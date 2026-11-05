"""Score each hotel's active bookings 14 days before arrival."""
import hashlib
import json
import os
import sys
from calendar import monthrange
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data/datamart/gold"
TRAIN = ROOT / "artifacts/train"
KEYS = ["Hotel", "Booking_ID"]
CONTACT_LIMIT = {"H1": 10, "H2": 20}
BOOKING_FIELDS = KEYS + ["ArrivalDate", "CutoffDate", "ADR", "TotalNights", "EstimatedRoomCount"]

# Saved models refer to the utils package under pipelines/model.
sys.path.insert(0, str(ROOT / "pipelines/model"))
from utils.model_outputs import generate_booking_shortlist
from utils.model_scoring import load_hotel_model, score_bookings


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def replay_range():
    start = date.fromisoformat(os.environ["REPLAY_START"])
    end = date.fromisoformat(os.environ["REPLAY_END"])
    if start > end:
        raise ValueError("Choose an ordered cutoff range")
    calendar = read_json(ROOT / "config/development_config.json")["split_calendar"]
    if not calendar["test_start"] <= start.isoformat() <= end.isoformat() < calendar["test_end"]:
        raise ValueError("Replay dates must fall inside the frozen historical Test window")
    return start, end


def run_folder(start, end):
    run = hashlib.sha256(os.environ["REPLAY_RUN_ID"].encode()).hexdigest()[:20]
    return ROOT / "artifacts/operations" / f"{start}_{end}" / run


def month_end(day):
    return day.replace(day=monthrange(day.year, day.month)[1])


def windows():
    """Group bookings by scoring month, including partial months at either end."""
    start, end = replay_range()
    while start <= end:
        last = min(month_end(start), end)
        yield start, last, run_folder(start, last)
        start = last + timedelta(days=1)


@contextmanager
def publish_folder(target):
    """Move the finished folder into place only after every file has been written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        stage = Path(tmp) / target.name
        stage.mkdir()
        yield stage
        stage.rename(target)


def daily_forecast(predictions, quality):
    # Each booking contributes its probability of cancellation or no-show to the daily total.
    keys = ["Hotel", "CutoffDate"]
    expected = predictions.groupby(keys).p_cancel.sum().rename("expected_cancellations")
    forecast = quality[keys + ["bookings"]].join(expected, on=keys)
    forecast["expected_cancellations"] = forecast.expected_cancellations.fillna(0.0)
    forecast["expected_arrivals"] = forecast.bookings - forecast.expected_cancellations
    forecast["ArrivalDate"] = forecast.CutoffDate + pd.Timedelta(days=14)
    return forecast


def run_month(start, end, out):
    target = out / "inference"
    if (target / "manifest.json").exists():
        return

    # Gold already contains the T−14 cohort. Leave the outcome in the label store until monitoring.
    contract = read_json(GOLD / "feature_contract.json")
    predictors = contract["numeric_predictors"] + contract["categorical_predictors"]
    fields = predictors + contract["metadata_only"]
    audit = read_json(TRAIN / "train_job_audit.json")
    if audit["status"] != "passed" or sorted(m["hotel"] for m in audit["models"]) != ["H1", "H2"]:
        raise ValueError("Expected two verified bootstrap model bundles")
    dates = pd.date_range(start, end)
    frames, quality = [], []
    for entry in audit["models"]:
        hotel = entry["hotel"]
        model = load_hotel_model(hotel, TRAIN / "models", entry["model_version"])
        data = pd.read_parquet(GOLD / f"features/gold_feature_store_{hotel}.parquet", columns=fields)
        data["CutoffDate"] = pd.to_datetime(data.CutoffDate)
        data = data[data.CutoffDate.between(dates[0], dates[-1])]
        if data.Booking_ID.isna().any() or data.duplicated(KEYS).any() or not data.Hotel.eq(hotel).all():
            raise ValueError(f"{hotel}: invalid booking identity")
        if not pd.to_datetime(data.ArrivalDate).sub(data.CutoffDate).eq(pd.Timedelta(days=14)).all():
            raise ValueError(f"{hotel}: invalid cutoff")
        for day in dates:
            batch = data[data.CutoffDate.eq(day)].sort_values("Booking_ID")
            # Keep a row for every scoring day, including days with no eligible bookings.
            quality.append(dict(Hotel=hotel, CutoffDate=day, bookings=len(batch),
                                missing_cells=int(batch[predictors].isna().sum().sum()),
                                predictor_cells=batch[predictors].size))
            if batch.empty:
                continue
            scores = score_bookings(model, batch, as_of_date=day)
            saved = batch[BOOKING_FIELDS].merge(scores, on=KEYS, validate="one_to_one")
            if len(saved) != len(batch) or not saved.p_cancel.between(0, 1).all():
                raise ValueError(f"{hotel} {day}: invalid predictions")
            frames.append(saved)
    cols = BOOKING_FIELDS + ["p_cancel", "prediction_as_of_date", "model_version", "generated_at_utc"]
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    with publish_folder(target) as stage:
        predictions.to_parquet(stage / "predictions.parquet", index=False)
        daily = pd.DataFrame(quality)
        daily.to_csv(stage / "daily_quality.csv", index=False)
        daily_forecast(predictions, daily).to_csv(stage / "daily_forecast.csv", index=False)
        shortlist = generate_booking_shortlist(predictions, CONTACT_LIMIT)
        shortlist.to_parquet(stage / "shortlist.parquet", index=False)
        manifest = dict(start=str(start), end=str(end), run_id=os.environ["REPLAY_RUN_ID"],
                        models=audit["models"], bookings=len(predictions))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Scored {len(predictions)} bookings across {len(dates)} cutoff dates.")


if __name__ == "__main__":
    for start, end, out in windows():
        run_month(start, end, out)
