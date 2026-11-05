"""Track changes in booking risk and compare saved predictions with observed outcomes."""
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from inference import CONTACT_LIMIT, GOLD, KEYS, ROOT, month_end, publish_folder, read_json, windows
from utils.model_outputs import aggregate_hotel_forecast

EXTERNAL_INPUTS = ["EURUSD", "HotelSearchInterest", "FlightSearchInterest"]


def assessment_dates(end):
    value = os.environ["REPLAY_AS_OF"].strip()
    if value == "auto":
        # At month-end some guests are still booked to arrive. Check the same bookings again next month.
        first_check = month_end(end)
        return [first_check, month_end(first_check + timedelta(days=1))]
    dates = sorted({date.fromisoformat(item.strip()) for item in value.split(",")})
    if min(dates) < end:
        raise ValueError("Assessment dates must cover the scoring window")
    return dates


def monitoring_folder(month_folder, dates):
    return month_folder / ("monitoring_" + "_".join(map(str, dates)))


def score_psi(reference, current):
    # Use the same ten bands for both periods. Adding 0.5 to each count avoids log(0).
    def proportions(values):
        counts = np.histogram(values, bins=np.linspace(0, 1, 11))[0] + 0.5
        return counts / counts.sum()
    if not len(current):
        return np.nan
    if not len(reference) or not reference.between(0, 1).all():
        raise ValueError("Invalid Development reference scores")
    ref, obs = proportions(reference), proportions(current)
    return float(((obs - ref) * np.log(obs / ref)).sum())


def known_outcome_dates(records):
    # A cancellation or no-show is known on its status date.
    # Count a stay as completed once both the status date and expected checkout have passed.
    status_date = pd.to_datetime(records.ReservationStatusDate)
    checkout = pd.to_datetime(records.ArrivalDate) + pd.to_timedelta(records.TotalNights, unit="D")
    stay_end = pd.concat([status_date, checkout], axis=1).max(axis=1)
    available = status_date.where(records.IsCanceled.eq(1), stay_end)
    available = available.where(records.IsCanceled.isin([0, 1]) & status_date.notna())
    if available.lt(records.CutoffDate).any():
        raise ValueError("Outcome predates scoring cutoff")
    return available


def outcome_metrics(known, bookings, detail, expected):
    # Early cancellations can be known while other guests are still due to arrive.
    # Report monthly accuracy once every booking in this hotel's scoring month has an outcome.
    result = dict(
        actual_cancellations=np.nan, bias_bookings=np.nan, daily_count_MAE=np.nan,
        Brier=np.nan, ROC_AUC=np.nan, outcome_status="awaiting labels" if bookings else "no bookings",
    )
    if bookings and len(known) == bookings:
        actual = int(known.IsCanceled.sum())
        result.update(
            actual_cancellations=actual,
            bias_bookings=expected - actual,
            daily_count_MAE=(detail.expected_cancellations - detail.actual_cancellations).abs().mean(),
            Brier=brier_score_loss(known.IsCanceled, known.p_cancel),
            outcome_status="complete",
        )
        if known.IsCanceled.nunique() == 2:
            result["ROC_AUC"] = roc_auc_score(known.IsCanceled, known.p_cancel)
    return result


def reconcile_bookings(active, scored):
    """Match each eligible Gold booking to exactly one score on its cutoff date."""
    keys = KEYS + ["CutoffDate"]
    actual = scored[keys].copy()
    expected = active[keys].copy()
    actual["CutoffDate"] = pd.to_datetime(actual.CutoffDate)
    expected["CutoffDate"] = pd.to_datetime(expected.CutoffDate)
    matched = expected.merge(actual, on=keys, how="outer", validate="one_to_one", indicator=True)
    mismatches = matched[matched["_merge"].ne("both")]
    if len(mismatches):
        raise ValueError(f"Gold/score mismatch: {len(mismatches)}. First IDs: "
                         f"{mismatches.head(5).to_dict('records')}")


def input_distributions(active, reference, hotel, start):
    metrics = {col + "_available": active[col].notna().mean() for col in EXTERNAL_INPUTS}
    groups = []
    for feature in ["LeadTime", "MarketSegment", "DistributionChannel"]:
        series = {}
        for name, data in [("validation", reference), ("current", active)]:
            values = data[feature]
            if feature == "LeadTime":
                values = pd.cut(values, [-np.inf, 30, 60, 90, 180, np.inf],
                                labels=["≤30 days", "31–60 days", "61–90 days", "91–180 days", "181+ days"])
            values = values.astype("string").fillna("(missing)")
            series[name] = values.value_counts()
        counts = pd.DataFrame(series).fillna(0).astype(int)
        for name in series:
            total = counts[name].sum()
            counts[name + "_share"] = counts[name] / total if total else np.nan
        counts["change_pp"] = 100 * (counts.current_share - counts.validation_share)
        counts.index.name = "category"
        groups.append(counts.reset_index().assign(Hotel=hotel, cutoff_month=start.strftime("%Y-%m"), feature=feature))
    return metrics, pd.concat(groups, ignore_index=True)


def training_rate(entry, labels):
    """Use the hotel's final training bookings to get its historical cancellation rate."""
    split_id = read_json(ROOT / "config/development_config.json")["split_id"]
    membership = pd.read_parquet(ROOT / "artifacts/development/splits" / split_id / "membership.parquet",
                                 columns=KEYS + ["Partition", "FinalRefit"])
    train = membership.loc[membership.Hotel.eq(entry["hotel"]) & membership.Partition.eq("Dev")
                           & membership.FinalRefit.eq(1), KEYS]
    train = train.merge(labels[KEYS + ["IsCanceled"]], on=KEYS, how="left", validate="one_to_one")
    if len(train) != entry["train_rows"] or not train.IsCanceled.isin([0, 1]).all():
        raise ValueError("Dummy reference does not match the frozen final-refit population")
    return train.IsCanceled.mean()


def business_metrics(known, shortlist, entry, labels):
    """Measure outreach precision and forecast error for a hotel with complete outcomes."""
    frame = known.rename(columns={"IsCanceled": "y_true"})
    if frame.EstimatedRoomCount.isna().any() or not frame.TotalNights.gt(0).all():
        raise ValueError("Gold overnight cohort needs usable room counts and positive stay lengths")
    flagged = shortlist[KEYS].merge(frame[KEYS + ["y_true", "CutoffDate"]], on=KEYS, validate="one_to_one")
    if len(flagged) != len(shortlist) or flagged.empty:
        raise ValueError("Unable to reconcile the persisted shortlist with known outcomes")
    daily_rate = frame.groupby("CutoffDate").y_true.mean()
    contacts = flagged.groupby("CutoffDate").size().reindex(daily_rate.index, fill_value=0)
    k = CONTACT_LIMIT[entry["hotel"]]
    if not contacts.eq(frame.groupby("CutoffDate").size().clip(upper=k)).all():
        raise ValueError("Shortlist differs from the daily contact budget")
    hits = int(flagged.y_true.sum())
    cancelled = frame.y_true.sum()
    precision = hits / len(flagged)
    # Give random selection the same number of contacts as the shortlist on each day.
    random_precision = (daily_rate * contacts).sum() / len(flagged)
    dummy_rate = training_rate(entry, labels)
    calculated = aggregate_hotel_forecast(frame)
    dummy = aggregate_hotel_forecast(frame.assign(p_cancel=dummy_rate))
    metrics = dict(contacts=len(flagged), shortlist_k=k, hits=hits, precision_at_k=precision,
                   recall_at_k=hits / cancelled if cancelled else np.nan,
                   random_precision=random_precision, mean_daily_prevalence=daily_rate.mean(),
                   lift_at_k=precision / random_precision if random_precision else np.nan,
                   dummy_training_rate=dummy_rate)
    nights = []
    for kind, key, expected, actual, date_col in [
            ("arrivals", "arrivals", "E_arrivals", "actual_arrivals", "arrival_date"),
            ("room_nights", "occupancy", "E_room_nights", "actual_room_nights", "night_date")]:
        join = ["Hotel", date_col]
        daily = calculated[key][join + [expected, actual]].merge(
            dummy[key][join + [expected]], on=join, validate="one_to_one", suffixes=("", "_dummy"))
        daily = daily.rename(columns={date_col: "service_date", expected: "model",
                                      expected + "_dummy": "dummy", actual: "observed"})
        metrics[kind + "_days"] = len(daily)
        for method in ["model", "dummy"]:
            metrics[kind + "_mae_" + method] = (daily[method] - daily.observed).abs().mean()
        nights.append(daily.assign(metric=kind))
    return metrics, pd.concat(nights, ignore_index=True)


def run_month(start, end, out):
    as_of_dates = assessment_dates(end)
    target = monitoring_folder(out, as_of_dates)

    source = out / "inference"
    manifest = read_json(source / "manifest.json")
    scores = pd.read_parquet(source / "predictions.parquet")
    if "y_true" in scores or "IsCanceled" in scores or scores.duplicated(KEYS).any():
        raise ValueError("Invalid saved inference schema or keys")
    quality = pd.read_csv(source / "daily_quality.csv", parse_dates=["CutoffDate"])
    contract = read_json(GOLD / "feature_contract.json")
    predictors = contract["numeric_predictors"] + contract["categorical_predictors"]
    shortlist = pd.read_parquet(source / "shortlist.parquet")
    monthly, daily, distributions, nights = [], [], [], []
    for entry in manifest["models"]:
        hotel = entry["hotel"]
        cohort = scores[scores.Hotel.eq(hotel)].copy()
        cohort["CutoffDate"] = pd.to_datetime(cohort.CutoffDate)
        if not cohort.model_version.eq(entry["model_version"]).all() or not cohort.p_cancel.between(0, 1).all():
            raise ValueError("Scored model version or probability is invalid")
        ref = pd.read_parquet(ROOT / "artifacts/development/experiments" /
                              entry["selected_artifact_id"] / "predictions.parquet")
        if ref.duplicated(KEYS).any() or not ref.Hotel.eq(hotel).all() or (pd.to_datetime(ref.CutoffDate).dt.date >= start).any():
            raise ValueError("Invalid validation reference")
        reference_features = pd.read_parquet(GOLD / f"features/gold_feature_store_{hotel}.parquet",
                                             columns=KEYS + ["CutoffDate"] + predictors)
        active = reference_features[pd.to_datetime(reference_features.CutoffDate).dt.date.between(start, end)]
        reconcile_bookings(active, cohort)
        # Use the same validation bookings for the score and booking-mix comparisons.
        reference_features = ref[KEYS].merge(reference_features, on=KEYS, how="left",
                                               validate="one_to_one", indicator=True)
        if not reference_features["_merge"].eq("both").all():
            raise ValueError("Validation features missing from Gold")
        inputs, mix = input_distributions(active, reference_features, hotel, start)
        distributions.append(mix)
        reference_missing = reference_features[predictors].isna().mean().mean()
        reference_auc = roc_auc_score(ref.y_true, ref.p_cancel) if ref.y_true.nunique() == 2 else np.nan
        reference_brier = brier_score_loss(ref.y_true, ref.p_cancel)
        psi = score_psi(ref.p_cancel, cohort.p_cancel)
        q = quality[quality.Hotel.eq(hotel)].set_index("CutoffDate").copy()
        observed = cohort.groupby("CutoffDate").p_cancel.agg(["count", "sum"]).reindex(q.index, fill_value=0)
        if not observed["count"].eq(q.bookings).all():
            raise ValueError("Scored count differs from input count")
        q["expected_cancellations"] = observed["sum"]
        labels = pd.read_parquet(GOLD / f"labels/gold_label_store_{hotel}.parquet")
        records = cohort.merge(labels[KEYS + ["ReservationStatusDate", "IsCanceled"]],
                               how="left", on=KEYS, validate="one_to_one")
        available = known_outcome_dates(records)
        bookings = len(cohort)
        expected = cohort.p_cancel.sum()
        missing = int(q.missing_cells.sum())
        cells = int(q.predictor_cells.sum())
        for day in as_of_dates:
            known = records[available.le(pd.Timestamp(day))]
            detail = q.copy()
            resolved = known.groupby("CutoffDate").IsCanceled.agg(["count", "sum"]).reindex(q.index, fill_value=0)
            detail["known_labels"] = resolved["count"]
            detail["actual_cancellations"] = resolved["sum"].where(resolved["count"].eq(q.bookings))
            detail["assessment_date"] = str(day)
            detail["model_version"] = entry["model_version"]
            daily.append(detail.reset_index())
            outcome = outcome_metrics(known, bookings, detail, expected)
            business = {}
            if outcome["outcome_status"] == "complete":
                business, forecast = business_metrics(known, shortlist[shortlist.Hotel.eq(hotel)], entry, labels)
                nights.append(forecast.assign(assessment_date=str(day)))
            monthly.append(dict(
                **inputs, **business,
                cutoff_month=start.strftime("%Y-%m"), Hotel=hotel, model_version=entry["model_version"],
                assessment_date=str(day), bookings=bookings, known_labels=len(known),
                label_coverage=len(known) / bookings if bookings else np.nan,
                validation_bookings=len(ref), validation_ROC_AUC=reference_auc, validation_Brier=reference_brier,
                validation_missing_fraction=reference_missing, missing_cells=missing, predictor_cells=cells,
                missing_fraction=missing / cells if cells else np.nan,
                score_PSI=psi, expected_cancellations=expected, **outcome,
            ))
    summary = pd.DataFrame(monthly).sort_values(["assessment_date", "Hotel"])
    detail = pd.concat(daily, ignore_index=True).sort_values(["assessment_date", "Hotel", "CutoffDate"])
    if target.is_dir():
        return
    with publish_folder(target) as stage:
        summary.to_csv(stage / "monthly.csv", index=False)
        pd.concat(distributions, ignore_index=True).to_csv(stage / "input_mix.csv", index=False)
        if nights:
            pd.concat(nights, ignore_index=True).to_csv(stage / "nightly_business.csv", index=False)
        detail.to_csv(stage / "daily.csv", index=False)
    print("Monthly metrics:", target / "monthly.csv")


if __name__ == "__main__":
    for start, end, out in windows():
        run_month(start, end, out)
