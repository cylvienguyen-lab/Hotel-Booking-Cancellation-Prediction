"""Score the historical Test cohort with saved models and evaluate its outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_ROOT.parents[1]
MODEL_ROOT = PROJECT_ROOT / "pipelines/model"
DEVELOPMENT_ROOT = PROJECT_ROOT / "artifacts/development"

# Resolve the Model package before third-party imports can claim this folder's
# separate package with the same name.
sys.path.insert(0, str(MODEL_ROOT))
from utils import model_outputs, model_refit, model_scoring  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


GOLD_ROOT = PROJECT_ROOT / "data/datamart/gold"
SPLIT_ROOT = DEVELOPMENT_ROOT / "splits/split_v2"
MODELS_ROOT = PROJECT_ROOT / "artifacts/train/models"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/evaluation"
NOTEBOOK = PROJECT_ROOT / "notebooks/evaluation_notebook.ipynb"
TRAIN_AUDIT = PROJECT_ROOT / "artifacts/train/train_job_audit.json"
TEST_AUDIT = OUTPUT_ROOT / "test_evaluation_job_audit.json"
TEST_CELLS = (34, 35, 38, 39, 40, 41, 42, 46, 47)
TEST_OUTPUTS = (
    "test_predictions.parquet",
    "layer_a_test.parquet",
    "hotel_forecast_arrivals.parquet",
    "hotel_forecast_occupancy.parquet",
    "hotel_forecast_coverage.parquet",
    "hotel_forecast_coverage_per_night.parquet",
    "hotel_forecast_summary.parquet",
    "booking_shortlist_workload_val.parquet",
    "booking_shortlist_workload_test.parquet",
    "booking_shortlist_eval.parquet",
    "booking_shortlist_operational.parquet",
)

def _load_evaluation_module(name: str):
    path = EVAL_ROOT / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"candidate_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_io = _load_evaluation_module("eval_io")
eval_metrics = _load_evaluation_module("eval_metrics")


def _preflight() -> tuple[dict, dict]:
    for path in (
        GOLD_ROOT / "data_build_audit.json",
        DEVELOPMENT_ROOT / "development_job_audit.json",
        OUTPUT_ROOT / "selection_job_audit.json",
        TRAIN_AUDIT,
        OUTPUT_ROOT / "finalist_choice.parquet",
        SPLIT_ROOT / "membership.parquet",
        SPLIT_ROOT / "config.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Test evaluation input: {path}")
    for path in (
        GOLD_ROOT / "data_build_audit.json",
        DEVELOPMENT_ROOT / "development_job_audit.json",
        OUTPUT_ROOT / "selection_job_audit.json",
        TRAIN_AUDIT,
    ):
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "passed":
            raise ValueError(f"Upstream job audit did not pass: {path}")

    train_audit = json.loads(TRAIN_AUDIT.read_text(encoding="utf-8"))
    choice_path = OUTPUT_ROOT / "finalist_choice.parquet"
    choice_hash = hashlib.sha256(choice_path.read_bytes()).hexdigest()
    if choice_hash != train_audit["selection_sha256"]:
        raise ValueError("Finalist choice changed after Train")
    choice = pd.read_parquet(choice_path)
    models = {entry["hotel"]: entry for entry in train_audit["models"]}
    if len(choice) != 2 or set(choice["hotel"]) != {"H1", "H2"} or set(models) != {"H1", "H2"}:
        raise ValueError("Test evaluation requires one selected model for each hotel")
    for row in choice.itertuples(index=False):
        if (
            row.artifact_id != models[row.hotel]["selected_artifact_id"]
            or row.calibration != "Uncalibrated"
        ):
            raise ValueError(f"{row.hotel}: Train and selection choices disagree")
        metadata_path = MODELS_ROOT / models[row.hotel]["model_version"] / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing saved model: {metadata_path}")
    return train_audit, {row.hotel: row for row in choice.itertuples(index=False)}


def _score_test(train_audit: dict) -> dict[str, int]:
    versions = {item["hotel"]: item["model_version"] for item in train_audit["models"]}
    split_audit = json.loads((SPLIT_ROOT / "audit.json").read_text(encoding="utf-8"))
    frames = []
    counts = {}
    for hotel in ("H1", "H2"):
        cohorts = model_refit.load_cohorts(
            hotel=hotel,
            gold_dataset_root=GOLD_ROOT / "model_dataset",
            membership_path=SPLIT_ROOT / "membership.parquet",
            split_config_path=SPLIT_ROOT / "config.json",
            feature_contract_path=GOLD_ROOT / "feature_contract.json",
        )
        test = cohorts.test
        expected = split_audit["by_hotel"][hotel]["test_metadata_only"]
        if len(test) != expected or not test["Partition"].eq("Test").all():
            raise ValueError(f"{hotel}: Test cohort differs from saved split")

        model = model_scoring.load_hotel_model(
            hotel, models_root=MODELS_ROOT, model_version=versions[hotel]
        )
        if set(model.metadata["components_skipped"]):
            raise ValueError(f"{hotel}: saved model has skipped components")
        scored = model_scoring.score_bookings(
            model, test, as_of_date=test["CutoffDate"].iloc[0]
        )
        base = test[
            ["Hotel", "Booking_ID", "ArrivalDate", "ADR", "TotalNights",
             "CutoffDate", "IsCanceled"]
        ].rename(columns={"IsCanceled": "y_true"})
        predictions = base.merge(
            scored, on=["Hotel", "Booking_ID"], how="inner", validate="one_to_one"
        )
        predictions["prediction_as_of_date"] = pd.to_datetime(
            predictions["CutoffDate"]
        ).dt.strftime("%Y-%m-%d")
        if (
            len(predictions) != expected
            or predictions[["Hotel", "Booking_ID"]].duplicated().any()
            or not predictions["y_true"].isin([0, 1]).all()
            or not np.isfinite(predictions["p_cancel"]).all()
            or not predictions["p_cancel"].between(0, 1).all()
            or not predictions["model_version"].eq(versions[hotel]).all()
        ):
            raise ValueError(f"{hotel}: invalid Test predictions")
        frames.append(predictions)
        counts[hotel] = len(predictions)
        print(f"Test score {hotel}: {len(predictions):,} bookings using {versions[hotel]}", flush=True)

    combined = pd.concat(frames, ignore_index=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_ROOT / "test_predictions.parquet", index=False)
    return counts


def _run_test_cells(choice: dict) -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index in TEST_CELLS:
        if notebook["cells"][index]["cell_type"] != "code":
            raise ValueError(f"Test notebook boundary changed at cell {index}")

    os.chdir(EVAL_ROOT)
    artifacts_all = eval_io.load_shortlist_uncalibrated(DEVELOPMENT_ROOT / "experiments")
    namespace = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK),
        "Path": Path,
        "json": json,
        "np": np,
        "pd": pd,
        "plt": plt,
        "display": display,
        "eval_io": eval_io,
        "eval_metrics": eval_metrics,
        "artifacts_all": artifacts_all,
        "FINALIST": {hotel: row.finalist_label for hotel, row in choice.items()},
        "CALIBRATION_CHOICE": {hotel: row.calibration for hotel, row in choice.items()},
        "out_dir": OUTPUT_ROOT,
    }
    for number, index in enumerate(TEST_CELLS, start=1):
        source = "".join(notebook["cells"][index]["source"])
        print(f"Test evaluation cell {number}/{len(TEST_CELLS)} (notebook index {index})", flush=True)
        exec(compile(source, f"{NOTEBOOK.name}:cell_{index}", "exec"), namespace)
    return namespace


def _write_shortlists(namespace: dict, versions: dict[str, str]) -> None:
    ranked = namespace["test"]
    k_per_hotel = namespace["SHORTLIST_K"]
    if k_per_hotel != {"H1": 10, "H2": 20}:
        raise ValueError(f"Unexpected frozen shortlist workload: {k_per_hotel}")
    if "EstimatedRoomCount" not in ranked or ranked["EstimatedRoomCount"].isna().any():
        raise ValueError("Test shortlist is missing room counts")
    for hotel, version in versions.items():
        if not ranked.loc[ranked["Hotel"].eq(hotel), "model_version"].eq(version).all():
            raise ValueError(f"{hotel}: shortlist model version differs from Train")
    for include_truth, name in (
        (True, "booking_shortlist_eval.parquet"),
        (False, "booking_shortlist_operational.parquet"),
    ):
        shortlist = model_outputs.generate_booking_shortlist(
            scored_bookings=ranked,
            k_per_hotel=k_per_hotel,
            rank_by="p_cancel",
            include_y_true=include_truth,
        )
        shortlist.to_parquet(OUTPUT_ROOT / name, index=False)


def _audit(train_audit: dict, choice_hash: str, counts: dict[str, int]) -> None:
    if hashlib.sha256((OUTPUT_ROOT / "finalist_choice.parquet").read_bytes()).hexdigest() != choice_hash:
        raise ValueError("Test job changed the frozen finalist choice")
    for name in TEST_OUTPUTS:
        path = OUTPUT_ROOT / name
        if not path.is_file() or pd.read_parquet(path).empty:
            raise ValueError(f"Missing or empty Test evaluation output: {path}")
    metric = pd.read_parquet(OUTPUT_ROOT / "layer_a_test.parquet")
    if len(metric) != 2 or set(metric["hotel"]) != {"H1", "H2"}:
        raise ValueError("Test metric table must contain H1 and H2")
    shortlist_eval = pd.read_parquet(OUTPUT_ROOT / "booking_shortlist_eval.parquet")
    shortlist_op = pd.read_parquet(OUTPUT_ROOT / "booking_shortlist_operational.parquet")
    if "y_true" not in shortlist_eval or "y_true" in shortlist_op:
        raise ValueError("Shortlist truth columns do not match evaluation/operations roles")
    if len(shortlist_eval) != len(shortlist_op):
        raise ValueError("Evaluation and operational shortlist row counts differ")
    versions = {item["hotel"]: item["model_version"] for item in train_audit["models"]}
    for hotel, version in versions.items():
        if not shortlist_op.loc[shortlist_op["Hotel"].eq(hotel), "model_version"].eq(version).all():
            raise ValueError(f"{hotel}: operational shortlist has wrong model version")
    audit = {
        "status": "passed",
        "historical_test_rows": counts,
        "model_versions": versions,
        "selection_sha256": choice_hash,
        "test_outputs": list(TEST_OUTPUTS),
    }
    TEST_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Test evaluation passed:", json.dumps(audit), flush=True)


def main() -> None:
    TEST_AUDIT.unlink(missing_ok=True)
    train_audit, choice = _preflight()
    choice_hash = train_audit["selection_sha256"]
    for name in TEST_OUTPUTS:
        (OUTPUT_ROOT / name).unlink(missing_ok=True)
    counts = _score_test(train_audit)
    namespace = _run_test_cells(choice)
    _write_shortlists(
        namespace,
        {item["hotel"]: item["model_version"] for item in train_audit["models"]},
    )
    _audit(train_audit, choice_hash, counts)


if __name__ == "__main__":
    main()
