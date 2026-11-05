"""Fit and persist the two frozen finalists using Development rows only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import model_refit, model_scoring


MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parents[1]
GOLD_ROOT = PROJECT_ROOT / "data/datamart/gold"
DEVELOPMENT_ROOT = PROJECT_ROOT / "artifacts/development"
SPLIT_ROOT = DEVELOPMENT_ROOT / "splits/split_v2"
SELECTION_ROOT = PROJECT_ROOT / "artifacts/evaluation"
MODELS_ROOT = PROJECT_ROOT / "artifacts/train/models"
EXPERIMENT_ROOT = DEVELOPMENT_ROOT / "experiments"
TRAIN_AUDIT = PROJECT_ROOT / "artifacts/train/train_job_audit.json"


def _verify_selected_experiment(hotel: str, artifact_id: str) -> None:
    """Confirm the frozen fit specs still match the selected OOF config."""
    supported = model_refit.FINALISTS[hotel]
    request = json.loads(
        (EXPERIMENT_ROOT / artifact_id / "config.json").read_text(encoding="utf-8")
    )["request"]
    split = json.loads((SPLIT_ROOT / "config.json").read_text(encoding="utf-8"))
    expected_sources = supported["source_artifacts"]
    actual_sources = {
        entry["label"]: entry["artifact_id"] for entry in request["components"]
    }
    if (
        request["experiment_id"] != artifact_id
        or request["hotel"] != hotel
        or request["split_id"] != "split_v2"
        or request["ensemble_id"] != supported["ensemble_id"]
        or request["source_fingerprint"] != split["source_fingerprint"]
        or request["method"] != "equal_average"
        or actual_sources != expected_sources
        or request["weights"] != {name: 0.5 for name in expected_sources}
    ):
        raise ValueError(f"{hotel}: selected OOF ensemble config differs from Train spec")
    for entry in request["components"]:
        component_id = entry["artifact_id"]
        folder = EXPERIMENT_ROOT / component_id
        for name, digest_key in (
            ("config.json", "config_sha256"),
            ("predictions.parquet", "predictions_sha256"),
        ):
            artifact = folder / name
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != entry[digest_key]:
                raise ValueError(f"{hotel}: source experiment changed: {artifact}")
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        fitted_spec = supported["components"][entry["label"]]
        if "request" in config:
            saved_spec = config["request"]["spec"]
            if any(saved_spec.get(key) != value for key, value in fitted_spec.items()):
                raise ValueError(f"{hotel}: component config differs: {component_id}")
        elif (
            config["model_name"] != fitted_spec["family"]
            or any(
                config["estimator_params"].get(key) != value
                for key, value in fitted_spec["estimator_params"].items()
            )
        ):
            raise ValueError(f"{hotel}: component config differs: {component_id}")


def _read_decisions() -> dict[str, dict]:
    for path in (
        GOLD_ROOT / "data_build_audit.json",
        GOLD_ROOT / "feature_contract.json",
        DEVELOPMENT_ROOT / "development_job_audit.json",
        SELECTION_ROOT / "selection_job_audit.json",
        SELECTION_ROOT / "finalist_choice.parquet",
        SPLIT_ROOT / "membership.parquet",
        SPLIT_ROOT / "config.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Train input: {path}")
    for path in (
        GOLD_ROOT / "data_build_audit.json",
        DEVELOPMENT_ROOT / "development_job_audit.json",
        SELECTION_ROOT / "selection_job_audit.json",
    ):
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "passed":
            raise ValueError(f"Upstream job audit did not pass: {path}")

    selection_audit = json.loads(
        (SELECTION_ROOT / "selection_job_audit.json").read_text(encoding="utf-8")
    )
    decisions = pd.read_parquet(SELECTION_ROOT / "finalist_choice.parquet")
    if set(decisions.columns) != {"hotel", "finalist_label", "calibration", "artifact_id"}:
        raise ValueError("Finalist choice has unexpected columns")
    if len(decisions) != 2 or set(decisions["hotel"]) != set(model_refit.FINALISTS):
        raise ValueError("Finalist choice must contain exactly H1 and H2")

    chosen = {}
    for row in decisions.itertuples(index=False):
        supported = model_refit.FINALISTS[row.hotel]
        expected = (supported["label"], "Uncalibrated", supported["artifact_id"])
        actual = (row.finalist_label, row.calibration, row.artifact_id)
        if actual != expected:
            raise ValueError(f"{row.hotel}: selected choice is not supported: {actual}")
        audit_choice = selection_audit.get("finalists", {}).get(row.hotel, {})
        if (
            audit_choice.get("label"),
            audit_choice.get("calibration"),
            audit_choice.get("artifact_id"),
        ) != actual:
            raise ValueError(f"{row.hotel}: selection audit and parquet disagree")
        _verify_selected_experiment(row.hotel, row.artifact_id)
        chosen[row.hotel] = {
            "artifact_id": row.artifact_id,
            "ensemble_id": supported["ensemble_id"],
        }
    return chosen


def _verify_bundle(hotel: str, model_version: str, train_row: pd.DataFrame) -> dict:
    folder = MODELS_ROOT / model_version
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = model_refit.FINALISTS[hotel]
    if metadata["hotel"] != hotel or metadata["ensemble_id"] != expected["ensemble_id"]:
        raise ValueError(f"{hotel}: saved model metadata has wrong identity")
    if metadata["components_skipped"]:
        raise ValueError(f"{hotel}: persisted model has skipped components")
    components = metadata["components"]
    if {entry["component"] for entry in components} != set(expected["components"]):
        raise ValueError(f"{hotel}: persisted model has incomplete components")

    for entry in components:
        for path_key, hash_key in (
            ("artifact_path", "sha256"),
            ("fe4_mapping_path", "fe4_mapping_sha256"),
            ("state_dict_path", "state_dict_sha256"),
            ("preprocess_path", "preprocess_sha256"),
            ("spec_path", "spec_sha256"),
        ):
            if path_key not in entry:
                continue
            artifact = folder / entry[path_key]
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != entry[hash_key]:
                raise ValueError(f"{hotel}: missing or corrupt saved component: {artifact}")

    loaded = model_scoring.load_hotel_model(
        hotel, models_root=MODELS_ROOT, model_version=model_version
    )
    probe = model_scoring.score_bookings(
        loaded, train_row, as_of_date=train_row["CutoffDate"].iloc[0]
    )
    if len(probe) != 1 or not np.isfinite(probe["p_cancel"].iloc[0]):
        raise ValueError(f"{hotel}: saved model did not score a Development probe row")
    if not 0 <= float(probe["p_cancel"].iloc[0]) <= 1:
        raise ValueError(f"{hotel}: saved model probability is out of range")
    return metadata


def main() -> None:
    TRAIN_AUDIT.unlink(missing_ok=True)
    chosen = _read_decisions()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for hotel in ("H1", "H2"):
        print(f"Train {hotel}: load eligible Development cohort", flush=True)
        cohorts = model_refit.load_cohorts(
            hotel=hotel,
            gold_dataset_root=GOLD_ROOT / "model_dataset",
            membership_path=SPLIT_ROOT / "membership.parquet",
            split_config_path=SPLIT_ROOT / "config.json",
            feature_contract_path=GOLD_ROOT / "feature_contract.json",
            include_test=False,
        )
        if cohorts.test is not None:
            raise RuntimeError(f"{hotel}: Train unexpectedly loaded a Test cohort")
        print(f"Train {hotel}: fit {chosen[hotel]['ensemble_id']} on {len(cohorts.train):,} rows", flush=True)
        summary = model_refit.fit_finalist(
            hotel, cohorts, artifact_id=chosen[hotel]["artifact_id"]
        )
        saved = model_refit.persist_hotel_models(
            hotel=hotel, cohorts=cohorts, summary=summary, models_root=MODELS_ROOT
        )
        metadata = _verify_bundle(
            hotel, saved["model_version"], cohorts.train.iloc[:1].copy()
        )
        results.append({
            "hotel": hotel,
            "selected_artifact_id": chosen[hotel]["artifact_id"],
            "model_version": saved["model_version"],
            "train_rows": len(cohorts.train),
            "components": [entry["component"] for entry in metadata["components"]],
            "split_fingerprint": metadata["split_v2_fingerprint"],
        })
        print(f"Train {hotel}: saved bundle reloaded and scored a Development row", flush=True)

    audit = {
        "status": "passed",
        "selection_sha256": hashlib.sha256(
            (SELECTION_ROOT / "finalist_choice.parquet").read_bytes()
        ).hexdigest(),
        "models": results,
    }
    TRAIN_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Train build passed:", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
