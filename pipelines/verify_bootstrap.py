"""Check the outputs that a complete bootstrap DAG run must produce."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDITS = {
    "data": PROJECT_ROOT / "data/datamart/gold/data_build_audit.json",
    "development": PROJECT_ROOT / "artifacts/development/development_job_audit.json",
    "selection": PROJECT_ROOT / "artifacts/evaluation/selection_job_audit.json",
    "train": PROJECT_ROOT / "artifacts/train/train_job_audit.json",
    "test_evaluation": PROJECT_ROOT / "artifacts/evaluation/test_evaluation_job_audit.json",
}


def main() -> None:
    audits = {}
    for name, path in AUDITS.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} audit missing: {path}")
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("status") != "passed":
            raise ValueError(f"{name} audit did not pass: {path}")
        audits[name] = audit

    expected_hotels = {"H1", "H2"}
    if {row["hotel"] for row in audits["data"]["hotels"]} != expected_hotels:
        raise ValueError("Data audit does not cover both hotels")
    if audits["development"]["complete_experiments"] < 2:
        raise ValueError("Development experiments are incomplete")
    if set(audits["selection"]["finalists"]) != expected_hotels:
        raise ValueError("Selection does not cover both hotels")
    if {row["hotel"] for row in audits["train"]["models"]} != expected_hotels:
        raise ValueError("Train does not cover both hotels")
    if set(audits["test_evaluation"]["historical_test_rows"]) != expected_hotels:
        raise ValueError("Test evaluation does not cover both hotels")

    choice = PROJECT_ROOT / "artifacts/evaluation/finalist_choice.parquet"
    digest = hashlib.sha256(choice.read_bytes()).hexdigest()
    if digest != audits["train"]["selection_sha256"]:
        raise ValueError("Train selection digest does not match finalist choice")
    if digest != audits["test_evaluation"]["selection_sha256"]:
        raise ValueError("Test selection digest does not match finalist choice")

    models_root = PROJECT_ROOT / "artifacts/train/models"
    for model in audits["train"]["models"]:
        if not (models_root / model["model_version"] / "metadata.json").is_file():
            raise FileNotFoundError(f"Saved model missing: {model['model_version']}")

    output_root = PROJECT_ROOT / "artifacts/evaluation"
    for name in audits["test_evaluation"]["test_outputs"]:
        if not (output_root / name).is_file():
            raise FileNotFoundError(f"Test output missing: {name}")
    predictions = pd.read_parquet(output_root / "test_predictions.parquet", columns=["Hotel"])
    counts = predictions["Hotel"].value_counts().to_dict()
    if counts != audits["test_evaluation"]["historical_test_rows"]:
        raise ValueError(f"Test prediction counts disagree with audit: {counts}")

    print(
        "Bootstrap demo passed: 5 jobs, 2 saved models, "
        f"{len(predictions):,} historical Test predictions",
        flush=True,
    )


if __name__ == "__main__":
    main()
