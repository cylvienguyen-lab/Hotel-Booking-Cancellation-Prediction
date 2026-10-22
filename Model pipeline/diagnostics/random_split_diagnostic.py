"""Compare random holdouts inside each eligible train window with future CV validation.

Diagnostic only: the temporal CV split remains the model-selection protocol.
Run from Model pipeline with a Python environment containing requirements.txt.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
from utils.model_pipeline import build_pipeline  # noqa: E402

RUN_ID = "random_split_diagnostic_v1"
OUT = ROOT / "reports" / RUN_ID
SEEDS = (17, 42, 73, 101, 2026)
TEST_SIZE = 0.20
SOURCE = ROOT / "reports" / "feature_ablation_round1_v1" / "run_config.json"
SPLIT = ROOT / "splits" / "split_v1"
GOLD = PROJECT / "Data pipeline" / "hotel_booking_pipeline" / "datamart" / "gold" / "model_dataset"


def metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    return {
        "Rows": len(y),
        "Positives": int(y.sum()),
        "ActualRate": float(y.mean()),
        "MeanProbability": float(np.mean(probability)),
        "ROC_AUC": float(roc_auc_score(y, probability)),
        "AveragePrecision": float(average_precision_score(y, probability)),
        "LogLoss": float(log_loss(y, probability, labels=[0, 1])),
        "BrierScore": float(brier_score_loss(y, probability)),
    }


def main() -> None:
    reference = json.loads(SOURCE.read_text(encoding="utf-8"))["protocol"]
    feature_config = reference["feature_sets"]["F1"]
    features = [name for group in feature_config.values() for name in group]
    assert len(features) == 17
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    for hotel in ("H1", "H2"):
        source = pd.read_parquet(GOLD / f"gold_model_dataset_{hotel}.parquet")
        assert source["Booking_ID"].is_unique and source["Hotel"].eq(hotel).all()
        membership = pd.read_csv(SPLIT / hotel / "cv_membership.csv")
        params = reference["model_params"][hotel]

        for fold in range(1, 5):
            fold_membership = membership.loc[membership["Fold"].eq(fold)]
            cohorts = {}
            for role in ("Train", "Validation"):
                keys = fold_membership.loc[fold_membership["Role"].eq(role), ["Hotel", "Booking_ID"]]
                cohort = keys.merge(source, on=["Hotel", "Booking_ID"], how="left", validate="one_to_one")
                assert len(cohort) == len(keys) and cohort["IsCanceled"].notna().all()
                cohorts[role] = cohort
            train, future = cohorts["Train"], cohorts["Validation"]
            assert not set(train["Booking_ID"]) & set(future["Booking_ID"])

            for seed in SEEDS:
                fit, random_holdout = train_test_split(
                    train, test_size=TEST_SIZE, random_state=seed,
                    stratify=train["IsCanceled"],
                )
                pipeline = build_pipeline("random_forest", feature_config, params)
                start = perf_counter()
                pipeline.fit(fit[features], fit["IsCanceled"])
                fit_seconds = perf_counter() - start

                for name, cohort in (
                    ("train_80pct", fit),
                    ("random_holdout", random_holdout),
                    ("future_validation", future),
                ):
                    probability = pipeline.predict_proba(cohort[features])[:, 1]
                    results.append({
                        "Hotel": hotel, "Fold": fold, "Seed": seed,
                        "Cohort": name, "FitSeconds": fit_seconds,
                        **metrics(cohort["IsCanceled"], probability),
                    })
                print(f"{hotel} fold {fold} seed {seed}: fit {fit_seconds:.1f}s", flush=True)

            pd.DataFrame(results).to_csv(OUT / "seed_metrics.csv", index=False)

    result = pd.DataFrame(results)
    summary = result.groupby(["Hotel", "Fold", "Cohort"], as_index=False).agg(
        Seeds=("Seed", "nunique"),
        Rows=("Rows", "mean"),
        ActualRate=("ActualRate", "mean"),
        MeanProbability=("MeanProbability", "mean"),
        ROC_AUC=("ROC_AUC", "mean"),
        ROC_AUC_SD=("ROC_AUC", "std"),
        AveragePrecision=("AveragePrecision", "mean"),
        LogLoss=("LogLoss", "mean"),
        BrierScore=("BrierScore", "mean"),
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "run_config.json").write_text(json.dumps({
        "run_id": RUN_ID,
        "source": str(SOURCE.relative_to(ROOT)),
        "split": str(SPLIT.relative_to(ROOT)),
        "feature_set": "F1",
        "features": features,
        "seeds": SEEDS,
        "test_size": TEST_SIZE,
        "comparison": "Same 80%-train fitted model scores random holdout and future validation",
        "selection_use": "diagnostic_only; do not use random split for final model selection",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }, indent=2), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
