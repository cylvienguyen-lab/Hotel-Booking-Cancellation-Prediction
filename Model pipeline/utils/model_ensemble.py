"""Probability-averaging ensembles from saved split_v2 OOF predictions.

No base model is fitted here. Components are immutable Section 2, 5 and 6
artifacts. Ensembles average booking-level probabilities and are scored with
the same Log Loss, ROC-AUC and ArrivalDate daily metrics as earlier runs, and
each ensemble is saved with the same three-file contract.
"""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from utils.model_screening import (
    ARTIFACT_FILES, KEYS, SPLIT_ID, _score, _verify_saved, _version,
)


COHORT = ["fold", "y_true", "CutoffDate", "StayDate"]
PREDICTION_COLUMNS = KEYS + [
    "CutoffDate", "StayDate", "fold", "y_true", "p_cancel",
    "experiment_id", "model_name", "split_id",
]
METHODS = {"equal_average", "temporal_weighted"}
ELIGIBILITY_RULE = (
    "earlier ValFold OOF rows with CutoffDate < FitDate, "
    "LabelAvailableDate < FitDate and ExpectedCheckoutDate < FitDate"
)
METRIC_DEFINITIONS = {
    "log_loss": "sklearn log_loss on booking-level ensemble probabilities, labels=[0, 1]",
    "roc_auc": "sklearn roc_auc_score on booking-level ensemble probabilities",
    "daily_bias_pct": "100 * sum(expected - actual) / sum(actual); ArrivalDate within ValFold",
    "daily_nrmse": "RMSE(expected - actual) / mean(actual); ArrivalDate within ValFold",
}


def file_sha256(path: Path) -> str:
    """Fingerprint an immutable source file by its exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_oof(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    hotel: str,
    membership: pd.DataFrame,
    labels: pd.DataFrame,
    test_keys: pd.DataFrame,
) -> None:
    """Recheck one saved OOF table against membership, Gold labels and metrics."""
    expected = membership.loc[
        membership.Hotel.eq(hotel) & membership.ValFold.notna(),
        KEYS + ["ValFold", "CutoffDate", "ArrivalDate"],
    ].merge(labels.loc[labels.Hotel.eq(hotel)], on=KEYS, validate="one_to_one")
    if len(predictions) != len(expected) or predictions.duplicated(KEYS).any():
        raise ValueError("OOF booking count/uniqueness differs from membership")
    aligned = predictions.merge(
        expected, on=KEYS, how="outer", validate="one_to_one",
        suffixes=("", "_expected"), indicator=True,
    )
    if (
        not aligned._merge.eq("both").all()
        or not aligned.fold.eq(aligned.ValFold).all()
        or not aligned.y_true.eq(aligned.IsCanceled).all()
        or not aligned.CutoffDate.eq(aligned.CutoffDate_expected).all()
        or not aligned.StayDate.eq(aligned.ArrivalDate).all()
    ):
        raise ValueError("OOF predictions differ from saved Development membership")
    p = predictions.p_cancel.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Missing, non-finite or out-of-range probability")
    if not predictions.Hotel.eq(hotel).all() or not predictions.split_id.eq(SPLIT_ID).all():
        raise ValueError("Prediction hotel or split identity differs")
    if len(predictions[KEYS].merge(test_keys, on=KEYS)):
        raise ValueError("Test booking found in OOF predictions")
    validation = metrics.loc[metrics.split.eq("val")].sort_values("fold")
    if validation.fold.tolist() != [1, 2, 3, 4]:
        raise ValueError("Saved Validation metrics do not cover four folds")
    for row in validation.itertuples(index=False):
        scored = predictions.loc[predictions.fold.eq(row.fold)]
        cohort = scored.rename(columns={"y_true": "IsCanceled", "StayDate": "ArrivalDate"})
        expected_score = _score(
            cohort, scored.p_cancel.to_numpy(), row.experiment_id,
            row.model_name, int(row.fold), "val",
        )
        for name in ("n_bookings", "log_loss", "roc_auc", "daily_bias_pct", "daily_nrmse"):
            if not np.isclose(getattr(row, name), expected_score[name]):
                raise ValueError(f"Saved {name} differs in fold {row.fold}")


def verify_component(
    *,
    experiment_root: Path,
    artifact_id: str,
    hotel: str,
    identity: dict,
    membership: pd.DataFrame,
    labels: pd.DataFrame,
    test_keys: pd.DataFrame,
    selected_features: dict,
    split_config: dict,
    source_fingerprint: str,
) -> dict:
    """Load one immutable base-model artifact after identity and OOF checks.

    ``identity`` holds either the exact screening ``spec`` of an ``_s1``/``_c1``
    run or the ``model_name``, ``experiment_version`` and ``estimator_params``
    of an initial ``_v2`` run. A missing artifact is an error, never a refit.
    """
    folder = Path(experiment_root) / artifact_id
    if not folder.is_dir():
        raise FileNotFoundError(f"Required component artifact is missing: {artifact_id}")
    if {path.name for path in folder.iterdir()} != ARTIFACT_FILES:
        raise ValueError(f"Incomplete component artifact: {artifact_id}")
    if "spec" in identity:
        request = {
            "experiment_id": artifact_id, "hotel": hotel,
            "split_id": SPLIT_ID, "source_fingerprint": source_fingerprint,
            "split_calendar": split_config, "base_features": selected_features,
            "spec": identity["spec"],
        }
        base_features = sum(
            (selected_features[group] for group in ("numeric", "binary", "categorical")), []
        )
        loaded = _verify_saved(folder, request, labels, membership, base_features)
        config, metrics, predictions = loaded["config"], loaded["metrics"], loaded["predictions"]
    else:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        expected = {
            "experiment_id": artifact_id, "split_id": SPLIT_ID, "hotel": hotel,
            "model_name": identity["model_name"],
            "experiment_version": identity["experiment_version"],
            "selected_features": selected_features,
            "estimator_params": identity["estimator_params"],
            "folds": split_config["folds"],
        }
        for name, value in expected.items():
            if config.get(name) != value:
                raise ValueError(f"{artifact_id}: saved {name} differs")
        metrics = pd.read_parquet(folder / "metrics.parquet")
        predictions = pd.read_parquet(folder / "predictions.parquet")
    if not predictions.experiment_id.eq(artifact_id).all():
        raise ValueError(f"{artifact_id}: prediction experiment ID differs")
    _check_oof(predictions, metrics, hotel, membership, labels, test_keys)
    return {
        "artifact_id": artifact_id, "config": config, "metrics": metrics,
        "predictions": predictions,
        "fingerprint": {
            "predictions_sha256": file_sha256(folder / "predictions.parquet"),
            "config_sha256": file_sha256(folder / "config.json"),
        },
        "status": "loaded and verified",
    }


def align_components(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join component OOF rows one-to-one; any unmatched booking is an error.

    The result has one ``p_cancel`` column per component label. Fold, label,
    CutoffDate and StayDate must be identical for every booking.
    """
    wide = None
    for label, predictions in components.items():
        # float32 TabM outputs are widened exactly so sums and means are float64.
        part = predictions[KEYS + COHORT + ["p_cancel"]].rename(
            columns={"p_cancel": label}
        ).astype({label: "float64"})
        if wide is None:
            wide = part
            continue
        merged = wide.merge(
            part, on=KEYS, how="outer", validate="one_to_one",
            suffixes=("", "_other"), indicator=True,
        )
        if not merged._merge.eq("both").all():
            raise ValueError(f"{label}: Validation booking membership differs")
        for column in COHORT:
            if not merged[column].eq(merged[f"{column}_other"]).all():
                raise ValueError(f"{label}: {column} differs for matched bookings")
        wide = merged.drop(columns=["_merge"] + [f"{column}_other" for column in COHORT])
    return wide.sort_values(["fold"] + KEYS).reset_index(drop=True)


def eligible_history(
    wide: pd.DataFrame,
    membership: pd.DataFrame,
    target_fold: int,
    fit_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return earlier-fold OOF rows whose outcomes were complete by FitDate."""
    dates = membership[KEYS + ["CutoffDate", "LabelAvailableDate", "ExpectedCheckoutDate"]]
    history = wide.loc[wide.fold.lt(target_fold)].merge(
        dates, on=KEYS, how="left", validate="one_to_one",
        suffixes=("", "_membership"),
    )
    if not history.CutoffDate.eq(history.CutoffDate_membership).all():
        raise ValueError("History CutoffDate differs from membership")
    return history.loc[
        history.CutoffDate.lt(fit_date)
        & history.LabelAvailableDate.lt(fit_date)
        & history.ExpectedCheckoutDate.lt(fit_date)
    ].drop(columns="CutoffDate_membership")


def fit_temporal_weight(history: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Minimize historical Log Loss of w*p_A + (1-w)*p_B over 0 <= w <= 1."""
    y = history.y_true.to_numpy()
    p_a = history[model_a].to_numpy(dtype=float)
    p_b = history[model_b].to_numpy(dtype=float)

    def objective(weight: float) -> float:
        return float(log_loss(y, weight * p_a + (1 - weight) * p_b, labels=[0, 1]))

    result = minimize_scalar(
        objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6}
    )
    weight = float(result.x)
    equal = objective(0.5)
    return {
        "weight_a": weight, "weight_b": 1.0 - weight,
        "historical_log_loss": float(result.fun),
        "historical_equal_log_loss": equal,
        "improvement_vs_equal": equal - float(result.fun),
        "optimizer_success": bool(result.success),
    }


def _temporal_weighted(
    wide: pd.DataFrame,
    labels: list[str],
    membership: pd.DataFrame,
    split_config: dict,
) -> tuple[pd.Series, list[dict]]:
    """Fit each target fold's weight from strictly eligible earlier OOF rows."""
    if len(labels) != 2:
        raise ValueError("Temporal weighting is defined for two models only")
    model_a, model_b = labels
    probability = pd.Series(np.nan, index=wide.index)
    details = []
    for fold_number, fold in enumerate(split_config["folds"], start=1):
        fit_date = pd.Timestamp(fold["fit_date"])
        detail = {"target_fold": fold_number, "fit_date": str(fit_date.date()),
                  "model_a": model_a, "model_b": model_b}
        history = eligible_history(wide, membership, fold_number, fit_date)
        if fold_number == 1:
            detail.update(status="unavailable: no prior validation fold", historical_n=0)
        elif history.empty or history.y_true.nunique() != 2:
            detail.update(
                status="unavailable: eligible history lacks both classes",
                historical_n=len(history),
            )
        else:
            fitted = fit_temporal_weight(history, model_a, model_b)
            extreme = min(fitted["weight_a"], fitted["weight_b"]) < 0.05
            detail.update(
                status="extreme weight (<0.05 on one model)" if extreme else "estimated",
                historical_n=len(history),
                historical_positive=int(history.y_true.sum()),
                historical_cancel_rate=float(history.y_true.mean()),
                history_rows_by_fold={
                    str(int(key)): int(value)
                    for key, value in history.fold.value_counts().sort_index().items()
                },
                **fitted,
            )
            target = wide.fold.eq(fold_number)
            probability[target] = (
                fitted["weight_a"] * wide.loc[target, model_a]
                + fitted["weight_b"] * wide.loc[target, model_b]
            )
        details.append(detail)
    return probability, details


def _score_folds(predictions: pd.DataFrame, experiment_id: str, method: str) -> pd.DataFrame:
    """Score each evaluated fold with the shared screening metric function."""
    rows = []
    for fold_number, scored in predictions.groupby("fold", sort=True):
        cohort = scored.rename(columns={"y_true": "IsCanceled", "StayDate": "ArrivalDate"})
        rows.append(_score(
            cohort, scored.p_cancel.to_numpy(), experiment_id, method,
            int(fold_number), "val",
        ))
    return pd.DataFrame(rows)


def _check_saved(target: Path, config: dict, metrics: pd.DataFrame,
                 predictions: pd.DataFrame) -> None:
    """Require a saved ensemble to reproduce exactly from its components."""
    if {path.name for path in target.iterdir()} != ARTIFACT_FILES:
        raise ValueError(f"Incomplete ensemble artifact: {target}")
    saved_config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    if saved_config["request"] != config["request"]:
        raise ValueError(f"Ensemble identity changed; use a new ID: {target.name}")
    for saved, current in zip(saved_config["fold_details"], config["fold_details"]):
        if saved.keys() != current.keys():
            raise ValueError("Saved weight-fitting history differs")
        for name, value in current.items():
            if isinstance(value, float):
                if not np.isclose(saved[name], value, rtol=0, atol=1e-9):
                    raise ValueError(f"Saved {name} differs")
            elif saved[name] != value:
                raise ValueError(f"Saved {name} differs")
    pd.testing.assert_frame_equal(
        pd.read_parquet(target / "predictions.parquet"), predictions,
        check_dtype=False, rtol=1e-12,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(target / "metrics.parquet"), metrics,
        check_dtype=False, rtol=1e-12,
    )


def run_ensemble(
    *,
    experiment_id: str,
    hotel: str,
    ensemble_id: str,
    method: str,
    components: dict[str, dict],
    membership: pd.DataFrame,
    split_config: dict,
    source_fingerprint: str,
    output_root: Path,
) -> dict:
    """Create or verify one versioned probability-averaging ensemble.

    ``components`` maps display labels to results of ``verify_component``.
    Equal averaging scores all four folds. Temporal weighting scores only folds
    with a forward-only weight; unavailable folds are recorded in the config.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown ensemble method: {method}")
    if not experiment_id.startswith(f"{hotel}_") or not experiment_id.endswith("_e1"):
        raise ValueError("Ensemble ID must start with the hotel and end with _e1")
    labels = list(components)
    wide = align_components({label: item["predictions"] for label, item in components.items()})
    if method == "equal_average":
        probability = wide[labels].mean(axis=1)
        weights = {label: 1 / len(labels) for label in labels}
        details = []
    else:
        probability, details = _temporal_weighted(wide, labels, membership, split_config)
        weights = "fold-specific; see fold_details"
    evaluated = probability.notna()
    predictions = wide.loc[evaluated, KEYS + ["CutoffDate", "StayDate", "fold", "y_true"]].copy()
    predictions["p_cancel"] = probability[evaluated].to_numpy(dtype=float)
    predictions["experiment_id"] = experiment_id
    predictions["model_name"] = method
    predictions["split_id"] = SPLIT_ID
    predictions = predictions[PREDICTION_COLUMNS].reset_index(drop=True)
    p = predictions.p_cancel.to_numpy()
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid ensemble probability")
    metrics = _score_folds(predictions, experiment_id, method)

    request = {
        "experiment_id": experiment_id, "hotel": hotel, "split_id": SPLIT_ID,
        "ensemble_id": ensemble_id, "method": method,
        "source_fingerprint": source_fingerprint, "split_calendar": split_config,
        "components": [
            {"label": label, "artifact_id": item["artifact_id"], **item["fingerprint"]}
            for label, item in components.items()
        ],
        "weights": weights,
        "combined_quantity": "booking-level p_cancel; daily totals are sums of ensemble p_cancel",
    }
    if method == "temporal_weighted":
        request["weight_protocol"] = {
            "objective": "historical Log Loss", "bounds": [0.0, 1.0],
            "optimizer": "scipy minimize_scalar bounded, xatol=1e-6",
            "eligibility": ELIGIBILITY_RULE,
            "extreme_flag": "minimum component weight < 0.05",
        }
    config = {
        "request": request,
        "evaluation_folds": sorted(int(fold) for fold in predictions.fold.unique()),
        "unavailable_folds": [
            {"fold": item["target_fold"], "reason": item["status"]}
            for item in details if item["status"].startswith("unavailable")
        ],
        "fold_details": details,
        "metric_definitions": METRIC_DEFINITIONS,
        "metric_rule": "unweighted; daily groups are ArrivalDate within each ValFold",
        "package_versions": {
            name: _version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")
        },
    }

    output_root = Path(output_root)
    target = output_root / experiment_id
    if target.exists():
        _check_saved(target, config, metrics, predictions)
        status = "loaded and verified"
    else:
        temporary = output_root / f".{experiment_id}.{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            (temporary / "config.json").write_text(
                json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            metrics.to_parquet(temporary / "metrics.parquet", index=False)
            predictions.to_parquet(temporary / "predictions.parquet", index=False)
            os.rename(temporary, target)
        except Exception:
            # Retain incomplete temporary evidence for diagnosis; never replace an ID.
            raise
        _check_saved(target, config, metrics, predictions)
        status = "created and verified"
    return {"config": config, "metrics": metrics, "predictions": predictions,
            "fold_details": details, "status": status}
