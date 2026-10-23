"""Forward-only calibration of immutable Development OOF probabilities.

Each target fold uses only earlier validation-fold OOF scores whose labels and
planned stays were complete before its FitDate. Source models and ensembles
are never fitted here. The output contract is config, metrics and predictions.
"""

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from utils.model_ensemble import _check_oof, align_components, file_sha256
from utils.model_screening import ARTIFACT_FILES, KEYS, SPLIT_ID, _score, _version


METHODS = ("Uncalibrated", "Sigmoid", "Isotonic")
CLIP_EPS = 1e-6
ISOTONIC_MIN_N = 1000
ISOTONIC_MIN_PER_CLASS = 50
SIGMOID_PARAMS = {
    "penalty": "l2", "C": 1.0, "solver": "lbfgs",
    "max_iter": 1000, "random_state": 42,
}
PREDICTION_COLUMNS = KEYS + [
    "CutoffDate", "StayDate", "fold", "y_true", "p_original", "p_raw",
    "p_cancel", "experiment_id", "model_name", "split_id",
]

# Exact IDs and component order were verified against Section 7's saved
# registry. Ensemble identity is checked again against its component files.
SHORTLIST = {
    "H1": {
        "H1-E2": {
            "artifact_id": "H1_E2_RF30_XGB30_e1", "kind": "ensemble",
            "components": {"RF30": "H1_RF_v2", "XGB30": "H1_XGB_v2"},
        },
        "RW365": {"artifact_id": "H1_RF_RW365_s1", "kind": "single"},
    },
    "H2": {
        "H2-E4": {
            "artifact_id": "H2_E4_C2_C4_TabME8_e1", "kind": "ensemble",
            "components": {
                "C2": "H2_C2_RW180_FE4_c1", "C4": "H2_C4_T5_RW90_c1",
                "TabM-E8": "H2_TabM_s1",
            },
        },
        "H2-E1": {
            "artifact_id": "H2_E1_C2_TabME8_e1", "kind": "ensemble",
            "components": {"C2": "H2_C2_RW180_FE4_c1", "TabM-E8": "H2_TabM_s1"},
        },
        "C2": {"artifact_id": "H2_C2_RW180_FE4_c1", "kind": "single"},
        "C4": {"artifact_id": "H2_C4_T5_RW90_c1", "kind": "single"},
    },
}


def _expected_single_spec(hotel: str, candidate: str, rf_params: dict) -> dict:
    """Restate only the shortlisted, already-saved RF interventions."""
    common = {"family": "RF", "seed": 42}
    if (hotel, candidate) == ("H1", "RW365"):
        return {**common, "track": "Recency Weighting",
                "estimator_params": rf_params, "half_life_days": 365}
    if (hotel, candidate) == ("H2", "C2"):
        return {**common, "track": "Combined RF",
                "estimator_params": rf_params, "half_life_days": 180,
                "feature_experiment": "FE4"}
    if (hotel, candidate) == ("H2", "C4"):
        return {**common, "track": "Combined RF",
                "estimator_params": {**rf_params, "max_depth": 12,
                                     "min_samples_leaf": 40},
                "half_life_days": 90}
    raise ValueError(f"Unknown single-model shortlist item: {hotel}/{candidate}")


def verify_sources(
    *, experiment_root: Path, membership: pd.DataFrame,
    labels: pd.DataFrame, split_config: dict, rf_params: dict,
    source_fingerprint: str,
) -> dict:
    """Load six shortlist artifacts and verify identities, OOF rows and scores.

    Inputs contain Development membership and Gold Development labels only.
    Exact one-to-one validation joins make an omitted or added booking fatal.
    For each ensemble, saved probabilities must equal the original equal
    average of the immutable component files, in the saved component order.
    """
    if not membership.Partition.eq("Dev").all() or membership.duplicated(KEYS).any():
        raise ValueError("Expected unique Development membership only")
    if labels.duplicated(KEYS).any() or not labels.IsCanceled.isin([0, 1]).all():
        raise ValueError("Invalid Development label input")
    empty_test_keys = pd.DataFrame(columns=KEYS)
    results = {}
    for hotel, candidates in SHORTLIST.items():
        for candidate, item in candidates.items():
            folder = Path(experiment_root) / item["artifact_id"]
            if not folder.is_dir() or {p.name for p in folder.iterdir()} != ARTIFACT_FILES:
                raise ValueError(f"Missing/incomplete source: {item['artifact_id']}")
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            request = config["request"]
            if any((
                request["experiment_id"] != item["artifact_id"],
                request["hotel"] != hotel,
                request["split_id"] != SPLIT_ID,
                request["split_calendar"] != split_config,
                request["source_fingerprint"] != source_fingerprint,
            )):
                raise ValueError(f"Source identity differs: {candidate}")
            if item["kind"] == "single":
                if request["spec"] != _expected_single_spec(hotel, candidate, rf_params):
                    raise ValueError(f"Saved RF specification differs: {candidate}")
            else:
                members = item["components"]
                if (request["ensemble_id"] != candidate
                    or request["method"] != "equal_average"
                    or request["weights"] != {
                        label: 1 / len(members) for label in members
                    }
                    or [(part["label"], part["artifact_id"])
                        for part in request["components"]] != list(members.items())):
                    raise ValueError(f"Saved ensemble definition differs: {candidate}")
                parts = {}
                for part in request["components"]:
                    source = Path(experiment_root) / part["artifact_id"]
                    if (part["config_sha256"] != file_sha256(source / "config.json")
                        or part["predictions_sha256"] != file_sha256(
                            source / "predictions.parquet"
                        )):
                        raise ValueError(f"Ensemble component changed: {candidate}")
                    parts[part["label"]] = pd.read_parquet(
                        source / "predictions.parquet"
                    )
                wide = align_components(parts)
            predictions = pd.read_parquet(folder / "predictions.parquet")
            metrics = pd.read_parquet(folder / "metrics.parquet")
            if not predictions.experiment_id.eq(item["artifact_id"]).all():
                raise ValueError(f"Source prediction ID differs: {candidate}")
            _check_oof(predictions, metrics, hotel, membership, labels, empty_test_keys)
            if item["kind"] == "ensemble":
                matched = predictions[KEYS + ["p_cancel"]].merge(
                    wide[KEYS + list(members)], on=KEYS, how="outer",
                    validate="one_to_one", indicator=True,
                )
                if (not matched._merge.eq("both").all()
                    or not np.allclose(
                        matched.p_cancel,
                        matched[list(members)].mean(axis=1),
                        rtol=0, atol=1e-12,
                    )):
                    raise ValueError(f"Ensemble mean differs: {candidate}")
            results[(hotel, candidate)] = {
                "artifact_id": item["artifact_id"], "config": config,
                "metrics": metrics, "predictions": predictions,
                "fingerprint": {
                    "config_sha256": file_sha256(folder / "config.json"),
                    "predictions_sha256": file_sha256(folder / "predictions.parquet"),
                    "metrics_sha256": file_sha256(folder / "metrics.parquet"),
                },
            }
    return results


def eligible_history(
    predictions: pd.DataFrame, membership: pd.DataFrame,
    target_fold: int, fit_date: pd.Timestamp,
) -> pd.DataFrame:
    """Select earlier genuine OOF rows with complete outcomes by target FitDate."""
    earlier = predictions.loc[predictions.fold.lt(target_fold)].copy()
    dates = membership[KEYS + [
        "CutoffDate", "LabelAvailableDate", "ExpectedCheckoutDate", "ValFold"
    ]]
    history = earlier.merge(dates, on=KEYS, how="left",
                            validate="one_to_one", suffixes=("", "_membership"))
    if (len(history) != len(earlier)
        or history.LabelAvailableDate.isna().any()
        or not history.CutoffDate.eq(history.CutoffDate_membership).all()
        or not history.fold.eq(history.ValFold).all()):
        raise ValueError("Historical OOF row differs from split membership")
    history = history.loc[
        history.CutoffDate.lt(fit_date)
        & history.LabelAvailableDate.lt(fit_date)
        & history.ExpectedCheckoutDate.lt(fit_date)
    ].copy()
    if not history.fold.lt(target_fold).all():
        raise ValueError("Calibration history includes current/future fold")
    return history


def readiness(history: pd.DataFrame) -> dict:
    """Apply fixed class-support criteria before observing target-fold scores."""
    n = len(history)
    positive = int(history.y_true.sum())
    negative = n - positive
    both = positive > 0 and negative > 0
    return {
        "historical_n": n, "historical_positive": positive,
        "historical_negative": negative,
        "historical_rate": positive / n if n else None,
        "history_folds": {
            str(int(fold)): int(count)
            for fold, count in history.fold.value_counts().sort_index().items()
        },
        "sigmoid_ready": both,
        "isotonic_ready": (
            n >= ISOTONIC_MIN_N
            and positive >= ISOTONIC_MIN_PER_CLASS
            and negative >= ISOTONIC_MIN_PER_CLASS
        ),
    }


def _history_fingerprint(history: pd.DataFrame) -> str:
    """Fingerprint exact eligible OOF booking IDs, fold, label and score."""
    rows = history[KEYS + ["fold", "y_true", "p_cancel"]].sort_values(KEYS)
    return hashlib.sha256(rows.to_csv(index=False).encode("utf-8")).hexdigest()


def _mapping(
    method: str, history: pd.DataFrame, target: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """Fit one mapping on eligible historical scores, then transform target."""
    p_history = history.p_cancel.to_numpy(dtype=float)
    p_target = target.p_cancel.to_numpy(dtype=float)
    if method == "Uncalibrated":
        return p_target.copy(), {"mapping": "identity"}
    if method == "Sigmoid":
        logit_history = np.log(np.clip(p_history, CLIP_EPS, 1 - CLIP_EPS) /
                               (1 - np.clip(p_history, CLIP_EPS, 1 - CLIP_EPS)))
        model = LogisticRegression(**SIGMOID_PARAMS)
        model.fit(logit_history.reshape(-1, 1), history.y_true.astype(int))
        logit_target = np.log(np.clip(p_target, CLIP_EPS, 1 - CLIP_EPS) /
                              (1 - np.clip(p_target, CLIP_EPS, 1 - CLIP_EPS)))
        output = model.predict_proba(logit_target.reshape(-1, 1))[:, 1]
        return output, {
            "mapping": "sigmoid of clipped input logit",
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
            "regularization": SIGMOID_PARAMS,
            "input_logit_clipped_n": int(np.sum(
                (p_history <= CLIP_EPS) | (p_history >= 1 - CLIP_EPS)
            )),
        }
    if method == "Isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(p_history, history.y_true.astype(int))
        output = model.predict(p_target)
        return output, {
            "mapping": "isotonic, out_of_bounds=clip",
            "n_thresholds": len(model.X_thresholds_),
            "x_thresholds": model.X_thresholds_.tolist(),
            "y_thresholds": model.y_thresholds_.tolist(),
        }
    raise ValueError(f"Unknown calibration method: {method}")


def run_calibration(
    *, hotel: str, candidate: str, method: str, source: dict,
    membership: pd.DataFrame, split_config: dict,
    output_root: Path,
) -> dict:
    """Create/verify a versioned calibration run from one saved OOF source.

    F1 is available only for the identity mapping. For F2–F4, a calibrator
    is fitted using earlier eligible OOF rows; target labels are used only in
    the established evaluation metric. The same fixed 1e-6 clipping is used
    for scored/saved probabilities and clipping counts are retained.
    """
    if method not in METHODS or candidate not in SHORTLIST.get(hotel, {}):
        raise ValueError("Unknown shortlist candidate or calibration method")
    experiment_id = f"{hotel}_{candidate.removeprefix(hotel + '-')}_{method}_p1"
    if not experiment_id.startswith(f"{hotel}_") or not experiment_id.endswith("_p1"):
        raise ValueError("Invalid calibration artifact ID")
    original = source["predictions"]
    request = {
        "experiment_id": experiment_id, "hotel": hotel, "candidate": candidate,
        "method": method, "split_id": SPLIT_ID,
        "source_artifact_id": source["artifact_id"],
        "source_fingerprint": source["fingerprint"],
        "split_calendar": split_config,
        "history_rule": (
            "earlier ValFold genuine OOF; CutoffDate, LabelAvailableDate and "
            "ExpectedCheckoutDate strictly before target FitDate"
        ),
        "probability_clip_epsilon": CLIP_EPS,
        "sigmoid_params": SIGMOID_PARAMS,
        "isotonic_criterion": {
            "min_n": ISOTONIC_MIN_N, "min_positive": ISOTONIC_MIN_PER_CLASS,
            "min_negative": ISOTONIC_MIN_PER_CLASS,
        },
        "package_versions": {
            name: _version(name) for name in ("numpy", "pandas", "scikit-learn")
        },
    }
    metric_rows, prediction_parts, fold_details = [], [], []
    for fold_number, fold in enumerate(split_config["folds"], start=1):
        fit_date = pd.Timestamp(fold["fit_date"])
        target = original.loc[original.fold.eq(fold_number)].copy()
        history = eligible_history(original, membership, fold_number, fit_date)
        ready = readiness(history)
        detail = {
            "fold": fold_number, "fit_date": str(fit_date.date()),
            **ready, "history_fingerprint": _history_fingerprint(history),
        }
        if method != "Uncalibrated" and fold_number == 1:
            detail.update(status="unavailable: no earlier OOF fold")
            fold_details.append(detail)
            continue
        if method == "Sigmoid" and not ready["sigmoid_ready"]:
            detail.update(status="unavailable: historical labels lack two classes")
            fold_details.append(detail)
            continue
        if method == "Isotonic" and not ready["isotonic_ready"]:
            detail.update(status="unavailable: predefined Isotonic class support")
            fold_details.append(detail)
            continue
        raw, mapping = _mapping(method, history, target)
        if len(raw) != len(target) or not np.isfinite(raw).all():
            raise ValueError("Non-finite or misaligned calibrated probability")
        if ((raw < 0) | (raw > 1)).any():
            raise ValueError("Calibrator produced out-of-range probability")
        scored = np.clip(raw, CLIP_EPS, 1 - CLIP_EPS)
        if not np.isfinite(scored).all():
            raise ValueError("Invalid clipped probability")
        clip_low = int(np.sum(raw < CLIP_EPS))
        clip_high = int(np.sum(raw > 1 - CLIP_EPS))
        detail.update({
            "status": "evaluated", "mapping": mapping,
            "target_n": len(target), "raw_exact_zero": int(np.sum(raw == 0)),
            "raw_exact_one": int(np.sum(raw == 1)),
            "clip_low_n": clip_low, "clip_high_n": clip_high,
        })
        prediction = target[KEYS + [
            "CutoffDate", "StayDate", "fold", "y_true"
        ]].copy()
        prediction["p_original"] = target.p_cancel.to_numpy(dtype=float)
        prediction["p_raw"] = raw
        prediction["p_cancel"] = scored
        prediction["experiment_id"] = experiment_id
        prediction["model_name"] = f"{candidate}:{method}"
        prediction["split_id"] = SPLIT_ID
        prediction_parts.append(prediction[PREDICTION_COLUMNS])
        cohort = target.rename(columns={"y_true": "IsCanceled",
                                        "StayDate": "ArrivalDate"})
        metric = _score(cohort, scored, experiment_id,
                        f"{candidate}:{method}", fold_number, "val")
        metric.update(raw_exact_zero=detail["raw_exact_zero"],
                      raw_exact_one=detail["raw_exact_one"],
                      clip_low_n=clip_low, clip_high_n=clip_high)
        metric_rows.append(metric)
        fold_details.append(detail)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    config = {
        "request": request, "fold_details": fold_details,
        "evaluation_folds": sorted(int(f) for f in predictions.fold.unique()),
        "metric_rule": "unweighted; daily groups are ArrivalDate within ValFold",
        "status": "complete" if len(metrics) >= 3 else "partial",
    }
    target_dir = Path(output_root) / experiment_id
    if target_dir.exists():
        if {p.name for p in target_dir.iterdir()} != ARTIFACT_FILES:
            raise ValueError(f"Incomplete calibration artifact: {experiment_id}")
        saved = json.loads((target_dir / "config.json").read_text(encoding="utf-8"))
        # Recomputed mappings, historical cohorts and clipping counts must
        # all agree, in addition to the immutable request identity.
        if saved != config:
            raise ValueError(f"Calibration config/history changed: {experiment_id}")
        pd.testing.assert_frame_equal(
            pd.read_parquet(target_dir / "predictions.parquet"), predictions,
            check_dtype=False, rtol=1e-10, atol=1e-12,
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(target_dir / "metrics.parquet"), metrics,
            check_dtype=False, rtol=1e-10, atol=1e-12,
        )
        status = "loaded and verified"
    else:
        Path(output_root).mkdir(parents=True, exist_ok=True)
        temporary = Path(output_root) / f".{experiment_id}.{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            (temporary / "config.json").write_text(
                json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            metrics.to_parquet(temporary / "metrics.parquet", index=False)
            predictions.to_parquet(temporary / "predictions.parquet", index=False)
            os.rename(temporary, target_dir)
        except Exception:
            # Preserve partial evidence for diagnosis; never overwrite an ID.
            raise
        status = "created and verified"
    return {"config": config, "metrics": metrics,
            "predictions": predictions, "status": status}
