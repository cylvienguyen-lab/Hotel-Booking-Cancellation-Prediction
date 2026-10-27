"""Pure metric functions for Section 4 model evaluation.

All functions are stateless and read from the predictions.parquet frames
saved by the Section 3 model pipeline. No model is fitted here, no
artifact is written. The output of every public function is either a
plain Python scalar / dict or a small pandas DataFrame.

Metric choice (set in agreement with Section 4 spec):

    Primary (model selection)
        daily_nrmse        — accuracy of per-arrival-night cancellation sums,
                             which is exactly the quantity the overbooking
                             policy and the "expected arrivals" layer consume.

    Diagnostic / sanity (reported alongside, no up-front cutoffs)
        daily_bias_pct     — signed systematic over/under-prediction per day;
                             matters because overbooking is asymmetric.
        log_loss           — proper scoring rule continuous with Section 3's
                             saved metrics; catches degenerate near-0/1
                             probability outputs.
        ap, ap_lift        — ranking quality for the minority (cancellation)
                             class; drives the Section 5 outreach shortlist.

    Calibration selection
        ece                — one-number summary of a reliability diagram, used
                             to pick between Uncalibrated / Sigmoid / Isotonic.
        reliability_bins   — the binned predicted-vs-observed table that
                             backs the reliability diagram plot.

    Operating-point reporting
        pr_curve_points    — the full precision-recall curve (visualisation).
        confusion_at_threshold — operating point for the outreach list.

Metrics deliberately dropped and why:

    ROC-AUC                — base-rate independent, so useful only for cross-
                             hotel ranking comparison; we never pool H1 and H2.
    Brier score            — overlaps with log_loss as an overall probability
                             quality score; the extra column adds no selection
                             information once ECE separates calibration out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
)


EPS = 1e-15  # numerical floor for log loss on near-0 / near-1 predictions


# -----------------------------------------------------------------------------
# Validation / preparation
# -----------------------------------------------------------------------------
def _prepare(predictions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pull (y_true, p_cancel) from a predictions frame with basic sanity checks.

    The saved OOF frames always have both columns and probabilities in [0, 1];
    this guard protects downstream code from quietly consuming NaN or
    out-of-range values if a different source frame is passed in.
    """
    needed = {"y_true", "p_cancel"}
    missing = needed - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    y_true = predictions["y_true"].to_numpy(dtype=int)
    p = predictions["p_cancel"].to_numpy(dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Found non-finite predicted probabilities")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Predicted probabilities outside [0, 1]")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Target must be 0/1 (1 = cancellation)")
    return y_true, np.clip(p, EPS, 1 - EPS)


# -----------------------------------------------------------------------------
# Primary + diagnostic metrics
# -----------------------------------------------------------------------------
def compute_log_loss(predictions: pd.DataFrame) -> float:
    """Mean binary cross-entropy on (y_true, p_cancel). Lower is better.

    Role: diagnostic / sanity. Section 3 reports this on every saved run;
    recomputing it here lets us verify the artifact and ranks probability
    outputs by how harshly they penalise confident-wrong predictions.
    """
    y_true, p = _prepare(predictions)
    return float(log_loss(y_true, p, labels=[0, 1]))


def compute_ap(predictions: pd.DataFrame) -> dict:
    """Average Precision, class prevalence, and AP lift.

    Role: diagnostic / outreach-list driver.

    * ``ap`` is the standard numeric summary of the Precision-Recall curve
      (sklearn.metrics.average_precision_score). It measures how well the
      model ranks the minority cancellation class.
    * A no-skill constant-rate model achieves ``ap = prevalence``, so
      ``ap_lift = ap - prevalence`` is the baseline-adjusted gain, directly
      comparable between H1 and H2 despite different cancellation rates.
    * AP is invariant to monotonic transformations of the probabilities,
      so it alone cannot tell us how well Sigma(1 - p) estimates arrivals.
      That is what the daily metrics are for.
    """
    y_true, p = _prepare(predictions)
    ap = float(average_precision_score(y_true, p))
    prevalence = float(y_true.mean())
    return {"ap": ap, "prevalence": prevalence, "ap_lift": ap - prevalence}


def daily_aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse booking-level predictions to one row per arrival night.

    Returns
    -------
    DataFrame with columns
        StayDate : calendar night of arrival
        expected : sum of p_cancel over bookings arriving that night
                   (= model's predicted cancellation count)
        actual   : sum of y_true over the same bookings
                   (= true cancellation count)
        n        : number of bookings scored for that night
    """
    needed = {"StayDate", "p_cancel", "y_true"}
    missing = needed - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    daily = predictions.groupby("StayDate", as_index=False).agg(
        expected=("p_cancel", "sum"),
        actual=("y_true", "sum"),
        n=("p_cancel", "size"),
    )
    return daily


def compute_daily_metrics(predictions: pd.DataFrame) -> dict:
    """Daily Bias % and Daily nRMSE on per-arrival-night cancellation sums.

    Definitions chosen to match Section 3's saved ``daily_bias_pct`` and
    ``daily_nrmse`` so results are directly comparable with the existing
    experiment artifacts:

        expected_cancels(day) = sum of p_cancel over bookings arriving that day
        actual_cancels(day)   = sum of y_true  over bookings arriving that day
        error(day)            = expected - actual
        daily_bias_pct        = 100 * sum(error) / sum(actual)
        daily_nrmse           = sqrt(mean(error**2)) / mean(actual)

    Interpretation
    --------------
    * ``daily_nrmse`` is the PRIMARY model-selection metric. It is the RMSE
      of per-day cancellation-count errors, normalised by mean actual daily
      cancellations. 0 is perfect; 1.0 means the typical daily miss is
      about the same size as a typical day's cancellation count.

    * ``daily_bias_pct`` is signed:
        > 0  model over-predicts cancellations systematically, which under
             the Sigma(1 - p) view means UNDER-estimating arrivals ->
             under-selling (leaves rooms empty).
        < 0  model under-predicts cancellations, OVER-estimating arrivals,
             which risks over-selling (walks).
      So the sign and magnitude map directly to a business failure mode.
    """
    daily = daily_aggregate(predictions)
    actual_total = daily["actual"].sum()
    if actual_total == 0 or len(daily) == 0:
        return {
            "n_days": int(len(daily)),
            "daily_bias_pct": np.nan,
            "daily_nrmse": np.nan,
        }
    error = daily["expected"] - daily["actual"]
    bias_pct = 100.0 * error.sum() / actual_total
    nrmse = float(np.sqrt((error ** 2).mean()) / daily["actual"].mean())
    return {
        "n_days": int(len(daily)),
        "daily_bias_pct": float(bias_pct),
        "daily_nrmse": nrmse,
    }


# -----------------------------------------------------------------------------
# Calibration selection
# -----------------------------------------------------------------------------
def reliability_bins(
    predictions: pd.DataFrame,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Bin predictions and compute per-bin predicted vs observed cancellation rate.

    Parameters
    ----------
    n_bins : int
        Target number of bins. Equal-frequency bins may collapse if the
        predicted distribution has many ties (e.g. a boundary at 0 or 1),
        so the returned frame may have fewer rows than ``n_bins``.
    strategy : {"quantile", "uniform"}
        quantile : equal-frequency bins (recommended for ECE; each bin has
                   the same number of bookings so weighting is unambiguous).
        uniform  : equal-width bins on [0, 1] (useful when the predicted
                   distribution is heavily skewed and we want to see which
                   probability regions are miscalibrated).
    """
    y_true, p = _prepare(predictions)
    if strategy == "quantile":
        # Use exactly n_bins+1 edges from the quantile grid. Open the top edge
        # slightly so the max value falls into the last bin. Do NOT prepend 0
        # separately — np.quantile already covers the min; adding 0.0 or 1.0
        # by hand produces an extra bin when the data's min > 0 or max < 1.
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        edges[-1] = edges[-1] + 1e-9
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
        edges[-1] = 1.0 + 1e-9
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    bin_idx = np.digitize(p, edges[1:-1], right=False)
    frame = pd.DataFrame({"bin": bin_idx, "p": p, "y": y_true})
    rows = []
    for bin_id, group in frame.groupby("bin"):
        if len(group) == 0:
            continue
        rows.append({
            "bin_index": int(bin_id),
            "lower_edge": float(edges[bin_id]),
            "upper_edge": float(edges[bin_id + 1]),
            "count": int(len(group)),
            "predicted_mean": float(group["p"].mean()),
            "observed_rate": float(group["y"].mean()),
        })
    return pd.DataFrame(rows).sort_values("bin_index").reset_index(drop=True)


def compute_ece(
    predictions: pd.DataFrame,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Expected Calibration Error (ECE).

    ECE = sum_k (n_k / N) * | predicted_mean_k - observed_rate_k |

    where k indexes the probability bins. Lower is better; 0 means perfect
    calibration at this binning. The number IS bin-dependent, which is why
    we fix the strategy (equal-frequency, 10 bins) and report both the
    number and the reliability plot. ECE is only used to break ties between
    calibration methods, not as a standalone model-selection metric.
    """
    bins = reliability_bins(predictions, n_bins=n_bins, strategy=strategy)
    total = bins["count"].sum()
    if total == 0:
        return float("nan")
    weights = bins["count"] / total
    gaps = (bins["predicted_mean"] - bins["observed_rate"]).abs()
    return float((weights * gaps).sum())


# -----------------------------------------------------------------------------
# Operating-point reporting
# -----------------------------------------------------------------------------
def pr_curve_points(predictions: pd.DataFrame) -> pd.DataFrame:
    """Full precision/recall curve points for plotting."""
    y_true, p = _prepare(predictions)
    precision, recall, thresholds = precision_recall_curve(y_true, p)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds; pad thresholds so columns align in one DataFrame.
    thresholds = np.concatenate([thresholds, [np.nan]])
    return pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
    })


def confusion_at_threshold(predictions: pd.DataFrame, threshold: float) -> dict:
    """Confusion matrix plus rates at a specific operating threshold.

    Used to describe the operating point of the "flag for review" shortlist
    that feeds Section 5's outreach workflow.
    """
    y_true, p = _prepare(predictions)
    y_hat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "precision": float(precision) if not np.isnan(precision) else float("nan"),
        "recall": float(recall) if not np.isnan(recall) else float("nan"),
        "specificity": float(specificity) if not np.isnan(specificity) else float("nan"),
        "flagged_share": float(y_hat.mean()),
    }


# -----------------------------------------------------------------------------
# Convenience: one-row summary of a candidate
# -----------------------------------------------------------------------------
def summarise_candidate(
    predictions: pd.DataFrame,
    label: str,
    hotel: str,
    calibration: str = "Uncalibrated",
    ece_bins: int = 10,
) -> dict:
    """One-row POOLED summary across all rows of ``predictions``.

    Use this when ``predictions`` is already filtered to the fold set you want
    pooled (e.g. all folds for the uncalibrated-only comparison, or the common
    folds after calling ``align_calibration_variants`` for the calibration
    comparison). Mixing fold sets across variants makes this number misleading.
    """
    ll = compute_log_loss(predictions)
    ap = compute_ap(predictions)
    daily = compute_daily_metrics(predictions)
    ece = compute_ece(predictions, n_bins=ece_bins)
    folds_present = sorted(predictions["fold"].unique().tolist()) if "fold" in predictions.columns else []
    return {
        "hotel": hotel,
        "candidate": label,
        "calibration": calibration,
        "n_bookings": int(len(predictions)),
        "n_days": daily["n_days"],
        "folds": tuple(int(f) for f in folds_present),
        "prevalence": ap["prevalence"],
        # Primary
        "daily_nrmse": daily["daily_nrmse"],
        # Diagnostic
        "daily_bias_pct": daily["daily_bias_pct"],
        "log_loss": ll,
        "ap": ap["ap"],
        "ap_lift": ap["ap_lift"],
        # Calibration
        "ece": ece,
    }


def summarise_candidate_per_fold(
    predictions: pd.DataFrame,
    label: str,
    hotel: str,
    calibration: str = "Uncalibrated",
    ece_bins: int = 10,
) -> pd.DataFrame:
    """Per-fold summary — one row per validation fold in ``predictions``.

    Returns the same columns as ``summarise_candidate`` plus a ``fold`` column.
    Use for temporal-stability analysis: a model whose primary metric is driven
    by one strong fold is risky, even if its pooled number looks good.

    The pooled headline and ``mean of per-fold`` are not identical in general
    because folds have different sizes; the pooled number weights each row
    equally, the per-fold mean weights each fold equally. Report both.
    """
    if "fold" not in predictions.columns:
        raise ValueError("predictions must have a 'fold' column for per-fold summary")
    rows = []
    for fold_id, chunk in predictions.groupby("fold"):
        row = summarise_candidate(
            chunk, label=label, hotel=hotel, calibration=calibration, ece_bins=ece_bins,
        )
        row["fold"] = int(fold_id)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("fold")
        .reset_index(drop=True)
    )


def stability_summary(
    per_fold: pd.DataFrame,
    group_cols: tuple[str, ...] = ("hotel", "candidate", "calibration"),
) -> pd.DataFrame:
    """Mean and std of each metric across folds, per (hotel, candidate, calibration).

    Used to show temporal stability alongside the pooled headline table.
    """
    metric_cols = [
        "daily_nrmse", "daily_bias_pct", "log_loss", "ap", "ap_lift", "ece",
    ]
    agg_dict: dict[str, list[str]] = {col: ["mean", "std"] for col in metric_cols}
    grouped = per_fold.groupby(list(group_cols), as_index=False).agg(agg_dict)
    grouped.columns = [
        "_".join(c for c in col if c).rstrip("_") if isinstance(col, tuple) else col
        for col in grouped.columns.to_flat_index()
    ]
    return grouped
