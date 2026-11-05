"""Run one initial model across the four fixed temporal folds."""

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline


def run_experiment(
    model_data: pd.DataFrame,
    membership: pd.DataFrame,
    hotel: str,
    selected_features: dict,
    estimator,
    preprocessor,
    experiment_id: str,
    split_id: str,
    split_config: dict,
    output_root: Path,
    experiment_version: int,
) -> dict:
    """Fit each fold in isolation and save raw Train/Val scores and OOF risk.

    Training uses only Development bookings whose cutoff, label availability
    and expected checkout are before that fold's fit date. Validation is the
    full cutoff cohort assigned by the master membership. Saved predictions
    are Val only;
    ``p_cancel`` is the estimator's probability for class 1.

    Args:
        model_data: Gold rows with Development predictors and target.
        membership: Shared booking-level temporal split table.
        hotel: H1 or H2, fitted separately.
        selected_features: All Gold predictors in numeric/binary/categorical roles.
        estimator: Unfitted sklearn-compatible probability classifier.
        preprocessor: Unfitted transformer or Dummy passthrough.
        experiment_id: Unique, versioned hotel/model run ID.
        split_id: Version of the shared split membership.
        split_config: Four fixed fit and validation windows.
        output_root: Parent of versioned experiment directories.
        experiment_version: Positive version of the experiment artifacts.

    Returns:
        Raw fold metrics, validation predictions and the saved config dict.
    """
    model_name = experiment_id.split("_")[1]
    if hotel not in {"H1", "H2"} or model_name not in {"Dummy", "LR", "RF", "XGB"}:
        raise ValueError("Unsupported hotel/model experiment")
    if experiment_version < 1:
        raise ValueError("Experiment version must be positive")
    expected_id = f"{hotel}_{model_name}_v{experiment_version}"
    if experiment_id != expected_id:
        raise ValueError(f"Experiment ID must be {expected_id}")
    output_root = Path(output_root)
    target = output_root / experiment_id
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")

    keys = ["Hotel", "Booking_ID"]
    features = [name for group in ("numeric", "binary", "categorical") for name in selected_features[group]]
    required = keys + features + ["IsCanceled"]
    if set(required) - set(model_data) or model_data.duplicated(keys).any():
        raise ValueError("Model data missing features/target or duplicate booking keys")
    if membership.duplicated(keys).any():
        raise ValueError("Duplicate master membership keys")
    required_membership = {
        "Partition", "CutoffDate", "LabelAvailableDate",
        "ExpectedCheckoutDate", "ValFold", "ArrivalDate",
    }
    if required_membership - set(membership.columns):
        raise ValueError("Master membership lacks Rule B dates or folds")
    dev_membership = membership.loc[
        membership["Hotel"].eq(hotel) & membership["Partition"].eq("Dev")
    ].copy()
    source = model_data.loc[
        model_data["Hotel"].eq(hotel), required
    ]
    data = dev_membership.merge(source, on=keys, how="left", validate="one_to_one")
    if len(data) != len(dev_membership) or data["IsCanceled"].isna().any():
        raise ValueError("Development feature join lost bookings or labels")
    if not data["IsCanceled"].isin([0, 1]).all():
        raise ValueError("Target must be 0/1")

    metric_rows = []
    prediction_parts = []
    for fold_number, fold_config in enumerate(split_config["folds"], start=1):
        fit_date = pd.Timestamp(fold_config["fit_date"])
        train_mask = (
            data["CutoffDate"].lt(fit_date)
            & data["LabelAvailableDate"].lt(fit_date)
            & data["ExpectedCheckoutDate"].lt(fit_date)
        )
        val_mask = data["ValFold"].eq(fold_number).fillna(False)
        train = data.loc[train_mask].copy()
        val = data.loc[val_mask].copy()
        if train.empty or val.empty or train["IsCanceled"].nunique() != 2:
            raise ValueError(f"Fold {fold_number}: empty cohort or single-class training")
        if set(map(tuple, train[keys].to_numpy())).intersection(
            set(map(tuple, val[keys].to_numpy()))
        ):
            raise AssertionError("Training and validation overlap within a fold")
        if not val["CutoffDate"].ge(fit_date).all() or not val["CutoffDate"].lt(
            pd.Timestamp(fold_config["val_end"])
        ).all():
            raise AssertionError("Validation cutoff outside specified window")

        # Cloning and fitting here keeps imputation, scaling and categories inside
        # the training fold. No saved model or full-Development fit is produced.
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor) if preprocessor != "passthrough" else "passthrough"),
            ("model", clone(estimator)),
        ])
        pipeline.fit(train[features], train["IsCanceled"].astype(int))
        classes = list(pipeline.named_steps["model"].classes_)
        if classes != [0, 1]:
            raise AssertionError(f"Unexpected probability class order: {classes}")

        for split_name, cohort in (("train", train), ("val", val)):
            y_true = cohort["IsCanceled"].astype(int).to_numpy()
            probabilities = pipeline.predict_proba(cohort[features])[:, classes.index(1)]
            if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError("Non-finite or out-of-range cancellation probability")
            score = {
                "experiment_id": experiment_id,
                "split_id": split_id,
                "hotel": hotel,
                "model_name": model_name,
                "fold": fold_number,
                "split": split_name,
                "n_bookings": len(cohort),
                "n_positive": int(y_true.sum()),
                "n_days": pd.NA,
                "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
                "roc_auc": np.nan,
                "daily_bias_pct": np.nan,
                "daily_nrmse": np.nan,
                "metric_warning": "",
            }
            if split_name == "val":
                if len(np.unique(y_true)) == 2:
                    score["roc_auc"] = float(roc_auc_score(y_true, probabilities))
                else:
                    score["metric_warning"] = "ROC-AUC undefined: one validation class"
                predictions = cohort[keys + ["CutoffDate", "ArrivalDate"]].copy()
                predictions = predictions.rename(columns={"ArrivalDate": "StayDate"})
                predictions["fold"] = fold_number
                predictions["y_true"] = y_true
                predictions["p_cancel"] = probabilities
                predictions["experiment_id"] = experiment_id
                predictions["model_name"] = model_name
                predictions["split_id"] = split_id
                prediction_parts.append(predictions)

                # Each booking contributes once to its arrival date, including
                # dates with zero actual cancellations.
                daily = predictions.groupby("StayDate", as_index=False).agg(
                    predicted=("p_cancel", "sum"),
                    actual=("y_true", "sum"),
                )
                score["n_days"] = len(daily)
                actual_total = daily["actual"].sum()
                if actual_total > 0:
                    error = daily["predicted"] - daily["actual"]
                    score["daily_bias_pct"] = float(100 * error.sum() / actual_total)
                    score["daily_nrmse"] = float(np.sqrt(np.mean(error**2)) / daily["actual"].mean())
                else:
                    score["metric_warning"] += "; daily metrics undefined: zero actual cancellations"
            metric_rows.append(score)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions.duplicated(keys).any() or len(predictions) != int(data["ValFold"].notna().sum()):
        raise AssertionError("OOF predictions do not match the validation membership")
    config = {
        "experiment_id": experiment_id,
        "split_id": split_id,
        "hotel": hotel,
        "model_name": model_name,
        "experiment_version": experiment_version,
        "feature_count": len(features),
        "selected_features": selected_features,
        "estimator": type(estimator).__name__,
        # XGBoost reports its default ``missing`` value as NaN. Encode that
        # explicitly as text so config.json stays valid strict JSON.
        "estimator_params": {
            name: "NaN" if isinstance(value, float) and np.isnan(value) else value
            for name, value in estimator.get_params().items()
        },
        "folds": split_config["folds"],
        "training_rule": (
            "Dev; CutoffDate < FitDate; LabelAvailableDate < FitDate; "
            "ExpectedCheckoutDate < FitDate"
        ),
        "stay_date_rule": "ArrivalDate, one row per booking",
    }

    # Write a complete run into a temporary sibling, then publish it by rename.
    # Existing run IDs are never overwritten, including on notebook reruns.
    output_root.mkdir(parents=True, exist_ok=True)
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
        # An incomplete temporary directory is intentionally retained for audit.
        raise
    return {"metrics": metrics, "predictions": predictions, "config": config}
