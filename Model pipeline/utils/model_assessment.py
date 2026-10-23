"""Read-only displays for initial model assessment."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display


def display_initial_assessment(
    experiment_root: Path,
    experiment_registry: list[dict],
    daily_fit_model: str = "XGB",
    daily_fit_fold: int | None = None,
) -> None:
    """Display four assessment outputs from saved experiment artifacts.

    Args:
        experiment_root: Directory containing one saved folder per experiment.
        experiment_registry: Exactly eight hotel/model entries with experiment
            IDs and the ordered 30-predictor groups used in training.
        daily_fit_model: Model whose saved OOF predictions appear in daily plots.
        daily_fit_fold: Optional Validation fold for daily plots. By default,
            use the combined out-of-fold predictions from all four folds.

    Returns:
        None. Tables and figures are displayed; no model is fitted and no file
        is written. All statistics use saved raw probabilities.
    """
    experiment_root = Path(experiment_root)
    model_order = ["Dummy", "LR", "RF", "XGB"]
    expected_pairs = {
        (hotel, model)
        for hotel in ("H1", "H2")
        for model in model_order
    }
    actual_pairs = {
        (entry["hotel"], entry["model_name"])
        for entry in experiment_registry
    }
    if len(experiment_registry) != 8 or actual_pairs != expected_pairs:
        raise ValueError("Assessment requires one run per hotel and model")
    if daily_fit_model not in model_order or (
        daily_fit_fold is not None and daily_fit_fold not in (1, 2, 3, 4)
    ):
        raise ValueError("Unknown daily-fit model or fold")

    # Read and validate complete saved runs. OOF membership must agree between
    # all four models of a hotel before any assessment is displayed.
    metric_parts = []
    prediction_parts = []
    reference_cohorts = {}
    for entry in experiment_registry:
        run_id = entry["experiment_id"]
        directory = experiment_root / run_id
        required_files = {"config.json", "metrics.parquet", "predictions.parquet"}
        if not directory.is_dir() or {
            path.name for path in directory.iterdir()
        } != required_files:
            raise ValueError(f"Incomplete saved experiment: {run_id}")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        metrics = pd.read_parquet(directory / "metrics.parquet")
        predictions = pd.read_parquet(directory / "predictions.parquet")
        if (
            config["experiment_id"] != run_id
            or config["hotel"] != entry["hotel"]
            or config["model_name"] != entry["model_name"]
            or config["selected_features"] != entry["features"]
            or config["feature_count"] != 30
        ):
            raise ValueError(f"Saved experiment identity differs: {run_id}")
        if len(metrics) != 8 or metrics.loc[
            metrics.split.eq("val"), "fold"
        ].tolist() != [1, 2, 3, 4]:
            raise ValueError(f"Missing fold metrics: {run_id}")
        if not predictions.p_cancel.between(0, 1).all():
            raise ValueError(f"Invalid saved probabilities: {run_id}")

        cohort = predictions[
            ["Booking_ID", "fold", "y_true", "CutoffDate", "StayDate"]
        ].sort_values(["Booking_ID", "fold"]).reset_index(drop=True)
        hotel = entry["hotel"]
        if hotel in reference_cohorts:
            pd.testing.assert_frame_equal(cohort, reference_cohorts[hotel])
        else:
            reference_cohorts[hotel] = cohort
        metric_parts.append(metrics)
        prediction_parts.append(predictions)

    all_metrics = pd.concat(metric_parts, ignore_index=True)
    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    validation = all_metrics.loc[all_metrics.split.eq("val")].copy()

    # Performance: equal-weight means and sample standard deviations across folds.
    performance_rows = []
    metric_labels = (
        ("log_loss", "Log Loss"),
        ("daily_nrmse", "Daily nRMSE"),
        ("daily_bias_pct", "Daily Bias %"),
        ("roc_auc", "ROC-AUC"),
    )
    for hotel in ("H1", "H2"):
        for model in model_order:
            fold_rows = validation.loc[
                validation.hotel.eq(hotel) & validation.model_name.eq(model)
            ]
            row = {"Hotel": hotel, "Model": model}
            for column, label in metric_labels:
                mean = fold_rows[column].mean()
                std = fold_rows[column].std(ddof=1)
                row[label] = (
                    f"{mean:+.2f} ± {std:.2f}"
                    if column == "daily_bias_pct"
                    else f"{mean:.4f} ± {std:.4f}"
                )
            performance_rows.append(row)
    display(Markdown("### 1. Performance (mean ± sample std across four Val folds)"))
    display(pd.DataFrame(performance_rows))

    # Stability: two complete per-fold tables for the same Validation cohorts.
    display(Markdown("### 2. Stability"))
    for column, title in (
        ("log_loss", "2a. Validation Log Loss"),
        ("daily_nrmse", "2b. Daily nRMSE"),
    ):
        rows = []
        for hotel in ("H1", "H2"):
            for model in model_order:
                scores = validation.loc[
                    validation.hotel.eq(hotel) & validation.model_name.eq(model)
                ].sort_values("fold")
                rows.append({
                    "Hotel": hotel,
                    "Model": model,
                    **{
                        f"Fold {fold}": float(scores.loc[
                            scores.fold.eq(fold), column
                        ].iloc[0])
                        for fold in range(1, 5)
                    },
                })
        display(Markdown(f"#### {title}"))
        display(pd.DataFrame(rows).round(4))

    # Generalization: each Train/Val pair comes from one fold-fitted pipeline.
    paired = all_metrics.pivot(
        index=["hotel", "model_name", "fold"],
        columns="split",
        values="log_loss",
    ).reset_index()
    paired["gap"] = paired["val"] - paired["train"]
    generalization_rows = []
    for hotel in ("H1", "H2"):
        for model in model_order:
            scores = paired.loc[
                paired.hotel.eq(hotel) & paired.model_name.eq(model)
            ]
            generalization_rows.append({
                "Hotel": hotel,
                "Model": model,
                "Train Log Loss": scores.train.mean(),
                "Validation Log Loss": scores.val.mean(),
                "Gap (Val - Train)": scores.gap.mean(),
            })
    display(Markdown("### 3. Generalization (mean across four matched folds)"))
    display(pd.DataFrame(generalization_rows).round(4))

    # Reliability: fixed-width bins retain their counts and empty bins.
    # All diagrams use saved raw probabilities without calibration fitting.
    edges = np.linspace(0, 1, 11)
    bin_ids = np.searchsorted(
        edges, all_predictions.p_cancel.to_numpy(), side="right"
    ) - 1
    all_predictions["bin_id"] = np.clip(bin_ids, 0, 9)
    bin_labels = [
        f"[{edges[index]:.1f}, {edges[index + 1]:.1f}{']' if index == 9 else ')'}"
        for index in range(10)
    ]
    reliability_by_hotel = {}
    for hotel in ("H1", "H2"):
        hotel_predictions = all_predictions.loc[
            all_predictions.Hotel.eq(hotel)
        ]
        reliability_rows = []
        for model in model_order:
            model_rows = hotel_predictions.loc[
                hotel_predictions.model_name.eq(model)
            ]
            for bin_id in range(10):
                bin_rows = model_rows.loc[model_rows.bin_id.eq(bin_id)]
                reliability_rows.append({
                    "Model": model,
                    "Bin": bin_labels[bin_id],
                    "Bookings": len(bin_rows),
                    "Mean predicted p": bin_rows.p_cancel.mean(),
                    "Observed rate": bin_rows.y_true.mean(),
                })
        reliability_by_hotel[hotel] = pd.DataFrame(reliability_rows)

    display(Markdown("### 4. Probability Reliability & Daily Fit"))
    display(Markdown("#### 4a. Reliability by Probability Bin"))
    for hotel in ("H1", "H2"):
        display(Markdown(f"##### {hotel}"))
        display(reliability_by_hotel[hotel].round(4))

    display(Markdown("#### 4b. Reliability Diagrams"))
    for hotel in ("H1", "H2"):
        reliability = reliability_by_hotel[hotel]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Ideal")
        for model in model_order:
            curve = reliability.loc[
                reliability.Model.eq(model) & reliability.Bookings.gt(0)
            ]
            ax.plot(
                curve["Mean predicted p"],
                curve["Observed rate"],
                marker="o",
                label=model,
            )
        ax.set(
            title=f"{hotel} — OOF reliability",
            xlabel="Mean predicted probability",
            ylabel="Observed cancellation rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
        display(Markdown(f"##### {hotel}"))
        display(fig)
        plt.close(fig)
    # Daily Fit aggregates each selected booking's OOF probability and label
    # by arrival date. The default combines all four disjoint Val cohorts.
    display(Markdown("#### 4c. Daily Fit: Predicted vs Actual Cancellations"))
    for hotel in ("H1", "H2"):
        hotel_predictions = all_predictions.loc[
            all_predictions.Hotel.eq(hotel)
        ]
        daily_predictions = hotel_predictions.loc[
            hotel_predictions.model_name.eq(daily_fit_model)
        ]
        if daily_fit_fold is not None:
            daily_predictions = daily_predictions.loc[
                daily_predictions.fold.eq(daily_fit_fold)
            ]
        if daily_predictions.empty:
            raise ValueError("No saved daily-fit predictions for requested slice")
        daily = daily_predictions.groupby("StayDate", as_index=False).agg(
            predicted=("p_cancel", "sum"),
            actual=("y_true", "sum"),
            n_bookings=("Booking_ID", "size"),
        )
        fold_label = (
            "All four Validation folds"
            if daily_fit_fold is None
            else f"Fold {daily_fit_fold}"
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(daily.StayDate, daily.actual, label="Actual", color="black")
        ax.plot(daily.StayDate, daily.predicted, label="Predicted")
        ax.set(
            title=f"{hotel} / {fold_label} / {daily_fit_model} — daily cancellations",
            xlabel="StayDate = ArrivalDate",
            ylabel="Cancellations among scored bookings",
        )
        ax.legend()
        display(Markdown(f"##### {hotel} — {daily_fit_model}, {fold_label}"))
        display(fig)
        plt.close(fig)
