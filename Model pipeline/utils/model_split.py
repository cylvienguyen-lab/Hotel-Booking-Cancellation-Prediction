"""Feature-independent temporal split and maturity audit."""

import numpy as np
import pandas as pd


def data_split(
    model_data: pd.DataFrame,
    label_dates: pd.DataFrame,
    split_config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Classify bookings once and audit complete-stay CV eligibility.

    ``model_data`` contains both hotels; Test rows carry metadata only and have
    a missing target. ``label_dates`` contains Development outcome dates only.
    Neither input is modified. Each fit requires a past cutoff, a mature label,
    and expected checkout strictly before FitDate. This prevents partially
    observed future stays from adding only early cancellations to training.

    Args:
        model_data: Gold predictors/target for Development and Test metadata.
        label_dates: Development outcome timestamps and matching targets.
        split_config: Fixed history, Test bounds and four fold windows.

    Returns:
        One-row-per-booking membership and an aggregate split audit dict.
    """
    keys = ["Hotel", "Booking_ID"]
    required_model = keys + ["CutoffDate", "ArrivalDate", "TotalNights", "IsCanceled"]
    required_labels = keys + ["ReservationStatusDate", "IsCanceled"]
    for name, frame, required in (
        ("model_data", model_data, required_model),
        ("label_dates", label_dates, required_labels),
    ):
        missing = set(required) - set(frame.columns)
        if missing or frame.empty or frame[keys].isna().any().any():
            raise ValueError(f"{name}: missing columns, rows or keys: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name}: duplicate (Hotel, Booking_ID)")

    start = pd.Timestamp(split_config["history_start"])
    test_start = pd.Timestamp(split_config["test_start"])
    test_end = pd.Timestamp(split_config["test_end"])
    folds = split_config["folds"]
    if len(folds) != 4 or not start < test_start < test_end:
        raise ValueError("Expected four ordered folds and valid history/Test bounds")

    data = model_data[required_model].copy()
    for column in ("CutoffDate", "ArrivalDate"):
        data[column] = pd.to_datetime(data[column], errors="raise").dt.normalize()
    if data[["CutoffDate", "ArrivalDate"]].isna().any().any():
        raise ValueError("Missing cutoff or arrival date")
    if (data["CutoffDate"] < start).any() or (data["CutoffDate"] >= test_end).any():
        raise ValueError("Source rows outside specified history/Test calendar")
    if (data["ArrivalDate"] - data["CutoffDate"]).ne(pd.Timedelta(days=14)).any():
        raise ValueError("Gold records violate the exact T-14 cutoff contract")
    if data["TotalNights"].isna().any() or (data["TotalNights"] < 0).any():
        raise ValueError("Invalid TotalNights")

    data["Partition"] = np.where(data["CutoffDate"] < test_start, "Dev", "Test")
    development = data["Partition"].eq("Dev")
    if data.loc[development, "IsCanceled"].isna().any():
        raise ValueError("Development target is missing")
    if not data.loc[development, "IsCanceled"].isin([0, 1]).all():
        raise ValueError("Development target must be binary")
    if data.loc[~development, "IsCanceled"].notna().any():
        raise ValueError("Test outcomes must be redacted before splitting")

    dates = label_dates[required_labels].copy()
    dates["ReservationStatusDate"] = pd.to_datetime(
        dates["ReservationStatusDate"], errors="raise"
    ).dt.normalize()
    if dates["ReservationStatusDate"].isna().any():
        raise ValueError("Missing Development outcome timestamp")
    if not dates["IsCanceled"].isin([0, 1]).all():
        raise ValueError("Label store target must be binary")
    if len(dates) != int(development.sum()):
        raise ValueError("Label dates must cover exactly Development")

    # Join only Development labels. Test retains null maturity and no outcome.
    joined = data.merge(
        dates, on=keys, how="left", validate="one_to_one", suffixes=("", "_label"),
    )
    if len(joined) != len(data) or joined.loc[development, "IsCanceled_label"].isna().any():
        raise ValueError("Incomplete or changing-grain Development label join")
    if not joined.loc[development, "IsCanceled"].eq(
        joined.loc[development, "IsCanceled_label"]
    ).all():
        raise ValueError("Model dataset and label store disagree on target")
    if joined.loc[~development, "IsCanceled_label"].notna().any():
        raise ValueError("Test label was supplied to the split")

    checkout_date = joined["ArrivalDate"] + pd.to_timedelta(
        joined["TotalNights"], unit="D"
    )
    joined["ExpectedCheckoutDate"] = checkout_date
    joined["LabelAvailableDate"] = joined["ReservationStatusDate"].where(
        joined["IsCanceled"].eq(1),
        pd.concat([joined["ReservationStatusDate"], checkout_date], axis=1).max(axis=1),
    )
    joined.loc[~development, "LabelAvailableDate"] = pd.NaT
    if joined.loc[development, "LabelAvailableDate"].isna().any():
        raise ValueError("Development label availability is missing")
    if (joined.loc[development, "LabelAvailableDate"] < joined.loc[development, "CutoffDate"]).any():
        raise ValueError("Outcome date precedes T-14 cutoff")

    joined["ValFold"] = pd.Series(pd.NA, index=joined.index, dtype="Int64")
    previous_end = None
    fold_audit = []
    for number, fold in enumerate(folds, start=1):
        fit_date = pd.Timestamp(fold["fit_date"])
        val_end = pd.Timestamp(fold["val_end"])
        if fit_date >= val_end or val_end > test_start or (
            previous_end is not None and fit_date != previous_end
        ):
            raise ValueError("Overlapping, unordered or Test-overlapping folds")
        validation = development & joined["CutoffDate"].ge(fit_date) & joined["CutoffDate"].lt(val_end)
        if joined.loc[validation, "ValFold"].notna().any():
            raise ValueError("Validation fold overlap")
        joined.loc[validation, "ValFold"] = number
        if joined.loc[validation, "LabelAvailableDate"].ge(test_start).any():
            raise ValueError(f"Fold {number}: validation label immature at final refit")

        # Recompute eligibility at every fit. A booking with a mature early
        # cancellation still waits until its planned stay has ended.
        past = development & joined["CutoffDate"].lt(fit_date)
        mature = joined["LabelAvailableDate"].lt(fit_date)
        stay_complete = joined["ExpectedCheckoutDate"].lt(fit_date)
        train = past & mature & stay_complete
        hotel_fit_counts = {}
        for hotel in sorted(joined["Hotel"].unique()):
            hotel_mask = joined["Hotel"].eq(hotel)
            hotel_fit_counts[hotel] = {
                "train_eligible": int((hotel_mask & train).sum()),
                "excluded_future_cutoff": int((hotel_mask & development & ~past).sum()),
                "excluded_immature_label": int((hotel_mask & past & ~mature).sum()),
                "excluded_unfinished_stay": int(
                    (hotel_mask & past & mature & ~stay_complete).sum()
                ),
                "validation": int((hotel_mask & validation).sum()),
            }
        fold_audit.append({
            "fold": number,
            "fit_date": fit_date.strftime("%Y-%m-%d"),
            "val_end": val_end.strftime("%Y-%m-%d"),
            "train_eligible": int(train.sum()),
            "excluded_future_cutoff": int((development & ~past).sum()),
            "excluded_immature_label": int((past & ~mature).sum()),
            "excluded_unfinished_stay": int((past & mature & ~stay_complete).sum()),
            "validation": int(validation.sum()),
            "validation_positives": int(joined.loc[validation, "IsCanceled"].sum()),
            "by_hotel": hotel_fit_counts,
        })
        previous_end = val_end

    joined["FinalRefit"] = (
        development
        & joined["CutoffDate"].lt(test_start)
        & joined["LabelAvailableDate"].lt(test_start)
        & joined["ExpectedCheckoutDate"].lt(test_start)
    ).astype("int8")
    membership = joined[
        keys + ["CutoffDate", "ArrivalDate", "ExpectedCheckoutDate",
                "LabelAvailableDate", "Partition", "ValFold", "FinalRefit"]
    ].sort_values(keys).reset_index(drop=True)
    if membership.duplicated(keys).any() or membership.loc[
        membership["Partition"].eq("Test"), "ValFold"
    ].notna().any():
        raise AssertionError("Membership key or Test/CV separation failed")

    dev_rows = joined.loc[development]
    audit = {
        "total_bookings": len(joined),
        "unique_keys": len(membership),
        "t14_deviations": 0,
        "development": int(development.sum()),
        "test_metadata_only": int((~development).sum()),
        "final_refit_eligible_count_only": int(joined["FinalRefit"].sum()),
        "by_hotel": {},
        "folds": fold_audit,
        "training_rule": (
            "Partition == Dev and CutoffDate < FitDate and "
            "LabelAvailableDate < FitDate and ExpectedCheckoutDate < FitDate"
        ),
        "final_refit_rule": (
            "Partition == Dev and CutoffDate < TestStart and "
            "LabelAvailableDate < TestStart and ExpectedCheckoutDate < TestStart"
        ),
    }
    for hotel in sorted(joined["Hotel"].unique()):
        hotel_rows = joined["Hotel"].eq(hotel)
        hotel_dev = dev_rows.loc[dev_rows["Hotel"].eq(hotel)]
        audit["by_hotel"][hotel] = {
            "dev": len(hotel_dev),
            "test_metadata_only": int((hotel_rows & ~development).sum()),
            "dev_positives": int(hotel_dev["IsCanceled"].sum()),
            "dev_prevalence": float(hotel_dev["IsCanceled"].mean()),
            "validation_by_fold": {
                str(fold): int((hotel_rows & joined["ValFold"].eq(fold).fillna(False)).sum())
                for fold in range(1, 5)
            },
        }
    return membership, audit
