"""Temporal split invariants, including strict label availability at fit time."""

import sys
from pathlib import Path

import pandas as pd
import pytest


MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from utils.model_split import make_splits  # noqa: E402


CONFIG = {
    "split_column": "CutoffDate",
    "first_fit_date": "2016-07-01",
    "validation_months": 2,
    "n_folds": 2,
    "test_start": "2016-11-01",
    "test_end": "2017-01-01",
}


def booking(key, cutoff, arrival, nights, status, status_date, canceled):
    return {
        "Hotel": "H1", "Booking_ID": key, "CutoffDate": cutoff,
        "ArrivalDate": arrival, "TotalNights": nights,
        "ReservationStatus": status, "ReservationStatusDate": status_date,
        "IsCanceled": canceled,
    }


@pytest.fixture
def bookings():
    # The two checkout rows distinguish actual status date from expected checkout.
    return pd.DataFrame([
        booking("known_cancel", "2016-06-01", "2016-07-15", 2,
                "Canceled", "2016-06-20", 1),
        booking("expected_after_fit", "2016-06-10", "2016-06-28", 5,
                "Check-Out", "2016-06-29", 0),
        booking("status_after_fit", "2016-06-11", "2016-06-20", 2,
                "Check-Out", "2016-07-04", 0),
        booking("exact_fit_date", "2016-06-12", "2016-06-25", 2,
                "No-Show", "2016-07-01", 1),
        booking("val1_zero", "2016-07-01", "2016-07-20", 1,
                "Check-Out", "2016-07-21", 0),
        booking("val1_one", "2016-08-31", "2016-09-10", 1,
                "Canceled", "2016-08-31", 1),
        booking("val2_zero", "2016-09-01", "2016-09-20", 1,
                "Check-Out", "2016-09-21", 0),
        booking("val2_one", "2016-10-31", "2016-10-31", 1,
                "No-Show", "2016-10-31", 1),
        booking("test_start", "2016-11-01", "2016-11-05", 1,
                "Canceled", "2016-11-03", 1),
    ])


def members(result, fold, role):
    table = result["cv_membership"]
    return set(table.loc[table["Fold"].eq(fold) & table["Role"].eq(role), "Booking_ID"])


def test_label_availability_and_boundaries(bookings):
    before = bookings.copy(deep=True)
    result = make_splits(bookings, CONFIG)
    pd.testing.assert_frame_equal(bookings, before)

    assert members(result, 1, "Train") == {"known_cancel"}
    assert members(result, 1, "Validation") == {"val1_zero", "val1_one"}
    assert members(result, 2, "Train") == {
        "known_cancel", "expected_after_fit", "status_after_fit",
        "exact_fit_date", "val1_zero", "val1_one",
    }
    assert members(result, 2, "Validation") == {"val2_zero", "val2_one"}

    eligibility = result["fit_eligibility"]
    fold1 = eligibility.loc[eligibility["Stage"].eq("CV") & eligibility["Fold"].eq(1)]
    availability = fold1.set_index("Booking_ID")["LabelAvailableDate"]
    assert availability["expected_after_fit"] == pd.Timestamp("2016-07-03")
    assert availability["status_after_fit"] == pd.Timestamp("2016-07-04")
    assert availability["exact_fit_date"] == pd.Timestamp("2016-07-01")
    assert set(result["refit_membership"]["Booking_ID"]) == set(bookings["Booking_ID"]) - {"test_start"}
    assert result["partition_membership"].set_index("Booking_ID").loc[
        "test_start", "Partition"] == "Test"


def test_no_test_booking_enters_cv_or_refit(bookings):
    result = make_splits(bookings, CONFIG)
    test_ids = set(result["partition_membership"].loc[
        lambda table: table["Partition"].eq("Test"), "Booking_ID"])
    for name in ("cv_membership", "refit_membership", "calibration_membership"):
        assert not test_ids.intersection(result[name]["Booking_ID"])
    validation = result["cv_membership"].loc[lambda table: table["Role"].eq("Validation")]
    assert validation["Booking_ID"].is_unique
    assert set(validation["Booking_ID"]) == set(result["calibration_membership"]["Booking_ID"])


def test_rejects_booking_at_exclusive_test_end(bookings):
    outside = bookings.copy()
    outside.loc[outside["Booking_ID"].eq("test_start"), "CutoffDate"] = "2017-01-01"
    with pytest.raises(ValueError, match="outside the configured test end"):
        make_splits(outside, CONFIG)
