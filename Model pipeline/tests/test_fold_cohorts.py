"""The shared runner must score the saved booking cohort in its saved order."""

import sys
from pathlib import Path

import pandas as pd
import unittest


MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / "src"))

from folds import fold_cohorts, run_fold  # noqa: E402


def check_dummy_oof_matches_membership_and_labels():
    dev = pd.DataFrame({
        "Hotel": ["H1"] * 4,
        "Booking_ID": ["a", "b", "c", "d"],
        "CutoffDate": pd.to_datetime(["2016-06-01", "2016-06-02", "2016-07-02", "2016-07-03"]),
        "IsCanceled": [0, 1, 1, 0],
        "LeadTime": [10, 20, 30, 40],
    })
    membership = pd.DataFrame({
        "Hotel": ["H1"] * 4,
        "Booking_ID": ["b", "a", "d", "c"],
        "Fold": [1] * 4,
        "FitDate": pd.to_datetime(["2016-07-01"] * 4),
        "Role": ["Train", "Train", "Validation", "Validation"],
    })
    groups = {"numeric": ["LeadTime"], "binary": [], "categorical": []}
    result = run_fold("H1", 1, "dummy", "prior", groups, {"strategy": "prior"},
                      dev, membership, validation_months=2)
    oof = result["predictions"]
    assert oof["Booking_ID"].tolist() == ["d", "c"]
    assert oof["IsCanceled"].tolist() == [0, 1]
    assert oof["RawProbability"].tolist() == [0.5, 0.5]
    assert result["metrics"]["ValidationRows"] == 2


def check_missing_membership_key_fails_before_fitting():
    dev = pd.DataFrame({
        "Hotel": ["H1"], "Booking_ID": ["a"],
        "CutoffDate": pd.to_datetime(["2016-06-01"]),
        "IsCanceled": [0], "LeadTime": [10],
    })
    membership = pd.DataFrame({
        "Hotel": ["H1", "H1"], "Booking_ID": ["a", "missing"],
        "Fold": [1, 1], "FitDate": pd.to_datetime(["2016-07-01"] * 2),
        "Role": ["Train", "Validation"],
    })
    with unittest.TestCase().assertRaisesRegex(ValueError, "unmatched or empty Validation cohort"):
        fold_cohorts(dev, membership, 1, ["LeadTime"])


class TestFoldCohorts(unittest.TestCase):
    def test_dummy_oof_matches_membership_and_labels(self):
        check_dummy_oof_matches_membership_and_labels()

    def test_missing_membership_key_fails_before_fitting(self):
        check_missing_membership_key_fails_before_fitting()
