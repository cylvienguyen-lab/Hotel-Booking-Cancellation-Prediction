"""Synthetic boundary fixtures and independent audits of development memberships."""
import sys
import unittest
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / 'src'))
from utils.model_split import make_splits
import split

CONFIG = {'split_column': 'CutoffDate', 'first_fit_date': '2016-07-01',
          'validation_months': 2, 'n_folds': 2,
          'test_start': '2016-11-01', 'test_end': '2017-01-01'}


def booking(key, cutoff, arrival, nights, status, status_date, canceled):
    return dict(Hotel='H1', Booking_ID=key, CutoffDate=cutoff, ArrivalDate=arrival,
                TotalNights=nights, ReservationStatus=status,
                ReservationStatusDate=status_date, IsCanceled=canceled)


def fixture():
    # All rows are synthetic; none are read from a real held-out partition.
    return pd.DataFrame([
        booking('known_cancel', '2016-06-01', '2016-07-15', 2, 'Canceled', '2016-06-20', 1),
        booking('expected_after_fit', '2016-06-10', '2016-06-28', 5, 'Check-Out', '2016-06-29', 0),
        booking('status_after_fit', '2016-06-11', '2016-06-20', 2, 'Check-Out', '2016-07-04', 0),
        booking('exact_fit_date', '2016-06-12', '2016-06-25', 2, 'No-Show', '2016-07-01', 1),
        booking('val1_zero', '2016-07-01', '2016-07-20', 1, 'Check-Out', '2016-07-21', 0),
        booking('val1_one', '2016-08-31', '2016-09-10', 1, 'Canceled', '2016-08-31', 1),
        booking('val2_zero', '2016-09-01', '2016-09-20', 1, 'Check-Out', '2016-09-21', 0),
        booking('val2_one', '2016-10-31', '2016-10-31', 1, 'No-Show', '2016-10-31', 1),
        booking('held_out_boundary', '2016-11-01', '2016-11-05', 1, 'Canceled', '2016-11-03', 1),
    ])


def members(result, fold, role):
    table = result['cv_membership']
    return set(table.loc[table.Fold.eq(fold) & table.Role.eq(role), 'Booking_ID'])


class TestModelSplit(unittest.TestCase):
    def setUp(self):
        self.rows = fixture()

    def test_strict_label_availability_and_half_open_boundaries(self):
        before = self.rows.copy(deep=True)
        result = make_splits(self.rows, CONFIG)
        pd.testing.assert_frame_equal(self.rows, before)
        self.assertEqual(members(result, 1, 'Train'), {'known_cancel'})
        self.assertEqual(members(result, 1, 'Validation'), {'val1_zero', 'val1_one'})
        self.assertEqual(members(result, 2, 'Train'), {
            'known_cancel', 'expected_after_fit', 'status_after_fit',
            'exact_fit_date', 'val1_zero', 'val1_one'})
        self.assertEqual(members(result, 2, 'Validation'), {'val2_zero', 'val2_one'})
        table = result['fit_eligibility']
        availability = table.loc[table.Stage.eq('CV') & table.Fold.eq(1).fillna(False)].set_index('Booking_ID').LabelAvailableDate
        self.assertEqual(availability['expected_after_fit'], pd.Timestamp('2016-07-03'))
        self.assertEqual(availability['status_after_fit'], pd.Timestamp('2016-07-04'))
        self.assertEqual(availability['exact_fit_date'], pd.Timestamp('2016-07-01'))

    def test_partition_exclusion_and_calibration_union(self):
        result = make_splits(self.rows, CONFIG)
        for name in ('cv_membership', 'refit_membership', 'calibration_membership'):
            self.assertNotIn('held_out_boundary', set(result[name].Booking_ID))
        valid = result['cv_membership'].loc[lambda t: t.Role.eq('Validation')]
        self.assertTrue(valid.Booking_ID.is_unique)
        self.assertEqual(set(valid.Booking_ID), set(result['calibration_membership'].Booking_ID))

    def test_exclusive_end_rejected(self):
        outside = self.rows.copy()
        outside.loc[outside.Booking_ID.eq('held_out_boundary'), 'CutoffDate'] = '2017-01-01'
        with self.assertRaisesRegex(ValueError, 'outside the configured test end'):
            make_splits(outside, CONFIG)

    def test_notebook_split_rejects_out_of_development_rows(self):
        outside = self.rows.copy()
        outside['CutoffDate'] = pd.to_datetime(outside.CutoffDate)
        with self.assertRaisesRegex(ValueError, 'development bookings only'):
            split.run_split('H1', outside, {'split': CONFIG}, Path('unused'))


def development_suite(source, result, config):
    """Re-derive memberships with independent filters; return one test per invariant."""
    checks = []
    data = source.copy(deep=True)
    before = source.copy(deep=True)
    expected_checkout = data.ArrivalDate + pd.to_timedelta(data.TotalNights, unit='D')
    data['Ready'] = data.ReservationStatusDate
    checkout = data.ReservationStatus.eq('Check-Out')
    data.loc[checkout, 'Ready'] = pd.concat(
        [data.loc[checkout, 'ReservationStatusDate'], expected_checkout.loc[checkout]], axis=1).max(axis=1)
    end = pd.Timestamp(config['test_start'])
    cv, eligibility = result['cv_membership'], result['fit_eligibility']
    hotel = source.Hotel.iloc[0]

    def equal_ids(name, actual, expected):
        values = list(actual)
        checks.append((name, len(values) == len(set(values)) and set(values) == set(expected)))

    checks.append(('development date bound', data.CutoffDate.lt(end).all()))
    equal_ids('development coverage', result['partition_membership'].Booking_ID, data.Booking_ID)
    checks.append(('development partition only', result['partition_membership'].Partition.eq('Development').all()))
    validation_ids = []
    for fold in range(1, config['n_folds'] + 1):
        fit_date = pd.Timestamp(config['first_fit_date']) + pd.DateOffset(months=(fold - 1) * config['validation_months'])
        valid_end = fit_date + pd.DateOffset(months=config['validation_months'])
        candidates = data.loc[data.CutoffDate.lt(fit_date)]
        train = candidates.loc[candidates.Ready.lt(fit_date)]
        valid = data.loc[data.CutoffDate.ge(fit_date) & data.CutoffDate.lt(valid_end)]
        train_ids, valid_ids = members(result, fold, 'Train'), members(result, fold, 'Validation')
        equal_ids(f'fold {fold} eligible train', cv.loc[cv.Fold.eq(fold) & cv.Role.eq('Train'), 'Booking_ID'], train.Booking_ID)
        equal_ids(f'fold {fold} complete validation', cv.loc[cv.Fold.eq(fold) & cv.Role.eq('Validation'), 'Booking_ID'], valid.Booking_ID)
        checks.append((f'fold {fold} disjoint', not train_ids & valid_ids))
        checks.append((f'fold {fold} nonempty', bool(train_ids) and bool(valid_ids)))
        checks.append((f'fold {fold} fit dates', cv.loc[cv.Fold.eq(fold), 'FitDate'].eq(fit_date).all()))
        audit = eligibility.loc[eligibility.Stage.eq('CV') & eligibility.Fold.eq(fold).fillna(False)]
        equal_ids(f'fold {fold} eligibility candidates', audit.Booking_ID, candidates.Booking_ID)
        equal_ids(f'fold {fold} eligibility agrees', audit.loc[audit.Eligible, 'Booking_ID'], train.Booking_ID)
        expected = audit.merge(data[['Booking_ID', 'Ready']], on='Booking_ID', validate='one_to_one')
        checks.append((f'fold {fold} availability dates', expected.LabelAvailableDate.eq(expected.Ready).all()))
        checks.append((f'fold {fold} validation labels ready', valid.Ready.lt(end).all()))
        validation_ids.extend(valid.Booking_ID)
    equal_ids('refit eligibility', result['refit_membership'].Booking_ID, data.loc[data.Ready.lt(end), 'Booking_ID'])
    equal_ids('calibration unique union', result['calibration_membership'].Booking_ID, validation_ids)
    checks.append(('source unchanged', source.equals(before)))
    suite = unittest.TestSuite()
    for name, passed in checks:
        def check(passed=bool(passed), name=name):
            if not passed:
                raise AssertionError(f'{hotel}: {name}')
        suite.addTest(unittest.FunctionTestCase(check, description=f'{hotel}: {name}'))
    return suite
