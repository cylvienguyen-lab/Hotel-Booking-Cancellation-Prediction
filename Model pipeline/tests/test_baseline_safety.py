"""Rehearsal isolation and Notebook 01 call restrictions, using synthetic inputs only."""
import ast
import inspect
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / 'src'))
import final


class TestBaselineSafety(unittest.TestCase):
    def test_calibration_restores_legacy_fold_booking_order(self):
        import calibration
        oof = pd.DataFrame({'Hotel': ['H1'] * 4, 'Booking_ID': ['b', 'a', 'd', 'c'],
                            'Fold': [1, 1, 2, 2], 'IsCanceled': [1, 0, 1, 0],
                            'PredictionDate': pd.to_datetime(['2016-07-01'] * 2 + ['2016-09-01'] * 2),
                            'RawProbability': [.2, .1, .2, .1]})
        dates = oof[['Hotel', 'Booking_ID']].assign(LabelAvailableDate=pd.to_datetime(
            ['2016-07-05'] * 2 + ['2016-09-05'] * 2))
        metrics = pd.DataFrame({'Fold': [1, 2], 'FitDate': pd.to_datetime(['2016-07-01', '2016-09-01']),
                                'ValidationEnd': pd.to_datetime(['2016-09-01', '2016-11-01']),
                                'LogLoss': [.3, .3], 'DummyLogLoss': [.4, .4]})
        before = oof.copy(deep=True)
        with patch.object(calibration, 'fit_and_apply', side_effect=lambda method, ready, scores: ({}, scores.to_numpy())) as fit:
            _, calibrated, _ = calibration.evaluate_folds('H1', 'sigmoid', oof, dates, metrics, '2017-04-18')
        self.assertEqual(fit.call_args.args[1].Booking_ID.tolist(), ['a', 'b'])
        self.assertEqual(calibrated.Booking_ID.tolist(), ['c', 'd'])
        pd.testing.assert_frame_equal(oof, before)

    def test_optional_run_metadata_does_not_change_numeric_results(self):
        from baseline import Workflow
        import common
        with TemporaryDirectory() as directory:
            flow = object.__new__(Workflow)
            flow.output_root, flow.run_id, flow.completed = Path(directory), 'synthetic', []
            expected = pd.DataFrame({'Variant': ['Dummy', 'F1'], 'ConfigID': [None, 'rf18_07'],
                                     'LogLoss': [0.2772, 0.2649]})
            flow.save('comparison', {'metrics.csv': expected})
            observed = common.read_tables(flow.output_root / 'comparison')['metrics.csv']
            self.assertEqual(observed.ConfigID.tolist(), ['', 'rf18_07'])
            pd.testing.assert_series_equal(observed.LogLoss, expected.LogLoss, check_exact=True)

    def test_non_rehearsal_rejected_before_any_input_or_io(self):
        with patch.object(final.common, 'load_contract', side_effect=AssertionError('unexpected IO')):
            with self.assertRaisesRegex(ValueError, 'rehearsal only'):
                final.run_final('H1', None, None, None, None, None, None, rehearsal=False)

    def test_rehearsal_api_has_no_held_out_input(self):
        self.assertNotIn('test', inspect.signature(final.run_final).parameters)

    def test_notebook_only_calls_rehearsal(self):
        notebook = json.loads((MODEL / 'model_train_main.ipynb').read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'run_final':
                    flag = next((kw.value for kw in node.keywords if kw.arg == 'rehearsal'), None)
                    self.assertIsInstance(flag, ast.Constant)
                    self.assertIs(flag.value, True)
        source = (MODEL / 'src' / 'baseline.py').read_text(encoding='utf-8')
        self.assertNotIn('rehearsal=False', source)

    def test_no_held_out_prediction_files_in_outputs(self):
        # Inspect names only; never read predictions left by an earlier run.
        names = [p.name for p in (MODEL / 'outputs').rglob('*') if p.is_file()]
        self.assertFalse(any('test' in name.lower() and 'prediction' in name.lower() for name in names))

    def test_rehearsal_uses_only_fold_four_cohort_and_validation_filename(self):
        keys = pd.DataFrame({'Hotel': ['H1'], 'Booking_ID': ['synthetic'],
                             'PredictionDate': pd.to_datetime(['2017-01-02']),
                             'IsCanceled': [0], 'RawProbability': [.1],
                             'ConfigID': ['manual'], 'FeatureSet': ['f1_17'], 'Model': ['dummy']})
        metric_names = ['LogLoss', 'BrierScore', 'ROC_AUC', 'AveragePrecision']
        values = dict(zip(metric_names, [.2, .1, .5, .2]))
        config = {'split': {'n_folds': 4, 'test_start': '2017-04-18'},
                  'features': 'f1_17', 'feature_sets': {'f1_17': {'numeric': ['LeadTime'], 'binary': [], 'categorical': []}},
                  'model': 'dummy', 'hyperparams': 'manual', 'calibration': 'raw',
                  'train_lookback_months': None, 'models': {'dummy': {'estimator': 'dummy', 'params': {'strategy': 'prior'}}},
                  'evaluation': {'metrics': metric_names}}
        dev = pd.DataFrame({'CutoffDate': pd.to_datetime(['2016-12-01'])})
        ready = keys[['Hotel', 'Booking_ID']].assign(LabelAvailableDate=pd.Timestamp('2017-01-03'))
        oof = keys.assign(Fold=4)
        cv = {'oof_predictions': oof,
              'fold_metrics': pd.DataFrame([{'Fold': 4, **values}]),
              'calibration_metrics': pd.DataFrame([{'Fold': 4, **values}])}
        train = pd.DataFrame({'CutoffDate': pd.to_datetime(['2016-12-01'])})
        with patch.object(final.common, 'load_contract', return_value={}), \
             patch.object(final.common, 'check_features'), \
             patch.object(final.folds, 'fold_cohorts', return_value=(pd.Timestamp('2017-01-01'), train, pd.DataFrame())) as cohort, \
             patch.object(final, 'refit_and_score', return_value=(None, {}, {k: values for k in ['raw', 'calibrated', 'dummy']}, pd.DataFrame())) as score, \
             patch.object(final.common, 'write_tables') as write, \
             patch.object(final.common, 'write_json'), \
             patch.object(pd, 'read_csv', side_effect=AssertionError('unexpected file read')):
            final.run_final('H1', config, dev, {'fit_eligibility': ready, 'cv_membership': pd.DataFrame()},
                            cv, {}, Path('unused'), rehearsal=True)
            self.assertEqual(cohort.call_args.args[2], 4)
            self.assertEqual(list(write.call_args.args[1]), ['validation_predictions.csv'])
            self.assertTrue(score.call_args.args[-1].empty)  # No previous-fold OOF in this synthetic fixture.
