"""Notebook 01 stages. All modelling uses the existing fold and estimator utilities."""
import copy
from contextlib import redirect_stdout
import io
import json
import pickle
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import calibration
import common
import error_analysis
import final
import folds
import split
import tune
from utils import model_evaluation, model_pipeline

METRICS = ['LogLoss', 'BrierScore', 'ROC_AUC', 'AveragePrecision']


def summarize(metrics, predictions, label='Variant'):
    rows = []
    for (hotel, variant), windows in metrics.groupby(['Hotel', label], sort=False):
        oof = predictions.loc[predictions['Hotel'].eq(hotel) & predictions[label].eq(variant)]
        pooled = model_evaluation.evaluate_predictions(oof['IsCanceled'], oof['RawProbability'])
        rows.append({'Hotel': hotel, label: variant,
                     **{metric: float(windows[metric].mean()) for metric in METRICS},
                     'PooledOOFLogLoss': pooled['LogLoss'],
                     'FoldsBeatingDummy': int(windows['GainVsDummy'].gt(common.TOLERANCE).sum()),
                     'MeanTrainLogLoss': float(windows['TrainLogLoss'].mean()),
                     'MeanGap': float(windows['GeneralizationGap'].mean()),
                     'MeanAbsoluteWindowBias': float(windows['ProbabilityBias'].abs().mean())})
    return pd.DataFrame(rows)


def reliability(predictions, label='Variant', probability='RawProbability', groups=5):
    parts = []
    for (hotel, fold, variant), frame in predictions.groupby(['Hotel', 'Fold', label]):
        scores = frame[probability]
        bins = pd.qcut(scores, q=groups, duplicates='drop') if scores.nunique() > 1 else None
        codes = np.ones(len(frame), dtype=int) if bins is None or bins.isna().all() else bins.cat.codes + 1
        table = frame.assign(ScoreGroup=np.asarray(codes)).groupby('ScoreGroup').agg(
            Rows=('IsCanceled', 'size'), Positives=('IsCanceled', 'sum'),
            MeanProbability=(probability, 'mean'), MinProbability=(probability, 'min'),
            MaxProbability=(probability, 'max')).reset_index()
        table['ActualRate'] = table['Positives'] / table['Rows']
        parts.append(table.assign(Hotel=hotel, Fold=fold, **{label: variant}))
    return pd.concat(parts, ignore_index=True)


class Workflow:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding='utf-8'))
        self.output_root = common.OUTPUTS / 'baseline01'
        self.run_id = datetime.now(timezone.utc).isoformat()
        self.data, self.splits, self.runs, self.tuned = {}, {}, {}, {}
        self.completed = []
        for groups in self.config['feature_sets'].values():
            common.check_features(groups, common.load_contract())
        common.write_json(self.output_root / 'run_config.json', self.config)
        self.save('setup', {'config_summary.csv': pd.DataFrame([{
            'Hotels': ', '.join(self.config['hotels']), 'Folds': self.config['split']['n_folds'],
            'PrimaryMetric': self.config['evaluation']['primary_metric'],
            'CandidateModel': self.config['model'], 'CandidateFeatures': self.config['features'],
            'Calibration': self.config['calibration']}])})

    def save(self, stage, tables):
        serialized = {}
        for name, table in tables.items():
            table = table.rename_axis(columns=None).copy()
            # CSV represents absent string metadata as an empty string, not a float NaN.
            for column in table.select_dtypes(include='object'):
                table[column] = table[column].fillna('')
            serialized[name] = table
        common.write_tables(self.output_root / stage, serialized, extra={'run_id': self.run_id})
        if stage not in self.completed:
            self.completed.append(stage)
        common.write_json(self.output_root / 'run_manifest.json', {
            'run_id': self.run_id, 'completed': self.completed,
            'held_out_access': 'CutoffDate count/min/max only; no labels, features or predictions'})

    def start_spark(self):
        from pyspark.sql import SparkSession
        self.spark = (SparkSession.builder.appName('notebook01_baseline')
                      .master('local[2]').getOrCreate())
        self.spark.sparkContext.setLogLevel('ERROR')

    def load(self):
        rows, partitions = [], []
        for hotel in self.config['hotels']:
            dev = common.load_gold(hotel, self.spark, self.config['split']['test_start'])
            if not dev['Hotel'].eq(hotel).all() or dev[common.KEYS].duplicated().any():
                raise AssertionError('Gold hotel/key contract failed')
            if not dev['IsCanceled'].isin([0, 1]).all():
                raise AssertionError('Development target must be binary')
            self.data[hotel] = dev
            counts = common.held_out_calendar_counts(hotel, self.spark, self.config['split'])
            partitions.append(counts)
            rows.append({'Hotel': hotel, 'TotalBookings': len(dev) + counts['Rows'],
                         'DevelopmentBookings': len(dev), 'DevelopmentPositives': int(dev.IsCanceled.sum()),
                         'DevelopmentCancelRate': float(dev.IsCanceled.mean()),
                         'DevelopmentMinCutoffDate': dev.CutoffDate.min(),
                         'DevelopmentMaxCutoffDate': dev.CutoffDate.max()})
        self.spark.stop()
        self.save('load', {'gold_summary.csv': pd.DataFrame(rows),
                           'held_out_calendar.csv': pd.DataFrame(partitions)})

    def temporal_split(self):
        for hotel, dev in self.data.items():
            self.splits[hotel] = split.run_split(hotel, dev, self.config, self.output_root)['tables']
        sys.path.insert(0, str(common.MODEL_DIR / 'tests'))
        import test_model_split
        import test_fold_cohorts
        import test_baseline_safety
        suite = unittest.TestSuite()
        for module in (test_model_split, test_fold_cohorts, test_baseline_safety):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for hotel, dev in self.data.items():
            suite.addTests(test_model_split.development_suite(dev, self.splits[hotel], self.config['split']))
        log = io.StringIO()
        with redirect_stdout(log):
            outcome = unittest.TextTestRunner(stream=log, verbosity=1).run(suite)
        self.save('checks', {'split_checks.csv': pd.DataFrame([{
            'ChecksPassed': outcome.testsRun - len(outcome.failures) - len(outcome.errors),
            'Failures': len(outcome.failures), 'Errors': len(outcome.errors)}])})
        if not outcome.wasSuccessful():
            raise AssertionError(log.getvalue())
        print(f'{outcome.testsRun} split, cohort and isolation checks passed')
        cv = pd.concat([r['audit']['cv_summary'] for r in self.splits.values()], ignore_index=True)
        parts = pd.concat([r['audit']['partition_summary'] for r in self.splits.values()], ignore_index=True)
        parts['Positives'] = parts['Positives'].astype(str)
        calendar = common.read_tables(self.output_root / 'load')['held_out_calendar.csv']
        parts = pd.concat([parts, calendar], ignore_index=True)
        self.save('split_summary', {'cv_summary.csv': cv, 'partition_summary.csv': parts})
        self.check_history('split')

    def preprocess(self):
        groups = common.feature_config(self.config, 'core21')
        columns = common.feature_columns(groups)
        rows, fallback_rows = [], []
        for hotel, dev in self.data.items():
            for fold in range(1, self.config['split']['n_folds'] + 1):
                _, train, valid = folds.fold_cohorts(dev, self.splits[hotel]['cv_membership'], fold, columns)
                processor = model_pipeline.build_preprocessor(groups, scale_numeric=True)
                a = processor.fit_transform(train[columns])
                state = pickle.dumps(processor)
                b = processor.transform(valid[columns])
                names = processor.get_feature_names_out()
                if not (np.isfinite(a).all() and np.isfinite(b).all()
                        and a.shape[1] == b.shape[1] == len(names) and state == pickle.dumps(processor)):
                    raise AssertionError('Preprocessing shape, finite values or fitted state changed')
                rows.append({'Hotel': hotel, 'Fold': fold, 'Inputs': len(columns),
                             'OutputColumns': len(names), 'TrainRows': len(train),
                             'ValidationRows': len(valid), 'MatchingColumns': True, 'AllFinite': True})
                if hotel == 'H1' and fold == 1:
                    fallback = valid[columns].head(3).copy()
                    fallback.loc[fallback.index[0], groups['numeric'] + groups['binary']] = np.nan
                    for column in groups['categorical']:
                        fallback[column] = fallback[column].astype(object)
                        fallback.loc[fallback.index[0], column] = '__UNSEEN__'
                        fallback.loc[fallback.index[1], column] = np.nan
                    transformed = processor.transform(fallback)
                    start = len(groups['numeric']) + len(groups['binary'])
                    encoder = processor.named_transformers_['categorical'].named_steps['encoder']
                    for column, levels in zip(groups['categorical'], encoder.categories_):
                        stop = start + len(levels)
                        if not np.all(transformed[0, start:stop] == 0):
                            raise AssertionError('Unknown category did not become an all-zero block')
                        expected_missing = np.asarray([value == '__MISSING__' for value in levels], dtype=float)
                        if not np.array_equal(transformed[1, start:stop], expected_missing):
                            raise AssertionError('Missing category did not use the fitted missing-token block')
                        fallback_rows.append({'Check': f'{column}: unseen category zero block', 'Passed': True})
                        fallback_rows.append({'Check': f'{column}: fitted missing-token block', 'Passed': True})
                        start = stop
                    imputer = processor.named_transformers_['numeric'].named_steps['imputer']
                    numeric = processor.named_transformers_['numeric'].named_steps['scaler'].transform(
                        imputer.statistics_.reshape(1, -1))[0]
                    if not np.allclose(transformed[0, :len(numeric)], numeric):
                        raise AssertionError('Numeric fallback did not use fitted train medians')
                    binary = processor.named_transformers_['binary'].named_steps['imputer'].statistics_
                    if not np.array_equal(transformed[0, len(numeric):len(numeric) + len(binary)], binary):
                        raise AssertionError('Binary fallback did not use fitted train modes')
                    if not np.isfinite(transformed).all() or state != pickle.dumps(processor):
                        raise AssertionError('Fallback changed fitted state or produced nonfinite values')
                    fallback_rows.extend([{'Check': name, 'Passed': True} for name in (
                        'Numeric train-median fallback', 'Binary train-mode fallback',
                        'Missing categorical fallback', 'Finite output and unchanged fitted state')])
        self.save('preprocessing', {'audit.csv': pd.DataFrame(rows), 'fallback_checks.csv': pd.DataFrame(fallback_rows)})

    def register(self, hotel, variant, run):
        # One shared cohort contract is applied to every experiment, including later stages.
        if variant != 'Dummy':
            reference = self.runs[(hotel, 'Dummy')]['oof_predictions']
            candidate = run['oof_predictions']
            columns = common.KEYS + ['Fold', 'PredictionDate', 'IsCanceled']
            pd.testing.assert_frame_equal(reference[columns].reset_index(drop=True),
                                          candidate[columns].reset_index(drop=True), check_exact=True)
        self.runs[(hotel, variant)] = run

    def experiment(self, hotel, variant, model_key, feature_name, parameters=None, keep_models=False):
        spec = self.config['models'][model_key]
        groups = common.feature_config(self.config, feature_name)
        parameters = parameters or common.manual_params(self.config, model_key, hotel)
        rows, parts, models = [], [], {}
        for fold in range(1, self.config['split']['n_folds'] + 1):
            result = folds.run_fold(hotel, fold, spec['estimator'], feature_name, groups, parameters,
                                     self.data[hotel], self.splits[hotel]['cv_membership'],
                                     self.config['split']['validation_months'], train_metrics=True)
            row, oof = result['metrics'], result['predictions']
            row['Variant'], oof['Variant'] = variant, variant
            if variant == 'Dummy':
                row['DummyLogLoss'] = row['LogLoss']
            else:
                dummy = self.runs[(hotel, 'Dummy')]['fold_metrics'].set_index('Fold').loc[fold]
                row['DummyLogLoss'] = dummy['LogLoss']
            row['GainVsDummy'] = row['DummyLogLoss'] - row['LogLoss']
            rows.append(row)
            parts.append(oof)
            if keep_models:
                models[fold] = result['pipeline']
        run = {'fold_metrics': pd.DataFrame(rows).drop(columns='Converged', errors='ignore'),
               'oof_predictions': pd.concat(parts, ignore_index=True),
               'models': models, 'params': parameters}
        self.register(hotel, variant, run)
        return run

    def save_runs(self, stage, variants):
        for hotel in self.config['hotels']:
            metrics = pd.concat([self.runs[(hotel, v)]['fold_metrics'] for v in variants], ignore_index=True)
            oof = pd.concat([self.runs[(hotel, v)]['oof_predictions'] for v in variants], ignore_index=True)
            self.save(f'{stage}/{hotel}', {'fold_metrics.csv': metrics, 'oof_predictions.csv': oof,
                                          'cv_summary.csv': summarize(metrics, oof)})

    def check_history(self, stage):
        sys.path.insert(0, str(common.MODEL_DIR / 'tools'))
        import check_legacy_parity
        check_legacy_parity.check_stage(stage, self.output_root, self.run_id)

    def initial(self):
        checks = metric_checks()
        self.save('metric_checks', {'checks.csv': checks})
        for hotel in self.config['hotels']:
            self.experiment(hotel, 'Dummy', 'dummy', 'core21')
            self.experiment(hotel, 'LR21', 'lr', 'core21', keep_models=True)
        self.save_runs('baseline', ['Dummy', 'LR21'])
        self.check_history('baseline')

    def diagnose_initial(self):
        from baseline_diagnostics import initial_diagnosis
        self.save('diagnosis', initial_diagnosis(self))

    def calendar(self):
        for hotel in self.config['hotels']:
            self.experiment(hotel, 'LR20', 'lr', 'no_year20')
            self.experiment(hotel, 'LR18', 'lr', 'no_time18')
        self.save_runs('calendar', ['LR20', 'LR18'])
        self.save_runs('calendar_comparison', ['Dummy', 'LR21', 'LR20', 'LR18'])
        self.check_history('calendar')

    def untuned(self):
        for hotel in self.config['hotels']:
            self.experiment(hotel, 'RF18', 'rf', 'no_time18')
            self.experiment(hotel, 'XGB18', 'xgb', 'no_time18')
        self.save_runs('untuned_rf', ['RF18'])
        self.save_runs('untuned_xgb', ['XGB18'])
        self.save_runs('comparison', ['Dummy', 'LR18', 'RF18', 'XGB18'])
        self.save('alignment', {'cohort_alignment.csv': pd.DataFrame([
            {'Hotel': hotel, 'ValidationBookings': len(self.runs[(hotel, 'Dummy')]['oof_predictions']),
             'VariantsAligned': len([key for key in self.runs if key[0] == hotel]),
             'KeysLabelsFoldsDatesMatch': True} for hotel in self.config['hotels']])})
        self.check_history('untuned')

    def diagnostics(self):
        from baseline_diagnostics import model_diagnostics
        self.save('diagnostics', model_diagnostics(self))

    def tuning(self):
        rows, windows = [], []
        for hotel in self.config['hotels']:
            result = tune.run_tuning(hotel, self.config, self.data[hotel],
                                     self.splits[hotel]['cv_membership'], self.output_root)
            self.tuned[hotel] = result
            for key, winner in result['winners'].items():
                label = {'lr': 'LR18', 'rf': 'RF18', 'xgb': 'XGB18'}[key]
                variant = label + ' tuned'
                winner['fold_metrics']['Variant'] = variant
                winner['oof_predictions']['Variant'] = variant
                self.register(hotel, variant, winner)
                summary = summarize(winner['fold_metrics'], winner['oof_predictions']).iloc[0].to_dict()
                summary.update({'ModelKey': key, 'ConfigID': winner['config_id'],
                                'Parameters': json.dumps(result['tuned'][key]['grid_params'], sort_keys=True),
                                'InitialLogLoss': float(self.runs[(hotel, label)]['fold_metrics'].LogLoss.mean()),
                                'InitialGap': float(self.runs[(hotel, label)]['fold_metrics'].GeneralizationGap.mean()),
                                'DummyLogLoss': float(self.runs[(hotel, 'Dummy')]['fold_metrics'].LogLoss.mean())})
                rows.append(summary)
                windows.append(winner['fold_metrics'])
        self.save('tuning_summary', {'winners.csv': pd.DataFrame(rows),
                                     'fold_metrics.csv': pd.concat(windows, ignore_index=True)})
        self.check_history('tuning')

    def features(self):
        for hotel in self.config['hotels']:
            reference = copy.deepcopy(self.tuned[hotel]['winners']['rf'])
            reference['fold_metrics']['Variant'] = 'F0'
            reference['oof_predictions']['Variant'] = 'F0'
            self.register(hotel, 'F0', reference)
            for variant, feature in [('F1', 'f1_17'), ('F2', 'f2_17'), ('F3', 'f3_16')]:
                self.experiment(hotel, variant, 'rf', feature, reference['params'])
            for variant in ('F0', 'F1', 'F2', 'F3'):
                run = self.runs[(hotel, variant)]
                metrics = run['fold_metrics']
                metrics['ConfigID'] = reference['config_id']
                metrics['ModelKey'] = 'rf'
                metrics['GridParameters'] = json.dumps(self.tuned[hotel]['tuned']['rf']['grid_params'], sort_keys=True)
                metrics['ReferenceConfigID'] = reference['config_id']
                metrics['F0LogLoss'] = reference['fold_metrics'].set_index('Fold').loc[metrics.Fold, 'LogLoss'].to_numpy()
                metrics['GainVsF0'] = metrics.F0LogLoss - metrics.LogLoss
                run['oof_predictions']['ConfigID'] = reference['config_id']
                run['oof_predictions']['DummyProbability'] = self.runs[(hotel, 'Dummy')]['oof_predictions'].RawProbability.to_numpy()
        self.save_runs('features', ['F0', 'F1', 'F2', 'F3'])
        summaries = []
        for hotel in self.config['hotels']:
            table = common.read_tables(self.output_root / 'features' / hotel)['cv_summary.csv']
            for i, row in table.iterrows():
                metrics = self.runs[(hotel, row.Variant)]['fold_metrics']
                table.loc[i, 'GainVsF0'] = metrics.GainVsF0.mean()
                table.loc[i, 'FoldsBeatingF0'] = metrics.GainVsF0.gt(common.TOLERANCE).sum()
                table.loc[i, 'InputColumns'] = metrics.InputColumns.iloc[0]
            summaries.append(table)
        self.save('feature_summary', {'cv_summary.csv': pd.concat(summaries, ignore_index=True)})
        self.check_history('features')

    def calibrate(self):
        rows, parts = [], []
        for hotel in self.config['hotels']:
            run = self.runs[(hotel, 'F1')]
            dates = common.label_dates(self.splits[hotel]['fit_eligibility'])
            for method in self.config['calibration_methods']:
                metrics, oof, states = calibration.evaluate_folds(
                    hotel, method, run['oof_predictions'], dates, run['fold_metrics'],
                    self.config['split']['test_start'])
                metrics['FeatureSet'], oof['FeatureSet'] = 'F1', 'F1'
                rows.append(metrics)
                parts.append(oof)
                common.write_json(self.output_root / 'calibration' / hotel / f'{method}.json', states)
                if method == 'raw':
                    run['calibration_metrics'] = metrics
                    run['calibrated_predictions'] = oof
            self.save(f'cv/{hotel}', {f'{name}.csv': run[name] for name in (
                'fold_metrics', 'oof_predictions', 'calibration_metrics', 'calibrated_predictions')})
        metrics = pd.concat(rows, ignore_index=True)
        oof = pd.concat(parts, ignore_index=True)
        summary = metrics.groupby(['Hotel', 'Method'], sort=False)[METRICS + ['GainVsRaw']].mean().reset_index()
        counts = metrics.groupby(['Hotel', 'Method']).agg(
            FoldsBeatingRaw=('GainVsRaw', lambda s: int(s.gt(common.TOLERANCE).sum())),
            MeanAbsoluteWindowBias=('ProbabilityBias', lambda s: s.abs().mean()),
            ExactZeroRows=('ExactZeroRows', 'sum')).reset_index()
        summary = summary.merge(counts, on=['Hotel', 'Method'], validate='one_to_one')
        zeros = oof.assign(CanceledAtZero=oof.Probability.eq(0) & oof.IsCanceled.eq(1)).groupby(
            ['Hotel', 'Method']).CanceledAtZero.sum().reset_index()
        summary = summary.merge(zeros, on=['Hotel', 'Method'], validate='one_to_one')
        bins = reliability(oof, 'Method', 'Probability', self.config['diagnostics']['score_groups'])
        self.save('calibration', {'fold_metrics.csv': metrics, 'oof_predictions.csv': oof,
                                  'cv_summary.csv': summary, 'reliability.csv': bins})
        self.check_history('calibration')

    def freeze(self):
        checks, groups, fold_tables, oof_tables = [], [], [], []
        for hotel in self.config['hotels']:
            run = self.runs[(hotel, 'F1')]
            analysis = error_analysis.run_error_analysis(hotel, self.config, self.data[hotel],
                self.splits[hotel]['cv_membership'], run['oof_predictions'], self.output_root)
            groups.append(analysis['group_errors'])
            rehearsal = final.run_final(hotel, self.config, self.data[hotel], self.splits[hotel],
                run, self.tuned[hotel]['tuned'], self.output_root, rehearsal=True)
            checks.extend({'Hotel': hotel, 'Fold': 4, 'Metric': name, 'MaxAbsDifference': difference}
                          for name, difference in rehearsal['check'].items())
            fold_tables.append(run['fold_metrics'])
            oof_tables.append(run['oof_predictions'])
        self.save_runs('frozen_comparison', ['Dummy', 'F1'])
        group_table = pd.concat(groups, ignore_index=True)
        large = group_table.loc[group_table.Rows.ge(self.config['diagnostics']['min_group_rows'])]
        largest = large.sort_values('CVExcessContribution', ascending=False).groupby('Hotel').head(6)
        self.save('frozen', {'fold_metrics.csv': pd.concat(fold_tables, ignore_index=True),
                             'oof_predictions.csv': pd.concat(oof_tables, ignore_index=True),
                             'hard_groups.csv': largest.reset_index(drop=True),
                             'rehearsal_check.csv': pd.DataFrame(checks)})
        common.write_json(self.output_root / 'frozen' / 'config.json', {
            'config': self.config, 'hotel_parameters': {hotel: self.tuned[hotel]['tuned']['rf']
                                                       for hotel in self.config['hotels']},
            'scope': 'RF17 raw development baseline; no held-out evaluation'})
        self.check_history('frozen')


def metric_checks():
    examples = [('Uniform balanced', [0, 1], [0.5, 0.5], [np.log(2), .25, .5, .5]),
                ('Correct ranking', [0, 1], [.1, .9], [-np.log(.9), .01, 1, 1]),
                ('Constant prior', [0, 0, 0, 1], [.25] * 4,
                 [-.75 * np.log(.75) - .25 * np.log(.25), .1875, .5, .25])]
    checks = []
    for name, labels, probabilities, expected in examples:
        observed = model_evaluation.evaluate_predictions(labels, probabilities)
        for metric, answer in zip(METRICS, expected):
            checks.append({'Check': f'{name}: {metric}', 'Passed': bool(np.isclose(observed[metric], answer))})
    for name, labels, probabilities in [
        ('Length mismatch', [0, 1], [.5]), ('Matrix', [0, 1], [[.5, .5], [.5, .5]]),
        ('Out of range', [0, 1], [.2, 1.2]), ('Nonfinite', [0, 1], [.2, np.nan]),
        ('Invalid labels', [0, 2], [.2, .8])]:
        try:
            model_evaluation.evaluate_predictions(labels, probabilities)
            passed = False
        except ValueError:
            passed = True
        checks.append({'Check': name, 'Passed': passed})
    table = pd.DataFrame(checks)
    if not table.Passed.all():
        raise AssertionError('Known-answer metric checks failed')
    return table
