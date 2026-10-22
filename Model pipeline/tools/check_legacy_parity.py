"""Compare Notebook 01 development outputs with frozen CV references; stop above 1e-4.

No Gold data, held-out partitions, models or held-out predictions are opened here.
    python tools/check_legacy_parity.py --output-root outputs/baseline01
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import common

LEGACY = common.MODEL_DIR / 'reports'
METRICS = ['LogLoss', 'BrierScore', 'ROC_AUC', 'AveragePrecision']
TOLERANCE = 1e-4
STAGES = ['split', 'baseline', 'calendar', 'untuned', 'tuning', 'features', 'calibration', 'frozen']


def legacy_table(path, **filters):
    columns = pd.read_csv(path, nrows=0).columns
    dates = [c for c in columns if c in ('PredictionDate', 'FitDate', 'LabelAvailableDate')]
    table = pd.read_csv(path, dtype={'Booking_ID': str, 'Group': str},
                        float_precision='round_trip', keep_default_na=False, parse_dates=dates)
    for column, value in filters.items():
        table = table.loc[table[column].eq(value)]
    return table.reset_index(drop=True)


def compare(name, current, historical, keys, exact, metrics):
    joined = current.merge(historical, on=keys, how='outer', validate='one_to_one',
                            suffixes=('_current', '_historical'), indicator=True)
    keys_match = bool(joined['_merge'].eq('both').all() and len(current) == len(historical))
    exact_match = bool(all(joined[f'{c}_current'].eq(joined[f'{c}_historical']).all() for c in exact))
    rows = []
    for metric in metrics or ['Membership']:
        if metric == 'Membership':
            a = b = pd.Series([0.0])
        else:
            a = pd.to_numeric(joined[f'{metric}_current'], errors='coerce')
            b = pd.to_numeric(joined[f'{metric}_historical'], errors='coerce')
        delta = np.abs(a.to_numpy() - b.to_numpy())
        maximum = float(delta.max()) if len(delta) and np.isfinite(delta).all() else float('inf')
        rows.append({'Check': name, 'Metric': metric, 'Rows': len(historical),
                     'RerunMean': float(a.mean()), 'HistoricalMean': float(b.mean()),
                     'MaxAbsDifference': maximum, 'KeysMatch': keys_match, 'ExactColumns': exact_match,
                     'Pass': keys_match and exact_match and maximum <= TOLERANCE})
    return rows


def check_stage(stage, output_root, run_id=None):
    root = Path(output_root)
    config = json.loads((root / 'run_config.json').read_text(encoding='utf-8'))
    run_id = run_id or json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))['run_id']
    records = []

    def compare_tables(name, current, old, keys, exact=(), metrics=METRICS):
        records.extend(compare(name, current, old, list(keys), list(exact), list(metrics)))

    def metric_pair(name, current, old, keys=('Fold',)):
        exact = [c for c in ('TrainRows', 'TrainPositives', 'ValidationRows', 'ValidationPositives',
                              'InputColumns', 'ProcessedColumns', 'FitDate', 'ExactZeroRows',
                              'ExactOneRows') if c in current and c in old]
        values = [c for c in METRICS + [f'Train{m}' for m in METRICS] + [
            'DummyLogLoss', 'GainVsDummy', 'GainVsF0', 'MeanProbability', 'ProbabilityBias',
            'GeneralizationGap', 'RawLogLoss', 'GainVsRaw'] if c in current and c in old]
        compare_tables(name, current, old, keys, exact, values)

    for hotel in config['hotels']:
        if stage == 'split':
            # Do not open historical partition membership or partition outcome summaries.
            for name, keys, exact in [
                ('cv_membership', common.KEYS + ['Fold', 'Role'], ['FitDate']),
                ('fit_eligibility', common.KEYS + ['Fold'], ['FitDate', 'LabelAvailableDate', 'Eligible']),
                ('refit_membership', common.KEYS, ['FitDate']),
                ('calibration_membership', common.KEYS + ['Fold'], ['PredictionDate'])]:
                current = common.read_tables(root / 'split' / hotel)[f'{name}.csv']
                old = legacy_table(common.MODEL_DIR / 'splits' / 'split_v1' / hotel / f'{name}.csv')
                if name == 'fit_eligibility':
                    current, old = current.loc[current.Stage.eq('CV')], old.loc[old.Stage.eq('CV')]
                    old = old.copy()
                    old['Fold'] = old.Fold.astype(int)
                    if old.Eligible.dtype == object:
                        old['Eligible'] = old.Eligible.astype(str).eq('True')
                compare_tables(f'{hotel} {name}', current, old, keys, exact, [])
        elif stage in ('baseline', 'calendar', 'untuned', 'features'):
            mapping = {
                'baseline': [('baseline', 'baseline_v1', 'Dummy', {'Model': 'dummy'}),
                             ('baseline', 'baseline_v1', 'LR21', {'Model': 'logistic_regression'})],
                'calendar': [('calendar', 'lr_calendar_ablation_v1', 'LR20', {'FeatureSet': 'core_no_year'}),
                             ('calendar', 'lr_calendar_ablation_v1', 'LR18', {'FeatureSet': 'core_no_year_month_week'})],
                'untuned': [('untuned_rf', 'rf_initial_v1', 'RF18', {}),
                            ('untuned_xgb', 'xgb_initial_v1', 'XGB18', {})],
                'features': [('features', 'feature_ablation_round1_v1', label, {'FeatureSet': label})
                             for label in ('F0', 'F1', 'F2', 'F3')],
            }
            for directory, reference, variant, filters in mapping[stage]:
                tables = common.read_tables(root / directory / hotel)
                current = tables['fold_metrics.csv'].loc[lambda t: t.Variant.eq(variant)]
                old = legacy_table(LEGACY / reference / hotel / 'fold_metrics.csv', **filters)
                metric_pair(f'{hotel} {variant}', current, old)
                current_oof = tables['oof_predictions.csv'].loc[lambda t: t.Variant.eq(variant)]
                old_oof = legacy_table(LEGACY / reference / hotel / 'oof_predictions.csv', **filters)
                compare_tables(f'{hotel} {variant} OOF', current_oof, old_oof, common.KEYS,
                               ['Fold', 'PredictionDate', 'IsCanceled'], ['RawProbability'])
        elif stage == 'tuning':
            current = common.read_tables(root / 'tuning' / hotel)['fold_metrics.csv']
            old = legacy_table(LEGACY / 'tuning_round1_v1' / hotel / 'fold_metrics.csv')
            metric_pair(f'{hotel} tuning: all configurations', current, old, ('ConfigID', 'Fold'))
            winners = json.loads((root / 'tuning' / hotel / 'tuned.json').read_text(encoding='utf-8'))
            old_summary = legacy_table(LEGACY / 'tuning_round1_v1' / hotel / 'cv_summary.csv')
            old_best = old_summary.loc[old_summary.BestWithinModel.astype(str).eq('True')]
            expected = set(old_best.ConfigID)
            observed = {v['config_id'] for v in winners.values()}
            records.append({'Check': f'{hotel} tuning selected configs', 'Metric': 'Selection', 'Rows': len(expected),
                            'RerunMean': 0.0, 'HistoricalMean': 0.0, 'MaxAbsDifference': 0.0,
                            'KeysMatch': expected == observed, 'ExactColumns': expected == observed,
                            'Pass': expected == observed})
        elif stage == 'calibration':
            tables = common.read_tables(root / 'calibration')
            for method in config['calibration_methods']:
                current = tables['fold_metrics.csv'].loc[lambda t: t.Hotel.eq(hotel) & t.Method.eq(method)]
                old = legacy_table(LEGACY / 'calibration_round1_v1' / 'fold_metrics.csv',
                                   Hotel=hotel, FeatureSet='F1', Method=method)
                metric_pair(f'{hotel} RF17 {method}', current, old)
                current_oof = tables['oof_predictions.csv'].loc[lambda t: t.Hotel.eq(hotel) & t.Method.eq(method)]
                old_oof = legacy_table(LEGACY / 'calibration_round1_v1' / 'oof_predictions.csv',
                                       Hotel=hotel, FeatureSet='F1', Method=method)
                compare_tables(f'{hotel} RF17 {method} OOF', current_oof, old_oof, common.KEYS,
                               ['Fold', 'PredictionDate', 'IsCanceled'], ['RawProbability', 'Probability'])
        elif stage == 'frozen':
            tables = common.read_tables(root / 'error_analysis' / hotel)
            for filename, keys, exact, metrics in [
                ('booking_errors.csv', common.KEYS, ['Fold', 'IsCanceled', 'LeadTimeBand', 'CustomerType', 'MarketSegment'],
                 ['RawProbability', 'RawProbability_dummy', 'IndividualLogLoss', 'DummyIndividualLogLoss', 'ExcessLogLoss']),
                ('group_errors.csv', ['Hotel', 'Fold', 'Dimension', 'Group'], ['Rows', 'Positives', 'Rows_window'],
                 ['MeanProbability', 'LogLoss', 'DummyLogLoss', 'TotalLoss', 'TotalExcessLoss',
                  'CancelRate', 'ProbabilityBias', 'ExcessLogLoss', 'CVExcessContribution'])]:
                old = legacy_table(LEGACY / 'rf17_error_analysis_v1' / filename, Hotel=hotel)
                compare_tables(f'{hotel} {filename}', tables[filename], old, keys, exact, metrics)
        else:
            raise ValueError(f'Unknown parity stage: {stage}')

    report = pd.DataFrame(records)
    report.insert(0, 'Stage', stage)
    report['RunID'] = run_id
    directory = root / 'parity'
    previous_path = directory / 'comparison.csv'
    previous = pd.read_csv(previous_path, float_precision='round_trip') if previous_path.exists() else pd.DataFrame()
    if not previous.empty:
        previous = previous.loc[previous.RunID.eq(run_id) & previous.Stage.ne(stage)]
        report = pd.concat([previous, report], ignore_index=True)
    summary = report.groupby(['Stage', 'Check'], sort=False).agg(
        MetricsCompared=('Metric', 'size'), MaxAbsDifference=('MaxAbsDifference', 'max'),
        Pass=('Pass', 'all')).reset_index()
    common.write_tables(directory, {'comparison.csv': report, 'summary.csv': summary})
    failures = report.loc[~report.Pass]
    if not failures.empty:
        raise AssertionError('Historical parity exceeded 1e-4 or cohort mismatch. Stop without changing values.\n'
                             + failures.to_string(index=False) + '\nLikely causes: input/cohort, feature order, '
                             'package versions or BLAS thread count; inspect before continuing.')
    print(f'{stage}: historical parity passed; max difference {report.MaxAbsDifference.max():.2e}')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=common.OUTPUTS / 'baseline01')
    parser.add_argument('--stage', choices=STAGES)
    args = parser.parse_args()
    for stage in [args.stage] if args.stage else STAGES:
        report = check_stage(stage, args.output_root)
    print(report.groupby('Stage').agg(MaxAbsDifference=('MaxAbsDifference', 'max'), Pass=('Pass', 'all')).to_string())


if __name__ == '__main__':
    main()
