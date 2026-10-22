"""Compare notebook outputs with frozen reports/ and splits/ at 1e-12.

Keys, counts, labels and dates must match exactly.

    python "Model pipeline/tools/check_legacy_parity.py" [--rolling]
"""
import argparse
import filecmp
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import common  # noqa: E402
import folds  # noqa: E402

LEGACY = common.MODEL_DIR / 'reports'
METRICS = ['LogLoss', 'BrierScore', 'ROC_AUC', 'AveragePrecision']
results = []


def legacy_table(path, **filters):
    table = pd.read_csv(path, float_precision='round_trip', keep_default_na=False,
                        dtype={'Booking_ID': str}, parse_dates=[c for c in ('PredictionDate', 'FitDate')
                                                                if c in pd.read_csv(path, nrows=0)])
    for column, value in filters.items():
        table = table.loc[table[column].eq(value)]
    return table.reset_index(drop=True)


def compare(name, new, old, keys, exact=(), floats=(), renames=None,
            tolerance=common.TOLERANCE):
    # Align on keys, then require exact equality or a bounded float difference per column
    new = new.rename(columns=renames or {})
    joined = new.merge(old, on=keys, how='outer', suffixes=('_new', '_old'),
                       indicator=True, validate='one_to_one')
    row = {'Check': name, 'Rows': len(old), 'KeysMatch': bool(joined['_merge'].eq('both').all()
                                                             and len(new) == len(old))}
    row['ExactColumns'] = all(joined[f'{c}_new'].eq(joined[f'{c}_old']).all() for c in exact)
    differences = {c: float(np.abs(joined[f'{c}_new'].astype(float) - joined[f'{c}_old'].astype(float)).max())
                   for c in floats}
    row['MaxFloatDiff'] = max(differences.values()) if differences else 0.0
    row['BitIdenticalFloats'] = all(joined[f'{c}_new'].equals(joined[f'{c}_old']) for c in floats)
    row['Pass'] = row['KeysMatch'] and row['ExactColumns'] and row['MaxFloatDiff'] <= tolerance
    results.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rolling', action='store_true', help='replay rolling_12m_v1 in memory')
    arguments = parser.parse_args()
    art = common.OUTPUTS
    params = json.loads((art / 'run_config.json').read_text(encoding='utf-8'))
    label = {'lr': 'LR18', 'rf': 'RF18', 'xgb': 'XGB18'}

    for hotel in params['hotels']:
        # Split memberships must be byte-identical
        for name in ('partition_membership', 'cv_membership', 'fit_eligibility', 'refit_membership',
                     'calibration_membership'):
            same = filecmp.cmp(art / 'split' / hotel / f'{name}.csv',
                               common.MODEL_DIR / 'splits' / 'split_v1' / hotel / f'{name}.csv', shallow=False)
            results.append({'Check': f'{hotel} split {name}.csv', 'Rows': None, 'KeysMatch': same,
                            'ExactColumns': same, 'MaxFloatDiff': 0.0, 'BitIdenticalFloats': same, 'Pass': same})

        # Tuning: all 25 configurations x folds, and the selected winners
        tuning = common.read_tables(art / 'tuning' / hotel)['fold_metrics.csv']
        old_tuning = legacy_table(LEGACY / 'tuning_round1_v1' / hotel / 'fold_metrics.csv')
        compare(f'{hotel} tuning validation/loss metrics (25 configs)', tuning,
                old_tuning, ['ConfigID', 'Fold'],
                exact=['TrainRows', 'TrainPositives', 'ValidationRows', 'ValidationPositives', 'ProcessedColumns'],
                floats=METRICS + ['TrainLogLoss', 'TrainBrierScore', 'DummyLogLoss', 'GainVsDummy'],
                tolerance=common.TOLERANCE)
        compare(f'{hotel} tuning train ranking metrics (25 configs)', tuning,
                old_tuning, ['ConfigID', 'Fold'],
                floats=['TrainROC_AUC', 'TrainAveragePrecision'], tolerance=common.TOLERANCE)
        tuned = json.loads((art / 'tuning' / hotel / 'tuned.json').read_text(encoding='utf-8'))
        legacy_best = legacy_table(LEGACY / 'tuning_round1_v1' / hotel / 'cv_summary.csv')
        legacy_best = legacy_best.loc[legacy_best['BestWithinModel'].astype(str).eq('True')]
        legacy_best = dict(zip(legacy_best['ModelLabel'], legacy_best['ConfigID']))
        same = all(legacy_best[label[key]] == value['config_id'] for key, value in tuned.items())
        results.append({'Check': f'{hotel} tuning winners {sorted(v["config_id"] for v in tuned.values())}',
                        'Rows': None, 'KeysMatch': same, 'ExactColumns': same, 'MaxFloatDiff': 0.0,
                        'BitIdenticalFloats': same, 'Pass': same})

        # cv default (RF F1, tuned, raw) against feature_ablation_round1_v1 F1 and calibration raw
        cv = common.read_tables(art / 'cv' / hotel)
        old_oof = legacy_table(LEGACY / 'feature_ablation_round1_v1' / hotel / 'oof_predictions.csv', FeatureSet='F1')
        dummy = legacy_table(LEGACY / 'baseline_v1' / hotel / 'oof_predictions.csv', Model='dummy')
        old_oof = old_oof.merge(dummy[['Hotel', 'Booking_ID', 'RawProbability']].rename(
            columns={'RawProbability': 'DummyProbability'}), on=['Hotel', 'Booking_ID'], validate='one_to_one')
        compare(f'{hotel} cv OOF vs feature_ablation F1 (+ baseline Dummy)',
                cv['oof_predictions.csv'].drop(columns='FeatureSet'), old_oof, ['Hotel', 'Booking_ID'],
                exact=['Fold', 'PredictionDate', 'IsCanceled', 'Model', 'ReferenceConfigID'],
                floats=['RawProbability', 'DummyProbability'], renames={'ConfigID': 'ReferenceConfigID'})
        compare(f'{hotel} cv fold metrics vs feature_ablation F1', cv['fold_metrics.csv'].drop(columns='FeatureSet'),
                legacy_table(LEGACY / 'feature_ablation_round1_v1' / hotel / 'fold_metrics.csv', FeatureSet='F1'),
                ['Fold'], exact=['TrainRows', 'TrainPositives', 'ValidationRows', 'ValidationPositives',
                                 'InputColumns', 'ProcessedColumns', 'FitDate'],
                floats=METRICS + [f'Train{m}' for m in METRICS] + ['DummyLogLoss', 'GainVsDummy'])
        old_calibration = legacy_table(LEGACY / 'calibration_round1_v1' / 'fold_metrics.csv',
                                       Hotel=hotel, FeatureSet='F1', Method=params['calibration'])
        compare(f'{hotel} {params["calibration"]} folds 2-4 vs calibration_round1_v1',
                cv['calibration_metrics.csv'], old_calibration, ['Hotel', 'Fold', 'Method'],
                exact=['ValidationRows', 'ValidationPositives', 'ExactZeroRows', 'ExactOneRows'],
                floats=METRICS + ['RawLogLoss', 'DummyLogLoss', 'GainVsRaw', 'MeanProbability'])
        old_probabilities = legacy_table(LEGACY / 'calibration_round1_v1' / 'oof_predictions.csv',
                                         Hotel=hotel, FeatureSet='F1', Method=params['calibration'])
        compare(f'{hotel} {params["calibration"]} calibrated probabilities', cv['calibrated_predictions.csv'],
                old_probabilities, ['Hotel', 'Booking_ID'], exact=['Fold', 'IsCanceled', 'PredictionDate'],
                floats=['RawProbability', 'Probability'])

        # Error analysis against rf17_error_analysis_v1
        errors = common.read_tables(art / 'error_analysis' / hotel)
        old = legacy_table(LEGACY / 'rf17_error_analysis_v1' / 'booking_errors.csv', Hotel=hotel)
        compare(f'{hotel} error analysis bookings', errors['booking_errors.csv'].drop(columns='FeatureSet'), old,
                ['Hotel', 'Booking_ID'], exact=['Fold', 'IsCanceled', 'LeadTime', 'LeadTimeBand', 'CustomerType',
                                                'MarketSegment', 'HighLossWithinClass', 'ReferenceConfigID'],
                floats=['RawProbability', 'RawProbability_dummy', 'IndividualLogLoss',
                        'DummyIndividualLogLoss', 'ExcessLogLoss'])
        compare(f'{hotel} error analysis groups', errors['group_errors.csv'],
                legacy_table(LEGACY / 'rf17_error_analysis_v1' / 'group_errors.csv', Hotel=hotel),
                ['Hotel', 'Fold', 'Dimension', 'Group'], exact=['Rows', 'Positives', 'Rows_window'],
                floats=['MeanProbability', 'LogLoss', 'DummyLogLoss', 'TotalLoss', 'TotalExcessLoss',
                        'CancelRate', 'ProbabilityBias', 'ExcessLogLoss', 'CVExcessContribution'])

        # Rolling 12-month history replayed with the shared fold runner
        if arguments.rolling:
            from pyspark.sql import SparkSession
            from utils import model_split
            spark = SparkSession.builder.appName('parity_rolling').master('local[2]').getOrCreate()
            gold = common.load_gold(hotel, spark)
            partition = model_split.make_splits(gold, params['split'])['partition_membership']
            labels = gold['Booking_ID'].map(partition.set_index('Booking_ID')['Partition'])
            dev = gold.loc[labels.eq('Development')].reset_index(drop=True)
            spark.stop()
            membership = common.read_tables(art / 'split' / hotel)['cv_membership.csv']
            groups = common.feature_config(params, 'no_time18')
            for key in ('rf', 'xgb'):
                estimator = params['models'][key]['estimator']
                runs = [folds.run_fold(hotel, fold, estimator, 'core_no_year_month_week', groups,
                                       common.manual_params(params, key, hotel), dev, membership,
                                       params['split']['validation_months'], lookback_months=12)
                        for fold in range(1, params['split']['n_folds'] + 1)]
                compare(f'{hotel} rolling_12m {key} OOF', pd.concat([r['predictions'] for r in runs]),
                        legacy_table(LEGACY / 'rolling_12m_v1' / hotel / 'oof_predictions.csv', Model=estimator),
                        ['Hotel', 'Booking_ID'], exact=['Fold', 'PredictionDate', 'IsCanceled', 'FeatureSet'],
                        floats=['RawProbability'])
                compare(f'{hotel} rolling_12m {key} fold metrics', pd.DataFrame([r['metrics'] for r in runs]),
                        legacy_table(LEGACY / 'rolling_12m_v1' / hotel / 'fold_metrics.csv', Model=estimator),
                        ['Model', 'Fold'], exact=['TrainRows', 'TrainPositives', 'HistoryRemovedRows',
                                                  'HistoryRemovedPositives', 'ValidationRows', 'MaxTreeDepth'],
                        floats=METRICS)

    report = pd.DataFrame(results)
    with pd.option_context('display.width', 200, 'display.max_colwidth', 70):
        print(report.to_string(index=False, formatters={'MaxFloatDiff': '{:.1e}'.format}))
    print('ALL PASS' if report['Pass'].all() else 'FAILURES PRESENT')
    sys.exit(0 if report['Pass'].all() else 1)


if __name__ == '__main__':
    main()
