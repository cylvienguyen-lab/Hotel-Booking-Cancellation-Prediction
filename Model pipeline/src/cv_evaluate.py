"""Stage cv: temporal CV of the configured candidate for one hotel, then temporal calibration."""
import argparse
import json

import pandas as pd

import calibration
import common
import folds
from utils import model_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hotel', required=True)
    hotel = parser.parse_args().hotel
    params = common.load_params()
    split_config = params['split']
    metric_columns = params['evaluation']['metrics']
    groups = common.feature_config(params, params['features'])
    common.check_features(groups, common.load_contract())
    tuned = json.loads((common.TUNED_PARAMS / f'{hotel}.json').read_text(encoding='utf-8'))
    model = common.model_settings(params, hotel, tuned)
    dev = common.read_parquet(common.ARTIFACTS / 'data' / hotel / 'dev.parquet')
    split_tables = common.read_tables(common.ARTIFACTS / 'split' / hotel)
    membership = split_tables['cv_membership.csv']

    # Fit the candidate and the expanding-history Dummy reference on every window
    rows, parts = [], []
    for fold in range(1, split_config['n_folds'] + 1):
        dummy = folds.run_fold(hotel, fold, 'dummy', 'prior', groups,
                               params['models']['dummy']['params'], dev, membership,
                               split_config['validation_months'])
        result = folds.run_fold(hotel, fold, model['estimator'], params['features'], groups,
                                model['params'], dev, membership, split_config['validation_months'],
                                lookback_months=params['train_lookback_months'], train_metrics=True)
        row, prediction = result['metrics'], result['predictions']
        row.update({'ModelKey': model['key'], 'ConfigID': model['config_id'],
                    'DummyLogLoss': dummy['metrics']['LogLoss'],
                    'GainVsDummy': dummy['metrics']['LogLoss'] - row['LogLoss']})
        if not prediction['Booking_ID'].reset_index(drop=True).equals(
                dummy['predictions']['Booking_ID'].reset_index(drop=True)):
            raise AssertionError(f'{hotel} fold {fold}: Dummy and model OOF rows differ')
        prediction['ConfigID'] = model['config_id']
        prediction['DummyProbability'] = dummy['predictions']['RawProbability'].to_numpy()
        rows.append(row)
        parts.append(prediction)
    fold_metrics = pd.DataFrame(rows)
    oof = pd.concat(parts, ignore_index=True)
    if oof.duplicated(common.KEYS).any():
        raise AssertionError('Duplicate OOF bookings')

    # Calibrate on earlier folds and score folds 2..n
    dates = common.label_dates(split_tables['fit_eligibility.csv'])
    calibration_metrics, calibrated, states = calibration.evaluate_folds(
        hotel, params['calibration'], oof, dates, fold_metrics, split_config['test_start'])

    # Save tables and the equal-weight fold means used for comparison
    directory = common.ARTIFACTS / 'cv' / hotel
    common.write_tables(directory, {
        'fold_metrics.csv': fold_metrics, 'oof_predictions.csv': oof,
        'calibration_metrics.csv': calibration_metrics, 'calibrated_predictions.csv': calibrated,
    }, extra={'model': model, 'features': params['features'], 'calibration': params['calibration'],
              'train_lookback_months': params['train_lookback_months']})
    common.write_json(directory / 'calibrators.json', states)
    pooled = model_evaluation.evaluate_predictions(oof['IsCanceled'], oof['RawProbability'])
    summary = {
        'cv': {**{name: float(fold_metrics[name].mean()) for name in metric_columns},
               'DummyLogLoss': float(fold_metrics['DummyLogLoss'].mean()),
               'GainVsDummy': float(fold_metrics['GainVsDummy'].mean()),
               'FoldsBeatingDummy': int(fold_metrics['GainVsDummy'].gt(common.TOLERANCE).sum()),
               'PooledOOFLogLoss': float(pooled['LogLoss']),
               'MeanTrainLogLoss': float(fold_metrics['TrainLogLoss'].mean()),
               'MeanGap': float(fold_metrics['GeneralizationGap'].mean())},
        'calibrated': {**{name: float(calibration_metrics[name].mean()) for name in metric_columns},
                       'RawLogLoss': float(calibration_metrics['RawLogLoss'].mean()),
                       'GainVsRaw': float(calibration_metrics['GainVsRaw'].mean()),
                       'FoldsBeatingRaw': int(calibration_metrics['GainVsRaw'].gt(common.TOLERANCE).sum()),
                       'DummyLogLoss': float(calibration_metrics['DummyLogLoss'].mean())},
    }
    common.write_json(common.METRICS / f'cv_{hotel}.json', summary)
    print(f"{hotel} {model['key']}/{params['features']} ({model['config_id']}): "
          f"cv LogLoss {summary['cv']['LogLoss']:.5f} | "
          f"{params['calibration']} folds 2-{split_config['n_folds']} {summary['calibrated']['LogLoss']:.5f}")


if __name__ == '__main__':
    main()
