"""Temporal grid search per model for one hotel."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

import common
import folds
from utils import model_evaluation


def run_tuning(hotel, params, dev, membership, output_root):
    split_config, tuning = params['split'], params['tuning']
    primary = params['evaluation']['primary_metric']
    metric_columns = params['evaluation']['metrics']
    groups = common.feature_config(params, tuning['features'])
    common.check_features(groups, common.load_contract())
    n_inputs = len(common.feature_columns(groups))
    fold_ids = range(1, split_config['n_folds'] + 1)

    # Score the training-prior Dummy once per window as the shared reference
    dummy = {fold: folds.run_fold(hotel, fold, 'dummy', 'prior', groups,
                                  params['models']['dummy']['params'], dev, membership,
                                  split_config['validation_months'])['metrics'][primary]
             for fold in fold_ids}

    # Evaluate every grid configuration on the same eligible train and validation windows
    rows, predictions = [], {}
    for model_key, grid in tuning['grids'].items():
        estimator = params['models'][model_key]['estimator']
        base = common.manual_params(params, model_key, hotel)
        for index, grid_params in enumerate(ParameterGrid(grid), start=1):
            config_id = f'{model_key}{n_inputs}_{index:02d}'
            parameters = {**base, **grid_params}
            parts = []
            for fold in fold_ids:
                result = folds.run_fold(hotel, fold, estimator, tuning['features'], groups, parameters,
                                        dev, membership, split_config['validation_months'],
                                        train_metrics=True)
                row = result['metrics']
                row.update({'ConfigID': config_id, 'ModelKey': model_key,
                            'GridParameters': json.dumps(grid_params, sort_keys=True),
                            'DummyLogLoss': dummy[fold],
                            'GainVsDummy': dummy[fold] - row['LogLoss']})
                rows.append(row)
                parts.append(result['predictions'])
            predictions[config_id] = (model_key, parameters, grid_params, pd.concat(parts))
            print(f'{hotel} {config_id} {json.dumps(grid_params)}', flush=True)
    # LR convergence is asserted per fit; drop its bool so mixed-model columns stay numeric
    fold_metrics = pd.DataFrame(rows).drop(columns='Converged', errors='ignore')

    # Rank configurations within each model by the equal-weight mean of fold scores
    summary_rows = []
    for config_id, (model_key, parameters, grid_params, oof) in predictions.items():
        windows = fold_metrics.loc[fold_metrics['ConfigID'].eq(config_id)].sort_values('Fold')
        pooled = model_evaluation.evaluate_predictions(oof['IsCanceled'], oof['RawProbability'])
        summary_rows.append({
            'Hotel': hotel, 'ModelKey': model_key, 'ConfigID': config_id,
            'GridParameters': json.dumps(grid_params, sort_keys=True),
            **{metric: windows[metric].mean() for metric in metric_columns},
            'MeanTrainLogLoss': windows['TrainLogLoss'].mean(),
            'MeanGap': windows['GeneralizationGap'].mean(),
            'DummyLogLoss': windows['DummyLogLoss'].mean(),
            'GainVsDummy': windows['GainVsDummy'].mean(),
            'FoldsBeatingDummy': int(windows['GainVsDummy'].gt(common.TOLERANCE).sum()),
            'PooledOOFLogLoss': pooled['LogLoss'],
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ['ModelKey', primary, 'ConfigID'], kind='stable').reset_index(drop=True)
    summary['RankWithinModel'] = summary.groupby('ModelKey').cumcount() + 1
    best = summary.loc[summary['RankWithinModel'].eq(1)]

    # Publish winners for the cv stage and summaries for review
    tuned = {}
    for row in best.itertuples():
        model_key, parameters, grid_params, _ = predictions[row.ConfigID]
        tuned[model_key] = {'config_id': row.ConfigID, 'grid_params': grid_params,
                            'features': tuning['features'], primary: float(getattr(row, primary)),
                            'params': parameters}
    common.write_json(output_root / 'tuning' / hotel / 'tuned.json', tuned)
    common.write_tables(output_root / 'tuning' / hotel,
                        {'fold_metrics.csv': fold_metrics, 'cv_summary.csv': summary})
    print(best[['ModelKey', 'ConfigID', 'GridParameters', primary]].to_string(index=False))
    return {'tuned': tuned, 'fold_metrics': fold_metrics, 'summary': summary}
