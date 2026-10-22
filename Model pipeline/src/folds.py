"""Fold cohorts and one fresh pipeline fit per hotel/fold (notebook run_candidate_fold)."""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from common import KEYS, TARGET, feature_columns
from utils import model_evaluation, model_pipeline


def fold_cohorts(dev, membership, fold, columns):
    # Join saved membership keys to development rows, keeping the membership order
    window = membership.loc[membership['Fold'].eq(fold)]
    fit_dates = window['FitDate'].unique()
    if len(fit_dates) != 1:
        raise ValueError(f'Fold {fold}: expected one fit date')
    cohorts = {}
    for role in ('Train', 'Validation'):
        keys = window.loc[window['Role'].eq(role), KEYS]
        joined = keys.merge(dev[KEYS + ['CutoffDate', TARGET] + columns], on=KEYS,
                            how='left', validate='one_to_one', indicator=True)
        if joined.empty or not joined['_merge'].eq('both').all():
            raise ValueError(f'Fold {fold}: unmatched or empty {role} cohort')
        cohorts[role] = joined.drop(columns='_merge')
    return pd.Timestamp(fit_dates[0]), cohorts['Train'], cohorts['Validation']


def fit_and_score(estimator_name, groups, model_params, train, columns):
    # Fit a fresh pipeline and capture convergence warnings separately
    pipeline = model_pipeline.build_pipeline(estimator_name, groups, model_params)
    if hasattr(pipeline.named_steps['model'], 'classes_'):
        raise AssertionError('Pipeline must start unfitted')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        pipeline.fit(train[columns].copy(), train[TARGET])
    convergence = [item for item in captured if issubclass(item.category, ConvergenceWarning)]
    for item in captured:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn(str(item.message), item.category)
    return pipeline, convergence


def class_one(pipeline, features):
    estimator = pipeline.named_steps['model']
    return pipeline.predict_proba(features)[:, list(estimator.classes_).index(1)].astype(float)


def tree_depth(node):
    return 0 if 'leaf' in node else 1 + max(tree_depth(child) for child in node['children'])


def run_fold(hotel, fold, estimator_name, feature_set_name, groups, model_params, dev, membership,
             validation_months, lookback_months=None, train_metrics=False):
    columns = feature_columns(groups)
    fit_date, train, valid = fold_cohorts(dev, membership, fold, columns)
    expanding_train = train

    # Restrict eligible train to the requested calendar history when configured
    if lookback_months is not None:
        if isinstance(lookback_months, bool) or not isinstance(lookback_months, int) or lookback_months < 1:
            raise ValueError('train_lookback_months must be a positive integer or null')
        train_start = fit_date - pd.DateOffset(months=lookback_months)
        train = train.loc[train['CutoffDate'].ge(train_start)].copy()
        if train.empty:
            raise ValueError(f'{hotel} fold {fold}: rolling train is empty')
    train_y, valid_y = train[TARGET], valid[TARGET]
    pipeline, convergence = fit_and_score(estimator_name, groups, model_params, train, columns)
    estimator = pipeline.named_steps['model']
    probabilities = class_one(pipeline, valid[columns].copy())
    metrics = model_evaluation.evaluate_predictions(valid_y, probabilities)
    validation_end = fit_date + pd.DateOffset(months=validation_months)

    # Audit cohorts, parameters and train-only preprocessing
    checks = {
        'Train and validation bookings are disjoint':
            not set(train['Booking_ID']).intersection(valid['Booking_ID']),
        'Train precedes the fit date': train['CutoffDate'].lt(fit_date).all(),
        'Validation inside its window': valid['CutoffDate'].ge(fit_date).all()
            and valid['CutoffDate'].lt(validation_end).all(),
        'Model parameters match config': all(
            estimator.get_params()[name] == value for name, value in model_params.items()),
        'One finite class-1 probability per booking': probabilities.shape == (len(valid),)
            and np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all(),
    }
    processed_columns = 0
    if estimator_name != 'dummy':
        preprocessor = pipeline.named_steps['preprocessor']
        numeric = preprocessor.named_transformers_['numeric']
        encoder = preprocessor.named_transformers_['categorical'].named_steps['encoder']
        processed_columns = len(preprocessor.get_feature_names_out())
        checks.update({
            'Preprocessing uses only configured inputs': list(preprocessor.feature_names_in_) == columns
                and estimator.n_features_in_ == processed_columns,
            'Numeric statistics fitted on eligible train': np.allclose(
                numeric.named_steps['imputer'].statistics_, train[groups['numeric']].median())
                and ('scaler' not in numeric.named_steps
                     or np.all(numeric.named_steps['scaler'].n_samples_seen_ == len(train))),
            'Encoder vocabulary fitted on eligible train': all(
                np.array_equal(categories, np.sort(train[column].fillna('__MISSING__').unique()))
                for column, categories in zip(groups['categorical'], encoder.categories_)),
            'Numeric scaling only for LR': ('scaler' in numeric.named_steps)
                == (estimator_name == 'logistic_regression'),
        })

    # Inspect optimization or tree construction for the selected model
    extra = {}
    if estimator_name == 'logistic_regression':
        checks['Optimization converged'] = not convergence
        extra = {'Iterations': int(estimator.n_iter_[0]), 'Converged': not convergence}
    elif estimator_name == 'random_forest':
        checks['All configured trees fitted'] = len(estimator.estimators_) == model_params['n_estimators']
        checks['Every leaf meets the minimum size'] = all(
            tree.tree_.n_node_samples[tree.tree_.children_left == -1].min()
            >= model_params['min_samples_leaf'] for tree in estimator.estimators_)
        extra = {'Trees': len(estimator.estimators_),
                 'MaxTreeDepth': max(tree.tree_.max_depth for tree in estimator.estimators_)}
    elif estimator_name == 'xgboost':
        booster = estimator.get_booster()
        depth = max(tree_depth(json.loads(tree)) for tree in booster.get_dump(dump_format='json'))
        checks['All configured boosting rounds fitted'] = (
            booster.num_boosted_rounds() == model_params['n_estimators'])
        checks['Trees respect the configured depth'] = depth <= model_params['max_depth']
        checks['Early stopping unused'] = estimator.get_params()['early_stopping_rounds'] is None
        extra = {'BoostingRounds': booster.num_boosted_rounds(), 'MaxTreeDepth': depth}
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f'{hotel} fold {fold} {estimator_name}/{feature_set_name}: {failed}')

    # Return comparable validation metrics and keyed OOF predictions
    row = {
        'Hotel': hotel, 'Model': estimator_name, 'FeatureSet': feature_set_name, 'Fold': fold,
        'FitDate': fit_date, 'ValidationStart': fit_date, 'ValidationEnd': validation_end,
        'TrainRows': len(train), 'TrainPositives': int(train_y.sum()),
        'TrainCancelRate': float(train_y.mean()),
        'ValidationRows': metrics.pop('Rows'), 'ValidationPositives': metrics.pop('Positives'),
        'ValidationCancelRate': metrics.pop('CancelRate'), **metrics,
        'InputColumns': len(columns), 'ProcessedColumns': processed_columns, **extra,
        'MeanProbability': float(probabilities.mean()),
        'ProbabilityBias': float(probabilities.mean() - valid_y.mean()),
    }
    if lookback_months is not None:
        row.update({'TrainWindowStart': train_start,
                    'HistoryRemovedRows': len(expanding_train) - len(train),
                    'HistoryRemovedPositives': int(expanding_train[TARGET].sum() - train_y.sum())})
    if train_metrics:
        scores = model_evaluation.evaluate_predictions(train_y, class_one(pipeline, train[columns]))
        row.update({f'Train{name}': scores[name] for name in ('LogLoss', 'BrierScore',
                                                              'ROC_AUC', 'AveragePrecision')})
        row['GeneralizationGap'] = row['LogLoss'] - row['TrainLogLoss']
        prior = np.full(len(train_y), train_y.mean())
        row['DummyTrainLogLoss'] = model_evaluation.evaluate_predictions(train_y, prior)['LogLoss']
    predictions = valid[KEYS + ['CutoffDate', TARGET]].rename(
        columns={'CutoffDate': 'PredictionDate'}).copy()
    predictions.insert(2, 'Model', estimator_name)
    predictions.insert(3, 'FeatureSet', feature_set_name)
    predictions.insert(4, 'Fold', fold)
    predictions['RawProbability'] = probabilities
    return {'pipeline': pipeline, 'metrics': row, 'predictions': predictions}
