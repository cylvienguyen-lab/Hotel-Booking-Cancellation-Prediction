"""Refit on eligible development rows, calibrate, and score test once.

Rehearsal uses the last CV fold as a stand-in for test and reproduces its scores.
"""
import pickle

import pandas as pd

import calibration
import common
import folds
from utils import model_evaluation


def refit_and_score(hotel, params, model, groups, train, evaluation, calibration_rows):
    # Fit the configured candidate and the Dummy prior on the same refit cohort
    columns = common.feature_columns(groups)
    pipeline, convergence = folds.fit_and_score(model['estimator'], groups, model['params'], train, columns)
    if model['estimator'] == 'logistic_regression' and convergence:
        raise AssertionError(f'{hotel}: final LR did not converge')
    dummy, _ = folds.fit_and_score('dummy', groups, params['models']['dummy']['params'], train, columns)
    raw = folds.class_one(pipeline, evaluation[columns].copy())
    prior = folds.class_one(dummy, evaluation[columns].copy())
    state, probabilities = calibration.fit_and_apply(params['calibration'], calibration_rows, raw)
    labels = evaluation[common.TARGET]
    scores = {name: model_evaluation.evaluate_predictions(labels, values)
              for name, values in [('calibrated', probabilities), ('raw', raw), ('dummy', prior)]}
    predictions = evaluation[common.KEYS + ['CutoffDate', common.TARGET]].rename(
        columns={'CutoffDate': 'PredictionDate'}).copy()
    predictions['RawProbability'], predictions['Probability'] = raw, probabilities
    predictions['DummyProbability'] = prior
    return pipeline, state, scores, predictions


def cohort(keys, rows, columns):
    joined = keys[common.KEYS].merge(rows[common.KEYS + ['CutoffDate', common.TARGET] + columns],
                                     on=common.KEYS, how='left', validate='one_to_one', indicator=True)
    if joined.empty or not joined['_merge'].eq('both').all():
        raise ValueError('Cohort keys missing from the supplied rows')
    return joined.drop(columns='_merge')


def lookback(train, end_date, months):
    if months is None:
        return train
    return train.loc[train['CutoffDate'].ge(end_date - pd.DateOffset(months=months))].copy()


def run_final(hotel, params, dev, test, split_tables, cv_tables, tuned, output_root,
              rehearsal=True):
    split_config = params['split']
    groups = common.feature_config(params, params['features'])
    common.check_features(groups, common.load_contract())
    columns = common.feature_columns(groups)
    model = common.model_settings(params, hotel, tuned)
    oof = cv_tables['oof_predictions']
    if not (oof['ConfigID'].eq(model['config_id']).all() and oof['FeatureSet'].eq(params['features']).all()
            and oof['Model'].eq(model['estimator']).all()):
        raise AssertionError('cv OOF predictions were produced by a different configuration')
    source = oof.merge(common.label_dates(split_tables['fit_eligibility']), on=common.KEYS,
                       how='left', validate='one_to_one')

    if rehearsal:
        # Use the last CV fold as a stand-in test window
        last = split_config['n_folds']
        fit_date, train, evaluation = folds.fold_cohorts(dev, split_tables['cv_membership'], last, columns)
        train = lookback(train, fit_date, params['train_lookback_months'])
        past = source.loc[source['Fold'].lt(last)]
        calibration_rows = past.loc[past['LabelAvailableDate'].lt(fit_date)]
        directory = output_root / 'final_rehearsal' / hotel
    else:
        # Refit on development rows whose labels were available at test_start; score every test row
        fit_date = pd.Timestamp(split_config['test_start'])
        if not (test['CutoffDate'].ge(fit_date).all()
                and test['CutoffDate'].lt(pd.Timestamp(split_config['test_end'])).all()
                and not set(test['Booking_ID']) & set(dev['Booking_ID'])):
            raise AssertionError('Test rows fall outside the held-out window')
        train = lookback(cohort(split_tables['refit_membership'], dev, columns), fit_date,
                         params['train_lookback_months'])
        evaluation = test
        calibration_rows = source.loc[source['LabelAvailableDate'].lt(fit_date)]
        directory = output_root / 'final' / hotel
    if not train['CutoffDate'].lt(fit_date).all():
        raise AssertionError('Refit rows must precede the fit date')

    pipeline, state, scores, predictions = refit_and_score(
        hotel, params, model, groups, train, evaluation, calibration_rows)
    common.write_tables(directory, {'test_predictions.csv': predictions},
                        extra={'model': model, 'features': params['features'],
                               'calibration': params['calibration'], 'fit_date': str(fit_date.date()),
                               'train_rows': len(train), 'calibration_rows': len(calibration_rows)})
    summary = {name: {key: float(value) for key, value in values.items()} for name, values in scores.items()}

    if rehearsal:
        # The rehearsal must reproduce the cv stage's last-fold raw and calibrated scores
        fold_row = cv_tables['fold_metrics'].set_index('Fold').loc[last]
        calibrated_row = cv_tables['calibration_metrics'].set_index('Fold').loc[last]
        differences = {name: max(abs(summary['raw'][name] - fold_row[name]),
                                 abs(summary['calibrated'][name] - calibrated_row[name]))
                       for name in params['evaluation']['metrics']}
        common.write_json(directory / 'rehearsal_check.json', {'max_abs_difference': differences})
        if max(differences.values()) > common.TOLERANCE:
            raise AssertionError(f'{hotel}: rehearsal differs from cv fold {last}: {differences}')
        print(f'{hotel}: rehearsal on fold {last} reproduces cv scores (max diff {max(differences.values()):.1e})')
        return {'scores': summary, 'predictions': predictions, 'check': differences}

    # Package the refit model, calibrator and test scores
    model_directory = output_root / 'models' / hotel
    model_directory.mkdir(parents=True, exist_ok=True)
    (model_directory / 'model.pkl').write_bytes(pickle.dumps(pipeline))
    common.write_json(model_directory / 'calibrator.json',
                      {'method': params['calibration'], 'state': state})
    common.write_json(model_directory / 'model_card.json', {
        'hotel': hotel, 'model': model, 'features': params['features'], 'feature_config': groups,
        'calibration': params['calibration'], 'fit_date': str(fit_date.date()),
        'train_rows': len(train), 'train_lookback_months': params['train_lookback_months']})
    common.write_json(directory / 'summary.json', summary)
    print(f"{hotel}: final test LogLoss {summary['calibrated']['LogLoss']:.5f}")
    return {'scores': summary, 'predictions': predictions, 'model': pipeline,
            'calibrator': state}
