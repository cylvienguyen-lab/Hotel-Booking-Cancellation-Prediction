"""Temporal calibration on earlier OOF predictions (notebook 6.8), shared by cv and final."""
import numpy as np
import pandas as pd

import common  # noqa: F401  (puts Model pipeline/ on sys.path for utils)
from utils import model_calibration, model_evaluation


def fit_and_apply(method, ready, scores):
    # raw keeps the base-model probabilities; other methods learn a mapping on ready rows only
    if method == 'raw':
        return {}, np.asarray(scores, dtype=float)
    state = model_calibration.fit_calibrator(ready['RawProbability'], ready['IsCanceled'], method)
    return state, model_calibration.apply_calibrator(state, scores)


def evaluate_folds(hotel, method, oof, dates, fold_metrics, test_start):
    # Fold k uses OOF rows from folds < k whose labels were available before fold k's fit date
    source = oof.merge(dates, on=['Hotel', 'Booking_ID'], how='left', validate='one_to_one')
    if source['LabelAvailableDate'].isna().any():
        raise AssertionError('OOF rows without a label-availability date')
    rows, parts, states = [], [], {}
    for reference in fold_metrics.sort_values('Fold').iloc[1:].itertuples():
        fold, fit_date = reference.Fold, pd.Timestamp(reference.FitDate)
        past = source.loc[source['Fold'].lt(fold)]
        ready = past.loc[past['LabelAvailableDate'].lt(fit_date)]
        valid = source.loc[source['Fold'].eq(fold)]
        if (ready.empty or ready['IsCanceled'].nunique() != 2
                or set(ready['Booking_ID']) & set(valid['Booking_ID'])
                or not ready['PredictionDate'].lt(fit_date).all()
                or not valid['PredictionDate'].ge(fit_date).all()
                or not valid['PredictionDate'].lt(pd.Timestamp(reference.ValidationEnd)).all()
                or not valid['LabelAvailableDate'].lt(pd.Timestamp(test_start)).all()):
            raise AssertionError(f'{hotel} fold {fold}: calibration chronology, labels or membership failed')
        state, probabilities = fit_and_apply(method, ready, valid['RawProbability'])
        scores = model_evaluation.evaluate_predictions(valid['IsCanceled'], probabilities)
        rows.append({
            'Hotel': hotel, 'Fold': fold, 'Method': method, 'FitDate': fit_date,
            'CalibrationRows': 0 if method == 'raw' else len(ready),
            'CalibrationPositives': 0 if method == 'raw' else int(ready['IsCanceled'].sum()),
            'ValidationRows': scores['Rows'], 'ValidationPositives': scores['Positives'],
            'ValidationCancelRate': scores['CancelRate'],
            **{name: scores[name] for name in ('LogLoss', 'BrierScore', 'ROC_AUC', 'AveragePrecision')},
            'RawLogLoss': reference.LogLoss, 'DummyLogLoss': reference.DummyLogLoss,
            'GainVsRaw': reference.LogLoss - scores['LogLoss'],
            'GainVsDummy': reference.DummyLogLoss - scores['LogLoss'],
            'MeanProbability': float(probabilities.mean()),
            'ProbabilityBias': float(probabilities.mean() - scores['CancelRate']),
            'ExactZeroRows': int((probabilities == 0).sum()),
            'ExactOneRows': int((probabilities == 1).sum()),
        })
        part = valid[['Hotel', 'Booking_ID', 'Fold', 'PredictionDate', 'IsCanceled',
                      'RawProbability']].copy()
        part['Method'], part['Probability'] = method, probabilities
        parts.append(part)
        states[str(fold)] = state
    return pd.DataFrame(rows), pd.concat(parts, ignore_index=True), states
