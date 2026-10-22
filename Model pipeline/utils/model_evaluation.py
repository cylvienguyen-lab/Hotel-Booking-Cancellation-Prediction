import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_prob):
    # Check aligned binary labels and class-1 probability vectors
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError('Labels and class-1 probabilities must be one-dimensional')
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError('Labels and probabilities must have the same nonzero length')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels must contain only 0 and 1')
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError('Probabilities must be finite values between 0 and 1')
    if len(np.unique(labels)) != 2:
        raise ValueError('Both label classes are required to report all four metrics')

    # Calculate validation size and cancellation prevalence
    positives = int(labels.sum())
    results = {
        'Rows': len(labels),
        'Positives': positives,
        'CancelRate': float(positives / len(labels)),
    }

    # Evaluate probability quality and risk ranking
    results.update({
        'LogLoss': float(log_loss(labels, probabilities, labels=[0, 1])),
        'BrierScore': float(brier_score_loss(labels, probabilities, pos_label=1)),
        'ROC_AUC': float(roc_auc_score(labels, probabilities)),
        'AveragePrecision': float(average_precision_score(labels, probabilities, pos_label=1)),
    })
    return results
