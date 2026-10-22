import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression


def _probability_vector(values):
    # Require a nonempty vector of finite probabilities
    probabilities = np.asarray(values, dtype=np.float64)
    if (probabilities.ndim != 1 or not len(probabilities)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError('Expected a nonempty vector of probabilities in [0, 1]')
    return probabilities


def fit_calibrator(probabilities, labels, method):
    # Validate the out-of-sample scores and binary outcomes supplied by the caller
    scores = _probability_vector(probabilities)
    outcomes = np.asarray(labels)
    if (outcomes.ndim != 1 or len(outcomes) != len(scores)
            or not np.isin(outcomes, [0, 1]).all()
            or len(np.unique(outcomes)) != 2):
        raise ValueError('Calibration requires aligned outcomes containing both classes')

    if method == 'sigmoid':
        # Fit Platt scaling on raw probabilities with smoothed binary targets
        positive = int(outcomes.sum())
        negative = len(outcomes) - positive
        targets = np.where(outcomes == 1, (positive + 1) / (positive + 2), 1 / (negative + 2))

        def objective(parameters):
            logits = -(parameters[0] * scores + parameters[1])
            residual = expit(logits) - targets
            loss = np.sum(np.logaddexp(0, logits) - targets * logits)
            gradient = np.array([-residual @ scores, -residual.sum()])
            return loss, gradient

        result = minimize(objective, [0, np.log((negative + 1) / (positive + 1))],
                          method='L-BFGS-B', jac=True,
                          options={'gtol': 1e-6, 'ftol': 64 * np.finfo(float).eps})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f'Platt optimization failed: {result.message}')
        return {'method': method, 'a': float(result.x[0]), 'b': float(result.x[1]),
                'iterations': int(result.nit), 'converged': True}

    if method == 'isotonic':
        # Learn a nondecreasing mapping and clip scores outside its fitted range
        estimator = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
        estimator.fit(scores, outcomes)
        return {'method': method, 'x': estimator.X_thresholds_.tolist(),
                'y': estimator.y_thresholds_.tolist()}

    raise ValueError(f'Unknown calibration method: {method}')


def apply_calibrator(calibrator, probabilities):
    # Apply the saved mapping without fitting or changing the base model
    scores = _probability_vector(probabilities)
    if calibrator['method'] == 'sigmoid':
        calibrated = expit(-(calibrator['a'] * scores + calibrator['b']))
    elif calibrator['method'] == 'isotonic':
        calibrated = np.interp(scores, calibrator['x'], calibrator['y'])
    else:
        raise ValueError(f"Unknown calibration method: {calibrator['method']}")
    return _probability_vector(calibrated)
