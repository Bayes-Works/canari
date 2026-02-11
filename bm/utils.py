import numpy as np

def p50(y: np.ndarray, ypred: np.ndarray, spred: np.ndarray) -> float:
    """Computes the (non-normalized) median absolute error.

    :param y: Observed target values.
    :type y: np.ndarray
    :param ypred: Predicted mean values.
    :type ypred: np.ndarray
    :return: Sum of absolute deviations between targets and predictions.
    :rtype: float
    """
    e = y - ypred
    denom = np.sum(np.abs(y))
    return np.sum(np.abs(e))/denom


def p90(y: np.ndarray, ypred: np.ndarray, spred: np.ndarray) -> float:
    """Computes the (non-normalized) tilted loss for the 90th percentile.

    :param y: Observed target values.
    :type y: np.ndarray
    :param ypred: Predicted mean values.
    :type ypred: np.ndarray
    :param spred: Predicted standard deviations.
    :type spred: np.ndarray
    :return: 90th percentile tilted loss aggregated over targets.
    :rtype: float
    """
    ypred_90q = ypred + 1.282 * spred  # 1.282 is the z-score for the 90th percentile
    Iq = y > ypred_90q
    Iq_ = ~Iq
    e = y - ypred_90q
    denom = np.sum(np.abs(y))
    return np.sum(2 * e * (0.9 * Iq - 0.1 * Iq_))/denom