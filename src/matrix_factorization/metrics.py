import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def round_probabilities(probabilities, threshold):

    """
    Round probabilities to labels based on the given threshold

    Parameters
    ----------
    probabilities : numpy.ndarray of shape (n_samples)
        Predicted probabilities

    threshold: float
        Rounding threshold

    Returns
    -------
    labels : numpy.ndarray of shape (n_samples)
        Rounded hard labels
    """

    labels = np.zeros_like(probabilities, dtype=np.uint8)
    labels[probabilities >= threshold] = 1

    return labels


def classification_scores(y_true, y_pred, threshold=0.5):

    """
    Calculate binary classification metrics on predicted probabilities and labels

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground-truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted probabilities

    threshold: float
        Rounding threshold

    Returns
    -------
    scores: dict
        Dictionary of classification scores
    """

    y_pred_labels = round_probabilities(y_pred, threshold=threshold)
    scores = {
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'roc_auc': roc_auc_score(y_true, y_pred),
    }

    return scores
