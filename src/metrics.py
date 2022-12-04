import numpy as np


def click_recall(y_true, y_pred):

    """
    Calculate recall for clicks on ground-truth and predictions

    Parameters
    ----------
    y_true: int
        Ground-truth click aid

    y_pred: array-like of shape (n_aids) (1 <= n_aids <= 20)
        Prediction click aids

    Returns
    -------
    recall: int
        Recall calculated on ground-truth and predictions
    """

    recall = int(y_true in y_pred)
    return recall


def cart_order_recall(y_true, y_pred):

    """
    Calculate recall for carts/orders on ground-truth and predictions

    Parameters
    ----------
    y_true: array-like of shape (n_aids) (1 <= n_aids)
        Ground-truth cart/order aids

    y_pred: array-like of shape (n_aids) (1 <= n_aids <= 20)
        Prediction cart/order aids

    Returns
    -------
    recall: int
        Recall calculated on ground-truth and predictions
    """

    y_true = set(y_true)
    y_pred = set(y_pred)

    tp = len(y_true.intersection(y_pred))
    fn = len(y_true - y_pred)
    try:
        recall = tp / min(20, (tp + fn))
    except ZeroDivisionError:
        recall = np.nan

    return recall
