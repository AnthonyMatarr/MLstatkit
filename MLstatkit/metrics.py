# MLstatkit/metrics.py
from typing import Callable
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    brier_score_loss,  # ADDED
)
from sklearn.calibration import calibration_curve  # ADDED


def _binarize(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


################ ADDED OPEN ################
def ici(y_true, y_proba):
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=3, strategy="uniform"
    )
    return np.mean(np.abs(prob_true - prob_pred))


################ ADDED CLOSE ################


def get_metric_fn(
    metric_str: str, threshold: float = 0.5, average: str = "macro"
) -> Callable:
    """
    Return a callable metric function f(y_true, y_pred_prob) -> float.

    Supported metrics:
      - 'f1', 'accuracy', 'recall', 'precision'
      - 'roc_auc'
      - 'average_precision'
      - 'pr_auc' (area under PR curve via trapezoid on precision-recall)
      - ADDED: 'ici', 'brier'
    """
    m = (metric_str or "").lower()

    if m == "f1":
        return lambda y, y_pred: f1_score(
            np.asarray(y),
            (np.asarray(y_pred) >= threshold).astype(int),
            average=average,
        )
    ################ ADDED OPEN ################
    if m == "ici":
        return lambda y, y_pred: ici(np.asarray(y), np.asarray(y_pred))

    if m == "brier":
        return lambda y, y_pred: brier_score_loss(np.asarray(y), np.asarray(y_pred))
    ################ ADDED CLOSE ################

    if m == "accuracy":
        return lambda y, y_pred: accuracy_score(
            np.asarray(y), (np.asarray(y_pred) >= threshold).astype(int)
        )

    if m == "recall":
        return lambda y, y_pred: recall_score(
            np.asarray(y),
            (np.asarray(y_pred) >= threshold).astype(int),
            average=average,
        )

    if m == "precision":
        return lambda y, y_pred: precision_score(
            np.asarray(y),
            (np.asarray(y_pred) >= threshold).astype(int),
            average=average,
        )

    if m in {"roc_auc", "auc"}:
        return lambda y, y_pred: roc_auc_score(np.asarray(y), np.asarray(y_pred))

    if m == "average_precision":
        return lambda y, y_pred: average_precision_score(
            np.asarray(y), np.asarray(y_pred)
        )

    if m == "pr_auc":

        def _pr_auc(y, y_pred):
            precision, recall, _ = precision_recall_curve(
                np.asarray(y), np.asarray(y_pred)
            )
            return float(auc(recall, precision))

        return _pr_auc

    raise ValueError(f"Unsupported metric_str: {metric_str!r}")
