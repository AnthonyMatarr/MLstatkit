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
def ici_w_thresholds(y_true, y_proba, thresholds):
    """
    Computes ICI given custom bin thresholds
    """
    thresholds = np.asarray(thresholds, dtype=float).flatten()
    n_bins = len(thresholds) + 1
    bin_indices = np.digitize(y_proba, thresholds, right=False)  # 0,1,...,n_bins-1

    event_rates = []
    mean_preds = []
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            event_rates.append(np.nan)
            mean_preds.append(np.nan)
        else:
            event_rates.append(np.nanmean(y_true[mask]))
            mean_preds.append(np.nanmean(y_proba[mask]))
    ici = np.nanmean(np.abs(np.asarray(event_rates) - np.asarray(mean_preds)))
    return ici


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
            average=average,  # type: ignore
        )
    ################ ADDED OPEN ################
    if m == "ici":

        def ici_metric(y, y_pred, **kwargs):
            bin_thresholds = kwargs.get("bin_thresholds", None)
            if bin_thresholds is None:
                raise ValueError("ICI metric requires thresholds to be passed.")
            return ici_w_thresholds(np.asarray(y), np.asarray(y_pred), bin_thresholds)

        return ici_metric

    if m == "brier":
        return lambda y, y_pred: brier_score_loss(np.asarray(y), np.asarray(y_pred))

    if m == "event_rate":
        # For event rate, we don't use y_pred at all - just compute mean of y_true
        # But the signature still needs y_pred for consistency with Bootstrapping
        return lambda y, y_pred: np.mean(np.asarray(y))
    ################ ADDED CLOSE ################

    if m == "accuracy":
        return lambda y, y_pred: accuracy_score(
            np.asarray(y), (np.asarray(y_pred) >= threshold).astype(int)
        )

    if m == "recall":
        return lambda y, y_pred: recall_score(
            np.asarray(y),
            (np.asarray(y_pred) >= threshold).astype(int),
            average=average,  # type: ignore
        )

    if m == "precision":
        return lambda y, y_pred: precision_score(
            np.asarray(y),
            (np.asarray(y_pred) >= threshold).astype(int),
            average=average,  # type: ignore
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
