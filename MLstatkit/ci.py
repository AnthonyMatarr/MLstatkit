# MLstatkit/ci.py
from typing import Tuple, NamedTuple, Optional
import numpy as np
from .metrics import get_metric_fn

try:
    from tqdm import tqdm
except Exception:  # 讓 tqdm 非強制 (make tqdm non-necessary)

    def tqdm(x, **kwargs):
        return x


class BootstrapResult(NamedTuple):
    # ADDED
    score: float
    ci_lower: float
    ci_upper: float


class PairedBootstrapResult(NamedTuple):
    # ADDED
    score_a: float
    score_b: float
    diff: float  # score_a - score_b
    ci_lower: float
    ci_upper: float
    # p_value: float


def Bootstrapping(
    y_true,
    y_prob,
    y_prob_b=None,  # added
    metric_str: str = "f1",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    threshold: float = 0.5,
    average: str = "macro",
    random_state: int = 0,
    show_progress: bool = True,  # Edited to add this flag for tqdm
    max_attempt_factor: int = 10,
    **metric_kwargs,  # Edited to add **metric_kwargs
) -> Tuple[float, float, float]:
    """
    Single model:  BootstrapResult(score, ci_lower, ci_upper, n_valid)
    Paired (y_prob_b given): PairedBootstrapResult, where diff = A - B and the
    CI/p-value describe the difference under the same resampled indices.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    assert y_true.shape[0] == y_prob.shape[0]

    paired = y_prob_b is not None
    if paired:
        y_prob_b = np.asarray(y_prob_b)
        assert (
            y_prob_b.shape == y_prob.shape
        ), "both prediction sets must align with y_true"

    rng = np.random.RandomState(random_state)
    metric_fn = get_metric_fn(metric_str, threshold, average)

    score_a = float(metric_fn(y_true, y_prob, **metric_kwargs))
    score_b = float(metric_fn(y_true, y_prob_b, **metric_kwargs)) if paired else None

    n = len(y_true)
    scores_a, scores_b = [], []
    attempts = 0
    max_attempts = n_bootstraps * max_attempt_factor

    desc = f"Bootstrapping {metric_str}" + (" (paired)" if paired else "")
    with tqdm(total=n_bootstraps, desc=desc, disable=not show_progress) as pbar:
        while len(scores_a) < n_bootstraps:
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Only {len(scores_a)}/{n_bootstraps} valid bootstrap samples "
                    f"after {attempts} draws. y_true is likely too imbalanced "
                    f"(minority count = {int(min(np.bincount(y_true.ravel().astype(int))))}); "
                    f"consider stratified resampling."
                )
            attempts += 1
            idx = rng.randint(0, n, n)
            y_sub_true = y_true[idx]
            if np.unique(y_sub_true).size < 2:
                continue

            scores_a.append(metric_fn(y_sub_true, y_prob[idx], **metric_kwargs))
            if paired:
                scores_b.append(metric_fn(y_sub_true, y_prob_b[idx], **metric_kwargs))
            pbar.update(1)

    alpha = (1 - confidence_level) / 2.0
    if not paired:
        return BootstrapResult(
            score_a,
            float(np.percentile(scores_a, 100 * alpha)),
            float(np.percentile(scores_a, 100 * (1 - alpha))),
        )
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    # n_le, n_ge = int(np.sum(diffs <= 0)), int(np.sum(diffs >= 0))
    # p_value = min(1.0, 2 * (min(n_le, n_ge) + 1) / (n_bootstraps + 1))

    return PairedBootstrapResult(
        score_a,
        score_b,
        score_a - score_b,
        float(np.percentile(diffs, 100 * alpha)),
        float(np.percentile(diffs, 100 * (1 - alpha))),
        # float(p_value),
    )
