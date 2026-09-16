"""Resampling primitives shared by every experiment in this work.

:func:`bootstrap_ci` and :func:`paired_bootstrap` both require ``cluster_ids``.
There is no default: the resampling unit is the one decision that must not be
made implicitly, and six separate implementations of it is how the external
lymph-node analysis once resampled 1,646 clusters over 978 patients. Where the
unit genuinely is the row, pass :func:`rows_are_units` at the call site.

All intervals are percentile intervals over the resampling distribution.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = ["rows_are_units", "cluster_resample_indices", "bootstrap_ci",
           "bootstrap_cis", "paired_bootstrap", "holm_adjust", "percentile_ci",
           "CI_ALPHA"]

#: Two-sided interval width used for every reported interval in this work.
#: Stated once: nine call sites wrote ``[2.5, 97.5]`` by hand, so moving to a
#: different convention would have meant finding all nine, and a missed one
#: would still have produced a plausible interval.
CI_ALPHA = 0.05


def rows_are_units(n: int) -> np.ndarray:
    """Cluster labels for the case where each row is its own unit.

    Spell this out at the call site rather than defaulting to it. An analysis
    over patches with no recoverable patient identifier is legitimately
    row-clustered; an analysis over slides from 978 patients is not, and the two
    should not look the same in the source.
    """
    return np.arange(int(n))


def cluster_resample_indices(cluster_ids: np.ndarray,
                             rng: np.random.Generator) -> np.ndarray:
    """Sample clusters with replacement and return every row they contain.

    Whole clusters move together, so correlated rows (slides of one patient,
    patches of one slide) are never split across a resample.
    """
    cluster_ids = np.asarray(cluster_ids)
    uniq, inverse = np.unique(cluster_ids, return_inverse=True)
    # Row positions per cluster, computed once rather than per draw.
    order = np.argsort(inverse, kind="stable")
    boundaries = np.searchsorted(inverse[order], np.arange(len(uniq) + 1))
    members = [order[boundaries[i]:boundaries[i + 1]] for i in range(len(uniq))]
    drawn = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[d] for d in drawn])


def percentile_ci(values: Sequence[float],
                  alpha: float = CI_ALPHA) -> tuple[float, float]:
    """Two-sided percentile interval, dropping non-finite draws."""
    finite = np.asarray([v for v in np.asarray(values, dtype=float)
                         if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_cis(statistics: dict[str, Callable[[np.ndarray], float]],
                  cluster_ids: np.ndarray, *, n_boot: int = 1000,
                  seed: int = 0, alpha: float = CI_ALPHA,
                  require: str | None = None,
                  progress: Callable[[int, int], None] | None = None
                  ) -> dict[str, tuple[float, float]]:
    """Intervals for several statistics over one shared set of resamples.

    Computing them together is not only cheaper than repeated
    :func:`bootstrap_ci` calls: it guarantees every interval describes the same
    resampled patients, which is what makes them comparable within a row.

    ``require`` names a statistic that must be finite for a draw to count; a
    resample with one class present yields no AUROC, and keeping its accuracy
    while dropping its AUROC would silently pool different draw sets.
    """
    if n_boot <= 0:
        return {k: (float("nan"), float("nan")) for k in statistics}
    cluster_ids = np.asarray(cluster_ids)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in statistics}
    step = max(1, n_boot // 20)
    for i in range(n_boot):
        idx = cluster_resample_indices(cluster_ids, rng)
        values = {k: fn(idx) for k, fn in statistics.items()}
        if require is not None and not np.isfinite(values[require]):
            continue
        for k, v in values.items():
            draws[k].append(float(v))
        if progress is not None and ((i + 1) % step == 0 or i + 1 == n_boot):
            progress(i + 1, n_boot)
    return {k: percentile_ci(v, alpha) for k, v in draws.items()}


def bootstrap_ci(statistic: Callable[[np.ndarray], float],
                 cluster_ids: np.ndarray, *, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval for ``statistic`` under cluster resampling.

    ``statistic`` receives the row indices of one resample and returns a scalar.
    Draws that produce a non-finite value (a resample with one class present,
    say) are dropped rather than propagated as NaN.
    """
    if n_boot <= 0:
        return float("nan"), float("nan")
    cluster_ids = np.asarray(cluster_ids)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        value = statistic(cluster_resample_indices(cluster_ids, rng))
        if np.isfinite(value):
            values.append(float(value))
    return percentile_ci(values, alpha)


def paired_bootstrap(statistic_a: Callable[[np.ndarray], float],
                     statistic_b: Callable[[np.ndarray], float],
                     cluster_ids: np.ndarray, *, n_boot: int = 1000,
                     seed: int = 0, alpha: float = 0.05) -> dict:
    """Difference ``a - b`` under a shared cluster resample.

    Both statistics see the same resampled rows, which is what makes the
    comparison paired and is why the difference is far better determined than
    either endpoint.

    Returns the observed difference, its interval, and a two-sided bootstrap
    p-value with an add-one correction so it can never be reported as zero.
    """
    cluster_ids = np.asarray(cluster_ids)
    all_rows = np.arange(len(cluster_ids))
    observed = float(statistic_a(all_rows) - statistic_b(all_rows))

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = cluster_resample_indices(cluster_ids, rng)
        delta = statistic_a(idx) - statistic_b(idx)
        if np.isfinite(delta):
            deltas.append(float(delta))
    if not deltas:
        return {"delta": observed, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p": float("nan"), "n_boot_valid": 0}

    arr = np.asarray(deltas)
    lo, hi = percentile_ci(arr, alpha)
    left = (np.sum(arr <= 0) + 1) / (len(arr) + 1)
    right = (np.sum(arr >= 0) + 1) / (len(arr) + 1)
    return {"delta": observed, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": float(min(1.0, 2 * min(left, right))),
            "n_boot_valid": len(arr)}


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    There were two implementations of this, one iterative and one vectorised,
    on different sides of the repository. They agreed, but a family-wise
    correction that disagreed between two tables would be almost impossible to
    spot in review, because both would still look like plausible p-values.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return []
    order = np.argsort(p)
    # Step down: each p is scaled by the number of hypotheses not yet rejected,
    # then made monotone so an adjusted value can never fall below an earlier one.
    scaled = (p.size - np.arange(p.size)) * p[order]
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return [float(v) for v in out]
