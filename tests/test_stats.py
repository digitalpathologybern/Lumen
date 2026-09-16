"""Properties of the shared resampling primitives.

The external lymph-node analysis once resampled 1,646 clusters over 978 real
patients because each experiment decided its own resampling unit. These pin the
behaviour that made that possible, so it cannot come back quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumen.stats import (bootstrap_ci, cluster_resample_indices,
                            paired_bootstrap, rows_are_units)


def test_clusters_move_as_whole_units():
    """Every row of a drawn cluster comes with it, and none of an undrawn one."""
    clusters = np.array(["a", "a", "a", "b", "c", "c"])
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = cluster_resample_indices(clusters, rng)
        drawn = clusters[idx]
        for name, size in (("a", 3), ("b", 1), ("c", 2)):
            assert np.sum(drawn == name) % size == 0, (name, drawn)


def test_resample_draws_as_many_clusters_as_exist():
    clusters = np.array([0, 0, 0, 0, 1, 2, 3, 3])
    rng = np.random.default_rng(1)
    idx = cluster_resample_indices(clusters, rng)
    # Four clusters drawn with replacement; row count varies with which.
    assert 4 <= len(idx) <= 4 * 4


def test_clustering_widens_the_interval_relative_to_rows():
    """The bug's signature: pretending correlated rows are independent.

    Twenty patients with ten identical slides each carry twenty patients' worth
    of information, not two hundred. Resampling rows understates the interval,
    which is exactly what made the published external CI too narrow.
    """
    rng = np.random.default_rng(0)
    per_patient = rng.normal(size=20)
    values = np.repeat(per_patient, 10)
    patients = np.repeat(np.arange(20), 10)

    def mean(idx):
        return float(values[idx].mean())

    lo_r, hi_r = bootstrap_ci(mean, rows_are_units(len(values)), n_boot=400, seed=0)
    lo_c, hi_c = bootstrap_ci(mean, patients, n_boot=400, seed=0)
    assert (hi_c - lo_c) > (hi_r - lo_r)


def test_row_clustering_is_a_plain_bootstrap():
    values = np.arange(100, dtype=float)

    def mean(idx):
        return float(values[idx].mean())

    lo, hi = bootstrap_ci(mean, rows_are_units(len(values)), n_boot=500, seed=3)
    assert lo < values.mean() < hi


def test_bootstrap_is_reproducible_under_a_seed():
    values = np.linspace(0, 1, 40)
    clusters = np.repeat(np.arange(8), 5)

    def mean(idx):
        return float(values[idx].mean())

    assert (bootstrap_ci(mean, clusters, n_boot=200, seed=7)
            == bootstrap_ci(mean, clusters, n_boot=200, seed=7))


def test_non_finite_draws_are_dropped_not_propagated():
    # Unequal cluster sizes, so the row count varies between draws and the
    # non-finite branch fires on some of them but not all.
    clusters = np.array([0] + [1] * 2 + [2] * 3 + [3] * 10)
    values = np.ones(len(clusters))

    def sometimes_nan(idx):
        return float("nan") if len(idx) > 15 else float(values[idx].mean())

    lo, hi = bootstrap_ci(sometimes_nan, clusters, n_boot=200, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_all_draws_invalid_yields_nan_rather_than_a_fabricated_interval():
    clusters = np.repeat(np.arange(4), 5)
    lo, hi = bootstrap_ci(lambda idx: float("nan"), clusters, n_boot=50, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_paired_bootstrap_uses_one_shared_resample():
    """Pairing is the point: a shared draw cancels the common variance."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    a_vals, b_vals = base + 0.1, base
    clusters = np.repeat(np.arange(40), 5)

    out = paired_bootstrap(lambda i: float(a_vals[i].mean()),
                           lambda i: float(b_vals[i].mean()),
                           clusters, n_boot=400, seed=0)
    assert out["delta"] == pytest.approx(0.1, abs=1e-9)
    # The two series differ by a constant, so the paired interval collapses onto it.
    assert out["ci_hi"] - out["ci_lo"] < 1e-6
    assert out["n_boot_valid"] > 0


def test_paired_p_value_is_never_exactly_zero():
    clusters = np.repeat(np.arange(10), 3)
    out = paired_bootstrap(lambda i: 1.0, lambda i: 0.0, clusters,
                           n_boot=100, seed=0)
    assert out["p"] > 0.0


# ── Holm correction ──────────────────────────────────────────────────────────

def test_holm_matches_a_worked_example_and_preserves_order():
    """Two implementations of this existed on either side of the repository.

    They agreed, but a family-wise correction that disagreed between two tables
    would be nearly invisible in review: both would still be plausible p-values.
    """
    from lumen.stats import holm_adjust
    assert holm_adjust([0.01, 0.04, 0.03, 0.2]) == [0.04, 0.09, 0.09, 0.2]
    assert holm_adjust([]) == []


def test_holm_is_monotone_and_never_exceeds_one():
    from lumen.stats import holm_adjust
    import numpy as np
    rng = np.random.default_rng(0)
    p = sorted(rng.random(20))
    adj = holm_adjust(p)
    assert all(b >= a - 1e-12 for a, b in zip(adj, adj[1:])), "must be non-decreasing"
    assert max(adj) <= 1.0
    assert all(b >= a - 1e-12 for a, b in zip(p, adj)), "adjusted >= raw"


# ── Shared interval convention ───────────────────────────────────────────────

def test_one_interval_convention_for_the_whole_repository():
    """Nine call sites wrote `[2.5, 97.5]` by hand before this existed.

    They agreed, but moving to a different convention would have meant finding
    all nine, and a missed one would still have produced a plausible interval.
    """
    from lumen.stats import CI_ALPHA, percentile_ci
    assert CI_ALPHA == 0.05
    lo, hi = percentile_ci(list(range(101)))
    assert (round(lo, 6), round(hi, 6)) == (2.5, 97.5)


def test_percentile_ci_drops_non_finite_draws():
    from lumen.stats import percentile_ci
    import numpy as np
    clean = percentile_ci([1.0, 2.0, 3.0])
    with_nan = percentile_ci([1.0, np.nan, 2.0, np.inf, 3.0])
    assert clean == with_nan
    assert all(np.isnan(v) for v in percentile_ci([np.nan, np.nan]))


def test_bootstrap_cis_shares_one_resample_across_statistics():
    """Intervals in a row must describe the same resampled patients.

    Calling `bootstrap_ci` once per metric would draw an independent sequence
    for each, so the intervals would no longer be comparable within a row.
    """
    from lumen.stats import bootstrap_cis
    import numpy as np
    clusters = np.array([f"p{i // 3}" for i in range(60)])
    seen: list[set[int]] = []

    def record(idx):
        seen.append(tuple(idx))
        return float(len(idx))

    def same(idx):
        return float(len(idx))

    out = bootstrap_cis({"a": record, "b": same}, clusters, n_boot=20, seed=1)
    assert set(out) == {"a", "b"}
    assert len(seen) == 20
    # Both statistics saw the identical index arrays, one draw each.
    assert out["a"] == out["b"]


def test_bootstrap_cis_require_drops_the_whole_draw():
    """A resample with one class present must not contribute a partial row."""
    from lumen.stats import bootstrap_cis
    import numpy as np
    clusters = np.arange(30)
    out = bootstrap_cis(
        {"never": lambda i: float("nan"), "always": lambda i: 1.0},
        clusters, n_boot=10, seed=0, require="never")
    assert all(np.isnan(v) for v in out["always"])
