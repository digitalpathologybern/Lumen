#!/usr/bin/env python3
"""Paired patient-cluster comparisons for WSI model scores.

The same resampled patients/cases and all of their slides are used for Lumen
and its comparator. Outputs effect sizes, percentile CIs, two-sided bootstrap
p-values, and Holm-adjusted p-values for exploratory multi-model comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.paths import project_root  # noqa: E402
from lumen.stats import holm_adjust as shared_holm, paired_bootstrap  # noqa: E402
from lumen.models.registry import PAPER_CHECKPOINT  # noqa: E402


def metric_pair(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                threshold_a: float, threshold_b: float) -> tuple[float, float]:
    if np.unique(y).size < 2:
        return np.nan, np.nan
    return (
        float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        float(balanced_accuracy_score(y, a >= threshold_a) -
              balanced_accuracy_score(y, b >= threshold_b)),
    )


def paired_cluster_bootstrap(frame: pd.DataFrame, threshold_a: float,
                             threshold_b: float, n_boot: int, seed: int) -> dict:
    """Paired AUROC and balanced-accuracy deltas over patient/case clusters.

    Delegates the resampling to :mod:`lumen.stats` so this shares one
    definition of the unit with every other experiment.
    """
    y = frame["gt"].to_numpy(dtype=int)
    a = frame["score_lumen"].to_numpy(dtype=float)
    b = frame["score_baseline"].to_numpy(dtype=float)
    clusters = frame["patient_id"].astype(str).to_numpy()

    out = {}
    for name, stat_a, stat_b in (
        ("auroc",
         lambda i: _auroc(y[i], a[i]), lambda i: _auroc(y[i], b[i])),
        ("balanced_accuracy",
         lambda i: _balacc(y[i], a[i], threshold_a),
         lambda i: _balacc(y[i], b[i], threshold_b)),
    ):
        res = paired_bootstrap(stat_a, stat_b, clusters,
                               n_boot=n_boot, seed=seed)
        out[f"delta_{name}"] = res["delta"]
        out[f"delta_{name}_ci_lo"] = res["ci_lo"]
        out[f"delta_{name}_ci_hi"] = res["ci_hi"]
        out[f"delta_{name}_p"] = res["p"]
        out["n_boot_valid"] = res["n_boot_valid"]
    return out


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    return np.nan if np.unique(y).size < 2 else float(roc_auc_score(y, s))


def _balacc(y: np.ndarray, s: np.ndarray, threshold: float) -> float:
    return (np.nan if np.unique(y).size < 2
            else float(balanced_accuracy_score(y, s >= threshold)))


def holm_adjust(values: pd.Series) -> pd.Series:
    """Index-preserving wrapper over :func:`lumen.stats.holm_adjust`."""
    return pd.Series(shared_holm(values.to_numpy()), index=values.index)


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores-dir", type=Path,
                    default=root / "outputs/wsi_benchmark/slide_scores")
    ap.add_argument("--out", type=Path,
                    default=root / "outputs/wsi_benchmark/paired_patient_comparisons.csv")
    ap.add_argument("--thresholds", type=Path,
                    default=root / "outputs/wsi_locked_threshold/summary.csv")
    # Must default to the paper's checkpoint. Defaulting to ``lumen``
    # produced paired deltas for a different model (+0.032 vs PathGen where the
    # paper reports +0.036), which is close enough to look right.
    ap.add_argument("--reference", default=PAPER_CHECKPOINT)
    ap.add_argument("--pooling", choices=("max", "top5"), default="max")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    score_col = f"ens_{args.pooling}_prob_pos"
    threshold_frame = pd.read_csv(args.thresholds)
    threshold_map = threshold_frame.set_index("model")["locked_threshold"].to_dict()
    if args.reference not in threshold_map:
        raise ValueError(f"No locked threshold for {args.reference} in {args.thresholds}")
    paths = sorted(args.scores_dir.glob("*.csv"))
    models = sorted({p.stem.split("__", 1)[1] for p in paths})
    tables = ["external"]
    rows = []
    for table in tables:
        ref_path = args.scores_dir / f"{table}__{args.reference}.csv"
        if not ref_path.exists():
            continue
        ref = pd.read_csv(ref_path)
        if "evaluable" in ref.columns:
            ref = ref[ref["evaluable"].astype(bool)]
        ref = ref[["slide", "group", "patient_id", "gt", score_col]].rename(
            columns={score_col: "score_lumen"}
        )
        for model in models:
            if model == args.reference:
                continue
            if model not in threshold_map:
                continue
            other_path = args.scores_dir / f"{table}__{model}.csv"
            if not other_path.exists():
                continue
            other = pd.read_csv(other_path)[["slide", "group", "gt", score_col]].rename(
                columns={"gt": "gt_baseline", score_col: "score_baseline"}
            )
            paired = ref.merge(
                other, on=["group", "slide"], how="inner", validate="one_to_one"
            )
            paired = paired[paired["gt"] == paired["gt_baseline"]].copy()
            if paired["gt"].nunique() < 2:
                continue
            rows.append({
                "table": table, "reference": args.reference, "baseline": model,
                "pooling": args.pooling, "n_slides": len(paired),
                "n_patients": paired["patient_id"].nunique(),
                **paired_cluster_bootstrap(
                    paired, threshold_map[args.reference], threshold_map[model],
                    args.n_boot, args.seed
                ),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No paired model score files were available for comparison")
    for table, idx in out.groupby("table").groups.items():
        for metric in ("auroc", "balanced_accuracy"):
            col = f"delta_{metric}_p"
            out.loc[idx, f"{col}_holm"] = holm_adjust(out.loc[idx, col])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print(f"\nOutput -> {args.out}")


if __name__ == "__main__":
    main()
