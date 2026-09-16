#!/usr/bin/env python3
"""Backfill patient IDs and clustered CIs from existing WSI slide scores.

This deliberately does not load model checkpoints or cached tile embeddings.
The continuous per-slide scores are already present and unchanged; only the
resampling unit and resulting confidence intervals are recomputed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumen.data.registry import SLIDE_TABLES  # noqa: E402
from lumen.inference.io import load_slide_table, safe_name  # noqa: E402
from lumen.paths import project_root  # noqa: E402
from lumen.wsi.slide_benchmark import bootstrap_ci  # noqa: E402


def patient_map(table: str) -> dict[tuple[str, str], str]:
    # Keys must be built under the same ``safe_name`` transform that produced
    # the cached slide stems these are later joined against.
    meta = load_slide_table(SLIDE_TABLES[table])
    return {
        (safe_name(row.group), safe_name(row.slide)): str(row.patient_id)
        for row in meta.itertuples(index=False)
    }


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-dir", type=Path,
                    default=root / "outputs/wsi_benchmark")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    summary_path = args.benchmark_dir / "summary.csv"
    summary = pd.read_csv(summary_path)
    maps = {table: patient_map(table) for table in SLIDE_TABLES}

    for path in sorted((args.benchmark_dir / "slide_scores").glob("*.csv")):
        table, model = path.stem.split("__", 1)
        if table not in maps:
            continue
        frame = pd.read_csv(path)
        mapping = maps[table]
        frame["patient_id"] = [
            mapping.get((str(group), str(slide)),
                        f"{table}::{group}::SLIDE::{slide}")
            for group, slide in zip(frame["group"], frame["slide"])
        ]
        n_fallback = sum(1 for g, s in zip(frame["group"], frame["slide"])
                         if (str(g), str(s)) not in mapping)
        if n_fallback:
            print(f"[warn] {table}/{model}: {n_fallback} of {len(frame)} slides "
                  f"fall back to one cluster per slide")
        frame.to_csv(path, index=False)

        evaluable = (frame["evaluable"].astype(bool) if "evaluable" in frame
                     else np.ones(len(frame), dtype=bool))
        used = frame[evaluable]
        for pooling in ("max", "top5"):
            # ``slide_scores`` writes ``ens_<pooling>_prob_pos``; the older
            # ``ens_<pooling>`` spelling is accepted so historical caches still
            # load.
            score_col = next(
                (c for c in (f"ens_{pooling}_prob_pos", f"ens_{pooling}")
                 if c in used.columns),
                None,
            )
            if score_col is None:
                raise KeyError(
                    f"{path.name} has no ensemble score column for pooling "
                    f"'{pooling}'; looked for ens_{pooling}_prob_pos and "
                    f"ens_{pooling}. Columns: {list(used.columns)}"
                )
            ci = bootstrap_ci(
                used["gt"].to_numpy(dtype=int),
                used[score_col].to_numpy(dtype=float),
                used["patient_id"].to_numpy(dtype=str),
                n_boot=args.n_boot,
                seed=args.seed,
            )
            mask = ((summary["table"] == table) &
                    (summary["model"] == model) &
                    (summary["pooling"] == pooling))
            summary.loc[mask, "n_patients"] = used["patient_id"].nunique()
            summary.loc[mask, "ensemble_auroc_ci_lo"] = ci["auroc_ci_lo"]
            summary.loc[mask, "ensemble_auroc_ci_hi"] = ci["auroc_ci_hi"]
            summary.loc[mask, "ensemble_accuracy_ci_lo"] = ci["accuracy_ci_lo"]
            summary.loc[mask, "ensemble_accuracy_ci_hi"] = ci["accuracy_ci_hi"]
            summary.loc[mask, "ensemble_balanced_accuracy_ci_lo"] = ci["balanced_accuracy_ci_lo"]
            summary.loc[mask, "ensemble_balanced_accuracy_ci_hi"] = ci["balanced_accuracy_ci_hi"]
        print(f"updated {table} x {model}", flush=True)

    summary.to_csv(summary_path, index=False)
    summary.to_json(args.benchmark_dir / "summary.json", orient="records", indent=2)
    print(f"Updated clustered statistics -> {summary_path}")


if __name__ == "__main__":
    main()
