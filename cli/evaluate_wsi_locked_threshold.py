#!/usr/bin/env python3
"""Calibrate once on 20% of internal patients and evaluate every held-out slide.

For every model, Youden's threshold is learned from the same patient-stratified
internal calibration split.  The model-specific threshold is then locked and
applied unchanged to the remaining internal patients and every external cohort.
TCGA melanoma is outside the benchmark and is removed defensively on input.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.models.registry import PAPER_CHECKPOINT  # noqa: E402
from lumen.paths import project_root  # noqa: E402
from lumen.wsi.calibration import (  # noqa: E402
    bootstrap_fixed_threshold,
    build_patient_split,
    fixed_threshold_metrics,
    learn_youden_threshold,
)
from lumen.wsi.cohorts import exclude_benchmark_groups  # noqa: E402


SCORE_COLUMNS = {
    "max": "ens_max_prob_pos",
    "top5": "ens_top5_prob_pos",
}

LOGGER = logging.getLogger("wsi_locked_threshold")


def _load_scores(path: Path, score_col: str) -> pd.DataFrame:
    frame = exclude_benchmark_groups(pd.read_csv(path))
    required = {"slide", "group", "patient_id", "gt", score_col}
    missing = required - set(frame)
    if missing:
        raise ValueError(
            f"{path} lacks {sorted(missing)}; rerun cli/wsi_benchmark_evaluate.py "
            "with the positive-probability scoring protocol"
        )
    return frame


def _split_metrics(frame: pd.DataFrame, score_col: str, threshold: float) -> dict:
    return fixed_threshold_metrics(
        frame["gt"].to_numpy(dtype=int),
        frame[score_col].to_numpy(dtype=float),
        threshold,
    )


def main() -> None:
    root = project_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores-dir", type=Path,
                    default=root / "outputs" / "wsi_benchmark" / "slide_scores")
    ap.add_argument("--out-dir", type=Path,
                    default=root / "outputs" / "wsi_locked_threshold")
    ap.add_argument("--pooling", choices=tuple(SCORE_COLUMNS), default="max")
    ap.add_argument("--calibration-fraction", type=float, default=0.20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    run_start = time.perf_counter()
    LOGGER.info(
        "Locked-threshold evaluation: pooling=%s score=%s n_boot=%d seed=%d",
        args.pooling, SCORE_COLUMNS[args.pooling], args.n_boot, args.seed,
    )
    LOGGER.info("Scores dir: %s", args.scores_dir)

    score_col = SCORE_COLUMNS[args.pooling]
    internal_models = {
        path.stem.split("__", 1)[1]
        for path in args.scores_dir.glob("internal__*.csv")
    }
    external_models = {
        path.stem.split("__", 1)[1]
        for path in args.scores_dir.glob("external__*.csv")
    }
    models = sorted(internal_models & external_models)
    if not models:
        raise SystemExit(f"No matching internal/external score files in {args.scores_dir}")
    LOGGER.info("Discovered %d models: %s", len(models), ", ".join(models))

    # Build the split once from the main model's scored population when
    # available; model score values are not used.  All other models map onto
    # these patient assignments below.
    split_reference = (PAPER_CHECKPOINT if PAPER_CHECKPOINT in models
                       else models[0])
    reference = _load_scores(
        args.scores_dir / f"internal__{split_reference}.csv", score_col
    )
    patient_split = build_patient_split(reference, args.calibration_fraction, args.seed)
    split_map = patient_split.set_index("patient_id")["split"]
    LOGGER.info(
        "Patient split (reference=%s): %d calibration / %d internal-holdout patients",
        split_reference,
        int((patient_split["split"] == "calibration").sum()),
        int((patient_split["split"] == "internal_holdout").sum()),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    patient_split.to_csv(args.out_dir / "calibration_split.csv", index=False)

    summary_rows, cohort_rows = [], []
    for model_idx, model in enumerate(models, start=1):
        model_start = time.perf_counter()
        LOGGER.info("[%d/%d] %s: loading scores", model_idx, len(models), model)
        internal = _load_scores(args.scores_dir / f"internal__{model}.csv", score_col)
        external = _load_scores(args.scores_dir / f"external__{model}.csv", score_col)
        internal["split"] = internal["patient_id"].astype(str).map(split_map)
        if internal["split"].isna().any():
            unknown = sorted(internal.loc[internal["split"].isna(), "patient_id"].astype(str).unique())
            raise ValueError(f"{model} has internal patients absent from shared split: {unknown[:5]}")

        calibration = internal[internal["split"] == "calibration"].copy()
        internal_holdout = internal[internal["split"] == "internal_holdout"].copy()
        threshold = learn_youden_threshold(
            calibration["gt"].to_numpy(dtype=int),
            calibration[score_col].to_numpy(dtype=float),
        )
        LOGGER.info(
            "[%d/%d] %s: youden threshold=%.4f (calib=%d, internal-holdout=%d, external=%d slides)",
            model_idx, len(models), model, threshold,
            len(calibration), len(internal_holdout), len(external),
        )
        external = external.copy()
        external["split"] = "external"
        heldout_all = pd.concat([internal_holdout, external], ignore_index=True)
        heldout_all["bootstrap_id"] = (
            heldout_all["split"].astype(str) + "::" +
            heldout_all["patient_id"].astype(str)
        )

        eval_sets = {
            "calibration": calibration,
            "internal_holdout": internal_holdout,
            "external": external,
            "heldout_all": heldout_all,
        }
        summary = {
            "model": model,
            "pooling": args.pooling,
            "score": score_col,
            "threshold_source": "20pct_internal_patient_stratified_youden",
            "split_reference": split_reference,
            "calibration_fraction": args.calibration_fraction,
            "split_seed": args.seed,
            "locked_threshold": threshold,
        }
        for name, frame in eval_sets.items():
            point = _split_metrics(frame, score_col, threshold)
            cluster_col = "bootstrap_id" if name == "heldout_all" else "patient_id"
            n_clusters = int(frame[cluster_col].nunique())
            LOGGER.info(
                "[%d/%d] %s: bootstrapping %s (%d resamples, %d clusters, %d slides)",
                model_idx, len(models), model, name, args.n_boot, n_clusters, len(frame),
            )
            boot_start = time.perf_counter()
            last_beat = [boot_start]

            def _heartbeat(done: int, total: int) -> None:
                now = time.perf_counter()
                if now - last_beat[0] >= 15.0:
                    last_beat[0] = now
                    LOGGER.info(
                        "      ... %s %s: %d/%d resamples (%.0f%%)",
                        model, name, done, total, 100.0 * done / total,
                    )

            ci = bootstrap_fixed_threshold(
                frame["gt"].to_numpy(dtype=int),
                frame[score_col].to_numpy(dtype=float),
                frame[cluster_col].astype(str).to_numpy(),
                threshold,
                args.n_boot,
                args.seed,
                progress=_heartbeat,
            )
            LOGGER.info(
                "[%d/%d] %s: %s done in %.1fs (auroc=%.3f, balacc=%.3f)",
                model_idx, len(models), model, name, time.perf_counter() - boot_start,
                point["auroc"], point["balanced_accuracy"],
            )
            summary[f"{name}_n_patients"] = int(frame[cluster_col].nunique())
            summary.update({f"{name}_{key}": value for key, value in point.items()})
            summary.update({f"{name}_{key}": value for key, value in ci.items()})
        summary_rows.append(summary)
        LOGGER.info(
            "[%d/%d] %s complete in %.1fs",
            model_idx, len(models), model, time.perf_counter() - model_start,
        )

        for eval_name, frame in (("internal_holdout", internal_holdout), ("external", external)):
            for group, cohort in frame.groupby("group", sort=True):
                cohort_rows.append({
                    "model": model,
                    "evaluation_set": eval_name,
                    "group": group,
                    "pooling": args.pooling,
                    "locked_threshold": threshold,
                    "n_patients": int(cohort["patient_id"].nunique()),
                    **_split_metrics(cohort, score_col, threshold),
                })

    LOGGER.info(
        "All %d models done in %.1fs; writing outputs to %s",
        len(models), time.perf_counter() - run_start, args.out_dir,
    )
    summary = pd.DataFrame(summary_rows).sort_values(
        "external_balanced_accuracy", ascending=False
    )
    cohorts = pd.DataFrame(cohort_rows).sort_values(["model", "evaluation_set", "group"])
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    cohorts.to_csv(args.out_dir / "per_cohort.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary.where(pd.notna(summary), None).to_dict("records"), indent=2)
    )
    print(summary[[
        "model", "locked_threshold", "internal_holdout_n", "external_n",
        "external_auroc", "external_balanced_accuracy", "external_sensitivity",
        "external_specificity", "external_f1",
    ]].to_string(index=False))
    print(f"\nOutputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
