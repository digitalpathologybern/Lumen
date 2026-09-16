#!/usr/bin/env python3
"""Prompt-ensemble metrics per class-name variant, with per-class detail.

Stage B reports the ensemble for the ``canonical`` variant only
(:func:`lumen.benchmark.prompts.canonical_prompts` hardcodes it), and the
per-prompt CSVs keep macro-F1 but not per-class F1. Neither can answer whether a
class is unreachable *because of how it is worded*.

This runs the same six-template ensemble for **every** variant a dataset defines
and adds per-class F1 and the prediction histogram, so a dead class is visible.

Motivating question: is a class unreachable because the representation lacks the
distinction, or because of how the class is worded? Some datasets ship more than
one class-name variant (see
:data:`lumen.benchmark.prompts.CLASS_NAME_VARIANTS`); this tool scores each
variant with the same six-template ensemble so a class killed by wording rather
than by the model is visible in the per-class F1 and the prediction histogram.

Runs on the Stage-A embedding caches plus text encoding.

Examples::

    python cli/analyze_prompt_variant.py --dataset lc25000 --model all
    python cli/analyze_prompt_variant.py --dataset lc25000 --model lumen,conch
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.benchmark import prompts as pb  # noqa: E402
from lumen.benchmark.extract import DEFAULT_OUT, resolve_keys  # noqa: E402
from lumen.benchmark.metrics import (  # noqa: E402
    class_matrix,
    compute_metrics,
    score_logits,
)
from lumen.data.registry import DATASETS  # noqa: E402
from lumen.models.registry import MODELS  # noqa: E402
from lumen.paths import project_root  # noqa: E402

DEFAULT_OUT_DIR = project_root() / "outputs" / "benchmark" / "prompt_variant"


def variant_prompts(dataset_key: str, variant: str) -> dict[str, list[str]]:
    """``canonical_prompts`` for an arbitrary variant: all templates x one variant."""
    names = pb.class_names_for(dataset_key, variant)
    frames = [f for fs in pb.TEMPLATES.values() for f in fs]
    return {name: [f.format(name) for f in frames] for name in names}


def per_class(logits: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    from sklearn.metrics import f1_score

    preds = logits.argmax(axis=1)
    f1 = f1_score(labels, preds, average=None, labels=list(range(n_classes)),
                  zero_division=0)
    return {
        "f1": [float(x) for x in f1],
        "n_pred": np.bincount(preds, minlength=n_classes).tolist(),
        "n_true": np.bincount(labels, minlength=n_classes).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lc25000", help="dataset key or comma-list")
    ap.add_argument("--model", default="all", help="model key, comma-list, or 'all'")
    ap.add_argument("--emb-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    from lumen.models.loaders import load_adapter

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    emb_dir, out_dir = Path(args.emb_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_keys = resolve_keys(args.model, MODELS)
    dataset_keys = resolve_keys(args.dataset, DATASETS)
    rows: list[dict] = []

    for mkey in model_keys:
        todo = [d for d in dataset_keys if (emb_dir / d / f"{mkey}.npz").exists()]
        if not todo:
            print(f"[skip] {mkey}: no embedding cache", flush=True)
            continue

        print(f"\n=== {mkey} ===", flush=True)
        adapter = load_adapter(MODELS[mkey], device)

        for dkey in todo:
            spec = DATASETS[dkey]
            d = np.load(emb_dir / dkey / f"{mkey}.npz")
            image_emb = d["embeddings"].astype(np.float32)
            labels = d["labels"].astype(np.int64)
            n_classes = spec.num_classes

            for variant in pb.variant_names(dkey):
                pset = variant_prompts(dkey, variant)
                logits = score_logits(image_emb, class_matrix(adapter, pset))
                m = compute_metrics(logits, labels)
                pc = per_class(logits, labels, n_classes)

                row = {
                    "dataset": dkey, "model": mkey, "variant": variant,
                    "n": int(len(labels)),
                    "balanced_acc": m["balanced_acc"],
                    "macro_f1": m["macro_f1"],
                    "macro_auroc": m["macro_auroc"],
                    "accuracy": m["accuracy"],
                    "dead_classes": sum(1 for x in pc["f1"] if x == 0.0),
                }
                for i, name in enumerate(spec.class_names):
                    row[f"f1__{name}"] = pc["f1"][i]
                    row[f"npred__{name}"] = pc["n_pred"][i]
                rows.append(row)

                dead = [spec.class_names[i] for i, x in enumerate(pc["f1"]) if x == 0.0]
                print(f"  {dkey:10s} {variant:10s} "
                      f"balacc={m['balanced_acc']:.3f} "
                      f"macroF1={m['macro_f1']:.3f} "
                      f"AUROC={m['macro_auroc']:.3f} "
                      f"pred={pc['n_pred']} "
                      f"dead={dead if dead else 'none'}", flush=True)

        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        sys.exit("ERROR: nothing evaluated (no caches matched)")

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out_csv = out_dir / f"{'_'.join(sorted(set(r['dataset'] for r in rows)))}_variants.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
