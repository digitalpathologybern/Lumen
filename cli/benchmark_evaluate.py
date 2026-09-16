#!/usr/bin/env python3
"""Stage B of the zero-shot benchmark: score cached embeddings against a prompt bank.

Thin CLI over :func:`lumen.benchmark.evaluate.evaluate`. Requires Stage A
caches (``cli/extract_embeddings.py``) to exist.

Examples::

    python cli/benchmark_evaluate.py --model all --dataset all
    python cli/benchmark_evaluate.py --model conch --dataset pcam --n-boot 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.benchmark.evaluate import (  # noqa: E402
    DEFAULT_EVAL_OUT, PROMPT_STYLES, evaluate)
from lumen.benchmark.extract import DEFAULT_OUT, resolve_keys  # noqa: E402
from lumen.benchmark.metrics import SELECTABLE_PRIMARY_METRICS  # noqa: E402
from lumen.data.registry import DATASETS  # noqa: E402
from lumen.models.registry import MODELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all", help="model key, comma-list, or 'all'")
    ap.add_argument("--dataset", default="all", help="dataset key, comma-list, or 'all'")
    ap.add_argument("--emb-dir", default=str(DEFAULT_OUT),
                    help="Stage A embedding cache dir")
    ap.add_argument("--out-dir", default=str(DEFAULT_EVAL_OUT))
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples for the ensemble CI (0 to skip)")
    ap.add_argument("--prompt-style", default="ours", choices=PROMPT_STYLES,
                    help="ensemble prompt bank: 'ours' is the six-template "
                         "comparison protocol, 'rich' is the primary paper "
                         "22-template bank over every class-name variant, and "
                         "'rich_canonical' is those 22 templates over the "
                         "canonical name only (uniform 22 per class)")
    ap.add_argument("--primary", default="balanced_acc",
                    choices=SELECTABLE_PRIMARY_METRICS)
    args = ap.parse_args()

    try:
        model_keys = resolve_keys(args.model, MODELS)
        dataset_keys = resolve_keys(args.dataset, DATASETS)
    except KeyError as exc:
        sys.exit(f"ERROR: {exc}")

    evaluate(model_keys, dataset_keys, emb_dir=args.emb_dir, out_dir=args.out_dir,
             device=args.device, n_boot=args.n_boot, primary=args.primary,
             prompt_style=args.prompt_style)


if __name__ == "__main__":
    main()
