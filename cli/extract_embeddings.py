#!/usr/bin/env python3
"""Stage A of the zero-shot benchmark: cache image embeddings.

Thin CLI over :func:`lumen.benchmark.extract.extract`.

Examples::

    python cli/extract_embeddings.py --model all --dataset all
    python cli/extract_embeddings.py --model conch --dataset pcam --limit 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.benchmark.extract import DEFAULT_OUT, extract, resolve_keys  # noqa: E402
from lumen.data.registry import DATASETS  # noqa: E402
from lumen.models.registry import MODELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all",
                    help="model key, comma-list, or 'all'")
    ap.add_argument("--dataset", default="all",
                    help="dataset key, comma-list, or 'all'")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="CPU workers for decode/preprocess "
                         "(default: SLURM_CPUS_PER_TASK-1, else min(8, ncpu))")
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap images per dataset (debugging)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    try:
        model_keys = resolve_keys(args.model, MODELS)
        dataset_keys = resolve_keys(args.dataset, DATASETS)
    except KeyError as exc:
        sys.exit(f"ERROR: {exc}")

    extract(model_keys, dataset_keys, out_dir=args.out_dir,
            batch_size=args.batch_size, device=args.device,
            limit=args.limit, overwrite=args.overwrite,
            num_workers=args.num_workers)


if __name__ == "__main__":
    main()
