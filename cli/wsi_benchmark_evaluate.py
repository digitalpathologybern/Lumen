#!/usr/bin/env python3
"""Slide-level zero-shot benchmark (WSI Stage B).

Thin CLI over :func:`lumen.wsi.slide_benchmark.evaluate`. Runs on the cached
per-slide tile embeddings from ``cli/extract_wsi_embeddings.py``. No slide is
re-tiled. For each (model, table) it applies three class vocabularies through the
six standard templates, computes tile-level positive-class probabilities, and
pools those probabilities over the slide. The primary score is max pooling.

Examples::

    python cli/wsi_benchmark_evaluate.py --model all --table all
    python cli/wsi_benchmark_evaluate.py --model conch --table external --n-boot 2000
    python cli/wsi_benchmark_evaluate.py --model lumen,conch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.models.registry import MODELS  # noqa: E402
from lumen.wsi.slide_benchmark import DEFAULT_EMB_DIR, DEFAULT_OUT, evaluate  # noqa: E402
from lumen.wsi.prompts import VOCABULARIES  # noqa: E402

TABLES = ("internal", "external")


def _resolve(arg: str, valid) -> list[str]:
    if arg == "all":
        return list(valid)
    keys = [k.strip() for k in str(arg).split(",") if k.strip()]
    bad = [k for k in keys if k not in valid]
    if bad:
        raise SystemExit(f"Unknown key(s) {bad}. Available: {list(valid)}")
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all",
                    help="registry model key, comma-list, or 'all'")
    ap.add_argument("--table", default="all",
                    help="'internal', 'external', comma-list, or 'all'")
    ap.add_argument("--emb-dir", default=str(DEFAULT_EMB_DIR),
                    help="root of cached WSI tile embeddings")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples for the ensemble CI (0 to skip)")
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    ap.add_argument("--groups", default=None,
                    help="comma-list of cohort/group names to restrict scoring to "
                         "(e.g. melanoma_histai_b1,melanoma_histai_b2); write to a "
                         "scratch --out-dir so the full outputs are not overwritten")
    ap.add_argument("--vocabulary", default="shared",
                    choices=sorted(VOCABULARIES),
                    help="class-name bank. 'shared' is the reported protocol and "
                         "names metastatic carcinoma. 'melanoma_agnostic' replaces "
                         "that with histology-neutral wording, for the melanoma "
                         "cohorts, which do not contain a carcinoma "
                         "(default: shared)")
    args = ap.parse_args()

    model_keys = _resolve(args.model, MODELS)
    table_keys = _resolve(args.table, TABLES)
    only_groups = ({g.strip() for g in args.groups.split(",")}
                   if args.groups else None)

    evaluate(model_keys, table_keys, emb_dir=args.emb_dir, out_dir=args.out_dir,
             device=args.device, n_boot=args.n_boot, only_groups=only_groups,
             vocabulary=args.vocabulary)


if __name__ == "__main__":
    main()
