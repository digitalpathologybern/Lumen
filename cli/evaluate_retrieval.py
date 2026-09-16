#!/usr/bin/env python3
"""Evaluate text-image retrieval from cached paired embeddings.

A general retrieval scorer over any ``(image_embeddings, text_embeddings,
manifest)`` cache (see :mod:`lumen.retrieval` for the reusable logic). Used
for the ARCH cross-modal retrieval benchmark; pass ``--embeddings``,
``--manifest`` and ``--out-dir`` explicitly.

Default input/output layout::

    outputs/retrieval/<sample>/
      embeddings.npz
      manifest.csv
      evaluation/
        metrics.json
        metrics.csv
        pair_scores.csv
        retrieval_summary.md
        topk_image_to_text.csv
        topk_text_to_image.csv
        cosine_histogram.png
        rank_cdf.png

For WSI-level retrieval, pass ``--unit wsi``. The CLI mean-pools all patch
image embeddings and all caption embeddings by ``file_id`` into one normalized
image/text vector pair per WSI, then performs the same bidirectional ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.paths import project_root  # noqa: E402
from lumen.retrieval import (  # noqa: E402
    aggregate_embeddings,
    compute_bidirectional_ranks,
    json_ready,
    load_manifest,
    modality_gap,
    random_nonmatch_cosine,
    summarize_scores,
    write_metrics_csv,
    write_pair_scores,
    write_plots,
    write_summary_markdown,
    write_topk_examples,
)


DEFAULT_ROOT = project_root() / "outputs" / "retrieval"
DEFAULT_EMBEDDINGS = DEFAULT_ROOT / "embeddings.npz"
DEFAULT_MANIFEST = DEFAULT_ROOT / "manifest.csv"
DEFAULT_OUT = DEFAULT_ROOT / "evaluation"
DEFAULT_WSI_OUT = DEFAULT_ROOT / "evaluation_wsi"


def parse_recall_k(value: str) -> tuple[int, ...]:
    try:
        out = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--recall-k must be a comma-list of integers") from exc
    if not out or any(k <= 0 for k in out):
        raise argparse.ArgumentTypeError("--recall-k must contain positive integers")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS),
                    help="Path to embeddings.npz")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="Path to manifest.csv with captions and WSI IDs")
    ap.add_argument("--out-dir", default=None,
                    help="Directory for retrieval outputs")
    ap.add_argument("--model-name", default=None,
                    help="Name to record in metrics (default: metadata/model)")
    ap.add_argument("--unit", default="patch", choices=["patch", "wsi"],
                    help="Evaluate row-level patch pairs or pooled WSI/file_id pairs")
    ap.add_argument("--aggregate-field", default="file_id",
                    help="Manifest field used to pool rows when --unit wsi")
    ap.add_argument("--aggregation", default="mean", choices=["mean"],
                    help="Pooling method for --unit wsi")
    ap.add_argument("--chunk-size", type=int, default=512,
                    help="Query chunk size for all-vs-all ranking")
    ap.add_argument("--recall-k", type=parse_recall_k, default=(1, 5, 10),
                    help="Comma-separated recall cutoffs")
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"],
                    help="Use torch+CUDA when available, otherwise NumPy")
    ap.add_argument("--device", default=None,
                    help="Torch device for --backend torch/auto, e.g. cuda or cpu")
    ap.add_argument("--group-field", default="file_id",
                    help="Manifest field for same-group retrieval positives")
    ap.add_argument("--no-group-metrics", action="store_true",
                    help="Skip same-group retrieval metrics")
    ap.add_argument("--nonmatch-samples", type=int, default=100_000,
                    help="Random non-matching image/text pairs for cosine baseline")
    ap.add_argument("--examples", type=int, default=128,
                    help="Number of query examples to save top-k retrievals for")
    ap.add_argument("--top-k-examples", type=int, default=5,
                    help="Top retrieved items per saved query")
    ap.add_argument("--seed", type=int, default=20260708,
                    help="Seed for nonmatch sampling and qualitative examples")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="Debug/smoke-test on the first N pairs")
    ap.add_argument("--skip-plots", action="store_true",
                    help="Do not write PNG diagnostic plots")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite an existing metrics.json")
    return ap.parse_args()


def _slice_or_none(array: np.ndarray | None, n: int) -> np.ndarray | None:
    if array is None:
        return None
    return array[:n]


def _rows_from_npz(data: dict[str, np.ndarray], n: int) -> list[dict]:
    coords = _slice_or_none(data.get("coords"), n)
    file_ids = _slice_or_none(data.get("file_ids"), n)
    wsi_ids = _slice_or_none(data.get("wsi_ids"), n)
    rows = []
    for idx in range(n):
        row = {"pair_index": str(idx)}
        if coords is not None:
            row["x"] = str(int(coords[idx, 0]))
            row["y"] = str(int(coords[idx, 1]))
        if file_ids is not None:
            row["file_id"] = str(file_ids[idx])
        if wsi_ids is not None:
            row["wsi_id"] = str(wsi_ids[idx])
        rows.append(row)
    return rows


def _load_metadata(embeddings_path: Path) -> dict:
    path = embeddings_path.parent / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        raise SystemExit(f"Missing embeddings file: {embeddings_path}")

    with np.load(embeddings_path, allow_pickle=False) as npz:
        image = npz["image_embeddings"]
        text = npz["text_embeddings"]
        if image.shape != text.shape:
            raise SystemExit(f"image/text embedding shapes differ: {image.shape} vs {text.shape}")
        n = int(image.shape[0])
        if args.max_pairs is not None:
            if args.max_pairs <= 1:
                raise SystemExit("--max-pairs must be greater than 1")
            n = min(n, int(args.max_pairs))
        image = np.asarray(image[:n], dtype=np.float32)
        text = np.asarray(text[:n], dtype=np.float32)
        npz_meta = {key: np.asarray(npz[key]) for key in npz.files if key not in {"image_embeddings", "text_embeddings"}}

    manifest_path = Path(args.manifest) if args.manifest else None
    try:
        rows = load_manifest(manifest_path, n)
    except FileNotFoundError:
        rows = _rows_from_npz(npz_meta, n)

    if rows and set(rows[0]) == {"pair_index"}:
        rows = _rows_from_npz(npz_meta, n)

    metadata = _load_metadata(embeddings_path)
    return image, text, rows, metadata


def _patch_count_summary(rows: list[dict]) -> dict[str, float]:
    values = [int(row["n_patches"]) for row in rows if str(row.get("n_patches", "")).isdigit()]
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def _write_wsi_embedding_cache(
    out_dir: Path,
    image: np.ndarray,
    text: np.ndarray,
    rows: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "wsi_embeddings.npz",
        image_embeddings=image.astype(np.float16, copy=False),
        text_embeddings=text.astype(np.float16, copy=False),
        group_ids=np.asarray([row.get("group_id", "") for row in rows], dtype=str),
        file_ids=np.asarray([row.get("file_id", "") for row in rows], dtype=str),
        wsi_ids=np.asarray([row.get("wsi_id", "") for row in rows], dtype=str),
        n_patches=np.asarray([int(row.get("n_patches", 0)) for row in rows], dtype=np.int32),
    )
    fields = ["pair_index", "group_id", "file_id", "wsi_id", "n_patches", "caption"]
    with (out_dir / "wsi_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = DEFAULT_WSI_OUT if args.unit == "wsi" else DEFAULT_OUT
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise SystemExit(f"{metrics_path} exists; pass --overwrite to recompute")

    image, text, rows, source_metadata = _load_inputs(args)
    n_patch_pairs = int(image.shape[0])
    if args.unit == "wsi":
        image, text, rows = aggregate_embeddings(
            image,
            text,
            rows,
            field=args.aggregate_field,
            method=args.aggregation,
        )
        _write_wsi_embedding_cache(out_dir, image, text, rows)

    n_pairs, dim = image.shape
    if n_pairs <= 1:
        raise SystemExit("Need at least two pairs for retrieval evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or source_metadata.get("model") or "unknown"
    group_ids = None
    if args.unit == "patch" and not args.no_group_metrics:
        if args.group_field not in rows[0]:
            print(f"[warn] group field {args.group_field!r} not found; skipping same-group metrics", flush=True)
        else:
            group_ids = [str(row.get(args.group_field, "")) for row in rows]

    # Datasets that attach several texts to one image (e.g. ARCH-OPEN QA) store
    # the image physically duplicated across rows. Keying the image candidate
    # pool on image_path collapses those copies so text->image ranks against
    # unique images. Unique-image datasets (PathGen patches, pooled WSIs) have
    # no repeats and fall back to identity ranking.
    image_ids = None
    if args.unit == "patch":
        candidate_ids = [str(row.get("image_path", "")) for row in rows]
        if any(candidate_ids):
            image_ids = candidate_ids

    print(
        f"[inputs] model={model_name} unit={args.unit} pairs={n_pairs:,} dim={dim} "
        f"chunks={args.chunk_size} recall_k={args.recall_k}",
        flush=True,
    )
    ranks = compute_bidirectional_ranks(
        image_embeddings=image,
        text_embeddings=text,
        chunk_size=args.chunk_size,
        recall_k=args.recall_k,
        backend=args.backend,
        device=args.device,
        group_ids=group_ids,
        image_ids=image_ids,
    )

    nonmatch_n = min(int(args.nonmatch_samples), n_pairs * (n_pairs - 1))
    nonmatch = random_nonmatch_cosine(image, text, nonmatch_n, seed=args.seed)
    pair_cosine = ranks["paired_cosine"]
    unique_file_ids = len({row.get("file_id", "") for row in rows if row.get("file_id", "")})
    unique_wsi_ids = len({row.get("wsi_id", "") for row in rows if row.get("wsi_id", "")})

    metrics = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "retrieval_unit": args.unit,
        "n_patch_pairs": n_patch_pairs,
        "n_pairs": int(n_pairs),
        "embedding_dim": int(dim),
        "embeddings_npz": str(Path(args.embeddings).resolve()),
        "manifest_csv": str(Path(args.manifest).resolve()) if args.manifest else None,
        "source_num_patches_requested": source_metadata.get("num_patches_requested"),
        "source_dataset": source_metadata.get("dataset"),
        "source_failures": len(source_metadata.get("failures", [])) if isinstance(source_metadata.get("failures"), list) else None,
        "unique_file_ids": int(unique_file_ids),
        "unique_wsi_ids": int(unique_wsi_ids),
        "aggregate_field": args.aggregate_field if args.unit == "wsi" else None,
        "aggregation": args.aggregation if args.unit == "wsi" else None,
        "patches_per_wsi": _patch_count_summary(rows) if args.unit == "wsi" else {},
        "recall_k": list(args.recall_k),
        "chunk_size": int(args.chunk_size),
        "backend": ranks["backend"],
        "n_image_candidates": ranks.get("n_image_candidates"),
        "n_text_candidates": ranks.get("n_text_candidates"),
        "candidate_pool_deduplicated": ranks.get("candidate_pool_deduplicated"),
        "group_field": args.group_field if args.unit == "patch" and not args.no_group_metrics else None,
        "i2t": ranks["i2t"],
        "t2i": ranks["t2i"],
        "paired_cosine": ranks["paired_cosine_summary"],
        "random_nonmatch_cosine": summarize_scores(nonmatch),
        "matched_minus_nonmatch_mean": float(pair_cosine.mean() - nonmatch.mean()),
        "modality_gap": modality_gap(image, text),
        "_paired_cosine_values": pair_cosine,
        "_random_nonmatch_values": nonmatch,
        "_i2t_ranks": ranks["i2t_ranks"],
        "_t2i_ranks": ranks["t2i_ranks"],
    }
    if "i2t_same_group" in ranks:
        metrics["i2t_same_group"] = ranks["i2t_same_group"]
        metrics["t2i_same_group"] = ranks["t2i_same_group"]

    metrics_json = json_ready(metrics)
    metrics_path.write_text(json.dumps(metrics_json, indent=2) + "\n", encoding="utf-8")
    write_metrics_csv(out_dir / "metrics.csv", metrics_json)
    write_pair_scores(
        out_dir / "pair_scores.csv",
        rows,
        paired_cosine=pair_cosine,
        i2t_ranks=ranks["i2t_ranks"],
        t2i_ranks=ranks["t2i_ranks"],
        i2t_group_ranks=ranks.get("i2t_group_ranks"),
        t2i_group_ranks=ranks.get("t2i_group_ranks"),
    )
    write_topk_examples(
        out_dir,
        image,
        text,
        rows,
        i2t_ranks=ranks["i2t_ranks"],
        t2i_ranks=ranks["t2i_ranks"],
        n_examples=args.examples,
        top_k=args.top_k_examples,
        seed=args.seed,
    )
    write_summary_markdown(out_dir / "retrieval_summary.md", metrics_json)
    if not args.skip_plots:
        write_plots(out_dir, metrics)

    print(f"[done] metrics -> {metrics_path}", flush=True)
    print(
        "[done] "
        f"i2t R@1={metrics_json['i2t']['R@1']:.4f} R@5={metrics_json['i2t']['R@5']:.4f} "
        f"t2i R@1={metrics_json['t2i']['R@1']:.4f} R@5={metrics_json['t2i']['R@5']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
