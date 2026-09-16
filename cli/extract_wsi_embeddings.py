#!/usr/bin/env python3
"""Cache WSI tile embeddings for internal/external slide-level benchmarks.

This is the WSI analogue of ``cli/extract_embeddings.py``: it tiles each WSI
at the configured resolution, applies the tissue filter, runs the model image
encoder, and stops after writing per-slide projected image embeddings. The
cache is one ``.npz`` per slide with 512d fp16 embeddings.

Examples
--------
    python cli/extract_wsi_embeddings.py --table internal
    python cli/extract_wsi_embeddings.py --table external --out-dir outputs/wsi_embeddings
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.inference import (  # noqa: E402
    apply_inference_overrides,
    load_slide_table,
    load_yaml,
    resolve_table_paths,
    safe_name as _safe_name,
)
from lumen.models.registry import MODELS, PAPER_CHECKPOINT  # noqa: E402
from lumen.wsi.cohorts import exclude_benchmark_groups  # noqa: E402


def _resolve_model_keys(arg: str) -> list[str]:
    """Expand ``'all'`` / comma-list into validated registry model keys."""
    if arg == "all":
        return list(MODELS)
    keys = [k.strip() for k in str(arg).split(",") if k.strip()]
    bad = [k for k in keys if k not in MODELS]
    if bad:
        raise SystemExit(
            f"Unknown model(s) {bad}. Available: {list(MODELS)}"
        )
    return keys


MANIFEST_FIELDS = [
    "table", "slide", "filepath", "group", "gt", "npz_path", "n_tiles",
    "embedding_dim", "bytes", "status", "seconds", "error",
]




def _table_key(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    if "internal" in stem:
        return "internal"
    if "external" in stem:
        return "external"
    return _safe_name(Path(path).stem)


def _select_tables(cfg: dict, table: str, table_paths: list[str] | None) -> list[Path]:
    if table_paths:
        return [Path(p) for p in table_paths]

    paths = resolve_table_paths(cfg)
    if table == "all":
        return paths

    selected = [p for p in paths if table in _table_key(p)]
    if not selected:
        raise SystemExit(f"No {table!r} table found in {cfg.get('tables', [])}")
    return selected


def _append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in MANIFEST_FIELDS})
        f.flush()


def _existing_cache_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as data:
            embeddings = data["embeddings"]
            coords = data["coords"]
            return (
                embeddings.ndim == 2
                and coords.ndim == 2
                and coords.shape[1] == 2
                and coords.shape[0] == embeddings.shape[0]
            )
    except Exception:
        return False


def _atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".npz",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--inference", default="configs/inference.yaml",
                    help="Inference YAML (default: configs/inference.yaml)")
    ap.add_argument("--table", choices=["internal", "external", "all"],
                    default="all",
                    help="Which configured slide-level table(s) to cache")
    ap.add_argument("--tables", nargs="+", default=None, metavar="PATH",
                    help="Explicit table path(s), overriding --table/config")
    ap.add_argument("--out-dir", default="outputs/wsi_embeddings",
                    help="Root for per-slide .npz caches")
    ap.add_argument("--model", default=PAPER_CHECKPOINT,
                    help="Registry model key(s): a key, comma-list, or 'all'. "
                         f"Available: {list(MODELS)}")
    ap.add_argument("--resolution", type=float, default=None,
                    help="Override inference.resolution")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override inference.batch_size")
    ap.add_argument("--tile-size", type=int, default=None,
                    help="Override inference.tile_size")
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug cap per table after loading/filtering")
    ap.add_argument("--group", default=None,
                    help="Optional group filter inside each table")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split each table into N row shards")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="Shard index to run, in [0, --num-shards)")
    ap.add_argument("--device", default=None,
                    help="cuda|cpu (auto if unset)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute slides even when their .npz already exists")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    cfg = apply_inference_overrides(
        load_yaml(args.inference),
        resolution=args.resolution,
    )
    inf_cfg = cfg["inference"]
    if args.batch_size is not None:
        inf_cfg["batch_size"] = args.batch_size
    if args.tile_size is not None:
        inf_cfg["tile_size"] = args.tile_size

    model_keys = _resolve_model_keys(args.model)
    print(f"[cache] models={model_keys}  out-dir={args.out_dir}")

    import torch

    from lumen.models.loaders import load_adapter
    from lumen.wsi.segment_classifier import extract_slide_tile_embeddings

    device = torch.device(args.device) if args.device else None
    table_paths = _select_tables(cfg, args.table, args.tables)

    for model_key in model_keys:
        adapter = load_adapter(MODELS[model_key], device)
        dev = adapter.device

        # The tiler needs a PIL->CPU tensor transform and a batched forward
        # returning a torch tensor. ``adapter.preprocess`` is the former;
        # ``adapter.encode_pixels`` handles the device move + autocast +
        # L2-normalize and returns numpy, so wrap it back into a tensor.
        vis_tfms = adapter.preprocess

        def img_forward(batch, _adapter=adapter):
            return torch.from_numpy(_adapter.encode_pixels(batch))

        out_root = Path(args.out_dir) / model_key
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#' * 60}\n  model={model_key}  root={out_root}\n{'#' * 60}")

        grand_total_tiles = 0
        grand_total_bytes = 0

        for table_path in table_paths:
            key = _table_key(table_path)
            df = exclude_benchmark_groups(load_slide_table(table_path))
            if args.group:
                df = df[df["group"] == args.group].copy()
            if args.limit is not None:
                df = df.head(args.limit).copy()
            total_before_shard = len(df)
            if args.num_shards > 1:
                df = df.iloc[args.shard_index::args.num_shards].copy()

            table_dir = out_root / key
            if args.num_shards > 1:
                manifest_path = (
                    table_dir
                    / f"{key}_manifest_shard{args.shard_index:02d}-of-{args.num_shards:02d}.csv"
                )
            else:
                manifest_path = table_dir / f"{key}_manifest.csv"
            table_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'=' * 60}\n  {key}: {table_path}\n{'=' * 60}")
            shard_msg = (
                f"  shard={args.shard_index}/{args.num_shards}"
                f" from {total_before_shard} slides"
                if args.num_shards > 1 else ""
            )
            print(f"[{key}] slides={len(df)}{shard_msg}  output={table_dir}")

            for idx, (_, row) in enumerate(df.iterrows(), 1):
                fp = Path(str(row["filepath"]))
                slide = _safe_name(row["slide"])
                group = _safe_name(row["group"])
                gt = int(row["gt"])
                npz_path = table_dir / group / f"{slide}.npz"
                meta_path = table_dir / group / f"{slide}.json"

                if not args.overwrite and _existing_cache_ok(npz_path):
                    print(f"  [{idx}/{len(df)}] skip existing {slide}")
                    continue
                if npz_path.exists() and not args.overwrite:
                    print(f"  [{idx}/{len(df)}] redo invalid cache {slide}")

                t0 = time.time()
                if not fp.exists():
                    msg = "file not found"
                    print(f"  [{idx}/{len(df)}] FAIL {slide}: {msg}")
                    _append_manifest(manifest_path, {
                        "table": key, "slide": slide, "filepath": str(fp),
                        "group": group, "gt": gt, "npz_path": str(npz_path),
                        "status": "missing", "error": msg,
                    })
                    continue

                try:
                    coords, feats = extract_slide_tile_embeddings(
                        sid=slide,
                        wsi_path=fp,
                        resolution=inf_cfg["resolution"],
                        vis_tfms=vis_tfms,
                        img_forward=img_forward,
                        device=dev,
                        batch_size=inf_cfg["batch_size"],
                        tile_size=inf_cfg.get("tile_size", 224),
                        tissue_thresh=inf_cfg.get("tissue_thresh", 0.12),
                        purple_thresh=inf_cfg.get("purple_thresh", 0.0),
                        ln_mask=None,
                    )
                    feats16 = feats.astype(np.float16, copy=False)
                    _atomic_save_npz(
                        npz_path,
                        embeddings=feats16,
                        coords=coords.astype(np.int32, copy=False),
                        gt=np.asarray(gt, dtype=np.int16),
                    )
                    meta = {
                        "table": key,
                        "model": model_key,
                        "slide": slide,
                        "filepath": str(fp),
                        "group": group,
                        "gt": gt,
                        "resolution": float(inf_cfg["resolution"]),
                        "tile_size": int(inf_cfg.get("tile_size", 224)),
                        "filter": "tissue",
                        "n_tiles": int(feats16.shape[0]),
                        "embedding_dim": int(feats16.shape[1]) if feats16.ndim == 2 else 0,
                        "dtype": "float16",
                    }
                    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

                    seconds = time.time() - t0
                    bytes_written = npz_path.stat().st_size
                    grand_total_tiles += int(feats16.shape[0])
                    grand_total_bytes += bytes_written
                    _append_manifest(manifest_path, {
                        **meta,
                        "npz_path": str(npz_path),
                        "bytes": bytes_written,
                        "status": "ok",
                        "seconds": f"{seconds:.1f}",
                        "error": "",
                    })
                    print(
                        f"  [{idx}/{len(df)}] ok   {slide:<38} "
                        f"[{group}] tiles={feats16.shape[0]:5d} "
                        f"dim={meta['embedding_dim']:4d} "
                        f"cache={bytes_written / 1024**2:6.1f} MiB "
                        f"({seconds:.0f}s)"
                    )
                except Exception as exc:
                    seconds = time.time() - t0
                    print(f"  [{idx}/{len(df)}] FAIL {slide}: {exc}")
                    _append_manifest(manifest_path, {
                        "table": key, "slide": slide, "filepath": str(fp),
                        "group": group, "gt": gt, "npz_path": str(npz_path),
                        "status": "failed", "seconds": f"{seconds:.1f}",
                        "error": repr(exc),
                    })

        print(
            f"\n[{model_key}] Done. New cache written: {grand_total_tiles:,} tiles, "
            f"{grand_total_bytes / 1024**3:.2f} GiB under {out_root}"
        )

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
