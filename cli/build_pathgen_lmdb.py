#!/usr/bin/env python3
"""Pack the PathGen patch corpus into an Lumen-format training LMDB.

The QUILT-1M run reads a single LMDB whose values are gzip-compressed pickles of
``{"image": <jpeg bytes>, "tokens": {input_ids, token_type_ids, attention_mask}}``
plus ``__len__``/``__keys__`` metadata. That format is what keeps a 1024-pair
GradCache step fed: one memory-mapped file instead of a million small GPFS reads,
and captions tokenized once at build time instead of in every collate. PathGen
arrives as 1.6M loose JPEGs plus a caption lookup, so it needs the same treatment
before it can be trained with the same recipe.

Two build-time choices, both numerically inert at training time:

* **Images are stored at 224x224.** Virchow2's timm eval transform is
  ``Resize(224, bicubic) -> CenterCrop(224)`` (``crop_pct: 1.0``), so resizing a
  square 672px patch here produces exactly the tensor the model would have seen,
  and the train-time ``Resize`` short-circuits on an already-224 image. The LMDB
  is ~32 GB instead of ~255 GB, which matters when every epoch streams the whole
  corpus off a shared filesystem.
* **Captions are padded to 128 tokens** (QUILT stored 384). PathGen captions run
  65 tokens on average and 87 at most, and the model reads the ``[CLS]`` position
  under an attention mask, so padding length changes nothing but the cost of
  BERT's quadratic attention.

Each record also carries its source WSI in the ``__groups__`` metadata, so
:mod:`lumen.training.data` can hold out whole slides rather than letting
neighbouring patches of one slide straddle the train/val boundary.

Run under the ``lumen`` env::

    python cli/build_pathgen_lmdb.py --workers 16

The build is resumable: records are committed in ordered blocks and
``__progress__`` records how many are durable, so a re-run continues from the
last committed block. ``__len__``/``__keys__`` are written only at the end, so a
partial LMDB is rejected by the loader rather than silently trained on.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BIOMEDBERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Per-worker globals, initialised once per process.
_TOK = None
_OPT: dict = {}


def _init_worker(root: str, image_size: int, quality: int, max_text_len: int) -> None:
    global _TOK, _OPT
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(BIOMEDBERT)
    _OPT = {"root": Path(root), "size": image_size, "quality": quality,
            "max_text_len": max_text_len}


def _encode_one(row: tuple[str, str]):
    """Resize, re-encode and tokenize one pair into a storable blob.

    Returns ``(blob, group_id, error)``; ``blob`` is None when the patch file is
    unreadable, which happens for the handful of patches lost to a full
    filesystem during extraction.
    """
    rel, caption = row
    path = _OPT["root"] / rel
    group = Path(rel).parent.name  # patches/<wsi_id>/<patch>.jpg
    try:
        with Image.open(path) as src:
            image = src.convert("RGB")
        size = _OPT["size"]
        if image.size != (size, size):
            # Matches torchvision Resize(size, BICUBIC) for a square input: the
            # same PIL call with the same filter, so the stored pixels are the
            # ones the training transform would have produced.
            image = image.resize((size, size), Image.BICUBIC)
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=_OPT["quality"])

        enc = _TOK(caption, padding="max_length", truncation=True,
                   max_length=_OPT["max_text_len"])
        tokens = {name: np.asarray(values, dtype=np.int64)
                  for name, values in enc.items()}
        blob = gzip.compress(
            pickle.dumps({"image": buf.getvalue(), "tokens": tokens}), 6)
        return blob, group, None
    except Exception as e:  # noqa: BLE001 - one bad patch must not kill the build
        return None, group, f"{type(e).__name__}: {e}"


def read_lookup(path: Path, limit: int = 0) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        if header is None or header[:2] != ["image_path", "caption"]:
            raise SystemExit(f"{path}: expected an image_path,caption header, got {header}")
        for rec in rd:
            if len(rec) < 2 or not rec[0] or not rec[1].strip():
                continue
            rows.append((rec[0], rec[1]))
            if limit and len(rows) >= limit:
                break
    return rows


def _resume_point(env) -> int:
    with env.begin() as txn:
        raw = txn.get(b"__progress__")
    return int(pickle.loads(raw)) if raw is not None else 0


def build(args: argparse.Namespace) -> None:
    import lmdb

    rows = read_lookup(args.lookup, args.limit)
    print(f"lookup: {len(rows)} pairs from {args.lookup}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(args.out), map_size=args.map_size_gb * (1 << 30),
                    subdir=True, readonly=False, meminit=False, map_async=True)
    start = _resume_point(env)
    if start:
        print(f"resuming after {start} already-committed pairs", flush=True)
    if start >= len(rows):
        print("nothing to do; finalising metadata", flush=True)

    groups: list[str] = []
    if start:
        with env.begin() as txn:
            raw = txn.get(b"__groups_partial__")
        groups = pickle.loads(raw) if raw is not None else []
        if len(groups) > start:
            raise SystemExit(
                f"partial LMDB is inconsistent: __progress__={start} rows "
                f"consumed but {len(groups)} records written; delete "
                f"{args.out} and rebuild"
            )

    # Records written, which trails rows consumed by the unreadable ones, so it
    # is the group list and not __progress__ that says where the keys resume.
    n_written = len(groups)
    n_failed = 0
    # One shared instance per WSI id: pickle memoises by identity, which keeps
    # the group list committed with every block small instead of re-writing
    # 1.6M separate strings each time.
    interned: dict[str, str] = {g: g for g in groups}
    t0 = time.time()
    fail_log = open(args.out.parent / "pathgen_lmdb_failures.log", "a")
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker,
        initargs=(str(args.root), args.image_size, args.jpeg_quality,
                  args.max_text_len),
    ) as ex:
        for block_start in range(start, len(rows), args.block):
            block = rows[block_start:block_start + args.block]
            results = list(ex.map(_encode_one, block, chunksize=32))
            with env.begin(write=True) as txn:
                for (rel, _), (blob, group, err) in zip(block, results):
                    if blob is None:
                        n_failed += 1
                        fail_log.write(f"{rel}\t{err}\n")
                        continue
                    # Keys are assigned as records are written, so skipped rows
                    # leave no holes in the index space.
                    txn.put(str(n_written).encode(), blob)
                    groups.append(interned.setdefault(group, group))
                    n_written += 1
                txn.put(b"__progress__", pickle.dumps(block_start + len(block)))
                txn.put(b"__groups_partial__", pickle.dumps(groups))
            fail_log.flush()
            done = block_start + len(block)
            rate = (done - start) / max(1e-6, time.time() - t0)
            eta = (len(rows) - done) / max(1e-6, rate) / 3600
            print(f"  {done}/{len(rows)} pairs  written={n_written} "
                  f"failed={n_failed}  {rate:.0f}/s  eta {eta:.1f}h", flush=True)

    keys = [str(i).encode() for i in range(n_written)]
    with env.begin(write=True) as txn:
        txn.put(b"__len__", pickle.dumps(n_written))
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__groups__", pickle.dumps(groups))
    env.sync()
    env.close()
    fail_log.close()
    # Completion marker for the self-chaining Slurm window: __len__/__keys__ are
    # only written above, so its presence means the LMDB is trainable.
    (args.out.parent / f"{args.out.name}.COMPLETE").write_text(
        f"records={n_written} unreadable={n_failed} "
        f"image_size={args.image_size} max_text_len={args.max_text_len}\n")
    print(f"done: {n_written} records ({n_failed} unreadable) -> {args.out} "
          f"in {(time.time() - t0) / 3600:.2f}h", flush=True)


def verify(args: argparse.Namespace) -> None:
    """Open the LMDB exactly as training will and report what it sees."""
    from lumen.training.data import QuiltLMDBDataset

    ds = QuiltLMDBDataset(args.out, image_transform=lambda im: im)
    image, tokens = ds[0]
    n_groups = len(set(ds.groups)) if getattr(ds, "groups", None) else 0
    print(f"records         : {len(ds)}")
    print(f"token length    : {ds.token_length}")
    print(f"token fields    : {sorted(tokens)}")
    print(f"first image     : {image.size} {image.mode}")
    print(f"source WSIs     : {n_groups}")
    if image.size != (args.image_size, args.image_size):
        raise SystemExit(
            f"stored image is {image.size}, expected "
            f"{(args.image_size, args.image_size)}: the training transform would "
            "resample instead of short-circuiting"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1] / "assets" / "datasets" / "pathgen"
    ap.add_argument("--root", type=Path, default=root,
                    help="corpus root that image_path entries are relative to")
    ap.add_argument("--lookup", type=Path, default=root / "pathgen_lookup.csv")
    ap.add_argument("--out", type=Path, default=root / "PathGen_dataset.lmdb")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--block", type=int, default=10_000,
                    help="pairs per commit; also the resume granularity")
    ap.add_argument("--image-size", type=int, default=224,
                    help="stored edge length; must match the Virchow2 input size")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--max-text-len", type=int, default=128)
    ap.add_argument("--map-size-gb", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs (smoke test)")
    ap.add_argument("--verify", action="store_true",
                    help="inspect an existing LMDB through the training loader and exit")
    args = ap.parse_args()

    if args.verify:
        verify(args)
        return
    build(args)


if __name__ == "__main__":
    main()
