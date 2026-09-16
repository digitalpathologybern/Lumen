"""Stage A of the zero-shot benchmark: cache image embeddings.

Encodes every evaluation image once per (model, dataset) and writes the
L2-normalized image embeddings to disk. Stage B (:mod:`lumen.benchmark.evaluate`)
then sweeps prompts entirely on these caches, so the image encoder is never
re-run per prompt.

Output layout (under ``out_dir``, default ``outputs/benchmark/embeddings``)::

    <dataset>/<model>.npz    embeddings[N,D] fp16, labels[N] int16
    <dataset>/<model>.json   metadata (class_names, dim, n, logit_scale, ...)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from lumen.data.loaders import load_eval_set
from lumen.data.registry import DATASETS
from lumen.models.registry import MODELS
from lumen.paths import project_root

DEFAULT_OUT = project_root() / "outputs" / "benchmark" / "embeddings"


def resolve_keys(arg: str, registry: dict) -> list[str]:
    """Expand ``'all'`` / comma-list into validated registry keys."""
    if arg == "all":
        return list(registry)
    keys = [k.strip() for k in arg.split(",") if k.strip()]
    bad = [k for k in keys if k not in registry]
    if bad:
        raise KeyError(f"unknown keys {bad}. Available: {list(registry)}")
    return keys


class _EncodeDataset:
    """Adapts an :class:`EvalDataset` to the torch ``Dataset`` protocol, applying
    the model's ``preprocess`` (decode is already inside ``get_image``).

    Runs entirely on CPU with no reference to the GPU model, so DataLoader
    workers can decode + resize + normalize in parallel.
    """

    def __init__(self, eval_ds, preprocess, n):
        self._eval_ds = eval_ds
        self._preprocess = preprocess
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._preprocess(self._eval_ds.get_image(i))


def _extract_one(adapter, ds, batch_size, limit, log_prefix, num_workers):
    from torch.utils.data import DataLoader

    n = len(ds) if limit is None else min(limit, len(ds))
    loader = DataLoader(
        _EncodeDataset(ds, adapter.preprocess, n),
        batch_size=batch_size,
        shuffle=False,                       # keep row order → labels stay aligned
        num_workers=num_workers,
        pin_memory=(adapter.device.type == "cuda"),
        prefetch_factor=(4 if num_workers > 0 else None),
        persistent_workers=False,
    )
    embeds = []
    t0 = time.time()
    done = 0
    for px in loader:
        embeds.append(adapter.encode_pixels(px))
        done += px.shape[0]
        if done <= batch_size or done == n or (done // batch_size) % 20 == 0:
            rate = done / max(1e-9, time.time() - t0)
            print(f"    {log_prefix} {done}/{n}  ({rate:.0f} img/s)", flush=True)
    return np.concatenate(embeds, axis=0), ds.labels[:n]


def _default_num_workers() -> int:
    """CPU workers for the input pipeline. Honour the SLURM allocation when set
    (``--cpus-per-task``), else fall back to a modest share of the machine."""
    import os
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return max(1, int(slurm) - 1)  # leave one core for the main/GPU-feed thread
    return min(8, (os.cpu_count() or 2))


def extract(model_keys, dataset_keys, out_dir=DEFAULT_OUT, batch_size=128,
            device=None, limit=None, overwrite=False, num_workers=None):
    """Run Stage A for the given models × datasets. Returns list of written .npz paths."""
    import torch
    from lumen.models.loaders import load_adapter

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if num_workers is None:
        num_workers = _default_num_workers()
    out_root = Path(out_dir)
    written: list[Path] = []
    eval_sets: dict = {}

    print(f"Device: {device}")
    print(f"Workers:  {num_workers}  (batch_size={batch_size})")
    print(f"Models:   {model_keys}")
    print(f"Datasets: {dataset_keys}\n")

    for mkey in model_keys:
        mspec = MODELS[mkey]
        pending = []
        for dkey in dataset_keys:
            npz = out_root / dkey / f"{mkey}.npz"
            if npz.exists() and not overwrite:
                print(f"[skip] {mkey} x {dkey} -> exists ({npz})")
            else:
                pending.append(dkey)
        if not pending:
            continue

        print(f"\n=== loading model: {mkey} ({mspec.family}) ===", flush=True)
        t0 = time.time()
        adapter = load_adapter(mspec, device)
        print(f"    loaded in {time.time() - t0:.1f}s", flush=True)

        for dkey in pending:
            dspec = DATASETS[dkey]
            if dkey not in eval_sets:
                eval_sets[dkey] = load_eval_set(dspec)
            ds = eval_sets[dkey]

            print(f"  -> {mkey} x {dkey}  (n={len(ds)})", flush=True)
            embeds, labels = _extract_one(adapter, ds, batch_size, limit,
                                          f"{mkey}/{dkey}", num_workers)

            out_sub = out_root / dkey
            out_sub.mkdir(parents=True, exist_ok=True)
            npz_path = out_sub / f"{mkey}.npz"
            np.savez(npz_path,
                     embeddings=embeds.astype(np.float16),
                     labels=labels.astype(np.int16))
            meta = {
                "model": mkey, "family": mspec.family, "dataset": dkey,
                "split": dspec.split, "class_names": ds.class_names,
                "n": int(len(labels)), "dim": int(embeds.shape[1]),
                "logit_scale": adapter.logit_scale,
                "label_dist": np.bincount(
                    labels, minlength=dspec.num_classes).tolist(),
                "created": datetime.now().isoformat(timespec="seconds"),
            }
            (out_sub / f"{mkey}.json").write_text(json.dumps(meta, indent=2))
            written.append(npz_path)
            print(f"     saved {embeds.shape} -> {npz_path}", flush=True)

        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nStage A done.")
    return written
