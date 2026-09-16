"""QUILT-1M image--text pair datasets and loaders.

The preferred backend reads the original Lumen LMDB, whose values are
gzip-compressed pickles containing JPEG bytes and fixed-length BioMedBERT token
arrays.  A CSV + extracted-image fallback is retained for smoke tests and new
datasets. Images are preprocessed with the *same* Virchow2 transform used at
inference, so the trained checkpoint sees exactly the distribution the
benchmark stack feeds it.

The train/val split is a deterministic function of ``seed`` and is drawn once,
so early stopping and model selection use a fixed held-out set across runs.
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from lumen.training.config import DataConfig

LOGGER = logging.getLogger("lumen.training.data")


class QuiltLMDBDataset(Dataset):
    """Original Lumen LMDB records: JPEG bytes plus token arrays.

    The environment is opened lazily in each DataLoader worker. Sharing an
    already-open LMDB environment across forked workers is unsafe and can also
    exhaust reader slots on long, persistent-worker runs.
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        image_transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        try:
            import lmdb
        except ImportError as exc:  # pragma: no cover - environment error path
            raise RuntimeError(
                "LMDB training requested, but the 'lmdb' package is not installed"
            ) from exc

        self.lmdb_path = str(Path(lmdb_path))
        self.tf = image_transform
        self._env = None
        env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False,
            meminit=False, max_readers=256,
        )
        try:
            with env.begin() as txn:
                raw_len = txn.get(b"__len__")
                raw_keys = txn.get(b"__keys__")
                raw_groups = txn.get(b"__groups__")
                if raw_len is None or raw_keys is None:
                    raise RuntimeError(
                        f"{self.lmdb_path} is missing __len__/__keys__ metadata"
                    )
                self.length = int(pickle.loads(raw_len))
                self.keys = pickle.loads(raw_keys)
                first_raw = txn.get(self.keys[0]) if self.keys else None
        finally:
            env.close()
        if self.length != len(self.keys):
            raise RuntimeError(
                f"LMDB metadata mismatch: __len__={self.length}, "
                f"but __keys__ has {len(self.keys)} entries"
            )
        if first_raw is None:
            raise RuntimeError(f"LMDB has no readable sample records: {self.lmdb_path}")
        # Optional per-record group id (PathGen stores the source WSI). When
        # present the split is drawn over groups, so patches cut from one slide
        # cannot straddle train and val. QUILT has no such key and splits flat.
        if raw_groups is not None:
            self.groups = list(pickle.loads(raw_groups))
            if len(self.groups) != self.length:
                raise RuntimeError(
                    f"LMDB metadata mismatch: __len__={self.length}, but "
                    f"__groups__ has {len(self.groups)} entries"
                )
        else:
            self.groups = None
        first_sample = pickle.loads(gzip.decompress(first_raw))
        first_tokens = first_sample.get("tokens", {})
        self.token_length = len(first_tokens.get("input_ids", []))
        if not self.token_length:
            raise RuntimeError(
                f"LMDB sample {self.keys[0]!r} has no stored input_ids"
            )

    def __len__(self) -> int:
        return self.length

    def _open(self):
        if self._env is None:
            import lmdb
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=False,
                meminit=False, max_readers=256,
            )
        return self._env

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __getitem__(self, i: int):
        key = self.keys[i]
        with self._open().begin() as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(f"LMDB record missing for key {key!r}")
        sample = pickle.loads(gzip.decompress(raw))
        if "image" not in sample or "tokens" not in sample:
            raise KeyError(f"LMDB record {key!r} lacks image/tokens fields")

        with Image.open(io.BytesIO(sample["image"])) as source:
            image = source.convert("RGB")
        image = self.tf(image)
        tokens = {
            name: torch.as_tensor(values, dtype=torch.long)
            for name, values in sample["tokens"].items()
        }
        return image, tokens


class QuiltPairDataset(Dataset):
    """Image--text pairs from a QUILT-1M lookup table.

    Rows whose image file is missing are dropped at construction, so every index
    is loadable; the number retained is reported in :attr:`n_dropped`.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        image_root: str | Path,
        image_transform: Callable[[Image.Image], torch.Tensor],
        image_col: str,
        caption_col: str,
    ) -> None:
        self.image_root = Path(image_root)
        self.tf = image_transform
        self.image_col = image_col
        self.caption_col = caption_col
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows.iloc[i]
        path = self.image_root / str(r[self.image_col])
        img = Image.open(path).convert("RGB")
        return self.tf(img), str(r[self.caption_col])


def _existing(rows: pd.DataFrame, image_root: Path, image_col: str) -> pd.DataFrame:
    """Keep only rows whose image file is present on disk.

    Scans in chunks and logs progress: on GPFS this is ~1M ``exists`` calls and
    otherwise runs silently for many minutes.
    """
    paths = rows[image_col].astype(str).to_numpy()
    root = str(image_root)
    present = np.empty(len(paths), dtype=bool)
    t0 = time.time()
    for i, p in enumerate(paths):
        present[i] = os.path.exists(os.path.join(root, p))
        if (i + 1) % 100_000 == 0 or (i + 1) == len(paths):
            LOGGER.info("  image scan %d/%d  present=%d  (%.0fs)",
                        i + 1, len(paths), int(present[: i + 1].sum()), time.time() - t0)
    return rows[present]


def load_lookup(cfg: DataConfig) -> pd.DataFrame:
    if not cfg.lookup_csv:
        raise ValueError("DataConfig.lookup_csv is empty; point it at quilt_1M_lookup.csv")
    df = pd.read_csv(cfg.lookup_csv)
    for col in (cfg.image_col, cfg.caption_col):
        if col not in df.columns:
            raise KeyError(
                f"column {col!r} not in lookup ({list(df.columns)[:12]}...); "
                "set DataConfig.image_col / caption_col"
            )
    df = df[[cfg.image_col, cfg.caption_col]].dropna()
    df = df[df[cfg.caption_col].str.strip().astype(bool)]
    df = df.reset_index(drop=True)
    if cfg.limit_rows and cfg.limit_rows > 0:
        df = df.iloc[: cfg.limit_rows].reset_index(drop=True)
    return df


def split_rows(df: pd.DataFrame, val_fraction: float, seed: int):
    """Deterministic train/val partition of the lookup rows."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_val = int(round(len(df) * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return df.iloc[train_idx], df.iloc[val_idx]


def split_indices(length: int, val_fraction: float, seed: int):
    """Deterministic train/val indices for an indexed storage backend."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(length)
    n_val = int(round(length * val_fraction))
    return perm[n_val:].tolist(), perm[:n_val].tolist()


def split_indices_grouped(groups: Sequence[str], val_fraction: float, seed: int):
    """Deterministic train/val indices that never split a group.

    Whole groups (for PathGen, whole WSIs) go to validation until the row target
    is met. A flat split would put ~229 patches of a held-out patch's own slide
    into training, so val loss would measure memorised slides rather than
    generalisation, and model selection would follow that.
    """
    order: list[str] = []
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        if g not in members:
            members[g] = []
            order.append(g)
        members[g].append(i)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(order))
    target = val_fraction * len(groups)
    val_idx: list[int] = []
    n_val_groups = 0
    for j in perm:
        if len(val_idx) >= target:
            break
        val_idx.extend(members[order[j]])
        n_val_groups += 1
    val_set = set(val_idx)
    train_idx = [i for i in range(len(groups)) if i not in val_set]
    return train_idx, sorted(val_idx), n_val_groups


def make_collate(tokenizer, max_text_len: int) -> Callable:
    """Collate ``(image, caption)`` samples into a training batch."""

    def collate(batch: Sequence):
        images = torch.stack([b[0] for b in batch])
        captions = [b[1] for b in batch]
        tokens = tokenizer(
            captions, padding=True, truncation=True,
            max_length=max_text_len, return_tensors="pt",
        )
        return images, dict(tokens)

    return collate


def make_token_collate() -> Callable:
    """Collate the pre-tokenized records stored in the original LMDB."""

    def collate(batch: Sequence):
        images = torch.stack([b[0] for b in batch])
        token_rows = [b[1] for b in batch]
        names = token_rows[0].keys()
        tokens = {
            name: torch.stack([row[name] for row in token_rows])
            for name in names
        }
        return images, tokens

    return collate


def build_loaders(
    cfg: DataConfig,
    image_transform: Callable[[Image.Image], torch.Tensor],
    tokenizer,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_text_len: int,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build train/val loaders and a small stats dict for logging."""
    if cfg.lmdb_path:
        t0 = time.time()
        dataset = QuiltLMDBDataset(cfg.lmdb_path, image_transform)
        n_stored = len(dataset)
        n_used = min(n_stored, cfg.limit_rows) if cfg.limit_rows > 0 else n_stored
        if dataset.groups is not None:
            train_idx, val_idx, n_val_groups = split_indices_grouped(
                dataset.groups[:n_used], cfg.val_fraction, seed)
            split_kind = "grouped"
        else:
            train_idx, val_idx = split_indices(n_used, cfg.val_fraction, seed)
            n_val_groups = 0
            split_kind = "flat"
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        stats = {
            "backend": "lmdb",
            "lmdb_path": cfg.lmdb_path,
            "rows_stored": n_stored,
            "rows_used": n_used,
            "stored_token_length": dataset.token_length,
            "split": split_kind,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_val_groups": n_val_groups,
        }
        LOGGER.info("LMDB loaded: %d/%d rows used (%.1fs)",
                    n_used, n_stored, time.time() - t0)

        def lmdb_loader(ds: Dataset, shuffle: bool) -> DataLoader:
            return DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers, collate_fn=make_token_collate(),
                pin_memory=True, drop_last=shuffle,
                persistent_workers=num_workers > 0,
            )

        return lmdb_loader(train_ds, True), lmdb_loader(val_ds, False), stats

    t0 = time.time()
    df = load_lookup(cfg)
    image_root = Path(cfg.image_root)
    n_before = len(df)
    LOGGER.info("lookup loaded: %d rows (%.0fs)", n_before, time.time() - t0)

    # The present-set is identical across ranks/windows, so cache it once and let
    # resuming windows skip the multi-minute GPFS scan. Only cache full runs.
    cache = None
    if not cfg.limit_rows:
        cache = image_root.parent / f"_present_cache_{n_before}.pkl"
    if cache is not None and cache.exists():
        df = pd.read_pickle(cache)
        LOGGER.info("present-set from cache: %d rows (%s)", len(df), cache.name)
    else:
        LOGGER.info("scanning %d image paths under %s ...", n_before, image_root)
        df = _existing(df, image_root, cfg.image_col)
        if cache is not None:
            df.to_pickle(cache)
            LOGGER.info("cached present-set -> %s", cache.name)
    stats = {"rows_in_lookup": n_before, "rows_with_image": len(df),
             "rows_missing_image": n_before - len(df)}
    if len(df) == 0:
        raise RuntimeError(
            f"no images found under {image_root}; check DataConfig.image_root "
            "and that QUILT frames were extracted"
        )

    train_df, val_df = split_rows(df, cfg.val_fraction, seed)
    stats["n_train"], stats["n_val"] = len(train_df), len(val_df)
    collate = make_collate(tokenizer, max_text_len)

    def loader(rows: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = QuiltPairDataset(rows, image_root, image_transform,
                              cfg.image_col, cfg.caption_col)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=collate,
            pin_memory=True, drop_last=shuffle, persistent_workers=num_workers > 0,
        )

    return loader(train_df, True), loader(val_df, False), stats
