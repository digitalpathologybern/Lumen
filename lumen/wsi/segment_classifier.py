#!/usr/bin/env python3
"""
segment_classifier.py: slide-level VLM evaluation (whole-slide tiling).

Public surface:
    BACKGROUND_LABEL, ORDERED_CLASSES, CATEGORY_DICT    label vocabulary
    classify_slide_whole(...)                           tile a WSI and score every passing tile

Tile filtering: HSV saturation + optional purple-fraction filter when no
LN mask is available; otherwise the (boolean) LN mask from MetAssist Stage 1
is the sole authority.
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import openslide
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.exceptions import UndefinedMetricWarning
from torch import amp

from lumen.utils.annotations import prepare_read_from_slide

# Some slides (e.g. HISTAI generic-tiff externals) store huge 4096x4096 internal
# tiles. A single decoded tile (~64 MiB RGBA) exceeds OpenSlide's default 32 MiB
# cache, so every small read_region re-decodes the whole tile with zero reuse
# (~50x slower). Give OpenSlide a per-slide cache large enough to hold a working
# set of big tiles. Size is configurable via LUMEN_OPENSLIDE_CACHE_MB.
_OPENSLIDE_CACHE_MB = int(os.environ.get("LUMEN_OPENSLIDE_CACHE_MB", "1024"))


def _open_slide_cached(wsi_path) -> "openslide.OpenSlide":
    """Open a slide with an enlarged tile cache when the API is available."""
    slide = openslide.OpenSlide(str(wsi_path))
    if _OPENSLIDE_CACHE_MB > 0 and hasattr(openslide, "OpenSlideCache"):
        try:
            slide.set_cache(openslide.OpenSlideCache(_OPENSLIDE_CACHE_MB * 1024 * 1024))
        except Exception:  # pragma: no cover - openslide too old / unsupported
            pass
    return slide


warnings.filterwarnings("ignore", message="A single label was found.*")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings(
    "ignore",
    message="networkx backend defined more than once.*",
    category=RuntimeWarning,
)


# ── LABELS ───────────────────────────────────────────────────────────────────

EXCLUDE_CLASSES: Set[str] = {"Training region", "Ink", "???", "Slide edge", "Folds"}
ORDERED_CLASSES: List[str] = [
    "Background", "Lymph node", "Tumor deposits", "Primary tumor",
    "Primary tissue", "Vessels", "Metastasis", "Necrosis",
    "Connective tissue", "Fat tissue", "Mucin",
]

RAW_COLORMAP: Dict[str, Tuple[int, int, int]] = {
    "Background":        (125, 125, 125),
    "Lymph node":        (229, 100,  84),
    "Tumor deposits":    (212, 185,  60),
    "Primary tumor":     ( 54,  90, 113),
    "Primary tissue":    (  0, 124, 169),
    "Ink":               ( 11,  72, 205),
    "Vessels":           (106,  29, 125),
    "Metastasis":        (117, 173,  81),
    "Necrosis":          ( 50,  50,  50),
    "Connective tissue": (250,  71, 102),
    "Folds":             ( 73, 103,  40),
    "Fat tissue":        (255, 255, 153),
    "Mucin":             (220, 220, 220),
    "Slide edge":        ( 48, 213, 200),
    "Training region":   (  0,   0,   0),
    "???":               (255, 255, 255),
}

TUMOR_CLASSES: Set[str] = {"Tumor deposits", "Primary tumor", "Metastasis"}

HEALTHY_LABEL    = "Healthy Lymphnode"
TUMOR_LABEL      = "Tumor"
BACKGROUND_LABEL = "Background"

COLORMAP       = {cls: RAW_COLORMAP[cls] for cls in ORDERED_CLASSES}
CATEGORY_DICT  = {cls: i for i, cls in enumerate(ORDERED_CLASSES)}
COLORMAP_FLOAT = {i: tuple(c / 255 for c in COLORMAP[cls]) for cls, i in CATEGORY_DICT.items()}
ID_LABELS      = {i: cls for cls, i in CATEGORY_DICT.items() if cls != BACKGROUND_LABEL}


# ── HELPERS ──────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _safe_float(v) -> "float | None":
    """Coerce to float; return None for nan/inf/non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f or f == float("inf") or f == float("-inf")) else f
    except (TypeError, ValueError):
        return None


def _is_tissue(
    tile_rgb: np.ndarray,
    sat_thresh: float = 0.12,
    purple_thresh: float = 0.0,
) -> bool:
    """
    Heuristic tissue filter.

    sat_thresh:    min mean HSV saturation; removes background, fat, dark artifacts.
    purple_thresh: min fraction of tissue pixels with hue in 130-220/255
                   (hematoxylin nuclei). 0.0 disables the purple gate.
    """
    hsv = np.array(Image.fromarray(tile_rgb).convert("HSV"), dtype=np.float32) / 255.0
    sat = hsv[:, :, 1]
    if sat.mean() < sat_thresh:
        return False
    if purple_thresh > 0.0:
        hue         = hsv[:, :, 0] * 255.0
        tissue_mask = sat >= sat_thresh
        if tissue_mask.sum() == 0:
            return False
        purple_frac = float(((hue >= 130) & (hue <= 220))[tissue_mask].mean())
        if purple_frac < purple_thresh:
            return False
    return True


def _quiet_call(fn, *args, **kwargs):
    """Run a noisy transform/model helper while suppressing stdout."""
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        return fn(*args, **kwargs)


# ── TILE EMBEDDING EXTRACTION ────────────────────────────────────────────────

def extract_slide_tile_embeddings(
    sid: str,
    wsi_path: Path,
    resolution: float,
    vis_tfms,
    img_forward,
    device: torch.device,
    batch_size: int = 32,
    tile_size: int = 224,
    tissue_thresh: float = 0.12,
    purple_thresh: float = 0.0,
    ln_mask: "np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tile an entire slide at *resolution* MPP and return kept tile embeddings.

    Tile selection:
    - If ``ln_mask`` is supplied (a boolean array at *resolution* MPP from any
      external region mask), a tile is kept whenever it overlaps any masked
      pixel and the HSV/purple filter is bypassed. (No mask is produced
      internally; callers currently pass ``ln_mask=None``.)
    - Otherwise the HSV saturation + optional purple-fraction filter is used.

    Returns ``(coords, feats)`` where coords is ``int32[N,2]`` in slide-level
    tile coordinates at the selected read resolution, and feats is normalized
    projected VLM image embeddings as ``float32[N,D]``.
    """
    slide = _open_slide_cached(wsi_path)
    lvl, ds, _, orig_dim, origin = prepare_read_from_slide(slide, resolution, wsi_path.suffix)
    ds = int(ds)

    ts, H, W = tile_size, *orig_dim
    imgs:   list = []
    coords: list = []
    use_ln_mask = ln_mask is not None
    lh, lw = (ln_mask.shape[:2] if use_ln_mask else (None, None))

    for y in range(0, H - ts + 1, ts):
        for x in range(0, W - ts + 1, ts):
            tile = slide.read_region(
                (origin[0] + x * ds, origin[1] + y * ds), lvl, (ts, ts)
            ).convert("RGB")
            tile_np = np.array(tile, dtype=np.uint8)

            if use_ln_mask:
                y2 = min(y + ts, lh)
                x2 = min(x + ts, lw)
                region = ln_mask[y:y2, x:x2]
                if region.size == 0 or not region.any():
                    continue
            else:
                if not _is_tissue(tile_np, sat_thresh=tissue_thresh, purple_thresh=purple_thresh):
                    continue

            t = _quiet_call(vis_tfms, tile)
            if isinstance(t, torch.Tensor) and t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            imgs.append(t)
            coords.append((x, y))

    slide.close()

    if not imgs:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 0), dtype=np.float32),
        )

    feats_list = []
    for i in range(0, len(imgs), batch_size):
        batch = torch.stack(imgs[i:i + batch_size]).to(device)
        with torch.no_grad(), amp.autocast("cuda", enabled=device.type == "cuda"):
            feats_list.append(_quiet_call(img_forward, batch))
    feats = torch.cat(feats_list, dim=0)

    if feats.dim() == 4:
        feats = F.adaptive_avg_pool2d(feats, 1).view(feats.size(0), -1)
    elif feats.dim() == 3:
        feats = feats.mean(1)
    feats = F.normalize(feats.float(), dim=1)
    return (
        np.asarray(coords, dtype=np.int32),
        feats.detach().cpu().numpy().astype(np.float32, copy=False),
    )


# ── MAIN ENTRY POINT ─────────────────────────────────────────────────────────

def classify_slide_whole(
    sid: str,
    wsi_path: Path,
    resolution: float,
    vis_tfms,
    img_forward,
    text_feats: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
    tile_size: int = 224,
    tissue_thresh: float = 0.12,
    purple_thresh: float = 0.0,
    ln_mask: "np.ndarray | None" = None,
):
    """
    Tile an entire slide at *resolution* MPP and classify each tile.

    Returns ``(tiles_info, result)`` where *result* contains ``max_prob_pos``,
    ``avg_prob_pos``, and ``n_tiles``.
    """
    coords_np, feats_np = extract_slide_tile_embeddings(
        sid=sid,
        wsi_path=wsi_path,
        resolution=resolution,
        vis_tfms=vis_tfms,
        img_forward=img_forward,
        device=device,
        batch_size=batch_size,
        tile_size=tile_size,
        tissue_thresh=tissue_thresh,
        purple_thresh=purple_thresh,
        ln_mask=ln_mask,
    )

    if feats_np.size == 0:
        return [], {"max_prob_pos": None, "avg_prob_pos": None, "n_tiles": 0}

    feats = torch.from_numpy(feats_np).to(device)
    tfs   = F.normalize(text_feats.to(device).float(), dim=1)
    probs = F.softmax(feats @ tfs.T, dim=1)

    probs_np  = probs.detach().cpu().numpy()
    probs_pos = probs_np[:, 1] if probs_np.shape[1] > 1 else np.zeros(len(imgs))
    preds     = probs_np.argmax(1)

    tiles_info = [
        {"x": int(x), "y": int(y), "pred": int(p), "prob_pos": float(pp)}
        for (x, y), p, pp in zip(coords_np, preds, probs_pos)
    ]
    result = {
        "max_prob_pos": _safe_float(probs_pos.max()),
        "avg_prob_pos": _safe_float(probs_pos.mean()),
        "n_tiles":      int(len(coords_np)),
    }
    return tiles_info, result
