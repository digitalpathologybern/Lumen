"""Generic image helpers shared across loaders and the server.

Model-specific preprocessing (resize/normalize) lives inside each adapter in
:mod:`lumen.models.loaders`; this module only holds format-level helpers that
are independent of the model.
"""

from __future__ import annotations

import io
from typing import Iterable

from PIL import Image


def ensure_rgb(img: Image.Image) -> Image.Image:
    """Return ``img`` as a 3-channel RGB image."""
    return img if img.mode == "RGB" else img.convert("RGB")


def decode_image(raw: bytes) -> Image.Image:
    """Decode raw image bytes (e.g. a POSTed tile) to an RGB PIL image."""
    return ensure_rgb(Image.open(io.BytesIO(raw)))


def iter_batches(seq, batch_size: int) -> Iterable[list]:
    """Yield ``seq`` in lists of at most ``batch_size`` items."""
    batch: list = []
    for item in seq:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
