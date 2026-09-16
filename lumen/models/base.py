"""The uniform adapter interface every VLM is wrapped in.

Downstream code (encoding, benchmark, server) only ever sees
:class:`VLMAdapter`, so it never needs to know whether a model came from
transformers, open_clip, the conch package, or the Lumen finetune.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class VLMAdapter:
    """Wraps encode_image / encode_text closures over a loaded model.

    Both methods return L2-normalized float32 embeddings in the model's shared
    image/text space, so cosine similarity is a plain dot product.
    """

    def __init__(self, name: str, device: torch.device,
                 preprocess: Callable[[Image.Image], torch.Tensor],
                 encode_pixels_fn: Callable[[torch.Tensor], torch.Tensor],
                 text_fn: Callable[[List[str]], torch.Tensor],
                 logit_scale: float | None = None) -> None:
        self.name = name
        self.device = device
        # ``preprocess`` maps one PIL image to a CPU [C,H,W] tensor. It holds no
        # reference to the (GPU) model, so it is safe to run inside DataLoader
        # worker processes. That's what lets image decode + resize/normalize run
        # in parallel across CPUs and overlap with the GPU forward pass, instead
        # of the old serial "decode-all → stack → forward" loop that left the GPU
        # idle most of the time.
        self.preprocess = preprocess
        # ``encode_pixels_fn`` takes a batched [N,C,H,W] tensor already on
        # ``device`` and returns the (un-normalized) image features.
        self._encode_pixels_fn = encode_pixels_fn
        self._text_fn = text_fn
        self.logit_scale = logit_scale
        # Forward passes run under autocast on CUDA (matches wsi/segment_classifier's
        # amp.autocast use). Tensor-core matmuls in fp16 roughly double throughput
        # on the ViT/CLIP backbones here, and results are cast back to fp32 in
        # _to_numpy before the L2-normalize so downstream cosine similarities are
        # unaffected.
        self._autocast = device.type == "cuda"

    @staticmethod
    def _to_numpy(feats: torch.Tensor) -> np.ndarray:
        feats = F.normalize(feats.float(), dim=-1)
        return feats.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_pixels(self, pixel_values: torch.Tensor) -> np.ndarray:
        """Encode an already-preprocessed [N,C,H,W] batch (the fast path).

        Callers that batch through a DataLoader hand the collated, pinned CPU
        tensor straight here; the ``non_blocking`` copy overlaps with the CPU
        preparing the next batch.
        """
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self._autocast):
            feats = self._encode_pixels_fn(pixel_values)
        return self._to_numpy(feats)

    @torch.inference_mode()
    def encode_image(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images (server / ad-hoc path). Preprocesses on
        the calling thread, then defers to :meth:`encode_pixels`."""
        pixel_values = torch.stack([self.preprocess(im) for im in images])
        return self.encode_pixels(pixel_values)

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self._autocast):
            feats = self._text_fn(list(texts))
        return self._to_numpy(feats)
