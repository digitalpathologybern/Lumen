"""High-level encoding facade, the main entry point most users want.

Switching models is a one-argument change::

    from lumen.encode import Encoder
    enc = Encoder("conch")                 # or "plip", "clip_l14", ...
    #   the reported Lumen run: Encoder(PAPER_CHECKPOINT)
    img_emb  = enc.encode_images(pil_list)          # [N, D] L2-normalized
    scores   = enc.zero_shot(pil_list, {             # {class: [prompt, ...]}
        "benign":  ["benign tissue", "a benign H&E image"],
        "tumor":   ["tumor tissue",  "carcinoma"],
    })                                               # [N, n_classes] probabilities
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from lumen.models.loaders import load_adapter
from lumen.models.registry import MODELS
from lumen.zeroshot import zero_shot_probs


class Encoder:
    """Thin, model-agnostic wrapper around a :class:`~lumen.models.base.VLMAdapter`."""

    def __init__(self, model_key: str, device: torch.device | None = None) -> None:
        if model_key not in MODELS:
            raise KeyError(f"Unknown model {model_key!r}. Available: {list(MODELS)}")
        self.model_key = model_key
        self.adapter = load_adapter(MODELS[model_key], device)

    @property
    def device(self):
        return self.adapter.device

    def encode_images(self, images: Sequence[Image.Image],
                      batch_size: int = 128) -> np.ndarray:
        out = [self.adapter.encode_image(images[i:i + batch_size])
               for i in range(0, len(images), batch_size)]
        return np.concatenate(out, axis=0) if out else np.empty((0, 0), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self.adapter.encode_text(list(texts))

    def zero_shot(self, images: Sequence[Image.Image],
                  class_prompts: Mapping[str, Sequence[str]],
                  batch_size: int = 128,
                  temperature: float | None = None) -> np.ndarray:
        """Zero-shot classify ``images`` against ``{class: [prompts]}``.

        Each class's prompt embeddings are averaged (prompt ensembling) and
        L2-renormalized, then image-text cosine similarities are softmaxed into
        per-class probabilities. Returns ``[N, n_classes]`` ordered by
        ``class_prompts`` insertion order.
        """
        img = self.encode_images(images, batch_size)              # [N, D]
        _, probs = zero_shot_probs(
            img, self.adapter.encode_text, class_prompts,
            logit_scale=self.adapter.logit_scale, temperature=temperature)
        return probs

    def class_names(self, class_prompts: Mapping[str, Sequence[str]]) -> list[str]:
        return list(class_prompts.keys())
