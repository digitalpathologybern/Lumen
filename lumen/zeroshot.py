"""Shared zero-shot classification primitives.

The prompt-ensemble -> cosine-logit -> softmax pipeline was previously copy-pasted
across the encoder facade (:mod:`lumen.encode`) and the benchmark metrics
(:mod:`lumen.benchmark.metrics`). It lives here once so a change to the
normalization epsilon, the temperature default or the ensembling rule takes
effect everywhere instead of drifting between three copies.

All functions assume :class:`~lumen.models.base.VLMAdapter`-style embeddings:
already L2-normalized ``float32`` rows, so cosine similarity is a plain dot
product.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

# CLIP's default inverse temperature (1 / 0.07). Used when a caller supplies
# neither an explicit temperature nor a model logit scale.
DEFAULT_TEMPERATURE = 1.0 / 0.07

EncodeText = Callable[[list[str]], np.ndarray]


def class_matrix(encode_text: EncodeText,
                 class_prompts: Mapping[str, Sequence[str]]) -> np.ndarray:
    """Ensemble each class's prompts into one L2-normalized row -> ``[C, D]``.

    ``encode_text`` must return already-L2-normalized ``[P, D]`` embeddings for a
    list of prompts (as every :class:`~lumen.models.base.VLMAdapter` does).
    Rows follow ``class_prompts`` insertion order.
    """
    vecs = []
    for prompts in class_prompts.values():
        emb = encode_text(list(prompts))                     # [P, D]
        mean = emb.mean(axis=0)
        vecs.append(mean / (np.linalg.norm(mean) + 1e-8))
    return np.stack(vecs, axis=0).astype(np.float32)


def cosine_logits(image_emb: np.ndarray, class_mat: np.ndarray) -> np.ndarray:
    """Cosine-similarity logits ``[N, C]`` (both inputs L2-normalized)."""
    return image_emb.astype(np.float32) @ class_mat.astype(np.float32).T


def softmax(logits: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Numerically stable row-wise softmax of ``scale * logits``."""
    z = logits.astype(np.float32) * scale
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def sigmoid(logits: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Numerically stable sigmoid of ``scale * logits``."""
    z = logits.astype(np.float32) * scale
    out = np.empty_like(z, dtype=np.float32)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def resolve_scale(logit_scale: float | None = None,
                  temperature: float | None = None) -> float:
    """Pick the softmax scale: explicit ``temperature`` > model ``logit_scale`` > default."""
    if temperature is not None:
        return temperature
    return logit_scale or DEFAULT_TEMPERATURE


def zero_shot_probs(image_emb: np.ndarray,
                    encode_text: EncodeText,
                    class_prompts: Mapping[str, Sequence[str]],
                    *,
                    logit_scale: float | None = None,
                    temperature: float | None = None,
                    class_mat: np.ndarray | None = None) -> tuple[list[str], np.ndarray]:
    """Full pipeline: ensemble prompts, cosine logits, probabilities.

    For multiple classes, probabilities are row-wise softmax values. For a
    single foreground prompt, a softmax would be exactly 1.0 for every patch, so
    the one-class case returns a sigmoid foreground probability instead.
    Returns ``(classes, probs)`` where ``probs`` is ``[N, C]`` ordered by
    ``class_prompts`` insertion order.

    ``class_mat`` may be a precomputed ``[C, D]`` matrix (from :func:`class_matrix`)
    to skip re-running the text encoder when the same class prompts are scored
    against many batches of patches.
    """
    classes = list(class_prompts.keys())
    if class_mat is None:
        class_mat = class_matrix(encode_text, class_prompts)
    logits = cosine_logits(image_emb, class_mat)
    scale = resolve_scale(logit_scale, temperature)
    probs = sigmoid(logits, scale) if logits.shape[1] == 1 else softmax(logits, scale)
    return classes, probs
