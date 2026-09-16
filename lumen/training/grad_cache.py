"""Exact large-batch contrastive gradients using microbatch replay.

The first pass encodes every microbatch without an autograd graph and computes
one InfoNCE loss over the concatenated projections. Gradients with respect to
those projections are cached. Each microbatch is then replayed with autograd and
the cached projection gradients as its upstream signal. This is the GradCache
algorithm: peak encoder activation memory follows the microbatch size, while
the contrastive negative pool follows the logical batch size.

RNG state is captured before every cache forward and restored for its replay so
dropout (including PEFT LoRA dropout) uses identical masks in both passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from lumen.training.loss import clip_contrastive_loss, contrastive_accuracy


@dataclass
class RNGSnapshot:
    cpu: torch.Tensor
    cuda: torch.Tensor | None


@dataclass
class CacheResult:
    loss: float
    accuracy: float
    pairs: int


def to_device(batch, device: torch.device):
    """Move one ``(images, tokens)`` batch to ``device``.

    Shared with the plain training loop in :mod:`lumen.training.engine`,
    which had a byte-equivalent copy differing only in whether the pair arrived
    as one argument or two. Non-blocking copies overlap with the CPU preparing
    the next batch, so both paths must keep that.
    """
    images, tokens = batch
    images = images.to(device, non_blocking=True)
    tokens = {name: value.to(device, non_blocking=True)
              for name, value in tokens.items()}
    return images, tokens


def _capture_rng(device: torch.device) -> RNGSnapshot:
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return RNGSnapshot(torch.get_rng_state(), cuda)


def _restore_rng(state: RNGSnapshot, device: torch.device) -> None:
    torch.set_rng_state(state.cpu)
    if state.cuda is not None:
        torch.cuda.set_rng_state(state.cuda, device)


def _project(model, images, tokens, device: torch.device, amp: bool,
             dtype: torch.dtype = torch.float16):
    with torch.autocast(device_type=device.type, dtype=dtype,
                        enabled=amp and device.type == "cuda"):
        out = model(images, tokens)
    try:
        return out["img_proj"], out["txt_proj"]
    except KeyError as exc:
        raise KeyError(
            "gradient caching requires model.forward() to return img_proj and txt_proj"
        ) from exc


def _global_loss(model, img_proj: torch.Tensor, txt_proj: torch.Tensor):
    # Compute the relatively small global similarity matrix in fp32. Encoder
    # forwards still use AMP; fp32 here improves softmax stability at B=1024.
    with torch.autocast(device_type=img_proj.device.type, enabled=False):
        img_norm = F.normalize(img_proj.float(), dim=-1)
        txt_norm = F.normalize(txt_proj.float(), dim=-1)
        logits = model.contrastive_head(img_norm, txt_norm)
        loss = clip_contrastive_loss(logits)
    return loss, logits


@torch.no_grad()
def cached_contrastive_forward(
    model,
    microbatches: Sequence,
    device: torch.device,
    amp: bool,
    dtype: torch.dtype = torch.float16,
) -> CacheResult:
    """Evaluate one logical batch without retaining encoder activations."""
    img_parts, txt_parts = [], []
    for batch in microbatches:
        images, tokens = to_device(batch, device)
        img_proj, txt_proj = _project(model, images, tokens, device, amp, dtype)
        img_parts.append(img_proj)
        txt_parts.append(txt_proj)
    img_all = torch.cat(img_parts, dim=0)
    txt_all = torch.cat(txt_parts, dim=0)
    loss, logits = _global_loss(model, img_all, txt_all)
    return CacheResult(loss.item(), contrastive_accuracy(logits), len(img_all))


def cached_contrastive_backward(
    model,
    microbatches: Sequence,
    device: torch.device,
    amp: bool,
    scaler,
    dtype: torch.dtype = torch.float16,
) -> CacheResult:
    """Backpropagate exact logical-batch InfoNCE through microbatch replay.

    ``optimizer.zero_grad()`` must be called before this function. The global
    loss is scaled once; its scaled projection gradients are used directly as
    replay signals, so the replay surrogate must *not* be scaled again.
    """
    if not microbatches:
        raise ValueError("gradient-cache step received no microbatches")

    rng_states: list[RNGSnapshot] = []
    sizes: list[int] = []
    img_parts, txt_parts = [], []
    with torch.no_grad():
        for batch in microbatches:
            images, tokens = to_device(batch, device)
            rng_states.append(_capture_rng(device))
            img_proj, txt_proj = _project(model, images, tokens, device, amp, dtype)
            sizes.append(images.size(0))
            img_parts.append(img_proj.detach())
            txt_parts.append(txt_proj.detach())

    img_leaf = torch.cat(img_parts, dim=0).requires_grad_(True)
    txt_leaf = torch.cat(txt_parts, dim=0).requires_grad_(True)
    del img_parts, txt_parts

    loss, logits = _global_loss(model, img_leaf, txt_leaf)
    loss_value = loss.item()
    accuracy = contrastive_accuracy(logits)
    scaler.scale(loss).backward()
    if img_leaf.grad is None or txt_leaf.grad is None:
        raise RuntimeError("global contrastive loss did not produce projection gradients")
    img_grad = img_leaf.grad.detach()
    txt_grad = txt_leaf.grad.detach()
    pairs = img_leaf.size(0)
    del loss, logits, img_leaf, txt_leaf

    offset = 0
    for batch, state, size in zip(microbatches, rng_states, sizes):
        images, tokens = to_device(batch, device)
        _restore_rng(state, device)
        img_proj, txt_proj = _project(model, images, tokens, device, amp, dtype)
        # The cached gradients already include GradScaler's scale. Accumulate
        # the dot product in fp32, then backpropagate without scaler.scale().
        surrogate = (
            (img_proj.float() * img_grad[offset:offset + size].float()).sum()
            + (txt_proj.float() * txt_grad[offset:offset + size].float()).sum()
        )
        surrogate.backward()
        offset += size

    return CacheResult(loss_value, accuracy, pairs)
