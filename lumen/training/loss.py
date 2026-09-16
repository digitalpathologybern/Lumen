"""Contrastive objective for image--text alignment.

The model's contrastive head returns a scaled cosine-similarity matrix
``logits[i, j]`` between image ``i`` and text ``j`` in a batch; the matched
pairs lie on the diagonal. Training minimises the symmetric InfoNCE (CLIP) loss:
the mean of the image->text and text->image cross-entropies against the
identity targets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE over a square ``[B, B]`` similarity matrix.

    ``logits`` must already be temperature-scaled (the contrastive head applies
    the learned scale). Rows are images, columns are texts, and row/column ``k``
    are the matched pair. Returns a scalar loss.
    """
    if logits.ndim != 2 or logits.size(0) != logits.size(1):
        raise ValueError(f"expected a square [B, B] logits matrix, got {tuple(logits.shape)}")
    target = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, target)
    loss_t2i = F.cross_entropy(logits.t(), target)
    return 0.5 * (loss_i2t + loss_t2i)


@torch.no_grad()
def contrastive_accuracy(logits: torch.Tensor) -> float:
    """Top-1 retrieval accuracy on the batch (diagnostic, not the objective).

    The fraction of images whose highest-scoring text is the matched one, a
    cheap in-batch signal to watch during training.
    """
    if logits.numel() == 0:
        return 0.0
    target = torch.arange(logits.size(0), device=logits.device)
    return (logits.argmax(dim=1) == target).float().mean().item()
