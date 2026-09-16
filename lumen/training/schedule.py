"""Learning-rate schedule for alignment training.

:func:`cosine_warmup` returns a linear warm-up followed by cosine decay. Its
``warmup_steps`` and ``total_steps`` are counted in **optimizer steps**, and the
training engine steps it once per optimizer step (see
:mod:`lumen.training.engine`).
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warm-up to the peak LR, then cosine decay to ``min_lr_ratio``.

    Both ``warmup_steps`` and ``total_steps`` are measured in optimizer steps,
    the unit the returned scheduler must be stepped in. The multiplier is applied
    to each param group's configured ``lr`` (the peak), so:

    * step ``0``                     -> ``1 / warmup_steps`` of peak (not zero),
    * step ``warmup_steps - 1``      -> peak,
    * step ``total_steps - 1``       -> ``min_lr_ratio`` of peak.

    Parameters
    ----------
    warmup_steps:
        Optimizer steps spent ramping linearly from ~0 to the peak LR.
    total_steps:
        Total optimizer steps in the run; the cosine phase spans
        ``total_steps - warmup_steps``.
    min_lr_ratio:
        Floor for the cosine decay as a fraction of the peak LR (``0`` decays to
        zero; a small value such as ``0.01`` keeps a nonzero tail).
    """
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
