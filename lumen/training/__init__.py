"""Clean, reproducible training for the Lumen alignment model.

Reuses the checkpoint-verified architecture in :mod:`lumen.utils.model` and
adds a training layer: a warm-up/cosine schedule stepped
per optimizer step (:mod:`.schedule`), symmetric InfoNCE (:mod:`.loss`), a typed
provenance-stamped config (:mod:`.config`), a QUILT-1M pair loader
(:mod:`.data`), and the loop (:mod:`.engine`).
"""

from lumen.training.config import (
    DataConfig, ModelConfig, OptimConfig, TrainConfig,
)

__all__ = ["TrainConfig", "ModelConfig", "OptimConfig", "DataConfig"]
