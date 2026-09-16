"""Typed training configuration and run provenance.

One :class:`TrainConfig` fully specifies a run. It is dumped to JSON alongside
the checkpoint so that a released model carries the exact settings that produced
it, which is the reproducibility gap the paper needs to close. The model
sub-config mirrors the fields consumed by :class:`lumen.utils.model.Model`, so
a checkpoint written here loads back with ``strict=True`` through the existing
inference stack.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Architecture, matching the recovered Lumen checkpoint config exactly."""

    vision_backbone: str = "Virchow2"
    text_backbone: str = "biomedBERT"
    proj_dim: int = 512
    img_comp: str = "direct"       # 2560 -> 512 linear + LayerNorm
    txt_comp: str = "direct"       # 768  -> 512 linear + LayerNorm
    img_hidden: int = 5            # unused for "direct"; kept for cfg parity
    txt_hidden: int = 3            # unused for "direct"; kept for cfg parity
    n_blocks: int = 3
    span: int = 3
    decay: float = 0.5
    lora: bool = True
    lora_mode: str = "both_encoders"
    #: Unfreeze both pretrained backbones. The backbone loaders freeze every
    #: tensor at load time, so this is the only switch that makes them
    #: trainable; ``lora=False`` alone yields the projection-only configuration
    #: of the adaptation ablation, not a full fine-tune.
    full_finetune: bool = False
    r: int = 4
    lora_dropout: float = 0.1


@dataclass
class OptimConfig:
    """AdamW and the warm-up/cosine schedule.

    ``peak_lr`` is the top of the warm-up ramp; ``warmup_steps`` and
    ``total_steps`` are in optimizer steps and are stepped per step (see
    :mod:`lumen.training.schedule`). Set ``peak_lr`` and the warm-up for the
    run you want.
    """

    peak_lr: float = 5e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    warmup_steps: int = 300
    min_lr_ratio: float = 0.0
    grad_clip_norm: float = 1.0

    # --- full fine-tuning only; both inert while the backbones are frozen ----
    #: Learning rate for pretrained backbone weights. ``None`` keeps every
    #: trainable tensor on ``peak_lr``, which is what the LoRA runs do: there the
    #: only trainable backbone tensors are freshly initialised adapters. Under
    #: full fine-tuning the backbones carry pretrained weights and the heads do
    #: not, so one shared rate cannot serve both -- the rate that trains a random
    #: projection destroys a pretrained ViT.
    backbone_lr: float | None = None
    #: Layer-wise learning-rate decay across encoder depth. ``1.0`` disables it.
    #: With ``d`` < 1 the rate is scaled by ``d ** (n_layers - layer_index)``, so
    #: early layers (generic features) move least and late layers (task-specific)
    #: move most.
    llrd: float = 1.0


@dataclass
class DataConfig:
    """QUILT-1M image--text pairs."""

    lmdb_path: str = ""            # prebuilt QUILT LMDB; takes precedence over CSV/files
    lookup_csv: str = ""           # official quilt_1M_lookup.csv
    image_root: str = ""           # directory of extracted QUILT frames
    caption_col: str = "caption"
    image_col: str = "image_path"
    val_fraction: float = 0.02     # held-out split for early stopping / model selection
    max_text_len: int = 128        # CSV backend only; LMDB uses its stored tokens
    image_size: int = 224
    limit_rows: int = 0            # 0 = use all; >0 caps lookup rows for smoke tests


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)

    epochs: int = 20
    steps_per_epoch: int = 0        # 0 = a full pass; >0 caps the epoch at N optimizer
                                    # steps so a corpus whose pass exceeds the 6h
                                    # preemptable wall still checkpoints inside a window
    batch_size: int = 256
    grad_accum: int = 1
    gradient_cache: bool = False    # true: grad_accum microbatches share one InfoNCE loss
    num_workers: int = 8
    amp: bool = True
    #: Autocast dtype. ``float16`` needs loss scaling and overflows readily on a
    #: 632M ViT under a contrastive loss; ``bfloat16`` has fp32's exponent range,
    #: needs no scaler, and is the stable choice for full fine-tuning on A100 or
    #: newer. Ignored when ``amp`` is false.
    amp_dtype: str = "float16"
    #: Recompute encoder activations in the backward pass instead of storing
    #: them. Roughly halves activation memory for a large multiplier on time; it
    #: is what makes a full fine-tune of Virchow2 fit beside its optimizer state.
    grad_checkpointing: bool = False
    seed: int = 0
    #: Write ``checkpoint_last.pt`` every N optimizer steps as well as at each
    #: epoch end. 0 keeps epoch-end-only. On a preemptable window this bounds
    #: what a kill discards: with a ~3h epoch and a 6h wall, epoch-end-only
    #: throws away most of every window.
    save_every_steps: int = 0
    early_stop_patience: int = 3
    min_epochs: int = 0             # early stopping is disabled before this epoch count
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "lumen-training"
    out_dir: str = "outputs/training/lumen_reproduction"
    run_name: str = "lumen"
    log_every: int = 50

    # Stamped at save time; do not set by hand.
    git_sha: str = ""
    created_utc: str = ""

    def stamp(self) -> "TrainConfig":
        """Record git SHA and timestamp for provenance."""
        self.created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.git_sha = _git_sha()
        return self

    # --- (de)serialisation ---------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        raw = json.loads(Path(path).read_text())
        return cls(
            model=ModelConfig(**raw.pop("model", {})),
            optim=OptimConfig(**raw.pop("optim", {})),
            data=DataConfig(**raw.pop("data", {})),
            **raw,
        )

    def model_cfg_json(self) -> dict[str, Any]:
        """The ``checkpoint_best_model_cfg.json`` payload the loader expects."""
        m = self.model
        return {
            "architecture": "LoRA",
            "vision_backbone": m.vision_backbone,
            "text_backbone": m.text_backbone,
            "proj_dim": m.proj_dim,
            "img_hidden": m.img_hidden,
            "txt_hidden": m.txt_hidden,
            "img_comp": m.img_comp,
            "txt_comp": m.txt_comp,
            "n_blocks": m.n_blocks,
            "span": m.span,
            "decay": m.decay,
            "gated": False,
            "attention": False,
            "pre_gating": False,
            "post_gating": False,
            "use_residuals": False,
            "lora": m.lora,
            "lora_mode": m.lora_mode,
            "r": m.r,
            "lora_dropout": m.lora_dropout,
        }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"
