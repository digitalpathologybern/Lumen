"""Training engine: model construction, loop, and checkpointing.

Reuses the verified architecture in :class:`lumen.utils.model.Model` (the
same class the inference and benchmark stack loads), so a checkpoint written
here is byte-compatible with the rest of the project. The base backbone weights
stay frozen; only the LoRA adapters and the two projection heads plus the
contrastive temperature are trained, matching Lumen's 0.40% trainable budget.

Checkpoints are saved in the original ``{epoch, model, optim, sched, scaler,
cfg}`` layout, where ``cfg`` is the model-config payload the loader expects, and
are accompanied by the full :class:`~lumen.training.config.TrainConfig` and a
SHA-256 of the weights for provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lumen.training.config import TrainConfig
from lumen.training.grad_cache import (
    to_device,
    cached_contrastive_backward,
    cached_contrastive_forward,
)
from lumen.training.loss import clip_contrastive_loss, contrastive_accuracy
from lumen.training.schedule import cosine_warmup
from lumen.training.tracking import MLflowTracker

LOGGER = logging.getLogger("lumen.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


_VISION_BLOCK = re.compile(r"^img_encoder\.blocks\.(\d+)\.")
_TEXT_LAYER = re.compile(r"^txt_encoder\.encoder\.layer\.(\d+)\.")
_VISION_DEPTH = 32   # Virchow2 blocks.0-31
_TEXT_DEPTH = 12     # BioMedBERT encoder.layer.0-11


def _depth_of(name: str) -> tuple[str, int, int]:
    """Return ``(tower, depth, max_depth)`` for layer-wise LR decay.

    Depth 0 is the embedding, then one step per transformer block, and the final
    norm sits one past the last block. A parameter that belongs to no tower is
    reported as a head, which never receives decay.
    """
    m = _VISION_BLOCK.match(name)
    if m:
        return "vision", int(m.group(1)) + 1, _VISION_DEPTH + 1
    m = _TEXT_LAYER.match(name)
    if m:
        return "text", int(m.group(1)) + 1, _TEXT_DEPTH + 1
    if name.startswith("img_encoder."):
        # patch_embed / cls_token / pos_embed are depth 0; the trailing norm is last.
        return "vision", (_VISION_DEPTH + 1 if name.startswith("img_encoder.norm") else 0), _VISION_DEPTH + 1
    if name.startswith("txt_encoder."):
        return "text", (_TEXT_DEPTH + 1 if name.startswith("txt_encoder.pooler") else 0), _TEXT_DEPTH + 1
    return "head", 0, 0


def build_param_groups(model, cfg: TrainConfig) -> list[dict]:
    """AdamW parameter groups for the run described by ``cfg``.

    Two things vary per group. **Learning rate**: heads always take
    ``peak_lr``; backbone tensors take ``backbone_lr`` when set, scaled by
    ``llrd ** (max_depth - depth)`` so early layers move least. **Weight
    decay**: every 1-D tensor (norm weights, biases, class/position tokens) and
    the logit scale are exempt, which is standard for transformers and matters
    more here because decaying a LayerNorm gain is a direct attack on a
    pretrained representation.

    With ``backbone_lr=None`` and ``llrd=1.0`` this collapses to a single rate
    for everything, reproducing the LoRA runs exactly.
    """
    peak = cfg.optim.peak_lr
    base = cfg.optim.backbone_lr if cfg.optim.backbone_lr is not None else peak
    d = cfg.optim.llrd
    groups: dict[tuple, dict] = {}
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        tower, depth, max_depth = _depth_of(name)
        if tower == "head":
            lr = peak
        else:
            lr = base * (d ** (max_depth - depth)) if d != 1.0 else base
        decay = cfg.optim.weight_decay
        if getattr(cfg.model, "full_finetune", False) and (
                prm.ndim <= 1 or name.endswith("logit_scale")):
            decay = 0.0
        key = (round(lr, 12), decay)
        g = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": decay,
                                    "names": []})
        g["params"].append(prm)
        g["names"].append(name)
    out = sorted(groups.values(), key=lambda g: -g["lr"])
    for g in out:
        g.pop("names")
    return out


def build_model(cfg: TrainConfig, device):
    """Construct the Lumen model with base frozen, LoRA + heads trainable.

    Returns ``(model, tokenizer, image_transform)``. Unlike
    :func:`lumen.utils.model.load_model`, this does *not* freeze everything:
    PEFT freezes the backbone base weights and marks the LoRA adapters
    trainable, and the projection/contrastive heads are trainable by default.
    """
    from lumen.utils.backbones import load_text_backbone, load_vision_backbone
    from lumen.utils.model import Model

    m = cfg.model
    txt_enc, tokenizer = load_text_backbone(m.text_backbone)
    vsn_enc, _, img_tfms, _, _ = load_vision_backbone(m.vision_backbone)

    model = Model(
        txt_encoder=txt_enc.to(device),
        img_encoder=vsn_enc.to(device),
        text_backbone=m.text_backbone,
        vision_backbone=m.vision_backbone,
        proj_dim=m.proj_dim,
        decay=m.decay,
        span=m.span,
        n_blocks=m.n_blocks,
        img_comp=m.img_comp,
        txt_comp=m.txt_comp,
        img_hidden=m.img_hidden,
        txt_hidden=m.txt_hidden,
        lora=m.lora,
        lora_mode=m.lora_mode,
        r=m.r,
        lora_dropout=m.lora_dropout,
    ).to(device)

    if getattr(m, "full_finetune", False):
        n = 0
        for prm in list(model.img_encoder.parameters()) + list(model.txt_encoder.parameters()):
            if not prm.requires_grad:
                prm.requires_grad = True
                n += 1
        LOGGER.info("full fine-tune: unfroze %d backbone tensors", n)

    if cfg.grad_checkpointing:
        # Both towers expose it, under different names. Without this a full
        # fine-tune stores activations for 32 Virchow2 blocks per micro-batch on
        # top of ~12 GB of optimizer state.
        enabled = []
        if hasattr(model.img_encoder, "set_grad_checkpointing"):
            model.img_encoder.set_grad_checkpointing(True); enabled.append("vision")
        if hasattr(model.txt_encoder, "gradient_checkpointing_enable"):
            model.txt_encoder.gradient_checkpointing_enable(); enabled.append("text")
        # HF disables the cache when checkpointing; it is unused here but the
        # warning is noisy and the flag is meaningless for an encoder.
        if hasattr(model.txt_encoder, "config"):
            model.txt_encoder.config.use_cache = False
        LOGGER.info("gradient checkpointing enabled on: %s", ", ".join(enabled) or "nothing")

    trainable, total = count_trainable(model)
    LOGGER.info("trainable params: %s / %s (%.3f%%)", f"{trainable:,}",
                f"{total:,}", 100.0 * trainable / total)
    return model, tokenizer, img_tfms


def _loader_groups(loader: DataLoader, size: int):
    """Yield lists of at most ``size`` consecutive DataLoader batches."""
    group = []
    for batch in loader:
        group.append(batch)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, amp: bool,
             amp_dtype: torch.dtype = torch.float16,
             logical_microbatches: int = 1) -> dict:
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    if logical_microbatches > 1:
        for group in _loader_groups(loader, logical_microbatches):
            result = cached_contrastive_forward(model, group, device, amp, amp_dtype)
            tot_loss += result.loss * result.pairs
            tot_acc += result.accuracy * result.pairs
            n += result.pairs
        return {"val_loss": tot_loss / max(1, n),
                "val_acc": tot_acc / max(1, n)}

    for images, tokens in loader:
        images, tokens = to_device((images, tokens), device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            out = model(images, tokens)
            loss = clip_contrastive_loss(out["logits"])
        bs = images.size(0)
        tot_loss += loss.item() * bs
        tot_acc += contrastive_accuracy(out["logits"]) * bs
        n += bs
    return {"val_loss": tot_loss / max(1, n), "val_acc": tot_acc / max(1, n)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path: Path, model, optim, sched, scaler, epoch: int,
                    cfg: TrainConfig, extra: dict, epoch_steps: int = 0) -> str:
    """Write a checkpoint in the original layout and return its SHA-256.

    ``epoch_steps`` is how many optimizer steps of ``epoch`` are already in
    these weights; 0 means the epoch finished. Resume uses it to re-enter a
    partly-done epoch instead of repeating it.

    The write goes to a temporary file and is then renamed. ``os.replace`` is
    atomic within a filesystem, so a preemption during the write leaves the
    previous checkpoint intact rather than a truncated file that the next
    window would fail to load -- which would lose the entire run, not one epoch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "epoch_steps": epoch_steps,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cfg": cfg.model_cfg_json(),
            "train_cfg": json_safe(cfg),
            "extra": extra,
        },
        tmp,
    )
    os.replace(tmp, path)
    digest = _sha256(path)
    (path.parent / f"{path.stem}.sha256").write_text(digest + "\n")
    return digest


def json_safe(cfg: TrainConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


def _assert_safe_out_dir(out_dir: Path) -> None:
    """Refuse to write training output into the original model assets.

    The released Lumen checkpoint lives under ``assets/models`` and must never
    be touched by a retrain. Any ``out_dir`` resolving inside that tree is a
    mistake, not a valid destination.
    """
    resolved = out_dir.resolve()
    protected = (Path.cwd() / "assets" / "models").resolve()
    if resolved == protected or protected in resolved.parents:
        raise SystemExit(
            f"refusing to train into {resolved}: it is inside the protected "
            f"model-assets tree ({protected}). Choose an out_dir under outputs/."
        )


def train(cfg: TrainConfig, train_loader: DataLoader, val_loader: DataLoader,
          model, device, data_stats: dict | None = None) -> dict:
    """Run the training loop with the per-step schedule."""
    out_dir = Path(cfg.out_dir)
    _assert_safe_out_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "train_config.json")

    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = build_param_groups(model, cfg)
    optim = torch.optim.AdamW(
        groups, lr=cfg.optim.peak_lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2), eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    if len(groups) > 1:
        LOGGER.info("optimizer: %d parameter groups, lr %.2e..%.2e, "
                    "%d tensors decay-exempt", len(groups),
                    min(g["lr"] for g in groups), max(g["lr"] for g in groups),
                    sum(len(g["params"]) for g in groups if g["weight_decay"] == 0))
    if cfg.grad_accum < 1:
        raise ValueError(f"grad_accum must be positive, got {cfg.grad_accum}")
    if cfg.gradient_cache:
        full_pass_steps = len(train_loader) // cfg.grad_accum
        if full_pass_steps < 1:
            raise ValueError(
                f"gradient cache needs at least {cfg.grad_accum} microbatches, "
                f"but the train loader has {len(train_loader)}"
            )
    else:
        full_pass_steps = max(1, math.ceil(len(train_loader) / cfg.grad_accum))
    if cfg.steps_per_epoch < 0:
        raise ValueError(f"steps_per_epoch must not be negative, got {cfg.steps_per_epoch}")
    steps_per_epoch = (min(full_pass_steps, cfg.steps_per_epoch)
                       if cfg.steps_per_epoch else full_pass_steps)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.epochs < 1:
        raise ValueError(f"epochs must be positive, got {cfg.epochs}")
    if not 0 <= cfg.min_epochs <= cfg.epochs:
        raise ValueError(
            f"min_epochs must be between 0 and epochs ({cfg.epochs}), "
            f"got {cfg.min_epochs}"
        )
    if cfg.optim.warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps={cfg.optim.warmup_steps} consumes the entire "
            f"{total_steps}-step run; reduce warmup_steps so training reaches "
            "the post-warmup schedule"
        )
    sched = cosine_warmup(optim, cfg.optim.warmup_steps, total_steps,
                          cfg.optim.min_lr_ratio)
    # bfloat16 carries fp32's exponent range, so gradient scaling is unnecessary
    # and an enabled scaler would silently no-op the unscale_/step protocol.
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_scaler = cfg.amp and amp_dtype is torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    if cfg.amp:
        LOGGER.info("autocast dtype=%s  grad_scaler=%s", cfg.amp_dtype, use_scaler)
    LOGGER.info("steps/epoch=%d  total_steps=%d  warmup_steps=%d  peak_lr=%.2e",
                steps_per_epoch, total_steps, cfg.optim.warmup_steps, cfg.optim.peak_lr)
    if steps_per_epoch < full_pass_steps:
        LOGGER.info("epoch capped at %d of %d full-pass steps; the run covers %.2f "
                    "passes over the corpus", steps_per_epoch, full_pass_steps,
                    total_steps / full_pass_steps)
    negative_pool = (cfg.batch_size * cfg.grad_accum
                     if cfg.gradient_cache else cfg.batch_size)
    LOGGER.info("micro_batch=%d  microbatches/step=%d  optimizer_batch=%d  "
                "InfoNCE_negative_pool=%d  gradient_cache=%s",
                cfg.batch_size, cfg.grad_accum, cfg.batch_size * cfg.grad_accum,
                negative_pool, cfg.gradient_cache)
    LOGGER.info("warmup ends in epoch %.2f; early stopping allowed after epoch %d",
                cfg.optim.warmup_steps / steps_per_epoch, cfg.min_epochs)
    tracker = MLflowTracker.start(cfg, out_dir, data_stats)

    best_val = float("inf")
    best_epoch = -1
    patience = 0
    history = []
    global_step = 0

    # Resume from a prior window if present (A100 preemptible, 6h wall). The
    # resumable state lives in checkpoint_last.pt, written every epoch below.
    start_epoch = 0
    resume_offset = 0
    last_path = out_dir / "checkpoint_last.pt"
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        ex = ckpt.get("extra", {})
        best_val = ex.get("best_val", best_val)
        best_epoch = ex.get("best_epoch", best_epoch)
        patience = ex.get("patience", patience)
        history = ex.get("history", history) or []
        global_step = ex.get("global_step", 0)
        # A mid-epoch checkpoint re-enters its own epoch with the completed
        # steps already banked; an epoch-end one starts the next epoch.
        resume_offset = int(ckpt.get("epoch_steps", 0))
        if resume_offset > 0:
            start_epoch = int(ckpt.get("epoch", 0))
        else:
            start_epoch = int(ckpt.get("epoch", -1)) + 1
        LOGGER.info("resumed from %s: start_epoch=%d offset=%d global_step=%d "
                    "best_val=%.4f patience=%d", last_path, start_epoch,
                    resume_offset, global_step, best_val, patience)

    for epoch in range(start_epoch, cfg.epochs):
        # Seeding epoch_steps makes the loop stop at the same cap after running
        # only the remainder, so the run's total step count is unchanged.
        resume_from = resume_offset if epoch == start_epoch else 0
        resume_offset = 0
        LOGGER.info("epoch %d/%d starting (%d steps%s)", epoch + 1, cfg.epochs,
                    steps_per_epoch - resume_from,
                    f", resuming after {resume_from}" if resume_from else "")
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optim.zero_grad(set_to_none=True)
        t0 = time.time()
        running = 0.0
        running_steps = 0
        epoch_steps = resume_from  # optimizer steps banked for this epoch
        if cfg.gradient_cache:
            for group in _loader_groups(train_loader, cfg.grad_accum):
                # Keep the negative pool fixed: at most 145/640,145 training
                # rows are dropped rather than taking a final 128-pair step.
                if len(group) < cfg.grad_accum:
                    break
                optim.zero_grad(set_to_none=True)
                cached = cached_contrastive_backward(
                    model, group, device, cfg.amp, scaler, amp_dtype)
                if cfg.optim.grad_clip_norm > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.optim.grad_clip_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()                       # once per optimizer step
                global_step += 1
                epoch_steps += 1
                running += cached.loss
                running_steps += 1
                if global_step % cfg.log_every == 0:
                    allocated = (torch.cuda.max_memory_allocated(device) / 2**30
                                 if device.type == "cuda" else 0.0)
                    reserved = (torch.cuda.max_memory_reserved(device) / 2**30
                                if device.type == "cuda" else 0.0)
                    LOGGER.info("epoch %d step %d  loss %.4f  lr %.2e",
                                epoch, global_step, running / running_steps,
                                sched.get_last_lr()[0])
                    tracker.log_step(
                        global_step, epoch, running / running_steps,
                        sched.get_last_lr()[0], scaler.get_scale(),
                        allocated, reserved)
                if (cfg.save_every_steps and epoch_steps < steps_per_epoch
                        and epoch_steps % cfg.save_every_steps == 0):
                    # Mid-epoch resume point. Without this a preemption discards
                    # everything since the last epoch boundary, which on a 6h
                    # window against a ~3h epoch is most of the window.
                    save_checkpoint(out_dir / "checkpoint_last.pt", model, optim,
                                    sched, scaler, epoch, cfg,
                                    {"best_val": best_val, "best_epoch": best_epoch,
                                     "patience": patience, "history": history,
                                     "global_step": global_step},
                                    epoch_steps=epoch_steps)
                    LOGGER.info("  mid-epoch checkpoint at step %d (epoch %d, %d/%d)",
                                global_step, epoch + 1, epoch_steps, steps_per_epoch)
                if epoch_steps >= steps_per_epoch:
                    break                          # capped epoch; reshuffles next epoch
        else:
            for it, (images, tokens) in enumerate(train_loader):
                images, tokens = to_device((images, tokens), device)
                group_start = (it // cfg.grad_accum) * cfg.grad_accum
                group_size = min(cfg.grad_accum, len(train_loader) - group_start)
                with torch.autocast("cuda", enabled=cfg.amp):
                    out = model(images, tokens)
                    loss = clip_contrastive_loss(out["logits"]) / group_size
                scaler.scale(loss).backward()
                running += loss.item() * group_size
                running_steps += 1

                if (it + 1) % cfg.grad_accum == 0 or (it + 1) == len(train_loader):
                    if cfg.optim.grad_clip_norm > 0:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(trainable, cfg.optim.grad_clip_norm)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    sched.step()
                    global_step += 1
                    epoch_steps += 1
                    if global_step % cfg.log_every == 0:
                        allocated = (torch.cuda.max_memory_allocated(device) / 2**30
                                     if device.type == "cuda" else 0.0)
                        reserved = (torch.cuda.max_memory_reserved(device) / 2**30
                                    if device.type == "cuda" else 0.0)
                        LOGGER.info("epoch %d step %d  loss %.4f  lr %.2e",
                                    epoch, global_step, running / running_steps,
                                    sched.get_last_lr()[0])
                        tracker.log_step(
                            global_step, epoch, running / running_steps,
                            sched.get_last_lr()[0], scaler.get_scale(),
                            allocated, reserved)
                    if epoch_steps >= steps_per_epoch:
                        break                      # capped epoch; reshuffles next epoch

        val = evaluate(model, val_loader, device, cfg.amp, amp_dtype,
                       cfg.grad_accum if cfg.gradient_cache else 1)
        rec = {"epoch": epoch, "train_loss": running / max(1, running_steps),
               "lr_end": sched.get_last_lr()[0], "secs": round(time.time() - t0, 1),
               **val}
        history.append(rec)
        tracker.log_epoch(epoch, rec)
        LOGGER.info("[epoch %d] train %.4f  val %.4f  val_acc %.3f  (%.0fs)",
                    epoch, rec["train_loss"], val["val_loss"], val["val_acc"], rec["secs"])

        warmup_finished = global_step >= cfg.optim.warmup_steps
        minimum_run_finished = (epoch + 1) >= cfg.min_epochs
        early_stop_eligible = warmup_finished and minimum_run_finished
        improved = val["val_loss"] < best_val - 1e-5
        if improved:
            best_val, best_epoch, patience = val["val_loss"], epoch, 0
            digest = save_checkpoint(
                out_dir / "checkpoint_best_model.pt", model, optim, sched, scaler,
                epoch, cfg, {"val": val, "history": history})
            # The loader expects a sibling *_cfg.json next to the checkpoint.
            (out_dir / "checkpoint_best_model_cfg.json").write_text(
                json.dumps(cfg.model_cfg_json(), indent=2) + "\n")
            LOGGER.info("  new best (val_loss %.4f); saved, sha256=%s", best_val, digest[:12])
        elif early_stop_eligible:
            patience += 1
            LOGGER.info("  no improvement (%d/%d)", patience, cfg.early_stop_patience)
        else:
            patience = 0
            LOGGER.info("  no improvement; early stopping inactive during "
                        "warmup/minimum-epoch guard")

        # Resumable checkpoint every epoch, so a 6h-wall or preemption restart
        # continues from here rather than from scratch.
        save_checkpoint(out_dir / "checkpoint_last.pt", model, optim, sched, scaler,
                        epoch, cfg, {"best_val": best_val, "best_epoch": best_epoch,
                                     "patience": patience, "history": history,
                                     "global_step": global_step})
        tracker.log_checkpoint_metadata()

        if early_stop_eligible and patience >= cfg.early_stop_patience:
            LOGGER.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    tracker.finish()
    return {"best_val_loss": best_val, "best_epoch": best_epoch, "history": history}
