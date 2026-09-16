#!/usr/bin/env python3
"""Train the Lumen alignment model cleanly and reproducibly.

Frozen Virchow2 + BioMedBERT backbones, LoRA adapters and two projection heads
trained with symmetric InfoNCE on QUILT-1M, using a per-step
warm-up/cosine schedule. Writes a checkpoint that loads back through the
existing inference stack with ``strict=True``.

Examples
--------
Smoke test on 512 pairs, 1 epoch (verifies the whole path end to end)::

    python cli/train_lumen.py --lmdb-path $Q/Quilt1M_dataset.lmdb \
        --limit-rows 512 --epochs 1 --batch-size 32 \
        --out-dir outputs/training/smoke

Full run::

    python cli/train_lumen.py --lmdb-path $Q/Quilt1M_dataset.lmdb \
        --peak-lr 5e-4 --epochs 20 --batch-size 128 --grad-accum 8 \
        --out-dir outputs/training/lumen_reproduction
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.training.config import TrainConfig
from lumen.training.data import build_loaders
from lumen.training.engine import build_model, set_seed, train


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="TrainConfig JSON; explicit flags override it")
    ap.add_argument("--lmdb-path", help="prebuilt QUILT LMDB (preferred over CSV/image files)")
    ap.add_argument("--lookup-csv")
    ap.add_argument("--image-root")
    ap.add_argument("--caption-col")
    ap.add_argument("--image-col")
    ap.add_argument("--out-dir")
    ap.add_argument("--run-name")
    ap.add_argument("--peak-lr", type=float)
    ap.add_argument("--warmup-steps", type=int)
    ap.add_argument("--min-lr-ratio", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--steps-per-epoch", type=int,
                    help="cap the epoch at N optimizer steps (0 = one full pass); "
                         "keeps epochs inside the 6h preemptable wall on large corpora")
    ap.add_argument("--min-epochs", type=int,
                    help="do not early-stop before this many completed epochs")
    ap.add_argument("--val-fraction", type=float,
                    help="held-out fraction for model selection (whole groups when "
                         "the LMDB carries group ids)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--gradient-cache", action="store_true",
                    help="use grad_accum microbatches as one true InfoNCE batch")
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--mlflow-tracking-uri")
    ap.add_argument("--mlflow-experiment")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--lora-r", type=int, help="LoRA rank (alpha follows as 2r); default 4")
    ap.add_argument("--limit-rows", type=int, help="cap lookup rows (smoke tests)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--full-finetune", action="store_true",
                    help="unfreeze both backbones (no LoRA); the matched full-tuning control")
    ap.add_argument("--backbone-lr", type=float,
                    help="LR for pretrained backbone weights; heads keep --peak-lr")
    ap.add_argument("--llrd", type=float, help="layer-wise LR decay (1.0 = off)")
    ap.add_argument("--beta2", type=float,
                    help="AdamW beta2; 0.95 is steadier than 0.999 for large backbones")
    ap.add_argument("--amp-dtype", choices=("float16", "bfloat16"),
                    help="autocast dtype; bfloat16 needs no loss scaling")
    ap.add_argument("--save-every-steps", type=int,
                    help="also write checkpoint_last.pt every N optimizer steps")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute encoder activations in backward to save memory")
    return ap.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig.from_json(args.config) if args.config else TrainConfig()
    # data
    if args.lmdb_path is not None:   cfg.data.lmdb_path = args.lmdb_path
    if args.lookup_csv is not None:  cfg.data.lookup_csv = args.lookup_csv
    if args.image_root is not None:  cfg.data.image_root = args.image_root
    if args.caption_col is not None: cfg.data.caption_col = args.caption_col
    if args.image_col is not None:   cfg.data.image_col = args.image_col
    if args.limit_rows is not None:  cfg.data.limit_rows = args.limit_rows
    if args.val_fraction is not None: cfg.data.val_fraction = args.val_fraction
    # optim / loop
    if args.peak_lr is not None:      cfg.optim.peak_lr = args.peak_lr
    if args.warmup_steps is not None: cfg.optim.warmup_steps = args.warmup_steps
    if args.min_lr_ratio is not None: cfg.optim.min_lr_ratio = args.min_lr_ratio
    if args.epochs is not None:       cfg.epochs = args.epochs
    if args.steps_per_epoch is not None: cfg.steps_per_epoch = args.steps_per_epoch
    if args.min_epochs is not None:   cfg.min_epochs = args.min_epochs
    if args.batch_size is not None:   cfg.batch_size = args.batch_size
    if args.grad_accum is not None:   cfg.grad_accum = args.grad_accum
    if args.gradient_cache:           cfg.gradient_cache = True
    if args.num_workers is not None:  cfg.num_workers = args.num_workers
    if args.seed is not None:         cfg.seed = args.seed
    if args.mlflow_tracking_uri is not None:
        cfg.mlflow_tracking_uri = args.mlflow_tracking_uri
    if args.mlflow_experiment is not None:
        cfg.mlflow_experiment = args.mlflow_experiment
    if args.mlflow:                    cfg.mlflow_enabled = True
    if args.lora_r is not None:       cfg.model.r = args.lora_r
    if args.full_finetune:
        # Nothing else freezes the backbones: the LoRA runs are frozen only as a
        # side effect of PEFT wrapping them, so dropping LoRA makes every tensor
        # trainable.
        cfg.model.lora = False
        cfg.model.lora_mode = "none"
        cfg.model.full_finetune = True
    if args.backbone_lr is not None:  cfg.optim.backbone_lr = args.backbone_lr
    if args.llrd is not None:         cfg.optim.llrd = args.llrd
    if args.beta2 is not None:        cfg.optim.beta2 = args.beta2
    if args.amp_dtype:                cfg.amp_dtype = args.amp_dtype
    if args.grad_checkpointing:       cfg.grad_checkpointing = True
    if args.save_every_steps is not None: cfg.save_every_steps = args.save_every_steps
    if args.out_dir is not None:      cfg.out_dir = args.out_dir
    if args.run_name is not None:     cfg.run_name = args.run_name
    if args.no_amp:                   cfg.amp = False
    return cfg.stamp()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse()
    cfg = build_config(args)

    if not cfg.data.lmdb_path and (not cfg.data.lookup_csv or not cfg.data.image_root):
        raise SystemExit("provide --lmdb-path, or both --lookup-csv and --image-root "
                         "(flags or config)")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("no CUDA device; training on CPU will be very slow")

    model, tokenizer, img_tfms = build_model(cfg, device)
    train_loader, val_loader, stats = build_loaders(
        cfg.data, img_tfms, tokenizer, cfg.batch_size, cfg.num_workers,
        cfg.seed, cfg.data.max_text_len)
    logging.info("data: %s", stats)

    result = train(cfg, train_loader, val_loader, model, device, data_stats=stats)
    logging.info("done: best val_loss %.4f at epoch %d -> %s",
                 result["best_val_loss"], result["best_epoch"], cfg.out_dir)
    # Completion marker: train() only returns when all epochs run or early
    # stopping triggers (a preempted window never reaches here), so the
    # self-chaining sweep uses this file to tell finished runs from interrupted
    # ones that must resume.
    (Path(cfg.out_dir) / "TRAINING_COMPLETE").write_text(
        f"best_val_loss={result['best_val_loss']:.6f} best_epoch={result['best_epoch']}\n")


if __name__ == "__main__":
    main()
