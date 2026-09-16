# Reproducing the results

## Install

```bash
conda env create -f environment.yml
conda activate lumen
pip install -e . --no-deps
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` keeps `~/.local` packages from overriding the environment.
The code refuses to load a model if they have.

## Get the weights and data

Lumen is already in `assets/models/`. You need the rest:

- **Baselines** into `assets/models/`. See [`docs/MODEL_SOURCES.md`](docs/MODEL_SOURCES.md).
- **Virchow2 and BioMedBERT** from the Hub. Virchow2 is gated: accept its
  licence, then `huggingface-cli login`.
- **Patch datasets** into `assets/datasets/`, in the layout
  `lumen/data/registry.py` expects.
- **Slide cohorts**, for stages 3 to 6 only. See
  [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md), then point the `file_path`
  column of the slide tables at your copy.

## Run

```bash
# 1. cache patch embeddings
python cli/extract_embeddings.py --model all --dataset all

# 2. patch benchmark
python cli/benchmark_evaluate.py --model all --dataset all \
    --prompt-style rich_canonical

# 3. cache WSI tile embeddings (~48 GB)
python cli/extract_wsi_embeddings.py --model all --table all

# 4. slide scoring
python cli/wsi_benchmark_evaluate.py --model all --table all

# 5. locked threshold
python cli/evaluate_wsi_locked_threshold.py

# 6. paired comparisons
python cli/recompute_wsi_clustered_stats.py
python cli/compare_wsi_models_clustered.py

# 7. retrieval
python cli/extract_arch_retrieval_embeddings.py --model all
python cli/evaluate_retrieval.py

# 8. linear probe
python cli/probe_vision_tower.py --dataset all
```

Everything lands under `outputs/`.

Notes:

- Stage 2 needs `--prompt-style rich_canonical`. The default is the
  six-template protocol from the supplement, which scores about 0.01 higher.
- Run 4, 5, 6 in that order. Each reads the last one's output.
- Stage 3 is the long one. It skips slides it has already done, and shards:
  `--num-shards 16 --shard-index $i`.
- Stages 3 to 6 need the cohorts. The external ones are public; the internal
  cohort is not distributed.
- The patch aggregate is the mean chance-corrected balanced accuracy over the
  nine datasets: 0.547 for `PAPER_CHECKPOINT`.

## Ablations

`lumen_retrain_pathgen` is the corpus ablation, trained on PathGen-1.6M instead
of QUILT-1M. It ships alongside the reported model, so rerun stages 1 and 2 with
`--model lumen_retrain_pathgen`.

The full-finetune ablation is not distributable: with LoRA off, training changes
the backbones themselves, and Virchow2's licence forbids sharing a modified
version. Retraining with `--full-finetune` is the only way to get it.

## Training

Not needed for any of the above.

```bash
python cli/train_lumen.py --config configs/quilt_clean.json \
    --lmdb-path "$QUILT_LMDB" \
    --out-dir outputs/training/lumen_quilt_r4 --run-name lumen_quilt_r4
```

20 epochs over QUILT-1M, batch 128 with 8-step accumulation. Budget GPU-days.
Add `--limit-rows 512 --epochs 1` first to check the corpus path works.

For PathGen: `cli/build_pathgen_patches.py`, then `cli/build_pathgen_lmdb.py`,
then the same with `--verify`.

To evaluate your own run, add a `ModelSpec` for it and put the checkpoint in
`assets/models/<key>/`. `load_model` reads the full training checkpoint or an
alignment-only `.safetensors`.
