# Lumen

Lumen is a LoRA alignment of two frozen backbones, Virchow2 (image) and
BioMedBERT (text), trained contrastively so that histopathology images and
free-text descriptions land in one embedding space. This repository trains it,
benchmarks it against eleven public vision-language baselines, and reproduces
every number reported for it.

All models share one interface, so switching between them is a one-argument
change:

```python
from lumen.encode import Encoder
from lumen.models.registry import PAPER_CHECKPOINT

enc = Encoder(PAPER_CHECKPOINT)             # or "conch", "plip", "clip_l14", ...
img_emb = enc.encode_images([pil_image])    # [N, D], L2-normalized

scores = enc.zero_shot([pil_image], {       # {class: [prompt variants]}
    "benign": ["benign tissue", "a benign H&E image"],
    "tumor":  ["tumor tissue", "carcinoma"],
})                                          # [N, n_classes] probabilities
```

`PAPER_CHECKPOINT` in `lumen/models/registry.py` is the reported checkpoint.

## Models

Fourteen registry keys. The twelve baselines load from `assets/models/`, which
you populate from [`docs/MODEL_SOURCES.md`](docs/MODEL_SOURCES.md); no
third-party weights are redistributed here.

Lumen itself ships, in `assets/models/`. It is the alignment only, 2,984,961
parameters, which attaches to the Virchow2 and BioMedBERT backbones you fetch
yourself, so it runs from a clone with no retraining. The corpus ablation
`lumen_retrain_pathgen` ships the same way.

| Family | Keys |
| --- | --- |
| HF `CLIPModel` | `clip_b16` `clip_b32` `clip_l14` `plip` `quilt_b16` `quilt_b32` `pathgen_l14` |
| open_clip `.pt` | `pathclip` `pathgen_b16` |
| BiomedCLIP | `biomedclip` |
| KEEP | `keep` |
| CONCH | `conch` |
| **Lumen** | `lumen_retrain_quilt` (= `PAPER_CHECKPOINT`), plus `lumen_retrain_pathgen` |

## Data

Nine public patch-classification datasets, resolved under `assets/datasets/`
and declared in `lumen/data/registry.py`:

`lc25000` (5 classes) · `osteo` (3) · `pcam` (2) · `sicap` (4) · `mhist` (2) ·
`databiox` (3) · `bach` (4) · `nct_crc` (9) · `wsss4luad` (3)

The whole-slide lymph-node cohorts are not part of this repository, and neither
are the slide tables that index them. Each table's `file_path` column is an
absolute path you repoint at your own copy. The internal cohort is patient data
under ethics approval. The external cohorts are public;
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) gives the download URI and
citation for each.

## Layout

```text
lumen/
  encode.py         # Encoder facade + zero_shot()  <- start here
  zeroshot.py       # shared prompt-ensemble -> logit -> softmax primitives
  models/           # registry.py (MODELS, PAPER_CHECKPOINT) + base.py + loaders.py
  data/             # registry.py (DATASETS, SLIDE_TABLES) + loaders.py + transforms.py
  benchmark/        # extract.py (Stage A), prompts.py, metrics.py, evaluate.py (Stage B)
  training/         # LoRA contrastive training: engine, loss, grad_cache, schedule
  wsi/              # whole-slide pipeline: engine, slide_benchmark, calibration
  inference/        # slide-table runner, metrics, io, records
  retrieval.py      # cross-modal retrieval (ARCH)
  utils/            # backbones, LoRA, the model itself, annotation readers
cli/                # thin entry points, one per stage
configs/            # inference and training settings
tests/              # runs without a GPU and without any data
docs/               # MODEL_SOURCES.md, DATA_SOURCES.md
```

## Benchmark

Each image is encoded once per model (Stage A). Stage B scores those cached
embeddings against a prompt bank, reporting the prompt-ensemble result with a
bootstrap CI. `--prompt-style` selects the bank, so the same models can be
re-scored under a second protocol without re-encoding anything.

Classification uses the highest-scoring class (`argmax`). Stage B also
reports AUROC and, for binary datasets, positive-class F1, precision,
sensitivity, specificity, and accuracy at both the explicit 0.5 threshold and
the Youden's-J threshold.

```bash
# Stage A: cache image embeddings (GPU)
python cli/extract_embeddings.py --model conch --dataset pcam

# Stage B: score against the prompt bank (reads Stage A caches, CPU)
python cli/benchmark_evaluate.py --model all --dataset all
#   -> outputs/benchmark/evaluation/summary.csv
```

## Install

```bash
conda env create -f environment.yml
conda activate lumen
pip install -e . --no-deps            # register the `lumen` package
pip install git+https://github.com/MahmoodLab/CONCH.git   # CONCH ships separately
python -m pytest tests/ -q            # no GPU and no data required
```

`PYTHONNOUSERSITE=1` must be set (an `activate.d` hook is the reliable way);
otherwise `~/.local/lib/pythonX.Y/site-packages` shadows the environment and
`environment.yml` stops describing what actually runs.

## Reproducing the results

[`REPRODUCE.md`](REPRODUCE.md) has the stage list: what each stage reads, what
it writes, and which stages need the whole-slide cohorts.

## Intended use

Research only. This is not a medical device and must not be used to diagnose,
treat or prevent disease, or as a substitute for a clinician's judgement. The
Virchow2 backbone's licence forbids clinical, diagnostic, Research Use Only and
Investigational Use Only applications outright, and that restriction reaches
anything built on it, including this work.

The reported numbers come from retrospective cohorts under one prompt protocol
and one operating point. Fairness across demographics has not been evaluated;
the training corpora's biases are not well characterised.

## License

CC BY-NC 4.0, see [`LICENSE`](LICENSE). The same licence the Lumen weights
carry on the Hugging Face Hub, applied to the source and to the alignment under
`assets/models/` alike. **Non-commercial.**

It grants no rights over the third-party weights the code loads. Virchow2, CONCH
and the rest in [`docs/MODEL_SOURCES.md`](docs/MODEL_SOURCES.md) come from their
own sources under their own terms. Virchow2 is CC BY-NC-ND 4.0 and gated, so the
assembled model is non-commercial regardless of this file, and its vision
encoder may not be modified. Read the backbone terms against your intended
use.
