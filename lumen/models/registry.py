"""Registry of vision-language models available to the framework.

Adding a model = one :class:`ModelSpec` entry here. ``family`` selects the
loader in :mod:`lumen.models.loaders`; ``rel_dir`` is relative to
``assets/models``; ``extra`` carries family-specific args (open_clip arch name,
checkpoint filename, ...). All checkpoints load from local ``assets/models``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lumen.paths import models_root


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    rel_dir: str
    extra: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    # --- HF CLIPModel family (get_image_features / get_text_features) ---------
    "clip_b16":  ModelSpec("clip_b16",  "hf_clip", "clip_vit_b16"),
    "clip_b32":  ModelSpec("clip_b32",  "hf_clip", "clip_vit_b32"),
    "clip_l14":  ModelSpec("clip_l14",  "hf_clip", "clip_vit_l14"),
    "plip":      ModelSpec("plip",      "hf_clip", "plip"),
    "quilt_b16": ModelSpec("quilt_b16", "hf_clip", "quiltnet_b16"),
    "quilt_b32": ModelSpec("quilt_b32", "hf_clip", "quiltnet_b32"),
    "pathgen_l14": ModelSpec("pathgen_l14", "hf_clip", "pathgen_l14_hf"),

    # --- open_clip local .pt checkpoints --------------------------------------
    "pathclip": ModelSpec(
        "pathclip", "open_clip_pt", "pathclip_base",
        extra={"arch": "ViT-B-16", "ckpt": "pathclip-base.pt"},
    ),
    "pathgen_b16": ModelSpec(
        "pathgen_b16", "open_clip_pt", "pathgen_b16",
        extra={"arch": "ViT-B-16", "ckpt": "pathgenclip.pt"},
    ),

    # --- BiomedCLIP (open_clip custom config + PubMedBERT text) ---------------
    "biomedclip": ModelSpec(
        "biomedclip", "biomedclip", "biomedclip_pubmedbert",
        extra={"config": "open_clip_config.json",
               "ckpt": "open_clip_pytorch_model.bin"},
    ),

    # --- KEEP (transformers custom code: ViT-L/16 + BERT) ---------------------
    "keep": ModelSpec("keep", "keep", "keep"),

    # --- CONCH (MahmoodLab conch package) -------------------------------------
    "conch": ModelSpec(
        "conch", "conch", "conch",
        extra={"arch": "conch_ViT-B-16", "ckpt": "pytorch_model.bin"},
    ),

    # --- Lumen ---------------------------------------------------------------
    # Ships in assets/models/, so a reviewer can evaluate it from a clone
    # without a retrain. What ships is the alignment only, 2,984,961 parameters:
    # the LoRA adapters, the two projection heads and the logit scale. The
    # Virchow2 and BioMedBERT backbones are not ours to redistribute and are
    # fetched from the Hub and assembled at load time. Training writes a full
    # checkpoint to outputs/training/ instead; see REPRODUCE.md.
    "lumen_retrain_quilt": ModelSpec(
        "lumen_retrain_quilt", "lumen", "lumen_quilt_r4",
        extra={"ckpt": "model.safetensors",
               "config": "checkpoint_best_model_cfg.json"},
    ),
    # Same architecture and recipe as the QUILT retrain, different corpus:
    # PathGen-1.6M (1,606,064 pairs over 7,203 TCGA WSIs), 21 x 625 steps.
    "lumen_retrain_pathgen": ModelSpec(
        "lumen_retrain_pathgen", "lumen", "lumen_pathgen_r4",
        extra={"ckpt": "model.safetensors",
               "config": "checkpoint_best_model_cfg.json"},
    ),
    # The full fine-tuning control for the adaptation ablation is NOT registered.
    # It trains with lora=False and both backbones unfrozen, so what it learns is
    # the modified Virchow2 and BioMedBERT weights: an 8.3 GB checkpoint with no
    # adapter tensors to extract, and a derivative of a model whose licence
    # forbids derivatives. It cannot be distributed, so registering it would put
    # back a key no reproducer can obtain. Its numbers are reported in the paper.
}

# The checkpoint the paper reports. Every headline number (patch benchmark,
# lymph-node internal and external, vision probe, paired comparisons) is this
# key, so anything that defaults to a different one silently reports a different
# model. The corpus ablation scores close but not equal to it, so a mix-up does
# not announce itself. Reach the reported model through this constant, never by
# key.
PAPER_CHECKPOINT = "lumen_retrain_quilt"

# Keys trained here rather than downloaded. PAPER_CHECKPOINT is one of them
# despite the ``retrain`` prefix; the other is the corpus ablation.
RETRAIN_MODELS = frozenset({"lumen_retrain_quilt", "lumen_retrain_pathgen"})


def model_dir(spec: ModelSpec) -> Path:
    return models_root() / spec.rel_dir

