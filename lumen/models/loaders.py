"""Family loaders that build a :class:`VLMAdapter` from a :class:`ModelSpec`.

All checkpoints load from local ``assets/models`` dirs (fully offline when
``HF_HUB_OFFLINE=1``). ``load_adapter`` is the single entry point.
"""

from __future__ import annotations

import json

import torch

from lumen.models.base import VLMAdapter
from lumen.models.registry import ModelSpec, model_dir


def _load_hf_clip(spec: ModelSpec, device) -> VLMAdapter:
    """transformers CLIPModel: openai CLIP, PLIP, QuiltNet, PathGen-L14-hf."""
    from transformers import CLIPModel, CLIPProcessor

    path = str(model_dir(spec))
    # sdpa dispatches to PyTorch's fused attention kernels (flash/mem-efficient on
    # CUDA) instead of the eager Python attention loop. Same exact attention
    # math, just faster/less memory, so outputs match eager up to ordinary
    # floating-point rounding. Falls back to eager if unsupported for this arch.
    try:
        model = CLIPModel.from_pretrained(path, attn_implementation="sdpa")
    except (ValueError, TypeError):
        model = CLIPModel.from_pretrained(path)
    model = model.eval().to(device)
    proc = CLIPProcessor.from_pretrained(path)
    logit_scale = float(model.logit_scale.exp().item())
    image_proc = proc.image_processor
    max_text_len = int(model.config.text_config.max_position_embeddings)

    def preprocess(im):
        return image_proc(im, return_tensors="pt")["pixel_values"][0]

    def encode_pixels(px):
        return model.get_image_features(pixel_values=px)

    def text_fn(texts):
        tok = proc(text=texts, padding=True, truncation=True,
                   max_length=max_text_len,
                   return_tensors="pt").to(device)
        return model.get_text_features(**tok)

    return VLMAdapter(spec.key, device, preprocess, encode_pixels, text_fn,
                      logit_scale)


def _load_open_clip_pt(spec: ModelSpec, device) -> VLMAdapter:
    """open_clip model from a local .pt (PathCLIP, PathGen-B16)."""
    import open_clip

    arch = spec.extra["arch"]
    ckpt = str(model_dir(spec) / spec.extra["ckpt"])
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch, pretrained=ckpt)
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(arch)
    logit_scale = float(model.logit_scale.exp().item())

    def encode_pixels(px):
        return model.encode_image(px)

    def text_fn(texts):
        tok = tokenizer(texts).to(device)
        return model.encode_text(tok)

    return VLMAdapter(spec.key, device, preprocess, encode_pixels, text_fn,
                      logit_scale)


def _load_biomedclip(spec: ModelSpec, device) -> VLMAdapter:
    """BiomedCLIP: open_clip custom config (timm ViT-B-16 + PubMedBERT text).

    The local dir ships an ``open_clip_config.json``; we register it as a
    named config so open_clip can build the model, then load the local weights.
    The HF text tower / tokenizer are read from local snapshots (offline).
    """
    import open_clip
    from open_clip import factory

    mdir = model_dir(spec)
    cfg = json.loads((mdir / spec.extra["config"]).read_text())
    model_cfg = cfg["model_cfg"]
    # The text tower architecture is a PubMedBERT skeleton; its trained weights
    # live inside the open_clip checkpoint (loaded below), not as a separate HF
    # model. Build the skeleton from the local biomedbert_base snapshot (it has
    # pytorch_model.bin + config), and tokenize with BiomedCLIP's own tokenizer
    # files (this dir). Both stay fully offline.
    text_skeleton = str(mdir.parent / "biomedbert_base")
    model_cfg["text_cfg"]["hf_model_name"] = text_skeleton
    model_cfg["text_cfg"]["hf_tokenizer_name"] = str(mdir)

    reg_name = "biomedclip_local"
    factory._MODEL_CONFIGS[reg_name] = model_cfg
    model, _, preprocess = open_clip.create_model_and_transforms(
        reg_name, pretrained=None,
        image_mean=cfg["preprocess_cfg"]["mean"],
        image_std=cfg["preprocess_cfg"]["std"],
    )
    state = torch.load(mdir / spec.extra["ckpt"], map_location="cpu")
    state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    # Newer transformers drops the non-persistent ``position_ids`` buffer that
    # the published checkpoint still carries; ignore only that key.
    state = {k: v for k, v in state.items() if not k.endswith("position_ids")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(
            f"BiomedCLIP state_dict mismatch: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}")
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(reg_name)
    logit_scale = float(model.logit_scale.exp().item())

    ctx_len = model_cfg["text_cfg"].get("context_length", 256)

    def encode_pixels(px):
        return model.encode_image(px)

    def text_fn(texts):
        tok = tokenizer(texts, context_length=ctx_len).to(device)
        return model.encode_text(tok)

    return VLMAdapter(spec.key, device, preprocess, encode_pixels, text_fn,
                      logit_scale)


def _load_conch(spec: ModelSpec, device) -> VLMAdapter:
    """CONCH via the MahmoodLab ``conch`` package (local checkpoint)."""
    from conch.open_clip_custom import (
        create_model_from_pretrained, get_tokenizer, tokenize,
    )

    ckpt = str(model_dir(spec) / spec.extra["ckpt"])
    model, preprocess = create_model_from_pretrained(
        spec.extra["arch"], checkpoint_path=ckpt)
    model = model.eval().to(device)
    tokenizer = get_tokenizer()
    # CONCH ships a *learned* inverse temperature (~56). Passing None here would
    # silently fall back to CLIP's 1/0.07 = 14.3, which is not the scale the model
    # was trained to produce probabilities at -- its maps come out washed out.
    logit_scale = float(model.logit_scale.exp().item())

    def encode_pixels(px):
        return model.encode_image(px, proj_contrast=True, normalize=False)

    def text_fn(texts):
        tok = tokenize(texts=list(texts), tokenizer=tokenizer).to(device)
        return model.encode_text(tok)

    return VLMAdapter(spec.key, device, preprocess, encode_pixels, text_fn,
                      logit_scale)


def _lumen_logit_scale(model) -> float | None:
    """Lumen's learned inverse temperature, as its ContrastiveHead actually uses it.

    The head does ``logit_scale.exp().clamp(1.0, 20.0)`` on every forward, so the
    clamp is part of the model, not a safety net -- read the *effective* value, not
    the raw exp. Returns ``None`` for a checkpoint with no head (the HF export drops
    it), which falls back to CLIP's 1/0.07.
    """
    head = getattr(model, "contrastive_head", None)
    if head is None or not hasattr(head, "logit_scale"):
        return None
    return float(head.logit_scale.exp().clamp(1.0, 20.0).item())


def _load_lumen(spec: ModelSpec, device) -> VLMAdapter:
    """The Lumen finetune (Virchow2 + BiomedBERT + contrastive projector)."""
    from lumen.utils.model import load_model

    mdir = model_dir(spec)
    ckpt = str(mdir / spec.extra["ckpt"])
    config = str(mdir / spec.extra["config"])
    model, tokenizer, img_tfms = load_model(ckpt, config, device)
    model.eval()
    logit_scale = _lumen_logit_scale(model)

    def encode_pixels(px):
        return model.encode_image(px)

    def text_fn(texts):
        tok = tokenizer(list(texts), padding=True, truncation=True,
                        return_tensors="pt").to(device)
        return model.encode_text(tok)

    return VLMAdapter(spec.key, device, img_tfms, encode_pixels, text_fn,
                      logit_scale)


def _load_keep(spec: ModelSpec, device) -> VLMAdapter:
    """KEEP (MAGIC-AI4Med): ViT-L/16 vision + BERT text, via trust_remote_code.

    Preprocessing and tokenizer call follow the model card exactly: bicubic resize
    to 224 then centre crop, ImageNet normalisation, and text padded to a fixed
    256 tokens. ``encode_image``/``encode_text`` already L2-normalise; VLMAdapter
    normalises again, which is idempotent.
    """
    import sys

    import timm.models.vision_transformer as tvit
    from torchvision import transforms
    from transformers import AutoModel, AutoTokenizer

    path = str(model_dir(spec))

    # KEEP's trust_remote_code module contains, at import scope:
    #     timm.models.vision_transformer.LayerScale = RenameLayerScale
    # Its LayerScale names the parameter `weight`; timm 1.0's names it `gamma`.
    # The swap is global and permanent, so *every timm ViT built later in the
    # same process* silently gets the wrong parameter names. Lumen's Virchow2
    # (vit_huge_patch14_224) then dies loading its own pretrained weights, which
    # is how a 13-model benchmark run crashed on the one model that happened
    # to come after KEEP. Confine the patch to this call.
    #
    # Restoring alone is not enough: the module-level swap only runs on *first*
    # import, so a second load would build KEEP's own ViT with timm's `gamma`
    # and fail on KEEP's `weight` checkpoint. Re-apply from the already-imported
    # module when that happens.
    original_layer_scale = tvit.LayerScale
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").endswith("modeling_keep") and \
                hasattr(mod, "RenameLayerScale"):
            tvit.LayerScale = mod.RenameLayerScale
            break
    try:
        model = AutoModel.from_pretrained(path, trust_remote_code=True).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    finally:
        tvit.LayerScale = original_layer_scale

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # KEEP carries a learned logit_scale (initialised at log(1/0.04)). Read the
    # trained value rather than defaulting to CLIP's 1/0.07: it is argmax-neutral
    # for classification but not for anything that averages probabilities.
    scale = getattr(model, "logit_scale", None)
    logit_scale = float(scale.exp().item()) if scale is not None else None

    def encode_pixels(px):
        return model.encode_image(px)

    def text_fn(texts):
        tok = tokenizer(list(texts), max_length=256, padding="max_length",
                        truncation=True, return_tensors="pt").to(device)
        return model.encode_text(tok)

    return VLMAdapter(spec.key, device, preprocess, encode_pixels, text_fn,
                      logit_scale)


_FAMILIES = {
    "hf_clip": _load_hf_clip,
    "open_clip_pt": _load_open_clip_pt,
    "biomedclip": _load_biomedclip,
    "conch": _load_conch,
    "lumen": _load_lumen,
    "keep": _load_keep,
}


def load_adapter(spec: ModelSpec, device: torch.device | None = None) -> VLMAdapter:
    """Load ``spec`` as a :class:`VLMAdapter` on ``device`` (auto-selected)."""
    from lumen import require_environment

    require_environment()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Inference-only process (server / benchmark CLIs): tensor-core matmuls in
        # TF32 and autotuned cuDNN algorithms are free throughput here since every
        # adapter here always runs the same fixed input shape per batch.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if spec.family not in _FAMILIES:
        raise ValueError(f"Unknown model family {spec.family!r} for {spec.key!r}")
    return _FAMILIES[spec.family](spec, device)
