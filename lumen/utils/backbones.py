import contextlib
import os
from pathlib import Path

import timm
import torch

# -- Utility functions  --

#: Scratch override for model weights.
SCRATCH_ENV_VARS = ("LUMEN_SCRATCH",)


def _resolve_weight(relative: str) -> str:
    """
    Return the first existing path for a model-weight file, checking in order:
      1. $LUMEN_SCRATCH/<relative>  (scratch storage, fast NVMe)
      2. The hardcoded shared-research path passed as `relative`
    Set it via:  export LUMEN_SCRATCH=/path/to/fast/local/scratch/lumen_models
    """
    scratch = next((os.environ[v] for v in SCRATCH_ENV_VARS if os.environ.get(v)), None)
    if scratch:
        candidate = str(Path(scratch) / relative)
        if os.path.exists(candidate):
            return candidate
    return relative

def supports_token_type_ids(model):
    """
    Return True if model.forward accepts 'token_type_ids'.
    """
    return "token_type_ids" in model.forward.__code__.co_varnames

def wrap_forward_remove_token_type_ids(model):
    """
    Wrap model.forward to drop token_type_ids if passed.
    """
    orig_forward = model.forward
    def forward_compatible(**kwargs):
        kwargs.pop("token_type_ids", None)
        return orig_forward(**kwargs)
    model.forward = forward_compatible
    return model

# ------------------------------------------------

accepted_text_backbones = {
    "biomedBERT":   "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "bioGPT":       "microsoft/biogpt",
    "MedBERT":      "Charangan/MedBERT",
    "ClinicalBERT": "medicalai/ClinicalBERT",
    "clip_vit_b16": "openai/clip-vit-base-patch16",
    "clip_vit_b32": "openai/clip-vit-base-patch32",
    "clip_vit_l14": "openai/clip-vit-large-patch14",
    "plip":         "vinid/plip",
    "quilt_b32":    "wisdomik/QuiltNet-B-32",
    "quilt_b16":    "wisdomik/QuiltNet-B-16",
    "pathgen_b16":  "pathgen-B-16/pathgenclip.pt",
    "pathgen_l14":  "hf-hub:jamessyx/pathgenclip-vit-large-patch14-hf",
}

accepted_vision_backbones = {
    "Virchow2":     "hf-hub:paige-ai/Virchow2",
    "clip_vit_b16": "openai/clip-vit-base-patch16",
    "clip_vit_b32": "openai/clip-vit-base-patch32",
    "clip_vit_l14": "openai/clip-vit-large-patch14",
    "plip":         "vinid/plip",
    "quilt_b32":    "wisdomik/QuiltNet-B-32",
    "quilt_b16":    "wisdomik/QuiltNet-B-16",
    "pathgen_b16":  "pathgen-B-16/pathgenclip.pt",
    "pathgen_l14":  "hf-hub:jamessyx/pathgenclip-vit-large-patch14-hf",
}

accepted_architectures = ["MLP", "gMLP", "AttnMLP", "CLIP"]

#: Commits the two Lumen backbones are pinned to. They are the only weights this
#: package pulls from the Hub, and an unpinned `main` is how a rerun silently
#: picks up a reupload. The same commits are in docs/MODEL_SOURCES.md, and
#: tests/test_backbone_pins.py checks the two agree.
BACKBONE_REVISION = {
    "Virchow2":   "3158645804b69e3f3bc4439d4116edddf0840a72",
    "biomedBERT": "e1354b7a3a09615f6aba48dfad4b7a613eef7062",
}

GATED_BACKBONES = {"hf-hub:paige-ai/Virchow2"}


@contextlib.contextmanager
def hub_errors(repo: str):
    """Turn a failed backbone fetch into an instruction.

    The backbones are not in this repository and are pulled from the Hub. Two
    things go wrong and neither says so plainly on its own: the repo is gated
    and the user has not been granted access, or REPRODUCE's offline flags were
    exported before the cache was primed, which transformers reports as a
    connection problem naming a repository that is in fact public.
    """
    try:
        yield
    except Exception as exc:
        name = repo.removeprefix("hf-hub:").split("@")[0]
        offline = (os.environ.get("HF_HUB_OFFLINE") not in (None, "", "0")
                   or os.environ.get("TRANSFORMERS_OFFLINE") not in (None, "", "0"))
        lines = [f"could not load the {name} backbone.", ""]
        if offline:
            lines += [
                "HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set, so only the local",
                "Hugging Face cache was searched and this backbone is not in it.",
                "Prime the cache once with those variables unset:", "",
                f"    huggingface-cli download {name}", "",
            ]
        if repo in GATED_BACKBONES:
            lines += [
                f"{name} is gated. Accept its licence at",
                f"    https://huggingface.co/{name}",
                "then authenticate with `huggingface-cli login`. Its terms",
                "require every user to register individually, which is why the",
                "weights are not redistributed here.", "",
            ]
        lines += [f"See docs/MODEL_SOURCES.md. Original error: {exc}"]
        raise RuntimeError("\n".join(lines)) from exc


# ------------------------------------------------

def load_text_backbone(backbone):
    from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel

    # allow projector file aliasing
    if os.path.isfile(backbone):
        fn = os.path.basename(backbone)
        key = fn.split("_projector")[0]
        if key not in accepted_text_backbones:
            raise ValueError(f"Unknown projector prefix '{key}' in {fn}")
        print(f"[BACKBONE txt]\tdetected projector file for '{key}' ({fn})")
        backbone = key

    # BERT-style
    if backbone in {"biomedBERT","MedBERT","ClinicalBERT","bioGPT"}:
        model_id = accepted_text_backbones[backbone]
        rev = BACKBONE_REVISION.get(backbone)
        with hub_errors(model_id):
            # The pinned commit ships pytorch_model.bin, not safetensors.
            # Saying so keeps the weights unambiguously the ones at that commit:
            # transformers otherwise prefers a converted safetensors file, which
            # is published on a different revision.
            txt_encoder = AutoModel.from_pretrained(
                model_id, revision=rev, use_safetensors=False)
            txt_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=rev)
        txt_encoder.resize_token_embeddings(len(txt_tokenizer))

    # CLIP/GPT-style
    elif backbone in {"clip_vit_b32","clip_vit_l14","plip","quilt_b32","quilt_b16"}:
        hf_id = accepted_text_backbones[backbone]
        txt_tokenizer = CLIPTokenizer.from_pretrained(hf_id)
        txt_encoder   = CLIPTextModel.from_pretrained(hf_id)

    # PathGen-B16 from local .pt via open_clip
    elif backbone == "pathgen_b16":
        import open_clip
        arch    = "ViT-B-16"
        pt_path = accepted_text_backbones[backbone]
        # load both vision+text into one model
        model, _, _ = open_clip.create_model_and_transforms(
            model_name=arch,
            pretrained=pt_path
        )
        txt_encoder   = model
        txt_tokenizer = open_clip.get_tokenizer(arch)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        print(f"[BACKBONE txt]\t{backbone} loaded via open_clip local .pt")
        return txt_encoder, txt_tokenizer

    # PathGen-L14 via open_clip hub (tokenizer only; text encoder is model from vision loader)
    elif backbone == "pathgen_l14":
        import open_clip
        hf_id = accepted_text_backbones[backbone]
        txt_tokenizer = open_clip.get_tokenizer(hf_id)
        txt_encoder = None
        print(f"[BACKBONE txt]\t{backbone} tokenizer loaded via open_clip hub")
        return txt_encoder, txt_tokenizer

    else:
        raise ValueError(f"Unable to load requested text backbone: {backbone}")

    # freeze & wrap
    txt_encoder.eval()
    if not supports_token_type_ids(txt_encoder):
        txt_encoder = wrap_forward_remove_token_type_ids(txt_encoder)
    for p in txt_encoder.parameters():
        p.requires_grad = False

    print(f"[BACKBONE txt]\t{backbone} loaded")
    return txt_encoder, txt_tokenizer


def load_vision_backbone(backbone):
    # file aliasing
    if os.path.isfile(backbone):
        fn = os.path.basename(backbone)
        key = fn.split("_projector")[0]
        if key not in accepted_vision_backbones:
            raise ValueError(f"Unknown projector prefix '{key}' in {fn}")
        print(f"[BACKBONE vsn]\tdetected projector file for '{key}' ({fn})")
        backbone = key

    # Virchow2
    if backbone == "Virchow2":
        from timm.data import resolve_data_config, create_transform
        from timm.layers import SwiGLUPacked
        model_ref = accepted_vision_backbones[backbone]
        rev = BACKBONE_REVISION.get(backbone)
        if rev:
            model_ref = f"{model_ref}@{rev}"      # timm parses the @revision
        with hub_errors(model_ref):
            vision_backbone = timm.create_model(
                model_ref, pretrained=True,
                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
            ).eval()
        # Weights are already loaded by timm from the HF cache above.
        # Local save/load is skipped: storage is full and the HF cache is the source of truth.
        vision_cfg = resolve_data_config(vision_backbone.pretrained_cfg, model=vision_backbone)
        img_tfms = create_transform(**vision_cfg)
        mean, std = vision_cfg["mean"], vision_cfg["std"]
        for p in vision_backbone.parameters():
            p.requires_grad = False
        print(f"[BACKBONE vsn]\t{backbone} loaded")
        return vision_backbone, vision_cfg, img_tfms, mean, std

    # PathGen-B16 local .pt via open_clip
    if backbone == "pathgen_b16":
        import open_clip
        arch    = "ViT-B-16"
        pt_path = accepted_vision_backbones[backbone]
        with open(os.devnull,"w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                model_name=arch, pretrained=pt_path
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        img_tfms = lambda imgs: (
            preprocess_val(imgs).unsqueeze(0)
            if not hasattr(imgs, "__len__")
            else torch.cat([preprocess_val(i).unsqueeze(0) for i in imgs], dim=0)
        )
        print(f"[BACKBONE vsn]\t{backbone} loaded via open_clip local .pt")
        return model, None, img_tfms, None, None

    # PathGen-L14 via open_clip hub
    if backbone in {"pathgen-l", "pathgen_l14"}:
        import open_clip
        hub_id = accepted_vision_backbones[backbone]
        with open(os.devnull,"w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(hub_id)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        img_tfms = lambda imgs: (
            preprocess_val(imgs).unsqueeze(0)
            if not hasattr(imgs, "__len__")
            else torch.cat([preprocess_val(i).unsqueeze(0) for i in imgs], dim=0)
        )
        print(f"[BACKBONE vsn]\t{backbone} loaded via open_clip hub")
        return model, None, img_tfms, None, None

    # HF CLIP / PLIP / QuiltNet
    elif backbone in {"clip_vit_b32","clip_vit_l14","plip","quilt_b32","quilt_b16"}:
        from transformers import CLIPFeatureExtractor, CLIPVisionModel
        model_id       = accepted_vision_backbones[backbone]
        feat_extractor = CLIPFeatureExtractor.from_pretrained(model_id)
        vision_backbone= CLIPVisionModel.from_pretrained(model_id).eval()
        img_tfms = lambda imgs: feat_extractor(images=imgs, return_tensors="pt")["pixel_values"]
        vision_cfg = None
        mean, std  = feat_extractor.image_mean, feat_extractor.image_std
        for p in vision_backbone.parameters():
            p.requires_grad = False
        print(f"[BACKBONE vsn]\t{backbone} loaded")
        return vision_backbone, vision_cfg, img_tfms, mean, std

    else:
        raise ValueError(f"Unable to load requested vision backbone: {backbone}")
