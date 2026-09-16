#!/usr/bin/env python3
"""Check that local .pt checkpoints actually land in the model that loads them.

``_load_open_clip_pt`` builds an architecture by name and points open_clip at a
local ``.pt``. Two things can go wrong silently and both look like a merely
mediocre model rather than an error:

1. **The weights do not land.** A key mismatch (wrapper dict, ``module.``
   prefixes, a renamed tower) can leave tensors at their random init. The test
   is not "did it raise" but "does the loaded tensor differ from a freshly
   initialised one".
2. **The preprocessing is the arch default, not the checkpoint's.** Building
   ``ViT-B-16`` by name yields OpenAI-CLIP normalisation whatever the
   checkpoint was trained with; a wrong mean/std costs accuracy quietly.

Motivation: PathGen-B/16 scores 10--20 points below its published values on
3/3 overlapping datasets, one-directionally, unlike the other baselines.

Reports, per model: missing/unexpected keys, how many tensors are still at
random init, the resolved preprocessing, and the tokenizer context length.

Examples::

    python cli/audit_checkpoint_loading.py
    python cli/audit_checkpoint_loading.py --model pathgen_b16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.models.registry import MODELS, model_dir  # noqa: E402


def _state_dict(obj):
    """Unwrap the common checkpoint containers to a bare tensor dict."""
    for key in ("state_dict", "model", "module"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    return obj


def audit_open_clip(key: str) -> None:
    import open_clip
    import torch

    spec = MODELS[key]
    arch = spec.extra["arch"]
    ckpt_path = model_dir(spec) / spec.extra["ckpt"]
    print(f"\n{'='*72}\n{key}   arch={arch}\n  ckpt={ckpt_path}\n{'='*72}")

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        print(f"  checkpoint top-level keys: {list(raw)[:6]}")
    sd = _state_dict(raw)
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items() if hasattr(v, "shape")}
    print(f"  tensors in checkpoint: {len(sd)}")

    # Reference: the architecture with NO pretrained weights (random init).
    rand, _, _ = open_clip.create_model_and_transforms(arch, pretrained=None)
    rsd = rand.state_dict()
    print(f"  tensors in {arch} skeleton: {len(rsd)}")

    missing = [k for k in rsd if k not in sd]
    unexpected = [k for k in sd if k not in rsd]
    shape_mismatch = [k for k in rsd if k in sd and tuple(rsd[k].shape) != tuple(sd[k].shape)]
    print(f"  missing from ckpt : {len(missing)}" + (f"  e.g. {missing[:3]}" if missing else ""))
    print(f"  unexpected in ckpt: {len(unexpected)}" + (f"  e.g. {unexpected[:3]}" if unexpected else ""))
    print(f"  shape mismatches  : {len(shape_mismatch)}" + (f"  e.g. {shape_mismatch[:3]}" if shape_mismatch else ""))

    # The real test: load the way the benchmark loads, then compare against the
    # random skeleton. Tensors that are still identical never got loaded.
    loaded, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=str(ckpt_path))
    lsd = loaded.state_dict()
    same, diff, skipped = [], [], []
    for k, v in lsd.items():
        if k not in rsd or tuple(rsd[k].shape) != tuple(v.shape):
            skipped.append(k); continue
        (same if torch.equal(v, rsd[k]) else diff).append(k)
    print(f"\n  DID THE WEIGHTS LAND?")
    print(f"    tensors differing from random init : {len(diff)}")
    print(f"    tensors IDENTICAL to random init   : {len(same)}"
          + ("   <== these did not load" if same else "   (good: none stranded)"))
    if same:
        for k in same[:8]:
            print(f"        {k}  {tuple(lsd[k].shape)}")

    print(f"\n  logit_scale: {float(loaded.logit_scale.exp().item()):.3f}")
    norm = [t for t in getattr(preprocess, "transforms", []) if t.__class__.__name__ == "Normalize"]
    if norm:
        print(f"  preprocessing Normalize mean={[round(x,4) for x in norm[0].mean]}")
        print(f"                          std ={[round(x,4) for x in norm[0].std]}")
    res = [t for t in getattr(preprocess, "transforms", []) if t.__class__.__name__ in ("Resize", "CenterCrop")]
    geom = ["{}({})".format(t.__class__.__name__, getattr(t, "size", None)) for t in res]
    print(f"  preprocessing geometry: {geom}")
    tok = open_clip.get_tokenizer(arch)
    print(f"  tokenizer context length: {getattr(tok, 'context_length', 'n/a')}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all",
                    help="model key, comma-list, or 'all' (open_clip_pt models only)")
    args = ap.parse_args()

    targets = [k for k, s in MODELS.items() if s.family == "open_clip_pt"]
    if args.model != "all":
        want = [m.strip() for m in args.model.split(",")]
        bad = [m for m in want if m not in targets]
        if bad:
            sys.exit(f"ERROR: not open_clip_pt models: {bad}. Available: {targets}")
        targets = want

    print(f"auditing open_clip_pt checkpoints: {targets}")
    for key in targets:
        audit_open_clip(key)


if __name__ == "__main__":
    main()
