"""Lumen: a pathology vision-language framework.

One interface over many zero-shot pathology VLMs (CLIP, PLIP, QuiltNet,
PathGen, PathCLIP, BiomedCLIP, CONCH) and the Lumen LoRA finetune. See
:data:`lumen.models.MODELS` and :data:`lumen.data.DATASETS`.

    from lumen.encode import Encoder
    emb = Encoder("conch").encode_images([pil_image])
"""

__all__ = ["Encoder"]
__version__ = "0.1.0"

import os as _os
import sys as _sys

# Set before any tokenizer is constructed, which is why it lives here rather
# than in the shell. setdefault, so an explicit value still wins.
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

#: Imports that decide what the numbers are. If any of these resolve to the
#: user site directory, environment.yml is not describing the run.
_PINNED_BY_ENV = ("torch", "numpy", "transformers", "timm", "peft", "PIL",
                  "safetensors", "open_clip")


def require_environment() -> None:
    """Raise if a dependency was actually imported from the user site directory.

    :func:`_warn_if_user_site_shadows` runs at import and can only say that
    shadowing is possible, because what gets imported happens later. This runs
    once the imports are done and reads their ``__file__``, so it reports what
    did happen rather than what might. Called from
    :func:`lumen.models.loaders.load_adapter`, which every evaluation goes
    through.
    """
    if _sys.flags.no_user_site:
        return
    import site

    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if not user_site:
        return
    user_site = _os.path.realpath(user_site)

    shadowed = []
    for name in _PINNED_BY_ENV:
        module = _sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if origin and _os.path.realpath(origin).startswith(user_site):
            shadowed.append(f"{name} ({origin})")
    if shadowed:
        listed = "\n  ".join(shadowed)
        raise RuntimeError(
            "these packages were imported from your user site directory, not "
            f"from the conda environment:\n  {listed}\n\n"
            "environment.yml does not describe this run, so the results are not "
            "comparable with the reported ones. Re-run with PYTHONNOUSERSITE=1 "
            "(or python -s). An activate.d hook makes it stick."
        )


def _warn_if_user_site_shadows() -> None:
    """Warn when ~/.local packages can outrank the environment's own.

    PYTHONNOUSERSITE has to be set before the interpreter starts, so this cannot
    fix it from here. It can say so, which beats finding out from a version
    mismatch three stages into a run.
    """
    if _sys.flags.no_user_site:
        return
    import site

    try:
        user_site = site.getusersitepackages()
    except Exception:
        return
    if not user_site or user_site not in _sys.path:
        return
    try:
        present = {n.split("-")[0].split(".")[0] for n in _os.listdir(user_site)}
    except OSError:
        return
    clashes = sorted(set(_PINNED_BY_ENV) & present)
    if not clashes:
        return

    import warnings

    warnings.warn(
        f"{user_site} is on sys.path and provides {', '.join(clashes)}, which "
        f"can take precedence over this conda environment. Results are then not "
        f"the environment.yml ones. Start Python with PYTHONNOUSERSITE=1 (or "
        f"python -s); an activate.d hook makes it stick.",
        RuntimeWarning, stacklevel=2,
    )


_warn_if_user_site_shadows()


def __getattr__(name: str):
    if name == "Encoder":
        from lumen.encode import Encoder

        return Encoder
    raise AttributeError(f"module 'lumen' has no attribute {name!r}")
