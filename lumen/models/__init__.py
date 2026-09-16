"""Model registry and uniform adapters for the Lumen framework."""

from lumen.models.registry import MODELS, ModelSpec, model_dir

__all__ = ["VLMAdapter", "load_adapter", "MODELS", "ModelSpec", "model_dir"]


def __getattr__(name: str):
    if name == "VLMAdapter":
        from lumen.models.base import VLMAdapter

        return VLMAdapter
    if name == "load_adapter":
        from lumen.models.loaders import load_adapter

        return load_adapter
    raise AttributeError(f"module 'lumen.models' has no attribute {name!r}")
