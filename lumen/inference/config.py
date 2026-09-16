"""Configuration helpers shared by inference CLIs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def apply_inference_overrides(
    cfg: dict,
    *,
    resolution: float | None = None,
) -> dict:
    """Return a config copy with CLI-style overrides applied."""
    out = deepcopy(cfg)
    if resolution is not None:
        out.setdefault("inference", {})["resolution"] = resolution
    return out


def resolve_table_paths(cfg: dict, tables: list[str] | tuple[str, ...] | None = None) -> list[Path]:
    """Resolve CLI table overrides or ``tables`` from config into Paths."""
    selected = tables if tables is not None else cfg.get("tables", [])
    return [Path(p) for p in selected]
