"""Canonical filesystem roots for the Lumen framework.

Everything that needs to resolve bundled assets (model checkpoints, public
datasets) goes through here so there is a single place that knows the repo
layout.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repo root (the dir that contains ``lumen/`` and ``assets/``)."""
    return Path(__file__).resolve().parents[1]


def assets_root() -> Path:
    return project_root() / "assets"


def models_root() -> Path:
    return assets_root() / "models"


def datasets_root() -> Path:
    return assets_root() / "datasets"
