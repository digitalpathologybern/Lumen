"""Slide-table I/O and label normalisation."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml


# ── Label maps ───────────────────────────────────────────────────────────────

INTERNAL_LABEL_MAP = {0: 0, 1: 1, 0.0: 0, 1.0: 1}

EXTERNAL_LABEL_MAP = {
    0: 0, "0": 0, "neg": 0, "negative": 0,
    1: 1, "1": 1, "pos": 1, "macro": 1, "micro": 1,
    1.0: 1, 0.0: 0,
}


_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: object) -> str:
    """Filesystem-safe form of a slide or group name.

    Embedding caches are written to ``<group>/<slide>.npz`` under this
    transform, so anything that later joins a cache path back to the slide
    table must apply it to *both* sides of the join. Building a lookup from
    raw table names and querying it with a sanitised path stem silently
    mismatches every slide whose name contains a filtered character, which is
    how a patient-level bootstrap degrades into a slide-level one.
    """
    text = str(value).strip() or "unknown"
    return _UNSAFE_RE.sub("_", text)


_BNUMBER_RE = re.compile(r"^(B\d{2,4}\.\d{2,5})", re.IGNORECASE)


def normalize_bnumber(value: object) -> str | None:
    """Return the base pathology accession used as an internal patient proxy.

    Block/stain suffixes such as ``_B_HE`` are removed. Rare patients with
    several consecutive accessions cannot be linked from the available table
    and therefore remain separate clusters.
    """
    if pd.isna(value):
        return None
    match = _BNUMBER_RE.match(str(value).strip().upper())
    return match.group(1) if match else None


def normalize_external_case(dataset: object, case: object, slide: object) -> str:
    """Build a source-scoped external patient/case cluster identifier.

    The gastric table lacks case identifiers, so those records conservatively
    fall back to one cluster per slide.
    """
    source = str(dataset).strip().lower()
    if not pd.isna(case) and str(case).strip():
        return f"{source}::{str(case).strip().upper()}"
    return f"{source}::SLIDE::{Path(str(slide)).stem}"


# ── Config ───────────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ── Slide tables ─────────────────────────────────────────────────────────────

def load_slide_table(path: str | Path) -> pd.DataFrame:
    """Load and normalise an internal, external, or legacy slide table.

    Returns a DataFrame with columns ``[filepath, gt, group, slide]``.
    Rows with unmappable labels (``discard``, ``itc``, ``ask``, …) are dropped.
    """
    df   = pd.read_excel(path)
    cols = set(df.columns)

    if "organ" in cols and "folder" in cols:
        # Internal format
        df["gt"] = df["label"].map(INTERNAL_LABEL_MAP)
        df = df.dropna(subset=["gt", "folder", "file_name"]).copy()
        df["gt"]       = df["gt"].astype(int)
        df["filepath"] = df.apply(
            lambda r: str(Path(str(r["folder"])) / str(r["file_name"])), axis=1
        )
        df["group"] = df["organ"].fillna("unknown")
        df["slide"] = df["file_name"].apply(lambda x: Path(str(x)).stem)
        df["patient_id"] = df["Bnumber"].map(normalize_bnumber)
        # Defensive fallback for malformed/missing accessions: never merge
        # unrelated slides merely because their patient identifier is absent.
        missing = df["patient_id"].isna()
        df.loc[missing, "patient_id"] = (
            df.loc[missing, "group"].astype(str) + "::SLIDE::" +
            df.loc[missing, "slide"].astype(str)
        )

    elif "dataset" in cols and "file_path" in cols:
        # External format
        df["gt"] = df["label"].map(EXTERNAL_LABEL_MAP)
        df = df.dropna(subset=["gt"]).copy()
        df["gt"]       = df["gt"].astype(int)
        df["filepath"] = df["file_path"].astype(str)
        df["group"]    = df["dataset"].fillna("unknown")
        df["slide"]    = df["file_name"].apply(lambda x: Path(str(x)).stem)
        df["patient_id"] = df.apply(
            lambda r: normalize_external_case(r["dataset"], r.get("case"), r["file_name"]),
            axis=1,
        )

    else:
        # Legacy format (merged_output_v2.xlsx)
        df = df[
            (df["consent"].fillna("").str.lower() != "abgelehnt") &
            (df["Topographie"].str.strip() == "Lymphknoten") &
            (df["class"].isin([0.0, 1.0]))
        ].copy()
        df["gt"]       = df["class"].astype(int)
        df["filepath"] = df.apply(
            lambda r: str(Path(str(r["Folder"])) / str(r["Filename"])), axis=1
        )
        df["group"] = "crc"
        df["slide"] = df["Filename"].apply(lambda x: Path(str(x)).stem)
        df["patient_id"] = df["slide"].map(lambda x: f"crc::SLIDE::{x}")

    return df[["filepath", "gt", "group", "slide", "patient_id"]].reset_index(drop=True)
