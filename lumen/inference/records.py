"""Reusable slide-record loading and grouping helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


INT_FIELDS = ("gt", "pred", "n_tiles")
FLOAT_FIELDS = ("max_prob", "avg_prob", "prob")


def normalize_slide_record(row: dict) -> dict:
    """Return a copy of *row* with common numeric fields coerced."""
    out = dict(row)
    for field in INT_FIELDS:
        value = out.get(field)
        if value not in (None, ""):
            out[field] = int(float(value))
    for field in FLOAT_FIELDS:
        value = out.get(field)
        if value not in (None, ""):
            out[field] = float(value)
    return out


def load_slide_records(paths: Iterable[str | Path]) -> list[dict]:
    """Load checkpoint/prediction CSV rows from one or more paths."""
    rows: list[dict] = []
    for path in paths:
        p = Path(path)
        with p.open(newline="") as f:
            rows.extend(normalize_slide_record(row) for row in csv.DictReader(f))
    return rows


def group_records(records: Iterable[dict], key: str = "group") -> dict[str, list[dict]]:
    """Group records by *key*, preserving each record object."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return dict(grouped)


def combined_parent_tag(paths: Iterable[str | Path]) -> str:
    """Build a stable output tag from the unique parent directory names."""
    tags: list[str] = []
    for path in paths:
        tag = Path(path).parent.name
        if tag not in tags:
            tags.append(tag)
    return "_".join(tags) if tags else "records"
