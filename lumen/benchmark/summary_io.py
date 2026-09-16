"""Merge-on-write for per-model summary tables.

Stages can run for a subset of models (``--model keep``), and a plain
``csv.DictWriter`` would replace the shared summary file, dropping every model
the current run did not compute. :func:`merge_rows` takes an exclusive lock,
re-reads the file after acquiring it, drops only the rows the new run
recomputed, and concatenates. Concurrent targeted jobs merge instead of
clobbering.
"""

from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path
from typing import Sequence


def merge_rows(out_path: Path, rows: list[dict],
               key_fields: Sequence[str] = ("model",),
               sort_fields: Sequence[str] | None = None) -> list[dict]:
    """Merge ``rows`` into the summary at ``out_path``; return the merged rows.

    ``key_fields`` identifies a row: existing rows matching a new row's key are
    replaced, all others are preserved. A sibling ``.json`` is written alongside.
    """
    import pandas as pd

    if not rows:
        return rows

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_name(f".{out_path.stem}.lock")
    key_fields = list(key_fields)
    sort_fields = list(sort_fields or key_fields)

    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out_path.exists():
            old = pd.read_csv(out_path)
            new = pd.DataFrame(rows)
            if all(k in old.columns for k in key_fields):
                keys = set(map(tuple, new[key_fields].astype(str).values))
                mask = old[key_fields].astype(str).apply(tuple, axis=1).isin(keys)
                old = old[~mask]
                merged = pd.concat([old, new], ignore_index=True, sort=False)
                have = [c for c in sort_fields if c in merged.columns]
                if have:
                    merged = merged.sort_values(have).reset_index(drop=True)
                rows = merged.where(pd.notna(merged), None).to_dict("records")

        with out_path.open("w", newline="") as f:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        out_path.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=str))

    return rows
