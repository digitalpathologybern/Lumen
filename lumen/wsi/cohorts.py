"""Cohort-scope rules for the lymph-node WSI benchmark."""

from __future__ import annotations

import pandas as pd


# TCGA melanoma is not part of this benchmark.  Keep this explicit and shared
# so extraction, scoring, and locked-threshold evaluation cannot drift.
EXCLUDED_GROUPS = frozenset({"melanoma_tcga"})


def exclude_benchmark_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Return benchmark rows after removing cohorts outside the study scope."""
    if "group" not in frame:
        return frame.copy()
    return frame.loc[~frame["group"].astype(str).isin(EXCLUDED_GROUPS)].copy()
