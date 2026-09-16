"""Inference pipeline package.

Public API:
    apply_inference_overrides, resolve_table_paths   config/CLI helpers
    load_yaml, safe_name, load_slide_table           table I/O
    load_slide_records, group_records                prediction records
    compute_probability_metrics                      slide-level metrics
"""

from .config import apply_inference_overrides, resolve_table_paths
from .evaluation import (
    compute_probability_metrics,
    compute_threshold_metrics,
    summarize_thresholds_by_group,
)
from .io import load_yaml, safe_name, load_slide_table
from .records import (
    load_slide_records,
    normalize_slide_record,
    group_records,
    combined_parent_tag,
)

__all__ = [
    "apply_inference_overrides", "resolve_table_paths",
    "compute_probability_metrics", "compute_threshold_metrics",
    "summarize_thresholds_by_group",
    "load_yaml", "safe_name", "load_slide_table",
    "load_slide_records", "normalize_slide_record", "group_records",
    "combined_parent_tag",
]
