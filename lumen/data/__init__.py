"""Dataset registry and loaders for the Lumen framework."""

from lumen.data.loaders import EvalDataset, load_eval_set
from lumen.data.registry import (
    DATASETS,
    SLIDE_TABLES,
    DatasetSpec,
    dataset_dir,
)

__all__ = [
    "EvalDataset", "load_eval_set", "DATASETS", "DatasetSpec", "dataset_dir",
    "SLIDE_TABLES",
]
