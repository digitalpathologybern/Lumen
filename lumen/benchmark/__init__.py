"""Central zero-shot benchmark: cache embeddings (Stage A), sweep prompts (Stage B).

Stage A (:mod:`lumen.benchmark.extract`) encodes each eval image once per
model. Stage B (:mod:`lumen.benchmark.evaluate`) scores those caches against a
prompt bank (:mod:`lumen.benchmark.prompts`) using
:mod:`lumen.benchmark.metrics`.
"""

from lumen.benchmark.evaluate import evaluate
from lumen.benchmark.extract import extract

__all__ = ["extract", "evaluate"]
