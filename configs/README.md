# Configuration

| File | Read by | Holds |
|---|---|---|
| `inference.yaml` | `cli/extract_wsi_embeddings.py` (`--inference`) | Whole-slide tiling and the slide tables |
| `quilt_clean.json` | `cli/train_lumen.py` (`--config`) | Training hyperparameters for the QUILT-1M run |

Models are not selected here. They come from the registry
(`lumen/models/registry.py`) via `--model`, defaulting to `PAPER_CHECKPOINT`.

Dataset locations are not selected here either. The slide tables
`inference.yaml` names are not part of this repository, and the slides they
index are located by their `file_path` column; see
[`docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md). The training corpus resolves
through `$QUILT_LMDB`, which is why `quilt_clean.json` carries a placeholder.
