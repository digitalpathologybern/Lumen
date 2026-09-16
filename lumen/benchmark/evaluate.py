"""Stage B of the zero-shot benchmark: score cached embeddings against a prompt bank.

Runs entirely on the Stage-A embedding caches plus cheap text encoding. For each
(model, dataset) it sweeps the prompt bank (templates × class-name variants),
computes the per-prompt metric distribution, the prompt-ensemble result with a
bootstrap CI, and a **prompt-sensitivity** (std) score. Writes per-(model,dataset)
per-prompt CSVs and a single summary table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lumen.benchmark import prompts as pb
from lumen.benchmark.extract import DEFAULT_OUT
from lumen.benchmark.metrics import (
    bootstrap_ci,
    class_matrix,
    compute_metrics,
    score_logits,
)
from lumen.benchmark.summary_io import merge_rows
from lumen.models.registry import MODELS
from lumen.paths import project_root

DEFAULT_EVAL_OUT = project_root() / "outputs" / "benchmark" / "evaluation"
ENSEMBLE_METRIC_COLUMNS = [
    "balanced_acc",
    "accuracy",
    "quadratic_kappa",
    "macro_f1",
    "macro_precision",
    "macro_sensitivity",
    "macro_auroc",
    "auroc",
    "binary_auroc",
    "binary_0p5_threshold",
    "binary_0p5_accuracy",
    "binary_0p5_f1",
    "binary_0p5_precision",
    "binary_0p5_sensitivity",
    "binary_0p5_specificity",
    "binary_youden_threshold",
    "binary_youden_j",
    "binary_youden_accuracy",
    "binary_youden_f1",
    "binary_youden_precision",
    "binary_youden_sensitivity",
    "binary_youden_specificity",
]


def _load_cache(emb_dir: Path, dataset_key: str, model_key: str):
    npz = emb_dir / dataset_key / f"{model_key}.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    return d["embeddings"].astype(np.float32), d["labels"].astype(np.int64)


# The selectable ensemble banks, keyed by the ``--prompt-style`` value. One
# mapping drives both the CLI choices and the dispatch.
_PROMPT_BANKS = {
    "ours": pb.canonical_prompts,
    "rich": pb.rich_prompts,
    "rich_canonical": pb.rich_canonical_prompts,
}

PROMPT_STYLES = tuple(_PROMPT_BANKS)


def evaluate(model_keys, dataset_keys, emb_dir=DEFAULT_OUT,
             out_dir=DEFAULT_EVAL_OUT, device=None, n_boot=1000,
             primary="balanced_acc", prompt_style="ours"):
    """Stage B over models × datasets. Returns the list of summary rows.

    ``prompt_style`` selects the ensemble bank:

    ``"ours"``
        Six templates over one canonical name per class.
    ``"rich"``
        The 22-template bank expanded over every registered class-name variant,
        giving 22 to 66 prompts per class.
    ``"rich_canonical"``
        The same 22 templates over the canonical name only, a uniform 22 per
        class. Differs from ``"ours"`` in the template set alone, so it isolates
        template richness from the vocabulary change that ``"rich"`` confounds
        with it.
    """
    import torch
    from lumen.models.loaders import load_adapter

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    emb_dir = Path(emb_dir)
    out_dir = Path(out_dir)
    (out_dir / "per_prompt").mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for mkey in model_keys:
        # Which datasets actually have a cache for this model?
        todo = [d for d in dataset_keys if (emb_dir / d / f"{mkey}.npz").exists()]
        if not todo:
            print(f"[skip] {mkey}: no embedding caches found")
            continue

        print(f"\n=== {mkey}: loading adapter for text encoding ===", flush=True)
        adapter = load_adapter(MODELS[mkey], device)

        for dkey in todo:
            image_emb, labels = _load_cache(emb_dir, dkey, mkey)
            print(f"  {mkey} x {dkey}  (n={len(labels)})", flush=True)

            # Prompt-ensemble + bootstrap CI, under the selected protocol.
            ens_set = _PROMPT_BANKS[prompt_style](dkey)
            ens_mat = class_matrix(adapter, ens_set)
            ens_logits = score_logits(image_emb, ens_mat)
            ens = compute_metrics(ens_logits, labels)
            ci_lo, ci_hi = bootstrap_ci(ens_logits, labels, primary, n_boot)

            row = {
                "model": mkey, "dataset": dkey, "n": int(len(labels)),
                # Which ensemble bank produced this row. A score is a property
                # of a model and a prompt protocol jointly, so a summary that
                # does not say which bank it used cannot be compared with one
                # that used another.
                "prompt_style": prompt_style,
                "n_prompts": len(ens_set[next(iter(ens_set))]),
                f"ensemble_{primary}_ci_lo": ci_lo,
                f"ensemble_{primary}_ci_hi": ci_hi,
            }
            for metric_name in ENSEMBLE_METRIC_COLUMNS:
                row[f"ensemble_{metric_name}"] = ens[metric_name]
            summary_rows.append(row)
            threshold_msg = ""
            if not np.isnan(ens["binary_youden_threshold"]):
                threshold_msg = (
                    f"  youden_t={ens['binary_youden_threshold']:.3f}"
                    f" youden_acc={ens['binary_youden_accuracy']:.3f}"
                )
            print(f"     {primary}: "
                  f"ensemble={ens['balanced_acc']:.3f} "
                  f"[{ci_lo:.3f},{ci_hi:.3f}]  "
                  f"auroc={ens['auroc']:.3f}{threshold_msg}", flush=True)

        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write the summary table. Merged on (model, dataset) under an exclusive lock,
    # so evaluating a subset of models tops the table up instead of replacing it
    # and two targeted jobs merge sequentially rather than clobbering each other.
    if summary_rows:
        summ_path = out_dir / "summary.csv"
        summary_rows = merge_rows(summ_path, summary_rows,
                                  key_fields=("model", "dataset"),
                                  sort_fields=("dataset", "model"))
        print(f"\nSummary -> {summ_path}  ({len(summary_rows)} rows)")

    return summary_rows
