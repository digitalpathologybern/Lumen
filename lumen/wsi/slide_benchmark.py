"""Slide-level zero-shot benchmark over cached tile embeddings.

This is the WSI analogue of :mod:`lumen.benchmark.evaluate` (the patch-level
Stage B). It runs entirely on the per-slide tile-embedding caches produced by
``cli/extract_wsi_embeddings.py`` (``outputs/wsi_embeddings/<model>/<table>/
<group>/<slide>.npz``) plus cheap text encoding. No slide is re-tiled.

The task is binary lymph-node-metastasis detection (``gt`` 0 = normal, 1 =
metastasis). For each (model, table) it:

* applies three prespecified negative/positive vocabularies through the same six
  templates used by the patch benchmark,
* converts each tile's two cosine similarities to a positive-class softmax
  probability and max-pools that probability over the slide,
* reports the across-prompt **variance** (std = prompt-sensitivity), and
* writes prompt-ensemble slide scores for a separately calibrated, locked
  operating threshold.

Outputs (under ``outputs/wsi_benchmark/`` by default):

* ``per_group/<table>__<model>.csv``: ensemble metrics per organ group.
* ``slide_scores/<table>__<model>.csv``: per-slide ensemble scores (for curves).
* ``summary.csv`` / ``summary.json``: one row per (model, table, pooling):
  across-prompt mean/std/min/max + ensemble metric with CI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from lumen import zeroshot
from lumen.inference.evaluation import compute_threshold_metrics
from lumen.inference.io import load_slide_table, safe_name
from lumen.data.registry import SLIDE_TABLES
from lumen.paths import project_root
from lumen.stats import bootstrap_cis, cluster_resample_indices
from lumen.wsi.cohorts import EXCLUDED_GROUPS
from lumen.wsi.prompts import ensemble_prompts, iter_prompt_pairs

DEFAULT_EMB_DIR = project_root() / "outputs" / "wsi_embeddings"
DEFAULT_OUT = project_root() / "outputs" / "wsi_benchmark"

# The primary MIL score is the maximum tile-level positive-class probability.
# Top-five averaging remains a prespecified secondary output, not a separate
# decision rule in the main experiment.
POOLINGS = ("max", "top5")
TOPK = 5

PRIMARY_POOLING = "max"

# Metrics summarised across the prompt sweep (the std of each is its
# prompt-sensitivity). These Stage-B values use the neutral 0.5 probability
# boundary only as a diagnostic; the reported operating point is learned on the
# internal calibration split by ``cli/evaluate_wsi_locked_threshold.py``.
ENSEMBLE_METRIC_KEYS = ("auroc", "aupr", "accuracy", "balanced_accuracy",
                        "sensitivity", "specificity", "f1", "decision_threshold")
PRIMARY = "auroc"


# ── tile → slide pooling ─────────────────────────────────────────────────────

def _slide_paths(emb_dir: Path, model: str, table: str,
                 only_groups: set[str] | None = None) -> list[Path]:
    return [
        path for path in sorted((emb_dir / model / table).glob("*/*.npz"))
        if path.parent.name not in EXCLUDED_GROUPS
        and (only_groups is None or path.parent.name in only_groups)
    ]


def score_slides(emb_dir: Path, model: str, table: str,
                 text_mat: np.ndarray, n_pairs: int,
                 only_groups: set[str] | None = None):
    """Score every cached slide against all prompt pairs + the ensemble.

    ``text_mat`` is ``[2 * n_pairs + 2, D]`` L2-normalized rows ordered
    ``neg0, pos0, neg1, pos1, ..., ens_neg, ens_pos`` (the last two are the
    prompt-ensemble class vectors).

    Returns ``(slides, groups, patient_ids, gts, pair_scores, ens_scores)`` where
    ``pair_scores[pool]`` is ``[S, n_pairs]`` and ``ens_scores[pool]`` is ``[S]``.
    """
    paths = _slide_paths(emb_dir, model, table, only_groups)
    slides, groups, patient_ids, gts = [], [], [], []
    patient_map: dict[tuple[str, str], str] = {}
    table_path = SLIDE_TABLES.get(table)
    if table_path is not None and Path(table_path).exists():
        meta = load_slide_table(table_path)
        # Cache paths are written under ``safe_name`` (see
        # ``cli/extract_wsi_embeddings.py``), so the key must be built under the
        # same transform that the lookup below applies. Keying on the raw table
        # names instead silently drops every slide whose name contains a
        # filtered character, and the per-slide fallback then turns a
        # patient-level bootstrap into a slide-level one.
        patient_map = {
            (safe_name(row.group), safe_name(row.slide)): str(row.patient_id)
            for row in meta.itertuples(index=False)
        }
    pair_scores = {p: [] for p in POOLINGS}
    ens_scores = {p: [] for p in POOLINGS}

    for npz_path in paths:
        try:
            with np.load(npz_path) as d:
                emb = d["embeddings"].astype(np.float32)     # [N, D]
                gt = int(d["gt"]) if "gt" in d else -1
        except Exception:
            continue
        if emb.ndim != 2 or emb.shape[0] == 0:
            continue

        # Cosine similarities to every prompt vector at once: [N, 2P+2].
        # Softmax is applied within each negative/positive pair for every tile,
        # then the positive probability is pooled across tiles.  Scale=1 is the
        # patch-benchmark convention and preserves the legacy WSI score.
        sims = emb @ text_mat.T
        n_tiles = emb.shape[0]
        k = min(TOPK, n_tiles)

        pair = sims[:, :2 * n_pairs].reshape(n_tiles, n_pairs, 2)
        ens = sims[:, 2 * n_pairs:2 * n_pairs + 2]           # [N, 2]
        pair_pos = zeroshot.softmax(pair.reshape(-1, 2)).reshape(
            n_tiles, n_pairs, 2
        )[:, :, 1]
        ens_pos = zeroshot.softmax(ens)[:, 1]

        pair_scores["max"].append(pair_pos.max(axis=0))
        pair_scores["top5"].append(np.sort(pair_pos, axis=0)[-k:].mean(axis=0))
        ens_scores["max"].append(float(ens_pos.max()))
        ens_scores["top5"].append(float(np.sort(ens_pos)[-k:].mean()))

        slides.append(npz_path.stem)
        groups.append(npz_path.parent.name)
        # The fallback must be namespaced by cohort as well as table: two
        # cohorts can legitimately contain a slide with the same stem, and
        # merging them would place two unrelated patients in one cluster.
        patient_ids.append(
            patient_map.get(
                (npz_path.parent.name, npz_path.stem),
                f"{table}::{npz_path.parent.name}::SLIDE::{npz_path.stem}",
            )
        )
        gts.append(gt)

    # A fallback identifier is legitimate only where the source table carries no
    # case column (the public gastric cohort). Anywhere else it means the join
    # failed, and a failed join silently costs the analysis its clustering, so
    # report it loudly rather than letting the bootstrap run on slides.
    if patient_map:
        unmatched = [
            (g, s) for g, s in zip(groups, slides)
            if (g, s) not in patient_map
        ]
        if unmatched:
            by_group: dict[str, int] = {}
            for g, _ in unmatched:
                by_group[g] = by_group.get(g, 0) + 1
            detail = ", ".join(f"{g}={n}" for g, n in sorted(by_group.items()))
            print(
                f"[warn] {model}/{table}: {len(unmatched)} of {len(slides)} slides "
                f"did not join to a patient identifier and fall back to one "
                f"cluster per slide ({detail}). Any patient-level bootstrap over "
                f"these cohorts is a slide-level bootstrap."
            )

    gts = np.asarray(gts, dtype=np.int64)
    pair_scores = {p: np.asarray(v, dtype=np.float64) for p, v in pair_scores.items()}
    ens_scores = {p: np.asarray(v, dtype=np.float64) for p, v in ens_scores.items()}
    return slides, groups, patient_ids, gts, pair_scores, ens_scores


# ── metrics ──────────────────────────────────────────────────────────────────

def _metrics(gts: np.ndarray, scores: np.ndarray) -> dict:
    """Diagnostic metrics at the neutral positive-probability boundary 0.5."""
    records = [{"gt": int(g), "score": float(s)} for g, s in zip(gts, scores)]
    m = compute_threshold_metrics(records, "score", threshold=0.5)
    m["balanced_accuracy"] = 0.5 * (m["sensitivity"] + m["specificity"])
    m.pop("youden_thresh")
    m["decision_threshold"] = 0.5
    m.pop("youden_j", None)
    return m


def _auroc_fixed_acc(gts: np.ndarray,
                     scores: np.ndarray) -> tuple[float, float, float]:
    """Fast (AUROC, accuracy, balanced accuracy) at probability 0.5."""
    if len(np.unique(gts)) < 2:
        return float("nan"), float("nan"), float("nan")
    auroc = float(roc_auc_score(gts, scores))
    pred = (scores >= 0.5).astype(int)
    pos = gts == 1
    neg = ~pos
    bal = 0.5 * (float(pred[pos].mean()) + float((pred[neg] == 0).mean()))
    return auroc, float((pred == gts).mean()), bal


def bootstrap_ci(gts: np.ndarray, scores: np.ndarray, cluster_ids: np.ndarray,
                 n_boot: int = 1000, seed: int = 0) -> dict:
    """Clustered 95% diagnostic CI at the fixed 0.5 probability boundary.

    Internal clusters are normalized B-number patient proxies; external
    clusters are source-scoped case identifiers. All slides in a sampled
    patient/case are retained together. The three intervals come from one
    shared set of resamples, so they describe the same patients.
    """
    if len(np.asarray(cluster_ids)) != len(gts):
        raise ValueError("cluster_ids must align with gts and scores")

    def at(idx, which):
        return _auroc_fixed_acc(gts[idx], scores[idx])[which]

    cis = bootstrap_cis(
        {"auroc": lambda i: at(i, 0),
         "accuracy": lambda i: at(i, 1),
         "balanced_accuracy": lambda i: at(i, 2)},
        np.asarray(cluster_ids, dtype=str),
        n_boot=n_boot, seed=seed, require="auroc")
    return {f"{k}_ci_{side}": v[j]
            for k, v in cis.items() for j, side in enumerate(("lo", "hi"))}
# ── driver ───────────────────────────────────────────────────────────────────

def _build_text_matrix(adapter, prompt_rows=None, vocabulary=None):
    """Encode prompt pairs + the ensemble into one L2-normalized matrix.

    Layout: ``[neg0, pos0, ..., neg{P-1}, pos{P-1}, ens_neg, ens_pos]``.
    The two ensemble rows are computed with exactly the patch-benchmark rule:
    average the normalized prompt embeddings for a class, then L2-normalize the
    mean with :func:`lumen.zeroshot.class_matrix`.

    ``vocabulary`` selects the class-name bank and must be the same one the
    ``prompt_rows`` came from. The ensemble used to be built from the default
    bank unconditionally, so passing rows from another vocabulary would have
    scored per-prompt metrics on one bank and the ensemble on a different one.
    """
    rows = list(prompt_rows or iter_prompt_pairs(vocabulary=vocabulary))
    prompts = [p for row in rows for p in row[2:4]]       # neg0,pos0,neg1,...
    vecs = adapter.encode_text(prompts).astype(np.float32)  # [2P, D]
    ens = zeroshot.class_matrix(
        adapter.encode_text, ensemble_prompts(vocabulary=vocabulary))  # [2, D]
    return np.concatenate([vecs, ens], axis=0)


def evaluate(model_keys, table_keys, emb_dir=DEFAULT_EMB_DIR, out_dir=DEFAULT_OUT,
             device=None, n_boot=1000, only_groups=None, vocabulary=None):
    """Run the slide-level benchmark. Returns summary rows.

    ``only_groups`` restricts scoring to the named cohorts (e.g. the melanoma
    cohorts for a prompt-sensitivity check); when set, write to a scratch
    ``out_dir`` so the full-benchmark outputs are not overwritten.

    ``vocabulary`` names the class-name bank (see
    :data:`lumen.wsi.prompts.VOCABULARIES`). It defaults to the shared bank
    that every reported number uses. ``melanoma_agnostic`` replaces "metastatic
    carcinoma" with histology-neutral wording, which matters because two
    external cohorts are melanoma and a carcinoma is not what they contain.
    """
    import torch

    from lumen.models.loaders import load_adapter
    from lumen.models.registry import MODELS

    emb_dir = Path(emb_dir)
    out_dir = Path(out_dir)
    for sub in ("per_group", "slide_scores"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    prompt_rows = list(iter_prompt_pairs(vocabulary=vocabulary))
    n_pairs = len(prompt_rows)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    summary_rows: list[dict] = []
    for mkey in model_keys:
        todo = [t for t in table_keys if (emb_dir / mkey / t).is_dir()]
        if not todo:
            print(f"[skip] {mkey}: no caches under {emb_dir/mkey}", flush=True)
            continue

        print(f"\n=== {mkey}: loading adapter for text encoding ===", flush=True)
        adapter = load_adapter(MODELS[mkey], device)
        text_mat = _build_text_matrix(adapter, prompt_rows, vocabulary=vocabulary)

        for tkey in todo:
            slides, groups, patient_ids, gts, pair_scores, ens_scores = score_slides(
                emb_dir, mkey, tkey, text_mat, n_pairs, only_groups)
            if len(slides) == 0:
                continue
            groups_arr = np.asarray(groups)
            patients_arr = np.asarray(patient_ids)

            n = len(slides)
            n_pos = int((gts == 1).sum())
            print(f"  {mkey} x {tkey}: {n} slides "
                  f"(pos={n_pos}, neg={n - n_pos})", flush=True)
            if n == 0:
                continue

            # per-group ensemble metrics + per-slide ensemble scores.
            group_rows, slide_rows = [], []
            for s, g, patient_id, gt in zip(slides, groups, patient_ids, gts):
                j = len(slide_rows)
                slide_rows.append({
                    "table": tkey, "model": mkey, "slide": s, "group": g,
                    "patient_id": patient_id, "gt": gt,
                    "ens_max_prob_pos": ens_scores["max"][j],
                    "ens_top5_prob_pos": ens_scores["top5"][j],
                })
            _write_csv(out_dir / "slide_scores" / f"{tkey}__{mkey}.csv", slide_rows)

            # Per organ: the prompt-ensemble point estimate.
            for pool in POOLINGS:
                for grp in sorted(set(groups)):
                    mask = groups_arr == grp
                    g_gts = gts[mask]
                    ens_g = _metrics(g_gts, ens_scores[pool][mask])
                    row = {
                        "table": tkey, "model": mkey, "pooling": pool, "group": grp,
                        "n": ens_g["n"], "n_pos": ens_g["n_pos"],
                        "n_neg": ens_g["n_neg"],
                    }
                    for k in ENSEMBLE_METRIC_KEYS:
                        row[f"ensemble_{k}"] = ens_g.get(k, float("nan"))
                    group_rows.append(row)
            _write_csv(out_dir / "per_group" / f"{tkey}__{mkey}.csv", group_rows)

            # summary row per pooling: ensemble + CI.
            for pool in POOLINGS:
                ens_m = _metrics(gts, ens_scores[pool])
                ci = bootstrap_ci(
                    gts, ens_scores[pool], patients_arr, n_boot=n_boot
                )
                row = {
                    "model": mkey, "table": tkey, "pooling": pool,
                    "primary": int(pool == PRIMARY_POOLING),
                    "prompt_protocol": "3_variants_x_6_templates",
                    "score": "pooled_positive_class_probability",
                    "n": n, "n_pos": n_pos, "n_neg": n - n_pos,
                    "n_patients": int(len(np.unique(patients_arr))),
                    "n_prompts": n_pairs,
                }
                for k in ENSEMBLE_METRIC_KEYS:
                    row[f"ensemble_{k}"] = ens_m.get(k, float("nan"))
                row.update({
                    "ensemble_auroc_ci_lo": ci["auroc_ci_lo"],
                    "ensemble_auroc_ci_hi": ci["auroc_ci_hi"],
                    "ensemble_accuracy_ci_lo": ci["accuracy_ci_lo"],
                    "ensemble_accuracy_ci_hi": ci["accuracy_ci_hi"],
                    "ensemble_balanced_accuracy_ci_lo": ci["balanced_accuracy_ci_lo"],
                    "ensemble_balanced_accuracy_ci_hi": ci["balanced_accuracy_ci_hi"],
                })
                summary_rows.append(row)
                print(f"     [{pool:4s}] "
                      f"ensemble auroc={ens_m['auroc']:.3f} "
                      f"[{ci['auroc_ci_lo']:.3f},{ci['auroc_ci_hi']:.3f}]  "
                      f"acc@0.5={ens_m['accuracy']:.3f}", flush=True)

        del adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if summary_rows:
        _write_csv(out_dir / "summary.csv", summary_rows)
        (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
        print(f"\nSummary -> {out_dir/'summary.csv'}  ({len(summary_rows)} rows)")
    return summary_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Merge ``rows`` into ``path``, keyed on (model, table).

    Was a bare truncating write, so ``--model keep`` replaced a 12-model summary
    with one row and nothing errored. Keyed on (model, table) because this file
    holds one row per model *per slide table* (internal/external).
    """
    if not rows:
        return
    from lumen.benchmark.summary_io import merge_rows

    keys = ("model", "table") if "table" in rows[0] else ("model",)
    merge_rows(path, rows, key_fields=keys, sort_fields=keys)
