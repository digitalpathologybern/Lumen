#!/usr/bin/env python3
"""Linear-probe Lumen's vision tower with its LoRA on and off.

Answers two questions the prompt-based benchmark cannot.

1. **How much does the prompt interface cost?** Zero-shot reads the tower through
   text; a linear probe reads it directly, with labels. The gap between them is
   what the prompt-based read-out leaves unrecovered. It is not a fair contest --
   the probe sees labels -- so it is a bound, not a competitor: it says whether
   the information is present at all.

2. **Did aligning the tower to text improve or degrade it?** Lumen's LoRA is
   injected into the attention of all blocks, so its tower *is not* Virchow2 any
   more. The control is exact: ``disable_adapter()`` recovers stock Virchow2 from
   the same checkpoint. Same weights, preprocessing, pooling, probe, folds.

Cross-validation is **grouped by biological source** wherever the source is
recoverable from the sample id (osteosarcoma by case, SICAPv2 and WSSS4LUAD by
slide/WSI, Databiox by specimen). Ungrouped folds let correlated patches from one
slide fall on both sides of the split, so the probe can exploit slide-specific
stain and morphology and the accuracy is optimistic. Where no grouping is
recoverable (PatchCamelyon, LC25000, MHIST, BACH Part A, NCT-CRC), each sample is
its own group, which is ordinary stratified CV; those estimates should be read as
upper bounds.

Feature extraction (GPU) is cached to ``--cache-dir`` so the cross-validation can
be re-run on CPU without re-encoding.

Measurement notes:

* Features are the **vision-tower output** (2560-d = CLS + mean patch tokens),
  *before* the projection head, so the probe measures the visual features, not a
  trained head.
* Feature movement is reported as **cosine** and **norm-relative** change, never a
  bare absolute delta.
* Both towers are probed on **identical folds** (paired). The signed-rank test is
  computed over **datasets** (the independent unit), not over folds, which are
  nested within datasets and share training data.

Examples::

    python cli/probe_vision_tower.py --dataset all
    python cli/probe_vision_tower.py --dataset all --refresh   # re-encode
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.data.loaders import load_eval_set  # noqa: E402
from lumen.data.registry import BENCHMARK_DATASETS  # noqa: E402
from lumen.models.registry import MODELS, PAPER_CHECKPOINT  # noqa: E402
from lumen.paths import project_root

OURS_KEY = PAPER_CHECKPOINT

DEFAULT_OUT = project_root() / "outputs" / "benchmark" / "vision_probe"


def _groups_for(key: str, ids: list[str]) -> np.ndarray | None:
    """Biological-source group per sample, or None if not recoverable.

    Returns an array of group labels the same length as ``ids``. ``None`` means no
    grouping is recoverable from the id and ordinary stratified CV is used.
    """
    def base(s: str) -> str:
        return os.path.basename(str(s))

    if key == "osteo":                     # 'Case-3-A10-...jpg' -> case
        g = [re.match(r"(Case-\d+)", base(s)) for s in ids]
        return np.array([m.group(1) if m else base(s) for m, s in zip(g, ids)])
    if key == "sicap":                     # '16B0003388_Block_...' -> slide
        return np.array([base(s).split("_")[0] for s in ids])
    if key == "wsss4luad":                 # '1003370-...' -> WSI
        return np.array([base(s).split("-")[0] for s in ids])
    if key == "databiox":                  # '01_BC_G1_9057_40x_1.JPG' -> specimen
        return np.array([re.sub(r"_40x_\d+\.\w+$", "", base(s)) for s in ids])
    # pcam / lc25000 / mhist / bach / nct_crc: no shared source recoverable.
    return None


def _tower_features(adapter, model, dataset, batch_size, workers, lora_on,
                    stock=None):
    """2560-d vision-tower features for one dataset, adapted or unadapted.

    ``stock`` supplies the unadapted reference explicitly. A LoRA model can be
    switched off in place via ``disable_adapter()``, but a fully fine-tuned model
    has no adapter to disable -- its pretrained weights are gone -- so the
    reference has to be a freshly loaded backbone.
    """
    import torch
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, i):
            return adapter.preprocess(dataset.get_image(i))

    loader = DataLoader(_DS(), batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    def _forward(px):
        feats = model._img_forward(px)
        return torch.cat([feats[:, 0], feats[:, 1:].mean(dim=1)], dim=-1)

    out = []
    with torch.inference_mode():
        for px in loader:
            px = px.to(adapter.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                if lora_on:
                    f = _forward(px)
                elif stock is not None:
                    feats = stock.forward_features(px)
                    f = torch.cat([feats[:, 0], feats[:, 1:].mean(dim=1)], dim=-1)
                else:
                    with model.img_encoder.disable_adapter():
                        f = _forward(px)
            out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def _feature_shift(f_on: np.ndarray, f_off: np.ndarray) -> dict:
    """How far the LoRA moved the representation, anchored to feature scale."""
    n_on = np.linalg.norm(f_on, axis=1)
    n_off = np.linalg.norm(f_off, axis=1)
    cos = np.sum(f_on * f_off, axis=1) / (n_on * n_off + 1e-12)
    rel = np.linalg.norm(f_on - f_off, axis=1) / (n_off + 1e-12)
    return {
        "cos_mean": float(cos.mean()), "cos_std": float(cos.std()),
        "cos_p05": float(np.percentile(cos, 5)),
        "rel_l2_mean": float(rel.mean()), "rel_l2_std": float(rel.std()),
        "norm_off_mean": float(n_off.mean()), "norm_on_mean": float(n_on.mean()),
        "norm_ratio": float((n_on / (n_off + 1e-12)).mean()),
        "mean_abs_delta": float(np.abs(f_on - f_off).mean()),
        "mean_abs_feature": float(np.abs(f_off).mean()),
    }


def _paired_probe(x_on, x_off, y, groups, folds, seed=0):
    """Probe both towers on the SAME folds; grouped by source when available.

    Returns ``(on_scores, off_scores, k, n_groups, cv_type)``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import (
        GroupKFold, StratifiedGroupKFold, StratifiedKFold,
    )
    from sklearn.preprocessing import normalize

    x_on, x_off = normalize(x_on), normalize(x_off)
    grouped = groups is not None and len(set(groups)) < len(y)

    if grouped:
        n_groups = len(set(groups))
        k = min(folds, n_groups)
        try:
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            splits = list(splitter.split(x_on, y, groups))
            cv_type = "stratified_group"
        except Exception:
            splits = list(GroupKFold(n_splits=k).split(x_on, y, groups))
            cv_type = "group"
    else:
        n_groups = len(y)
        k = min(folds, int(np.bincount(y).min()))
        if k < 2:
            return np.array([]), np.array([]), 0, n_groups, "none"
        splits = list(StratifiedKFold(n_splits=k, shuffle=True,
                                      random_state=seed).split(x_on, y))
        cv_type = "stratified"

    on, off = [], []
    for tr, te in splits:
        for x, acc in ((x_on, on), (x_off, off)):
            clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
            clf.fit(x[tr], y[tr])
            acc.append(balanced_accuracy_score(y[te], clf.predict(x[te])))
    return np.array(on), np.array(off), k, n_groups, cv_type


def _cache_path(cache_dir: Path, key: str, model_key: str) -> Path:
    """Per-(model, dataset) feature cache.

    Keyed by model as well as dataset: these are one model's tower features, and
    keying by dataset alone would hand a different model's features to the probe
    after a model change, with no error.
    """
    return cache_dir / f"{model_key}_{key}_tower_features.npz"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=OURS_KEY,
                    help="registry key of the adapted model to probe")
    ap.add_argument("--dataset", default="all", help="dataset key, comma-list, or 'all'")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--cache-dir", default=str(DEFAULT_OUT / "cache"),
                    help="where extracted tower features are cached")
    ap.add_argument("--refresh", action="store_true",
                    help="re-encode features even if a cache exists")
    args = ap.parse_args()

    keys = (list(BENCHMARK_DATASETS) if args.dataset == "all"
            else [k.strip() for k in args.dataset.split(",")])
    bad = [k for k in keys if k not in BENCHMARK_DATASETS]
    if bad:
        sys.exit(f"ERROR: unknown/excluded datasets {bad}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Only load the model if something actually needs encoding.
    need_encode = args.refresh or any(
        not _cache_path(cache_dir, k, args.model).exists() for k in keys)
    adapter = model = stock_enc = None
    if need_encode:
        import torch

        from lumen.models.loaders import load_adapter
        from lumen.models.registry import model_dir
        from lumen.utils.model import load_model

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        spec = MODELS[args.model]
        adapter = load_adapter(spec, dev)
        mdir = model_dir(spec)
        model, _, _ = load_model(str(mdir / spec.extra["ckpt"]),
                                 str(mdir / spec.extra["config"]), dev)
        model.eval()
        if not hasattr(model.img_encoder, "disable_adapter"):
            # Fully fine-tuned tower: the unadapted reference is stock Virchow2.
            from lumen.utils.backbones import load_vision_backbone
            stock_enc = load_vision_backbone("Virchow2")[0].eval().to(adapter.device)
            for prm in stock_enc.parameters():
                prm.requires_grad = False
            print("[probe] no adapter to disable; using freshly loaded stock "
                  "Virchow2 as the unadapted reference", flush=True)

    rows, fold_rows = [], []
    for key in keys:
        ds_spec = BENCHMARK_DATASETS[key]
        ds = load_eval_set(ds_spec)
        y = np.asarray(ds.labels)
        groups = _groups_for(key, ds.ids)
        cache = _cache_path(cache_dir, key, args.model)

        if cache.exists() and not args.refresh:
            z = np.load(cache)
            f_on, f_off = z["f_on"], z["f_off"]
            print(f"\n=== {key}  (n={len(y)}, cached) ===", flush=True)
        else:
            print(f"\n=== {key}  (n={len(y)}, {ds_spec.num_classes} classes) ===", flush=True)
            f_on = _tower_features(adapter, model, ds, args.batch_size, args.workers, True)
            f_off = _tower_features(adapter, model, ds, args.batch_size, args.workers,
                                    False, stock=stock_enc)
            np.savez_compressed(cache, f_on=f_on, f_off=f_off, y=y)

        shift = _feature_shift(f_on, f_off)
        if shift["mean_abs_delta"] == 0.0:
            sys.exit(f"ERROR: {key}: LoRA on/off features identical.")

        t0 = time.perf_counter()
        on, off, k, n_groups, cv_type = _paired_probe(f_on, f_off, y, groups, args.folds)
        d = on - off
        dt = time.perf_counter() - t0

        print(f"  feature shift: cos(on,off) = {shift['cos_mean']:.4f} "
              f"(5th pct {shift['cos_p05']:.4f}), rel L2 {shift['rel_l2_mean']:.4f}")
        print(f"  CV: {cv_type}, {k}-fold over {n_groups} group(s)"
              f"{' [GROUPED]' if cv_type.endswith('group') else ''}")
        print(f"  probe (paired): stock {off.mean():.4f}+/-{off.std():.4f} | "
              f"lumen {on.mean():.4f}+/-{on.std():.4f}")
        print(f"  per-fold delta = {d.mean():+.4f} +/- {d.std():.4f}  "
              f"folds better: {int((d > 0).sum())}/{len(d)}  [{dt:.0f}s]", flush=True)

        rows.append({"dataset": key, "n": len(y), "n_classes": ds_spec.num_classes,
                     "n_groups": n_groups, "cv_type": cv_type, "folds": k,
                     "probe_balacc_virchow2": off.mean(), "probe_std_virchow2": off.std(),
                     "probe_balacc_lumen_tower": on.mean(),
                     "probe_std_lumen_tower": on.std(),
                     "probe_delta": d.mean(), "probe_delta_std": d.std(),
                     "folds_better": int((d > 0).sum()), **shift})
        for i, (a, b) in enumerate(zip(on, off)):
            fold_rows.append({"dataset": key, "fold": i, "lumen_tower": a,
                              "virchow2": b, "delta": a - b})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("vision_probe.csv", rows), ("vision_probe_folds.csv", fold_rows)):
        with (out_dir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {out_dir / name}  ({len(data)} rows)")

    # The independent unit is the dataset, not the fold: folds are nested within a
    # dataset and share training data. Test the per-dataset mean deltas.
    from scipy.stats import wilcoxon
    ds_d = np.array([r["probe_delta"] for r in rows])
    on_mean = np.array([r["probe_balacc_lumen_tower"] for r in rows])
    off_mean = np.array([r["probe_balacc_virchow2"] for r in rows])
    print(f"\n=== paired over {len(rows)} datasets (the independent unit) ===")
    print(f"  probe mean: stock {off_mean.mean():.4f} | lumen tower {on_mean.mean():.4f}")
    print(f"  per-dataset delta: mean {ds_d.mean():+.4f}, "
          f"range [{ds_d.min():+.4f}, {ds_d.max():+.4f}], "
          f"better: {(ds_d > 0).sum()}/{len(ds_d)}")
    if len(ds_d) >= 2 and np.any(ds_d != 0):
        stat, p = wilcoxon(ds_d)
        print(f"  Wilcoxon signed-rank over datasets: stat={stat:.1f}  p={p:.4g}")
    grouped = [r for r in rows if r["cv_type"].endswith("group")]
    if grouped:
        gon = np.mean([r["probe_balacc_lumen_tower"] for r in grouped])
        print(f"  (grouped-CV datasets only, n={len(grouped)}: "
              f"lumen-tower mean {gon:.4f})")


if __name__ == "__main__":
    main()
