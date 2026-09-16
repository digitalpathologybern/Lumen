"""Chunked image-text retrieval evaluation utilities.

The PathGen retrieval cache stores paired image and caption embeddings in the
same row order. Exact-pair retrieval treats row ``i`` in the opposite modality
as the only positive. Same-slide retrieval is also useful for PathGen because
many patches/captions come from each WSI and near-duplicate descriptions can be
reasonable semantic matches.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_RECALL_K = (1, 5, 10)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return float32 row-normalized embeddings."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _percentile_dict(values: np.ndarray) -> dict[str, float]:
    qs = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p25": float(qs[2]),
        "median": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
    }


def summarize_scores(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        **_percentile_dict(values),
    }


def summarize_ranks(ranks: np.ndarray, recall_k: Iterable[int] = DEFAULT_RECALL_K) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    out = {
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    out.update({f"R@{int(k)}": float(np.mean(ranks <= int(k))) for k in recall_k})
    return out


def load_manifest(path: Path | None, n: int) -> list[dict]:
    if path is None or not path.exists():
        return [{"pair_index": str(i)} for i in range(n)]
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) == n:
                break
    if len(rows) != n:
        raise ValueError(f"Manifest has {len(rows):,} rows but embeddings have {n:,} pairs")
    return rows


def aggregate_embeddings(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    manifest_rows: list[dict],
    field: str = "file_id",
    method: str = "mean",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Aggregate patch-level paired embeddings into one row per group.

    Embeddings are first L2-normalized per patch, then mean-pooled inside each
    group and L2-normalized again. This gives one image vector and one text
    vector per WSI/file_id while keeping row ``i`` aligned across modalities.
    """
    if method != "mean":
        raise ValueError("Only mean aggregation is currently supported")
    if len(manifest_rows) != image_embeddings.shape[0]:
        raise ValueError(
            f"Manifest has {len(manifest_rows):,} rows but embeddings have "
            f"{image_embeddings.shape[0]:,} pairs"
        )
    if not manifest_rows or field not in manifest_rows[0]:
        raise ValueError(f"Aggregation field {field!r} is not present in the manifest")

    image = normalize_rows(image_embeddings)
    text = normalize_rows(text_embeddings)
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for idx, row in enumerate(manifest_rows):
        key = str(row.get(field, ""))
        if not key:
            raise ValueError(f"Manifest row {idx} has an empty {field!r}")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(idx)

    image_out: list[np.ndarray] = []
    text_out: list[np.ndarray] = []
    rows_out: list[dict] = []
    for out_idx, key in enumerate(order):
        idx = np.asarray(groups[key], dtype=np.int64)
        first = manifest_rows[int(idx[0])]
        image_out.append(image[idx].mean(axis=0))
        text_out.append(text[idx].mean(axis=0))
        rows_out.append({
            "pair_index": str(out_idx),
            "group_id": key,
            "file_id": first.get("file_id", key if field == "file_id" else ""),
            "wsi_id": first.get("wsi_id", key if field == "wsi_id" else ""),
            "x": "",
            "y": "",
            "n_patches": str(len(idx)),
            "caption": first.get("caption", ""),
        })

    return (
        normalize_rows(np.vstack(image_out)),
        normalize_rows(np.vstack(text_out)),
        rows_out,
    )


def group_index(values: list[str]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, value in enumerate(values):
        groups[str(value)].append(idx)
    return [np.asarray(groups[str(value)], dtype=np.int64) for value in values]


def build_candidate_pool(
    embeddings: np.ndarray,
    ids: list[str] | None,
    group_ids: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collapse rows that reference the *same* item into one candidate.

    When a dataset stores multiple text rows per image (e.g. several QA pairs
    for one figure), the image side is physically duplicated. Ranking a query
    against those duplicated rows makes near-duplicate slots occupy several
    consecutive ranks, so recall@k jumps in steps of the duplication factor and
    the "same-group" positive is a tie of the exact positive. Both are artefacts.

    Rows sharing a non-empty ``ids`` key are mean-pooled into a single
    L2-normalized candidate; rows with an empty key (or ``ids is None``) stay as
    their own singleton, so this reduces to the identity pool for datasets whose
    images are already unique (PathGen patches, ARCH captions, pooled WSIs).

    Returns the pooled candidate matrix ``[K, D]``, a ``row -> pool index`` map
    of length ``n``, and the group label of each pooled candidate.
    """
    n = embeddings.shape[0]
    if ids is None:
        ids = [str(i) for i in range(n)]
    order: list[str] = []
    key_to_pool: dict[str, int] = {}
    row_to_pool = np.empty(n, dtype=np.int64)
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        key = str(ids[i])
        if key != "" and key in key_to_pool:
            pool_idx = key_to_pool[key]
        else:
            pool_idx = len(order)
            order.append(key)
            if key != "":
                key_to_pool[key] = pool_idx
        row_to_pool[i] = pool_idx
        members[pool_idx].append(i)

    pool = np.empty((len(order), embeddings.shape[1]), dtype=np.float32)
    pool_group = [""] * len(order)
    for pool_idx, rows in members.items():
        pool[pool_idx] = embeddings[rows].mean(axis=0)
        if group_ids is not None:
            pool_group[pool_idx] = str(group_ids[rows[0]])
    return normalize_rows(pool), row_to_pool, pool_group


def _group_pool_targets(group_ids: list[str], pool_group: list[str]) -> list[np.ndarray]:
    """For each query row, the pooled-candidate indices sharing its group."""
    by_group: dict[str, list[int]] = defaultdict(list)
    for pool_idx, group in enumerate(pool_group):
        by_group[group].append(pool_idx)
    lookup = {group: np.asarray(idx, dtype=np.int64) for group, idx in by_group.items()}
    return [lookup[str(group)] for group in group_ids]


def _resolve_backend(backend: str, device: str | None) -> tuple[str, object | None]:
    if backend not in {"auto", "numpy", "torch"}:
        raise ValueError("--backend must be one of auto, numpy, torch")
    if backend == "numpy":
        return "numpy", None
    try:
        import torch
    except ImportError:
        if backend == "torch":
            raise
        return "numpy", None

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if backend == "auto" and torch_device.type != "cuda":
        return "numpy", None
    return "torch", torch_device


def _compute_ranks_numpy(
    query: np.ndarray,
    target: np.ndarray,
    chunk_size: int,
    positive_index: np.ndarray,
    group_targets: list[np.ndarray] | None = None,
    direction: str = "retrieval",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n = query.shape[0]
    target_t = np.ascontiguousarray(target.T)
    ranks = np.empty(n, dtype=np.int32)
    matched = np.empty(n, dtype=np.float32)
    group_ranks = np.empty(n, dtype=np.int32) if group_targets is not None else None
    n_chunks = math.ceil(n / chunk_size)
    t0 = time.time()

    for chunk_idx, start in enumerate(range(0, n, chunk_size), 1):
        end = min(start + chunk_size, n)
        sims = query[start:end] @ target_t
        local = np.arange(end - start)
        pos = sims[local, positive_index[start:end]]
        matched[start:end] = pos
        ranks[start:end] = 1 + np.sum(sims > pos[:, None], axis=1, dtype=np.int32)

        if group_targets is not None and group_ranks is not None:
            for j, idx in enumerate(range(start, end)):
                best_group = float(np.max(sims[j, group_targets[idx]]))
                group_ranks[idx] = 1 + int(np.sum(sims[j] > best_group))

        if chunk_idx == 1 or chunk_idx == n_chunks or chunk_idx % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{direction}] chunk {chunk_idx}/{n_chunks} rows={end:,}/{n:,} ({elapsed:.1f}s)", flush=True)

    return ranks, matched, group_ranks


def _compute_ranks_torch(
    query: np.ndarray,
    target: np.ndarray,
    chunk_size: int,
    device,
    positive_index: np.ndarray,
    group_targets: list[np.ndarray] | None = None,
    direction: str = "retrieval",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import torch

    n = query.shape[0]
    q = torch.from_numpy(np.ascontiguousarray(query)).to(device)
    t = torch.from_numpy(np.ascontiguousarray(target)).to(device)
    t_t = t.T.contiguous()
    pos_index = torch.from_numpy(np.ascontiguousarray(positive_index)).to(device)
    ranks = np.empty(n, dtype=np.int32)
    matched = np.empty(n, dtype=np.float32)
    group_ranks = np.empty(n, dtype=np.int32) if group_targets is not None else None
    group_tensors = (
        [torch.from_numpy(idx).to(device) for idx in group_targets]
        if group_targets is not None
        else None
    )
    n_chunks = math.ceil(n / chunk_size)
    t0 = time.time()

    with torch.no_grad():
        for chunk_idx, start in enumerate(range(0, n, chunk_size), 1):
            end = min(start + chunk_size, n)
            sims = q[start:end] @ t_t
            local = torch.arange(end - start, device=device)
            pos = sims[local, pos_index[start:end]]
            matched[start:end] = pos.detach().cpu().numpy().astype(np.float32, copy=False)
            ranks[start:end] = (
                1 + torch.sum(sims > pos[:, None], dim=1)
            ).detach().cpu().numpy().astype(np.int32, copy=False)

            if group_tensors is not None and group_ranks is not None:
                for j, idx in enumerate(range(start, end)):
                    best_group = torch.max(sims[j, group_tensors[idx]])
                    group_ranks[idx] = int(1 + torch.sum(sims[j] > best_group).item())

            if chunk_idx == 1 or chunk_idx == n_chunks or chunk_idx % 10 == 0:
                elapsed = time.time() - t0
                print(f"[{direction}] chunk {chunk_idx}/{n_chunks} rows={end:,}/{n:,} ({elapsed:.1f}s)", flush=True)

    return ranks, matched, group_ranks


def compute_bidirectional_ranks(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    chunk_size: int = 512,
    recall_k: Iterable[int] = DEFAULT_RECALL_K,
    backend: str = "auto",
    device: str | None = None,
    group_ids: list[str] | None = None,
    image_ids: list[str] | None = None,
    text_ids: list[str] | None = None,
) -> dict:
    """Compute exact-pair and optional same-group bidirectional retrieval.

    Each query is ranked against a *de-duplicated* candidate pool: image→text
    ranks the image against the unique texts, text→image ranks the text against
    the unique images. ``image_ids`` / ``text_ids`` supply the identity keys
    that define "unique" (typically the image path and the caption/query id).
    When every item is already unique (the common one-text-per-image case),
    the pools are the identity and this is exact row-matched retrieval.
    """
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            f"image/text embedding shapes differ: {image_embeddings.shape} vs {text_embeddings.shape}"
        )
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    image = normalize_rows(image_embeddings)
    text = normalize_rows(text_embeddings)
    recall_k = tuple(int(k) for k in recall_k)
    backend_name, backend_device = _resolve_backend(backend, device)

    # image->text ranks images against the unique text pool; text->image ranks
    # texts against the unique image pool. ``*_pos`` maps each query row to its
    # positive candidate in the opposite pool.
    text_pool, text_pos, text_pool_group = build_candidate_pool(text, text_ids, group_ids)
    image_pool, image_pos, image_pool_group = build_candidate_pool(image, image_ids, group_ids)
    i2t_group_targets = (
        _group_pool_targets(group_ids, text_pool_group) if group_ids is not None else None
    )
    t2i_group_targets = (
        _group_pool_targets(group_ids, image_pool_group) if group_ids is not None else None
    )
    print(
        f"[retrieval] backend={backend_name}"
        + (f" device={backend_device}" if backend_device else "")
        + f" text_candidates={text_pool.shape[0]:,} image_candidates={image_pool.shape[0]:,}",
        flush=True,
    )

    if backend_name == "torch":
        i2t_ranks, pair_cosine, i2t_group = _compute_ranks_torch(
            image, text_pool, chunk_size, backend_device, text_pos, i2t_group_targets, "image->text"
        )
        t2i_ranks, pair_cosine_t, t2i_group = _compute_ranks_torch(
            text, image_pool, chunk_size, backend_device, image_pos, t2i_group_targets, "text->image"
        )
    else:
        i2t_ranks, pair_cosine, i2t_group = _compute_ranks_numpy(
            image, text_pool, chunk_size, text_pos, i2t_group_targets, "image->text"
        )
        t2i_ranks, pair_cosine_t, t2i_group = _compute_ranks_numpy(
            text, image_pool, chunk_size, image_pos, t2i_group_targets, "text->image"
        )

    # Numerical roundoff can differ across the two matmul orders; average the
    # diagonal estimates for the reporting statistic.
    pair_cosine = ((pair_cosine.astype(np.float32) + pair_cosine_t.astype(np.float32)) / 2.0).astype(np.float32)
    result = {
        "backend": backend_name,
        "n_image_candidates": int(image_pool.shape[0]),
        "n_text_candidates": int(text_pool.shape[0]),
        "candidate_pool_deduplicated": bool(
            image_pool.shape[0] != image.shape[0] or text_pool.shape[0] != text.shape[0]
        ),
        "i2t_ranks": i2t_ranks,
        "t2i_ranks": t2i_ranks,
        "paired_cosine": pair_cosine,
        "i2t": summarize_ranks(i2t_ranks, recall_k),
        "t2i": summarize_ranks(t2i_ranks, recall_k),
        "paired_cosine_summary": summarize_scores(pair_cosine),
    }
    if i2t_group is not None and t2i_group is not None:
        result.update({
            "i2t_group_ranks": i2t_group,
            "t2i_group_ranks": t2i_group,
            "i2t_same_group": summarize_ranks(i2t_group, recall_k),
            "t2i_same_group": summarize_ranks(t2i_group, recall_k),
        })
    return result


def random_nonmatch_cosine(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    image = normalize_rows(image_embeddings)
    text = normalize_rows(text_embeddings)
    n = image.shape[0]
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, size=n_samples)
    dst = rng.integers(0, n - 1, size=n_samples)
    dst = dst + (dst >= src)
    return np.sum(image[src] * text[dst], axis=1).astype(np.float32, copy=False)


def modality_gap(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> dict[str, float]:
    image = normalize_rows(image_embeddings)
    text = normalize_rows(text_embeddings)
    image_mean = image.mean(axis=0)
    text_mean = text.mean(axis=0)
    return {
        "centroid_l2": float(np.linalg.norm(image_mean - text_mean)),
        "centroid_cosine": float(
            np.dot(image_mean, text_mean)
            / max(np.linalg.norm(image_mean) * np.linalg.norm(text_mean), 1e-12)
        ),
        "image_centroid_norm": float(np.linalg.norm(image_mean)),
        "text_centroid_norm": float(np.linalg.norm(text_mean)),
    }


def flatten_metrics(metrics: dict, prefix: str = "") -> dict[str, float | int | str]:
    flat: dict[str, float | int | str] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
    return flat


def write_metrics_csv(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_metrics(metrics)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(flat):
            writer.writerow({"metric": key, "value": flat[key]})


def write_pair_scores(
    path: Path,
    manifest_rows: list[dict],
    paired_cosine: np.ndarray,
    i2t_ranks: np.ndarray,
    t2i_ranks: np.ndarray,
    i2t_group_ranks: np.ndarray | None = None,
    t2i_group_ranks: np.ndarray | None = None,
) -> None:
    fields = [
        "pair_index",
        "group_id",
        "wsi_id",
        "file_id",
        "n_patches",
        "x",
        "y",
        "matched_cosine",
        "i2t_rank",
        "t2i_rank",
    ]
    if i2t_group_ranks is not None and t2i_group_ranks is not None:
        fields += ["i2t_same_group_rank", "t2i_same_group_rank"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(manifest_rows):
            out = {
                "pair_index": row.get("pair_index", idx),
                "group_id": row.get("group_id", ""),
                "wsi_id": row.get("wsi_id", ""),
                "file_id": row.get("file_id", ""),
                "n_patches": row.get("n_patches", ""),
                "x": row.get("x", ""),
                "y": row.get("y", ""),
                "matched_cosine": float(paired_cosine[idx]),
                "i2t_rank": int(i2t_ranks[idx]),
                "t2i_rank": int(t2i_ranks[idx]),
            }
            if i2t_group_ranks is not None and t2i_group_ranks is not None:
                out["i2t_same_group_rank"] = int(i2t_group_ranks[idx])
                out["t2i_same_group_rank"] = int(t2i_group_ranks[idx])
            writer.writerow(out)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def write_topk_examples(
    out_dir: Path,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    manifest_rows: list[dict],
    i2t_ranks: np.ndarray,
    t2i_ranks: np.ndarray,
    n_examples: int,
    top_k: int,
    seed: int,
) -> None:
    if n_examples <= 0 or top_k <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    image = normalize_rows(image_embeddings)
    text = normalize_rows(text_embeddings)
    text_t = np.ascontiguousarray(text.T)
    image_t = np.ascontiguousarray(image.T)
    n = image.shape[0]
    rng = np.random.default_rng(seed)
    query_indices = np.sort(rng.choice(n, size=min(n_examples, n), replace=False))

    fields = [
        "query_index",
        "query_file_id",
        "query_wsi_id",
        "query_n_patches",
        "query_caption",
        "true_rank",
        "retrieved_rank",
        "retrieved_index",
        "retrieved_file_id",
        "retrieved_wsi_id",
        "retrieved_n_patches",
        "retrieved_caption",
        "score",
        "is_exact_pair",
        "is_same_file",
    ]
    with (out_dir / "topk_image_to_text.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for qidx in query_indices:
            sims = image[qidx] @ text_t
            for rank, ridx in enumerate(_topk_indices(sims, top_k), 1):
                qrow = manifest_rows[int(qidx)]
                rrow = manifest_rows[int(ridx)]
                writer.writerow({
                    "query_index": int(qidx),
                    "query_file_id": qrow.get("file_id", ""),
                    "query_wsi_id": qrow.get("wsi_id", ""),
                    "query_n_patches": qrow.get("n_patches", ""),
                    "query_caption": qrow.get("caption", ""),
                    "true_rank": int(i2t_ranks[int(qidx)]),
                    "retrieved_rank": rank,
                    "retrieved_index": int(ridx),
                    "retrieved_file_id": rrow.get("file_id", ""),
                    "retrieved_wsi_id": rrow.get("wsi_id", ""),
                    "retrieved_n_patches": rrow.get("n_patches", ""),
                    "retrieved_caption": rrow.get("caption", ""),
                    "score": float(sims[int(ridx)]),
                    "is_exact_pair": int(qidx == ridx),
                    "is_same_file": int(qrow.get("file_id", "") == rrow.get("file_id", "")),
                })

    with (out_dir / "topk_text_to_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for qidx in query_indices:
            sims = text[qidx] @ image_t
            for rank, ridx in enumerate(_topk_indices(sims, top_k), 1):
                qrow = manifest_rows[int(qidx)]
                rrow = manifest_rows[int(ridx)]
                writer.writerow({
                    "query_index": int(qidx),
                    "query_file_id": qrow.get("file_id", ""),
                    "query_wsi_id": qrow.get("wsi_id", ""),
                    "query_n_patches": qrow.get("n_patches", ""),
                    "query_caption": qrow.get("caption", ""),
                    "true_rank": int(t2i_ranks[int(qidx)]),
                    "retrieved_rank": rank,
                    "retrieved_index": int(ridx),
                    "retrieved_file_id": rrow.get("file_id", ""),
                    "retrieved_wsi_id": rrow.get("wsi_id", ""),
                    "retrieved_n_patches": rrow.get("n_patches", ""),
                    "retrieved_caption": rrow.get("caption", ""),
                    "score": float(sims[int(ridx)]),
                    "is_exact_pair": int(qidx == ridx),
                    "is_same_file": int(qrow.get("file_id", "") == rrow.get("file_id", "")),
                })


def write_summary_markdown(path: Path, metrics: dict) -> None:
    i2t = metrics["i2t"]
    t2i = metrics["t2i"]
    paired = metrics["paired_cosine"]
    nonmatch = metrics["random_nonmatch_cosine"]
    gap = metrics["modality_gap"]
    unit = metrics.get("retrieval_unit", "patch")
    unit_label = "WSIs" if unit == "wsi" else "Pairs"
    dataset_label = str(metrics.get("source_dataset") or "Text-Image")
    title = (
        f"{dataset_label} WSI-Level Text-Image Retrieval"
        if unit == "wsi"
        else f"{dataset_label} Text-Image Retrieval"
    )
    lines = [
        f"# {title}",
        "",
        f"- {unit_label}: {metrics['n_pairs']:,}",
        f"- Embedding dimension: {metrics['embedding_dim']}",
        f"- Backend: {metrics['backend']}",
        f"- Image -> text: R@1={i2t['R@1']:.4f}, R@5={i2t['R@5']:.4f}, R@10={i2t['R@10']:.4f}, median rank={i2t['median_rank']:.1f}, MRR={i2t['mrr']:.4f}",
        f"- Text -> image: R@1={t2i['R@1']:.4f}, R@5={t2i['R@5']:.4f}, R@10={t2i['R@10']:.4f}, median rank={t2i['median_rank']:.1f}, MRR={t2i['mrr']:.4f}",
        f"- Matched-pair cosine: mean={paired['mean']:.4f}, median={paired['median']:.4f}",
        f"- Random non-match cosine: mean={nonmatch['mean']:.4f}, median={nonmatch['median']:.4f}",
        f"- Matched minus non-match mean cosine: {metrics['matched_minus_nonmatch_mean']:.4f}",
        f"- Modality-gap centroid L2: {gap['centroid_l2']:.4f}; centroid cosine={gap['centroid_cosine']:.4f}",
    ]
    if "i2t_same_group" in metrics and "t2i_same_group" in metrics:
        sg_i = metrics["i2t_same_group"]
        sg_t = metrics["t2i_same_group"]
        group_field = str(metrics.get("group_field") or "group")
        lines += [
            "",
            f"Same-group retrieval treats any row with the same {group_field} as a positive.",
            f"- Image -> text same-group: R@1={sg_i['R@1']:.4f}, R@5={sg_i['R@5']:.4f}, R@10={sg_i['R@10']:.4f}, median rank={sg_i['median_rank']:.1f}",
            f"- Text -> image same-group: R@1={sg_t['R@1']:.4f}, R@5={sg_t['R@5']:.4f}, R@10={sg_t['R@10']:.4f}, median rank={sg_t['median_rank']:.1f}",
        ]
    if unit == "wsi":
        lines += [
            "",
            "Methods note: WSI embeddings are mean-pooled by file_id from all",
            "sampled patch embeddings in each slide, then L2-normalized. The",
            "row-matched WSI image and WSI text centroids are the positives.",
        ]
    else:
        lines += [
            "",
            "Methods note: exact-pair ranks are computed by chunked cosine-similarity",
            "search over all text/image rows; row-matched image-text pairs are the",
            "primary positives and all other rows are negatives.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(out_dir: Path, metrics: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plots] skipped: {exc}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    paired = np.asarray(metrics["_paired_cosine_values"], dtype=np.float32)
    nonmatch = np.asarray(metrics["_random_nonmatch_values"], dtype=np.float32)
    plt.figure(figsize=(6, 4))
    plt.hist(nonmatch, bins=80, alpha=0.55, density=True, label="random non-match")
    plt.hist(paired, bins=80, alpha=0.55, density=True, label="matched pair")
    plt.xlabel("cosine similarity")
    plt.ylabel("density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "cosine_histogram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    for label, ranks in [
        ("image -> text", np.asarray(metrics["_i2t_ranks"])),
        ("text -> image", np.asarray(metrics["_t2i_ranks"])),
    ]:
        x = np.sort(ranks)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=label)
    plt.xscale("log")
    plt.xlabel("exact-pair rank")
    plt.ylabel("cumulative fraction")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "rank_cdf.png", dpi=200)
    plt.close()


def json_ready(metrics: dict) -> dict:
    """Drop raw arrays and convert NumPy scalars for JSON writing."""
    out = {}
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            out[key] = json_ready(value)
        elif isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out
