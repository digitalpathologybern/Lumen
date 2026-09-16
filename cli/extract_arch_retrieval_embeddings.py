#!/usr/bin/env python3
"""Extract ARCH / ARCH-OPEN image-text retrieval embeddings for all VLMs.

ARCH provides pathology images with dense captions. ARCH-OPEN, from Path-RAG,
adds five synthetic open-ended question-answer pairs per ARCH image/caption. The
script expects the image archives to already be extracted locally, then writes
compact embedding caches and retrieval metrics for each selected model.

Expected input layout::

    assets/datasets/arch/
      book_set/     or books_set/
        images/
        captions.json
      pubmed_set/
        images/
        captions.json

    assets/datasets/path-rag/ARCH-OPEN/
      pubmed_qa_pairs.json
      textbook_qa_pairs.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumen.benchmark.extract import resolve_keys  # noqa: E402
from lumen.models.registry import MODELS  # noqa: E402
from lumen.paths import datasets_root, project_root  # noqa: E402
from lumen.retrieval import normalize_rows  # noqa: E402


DEFAULT_ARCH_ROOT = datasets_root() / "arch"
DEFAULT_ARCH_OPEN_ROOT = datasets_root() / "path-rag" / "ARCH-OPEN"
DEFAULT_OUT = project_root() / "outputs" / "arch_retrieval"
EVAL_SCRIPT = Path(__file__).resolve().with_name("evaluate_retrieval.py")


@dataclass
class RetrievalRow:
    dataset: str
    split: str
    item_id: str
    query_id: str
    image_path: Path
    caption: str
    texts: list[str]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="all",
                    help="model key, comma-list, or 'all'")
    ap.add_argument(
        "--dataset",
        default="all",
        help=(
            "arch_caption, arch_open_qa, arch_open_qa_pooled, "
            "arch_caption_qa_pooled, or all"
        ),
    )
    ap.add_argument("--arch-root", default=str(DEFAULT_ARCH_ROOT))
    ap.add_argument("--arch-open-root", default=str(DEFAULT_ARCH_OPEN_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--image-batch-size", type=int, default=128)
    ap.add_argument("--text-batch-size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    ap.add_argument("--evaluate", action="store_true",
                    help="Run retrieval evaluation after each embedding cache")
    ap.add_argument("--eval-backend", default="numpy", choices=["auto", "numpy", "torch"])
    ap.add_argument("--eval-chunk-size", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: cap rows per dataset")
    ap.add_argument("--skip-missing-images", action="store_true",
                    help="Skip rows whose image file is missing")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build manifests and report counts; do not load models")
    return ap.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_set_dir(root: Path, split: str) -> Path | None:
    candidates = {
        "book": ["book_set", "books_set", "book", "books"],
        "pubmed": ["pubmed_set", "pubmed"],
    }[split]
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return None


def _normalise_caption_records(data) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        rows = []
        for key, value in data.items():
            if isinstance(value, str):
                rows.append({"uuid": key, "caption": value})
            elif isinstance(value, dict):
                row = dict(value)
                row.setdefault("uuid", key)
                rows.append(row)
        return rows
    return []


def _get_text(record: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _find_image(images_dir: Path, item_id: str) -> Path | None:
    raw = Path(item_id)
    candidates = [images_dir / raw.name]
    if raw.suffix:
        candidates.append(images_dir / raw.stem / raw.name)
    else:
        for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            candidates.append(images_dir / f"{item_id}{suffix}")
    for path in candidates:
        if path.is_file():
            return path
    matches = list(images_dir.glob(f"{item_id}.*"))
    return matches[0] if matches else None


def _caption_rows(arch_root: Path, skip_missing: bool) -> list[RetrievalRow]:
    rows: list[RetrievalRow] = []
    for split in ("book", "pubmed"):
        set_dir = _candidate_set_dir(arch_root, split)
        if set_dir is None:
            raise FileNotFoundError(f"Missing ARCH {split} set under {arch_root}")
        captions_path = set_dir / "captions.json"
        images_dir = set_dir / "images"
        if not captions_path.exists() or not images_dir.exists():
            raise FileNotFoundError(f"Expected {captions_path} and {images_dir}")
        for record in _normalise_caption_records(_load_json(captions_path)):
            item_id = _get_text(record, ("uuid", "image_id", "id", "filename", "file_name"))
            caption = _get_text(record, ("caption", "title", "text", "description"))
            if not item_id or not caption:
                continue
            image_path = _find_image(images_dir, item_id)
            if image_path is None:
                if skip_missing:
                    continue
                raise FileNotFoundError(f"Missing image for {split}/{item_id}")
            rows.append(RetrievalRow(
                dataset="arch_caption",
                split=split,
                item_id=item_id,
                query_id=item_id,
                image_path=image_path,
                caption=caption,
                texts=[caption],
            ))
    return rows


def _arch_open_jsons(root: Path) -> list[tuple[str, Path]]:
    return [
        ("pubmed", root / "pubmed_qa_pairs.json"),
        ("book", root / "textbook_qa_pairs.json"),
    ]


def _qa_texts(record: dict) -> list[tuple[int, str]]:
    texts = []
    for idx in range(1, 6):
        question = _get_text(record, (f"Question_{idx}", f"question_{idx}"))
        answer = _get_text(record, (f"Answer_{idx}", f"answer_{idx}"))
        if question and answer:
            texts.append((idx, f"Question: {question} Answer: {answer}"))
    return texts


def _arch_open_rows(
    arch_root: Path,
    arch_open_root: Path,
    skip_missing: bool,
    dataset_key: str,
) -> list[RetrievalRow]:
    rows: list[RetrievalRow] = []
    for split, qa_path in _arch_open_jsons(arch_open_root):
        if not qa_path.exists():
            raise FileNotFoundError(f"Missing ARCH-OPEN QA file: {qa_path}")
        set_dir = _candidate_set_dir(arch_root, split)
        if set_dir is None:
            raise FileNotFoundError(f"Missing ARCH {split} set under {arch_root}")
        images_dir = set_dir / "images"
        for record in _load_json(qa_path):
            if not isinstance(record, dict):
                continue
            item_id = _get_text(record, ("uuid", "image_id", "id", "filename", "file_name"))
            caption = _get_text(record, ("caption", "title", "text", "description"))
            texts = _qa_texts(record)
            if not item_id or not texts:
                continue
            image_path = _find_image(images_dir, item_id)
            if image_path is None:
                if skip_missing:
                    continue
                raise FileNotFoundError(f"Missing image for ARCH-OPEN {split}/{item_id}")
            if dataset_key == "arch_open_qa":
                for qa_idx, qa_text in texts:
                    rows.append(RetrievalRow(
                        dataset=dataset_key,
                        split=split,
                        item_id=item_id,
                        query_id=f"{item_id}::qa{qa_idx}",
                        image_path=image_path,
                        caption=caption,
                        texts=[qa_text],
                    ))
            elif dataset_key == "arch_open_qa_pooled":
                rows.append(RetrievalRow(
                    dataset=dataset_key,
                    split=split,
                    item_id=item_id,
                    query_id=item_id,
                    image_path=image_path,
                    caption=caption,
                    texts=[qa_text for _, qa_text in texts],
                ))
            elif dataset_key == "arch_caption_qa_pooled":
                pooled_texts = ([caption] if caption else []) + [qa_text for _, qa_text in texts]
                if not pooled_texts:
                    continue
                rows.append(RetrievalRow(
                    dataset=dataset_key,
                    split=split,
                    item_id=item_id,
                    query_id=item_id,
                    image_path=image_path,
                    caption=caption,
                    texts=pooled_texts,
                ))
            else:
                raise ValueError(dataset_key)
    return rows


def _resolve_datasets(arg: str) -> list[str]:
    default = ["arch_caption", "arch_open_qa", "arch_caption_qa_pooled"]
    valid = default + ["arch_open_qa_pooled"]
    if arg == "all":
        return default
    keys = [x.strip() for x in arg.split(",") if x.strip()]
    bad = [x for x in keys if x not in valid]
    if bad:
        raise SystemExit(f"Unknown dataset(s) {bad}; available={valid}")
    return keys


def _load_rows(dataset_key: str, args: argparse.Namespace) -> list[RetrievalRow]:
    arch_root = Path(args.arch_root)
    arch_open_root = Path(args.arch_open_root)
    if dataset_key == "arch_caption":
        rows = _caption_rows(arch_root, args.skip_missing_images)
    elif dataset_key in {"arch_open_qa", "arch_open_qa_pooled", "arch_caption_qa_pooled"}:
        rows = _arch_open_rows(arch_root, arch_open_root, args.skip_missing_images, dataset_key)
    else:
        raise ValueError(dataset_key)
    if args.limit is not None:
        rows = rows[:args.limit]
    if len(rows) < 2:
        raise SystemExit(f"{dataset_key}: need at least two rows, found {len(rows)}")
    return rows


def _encode_images(adapter, rows: list[RetrievalRow], batch_size: int) -> np.ndarray:
    import torch

    unique_rows: list[RetrievalRow] = []
    inverse: list[int] = []
    seen: dict[str, int] = {}
    for row in rows:
        key = str(row.image_path)
        if key not in seen:
            seen[key] = len(unique_rows)
            unique_rows.append(row)
        inverse.append(seen[key])

    chunks = []
    batch = []
    done = 0
    t0 = time.time()
    for row in unique_rows:
        with Image.open(row.image_path) as image:
            tensor = adapter.preprocess(image.convert("RGB"))
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        batch.append(tensor)
        if len(batch) >= batch_size:
            chunks.append(adapter.encode_pixels(torch.stack(batch)))
            done += len(batch)
            batch.clear()
            if done <= batch_size or done == len(unique_rows) or (done // batch_size) % 20 == 0:
                rate = done / max(1e-9, time.time() - t0)
                print(f"    images {done}/{len(unique_rows)} ({rate:.0f} img/s)", flush=True)
    if batch:
        chunks.append(adapter.encode_pixels(torch.stack(batch)))
        done += len(batch)
        rate = done / max(1e-9, time.time() - t0)
        print(f"    images {done}/{len(unique_rows)} ({rate:.0f} img/s)", flush=True)
    unique_embeddings = np.concatenate(chunks, axis=0)
    return unique_embeddings[np.asarray(inverse, dtype=np.int64)]


def _encode_texts(adapter, rows: list[RetrievalRow], batch_size: int) -> np.ndarray:
    flat_texts: list[str] = []
    owners: list[int] = []
    for row_idx, row in enumerate(rows):
        for text in row.texts:
            flat_texts.append(text)
            owners.append(row_idx)

    per_row: list[list[np.ndarray]] = [[] for _ in rows]
    done = 0
    t0 = time.time()
    for start in range(0, len(flat_texts), batch_size):
        end = min(start + batch_size, len(flat_texts))
        emb = adapter.encode_text(flat_texts[start:end])
        for local_idx, vector in enumerate(emb):
            per_row[owners[start + local_idx]].append(vector)
        done = end
        if done <= batch_size or done == len(flat_texts) or (done // batch_size) % 20 == 0:
            rate = done / max(1e-9, time.time() - t0)
            print(f"    texts {done}/{len(flat_texts)} ({rate:.0f} text/s)", flush=True)

    pooled = [np.mean(np.stack(parts), axis=0) for parts in per_row]
    return normalize_rows(np.stack(pooled))


def _write_outputs(out_dir: Path, rows: list[RetrievalRow], image: np.ndarray, text: np.ndarray, metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "embeddings.npz",
        image_embeddings=image.astype(np.float16, copy=False),
        text_embeddings=text.astype(np.float16, copy=False),
        ids=np.asarray([row.item_id for row in rows], dtype=str),
        query_ids=np.asarray([row.query_id for row in rows], dtype=str),
        file_ids=np.asarray([row.item_id for row in rows], dtype=str),
        group_ids=np.asarray([row.item_id for row in rows], dtype=str),
        splits=np.asarray([row.split for row in rows], dtype=str),
        n_texts=np.asarray([len(row.texts) for row in rows], dtype=np.int32),
    )
    fields = [
        "pair_index",
        "dataset",
        "split",
        "id",
        "query_id",
        "file_id",
        "group_id",
        "image_path",
        "caption",
        "text",
        "n_texts",
    ]
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow({
                "pair_index": idx,
                "dataset": row.dataset,
                "split": row.split,
                "id": row.item_id,
                "query_id": row.query_id,
                "file_id": row.item_id,
                "group_id": row.item_id,
                "image_path": str(row.image_path),
                "caption": row.caption,
                "text": " || ".join(row.texts),
                "n_texts": len(row.texts),
            })
    metadata = dict(metadata)
    metadata.update({
        "n_pairs": int(image.shape[0]),
        "n_unique_images": int(len({row.item_id for row in rows})),
        "n_text_entries": int(sum(len(row.texts) for row in rows)),
        "embedding_dim": int(image.shape[1]) if image.ndim == 2 else 0,
        "embedding_dtype": "float16",
        "embeddings_npz": str(out_dir / "embeddings.npz"),
        "manifest_csv": str(out_dir / "manifest.csv"),
    })
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _evaluate(model_key: str, dataset_key: str, out_dir: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--model-name", model_key,
        "--embeddings", str(out_dir / "embeddings.npz"),
        "--manifest", str(out_dir / "manifest.csv"),
        "--out-dir", str(out_dir / "evaluation"),
        "--unit", "patch",
        "--group-field", "file_id",
        "--backend", args.eval_backend,
        "--chunk-size", str(args.eval_chunk_size),
        "--overwrite",
    ]
    print(f"[eval] {model_key} x {dataset_key}", flush=True)
    subprocess.run(cmd, check=True)


def _summary(out_root: Path, model_keys: list[str], dataset_keys: list[str]) -> None:
    rows = []
    for dataset_key in dataset_keys:
        for model_key in model_keys:
            path = out_root / dataset_key / model_key / "evaluation" / "metrics.json"
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "dataset": dataset_key,
                "model": model_key,
                "n_pairs": metrics.get("n_pairs"),
                "unique_file_ids": metrics.get("unique_file_ids"),
                "embedding_dim": metrics.get("embedding_dim"),
                "i2t_R@1": metrics["i2t"].get("R@1"),
                "i2t_R@5": metrics["i2t"].get("R@5"),
                "i2t_R@10": metrics["i2t"].get("R@10"),
                "i2t_median_rank": metrics["i2t"].get("median_rank"),
                "t2i_R@1": metrics["t2i"].get("R@1"),
                "t2i_R@5": metrics["t2i"].get("R@5"),
                "t2i_R@10": metrics["t2i"].get("R@10"),
                "t2i_median_rank": metrics["t2i"].get("median_rank"),
                "i2t_same_image_R@1": metrics.get("i2t_same_group", {}).get("R@1"),
                "i2t_same_image_R@5": metrics.get("i2t_same_group", {}).get("R@5"),
                "i2t_same_image_R@10": metrics.get("i2t_same_group", {}).get("R@10"),
                "i2t_same_image_median_rank": metrics.get("i2t_same_group", {}).get("median_rank"),
                "t2i_same_image_R@1": metrics.get("t2i_same_group", {}).get("R@1"),
                "t2i_same_image_R@5": metrics.get("t2i_same_group", {}).get("R@5"),
                "t2i_same_image_R@10": metrics.get("t2i_same_group", {}).get("R@10"),
                "t2i_same_image_median_rank": metrics.get("t2i_same_group", {}).get("median_rank"),
                "matched_cosine_mean": metrics["paired_cosine"].get("mean"),
                "nonmatch_cosine_mean": metrics["random_nonmatch_cosine"].get("mean"),
                "matched_minus_nonmatch_mean": metrics.get("matched_minus_nonmatch_mean"),
            })
    if not rows:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    # Merge, do not replace: this function only iterates the models of the current
    # run, so a targeted `--model keep --evaluate` would otherwise rewrite the file
    # with a single row and silently drop the other twelve.
    from lumen.benchmark.summary_io import merge_rows
    rows = merge_rows(out_root / "retrieval_summary.csv", rows,
                      key_fields=("model", "dataset"),
                      sort_fields=("dataset", "model"))
    (out_root / "retrieval_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        model_keys = resolve_keys(args.model, MODELS)
    except KeyError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    dataset_keys = _resolve_datasets(args.dataset)
    try:
        rows_by_dataset = {key: _load_rows(key, args) for key in dataset_keys}
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    for key, rows in rows_by_dataset.items():
        n_texts = sum(len(row.texts) for row in rows)
        n_unique_images = len({row.item_id for row in rows})
        print(
            f"[dataset] {key}: {len(rows):,} rows, "
            f"{n_unique_images:,} unique images, {n_texts:,} text entries",
            flush=True,
        )
    print(f"[models] {model_keys}", flush=True)
    if args.dry_run:
        return

    import torch
    from lumen.models.loaders import load_adapter

    device = torch.device(args.device) if args.device else None
    out_root = Path(args.out_root)
    for model_key in model_keys:
        print(f"\n=== loading model: {model_key} ===", flush=True)
        t0 = time.time()
        adapter = load_adapter(MODELS[model_key], device)
        print(f"[model] {model_key} loaded on {adapter.device} in {time.time() - t0:.1f}s", flush=True)
        try:
            for dataset_key in dataset_keys:
                out_dir = out_root / dataset_key / model_key
                if (out_dir / "embeddings.npz").exists() and not args.overwrite:
                    print(f"[skip] {dataset_key} x {model_key}: embeddings exist", flush=True)
                    if args.evaluate and not (out_dir / "evaluation" / "metrics.json").exists():
                        _evaluate(model_key, dataset_key, out_dir, args)
                    continue
                rows = rows_by_dataset[dataset_key]
                print(f"[extract] {dataset_key} x {model_key} n={len(rows):,}", flush=True)
                t_ds = time.time()
                image = _encode_images(adapter, rows, args.image_batch_size)
                text = _encode_texts(adapter, rows, args.text_batch_size)
                _write_outputs(
                    out_dir,
                    rows,
                    image,
                    text,
                    {
                        "created": datetime.now().isoformat(timespec="seconds"),
                        "dataset": dataset_key,
                        "model": model_key,
                        "arch_root": str(args.arch_root),
                        "arch_open_root": str(args.arch_open_root),
                        "image_batch_size": int(args.image_batch_size),
                        "text_batch_size": int(args.text_batch_size),
                        "seconds": round(time.time() - t_ds, 1),
                    },
                )
                if args.evaluate:
                    _evaluate(model_key, dataset_key, out_dir, args)
        finally:
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if args.evaluate:
        _summary(out_root, model_keys, dataset_keys)


if __name__ == "__main__":
    main()
