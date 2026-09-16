"""Uniform eval-set loading for the bundled patch datasets.

Every dataset is exposed as an :class:`EvalDataset` with integer labels aligned
to ``spec.class_names`` and lazy per-index image decoding, so callers can batch
over ``range(len(ds))`` regardless of the underlying storage format (flat
parquet, nested-struct parquet, or loose image files + xlsx).
"""

from __future__ import annotations

import glob
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from lumen.data.registry import DatasetSpec, dataset_dir


class EvalDataset:
    """In-memory index over an evaluation split with lazy image decoding."""

    def __init__(self, class_names: list[str], ids: list[str],
                 labels: np.ndarray, image_getter) -> None:
        self.class_names = class_names
        self.ids = ids
        self.labels = labels.astype(np.int64)
        self._get = image_getter
        assert len(ids) == len(labels), (len(ids), len(labels))

    def __len__(self) -> int:
        return len(self.ids)

    def get_image(self, i: int) -> Image.Image:
        return self._get(i)


def _decode(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _sorted_split_files(dir_path: Path, split: str) -> list[str]:
    files = []
    for f in sorted(glob.glob(str(dir_path / "*.parquet"))):
        stem = Path(f).name
        if stem.startswith(f"{split}-") or stem == f"{split}.parquet":
            files.append(f)
    if not files:
        raise FileNotFoundError(f"No parquet files for split={split!r} in {dir_path}")
    return files


def _read_image_struct(files: list[str], extra_cols: list[str]):
    """Read the HF ``image: {bytes, path}`` struct plus ``extra_cols``."""
    tbl = pq.read_table(files, columns=["image", *extra_cols])
    img = tbl.column("image").combine_chunks()
    raw = img.field("bytes")
    paths = img.field("path").to_pylist()
    extra = {c: tbl.column(c).to_pylist() for c in extra_cols}
    return raw, paths, extra


def _load_parquet_flat(spec: DatasetSpec) -> EvalDataset:
    """HF image-classification parquet: image struct + int/bool ``label``."""
    files = _sorted_split_files(dataset_dir(spec), spec.split)
    raw, paths, extra = _read_image_struct(files, ["label"])
    labels = np.array([int(x) for x in extra["label"]])
    ids = [p if p else f"row{i}" for i, p in enumerate(paths)]

    def getter(i: int) -> Image.Image:
        return _decode(raw[i].as_py())

    return EvalDataset(spec.class_names, ids, labels, getter)


def _load_lc25000(spec: DatasetSpec) -> EvalDataset:
    """Nested 'image' struct + (organ, label) ints -> 5-class id.

    organ 0 (lung): label 0/1/2 -> classes 0/1/2
    organ 1 (colon): label 0/1  -> classes 3/4
    """
    files = _sorted_split_files(dataset_dir(spec), spec.split)
    raw, paths, extra = _read_image_struct(files, ["organ", "label"])
    organ = [int(x) for x in extra["organ"]]
    lab = [int(x) for x in extra["label"]]

    def to_class(o: int, l: int) -> int:
        return l if o == 0 else 3 + l

    labels = np.array([to_class(o, l) for o, l in zip(organ, lab)])
    ids = [p if p else f"row{i}" for i, p in enumerate(paths)]

    def getter(i: int) -> Image.Image:
        return _decode(raw[i].as_py())

    return EvalDataset(spec.class_names, ids, labels, getter)


def _load_lc25000_kaggle(spec: DatasetSpec) -> EvalDataset:
    """Load the five class directories in the Kaggle distribution's Test Set."""
    root = dataset_dir(spec)
    class_map = {
        "lung_n": 0,
        "lung_aca": 1,
        "lung_scc": 2,
        "colon_n": 3,
        "colon_aca": 4,
    }
    paths: list[Path] = []
    labels: list[int] = []
    for dirname, label in class_map.items():
        class_dir = root / dirname
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing LC25000 test class directory: {class_dir}")
        images = sorted(p for p in class_dir.iterdir()
                        if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
        paths.extend(images)
        labels.extend([label] * len(images))
    if not paths:
        raise FileNotFoundError(f"No LC25000 test images found in {root}")

    def getter(i: int) -> Image.Image:
        return Image.open(paths[i]).convert("RGB")

    return EvalDataset(
        spec.class_names,
        [str(p.relative_to(root)) for p in paths],
        np.asarray(labels, dtype=np.int64),
        getter,
    )


# SICAP one-hot columns -> class id (G4C cribriform merged into G4).
_SICAP_COLS = ["NC", "G3", "G4", "G5", "G4C"]
_SICAP_TO_CLASS = {"NC": 0, "G3": 1, "G4": 2, "G5": 3, "G4C": 2}


def _load_sicap(spec: DatasetSpec) -> EvalDataset:
    import pandas as pd

    root = dataset_dir(spec)
    split_map = {"test": "partition/Test/Test.xlsx",
                 "train": "partition/Test/Train.xlsx"}
    xlsx = root / split_map[spec.split]
    df = pd.read_excel(xlsx)
    img_dir = root / "images"

    ids = df["image_name"].tolist()
    onehot = df[_SICAP_COLS].to_numpy()
    col_idx = onehot.argmax(axis=1)
    labels = np.array([_SICAP_TO_CLASS[_SICAP_COLS[j]] for j in col_idx])

    def getter(i: int) -> Image.Image:
        return Image.open(img_dir / ids[i]).convert("RGB")

    return EvalDataset(spec.class_names, ids, labels, getter)


def _load_mhist(spec: DatasetSpec) -> EvalDataset:
    """MHIST: loose PNGs + annotations.csv. Binary HP(0)/SSA(1).

    ``split='all'`` uses every annotated image; ``split in {train,test}`` filters
    on the CSV ``Partition`` column.
    """
    import csv

    root = dataset_dir(spec)
    img_dir = root / "images"
    label_map = {"HP": 0, "SSA": 1}
    with (root / "annotations.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if spec.split != "all":
        rows = [r for r in rows if r["Partition"] == spec.split]

    ids = [r["Image Name"] for r in rows]
    labels = np.array([label_map[r["Majority Vote Label"]] for r in rows])

    def getter(i: int) -> Image.Image:
        return Image.open(img_dir / ids[i]).convert("RGB")

    return EvalDataset(spec.class_names, ids, labels, getter)


def _load_databiox(spec: DatasetSpec) -> EvalDataset:
    """Databiox breast IDC grading: loose JPGs under ``BC_IDC_Grade_{1,2,3}/<mag>``.

    ``split`` selects the magnification subfolder (e.g. ``40x``). Grade folder ->
    label id (G1->0, G2->1, G3->2).
    """
    root = dataset_dir(spec)
    grade_dirs = [("BC_IDC_Grade_1", 0), ("BC_IDC_Grade_2", 1),
                  ("BC_IDC_Grade_3", 2)]
    ids: list[str] = []
    labels_list: list[int] = []
    for sub, lab in grade_dirs:
        mag_dir = root / sub / spec.split
        files = sorted(p for p in mag_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for p in files:
            ids.append(str(p))
            labels_list.append(lab)
    labels = np.array(labels_list)

    def getter(i: int) -> Image.Image:
        return Image.open(ids[i]).convert("RGB")

    return EvalDataset(spec.class_names, ids, labels, getter)


_BACH_DIR_LABELS = {
    "normal": 0,
    "benign": 1,
    "insitu": 2,
    "in_situ": 2,
    "in-situ": 2,
    "carcinoma in situ": 2,
    "invasive": 3,
}
_BACH_PREFIX_LABELS = [("is", 2), ("iv", 3), ("b", 1), ("n", 0)]


def _load_bach(spec: DatasetSpec) -> EvalDataset:
    """BACH/ICIAR 2018 microscopy classification images.

    Supports both common extracted layouts: class-named subdirectories
    (Normal/Benign/InSitu/Invasive) and the original flat microscopy filenames
    prefixed as n*, b*, is*, iv*.
    """
    root = dataset_dir(spec)
    candidates = [
        root / "Photos",
        root / "ICIAR2018_BACH_Challenge" / "Photos",
        root / "Microscopy" / "Photos",
        root / "Histology_Dataset",
        root,
    ]
    img_root = next((p for p in candidates if p.exists()), None)
    if img_root is None:
        raise FileNotFoundError(
            f"BACH files not found under {root}. Download/extract "
            "ICIAR2018_BACH_Challenge.zip into this directory.")

    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    ids: list[str] = []
    labels_list: list[int] = []
    for p in sorted(img_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        parent_key = p.parent.name.strip().lower().replace(" ", "_")
        label = _BACH_DIR_LABELS.get(parent_key)
        if label is None:
            stem = p.stem.lower()
            for prefix, lab in _BACH_PREFIX_LABELS:
                if stem.startswith(prefix):
                    label = lab
                    break
        if label is None:
            continue
        ids.append(str(p))
        labels_list.append(label)

    if not ids:
        raise FileNotFoundError(f"No labeled BACH microscopy images found under {img_root}")
    labels = np.array(labels_list)

    def getter(i: int) -> Image.Image:
        return Image.open(ids[i]).convert("RGB")

    return EvalDataset(spec.class_names, ids, labels, getter)


# NCT-CRC-HE-100K / CRC-VAL-HE-7K: class code is the filename prefix before the
# first '-'. Order below == integer label id and must match spec.class_names.
_NCT_CODES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
_NCT_SPLIT_DIRS = {"train": "NCT-CRC-HE-100K-png", "val": "CRC-VAL-HE-7K-png"}


def _load_nct_crc(spec: DatasetSpec) -> EvalDataset:
    """NCT-CRC colorectal tissue: flat PNGs, class = filename prefix.

    ``split`` selects ``train`` (NCT-CRC-HE-100K, 100k), ``val`` (CRC-VAL-HE-7K,
    7180), or ``all`` (both concatenated). Row order is deterministic (sorted
    per split dir) so cached embeddings stay aligned to labels.
    """
    root = dataset_dir(spec)
    code_to_label = {c: i for i, c in enumerate(_NCT_CODES)}
    subdirs = (list(_NCT_SPLIT_DIRS.values()) if spec.split == "all"
               else [_NCT_SPLIT_DIRS[spec.split]])

    ids: list[str] = []
    labels_list: list[int] = []
    for sub in subdirs:
        d = root / sub
        for p in sorted(d.iterdir()):
            if p.suffix.lower() == ".png":
                ids.append(str(p))
                labels_list.append(code_to_label[p.name.split("-", 1)[0]])
    labels = np.array(labels_list)

    def getter(i: int) -> Image.Image:
        return Image.open(ids[i]).convert("RGB")

    return EvalDataset(spec.class_names, ids, labels, getter)


_WSSS_LABEL_COLS = ["label_tumor", "label_stroma", "label_normal"]


def _load_wsss4luad(spec: DatasetSpec) -> EvalDataset:
    """WSSS4LUAD image-level labels, filtered to pure one-hot patches.

    The training parquet has multi-label rows for mixed tissue patches. The
    external classification benchmark reports the 4,693 pure patches only
    (tumor/stroma/normal), so mixed rows are intentionally skipped here.
    """
    files = _sorted_split_files(dataset_dir(spec), spec.split)
    raw, paths, extra = _read_image_struct(files, ["filename", *_WSSS_LABEL_COLS])

    keep: list[int] = []
    labels_list: list[int] = []
    for i, vals in enumerate(zip(*(extra[c] for c in _WSSS_LABEL_COLS))):
        if any(v is None for v in vals):
            continue
        label_vec = [int(v) for v in vals]
        if sum(label_vec) == 1:
            keep.append(i)
            labels_list.append(label_vec.index(1))

    ids = [
        paths[i] or extra["filename"][i] or f"row{i}"
        for i in keep
    ]
    labels = np.array(labels_list)

    def getter(i: int) -> Image.Image:
        return _decode(raw[keep[i]].as_py())

    return EvalDataset(spec.class_names, ids, labels, getter)


_LOADERS = {
    "parquet_flat": _load_parquet_flat,
    "lc25000": _load_lc25000,
    "lc25000_kaggle": _load_lc25000_kaggle,
    "sicap": _load_sicap,
    "mhist": _load_mhist,
    "databiox": _load_databiox,
    "bach": _load_bach,
    "nct_crc": _load_nct_crc,
    "wsss4luad": _load_wsss4luad,
}


def load_eval_set(spec: DatasetSpec) -> EvalDataset:
    """Load the evaluation split for ``spec`` as an :class:`EvalDataset`."""
    if spec.loader not in _LOADERS:
        raise ValueError(f"Unknown loader {spec.loader!r} for dataset {spec.key!r}")
    ds = _LOADERS[spec.loader](spec)
    n_seen = int(ds.labels.max()) + 1 if len(ds) else 0
    if n_seen > spec.num_classes:
        raise ValueError(
            f"{spec.key}: found label id {n_seen - 1} but only "
            f"{spec.num_classes} class names defined")
    return ds
