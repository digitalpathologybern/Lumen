"""Registry of datasets the framework knows about.

Two kinds:

* :data:`DATASETS`: public, patch-level, bundled under ``assets/datasets``.
  Used by the benchmark; loaded eagerly by :func:`lumen.data.loaders.load_eval_set`.
* :data:`SLIDE_TABLES`: the two whole-slide tables, read with
  :func:`lumen.inference.io.load_slide_table`. Neither the tables nor the
  slides are in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lumen.paths import datasets_root, project_root


# ─────────────────────────────────────────────────────────────────────────────
# Public patch-classification datasets (bundled)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetSpec:
    """One bundled evaluation dataset.

    ``loader`` names the reader in :mod:`lumen.data.loaders`. ``class_names``
    is ordered by integer label id (``class_names[label]`` is the canonical
    class token used in prompts). ``rel_dir`` is relative to ``assets/datasets``.
    """

    key: str
    loader: str
    rel_dir: str
    split: str
    class_names: list[str]
    notes: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


DATASETS: dict[str, DatasetSpec] = {
    "lc25000": DatasetSpec(
        key="lc25000", loader="lc25000_kaggle",
        rel_dir="lc25000_kaggle/lung_colon_image_set/Test Set", split="test",
        class_names=[
            "benign lung tissue",
            "lung adenocarcinoma",
            "lung squamous cell carcinoma",
            "benign colon tissue",
            "colon adenocarcinoma",
        ],
        notes=("Predefined Kaggle-distribution Test Set, 5-class lung+colon. "
               "Not established as patient-independent; augmented derivatives may be correlated."),
    ),
    "osteo": DatasetSpec(
        key="osteo", loader="parquet_flat", rel_dir="osteo_aug_parquet/data",
        split="test",
        class_names=["non-tumor tissue", "necrotic tumor", "viable tumor"],
        notes="283 patches, 3-class osteosarcoma (Non-Tumor / Non-Viable / Viable).",
    ),
    "pcam": DatasetSpec(
        key="pcam", loader="parquet_flat", rel_dir="pcam_patchcamelyon/data",
        split="test",
        class_names=["normal lymph node tissue", "lymph node metastasis"],
        notes="32768 patches, balanced binary (PatchCamelyon).",
    ),
    "sicap": DatasetSpec(
        key="sicap", loader="sicap", rel_dir="sicapv2/extracted/SICAPv2",
        split="test",
        class_names=[
            "benign prostate tissue",
            "Gleason grade 3 prostate cancer",
            "Gleason grade 4 prostate cancer",
            "Gleason grade 5 prostate cancer",
        ],
        notes="2122 patches, 4-class Gleason (NC/G3/G4/G5, G4C merged into G4).",
    ),
    "mhist": DatasetSpec(
        key="mhist", loader="mhist", rel_dir="MHIST", split="test",
        class_names=[
            "hyperplastic colorectal polyp",
            "sessile serrated adenoma",
        ],
        notes="977-image official test partition, binary colorectal polyp (HP=0/SSA=1).",
    ),
    "databiox": DatasetSpec(
        key="databiox", loader="databiox", rel_dir="Databiox", split="40x",
        class_names=[
            "grade 1 invasive ductal carcinoma",
            "grade 2 invasive ductal carcinoma",
            "grade 3 invasive ductal carcinoma",
        ],
        notes="454 images, 3-class breast IDC Nottingham grade (G1/G2/G3), 40x only.",
    ),
    "bach": DatasetSpec(
        key="bach", loader="bach", rel_dir="bach_iciar2018",
        split="microscopy",
        class_names=[
            "normal breast tissue",
            "benign breast lesion",
            "breast carcinoma in situ",
            "invasive breast carcinoma",
        ],
        notes="BACH/ICIAR 2018 **Part A**: 400 microscopy patches, 4-class breast, "
              "from the Zenodo challenge archive. NOT to be confused with **Part B** "
              "of the same challenge, the 10 annotated WSIs, which this benchmark "
              "does not use. Always disambiguate in prose and in figure labels; "
              "'BACH' alone reads as double-counting.",
    ),
    "nct_crc": DatasetSpec(
        key="nct_crc", loader="nct_crc", rel_dir="crc100k_nct_val/extracted",
        split="val",
        class_names=[
            "adipose tissue",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth muscle",
            "normal colon mucosa",
            "colorectal cancer-associated stroma",
            "colorectal adenocarcinoma epithelium",
        ],
        notes="7180 patches from the independent CRC-VAL-HE-7K cohort, "
              "9-class colorectal tissue (Kather); class = filename prefix.",
    ),
    "wsss4luad": DatasetSpec(
        key="wsss4luad", loader="wsss4luad", rel_dir="wsss4luad_v2",
        split="train",
        class_names=[
            "lung adenocarcinoma tumor epithelium",
            "lung adenocarcinoma tumor-associated stroma",
            "normal lung tissue",
        ],
        notes="4693 pure one-hot patches from the labelled WSSS4LUAD training set; "
              "mixed-label patches are excluded to match CONCH's published "
              "patch-classification benchmark. Validation/test are segmentation "
              "sets and do not expose patch-classification labels.",
    ),
}


def dataset_dir(spec: DatasetSpec) -> Path:
    return datasets_root() / spec.rel_dir


# ─────────────────────────────────────────────────────────────────────────────
# Reported benchmark
# ─────────────────────────────────────────────────────────────────────────────

# ``DATASETS`` is everything the framework loads, and every one is reported.
# ``BENCHMARK_DATASETS`` is retained as the name the figures and tables import.
BENCHMARK_DATASETS: dict[str, DatasetSpec] = dict(DATASETS)


# ─────────────────────────────────────────────────────────────────────────────
# Whole-slide cohorts
# ─────────────────────────────────────────────────────────────────────────────

# The cohorts themselves live outside the repo and are located by the
# ``file_path`` column of these tables, which is an absolute path the reader
# repoints at their own copy. See docs/DATA_SOURCES.md.
SLIDE_TABLES = {
    "internal": project_root() / "datasets" / "internal-slide-lvl-data.xlsx",
    "external": project_root() / "datasets" / "external-slide-lvl-data.xlsx",
}
