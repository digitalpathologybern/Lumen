"""Structured prompt bank for zero-shot classification.

Two independent axes of variation are kept separate so the benchmark can
attribute variance to each:

* **templates**: sentence frames with a ``{}`` slot (phrasing variance).
* **class-name variants**: alternative wordings per class (label variance).

A *prompt set* pairs one template list with one class-name-variant map, and
expands to ``{class: [filled prompts]}``. The benchmark evaluates every
(template, variant) combination to get a metric distribution per (model, dataset).
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

# ── Templates (phrasing axis) ────────────────────────────────────────────────
# CLIP-style + pathology-specific frames used across the PLIP / BiomedCLIP /
# CONCH papers.
TEMPLATES: dict[str, list[str]] = {
    "bare":        ["{}"],
    "a_photo":     ["a photo of {}."],
    "he_image":    ["an H&E stained image of {}."],
    "histo":       ["histopathology of {}."],
    "patch":       ["a histopathology image showing {}."],
    "wsi":         ["a whole-slide image of {}."],
}

# The default template set (one frame each).
DEFAULT_TEMPLATE_KEYS = list(TEMPLATES)


# ── Class-name variants (label axis), per dataset ────────────────────────────
# variant "canonical" MUST match DatasetSpec.class_names ordering. Additional
# variants are alternative clinical wordings, same class order.
CLASS_NAME_VARIANTS: dict[str, dict[str, list[list[str]]]] = {
    "lc25000": {
        "canonical": [["benign lung tissue", "lung adenocarcinoma",
                       "lung squamous cell carcinoma", "benign colon tissue",
                       "colon adenocarcinoma"]],
        "short":     [["normal lung", "lung adenocarcinoma", "lung SCC",
                       "normal colon", "colon adenocarcinoma"]],
    },
    "osteo": {
        "canonical": [["non-tumor tissue", "necrotic tumor", "viable tumor"]],
        "clinical":  [["non-tumor bone tissue", "necrotic osteosarcoma",
                       "viable osteosarcoma"]],
    },
    "pcam": {
        "canonical": [["normal lymph node tissue", "lymph node metastasis"]],
        "short":     [["normal", "tumor"]],
        "clinical":  [["benign lymph node", "metastatic tumor"]],
    },
    "sicap": {
        # canonical = reported paper wording (Table 2 ensemble 0.452).
        # Reordering the cancer classes to "prostate cancer, Gleason grade N"
        # was tested and lowered the ensemble to 0.424, so it is not used.
        "canonical": [["benign prostate tissue",
                       "Gleason grade 3 prostate cancer",
                       "Gleason grade 4 prostate cancer",
                       "Gleason grade 5 prostate cancer"]],
        "short":     [["benign", "Gleason 3", "Gleason 4", "Gleason 5"]],
    },
    "mhist": {
        # canonical = the wording used for the reported paper numbers
        # (Table 2, ensemble balanced accuracy 0.591). The more verbose
        # "colorectal polyp on both classes" variant was tested and did not
        # help (ensemble 0.532), so it is kept only as a non-canonical variant.
        "canonical": [["hyperplastic colorectal polyp",
                       "sessile serrated adenoma"]],
        "verbose":   [["benign hyperplastic colorectal polyp",
                       "sessile serrated colorectal polyp"]],
    },
    "databiox": {
        # canonical = reported paper wording (Table 2 ensemble 0.386). The
        # fully spelled-out "of the breast, well/moderately/poorly differentiated"
        # variant was tested and did not help (ensemble 0.397); kept as verbose.
        "canonical": [["grade 1 invasive ductal carcinoma",
                       "grade 2 invasive ductal carcinoma",
                       "grade 3 invasive ductal carcinoma"]],
        "verbose":   [["invasive ductal carcinoma of the breast, grade 1, well differentiated",
                       "invasive ductal carcinoma of the breast, grade 2, moderately differentiated",
                       "invasive ductal carcinoma of the breast, grade 3, poorly differentiated"]],
    },
    "bach": {
        # BACH/ICIAR 2018 microscopy classes: normal, benign, in situ, invasive.
        "canonical": [["normal breast tissue", "benign breast lesion",
                       "breast carcinoma in situ", "invasive breast carcinoma"]],
        "clinical":  [["normal breast histology", "benign breast tumor",
                       "ductal carcinoma in situ", "invasive ductal carcinoma"]],
    },
    "nct_crc": {
        # 9-class Kather colorectal tissue (NCT-CRC-HE-100K + CRC-VAL-HE-7K).
        # canonical MUST match registry class_names order (ADI, BACK, DEB, LYM,
        # MUC, MUS, NORM, STR, TUM).
        "canonical": [["adipose tissue", "background", "debris", "lymphocytes",
                       "mucus", "smooth muscle", "normal colon mucosa",
                       "colorectal cancer-associated stroma",
                       "colorectal adenocarcinoma epithelium"]],
        "clinical":  [["adipose tissue", "empty background slide", "necrotic debris",
                       "lymphocyte aggregate", "extracellular mucus",
                       "smooth muscle tissue", "normal colon mucosa glands",
                       "cancer-associated stroma", "colorectal adenocarcinoma"]],
    },
    "wsss4luad": {
        # WSSS4LUAD pure one-hot patch subset. Order must match registry:
        # tumor epithelium, tumor-associated stroma, normal.
        "canonical": [["lung adenocarcinoma tumor epithelium",
                       "lung adenocarcinoma tumor-associated stroma",
                       "normal lung tissue"]],
        "short":     [["tumor epithelium", "tumor-associated stroma",
                       "normal tissue"]],
        "clinical":  [["lung adenocarcinoma epithelial tumor cells",
                       "desmoplastic tumor stroma in lung adenocarcinoma",
                       "benign normal lung parenchyma"]],
    },
}


def variant_names(dataset_key: str) -> list[str]:
    return list(CLASS_NAME_VARIANTS[dataset_key])


def class_names_for(dataset_key: str, variant: str) -> list[str]:
    """Return the ordered class-name list for one variant."""
    return CLASS_NAME_VARIANTS[dataset_key][variant][0]


def build_prompt_set(dataset_key: str, template_key: str,
                     variant: str) -> dict[str, list[str]]:
    """Return ``{class_name: [filled prompts]}`` for one (template, variant)."""
    names = class_names_for(dataset_key, variant)
    frames = TEMPLATES[template_key]
    return {name: [f.format(name) for f in frames] for name in names}


def iter_prompt_sets(dataset_key: str,
                     template_keys: list[str] | None = None,
                     variants: list[str] | None = None):
    """Yield ``(template_key, variant, prompt_set)`` for every combination."""
    template_keys = template_keys or DEFAULT_TEMPLATE_KEYS
    variants = variants or variant_names(dataset_key)
    for tkey, variant in product(template_keys, variants):
        yield tkey, variant, build_prompt_set(dataset_key, tkey, variant)


def canonical_prompts(dataset_key: str) -> dict[str, list[str]]:
    """Prompt-ensemble set: canonical class names across all templates."""
    names = class_names_for(dataset_key, "canonical")
    all_frames = [f for frames in TEMPLATES.values() for f in frames]
    return {name: [f.format(name) for f in all_frames] for name in names}


# ── Richer prompt bank: the second protocol, CONCH's 22 released frames ──────
# A zero-shot score is a property of a model and a prompt protocol jointly, so a
# ranking produced under one bank has to be checked against another. These are
# CONCH's released frames, verbatim from its ``*_all_per_class.json`` files, and
# they are the second protocol every prompted experiment in this work is
# re-scored under. They live here rather than beside any one experiment so that
# every experiment re-scored under "the richer bank" uses the identical one.
RICH_TEMPLATES: tuple[str, ...] = (
    "CLASSNAME.", "a photomicrograph showing CLASSNAME.",
    "a photomicrograph of CLASSNAME.", "an image of CLASSNAME.",
    "an image showing CLASSNAME.", "an example of CLASSNAME.",
    "CLASSNAME is shown.", "this is CLASSNAME.", "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.",
    "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.", "presence of CLASSNAME.", "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.", "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.", "an H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.", "CLASSNAME, H&E.",
)


def fill_templates(synonyms: Sequence[str]) -> list[str]:
    """Expand every synonym across :data:`RICH_TEMPLATES`.

    Shared by every experiment re-scored under the richer protocol, so that "the
    richer bank" means one thing. Order is synonym-major, matching CONCH's files.
    """
    return [t.replace("CLASSNAME", syn) for syn in synonyms for t in RICH_TEMPLATES]


def synonyms_for(dataset_key: str) -> dict[str, list[str]]:
    """``{canonical class name: [synonyms]}`` from every registered variant.

    The variants carry the same class order by construction, so column *i*
    across variants is the synonym set for class *i*. Duplicates are dropped
    while keeping first-seen order, so a dataset whose variants coincide on a
    class simply contributes fewer synonyms for it rather than repeating one.
    """
    variants = [class_names_for(dataset_key, v) for v in variant_names(dataset_key)]
    canonical = variants[0]
    return {name: list(dict.fromkeys(v[i] for v in variants))
            for i, name in enumerate(canonical)}


def rich_canonical_prompts(dataset_key: str) -> dict[str, list[str]]:
    """The 22 templates over the canonical class name only: 22 prompts per class.

    This is the single-axis counterpart to :func:`rich_prompts`. Against
    :func:`canonical_prompts` it changes the template set and nothing else, so a
    score difference between the two is attributable to template richness alone.
    :func:`rich_prompts` moves the vocabulary at the same time, and gives a bank
    whose size depends on how many variants a dataset happens to register, so it
    cannot separate the two effects or weight the datasets equally.
    """
    names = class_names_for(dataset_key, "canonical")
    return {name: [t.replace("CLASSNAME", name) for t in RICH_TEMPLATES]
            for name in names}


def rich_prompts(dataset_key: str) -> dict[str, list[str]]:
    """Richer-protocol prompt set: every class-name variant across 22 templates.

    The counterpart to :func:`canonical_prompts`, which is one name across six
    templates. Note the bank is as rich as the registered variants allow: with
    two variants a class gets 44 prompts, not the 110 that five synonyms would
    give. Adding variants to :data:`CLASS_NAME_VARIANTS` widens it here
    automatically.
    """
    return {name: fill_templates(syns)
            for name, syns in synonyms_for(dataset_key).items()}
