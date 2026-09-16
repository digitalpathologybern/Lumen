"""Prespecified lymph-node prompt grid shared by WSI scoring and reporting.

The WSI benchmark uses the same six sentence templates as the patch benchmark,
but expands three binary class vocabularies.  The class representation averages
all 18 normalized text embeddings of each polarity and L2-normalizes the mean.
"""

from __future__ import annotations

from lumen.benchmark.prompts import DEFAULT_TEMPLATE_KEYS, TEMPLATES


LYMPH_NODE_CLASS_VARIANTS: dict[str, tuple[str, str]] = {
    "canonical": (
        "normal lymph node tissue",
        "lymph node tissue with metastatic carcinoma",
    ),
    "status": (
        "a lymph node negative for metastatic carcinoma",
        "a lymph node positive for metastatic carcinoma",
    ),
    "healthy_tumor": (
        "healthy lymph node tissue",
        "metastatic tumor in lymph node tissue",
    ),
}

# The external cohorts include melanoma, which is not a carcinoma, so the shared
# "metastatic carcinoma" vocabulary names a malignancy two cohorts do not have.
# As a prompt-sensitivity check the melanoma cohorts were re-scored with this
# histology-agnostic positive vocabulary (see cli/wsi_benchmark_evaluate.py
# --groups melanoma_histai_b1,melanoma_histai_b2). Melanoma discrimination was
# unchanged (AUROC 0.994 / 0.984 vs 0.990 / 0.983 under the shared bank), so the
# result is not an artifact of the carcinoma wording. The shared bank above is the
# reported protocol; this variant is documented, not the default, because the
# locked threshold is calibrated on the shared-vocabulary score scale.
LYMPH_NODE_MELANOMA_SENSITIVITY: dict[str, tuple[str, str]] = {
    "canonical": (
        "normal lymph node tissue",
        "lymph node tissue with metastatic tumor",
    ),
    "status": (
        "a lymph node negative for metastatic malignancy",
        "a lymph node positive for metastatic malignancy",
    ),
    "healthy_tumor": (
        "healthy lymph node tissue",
        "metastatic tumor in lymph node tissue",
    ),
}


# The two vocabularies, by name, so a caller selects one instead of editing a
# module-level dict. The melanoma check previously had no way in: the sensitivity
# bank above was defined but nothing read it, so the numbers once quoted for it
# could not be reproduced or attributed to a checkpoint.
VOCABULARIES: dict[str, dict[str, tuple[str, str]]] = {
    "shared": LYMPH_NODE_CLASS_VARIANTS,
    "melanoma_agnostic": LYMPH_NODE_MELANOMA_SENSITIVITY,
}


def resolve_vocabulary(name_or_variants=None):
    """Accept a registered name, an explicit variants dict, or None."""
    if name_or_variants is None:
        return LYMPH_NODE_CLASS_VARIANTS
    if isinstance(name_or_variants, str):
        try:
            return VOCABULARIES[name_or_variants]
        except KeyError:
            raise SystemExit(
                f"unknown vocabulary {name_or_variants!r}. "
                f"Known: {', '.join(sorted(VOCABULARIES))}") from None
    return name_or_variants


def iter_prompt_pairs(template_keys: list[str] | None = None, vocabulary=None):
    """Yield ``(template, variant, negative_prompt, positive_prompt)`` rows."""
    keys = template_keys or DEFAULT_TEMPLATE_KEYS
    for variant, (negative, positive) in resolve_vocabulary(vocabulary).items():
        for template_key in keys:
            for frame in TEMPLATES[template_key]:
                yield (
                    template_key,
                    variant,
                    frame.format(negative),
                    frame.format(positive),
                )


def ensemble_prompts(template_keys: list[str] | None = None,
                     vocabulary=None) -> dict[str, list[str]]:
    """Return the ordered 18-prompt bank for each binary class."""
    rows = list(iter_prompt_pairs(template_keys, vocabulary))
    return {
        "negative": [row[2] for row in rows],
        "positive": [row[3] for row in rows],
    }
