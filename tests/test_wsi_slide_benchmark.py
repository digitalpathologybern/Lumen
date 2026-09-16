from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lumen.wsi.calibration import (
    build_patient_split,
    learn_youden_threshold,
)
from lumen.wsi.prompts import ensemble_prompts, iter_prompt_pairs
from lumen.inference.io import safe_name
from lumen.wsi.slide_benchmark import _build_text_matrix, _metrics, score_slides


def test_prompt_grid_has_three_vocabularies_by_six_templates():
    rows = list(iter_prompt_pairs())
    prompts = ensemble_prompts()

    assert len(rows) == 18
    assert {row[1] for row in rows} == {"canonical", "status", "healthy_tumor"}
    assert len(prompts["negative"]) == len(prompts["positive"]) == 18
    assert len(set(prompts["negative"])) == len(set(prompts["positive"])) == 18


def test_text_ensemble_is_normalized_mean_of_normalized_prompt_vectors():
    class Adapter:
        def encode_text(self, prompts):
            vecs = []
            for prompt in prompts:
                raw = np.array([
                    1.0 + (len(prompt) % 7),
                    1.0 + (sum(map(ord, prompt)) % 11),
                    1.0 + (sum(map(ord, prompt)) % 5),
                ], dtype=np.float32)
                vecs.append(raw / np.linalg.norm(raw))
            return np.stack(vecs)

    matrix = _build_text_matrix(Adapter())
    negative_rows = matrix[:-2:2]
    positive_rows = matrix[1:-2:2]
    expected_negative = negative_rows.mean(axis=0)
    expected_negative /= np.linalg.norm(expected_negative)
    expected_positive = positive_rows.mean(axis=0)
    expected_positive /= np.linalg.norm(expected_positive)

    assert matrix.shape == (38, 3)
    np.testing.assert_allclose(matrix[-2], expected_negative, atol=1e-6)
    np.testing.assert_allclose(matrix[-1], expected_positive, atol=1e-6)


def test_metrics_use_neutral_probability_boundary():
    result = _metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.2, 0.8, 0.7, 0.4]),
    )

    assert result["decision_threshold"] == 0.5
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (1, 1, 1, 1)
    assert result["balanced_accuracy"] == 0.5


def test_slide_score_max_pools_positive_class_probability_and_drops_tcga():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        included = root / "model" / "table" / "group"
        excluded = root / "model" / "table" / "melanoma_tcga"
        included.mkdir(parents=True)
        excluded.mkdir(parents=True)
        embeddings = np.array([[0.9, 0.1], [0.2, 0.6]], dtype=np.float32)
        np.savez(included / "slide.npz", embeddings=embeddings, gt=np.array(1))
        np.savez(excluded / "tcga.npz", embeddings=embeddings, gt=np.array(1))
        text = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        slides, groups, _, _, pair_scores, ensemble_scores = score_slides(
            root, "model", "table", text, n_pairs=1
        )

    expected = 1.0 / (1.0 + np.exp(-0.4))
    assert slides == ["slide"]
    assert groups == ["group"]
    assert pair_scores["max"][0, 0] == pytest.approx(expected)
    assert ensemble_scores["max"][0] == pytest.approx(expected)


def test_safe_name_matches_between_cache_writer_and_slide_table_join():
    """The embedding writer and the patient-ID join must share one transform.

    Slide caches are written to ``<group>/<slide>.npz`` under ``safe_name``.
    When the patient lookup was keyed on raw table names and queried with the
    sanitised stem, every slide whose name held a filtered character missed the
    join and fell back to one cluster per slide, silently turning the external
    patient-level bootstrap into a slide-level one (1,646 clusters for 978
    patients).
    """
    raw = "case 12 & 13/H&E"
    assert safe_name(raw) == "case_12_13_H_E"
    # Idempotent, so an already-sanitised stem still joins.
    assert safe_name(safe_name(raw)) == safe_name(raw)


def test_unmatched_slides_get_cohort_scoped_fallback_ids():
    """Same-named slides in different cohorts must not share a cluster.

    The fallback identifier used to be namespaced by table only, so a
    ``case_601_slide_H_E_0`` in colorectal and one in lung merged into a single
    bootstrap cluster, placing two unrelated patients in one unit.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        embeddings = np.array([[0.9, 0.1], [0.2, 0.6]], dtype=np.float32)
        for cohort in ("crc_histai_b1", "lung_histai_thorax"):
            d = root / "model" / "nosuchtable" / cohort
            d.mkdir(parents=True)
            np.savez(d / "case_601_slide_H_E_0.npz",
                     embeddings=embeddings, gt=np.array(0))
        text = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        _, groups, patient_ids, _, _, _ = score_slides(
            root, "model", "nosuchtable", text, n_pairs=1
        )

    assert len(patient_ids) == 2
    assert len(set(patient_ids)) == 2, (
        f"same-named slides in {groups} collapsed into one cluster: {patient_ids}"
    )
    assert all(g in pid for g, pid in zip(groups, patient_ids))


def test_patient_split_is_reproducible_stratified_and_has_no_leakage():
    rows = []
    for group in ("breast", "crc"):
        for pattern in ("negative_only", "positive_only", "mixed"):
            for patient_idx in range(10):
                patient = f"{group}-{pattern}-{patient_idx}"
                labels = [0, 1] if pattern == "mixed" else [int(pattern == "positive_only")]
                for slide_idx, label in enumerate(labels):
                    rows.append({
                        "patient_id": patient,
                        "group": group,
                        "gt": label,
                        "slide": f"{patient}-{slide_idx}",
                    })
    internal = pd.DataFrame(rows)

    first = build_patient_split(internal, calibration_fraction=0.2, seed=42)
    second = build_patient_split(internal, calibration_fraction=0.2, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert (first["split"] == "calibration").sum() == 12
    by_stratum = first.groupby(["group", "label_pattern", "split"]).size()
    for group in ("breast", "crc"):
        for pattern in ("negative_only", "positive_only", "mixed"):
            assert by_stratum[group, pattern, "calibration"] == 2
            assert by_stratum[group, pattern, "internal_holdout"] == 8


def test_youden_ties_choose_the_more_conservative_threshold():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.4, 0.9])
    threshold = learn_youden_threshold(labels, scores)

    assert threshold == 0.9


# ── melanoma prompt-sensitivity vocabulary ──────────────────────────────────
# Two external cohorts are melanoma, which is not a carcinoma, so the reported
# bank names a malignancy those slides do not contain. The alternative bank
# existed in the source but nothing read it, and the numbers once quoted for it
# turned out to have come from a checkpoint the paper does not report. These
# pin the switch that makes it runnable and attributable.

def test_the_two_lymph_node_vocabularies_differ_only_in_wording():
    from lumen.wsi.prompts import iter_prompt_pairs

    shared = list(iter_prompt_pairs(vocabulary="shared"))
    agnostic = list(iter_prompt_pairs(vocabulary="melanoma_agnostic"))

    assert len(shared) == len(agnostic) == 18
    # Same templates in the same order, so a difference is the class name alone.
    assert [r[0] for r in shared] == [r[0] for r in agnostic]
    assert [r[1] for r in shared] == [r[1] for r in agnostic]
    assert shared != agnostic


def test_the_agnostic_bank_never_says_carcinoma():
    """That word is the entire reason the alternative bank exists."""
    from lumen.wsi.prompts import ensemble_prompts

    bank = ensemble_prompts(vocabulary="melanoma_agnostic")
    offenders = [p for side in bank.values() for p in side if "carcinoma" in p]
    assert not offenders, offenders


def test_the_default_vocabulary_is_still_the_reported_one():
    """Every published lymph-node number came from the shared bank."""
    from lumen.wsi.prompts import (LYMPH_NODE_CLASS_VARIANTS,
                                      iter_prompt_pairs, resolve_vocabulary)

    assert resolve_vocabulary(None) is LYMPH_NODE_CLASS_VARIANTS
    assert list(iter_prompt_pairs()) == list(iter_prompt_pairs(vocabulary="shared"))


def test_the_ensemble_follows_the_requested_vocabulary():
    """The ensemble rows used to be built from the default bank unconditionally.

    That would have scored the per-prompt metrics on one vocabulary and the
    ensemble, which is the number actually reported, on another. The mismatch is
    invisible in the output: both are plausible AUROCs.
    """
    import numpy as np

    from lumen.wsi.prompts import iter_prompt_pairs
    from lumen.wsi.slide_benchmark import _build_text_matrix

    seen: list[str] = []

    class SpyAdapter:
        def encode_text(self, prompts):
            seen.extend(prompts)
            return np.zeros((len(prompts), 4), dtype=np.float32)

    rows = list(iter_prompt_pairs(vocabulary="melanoma_agnostic"))
    _build_text_matrix(SpyAdapter(), rows, vocabulary="melanoma_agnostic")

    assert seen, "adapter was never asked to encode anything"
    assert not [p for p in seen if "carcinoma" in p]
