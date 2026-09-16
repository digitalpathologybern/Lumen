"""Multiclass zero-shot metrics for the patch benchmark.

A (model, dataset, prompt bank) is scored by taking cosine image-text
similarity to per-class logits, then predictions. :func:`bootstrap_ci` gives the
interval on the reported metric.
"""

from __future__ import annotations

import numpy as np

from lumen.stats import bootstrap_ci as shared_bootstrap_ci, rows_are_units
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from lumen import zeroshot

NAN = float("nan")
BINARY_THRESHOLD_PREFIXES = ("binary_0p5", "binary_youden")
SELECTABLE_PRIMARY_METRICS = [
    "balanced_acc",
    "macro_f1",
    "macro_precision",
    "macro_sensitivity",
    "macro_auroc",
    "auroc",
    "accuracy",
    "binary_0p5_accuracy",
    "binary_0p5_f1",
    "binary_0p5_precision",
    "binary_0p5_sensitivity",
    "binary_youden_accuracy",
    "binary_youden_f1",
    "binary_youden_precision",
    "binary_youden_sensitivity",
]


def class_matrix(adapter, prompt_set: dict[str, list[str]]) -> np.ndarray:
    """Ensemble each class's prompts into one L2-normalized vector -> [C, D].

    Thin adapter over :func:`lumen.zeroshot.class_matrix` (shared with the
    encoder facade and the inference server).
    """
    return zeroshot.class_matrix(adapter.encode_text, prompt_set)


def score_logits(image_emb: np.ndarray, text_mat: np.ndarray) -> np.ndarray:
    """Cosine-similarity logits ``[N, C]`` (both inputs L2-normalized)."""
    return zeroshot.cosine_logits(image_emb, text_mat)


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax probabilities from logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    return probs / probs.sum(axis=1, keepdims=True)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else NAN


def _is_nan(value: float) -> bool:
    return bool(np.isnan(value))


def _binary_stats(labels: np.ndarray, preds: np.ndarray) -> dict:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = _safe_div(tp, tp + fp)
    sensitivity = _safe_div(tp, tp + fn)
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if not (_is_nan(precision) or _is_nan(sensitivity)) and (precision + sensitivity)
        else NAN
    )
    return {
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "f1": f1,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": _safe_div(tn, tn + fp),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _empty_binary_threshold_metrics() -> dict:
    out = {
        "binary_auroc": NAN,
        "binary_0p5_threshold": 0.5,
        "binary_youden_threshold": NAN,
        "binary_youden_j": NAN,
    }
    for prefix in BINARY_THRESHOLD_PREFIXES:
        for metric in ("accuracy", "f1", "precision", "sensitivity", "specificity"):
            out[f"{prefix}_{metric}"] = NAN
        for count in ("tp", "tn", "fp", "fn"):
            out[f"{prefix}_{count}"] = NAN
    return out


def _binary_threshold_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Positive-class metrics at 0.5 and Youden thresholds for binary tasks."""
    out = _empty_binary_threshold_metrics()
    if probs.shape[1] != 2:
        return out

    prob_pos = probs[:, 1]
    labels = labels.astype(int)
    if not set(np.unique(labels)).issubset({0, 1}):
        return out

    fixed = _binary_stats(labels, (prob_pos >= 0.5).astype(int))
    for key, value in fixed.items():
        out[f"binary_0p5_{key}"] = value

    if len(np.unique(labels)) < 2:
        return out

    try:
        out["binary_auroc"] = float(roc_auc_score(labels, prob_pos))
        fpr, tpr, thresholds = roc_curve(labels, prob_pos)
    except ValueError:
        return out

    youden_j = tpr - fpr
    idx = int(np.nanargmax(youden_j))
    threshold = float(thresholds[idx])
    youden = _binary_stats(labels, (prob_pos >= threshold).astype(int))
    out["binary_youden_threshold"] = threshold
    out["binary_youden_j"] = float(youden_j[idx])
    for key, value in youden.items():
        out[f"binary_youden_{key}"] = value
    return out


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Argmax/macro metrics plus binary threshold operating-point metrics.

    ``accuracy`` and the macro classification metrics use ``argmax`` over class
    logits. For binary datasets, that is equivalent to thresholding the class-1
    softmax probability at 0.5. The ``binary_*`` fields make that threshold
    explicit and also report the same positive-class metrics at Youden's J.
    """
    preds = logits.argmax(axis=1)
    n_classes = logits.shape[1]
    probs = softmax_probs(logits)

    out = {
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro",
                                   labels=list(range(n_classes)), zero_division=0)),
        "macro_precision": float(precision_score(
            labels, preds, average="macro", labels=list(range(n_classes)),
            zero_division=0)),
        "macro_sensitivity": float(recall_score(
            labels, preds, average="macro", labels=list(range(n_classes)),
            zero_division=0)),
        "accuracy": float((preds == labels).mean()),
    }
    # Quadratic-weighted Cohen kappa: the standard metric for ordinal grading
    # (e.g. Gleason on SICAP), giving partial credit for adjacent-grade errors.
    # Meaningful only when class ids are ordinal; computed for all datasets but
    # interpret for graded ones (sicap/databiox/osteo).
    try:
        out["quadratic_kappa"] = float(cohen_kappa_score(
            labels, preds, labels=list(range(n_classes)), weights="quadratic"))
    except ValueError:
        out["quadratic_kappa"] = NAN
    try:
        if n_classes == 2:
            out["macro_auroc"] = float(roc_auc_score(labels, probs[:, 1]))
        else:
            out["macro_auroc"] = float(roc_auc_score(
                labels, probs, multi_class="ovr", average="macro",
                labels=list(range(n_classes))))
    except ValueError:
        out["macro_auroc"] = NAN
    out["auroc"] = out["macro_auroc"]
    out.update(_binary_threshold_metrics(labels, probs))
    return out


def bootstrap_ci(logits: np.ndarray, labels: np.ndarray, metric: str,
                 n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for one metric from fixed logits, resampling patches.

    The patch is the resampling unit here, declared through ``rows_are_units``
    rather than assumed. Most of these datasets ship no patient or slide
    identifier, so the patch is all there is; where one does exist the interval
    is optimistic, and the Methods say so. Routing through
    :mod:`lumen.stats` keeps that decision visible instead of leaving it
    implicit in a local resampling loop.
    """
    return shared_bootstrap_ci(
        lambda idx: compute_metrics(logits[idx], labels[idx])[metric],
        rows_are_units(len(labels)), n_boot=n_boot, seed=seed)


def chance_level(n_classes: int) -> float:
    """Balanced accuracy of a label-blind predictor over ``n_classes``."""
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    return 1.0 / n_classes


def chance_corrected(balanced_acc, n_classes: int):
    """Rescale balanced accuracy so chance is 0 and perfect is 1.

    This is the paper's primary patch metric, and it was previously written out
    at each call site. Two figure scripts each derived the chance level from
    ``DatasetSpec.num_classes`` and applied the formula themselves, so a change
    of convention would have had to be made in every place at once, and a place
    missed would still produce a number in the right range.

    Accepts a scalar or an array. Note the rescaling divides by ``1 - 1/K``, so
    a binary task's differences are magnified twice as much as a nine-class
    task's: this makes datasets comparable on the chance axis, not on effect
    size, and an aggregate over datasets weights the low-K ones more heavily.
    """
    chance = chance_level(n_classes)
    return (np.asarray(balanced_acc, dtype=float) - chance) / (1.0 - chance)
