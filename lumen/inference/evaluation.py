"""Reusable slide-level binary classification metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


NAN = float("nan")


def safe_div(num: float, den: float) -> float:
    """Return ``num / den`` or NaN when the denominator is zero."""
    return num / den if den else NAN


def metric_is_nan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def round_metric(value, digits: int = 4):
    """Round finite numeric metrics, preserving NaN as an empty CSV cell."""
    return "" if metric_is_nan(value) else round(float(value), digits)


def _arrays(records: list[dict], score_col: str, pred_col: str = "pred"):
    y_true = np.array([int(r["gt"]) for r in records])
    y_prob = np.array([float(r[score_col]) for r in records])
    y_pred = []
    for record, prob in zip(records, y_prob):
        if pred_col in record and record[pred_col] not in (None, ""):
            y_pred.append(int(record[pred_col]))
        else:
            y_pred.append(int(prob >= 0.5))
    return y_true, y_prob, np.array(y_pred)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``tp, tn, fp, fn`` for binary labels."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def classification_stats(tp: int, tn: int, fp: int, fn: int) -> dict:
    """Accuracy, precision, recall/sensitivity, specificity, NPV, and F1."""
    total = tp + tn + fp + fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if not (metric_is_nan(precision) or metric_is_nan(recall)) and (precision + recall)
        else NAN
    )
    return {
        "accuracy": safe_div(tp + tn, total),
        "acc": safe_div(tp + tn, total),
        "precision": precision,
        "prec": precision,
        "recall": recall,
        "rec": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
    }


def _class_counts(y_true: np.ndarray) -> tuple[int, int]:
    n_pos = int(y_true.sum())
    return n_pos, int(len(y_true) - n_pos)


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else NAN


def youden_operating_point(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float, float]:
    """Return threshold, sensitivity, specificity at max Youden's J."""
    spec = 1.0 - fpr
    idx = int(np.argmax(tpr + spec - 1.0))
    return float(thresholds[idx]), float(tpr[idx]), float(spec[idx])


def sensitivity_operating_point(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    min_sensitivity: float,
) -> tuple[float, float, float]:
    """Return threshold meeting sensitivity floor with highest specificity."""
    spec = 1.0 - fpr
    candidates = np.where(tpr >= min_sensitivity)[0]
    if len(candidates) == 0:
        idx = int(np.argmax(tpr))
    else:
        candidate_specs = spec[candidates]
        idx = int(candidates[int(np.argmax(candidate_specs))])
    return float(thresholds[idx]), float(tpr[idx]), float(spec[idx])


def compute_probability_metrics(
    records: list[dict],
    score_col: str,
    pred_col: str = "pred",
) -> dict:
    """Compute AUC, confusion, and common operating points from slide rows."""
    if not records:
        return {
            "n": 0, "n_pos": 0, "n_neg": 0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "acc": NAN, "prec": NAN, "rec": NAN, "f1": NAN,
            "accuracy": NAN, "precision": NAN, "recall": NAN,
            "sensitivity": NAN, "specificity": NAN, "npv": NAN,
            "auroc": NAN, "aupr": NAN,
            "fpr": None, "tpr": None, "prec_curve": None, "rec_curve": None,
            "mean_prob_pos": NAN, "mean_prob_neg": NAN,
            "youden_thresh": NAN, "youden_sens": NAN, "youden_spec": NAN,
            "sens90_thresh": NAN, "sens90_sens": NAN, "sens90_spec": NAN,
            "sens95_thresh": NAN, "sens95_sens": NAN, "sens95_spec": NAN,
        }

    y_true, y_prob, y_pred = _arrays(records, score_col, pred_col)
    n_pos, n_neg = _class_counts(y_true)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    stats = classification_stats(tp, tn, fp, fn)
    base = {
        "n": len(y_true), "n_pos": n_pos, "n_neg": n_neg,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        **stats,
        "mean_prob_pos": _mean_or_nan(y_prob[y_true == 1]),
        "mean_prob_neg": _mean_or_nan(y_prob[y_true == 0]),
    }

    if n_pos == 0 or n_neg == 0:
        return {
            **base,
            "auroc": NAN, "aupr": NAN,
            "fpr": None, "tpr": None, "prec_curve": None, "rec_curve": None,
            "youden_thresh": NAN, "youden_sens": NAN, "youden_spec": NAN,
            "sens90_thresh": NAN, "sens90_sens": NAN, "sens90_spec": NAN,
            "sens95_thresh": NAN, "sens95_sens": NAN, "sens95_spec": NAN,
        }

    auroc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    y_t, y_sens, y_spec = youden_operating_point(fpr, tpr, thresholds)
    s90_t, s90_sens, s90_spec = sensitivity_operating_point(fpr, tpr, thresholds, 0.90)
    s95_t, s95_sens, s95_spec = sensitivity_operating_point(fpr, tpr, thresholds, 0.95)

    return {
        **base,
        "auroc": float(auroc), "aupr": float(aupr),
        "fpr": fpr, "tpr": tpr, "prec_curve": prec_curve, "rec_curve": rec_curve,
        "youden_thresh": y_t, "youden_sens": y_sens, "youden_spec": y_spec,
        "sens90_thresh": s90_t, "sens90_sens": s90_sens, "sens90_spec": s90_spec,
        "sens95_thresh": s95_t, "sens95_sens": s95_sens, "sens95_spec": s95_spec,
    }


def compute_threshold_metrics(
    records: list[dict],
    score_col: str,
    threshold: float | None = None,
    min_sensitivity: float | None = None,
) -> dict:
    """Compute metrics after selecting or applying an operating threshold."""
    if not records:
        return {
            "n": 0, "n_pos": 0, "n_neg": 0,
            "youden_thresh": NAN, "youden_j": NAN,
            "sensitivity": NAN, "specificity": NAN,
            "precision": NAN, "npv": NAN,
            "f1": NAN, "accuracy": NAN,
            "auroc": NAN, "aupr": NAN,
            "tp": NAN, "tn": NAN, "fp": NAN, "fn": NAN,
        }

    y_true, y_prob, _ = _arrays(records, score_col)
    n_pos, n_neg = _class_counts(y_true)
    base = {"n": len(y_true), "n_pos": n_pos, "n_neg": n_neg}
    if n_pos == 0 or n_neg == 0:
        return {
            **base,
            "youden_thresh": NAN, "youden_j": NAN,
            "sensitivity": NAN, "specificity": NAN,
            "precision": NAN, "npv": NAN,
            "f1": NAN, "accuracy": NAN,
            "auroc": NAN, "aupr": NAN,
            "tp": NAN, "tn": NAN, "fp": NAN, "fn": NAN,
        }

    auroc = float(roc_auc_score(y_true, y_prob))
    aupr = float(average_precision_score(y_true, y_prob))
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if threshold is None:
        if min_sensitivity is not None:
            threshold, _, _ = sensitivity_operating_point(
                fpr, tpr, thresholds, min_sensitivity
            )
        else:
            threshold, _, _ = youden_operating_point(fpr, tpr, thresholds)

    y_pred = (y_prob >= float(threshold)).astype(int)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    stats = classification_stats(tp, tn, fp, fn)
    youden_j = (
        stats["sensitivity"] + stats["specificity"] - 1.0
        if not (metric_is_nan(stats["sensitivity"]) or metric_is_nan(stats["specificity"]))
        else NAN
    )
    return {
        **base,
        "youden_thresh": float(threshold),
        "youden_j": youden_j,
        "sensitivity": stats["sensitivity"],
        "specificity": stats["specificity"],
        "precision": stats["precision"],
        "npv": stats["npv"],
        "f1": stats["f1"],
        "accuracy": stats["accuracy"],
        "auroc": auroc,
        "aupr": aupr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def summarize_thresholds_by_group(
    records: Iterable[dict],
    split_name: str,
    score_col: str,
    min_sensitivity: float | None = None,
    overall_group: str = "__overall__",
) -> list[dict]:
    """Return per-group rows plus a split-level overall threshold row."""
    rows = list(records)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(row)

    summary: list[dict] = []
    for group in sorted(by_group):
        summary.append({
            "split": split_name,
            "group": group,
            "score_col": score_col,
            **compute_threshold_metrics(
                by_group[group], score_col, min_sensitivity=min_sensitivity
            ),
        })
    summary.append({
        "split": split_name,
        "group": overall_group,
        "score_col": score_col,
        **compute_threshold_metrics(rows, score_col, min_sensitivity=min_sensitivity),
    })
    return summary
