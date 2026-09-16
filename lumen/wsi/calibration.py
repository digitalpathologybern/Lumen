"""Patient-level calibration and fixed-threshold evaluation primitives."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lumen.stats import cluster_resample_indices, percentile_ci
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def learn_youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return the highest threshold among operating points with maximal J."""
    if np.unique(labels).size != 2:
        raise ValueError("Youden calibration requires both negative and positive slides")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    youden = tpr - fpr
    best = np.flatnonzero(np.isclose(youden, np.nanmax(youden)))
    return float(np.max(thresholds[best]))


def build_patient_split(
    internal: pd.DataFrame,
    calibration_fraction: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Create one patient split stratified by organ and slide-label pattern."""
    required = {"patient_id", "group", "gt"}
    missing = required - set(internal)
    if missing:
        raise ValueError(f"internal scores are missing columns: {sorted(missing)}")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")

    records = []
    for patient_id, frame in internal.groupby("patient_id", sort=True):
        groups = sorted(frame["group"].astype(str).unique())
        if len(groups) != 1:
            raise ValueError(f"patient {patient_id!r} occurs in multiple organs: {groups}")
        labels = set(frame["gt"].astype(int))
        pattern = "mixed" if labels == {0, 1} else (
            "positive_only" if labels == {1} else "negative_only"
        )
        records.append({
            "patient_id": str(patient_id),
            "group": groups[0],
            "label_pattern": pattern,
            "n_slides": int(len(frame)),
            "n_pos": int(frame["gt"].astype(int).sum()),
        })

    patients = pd.DataFrame(records)
    patients["stratum"] = patients["group"] + "::" + patients["label_pattern"]
    counts = patients["stratum"].value_counts()
    rare = counts[counts < 2]
    if not rare.empty:
        raise ValueError(
            "patient stratification requires at least two patients per organ/label "
            f"stratum; too small: {rare.to_dict()}"
        )
    _holdout, calibration = train_test_split(
        patients,
        test_size=calibration_fraction,
        random_state=seed,
        stratify=patients["stratum"],
    )
    split = patients.copy()
    split["split"] = "internal_holdout"
    split.loc[split["patient_id"].isin(calibration["patient_id"]), "split"] = "calibration"
    return split.sort_values(["group", "label_pattern", "patient_id"]).reset_index(drop=True)


def fixed_threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Binary discrimination and classification metrics at a fixed threshold."""
    if len(labels) == 0 or np.unique(labels).size != 2:
        return {
            "n": int(len(labels)),
            "n_pos": int(labels.sum()) if len(labels) else 0,
            "n_neg": int((labels == 0).sum()) if len(labels) else 0,
            **{key: float("nan") for key in (
                "auroc", "aupr", "accuracy", "balanced_accuracy",
                "sensitivity", "specificity", "precision", "f1",
            )},
        }
    pred = (scores >= threshold).astype(int)
    return {
        "n": int(len(labels)),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "sensitivity": float(recall_score(labels, pred, pos_label=1)),
        "specificity": float(recall_score(labels, pred, pos_label=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
    }


def bootstrap_fixed_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    patient_ids: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Patient-clustered percentile intervals without refitting the threshold.

    ``progress``, when given, is called as ``progress(done, n_boot)`` roughly
    every 5% of resamples so long-running bootstraps can report a heartbeat.
    """
    keys = ("auroc", "balanced_accuracy", "sensitivity", "specificity")
    if n_boot <= 0:
        return {f"{key}_ci_{side}": float("nan") for key in keys for side in ("lo", "hi")}
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(patient_ids, dtype=str)
    vals = {key: [] for key in keys}
    report_step = max(1, n_boot // 20)
    for i in range(n_boot):
        idx = cluster_resample_indices(patient_ids, rng)
        sample = fixed_threshold_metrics(labels[idx], scores[idx], threshold)
        if not np.isnan(sample["auroc"]):
            for key in keys:
                vals[key].append(sample[key])
        if progress is not None and ((i + 1) % report_step == 0 or i + 1 == n_boot):
            progress(i + 1, n_boot)
    out = {}
    for key, samples in vals.items():
        lo, hi = percentile_ci(samples)
        out[f"{key}_ci_lo"] = float(lo)
        out[f"{key}_ci_hi"] = float(hi)
    return out
