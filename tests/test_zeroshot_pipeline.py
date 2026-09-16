"""Zero-shot scoring primitives: prompt ensembling, cosine logits, and the
balanced-accuracy chance level.

These lock the contract the whole patch benchmark relies on: each class is one
L2-normalized mean of its prompt embeddings, scoring is cosine similarity, and a
label-blind predictor scores exactly 1/C balanced accuracy (the dotted chance
line in Fig. 3) regardless of class imbalance.
"""

import unittest

import numpy as np

from lumen.benchmark.metrics import compute_metrics, score_logits
from lumen.zeroshot import class_matrix


def _encoder(table):
    """Fake ``encode_text``: looks each prompt up in ``table`` -> stacked rows."""
    return lambda prompts: np.stack([table[p] for p in prompts]).astype(np.float32)


class ClassMatrixTests(unittest.TestCase):
    def test_rows_are_unit_norm_and_follow_insertion_order(self):
        table = {
            "cat": np.array([1.0, 0.0, 0.0]),
            "dog1": np.array([0.0, 1.0, 0.0]),
            "dog2": np.array([0.0, 1.0, 0.0]),
        }
        mat = class_matrix(_encoder(table), {"cat": ["cat"], "dog": ["dog1", "dog2"]})
        self.assertEqual(mat.shape, (2, 3))
        np.testing.assert_allclose(np.linalg.norm(mat, axis=1), [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(mat[0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(mat[1], [0.0, 1.0, 0.0], atol=1e-6)

    def test_mean_pool_happens_before_normalization(self):
        # two orthogonal prompts -> mean is the diagonal -> normalized to 1/sqrt(2)
        table = {"p1": np.array([1.0, 0.0]), "p2": np.array([0.0, 1.0])}
        mat = class_matrix(_encoder(table), {"c": ["p1", "p2"]})
        np.testing.assert_allclose(mat[0], [1 / np.sqrt(2)] * 2, atol=1e-6)


class ScoreLogitsTests(unittest.TestCase):
    def test_score_logits_is_cosine_and_argmax_picks_nearest_class(self):
        img = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
        mat = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
        logits = score_logits(img, mat)
        np.testing.assert_allclose(logits, [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)
        np.testing.assert_array_equal(logits.argmax(axis=1), [0, 1])


class BalancedAccuracyChanceTests(unittest.TestCase):
    def test_label_blind_predictor_scores_one_over_c_under_imbalance(self):
        for c in (2, 3, 4, 5):
            labels = np.concatenate(
                [np.zeros(100, int)] + [np.full(5, k) for k in range(1, c)])
            logits = np.zeros((len(labels), c))
            logits[:, 0] = 10.0  # always predict the majority class
            m = compute_metrics(logits, labels)
            self.assertAlmostEqual(m["balanced_acc"], 1.0 / c, places=6,
                                   msg=f"chance != 1/{c} for {c}-class imbalance")

    def test_perfect_predictions_give_balanced_accuracy_one(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        logits = np.eye(3)[labels] * 5.0
        self.assertAlmostEqual(compute_metrics(logits, labels)["balanced_acc"], 1.0)


class QuadraticKappaTests(unittest.TestCase):
    """Ordinal metric: adjacent-grade errors get partial credit (the SICAP
    Gleason story) where balanced accuracy gives none."""

    def test_perfect_predictions_give_kappa_one(self):
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        logits = np.eye(4)[labels] * 5.0
        self.assertAlmostEqual(compute_metrics(logits, labels)["quadratic_kappa"], 1.0)

    def test_adjacent_grade_confusion_beats_distant_confusion(self):
        # 4 ordinal grades; one predictor shifts every label by one (adjacent),
        # the other by the maximum ordinal distance. Balanced accuracy is 0 for
        # both; QWK ranks the adjacent-error predictor much higher.
        labels = np.array([0, 1, 2, 3] * 25)
        adjacent = np.clip(labels + 1, 0, 3)
        distant = 3 - labels  # 0<->3, 1<->2 (max ordinal distance)
        k_adj = compute_metrics(np.eye(4)[adjacent] * 5.0, labels)["quadratic_kappa"]
        k_dist = compute_metrics(np.eye(4)[distant] * 5.0, labels)["quadratic_kappa"]
        self.assertGreater(k_adj, k_dist)


if __name__ == "__main__":
    unittest.main()
