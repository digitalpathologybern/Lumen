import math
import unittest

import numpy as np

from lumen.benchmark.metrics import compute_metrics


def binary_logits_from_positive_probs(values):
    probs = np.array(values, dtype=float)
    return np.column_stack([np.zeros_like(probs), np.log(probs / (1.0 - probs))])


class BenchmarkMetricTests(unittest.TestCase):
    def test_binary_metrics_report_0p5_and_youden_thresholds(self):
        labels = np.array([0, 0, 1, 1])
        logits = binary_logits_from_positive_probs([0.1, 0.3, 0.4, 0.8])

        metrics = compute_metrics(logits, labels)

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["binary_0p5_threshold"], 0.5)
        self.assertAlmostEqual(metrics["binary_0p5_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["binary_0p5_precision"], 1.0)
        self.assertAlmostEqual(metrics["binary_0p5_sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["binary_0p5_f1"], 2.0 / 3.0)

        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["binary_auroc"], 1.0)
        self.assertAlmostEqual(metrics["binary_youden_threshold"], 0.4)
        self.assertAlmostEqual(metrics["binary_youden_j"], 1.0)
        self.assertAlmostEqual(metrics["binary_youden_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["binary_youden_precision"], 1.0)
        self.assertAlmostEqual(metrics["binary_youden_sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["binary_youden_f1"], 1.0)

    def test_multiclass_metrics_leave_binary_thresholds_empty(self):
        labels = np.array([0, 1, 2, 0])
        logits = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.2, 0.4, 0.1],
            ]
        )

        metrics = compute_metrics(logits, labels)

        self.assertIn("macro_precision", metrics)
        self.assertIn("macro_sensitivity", metrics)
        self.assertTrue(math.isnan(metrics["binary_auroc"]))
        self.assertTrue(math.isnan(metrics["binary_youden_threshold"]))
        self.assertTrue(math.isnan(metrics["binary_youden_accuracy"]))


if __name__ == "__main__":
    unittest.main()


# ── Chance correction ────────────────────────────────────────────────────────

def test_chance_corrected_reproduces_reported_values():
    """The paper's primary patch metric, pinned to published numbers.

    It was written out at each call site, so a change of convention had to be
    made everywhere at once and a missed place still produced a number in the
    right range.
    """
    from lumen.benchmark.metrics import chance_corrected
    assert round(float(chance_corrected(0.861, 2)), 3) == 0.722   # PatchCamelyon
    assert round(float(chance_corrected(0.716, 9)), 3) == 0.680   # NCT-CRC
    assert round(float(chance_corrected(0.412, 4)), 3) == 0.216   # SICAPv2


def test_chance_is_zero_and_perfect_is_one():
    from lumen.benchmark.metrics import chance_corrected, chance_level
    for k in (2, 3, 4, 5, 9):
        assert abs(float(chance_corrected(chance_level(k), k))) < 1e-12
        assert abs(float(chance_corrected(1.0, k)) - 1.0) < 1e-12


def test_chance_corrected_rejects_a_degenerate_class_count():
    from lumen.benchmark.metrics import chance_level
    import pytest
    with pytest.raises(ValueError):
        chance_level(1)
