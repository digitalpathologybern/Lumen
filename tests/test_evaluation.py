from __future__ import annotations

import math
import unittest

from lumen.inference.evaluation import (
    compute_probability_metrics,
    compute_threshold_metrics,
    summarize_thresholds_by_group,
)


class EvaluationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"group": "crc", "gt": 0, "pred": 0, "max_prob": 0.1, "avg_prob": 0.1},
            {"group": "crc", "gt": 0, "pred": 1, "max_prob": 0.8, "avg_prob": 0.7},
            {"group": "crc", "gt": 1, "pred": 1, "max_prob": 0.9, "avg_prob": 0.8},
            {"group": "crc", "gt": 1, "pred": 0, "max_prob": 0.4, "avg_prob": 0.3},
        ]

    def test_probability_metrics_include_confusion_auc_and_operating_points(self):
        metrics = compute_probability_metrics(self.records, "max_prob")

        self.assertEqual(metrics["n"], 4)
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]),
                         (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["acc"], 0.5)
        self.assertAlmostEqual(metrics["auroc"], 0.75)
        self.assertAlmostEqual(metrics["mean_prob_pos"], 0.65)
        self.assertAlmostEqual(metrics["mean_prob_neg"], 0.45)
        self.assertAlmostEqual(metrics["youden_thresh"], 0.9)
        self.assertAlmostEqual(metrics["youden_sens"], 0.5)
        self.assertAlmostEqual(metrics["youden_spec"], 1.0)

    def test_threshold_metrics_can_apply_fixed_threshold(self):
        metrics = compute_threshold_metrics(self.records, "max_prob", threshold=0.5)

        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]),
                         (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["youden_j"], 0.0)

    def test_single_class_auc_is_nan_without_crashing(self):
        rows = [
            {"group": "crc", "gt": 1, "pred": 1, "max_prob": 0.8},
            {"group": "crc", "gt": 1, "pred": 1, "max_prob": 0.9},
        ]

        metrics = compute_probability_metrics(rows, "max_prob")
        threshold_metrics = compute_threshold_metrics(rows, "max_prob")

        self.assertTrue(math.isnan(metrics["auroc"]))
        self.assertTrue(math.isnan(threshold_metrics["auroc"]))

    def test_summarize_thresholds_by_group_adds_overall_row(self):
        rows = self.records + [
            {"group": "breast", "gt": 0, "pred": 0, "max_prob": 0.2},
            {"group": "breast", "gt": 1, "pred": 1, "max_prob": 0.7},
        ]

        summary = summarize_thresholds_by_group(rows, "internal", "max_prob")
        groups = [row["group"] for row in summary]

        self.assertEqual(groups, ["breast", "crc", "__overall__"])
        self.assertTrue(all(row["split"] == "internal" for row in summary))


if __name__ == "__main__":
    unittest.main()
