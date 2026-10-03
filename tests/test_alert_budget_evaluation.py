from __future__ import annotations

import unittest

import numpy as np

from src.syndicai_v4.modeling import evaluate_scores, select_alert_budget_threshold


class AlertBudgetThresholdTests(unittest.TestCase):
    def test_selector_chooses_nearest_cutoff_across_repeated_score_groups(self) -> None:
        scores = np.concatenate(
            (
                np.full(964, 0.9657666),
                np.full(252, 0.9657165),
                np.full(100, 0.4),
            )
        )

        threshold, validation_alerts = select_alert_budget_threshold(
            scores,
            980 / len(scores),
        )

        self.assertEqual(validation_alerts, 964)
        self.assertAlmostEqual(threshold, 0.9657666)

    def test_threshold_uses_validation_only_and_selects_nearest_tied_budget(self) -> None:
        validation_scores = np.array([0.99, 0.98, 0.98, 0.90, 0.70])

        threshold, validation_alerts = select_alert_budget_threshold(
            validation_scores, 0.40
        )

        self.assertEqual(validation_alerts, 1)
        self.assertGreater(threshold, 0.98)
        self.assertEqual(
            select_alert_budget_threshold(validation_scores, 0.40),
            (threshold, validation_alerts),
        )

    def test_metrics_are_test_outcomes_at_selected_threshold(self) -> None:
        validation_scores = np.array([0.99, 0.98, 0.97, 0.90, 0.70])
        threshold, _ = select_alert_budget_threshold(validation_scores, 0.40)
        test_labels = np.array([1, 0, 1, 0, 0, 1])
        test_scores = np.array([0.995, 0.985, 0.975, 0.94, 0.89, 0.65])

        metrics = evaluate_scores(test_labels, test_scores, threshold)

        self.assertEqual(metrics["alerts"], 2)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 2)
        self.assertAlmostEqual(metrics["alert_burden_pct"], 100 * 2 / 6)
        self.assertAlmostEqual(
            metrics["false_positives_per_10000_negatives"],
            10_000 / 3,
        )
        self.assertAlmostEqual(
            metrics["pr_auc"],
            evaluate_scores(test_labels, test_scores, 0.5)["pr_auc"],
        )

    def test_invalid_budget_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            select_alert_budget_threshold(np.array([0.1, 0.2]), 0.0)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            select_alert_budget_threshold(np.array([]), 0.1)


if __name__ == "__main__":
    unittest.main()
