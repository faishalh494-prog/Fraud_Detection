from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.syndicai_v4.anomaly import (
    ANOMALY_FEATURES,
    NormalBehaviourDetector,
    causal_training_scores,
    transform_anomaly_features,
)
from src.syndicai_v4.investigations import InvestigationStore
from src.syndicai_v4.modeling import (
    evaluate_scores,
    explain_prediction,
    risk_level,
    select_f1_threshold,
)
from src.syndicai_v4.network import (
    NETWORK_FEATURES,
    build_investigation_network,
    build_network_features,
)


class NetworkFeatureTests(unittest.TestCase):
    def test_network_features_exclude_same_step_and_future_edges(self) -> None:
        rows = pd.DataFrame(
            {
                "step": [1, 2, 2, 3, 4],
                "nameOrig": ["A", "A", "C", "B", "A"],
                "nameDest": ["B", "B", "B", "A", "C"],
            }
        )
        features = build_network_features(rows)

        self.assertEqual(features.columns.tolist(), NETWORK_FEATURES)
        self.assertEqual(features.loc[0].tolist(), [0, 0, 0, 0])
        self.assertEqual(features.loc[1].tolist(), [0, 0, 0, 0])
        self.assertEqual(features.loc[2].tolist(), [0, 0, 0, 0])
        self.assertEqual(features.loc[3, "network_sender_prior_in_degree"], 3)
        self.assertEqual(features.loc[3, "network_sender_in_degree_last24"], 3)
        self.assertEqual(features.loc[3, "network_receiver_prior_out_degree"], 2)
        self.assertEqual(features.loc[3, "network_receiver_out_degree_last24"], 2)
        self.assertEqual(features.loc[4, "network_sender_prior_in_degree"], 1)

    def test_network_features_require_chronological_rows(self) -> None:
        rows = pd.DataFrame(
            {"step": [2, 1], "nameOrig": ["A", "B"], "nameDest": ["B", "A"]}
        )
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            build_network_features(rows)

    def test_investigation_context_contains_only_prior_edges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.parquet"
            pd.DataFrame(
                {
                    "step": [1, 2],
                    "nameOrig": ["A", "A"],
                    "nameDest": ["B", "C"],
                    "amount": [50.0, 75.0],
                }
            ).to_parquet(path)
            context = build_investigation_network(
                path, step=2, sender="A", receiver="B"
            )
            at_first_step = build_investigation_network(
                path, step=1, sender="A", receiver="B"
            )

        self.assertTrue(context["prior_relationship_seen"])
        self.assertEqual(context["sender"]["prior_out_degree"], 1)
        self.assertEqual(context["sender"]["prior_receivers"][0]["account"], "B")
        self.assertFalse(at_first_step["prior_relationship_seen"])
        self.assertEqual(at_first_step["observed_edge_count"], 0)


class ModelEvidenceTests(unittest.TestCase):
    def test_validation_threshold_and_test_burden_metrics(self) -> None:
        labels = np.array([0, 0, 0, 1, 1])
        scores = np.array([0.05, 0.2, 0.4, 0.7, 0.9])
        threshold = select_f1_threshold(labels, scores)
        metrics = evaluate_scores(labels, scores, threshold)

        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertEqual(metrics["evaluation_rows"], 5)
        self.assertEqual(metrics["actual_fraud_rows"], 2)
        self.assertIn("false_positives_per_10000_negatives", metrics)

    def test_explanation_uses_native_model_contributions(self) -> None:
        rng = np.random.default_rng(14)
        frame = pd.DataFrame({"amount": rng.normal(size=80), "history": rng.normal(size=80)})
        labels = (frame["amount"] + frame["history"] > 0).astype(int)
        model = XGBClassifier(
            n_estimators=3,
            max_depth=2,
            n_jobs=1,
            eval_metric="logloss",
            random_state=1,
        ).fit(frame, labels)
        explanation = explain_prediction(model, frame.iloc[[0]])

        self.assertEqual(len(explanation), 2)
        self.assertTrue(all("contribution" in item and "feature" in item for item in explanation))

    def test_risk_labels_are_review_priorities_not_verdicts(self) -> None:
        self.assertEqual(risk_level(0.1, 0.2), "Low")
        self.assertEqual(risk_level(0.4, 0.2), "Review")
        self.assertEqual(risk_level(0.9, 0.2), "High review priority")


class NormalBehaviourAnomalyTests(unittest.TestCase):
    @staticmethod
    def _rows() -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "step": np.repeat(np.arange(1, 11), 3),
                "receiver_txn_count_before": np.tile([0, 1, 4], 10),
                "receiver_total_amount_before": np.tile([0.0, 50.0, 400.0], 10),
                "receiver_avg_amount_before": np.tile([0.0, 50.0, 100.0], 10),
                "receiver_steps_since_last": np.tile([-1, 2, 7], 10),
                "receiver_is_new": np.tile([1, 0, 0], 10),
                "receiver_txn_count_last24_before": np.tile([0, 1, 3], 10),
                "sender_txn_count_before": np.tile([0, 1, 0], 10),
                "sender_is_new": np.tile([1, 0, 1], 10),
            }
        )
        return frame

    def test_detector_fits_only_legitimate_rows(self) -> None:
        frame = self._rows()[ANOMALY_FEATURES]
        labels = np.zeros(len(frame), dtype=np.int8)
        labels[[2, 8, 15]] = 1
        detector = NormalBehaviourDetector(estimators=5).fit(frame, labels)

        self.assertEqual(detector.training_rows, len(frame))
        self.assertEqual(detector.normal_training_rows, len(frame) - 3)
        self.assertTrue(np.isfinite(detector.score(frame)).all())

    def test_causal_training_scores_leave_warmup_unscored_and_ignore_future(self) -> None:
        frame = self._rows()
        labels = np.zeros(len(frame), dtype=np.int8)
        labels[[4, 14, 22, 28]] = 1
        scores, eligible, _ = causal_training_scores(
            frame,
            labels,
            folds=2,
            estimators=5,
        )
        changed_future = frame.copy()
        changed_future.loc[changed_future["step"] >= 8, "receiver_avg_amount_before"] = 100_000
        changed_scores, changed_eligible, _ = causal_training_scores(
            changed_future,
            labels,
            folds=2,
            estimators=5,
        )

        self.assertTrue(np.isnan(scores[~eligible]).all())
        self.assertTrue(np.isfinite(scores[eligible]).all())
        self.assertTrue(np.array_equal(eligible, changed_eligible))
        first_scored_step = frame.loc[eligible, "step"].min()
        first_block = frame["step"] == first_scored_step
        np.testing.assert_array_equal(scores[first_block], changed_scores[first_block])

    def test_anomaly_transform_is_finite_and_rejects_missing_features(self) -> None:
        frame = self._rows()[ANOMALY_FEATURES]
        transformed = transform_anomaly_features(frame)
        self.assertTrue(np.isfinite(transformed).all())
        with self.assertRaisesRegex(ValueError, "Missing anomaly features"):
            transform_anomaly_features(frame.drop(columns=[ANOMALY_FEATURES[0]]))


class InvestigationStoreTests(unittest.TestCase):
    def test_status_and_note_persist_between_store_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "investigations.sqlite"
            store = InvestigationStore(path)
            saved = store.update(42, "Investigating", "Review beneficiary history")
            reopened = InvestigationStore(path).get(42)

        self.assertEqual(saved["status"], "Investigating")
        self.assertEqual(reopened["note"], "Review beneficiary history")
        self.assertTrue(reopened["updated_at"])

    def test_unknown_investigation_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InvestigationStore(Path(directory) / "investigations.sqlite")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                store.update(1, "Fraudulent")


if __name__ == "__main__":
    unittest.main()
