from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.normal_behaviour_anomaly.anomaly import (
    ANOMALY_FEATURES,
    NormalBehaviourDetector,
    causal_training_scores,
    transform_anomaly_features,
)


class NormalBehaviourAnomalyTests(unittest.TestCase):
    @staticmethod
    def _rows() -> pd.DataFrame:
        return pd.DataFrame(
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


if __name__ == "__main__":
    unittest.main()
