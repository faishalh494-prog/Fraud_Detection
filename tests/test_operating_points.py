from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.syndicai_v4.modeling import MODEL_FEATURES
from src.syndicai_v4.service import RiskService


class AmountScoreModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = frame["amount"].to_numpy(dtype=float)
        return np.column_stack((1 - positive, positive))


class AlertOperatingPointTests(unittest.TestCase):
    def test_policy_thresholds_derive_from_validation_and_report_test_workload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            val_scores = np.array([0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.20, 0.10])
            test_scores = np.array([0.98, 0.92, 0.85, 0.40])
            for split, scores, labels in (
                ("val", val_scores, [1, 1, 0, 1, 0, 0, 0, 0]),
                ("test", test_scores, [1, 0, 1, 0]),
            ):
                frame = pd.DataFrame(
                    {feature: np.zeros(len(scores)) for feature in MODEL_FEATURES["B"]}
                )
                frame["amount"] = scores
                frame["isFraud"] = labels
                frame.to_parquet(data_dir / f"syndicai_v1_{split}.parquet")

            service = RiskService.__new__(RiskService)
            service.data_dir = data_dir
            service.bundle = {"B": AmountScoreModel()}
            service.metrics = {
                "models": {
                    "B": {
                        "validation": {
                            "threshold": 0.6,
                        }
                    }
                }
            }
            service._operating_points_cache = None
            policies = service.alert_operating_points()

        self.assertEqual(len(policies), 6)
        baseline = policies[0]
        self.assertEqual(baseline["id"], "max_f1")
        self.assertEqual(baseline["threshold"], 0.6)
        self.assertEqual(policies[1]["id"], "budget_0_10")
        self.assertGreater(policies[1]["threshold"], max(val_scores))
        self.assertEqual(policies[1]["validation"]["alerts"], 0)
        self.assertEqual(policies[1]["test"]["alerts"], 0)
        self.assertTrue(policies[1]["test_workload_is_historical"])


if __name__ == "__main__":
    unittest.main()
