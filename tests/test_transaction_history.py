from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic import ValidationError

from backend.app import TransactionScoreRequest
from src.syndicai_v4.modeling import MODEL_FEATURES
from src.syndicai_v4.online_features import (
    OnlineFeatureBuilder,
)
from src.syndicai_v4.service import RiskService
from src.syndicai_v4.transaction_history import (
    DuplicateTransactionError,
    TransactionHistory,
)


def transaction(
    step: int,
    *,
    amount: float = 25.0,
    sender: str = "sender",
    receiver: str = "receiver",
) -> dict[str, object]:
    return {
        "step": step,
        "type": "TRANSFER",
        "amount": amount,
        "nameOrig": sender,
        "nameDest": receiver,
    }


class TransactionHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.history_path = Path(self.temp_dir.name) / "history.sqlite"
        self.history = TransactionHistory(self.history_path, reference_max_step=0)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build_features(
        self,
        event: dict[str, object],
        history: TransactionHistory | None = None,
    ) -> dict[str, int | float]:
        store = history or self.history
        past_events = store.history_for(
            sender=str(event["nameOrig"]),
            receiver=str(event["nameDest"]),
            before_step=int(event["step"]),
        )
        return OnlineFeatureBuilder.build_from_history_frame(
            event,
            past_events,
        )

    def test_first_transaction_has_new_account_defaults_then_second_sees_first(self) -> None:
        first = transaction(10)
        first_features = self.build_features(first)
        self.assertEqual(first_features["receiver_is_new"], 1)
        self.assertEqual(first_features["sender_is_new"], 1)
        self.assertEqual(first_features["receiver_txn_count_before"], 0)

        self.history.record_scored_transaction(first)
        second = transaction(11, amount=40.0)
        second_features = self.build_features(second)

        self.assertEqual(second_features["receiver_is_new"], 0)
        self.assertEqual(second_features["sender_is_new"], 0)
        self.assertEqual(second_features["receiver_txn_count_before"], 1)
        self.assertEqual(second_features["receiver_total_amount_before"], 25.0)
        self.assertEqual(second_features["receiver_avg_amount_before"], 25.0)
        self.assertEqual(second_features["receiver_steps_since_last"], 1)
        self.assertEqual(second_features["receiver_txn_count_last24_before"], 1)
        self.assertEqual(second_features["sender_txn_count_before"], 1)

    def test_same_step_events_do_not_enter_each_others_history(self) -> None:
        same_step = transaction(10, amount=100.0)

        before = self.build_features(same_step)
        self.history.record_scored_transaction(same_step)
        features_after_record = self.build_features(same_step)

        self.assertEqual(before, features_after_record)
        self.assertEqual(features_after_record["receiver_txn_count_before"], 0)

    def test_future_events_cannot_affect_past_transaction_features(self) -> None:
        self.history.record_scored_transaction(transaction(20, amount=300.0))

        features = self.build_features(transaction(10))

        self.assertEqual(features["receiver_txn_count_before"], 0)
        self.assertEqual(features["sender_txn_count_before"], 0)

    def test_online_history_is_combined_with_read_only_reference_history(self) -> None:
        reference_path = Path(self.temp_dir.name) / "reference.parquet"
        pd.DataFrame(
            [
                {
                    "step": 1,
                    "amount": 10.0,
                    "nameOrig": "prior-sender",
                    "nameDest": "receiver",
                }
            ],
            columns=["step", "amount", "nameOrig", "nameDest"],
        ).to_parquet(reference_path)
        isolated_history = TransactionHistory(
            Path(self.temp_dir.name) / "isolated.sqlite",
            reference_max_step=2,
        )
        isolated_history.record_scored_transaction(
            transaction(3, amount=20.0, sender="online-sender")
        )
        with isolated_history._connect() as connection:
            connection.execute(
                """
                INSERT INTO scored_transactions
                    (event_id, step, amount, nameOrig, nameDest)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("legacy-overlap", 2, 500.0, "old-sender", "receiver"),
            )
        builder = OnlineFeatureBuilder(reference_path)

        features = builder.build(
            transaction(4, sender="new-sender"),
            additional_history=isolated_history.history_for(
                sender="new-sender",
                receiver="receiver",
                before_step=4,
            ),
        )

        self.assertEqual(features["receiver_txn_count_before"], 2)
        self.assertEqual(features["receiver_total_amount_before"], 30.0)
        self.assertEqual(features["receiver_avg_amount_before"], 15.0)

    @staticmethod
    def make_scoring_service(history: TransactionHistory, reference_max_step: int) -> RiskService:
        class FeatureBuilder:
            def build(
                self,
                transaction: dict[str, object],
                *,
                model: str,
                additional_history: pd.DataFrame,
            ) -> dict[str, int]:
                del transaction, additional_history
                return {feature: 0 for feature in MODEL_FEATURES[model]}

        class Model:
            def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
                del frame
                return np.array([[0.1, 0.9]])

        service = RiskService.__new__(RiskService)
        service.reference_max_step = reference_max_step
        service.transaction_history = history
        service.online_features = FeatureBuilder()
        service.bundle = {"B": Model()}
        service.metrics = {"models": {"B": {"validation": {"threshold": 0.5}}}}
        return service

    def test_reference_max_step_is_rejected_and_next_step_is_accepted(self) -> None:
        reference_max_step = 743
        history = TransactionHistory(
            Path(self.temp_dir.name) / "boundary.sqlite",
            reference_max_step=reference_max_step,
        )
        service = self.make_scoring_service(history, reference_max_step)

        with self.assertRaisesRegex(ValueError, "greater than the immutable reference maximum step \\(743\\)"):
            service.score_transaction(transaction(reference_max_step))

        with patch(
            "src.syndicai_v4.service.explain_prediction",
            return_value=[],
        ):
            result = service.score_transaction(transaction(reference_max_step + 1))
        self.assertTrue(result["history_updated"])
        self.assertEqual(
            len(
                history.history_for(
                    sender="sender",
                    receiver="receiver",
                    before_step=reference_max_step + 2,
                )
            ),
            1,
        )

    def test_restart_preserves_history_and_event_id_deduplicates(self) -> None:
        self.history.record_scored_transaction(
            transaction(4, amount=12.5),
            event_id="event-1",
        )
        restarted = TransactionHistory(self.history_path, reference_max_step=0)

        features = self.build_features(transaction(5), restarted)
        self.assertEqual(features["receiver_txn_count_before"], 1)
        self.assertEqual(features["receiver_total_amount_before"], 12.5)
        with self.assertRaises(DuplicateTransactionError):
            restarted.ensure_event_is_new("event-1")
        with self.assertRaises(DuplicateTransactionError):
            restarted.record_scored_transaction(transaction(5), event_id="event-1")

    def test_service_records_only_after_successful_scoring(self) -> None:
        service = self.make_scoring_service(self.history, reference_max_step=0)
        event = transaction(5)

        with patch("src.syndicai_v4.service.explain_prediction", return_value=[]):
            result = service.score_transaction(event, event_id="scored-1")

        self.assertTrue(result["history_updated"])
        later = self.build_features(transaction(6))
        self.assertEqual(later["receiver_txn_count_before"], 1)
        self.assertEqual(later["sender_txn_count_before"], 1)

    def test_service_does_not_record_when_scoring_fails(self) -> None:
        class BrokenModel:
            def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
                del frame
                raise RuntimeError("inference failed")

        service = self.make_scoring_service(self.history, reference_max_step=0)
        service.bundle = {"B": BrokenModel()}

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            service.score_transaction(transaction(5), event_id="failed")
        self.assertTrue(self.history.history_for(
            sender="sender",
            receiver="receiver",
            before_step=6,
        ).empty)

    def test_request_accepts_optional_event_id_and_rejects_blank(self) -> None:
        base = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 10.0,
            "nameOrig": "sender",
            "nameDest": "receiver",
        }
        self.assertIsNone(TransactionScoreRequest.model_validate(base).event_id)
        self.assertEqual(
            TransactionScoreRequest.model_validate(
                {**base, "event_id": " event-7 "}
            ).event_id,
            "event-7",
        )
        with self.assertRaises(ValidationError):
            TransactionScoreRequest.model_validate({**base, "event_id": "  "})


if __name__ == "__main__":
    unittest.main()
