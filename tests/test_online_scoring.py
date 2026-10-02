from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pydantic import ValidationError

from backend.app import TransactionScoreRequest
from src.features.syndicai_preprocessing import (
    build_behavioral_features,
    build_transaction_features,
    time_based_split,
)
from src.syndicai_v4.modeling import MODEL_FEATURES
from src.syndicai_v4.online_features import (
    BEHAVIOUR_FEATURES,
    HISTORY_COLUMNS,
    OnlineFeatureBuilder,
    build_behavioural_features_from_history,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "data" / "processed" / "syndicai_v1_processed.parquet"


def transaction(
    *,
    step: int = 5,
    transaction_type: str = "TRANSFER",
    amount: float = 10.0,
    sender: str = "sender",
    receiver: str = "receiver",
) -> dict[str, object]:
    return {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "nameOrig": sender,
        "nameDest": receiver,
    }


class OnlineFeatureTests(unittest.TestCase):
    def test_matches_v1_features_for_fixture_rows_and_excludes_same_step(self) -> None:
        rows = pd.DataFrame(
            [
                (1, "sender", "receiver", 10.25),
                (2, "sender", "receiver", 11.50),
                (2, "other", "receiver", 15.25),
                (3, "receiver", "other", 20.0),
                (5, "sender", "receiver", 5.0),
            ],
            columns=["step", "nameOrig", "nameDest", "amount"],
        )
        expected = rows.copy()
        expected["type"] = "TRANSFER"
        for column in (
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ):
            expected[column] = 0.0
        build_transaction_features(expected)
        build_behavioral_features(expected)
        time_based_split(expected)

        for row in expected.itertuples(index=False):
            event = transaction(
                step=row.step,
                sender=row.nameOrig,
                receiver=row.nameDest,
                amount=row.amount,
            )
            online = OnlineFeatureBuilder.build_from_history_frame(event, rows)
            for feature in BEHAVIOUR_FEATURES:
                self.assertEqual(online[feature], getattr(row, feature), feature)

        at_step_two = OnlineFeatureBuilder.build_from_history_frame(
            transaction(step=2), rows
        )
        self.assertEqual(at_step_two["receiver_txn_count_before"], 1)
        self.assertEqual(at_step_two["receiver_total_amount_before"], np.float32(10.25))

    def test_unseen_accounts_and_empty_history_have_v1_defaults(self) -> None:
        result = build_behavioural_features_from_history(
            transaction(step=4, sender="unseen-sender", receiver="unseen-receiver"),
            pd.DataFrame(columns=HISTORY_COLUMNS),
        )

        self.assertEqual(result["receiver_txn_count_before"], 0)
        self.assertEqual(result["receiver_total_amount_before"], 0.0)
        self.assertEqual(result["receiver_avg_amount_before"], 0.0)
        self.assertEqual(result["receiver_steps_since_last"], -1)
        self.assertEqual(result["receiver_is_new"], 1)
        self.assertEqual(result["receiver_txn_count_last24_before"], 0)
        self.assertEqual(result["sender_txn_count_before"], 0)
        self.assertEqual(result["sender_is_new"], 1)

    def test_same_step_and_future_history_do_not_change_features(self) -> None:
        event = transaction(step=10, sender="s", receiver="r")
        history = pd.DataFrame(
            [
                (2, 25.0, "s", "r"),
                (10, 50.0, "s", "r"),
                (11, 500.0, "s", "r"),
            ],
            columns=HISTORY_COLUMNS,
        )
        before = build_behavioural_features_from_history(event, history)
        past_only = build_behavioural_features_from_history(
            event,
            history.loc[history["step"] < 10],
        )
        history_with_more_future = pd.concat(
            [
                history,
                pd.DataFrame(
                    [(99, 10_000.0, "s", "r")],
                    columns=HISTORY_COLUMNS,
                ),
            ],
            ignore_index=True,
        )
        after = build_behavioural_features_from_history(event, history_with_more_future)

        self.assertEqual(before, past_only)
        self.assertEqual(before, after)
        self.assertEqual(before["receiver_txn_count_before"], 1)
        self.assertEqual(before["sender_txn_count_before"], 1)

    def test_online_builder_has_exact_model_feature_contract(self) -> None:
        class InMemoryBuilder(OnlineFeatureBuilder):
            def _account_history(self, *, sender: str, receiver: str, step: int) -> pd.DataFrame:
                del sender, receiver, step
                return pd.DataFrame(columns=HISTORY_COLUMNS)

        builder = InMemoryBuilder.__new__(InMemoryBuilder)
        self.assertEqual(list(builder.build(transaction(), model="A")), MODEL_FEATURES["A"])
        self.assertEqual(list(builder.build(transaction(), model="B")), MODEL_FEATURES["B"])
        with self.assertRaisesRegex(ValueError, "Model A or B"):
            builder.build(transaction(), model="C")

    @unittest.skipUnless(REFERENCE_PATH.is_file(), "Local validated processed data is unavailable")
    def test_sampled_real_transactions_match_v1_precomputed_features(self) -> None:
        parquet = pq.ParquetFile(REFERENCE_PATH)
        rng = np.random.default_rng(2402)
        sample_group_ids = rng.choice(
            parquet.num_row_groups,
            size=min(16, parquet.num_row_groups),
            replace=False,
        )
        required_columns = list(
            dict.fromkeys(HISTORY_COLUMNS + BEHAVIOUR_FEATURES + ["type"])
        )
        sampled_rows: list[dict[str, object]] = []
        for group_id in sample_group_ids:
            group = parquet.read_row_group(int(group_id), columns=required_columns)
            local_row = int(rng.integers(0, group.num_rows))
            sampled_rows.append(group.slice(local_row, 1).to_pylist()[0])

        builder = OnlineFeatureBuilder(REFERENCE_PATH)
        mismatches: list[tuple[int, str, object, object]] = []
        for row_index, row in enumerate(sampled_rows):
            event = {
                "step": row["step"],
                "type": row["type"],
                "amount": row["amount"],
                "nameOrig": row["nameOrig"],
                "nameDest": row["nameDest"],
            }
            online = builder.build(event, model="B")
            for feature in BEHAVIOUR_FEATURES:
                expected = row[feature]
                actual = online[feature]
                if actual != expected:
                    mismatches.append((row_index, feature, expected, actual))

        non_float_mismatches = [
            mismatch
            for mismatch in mismatches
            if mismatch[1] not in {"receiver_total_amount_before", "receiver_avg_amount_before"}
        ]
        float_differences = [
            abs(float(expected) - float(actual))
            for _, _, expected, actual in mismatches
        ]
        self.assertEqual(non_float_mismatches, [])
        self.assertTrue(
            all(difference <= 0.05 for difference in float_differences),
            f"Online/V1 amount feature differences exceed one small currency fraction: {mismatches}",
        )


class TransactionScoreRequestTests(unittest.TestCase):
    def test_valid_request_defaults_to_model_b(self) -> None:
        request = TransactionScoreRequest.model_validate(
            {
                "step": 10,
                "type": "TRANSFER",
                "amount": 50.5,
                "nameOrig": " C123 ",
                "nameDest": "C456",
            }
        )

        self.assertEqual(request.model, "B")
        self.assertEqual(request.nameOrig, "C123")

    def test_invalid_step_type_amount_and_blank_account_are_rejected(self) -> None:
        valid = {
            "step": 10,
            "type": "TRANSFER",
            "amount": 50.5,
            "nameOrig": "C123",
            "nameDest": "C456",
        }
        invalid_cases = [
            {**valid, "step": 0},
            {**valid, "step": 1.5},
            {**valid, "type": "UNKNOWN"},
            {**valid, "amount": -1},
            {**valid, "amount": float("nan")},
            {**valid, "nameOrig": "   "},
            {**valid, "model": "C"},
            {**valid, "oldbalanceOrg": 100},
        ]
        for case in invalid_cases:
            with self.subTest(case=case), self.assertRaises(ValidationError):
                TransactionScoreRequest.model_validate(case)


if __name__ == "__main__":
    unittest.main()
