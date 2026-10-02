"""Transaction-time feature construction from strictly earlier V1 history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.syndicai_preprocessing import (
    VELOCITY_WINDOW_STEPS,
    _build_entity_step_summary,
)
from src.syndicai_v4.modeling import MODEL_FEATURES

HISTORY_COLUMNS = ["step", "amount", "nameOrig", "nameDest"]
BEHAVIOUR_FEATURES = [
    "receiver_txn_count_before",
    "receiver_total_amount_before",
    "receiver_avg_amount_before",
    "receiver_steps_since_last",
    "receiver_is_new",
    "receiver_txn_count_last24_before",
    "sender_txn_count_before",
    "sender_is_new",
]
TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


def _step_history_summary(
    history: pd.DataFrame,
    *,
    entity: str,
    account: str,
    current_step: int,
) -> pd.DataFrame:
    prior = history.loc[
        (history[entity] == account) & (history["step"] < current_step),
        ["step", "amount", entity],
    ]
    if prior.empty:
        return prior
    return _build_entity_step_summary(prior, entity)


def build_behavioural_features_from_history(
    transaction: dict[str, Any],
    history: pd.DataFrame,
) -> dict[str, int | float]:
    """Build Model B's history features from supplied history and no other rows."""
    required = set(HISTORY_COLUMNS)
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Historical frame is missing columns: {sorted(missing)}")

    step = int(transaction["step"])
    receiver = str(transaction["nameDest"])
    sender = str(transaction["nameOrig"])
    receiver_summary = _step_history_summary(
        history,
        entity="nameDest",
        account=receiver,
        current_step=step,
    )
    sender_summary = _step_history_summary(
        history,
        entity="nameOrig",
        account=sender,
        current_step=step,
    )

    if receiver_summary.empty:
        receiver_count = 0
        receiver_total = np.float32(0)
        receiver_last_step = None
        receiver_velocity = 0
    else:
        receiver_count = int(receiver_summary["cum_count_incl"].iloc[-1])
        receiver_total = np.float32(receiver_summary["cum_amount_incl"].iloc[-1])
        receiver_last_step = int(receiver_summary["step"].iloc[-1])
        window_start = step - VELOCITY_WINDOW_STEPS
        receiver_velocity = int(
            receiver_summary.loc[
                receiver_summary["step"] >= window_start, "step_count"
            ].sum()
        )

    if sender_summary.empty:
        sender_count = 0
    else:
        sender_count = int(sender_summary["cum_count_incl"].iloc[-1])

    receiver_average = (
        np.float32(receiver_total / receiver_count)
        if receiver_count
        else np.float32(0)
    )
    return {
        "receiver_txn_count_before": np.int32(receiver_count).item(),
        "receiver_total_amount_before": receiver_total.item(),
        "receiver_avg_amount_before": receiver_average.item(),
        "receiver_steps_since_last": (
            step - receiver_last_step if receiver_last_step is not None else -1
        ),
        "receiver_is_new": int(receiver_count == 0),
        "receiver_txn_count_last24_before": np.int32(receiver_velocity).item(),
        "sender_txn_count_before": np.int32(sender_count).item(),
        "sender_is_new": int(sender_count == 0),
    }


class OnlineFeatureBuilder:
    """Build online Model A/B features using the processed reference history."""

    def __init__(self, reference_path: str | Path):
        self.reference_path = Path(reference_path)
        if not self.reference_path.is_file():
            raise FileNotFoundError(f"Processed reference dataset not found: {self.reference_path}")

    @staticmethod
    def _transaction_features(transaction: dict[str, Any]) -> dict[str, int | float]:
        step = int(transaction["step"])
        amount = float(transaction["amount"])
        transaction_type = str(transaction["type"])
        features: dict[str, int | float] = {
            "step": step,
            "hour_of_day": step % 24,
            "day_of_sim": step // 24 + 1,
            "is_night": int(step % 24 in range(0, 6)),
            "amount": amount,
            "amount_log1p": np.float32(np.log1p(amount)).item(),
        }
        for category in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"):
            features[f"type_{category}"] = int(transaction_type == category)
        return features

    def _account_history(
        self,
        *,
        sender: str,
        receiver: str,
        step: int,
    ) -> pd.DataFrame:
        return pd.read_parquet(
            self.reference_path,
            columns=HISTORY_COLUMNS,
            filters=[
                [("nameDest", "==", receiver), ("step", "<", step)],
                [("nameOrig", "==", sender), ("step", "<", step)],
            ],
        )

    def build(
        self,
        transaction: dict[str, Any],
        *,
        model: str = "B",
    ) -> dict[str, int | float]:
        name = model.upper()
        if name not in {"A", "B"}:
            raise ValueError("Transaction-time scoring supports Model A or B only")

        features = self._transaction_features(transaction)
        if name == "B":
            step = int(transaction["step"])
            relevant_history = self._account_history(
                sender=str(transaction["nameOrig"]),
                receiver=str(transaction["nameDest"]),
                step=step,
            )
            features.update(
                build_behavioural_features_from_history(transaction, relevant_history)
            )

        expected = MODEL_FEATURES[name]
        missing = set(expected).difference(features)
        if missing:
            raise RuntimeError(f"Online feature builder omitted model features: {sorted(missing)}")
        return {feature: features[feature] for feature in expected}

    @classmethod
    def build_from_history_frame(
        cls,
        transaction: dict[str, Any],
        history: pd.DataFrame,
        *,
        model: str = "B",
    ) -> dict[str, int | float]:
        """Pure history-frame path used for parity checks and focused tests."""
        name = model.upper()
        if name not in {"A", "B"}:
            raise ValueError("Transaction-time scoring supports Model A or B only")
        features = cls._transaction_features(transaction)
        if name == "B":
            features.update(build_behavioural_features_from_history(transaction, history))
        expected = MODEL_FEATURES[name]
        missing = set(expected).difference(features)
        if missing:
            raise RuntimeError(f"Online feature builder omitted model features: {sorted(missing)}")
        return {feature: features[feature] for feature in expected}
