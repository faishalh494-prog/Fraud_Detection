"""Transaction retrieval and explainable investigation service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.syndicai_v4.investigations import InvestigationStore
from src.syndicai_v4.modeling import (
    MODEL_FEATURES,
    explain_prediction,
    risk_level,
)
from src.syndicai_v4.network import NETWORK_FEATURES, build_investigation_network
from src.syndicai_v4.online_features import OnlineFeatureBuilder
from src.syndicai_v4.transaction_history import TransactionHistory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models"
REFERENCE_NAME = "syndicai_v1_processed.parquet"


class ParquetRowStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file = pq.ParquetFile(self.path)
        self.row_counts = [self.file.metadata.row_group(i).num_rows for i in range(self.file.num_row_groups)]
        self.total_rows = sum(self.row_counts)

    def read_row(self, row_index: int, columns: list[str] | None = None) -> dict[str, Any]:
        if row_index < 0 or row_index >= self.total_rows:
            raise IndexError(f"Transaction row index must be from 0 to {self.total_rows - 1}")
        offset = 0
        for group_index, row_count in enumerate(self.row_counts):
            if row_index < offset + row_count:
                table = self.file.read_row_group(group_index, columns=columns)
                row = table.slice(row_index - offset, 1).to_pylist()[0]
                return row
            offset += row_count
        raise IndexError(f"Transaction row index is out of range: {row_index}")


class RiskService:
    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    ):
        self.data_dir = Path(data_dir)
        self.artifact_dir = Path(artifact_dir)
        bundle_path = self.artifact_dir / "model_bundle.joblib"
        metrics_path = self.artifact_dir / "metrics.json"
        network_path = self.artifact_dir / "network_features.parquet"
        required = [
            self.data_dir / REFERENCE_NAME,
            bundle_path,
            metrics_path,
            network_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required V4 artifacts are missing: "
                + ", ".join(missing)
                + ". Run `python -m src.syndicai_v4.train` first."
            )

        self.transactions = ParquetRowStore(self.data_dir / REFERENCE_NAME)
        self.reference_max_step = self._reference_max_step(
            self.data_dir / REFERENCE_NAME
        )
        self.network = ParquetRowStore(network_path)
        self.online_features = OnlineFeatureBuilder(self.data_dir / REFERENCE_NAME)
        self.transaction_history = TransactionHistory(
            self.artifact_dir / "online_history.sqlite",
            reference_max_step=self.reference_max_step,
        )
        self.bundle = joblib.load(bundle_path)
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.investigations = InvestigationStore(self.artifact_dir / "investigations.sqlite")
        self.alerts = {
            model: pd.read_parquet(self.artifact_dir / f"test_alerts_{model}.parquet")
            for model in MODEL_FEATURES
        }

    @staticmethod
    def _reference_max_step(path: Path) -> int:
        parquet = pq.ParquetFile(path)
        step_column_index = parquet.schema_arrow.names.index("step")
        row_group_maxima: list[int] = []
        for row_group_index in range(parquet.num_row_groups):
            statistics = parquet.metadata.row_group(row_group_index).column(
                step_column_index
            ).statistics
            if statistics is None or not statistics.has_min_max:
                steps = pd.read_parquet(path, columns=["step"])["step"]
                return int(steps.max())
            row_group_maxima.append(int(statistics.max))
        if not row_group_maxima:
            raise ValueError(f"Reference dataset has no rows: {path}")
        return max(row_group_maxima)

    def model_summary(self) -> dict[str, Any]:
        return self.metrics

    def alert_queue(self, model: str, limit: int = 100) -> list[dict[str, Any]]:
        name = model.upper()
        if name not in self.alerts:
            raise ValueError("Model must be A, B, or C")
        if limit < 1 or limit > 500:
            raise ValueError("Alert limit must be between 1 and 500")
        queue = self.alerts[name].head(limit)
        return [
            {
                "row_index": int(row.row_index),
                "step": int(row.step),
                "transaction_type": str(row.type),
                "amount": float(row.amount),
                "risk_score": round(float(row.risk_score) * 100, 1),
                "risk_level": str(row.risk_level),
                "status": str(self.investigations.get(int(row.row_index))["status"]),
            }
            for row in queue.itertuples(index=False)
        ]

    def inspect(self, row_index: int, model_name: str = "C") -> dict[str, Any]:
        name = model_name.upper()
        if name not in MODEL_FEATURES:
            raise ValueError("Model must be A, B, or C")
        raw = self.transactions.read_row(row_index)
        network_values = self.network.read_row(row_index)
        features = {feature: raw[feature] for feature in MODEL_FEATURES[name] if feature not in NETWORK_FEATURES}
        if name == "C":
            features.update({feature: network_values[feature] for feature in NETWORK_FEATURES})
        feature_frame = pd.DataFrame([features], columns=MODEL_FEATURES[name])
        score = float(self.bundle[name].predict_proba(feature_frame)[0, 1])
        threshold = float(self.metrics["models"][name]["validation"]["threshold"])

        reasons = explain_prediction(self.bundle[name], feature_frame)
        for reason in reasons:
            reason["contribution"] = round(reason["contribution"], 4)
        history = build_investigation_network(
            self.data_dir / REFERENCE_NAME,
            step=int(raw["step"]),
            sender=str(raw["nameOrig"]),
            receiver=str(raw["nameDest"]),
        )
        ranges = self.metrics["dataset"]["split_row_ranges"]
        period = next(
            split
            for split, limits in ranges.items()
            if limits["start_inclusive"] <= row_index < limits["end_exclusive"]
        )

        def finite(value: Any) -> Any:
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return None
            if isinstance(value, np.generic):
                return value.item()
            return value

        return {
            "row_index": row_index,
            "evaluation_period": period,
            "score_context": (
                "Held-out test prediction"
                if period == "test"
                else f"{period.title()}-period score; this is not an independent test prediction"
            ),
            "model": name,
            "transaction": {
                "step": int(raw["step"]),
                "transaction_type": str(raw["type"]),
                "amount": float(raw["amount"]),
                "sender": str(raw["nameOrig"]),
                "receiver": str(raw["nameDest"]),
                "sender_balance_before": finite(raw["oldbalanceOrg"]),
                "receiver_balance_before": finite(raw["oldbalanceDest"]),
            },
            "risk": {
                "score": round(score * 100, 2),
                "score_kind": "model score; not a calibrated probability",
                "review_threshold": round(threshold * 100, 2),
                "flagged_for_review": score >= threshold,
                "level": risk_level(score, threshold),
            },
            "explanation": {
                "method": "XGBoost TreeSHAP contributions",
                "caveat": "Contributions are model evidence, not proof of fraud.",
                "reasons": reasons,
            },
            "behaviour": {
                key: finite(value)
                for key, value in features.items()
                if key.startswith("receiver_") or key.startswith("sender_")
            },
            "network_features": (
                {key: int(features[key]) for key in NETWORK_FEATURES} if name == "C" else {}
            ),
            "network_context": history,
            "investigation": self.investigations.get(row_index),
        }

    def score_transaction(
        self,
        transaction: dict[str, Any],
        model_name: str = "B",
        *,
        event_id: str | None = None,
    ) -> dict[str, Any]:
        name = model_name.upper()
        if name not in {"A", "B"}:
            raise ValueError("Transaction-time scoring supports Model A or B only")
        step = int(transaction["step"])
        if step <= self.reference_max_step:
            raise ValueError(
                "Transaction-time scoring requires step greater than the "
                f"immutable reference maximum step ({self.reference_max_step})"
            )
        self.transaction_history.ensure_event_is_new(event_id)
        prior_online_history = self.transaction_history.history_for(
            sender=str(transaction["nameOrig"]),
            receiver=str(transaction["nameDest"]),
            before_step=step,
        )
        values = self.online_features.build(
            transaction,
            model=name,
            additional_history=prior_online_history,
        )
        feature_frame = pd.DataFrame([values], columns=MODEL_FEATURES[name])
        score = float(self.bundle[name].predict_proba(feature_frame)[0, 1])
        threshold = float(self.metrics["models"][name]["validation"]["threshold"])
        contributions = explain_prediction(self.bundle[name], feature_frame)
        reasons = [
            {
                "feature": reason["feature"],
                "label": reason["label"],
                "value": reason["value"],
                "contribution": round(float(reason["contribution"]), 4),
                "direction": reason["direction"],
            }
            for reason in contributions
        ]
        positive_reasons = [reason for reason in reasons if reason["contribution"] > 0][:3]
        if positive_reasons:
            explanation_text = "Score-increasing model evidence: " + "; ".join(
                f"{reason['label']}={reason['value']}"
                for reason in positive_reasons
            ) + "."
        else:
            explanation_text = "No listed feature increased this model score."
        behavior = {
            key: values[key]
            for key in values
            if key.startswith("receiver_") or key.startswith("sender_")
        }
        result = {
            "model": name,
            "risk": {
                "score": round(score * 100, 2),
                "score_kind": "model score; not a calibrated probability",
                "risk_level": risk_level(score, threshold),
                "review_threshold": round(threshold * 100, 2),
                "flagged_for_review": score >= threshold,
            },
            "explanation": {
                "method": "XGBoost TreeSHAP contributions",
                "summary": explanation_text,
                "caveat": "Model evidence supports review; it does not establish fraud.",
                "reasons": reasons,
            },
            "behavioural_evidence": behavior,
            "features": values,
            "history_rule": (
                "Only reference and previously scored online transactions with "
                "step strictly less than this event were used."
            ),
            "history_updated": True,
        }
        self.transaction_history.record_scored_transaction(
            transaction,
            event_id=event_id,
        )
        return result
