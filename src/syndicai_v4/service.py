"""Transaction retrieval and explainable investigation service."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.syndicai_v4.investigations import InvestigationStore
from src.syndicai_v4.modeling import (
    MODEL_FEATURES,
    evaluate_scores,
    explain_prediction,
    risk_level,
    select_alert_budget_threshold,
)
from src.syndicai_v4.network import NETWORK_FEATURES, build_investigation_network
from src.syndicai_v4.online_features import OnlineFeatureBuilder
from src.syndicai_v4.transaction_history import TransactionHistory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models"
REFERENCE_NAME = "syndicai_v1_processed.parquet"
ALERT_BUDGETS = (
    ("budget_0_10", "Validation target 0.10%", 0.001),
    ("budget_0_25", "Validation target 0.25%", 0.0025),
    ("budget_0_50", "Validation target 0.50%", 0.005),
    ("budget_1_00", "Validation target 1.00%", 0.01),
    ("budget_2_00", "Validation target 2.00%", 0.02),
)
ESTABLISHED_HISTORY_MINIMUM = 5


def history_evidence(
    model_name: str,
    features: dict[str, Any],
) -> dict[str, Any]:
    """Describe behavioural history coverage without implying model confidence."""
    name = model_name.upper()
    if name == "A":
        return {
            "status": "Not used",
            "sender_prior_transactions": None,
            "receiver_prior_transactions": None,
            "established_history_minimum": ESTABLISHED_HISTORY_MINIMUM,
            "interpretation": "Model A does not use behavioural history.",
        }

    sender_count = int(features.get("sender_txn_count_before", 0))
    receiver_count = int(features.get("receiver_txn_count_before", 0))
    if sender_count == 0 and receiver_count == 0:
        status = "New"
    elif (
        sender_count >= ESTABLISHED_HISTORY_MINIMUM
        and receiver_count >= ESTABLISHED_HISTORY_MINIMUM
    ):
        status = "Established history"
    else:
        status = "Limited history"
    return {
        "status": status,
        "sender_prior_transactions": sender_count,
        "receiver_prior_transactions": receiver_count,
        "established_history_minimum": ESTABLISHED_HISTORY_MINIMUM,
        "interpretation": (
            "Coverage describes the available prior behavioural history only; "
            "limited or absent history is not evidence of low or high fraud risk."
        ),
    }


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
        self._operating_points_cache: list[dict[str, Any]] | None = None
        self.reference_max_step = self._reference_max_step(
            self.data_dir / REFERENCE_NAME
        )
        self.network = ParquetRowStore(network_path)
        self.online_features = OnlineFeatureBuilder(self.data_dir / REFERENCE_NAME)
        state_dir = Path(os.environ.get("SYNDICAI_STATE_DIR", self.artifact_dir))
        self.transaction_history = TransactionHistory(
            state_dir / "online_history.sqlite",
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

    def live_status(self) -> dict[str, Any]:
        return {
            **self.transaction_history.live_status(),
            "reference_max_step": self.reference_max_step,
        }

    def recent_live_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.transaction_history.recent_live_events(limit)

    def alert_operating_points(self) -> list[dict[str, Any]]:
        """Return Model B policies selected on validation and evaluated on test."""
        if self._operating_points_cache is not None:
            return self._operating_points_cache

        validation_path = self.data_dir / "syndicai_v1_val.parquet"
        test_path = self.data_dir / "syndicai_v1_test.parquet"
        required = [validation_path, test_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Required validation/test data for alert operating points is missing."
            )

        features = MODEL_FEATURES["B"]
        validation = pd.read_parquet(
            validation_path,
            columns=["isFraud", *features],
        )
        validation_labels = validation.pop("isFraud").to_numpy(dtype=np.int8)
        validation_scores = self.bundle["B"].predict_proba(validation[features])[:, 1]

        choices: list[tuple[str, str, float, int | None]] = [
            (
                "max_f1",
                "Existing validation maximum-F1 default",
                float(self.metrics["models"]["B"]["validation"]["threshold"]),
                None,
            )
        ]
        for policy_id, label, target in ALERT_BUDGETS:
            threshold, validation_alerts = select_alert_budget_threshold(
                validation_scores,
                target,
            )
            choices.append((policy_id, label, threshold, validation_alerts))

        test = pd.read_parquet(test_path, columns=["isFraud", *features])
        test_labels = test.pop("isFraud").to_numpy(dtype=np.int8)
        test_scores = self.bundle["B"].predict_proba(test[features])[:, 1]
        result = []
        for policy_id, label, threshold, validation_alerts in choices:
            validation_metrics = evaluate_scores(
                validation_labels,
                validation_scores,
                threshold,
            )
            test_metrics = evaluate_scores(test_labels, test_scores, threshold)
            result.append(
                {
                    "id": policy_id,
                    "label": label,
                    "threshold": threshold,
                    "validation": validation_metrics,
                    "test": test_metrics,
                    "validation_alerts_from_budget_selector": validation_alerts,
                    "test_workload_is_historical": True,
                }
            )
        self._operating_points_cache = result
        return result

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
                "review_priority": str(row.risk_level),
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
                "calibrated_probability": None,
                "review_threshold": round(threshold * 100, 2),
                "flagged_for_review": score >= threshold,
                "review_priority": risk_level(score, threshold),
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
            "evidence_strength": history_evidence(name, features),
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
        operating_point: str = "max_f1",
    ) -> dict[str, Any]:
        name = model_name.upper()
        if name not in {"A", "B"}:
            raise ValueError("Transaction-time scoring supports Model A or B only")
        if name == "A" and operating_point != "max_f1":
            raise ValueError("Alert-budget operating points are available for Model B only")
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
        feature_started = time.perf_counter()
        values = self.online_features.build(
            transaction,
            model=name,
            additional_history=prior_online_history,
        )
        feature_ms = (time.perf_counter() - feature_started) * 1000
        feature_frame = pd.DataFrame([values], columns=MODEL_FEATURES[name])
        inference_started = time.perf_counter()
        score = float(self.bundle[name].predict_proba(feature_frame)[0, 1])
        inference_ms = (time.perf_counter() - inference_started) * 1000
        if operating_point == "max_f1":
            threshold = float(self.metrics["models"][name]["validation"]["threshold"])
        else:
            selected = next(
                (
                    point
                    for point in self.alert_operating_points()
                    if point["id"] == operating_point
                ),
                None,
            )
            if selected is None:
                raise ValueError("Unknown validation-selected alert operating point")
            threshold = float(selected["threshold"])
        explanation_started = time.perf_counter()
        contributions = explain_prediction(self.bundle[name], feature_frame)
        explanation_ms = (time.perf_counter() - explanation_started) * 1000
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
        transaction_summary = {
            "step": step,
            "type": str(transaction["type"]),
            "amount": float(transaction["amount"]),
            "sender": str(transaction["nameOrig"]),
            "receiver": str(transaction["nameDest"]),
        }
        result = {
            "model": name,
            "operating_point": operating_point,
            "event_id": event_id,
            "transaction": transaction_summary,
            "risk": {
                "score": round(score * 100, 2),
                "score_kind": "model score; not a calibrated probability",
                "calibrated_probability": None,
                "review_priority": risk_level(score, threshold),
                "review_threshold": round(threshold * 100, 2),
                "flagged_for_review": score >= threshold,
            },
            "evidence_strength": history_evidence(name, values),
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
            "timings": {
                "feature_ms": round(feature_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "explanation_ms": round(explanation_ms, 3),
                "processing_ms": round(
                    feature_ms + inference_ms + explanation_ms,
                    3,
                ),
            },
        }
        state_update_ms = self.transaction_history.record_scored_transaction(
            transaction,
            event_id=event_id,
            result=result,
        )
        result["timings"]["state_update_ms"] = round(state_update_ms, 3)
        result["timings"]["processing_ms"] = round(
            feature_ms + inference_ms + explanation_ms + state_update_ms,
            3,
        )
        return result
