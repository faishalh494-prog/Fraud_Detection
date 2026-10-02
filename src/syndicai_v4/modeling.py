"""Model definitions, threshold selection, evaluation, and explanations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from xgboost import DMatrix, XGBClassifier

from src.features.syndicai_preprocessing import MODEL_A_FEATURES, MODEL_B_FEATURES
from src.syndicai_v4.network import NETWORK_FEATURES

MODEL_FEATURES = {
    "A": list(MODEL_A_FEATURES),
    "B": list(MODEL_B_FEATURES),
    "C": list(MODEL_B_FEATURES) + NETWORK_FEATURES,
}

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "amount_log1p": "Log transaction amount",
    "step": "Simulation time step",
    "hour_of_day": "Hour of day",
    "day_of_sim": "Day of simulation",
    "is_night": "Night-time indicator",
    "type_CASH_IN": "Cash-in transaction",
    "type_CASH_OUT": "Cash-out transaction",
    "type_DEBIT": "Debit transaction",
    "type_PAYMENT": "Payment transaction",
    "type_TRANSFER": "Transfer transaction",
    "receiver_txn_count_before": "Prior transactions to receiver",
    "receiver_total_amount_before": "Prior received amount",
    "receiver_avg_amount_before": "Average prior received amount",
    "receiver_steps_since_last": "Steps since receiver activity",
    "receiver_is_new": "Receiver has no prior observed activity",
    "receiver_txn_count_last24_before": "Receiver transactions in prior 24 steps",
    "sender_txn_count_before": "Prior transactions from sender",
    "sender_is_new": "Sender has no prior observed activity",
    "network_sender_prior_in_degree": "Sender's prior counterparties as receiver",
    "network_sender_in_degree_last24": "Sender's recent inbound network activity",
    "network_receiver_prior_out_degree": "Receiver's prior counterparties as sender",
    "network_receiver_out_degree_last24": "Receiver's recent outbound network activity",
}


def new_model(negative_count: int, positive_count: int, *, estimators: int = 80) -> XGBClassifier:
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("Training data must contain both classes")
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        max_bin=128,
        n_estimators=estimators,
        max_depth=5,
        learning_rate=0.08,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=1.0,
        reg_lambda=2.0,
        scale_pos_weight=negative_count / positive_count,
        n_jobs=4,
        random_state=42,
    )


def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.nanargmax(f1))
    return float(thresholds[best])


def evaluate_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    negatives = max(int(np.sum(labels == 0)), 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "alerts": int(tp + fp),
        "alert_burden_pct": float(100 * (tp + fp) / len(labels)),
        "false_positive_rate_pct": float(100 * fp / negatives),
        "false_positives_per_10000_negatives": float(10_000 * fp / negatives),
        "evaluation_rows": int(len(labels)),
        "actual_fraud_rows": int(np.sum(labels)),
    }


def explain_prediction(
    model: XGBClassifier,
    features: pd.DataFrame,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return feature contributions from XGBoost's native TreeSHAP values."""
    if len(features) != 1:
        raise ValueError("Explanation expects exactly one transaction")
    contributions = model.get_booster().predict(
        DMatrix(features, feature_names=list(features.columns)),
        pred_contribs=True,
    )[0][:-1]
    ranked = sorted(
        zip(features.columns, features.iloc[0].tolist(), contributions.tolist()),
        key=lambda item: item[2],
        reverse=True,
    )
    return [
        {
            "feature": name,
            "label": FEATURE_LABELS.get(name, name.replace("_", " ").title()),
            "value": value.item() if isinstance(value, np.generic) else value,
            "contribution": float(contribution),
            "direction": "increases" if contribution > 0 else "decreases",
        }
        for name, value, contribution in ranked[:limit]
    ]


def risk_level(score: float, threshold: float) -> str:
    if score < threshold:
        return "Low"
    if score >= max(threshold, 0.8):
        return "High review priority"
    return "Review"
