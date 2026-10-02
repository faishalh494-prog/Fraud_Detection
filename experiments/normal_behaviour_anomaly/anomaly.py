"""Offline normal-behaviour anomaly scoring using prior-only V1 features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = [
    "receiver_txn_count_before",
    "receiver_total_amount_before",
    "receiver_avg_amount_before",
    "receiver_steps_since_last",
    "receiver_is_new",
    "receiver_txn_count_last24_before",
    "sender_txn_count_before",
    "sender_is_new",
]
LOG_FEATURES = [
    "receiver_txn_count_before",
    "receiver_total_amount_before",
    "receiver_avg_amount_before",
    "receiver_txn_count_last24_before",
    "sender_txn_count_before",
]


def transform_anomaly_features(frame: pd.DataFrame) -> np.ndarray:
    missing = set(ANOMALY_FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing anomaly features: {sorted(missing)}")
    values = frame[ANOMALY_FEATURES].to_numpy(dtype=np.float32, copy=True)
    columns = {name: index for index, name in enumerate(ANOMALY_FEATURES)}
    for feature in LOG_FEATURES:
        index = columns[feature]
        if np.any(values[:, index] < 0):
            raise ValueError(f"{feature} must be nonnegative")
        values[:, index] = np.log1p(values[:, index])
    steps_index = columns["receiver_steps_since_last"]
    values[:, steps_index] = np.log1p(np.maximum(values[:, steps_index], 0))
    if not np.isfinite(values).all():
        raise ValueError("Anomaly features must be finite")
    return values


class NormalBehaviourDetector:
    """Isolation Forest fitted only to legitimate training examples."""

    def __init__(self, *, estimators: int = 100, random_state: int = 42):
        if estimators < 1:
            raise ValueError("estimators must be positive")
        self.estimators = estimators
        self.random_state = random_state
        self.model: IsolationForest | None = None
        self.training_rows = 0
        self.normal_training_rows = 0

    def fit(self, frame: pd.DataFrame, labels: np.ndarray) -> NormalBehaviourDetector:
        if len(frame) != len(labels):
            raise ValueError("Training rows and labels must have equal lengths")
        normal = np.asarray(labels) == 0
        self.training_rows = len(frame)
        self.normal_training_rows = int(normal.sum())
        if self.normal_training_rows < 2:
            raise ValueError("At least two legitimate training rows are required")
        values = transform_anomaly_features(frame)
        self.model = IsolationForest(
            n_estimators=self.estimators,
            max_samples=min(512, self.normal_training_rows),
            contamination="auto",
            random_state=self.random_state,
            n_jobs=4,
        )
        self.model.fit(values[normal])
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Anomaly detector has not been fitted")
        values = transform_anomaly_features(frame)
        return -self.model.score_samples(values)


def causal_training_scores(
    frame: pd.DataFrame,
    labels: np.ndarray,
    *,
    folds: int = 5,
    estimators: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, NormalBehaviourDetector]:
    """Score later training blocks using only earlier legitimate training rows.

    Rows in the initial warm-up block are deliberately excluded from the
    supervised comparison because no earlier training history exists to score
    them without looking ahead.
    """
    if folds < 2:
        raise ValueError("At least two folds are required for cross-fitting")
    if len(frame) != len(labels):
        raise ValueError("Training rows and labels must have equal lengths")
    if "step" not in frame:
        raise ValueError("Causal training scores require the chronological step column")

    labels = np.asarray(labels)
    steps = np.sort(frame["step"].unique())
    step_blocks = np.array_split(steps, folds + 1)
    if any(len(block) == 0 for block in step_blocks):
        raise ValueError("Not enough distinct training steps for the requested folds")

    scores = np.full(len(frame), np.nan, dtype=np.float32)
    eligible = np.zeros(len(frame), dtype=bool)
    for fold, block_steps in enumerate(step_blocks[1:]):
        first_step = block_steps[0]
        holdout = frame["step"].isin(block_steps).to_numpy()
        prior = frame["step"].to_numpy() < first_step
        if not prior.any() or not holdout.any():
            raise ValueError("Each scored training block requires earlier training history")
        detector = NormalBehaviourDetector(
            estimators=estimators,
            random_state=random_state + fold,
        ).fit(frame.loc[prior, ANOMALY_FEATURES], labels[prior])
        scores[holdout] = detector.score(frame.loc[holdout])
        eligible[holdout] = True

    final_detector = NormalBehaviourDetector(
        estimators=estimators,
        random_state=random_state,
    ).fit(frame[ANOMALY_FEATURES], labels)
    return scores, eligible, final_detector


def anomaly_metadata(detector: NormalBehaviourDetector) -> dict[str, Any]:
    return {
        "estimator": "IsolationForest",
        "features": ANOMALY_FEATURES,
        "transformed_features": {
            "log1p": LOG_FEATURES,
            "receiver_steps_since_last": "log1p(max(value, 0)); new-receiver flag remains a separate feature",
        },
        "fit_rows": detector.training_rows,
        "normal_fit_rows": detector.normal_training_rows,
        "fraud_rows_used_for_fit": 0,
        "label_use": (
            "Training labels are used only to exclude positive rows; labels are not "
            "features or a detector training target."
        ),
        "estimators": detector.estimators,
        "max_samples": min(512, detector.normal_training_rows),
        "contamination": "auto; score threshold selected on validation only",
        "training_score_method": (
            "chronological expanding-window Isolation Forest scores; each scored block "
            "uses legitimate rows from strictly earlier steps only"
        ),
        "score_direction": "higher is more anomalous",
    }
