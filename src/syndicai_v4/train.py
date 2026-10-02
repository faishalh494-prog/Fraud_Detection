"""Train and compare SyndicAI V4 models using the validated V1 splits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.syndicai_v4.modeling import (
    MODEL_FEATURES,
    evaluate_scores,
    new_model,
    risk_level,
    select_f1_threshold,
)
from src.syndicai_v4.network import NETWORK_FEATURES, build_network_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models"
SPLITS = ("train", "val", "test")


def _read_split(
    data_dir: Path,
    split: str,
    model_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    columns = list(dict.fromkeys(model_features + ["isFraud", "type"]))
    frame = pd.read_parquet(data_dir / f"syndicai_v1_{split}.parquet", columns=columns)
    labels = frame.pop("isFraud").to_numpy(dtype=np.int8)
    return frame, labels


def _json_number(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_number(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def train(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    estimators: int = 80,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        split: data_dir / f"syndicai_v1_{split}.parquet"
        for split in SPLITS
    }
    reference_path = data_dir / "syndicai_v1_processed.parquet"
    missing = [str(path) for path in [*paths.values(), reference_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Expected validated V1 Parquet artifacts; missing: " + ", ".join(missing)
        )

    counts = {
        split: int(pd.read_parquet(path, columns=["isFraud"])["isFraud"].size)
        for split, path in paths.items()
    }
    total_rows = sum(counts.values())
    network_path = artifact_dir / "network_features.parquet"
    print(f"Building four prior-step network features for {total_rows:,} transactions...")
    network_input = pd.read_parquet(reference_path, columns=["step", "nameOrig", "nameDest"])
    if len(network_input) != total_rows:
        raise ValueError(
            f"Split rows total {total_rows:,}, reference rows {len(network_input):,}"
        )
    if not network_input["step"].is_monotonic_increasing:
        raise ValueError("Reference dataset is not chronologically ordered")
    network = build_network_features(network_input)
    network_summary = {
        feature: {
            "nonzero_rows": int((network[feature] > 0).sum()),
            "nonzero_pct": float(100 * (network[feature] > 0).mean()),
            "maximum": int(network[feature].max()),
        }
        for feature in NETWORK_FEATURES
    }
    network.insert(0, "row_index", np.arange(total_rows, dtype=np.int64))
    pq.write_table(
        pa.Table.from_pandas(network, preserve_index=False),
        network_path,
        row_group_size=65_536,
    )
    del network_input, network

    split_frames: dict[str, pd.DataFrame] = {}
    split_labels: dict[str, np.ndarray] = {}
    offsets: dict[str, int] = {}
    offset = 0
    graph_rows = pd.read_parquet(network_path, columns=["row_index", *NETWORK_FEATURES])
    for split in SPLITS:
        offsets[split] = offset
        frame, labels = _read_split(data_dir, split, MODEL_FEATURES["C"][:-len(NETWORK_FEATURES)])
        graph_values = graph_rows.iloc[offset : offset + len(frame)]
        expected_ids = np.arange(offset, offset + len(frame), dtype=np.int64)
        if not np.array_equal(graph_values["row_index"].to_numpy(), expected_ids):
            raise ValueError(f"Network feature row alignment failed for {split}")
        for feature in NETWORK_FEATURES:
            frame[feature] = graph_values[feature].to_numpy(dtype=np.int32)
        split_frames[split] = frame
        split_labels[split] = labels
        offset += len(frame)
    del graph_rows
    if offset != total_rows:
        raise ValueError("Split row counts do not match the full reference dataset")

    bundle: dict[str, Any] = {}
    report: dict[str, Any] = {
        "dataset": {
            "reference": reference_path.name,
            "split_files": {key: path.name for key, path in paths.items()},
            "split_rows": counts,
            "training_rows": counts["train"],
            "validation_rows": counts["val"],
            "test_rows": counts["test"],
            "training_positives": int(split_labels["train"].sum()),
            "validation_positives": int(split_labels["val"].sum()),
            "test_positives": int(split_labels["test"].sum()),
            "split_strategy": "existing chronological step split; no re-splitting",
            "split_row_ranges": {
                split: {
                    "start_inclusive": offsets[split],
                    "end_exclusive": offsets[split] + counts[split],
                }
                for split in SPLITS
            },
        },
        "training": {
            "estimator": "XGBoostClassifier",
            "estimators": estimators,
            "max_depth": 5,
            "learning_rate": 0.08,
            "min_child_weight": 2,
            "subsample": 0.85,
            "colsample_bytree": 1.0,
            "max_bin": 128,
            "random_seed": 42,
            "n_jobs": 4,
            "imbalance": "scale_pos_weight=negative/positive rows in training split",
            "threshold_selection": "maximum F1 on validation split",
        },
        "models": {},
        "network_features": NETWORK_FEATURES,
        "network_feature_summary": network_summary,
        "test_alerts_note": "Test-split predictions are for final evaluation/demo investigation only.",
    }

    for model_name in ("A", "B", "C"):
        start = time.monotonic()
        features = MODEL_FEATURES[model_name]
        train_x = split_frames["train"][features].to_numpy(dtype=np.float32)
        val_x = split_frames["val"][features].to_numpy(dtype=np.float32)
        test_x = split_frames["test"][features].to_numpy(dtype=np.float32)
        y_train = split_labels["train"]
        y_val = split_labels["val"]
        y_test = split_labels["test"]
        model = new_model(
            int(np.sum(y_train == 0)),
            int(np.sum(y_train == 1)),
            estimators=estimators,
        )
        print(
            f"Training Model {model_name} ({len(features)} features, "
            f"{len(y_train):,} training rows, {estimators} trees)..."
        )
        model.fit(train_x, y_train, verbose=False)
        val_scores = model.predict_proba(val_x)[:, 1]
        threshold = select_f1_threshold(y_val, val_scores)
        test_scores = model.predict_proba(test_x)[:, 1]
        validation_metrics = evaluate_scores(y_val, val_scores, threshold)
        test_metrics = evaluate_scores(y_test, test_scores, threshold)
        raw_gain = model.get_booster().get_score(importance_type="gain")
        total_gain = sum(raw_gain.values()) or 1.0
        gain_by_feature = {
            feature: float(raw_gain.get(f"f{index}", 0.0) / total_gain)
            for index, feature in enumerate(features)
        }

        queue_mask = test_scores >= threshold
        queue = pd.DataFrame(
            {
                "row_index": np.arange(
                    offsets["test"], offsets["test"] + len(test_scores), dtype=np.int64
                )[queue_mask],
                "step": split_frames["test"].loc[queue_mask, "step"].to_numpy(),
                "type": split_frames["test"].loc[queue_mask, "type"].astype(str).to_numpy(),
                "amount": split_frames["test"].loc[queue_mask, "amount"].to_numpy(),
                "risk_score": test_scores[queue_mask],
                "risk_level": [
                    risk_level(float(score), threshold) for score in test_scores[queue_mask]
                ],
            }
        ).sort_values("risk_score", ascending=False)
        queue.to_parquet(artifact_dir / f"test_alerts_{model_name}.parquet", index=False)
        bundle[model_name] = model
        report["models"][model_name] = {
            "features": features,
            "validation": validation_metrics,
            "test": test_metrics,
            "gain_importance_share": gain_by_feature,
            "fit_seconds": round(time.monotonic() - start, 2),
            "test_alert_queue_rows": int(len(queue)),
        }
        print(
            f"  Model {model_name}: validation F1={validation_metrics['f1']:.4f}, "
            f"test PR-AUC={test_metrics['pr_auc']:.4f}, "
            f"test alerts={test_metrics['alerts']:,}"
        )
        del train_x, val_x, test_x, val_scores, test_scores, queue

    report["comparisons_vs_B"] = {
        model_name: {
            metric: float(
                report["models"][model_name]["test"][metric]
                - report["models"]["B"]["test"][metric]
            )
            for metric in ("precision", "recall", "f1", "pr_auc", "alerts")
        }
        for model_name in ("A", "C")
    }
    joblib.dump(bundle, artifact_dir / "model_bundle.joblib", compress=3)
    (artifact_dir / "metrics.json").write_text(
        json.dumps(_json_number(report), indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--estimators", type=int, default=80)
    args = parser.parse_args()
    if args.estimators < 1:
        parser.error("--estimators must be positive")
    result = train(args.data_dir, args.artifact_dir, estimators=args.estimators)
    print(json.dumps(_json_number(result), indent=2))


if __name__ == "__main__":
    main()
