"""Controlled B versus B + normal-behaviour anomaly experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.syndicai_v4.anomaly import (
    ANOMALY_FEATURES,
    anomaly_metadata,
    causal_training_scores,
)
from src.syndicai_v4.modeling import evaluate_scores, new_model, select_f1_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "anomaly_experiment"
SPLITS = ("train", "val", "test")
BASELINE_NAME = "B"


def _read_split(
    data_dir: Path,
    split: str,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_parquet(
        data_dir / f"syndicai_v1_{split}.parquet",
        columns=list(dict.fromkeys(features + ANOMALY_FEATURES + ["isFraud"])),
    )
    labels = frame.pop("isFraud").to_numpy(dtype=np.int8)
    return frame, labels


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_experiment(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    estimators: int = 100,
    folds: int = 5,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        split: data_dir / f"syndicai_v1_{split}.parquet"
        for split in SPLITS
    }
    baseline_path = model_dir / "model_bundle.joblib"
    baseline_metrics_path = model_dir / "metrics.json"
    missing = [
        str(path)
        for path in [*split_paths.values(), baseline_path, baseline_metrics_path]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing V1 data or baseline model artifacts: " + ", ".join(missing))

    baseline_bundle = joblib.load(baseline_path)
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline = baseline_bundle[BASELINE_NAME]
    baseline_features = baseline_metrics["models"][BASELINE_NAME]["features"]
    frames: dict[str, pd.DataFrame] = {}
    labels: dict[str, np.ndarray] = {}
    for split in SPLITS:
        frame, target = _read_split(data_dir, split, baseline_features)
        frames[split] = frame
        labels[split] = target

    train_frame = frames["train"]
    started = time.monotonic()
    print(
        f"Scoring {folds} causal normal-only Isolation Forest blocks "
        f"over {len(train_frame):,} training rows..."
    )
    train_anomaly, eligible_train, detector = causal_training_scores(
        train_frame,
        labels["train"],
        folds=folds,
        estimators=estimators,
    )
    val_anomaly = detector.score(frames["val"][ANOMALY_FEATURES])
    test_anomaly = detector.score(frames["test"][ANOMALY_FEATURES])
    anomaly_fit_seconds = time.monotonic() - started

    val_anomaly_threshold = select_f1_threshold(labels["val"], val_anomaly)
    test_anomaly_metrics = evaluate_scores(
        labels["test"],
        test_anomaly,
        val_anomaly_threshold,
    )

    baseline_val = baseline.predict_proba(
        frames["val"][baseline_features].to_numpy(dtype=np.float32)
    )[:, 1]
    baseline_test = baseline.predict_proba(
        frames["test"][baseline_features].to_numpy(dtype=np.float32)
    )[:, 1]
    baseline_threshold = float(baseline_metrics["models"]["B"]["validation"]["threshold"])
    baseline_val_metrics = evaluate_scores(labels["val"], baseline_val, baseline_threshold)
    baseline_test_metrics = evaluate_scores(labels["test"], baseline_test, baseline_threshold)

    eligible_labels = labels["train"][eligible_train]
    eligible_features = train_frame.loc[eligible_train, baseline_features]
    matched_baseline = new_model(
        int(np.sum(eligible_labels == 0)),
        int(np.sum(eligible_labels == 1)),
        estimators=int(baseline_metrics["training"]["estimators"]),
    )
    augmented = new_model(
        int(np.sum(eligible_labels == 0)),
        int(np.sum(eligible_labels == 1)),
        estimators=int(baseline_metrics["training"]["estimators"]),
    )
    train_augmented = eligible_features.copy()
    train_augmented["normal_anomaly_score"] = train_anomaly[eligible_train]
    val_augmented = frames["val"][baseline_features].copy()
    val_augmented["normal_anomaly_score"] = val_anomaly
    test_augmented = frames["test"][baseline_features].copy()
    test_augmented["normal_anomaly_score"] = test_anomaly
    print("Training supervised B + anomaly model with the same XGBoost configuration...")
    matched_baseline.fit(
        eligible_features.to_numpy(dtype=np.float32),
        eligible_labels,
        verbose=False,
    )
    augmented.fit(train_augmented.to_numpy(dtype=np.float32), eligible_labels, verbose=False)
    matched_val = matched_baseline.predict_proba(
        frames["val"][baseline_features].to_numpy(dtype=np.float32)
    )[:, 1]
    matched_threshold = select_f1_threshold(labels["val"], matched_val)
    matched_test = matched_baseline.predict_proba(
        frames["test"][baseline_features].to_numpy(dtype=np.float32)
    )[:, 1]
    val_combined = augmented.predict_proba(
        val_augmented.to_numpy(dtype=np.float32)
    )[:, 1]
    combined_threshold = select_f1_threshold(labels["val"], val_combined)
    test_combined = augmented.predict_proba(
        test_augmented.to_numpy(dtype=np.float32)
    )[:, 1]

    report: dict[str, Any] = {
        "dataset": {
            "split_files": {split: path.name for split, path in split_paths.items()},
            "split_rows": {split: int(len(labels[split])) for split in SPLITS},
            "positive_rows": {
                split: int(labels[split].sum())
                for split in SPLITS
            },
            "supervised_rows_with_causal_anomaly_score": int(eligible_train.sum()),
            "warmup_rows_excluded_from_supervised_comparison": int((~eligible_train).sum()),
            "training_score_blocks": int(folds),
            "split_strategy": "existing chronological V1 splits; no rows re-split",
        },
        "anomaly_detector": anomaly_metadata(detector),
        "threshold_selection": (
            "Anomaly-only and B+anomaly F1-maximizing thresholds are independently "
            "selected on validation labels; test data is not used for selection."
        ),
        "models": {
            "B": {
                "description": "Existing trained Model B and its existing validation-selected threshold",
                "validation": baseline_val_metrics,
                "test": baseline_test_metrics,
            },
            "B_matched_training_rows": {
                "description": (
                    "B-only control refitted on the same later training rows "
                    "available to the causal B + anomaly experiment."
                ),
                "validation": evaluate_scores(labels["val"], matched_val, matched_threshold),
                "test": evaluate_scores(labels["test"], matched_test, matched_threshold),
            },
            "anomaly_only": {
                "description": "Isolation Forest anomaly score thresholded as an investigation signal",
                "validation": evaluate_scores(
                    labels["val"], val_anomaly, val_anomaly_threshold
                ),
                "test": test_anomaly_metrics,
            },
            "B_plus_anomaly": {
                "description": "Same Model B feature set plus a strictly prior normal anomaly score",
                "features": baseline_features + ["normal_anomaly_score"],
                "validation": evaluate_scores(
                    labels["val"], val_combined, combined_threshold
                ),
                "test": evaluate_scores(
                    labels["test"], test_combined, combined_threshold
                ),
                "gain_importance_share": {},
            },
        },
        "comparison_B_plus_anomaly_minus_B": {},
        "comparison_B_plus_anomaly_minus_matched_B": {},
        "anomaly_fit_seconds": round(anomaly_fit_seconds, 2),
        "decision": {},
    }
    raw_gain = augmented.get_booster().get_score(importance_type="gain")
    total_gain = sum(raw_gain.values()) or 1.0
    all_features = baseline_features + ["normal_anomaly_score"]
    report["models"]["B_plus_anomaly"]["gain_importance_share"] = {
        feature: float(raw_gain.get(f"f{index}", 0.0) / total_gain)
        for index, feature in enumerate(all_features)
    }
    comparison_metrics = (
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_positives",
        "alerts",
        "alert_burden_pct",
        "false_positives_per_10000_negatives",
    )
    report["comparison_B_plus_anomaly_minus_B"] = {
        metric: float(
            report["models"]["B_plus_anomaly"]["test"][metric]
            - report["models"]["B"]["test"][metric]
        )
        for metric in comparison_metrics
    }
    report["comparison_B_plus_anomaly_minus_matched_B"] = {
        metric: float(
            report["models"]["B_plus_anomaly"]["test"][metric]
            - report["models"]["B_matched_training_rows"]["test"][metric]
        )
        for metric in comparison_metrics
    }
    report["decision"] = {
        "improves_test_pr_auc_vs_existing_B": (
            report["models"]["B_plus_anomaly"]["test"]["pr_auc"]
            > report["models"]["B"]["test"]["pr_auc"]
        ),
        "improves_test_f1_vs_existing_B": (
            report["models"]["B_plus_anomaly"]["test"]["f1"]
            > report["models"]["B"]["test"]["f1"]
        ),
        "improves_test_pr_auc_vs_matched_B": (
            report["models"]["B_plus_anomaly"]["test"]["pr_auc"]
            > report["models"]["B_matched_training_rows"]["test"]["pr_auc"]
        ),
        "improves_test_f1_vs_matched_B": (
            report["models"]["B_plus_anomaly"]["test"]["f1"]
            > report["models"]["B_matched_training_rows"]["test"]["f1"]
        ),
        "model_feature_gain_share": report["models"]["B_plus_anomaly"][
            "gain_importance_share"
        ]["normal_anomaly_score"],
        "available_in_investigation_api": False,
        "recommendation": (
            "Promote only if held-out PR-AUC and F1 improve over both existing B and the matched-row B control."
            if (
                report["models"]["B_plus_anomaly"]["test"]["pr_auc"]
                > report["models"]["B"]["test"]["pr_auc"]
                and report["models"]["B_plus_anomaly"]["test"]["pr_auc"]
                > report["models"]["B_matched_training_rows"]["test"]["pr_auc"]
                and report["models"]["B_plus_anomaly"]["test"]["f1"]
                > report["models"]["B"]["test"]["f1"]
                and report["models"]["B_plus_anomaly"]["test"]["f1"]
                > report["models"]["B_matched_training_rows"]["test"]["f1"]
            )
            else (
                "Do not add anomaly score to the deployed risk or investigation API; "
                "retain it as an offline experimental signal."
            )
        ),
    }

    joblib.dump(
        {
            "detector": detector,
            "anomaly_features": ANOMALY_FEATURES,
            "score_direction": "higher is more anomalous",
            "validation_threshold": val_anomaly_threshold,
        },
        output_dir / "normal_behaviour_detector.joblib",
        compress=3,
    )
    joblib.dump(augmented, output_dir / "model_b_plus_anomaly.joblib", compress=3)
    scores = pd.DataFrame(
        {
            "row_index": np.arange(
                len(labels["train"]),
                len(labels["train"]) + len(labels["val"]) + len(labels["test"]),
                dtype=np.int64,
            ),
            "normal_anomaly_score": np.concatenate([val_anomaly, test_anomaly]),
        }
    )
    scores.to_parquet(output_dir / "val_test_anomaly_scores.parquet", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(_normalize_json(report), indent=2),
        encoding="utf-8",
    )
    print(
        "B test PR-AUC/F1: "
        f"{report['models']['B']['test']['pr_auc']:.4f}/"
        f"{report['models']['B']['test']['f1']:.4f}; "
        "B + anomaly: "
        f"{report['models']['B_plus_anomaly']['test']['pr_auc']:.4f}/"
        f"{report['models']['B_plus_anomaly']['test']['f1']:.4f}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--estimators", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if args.estimators < 1:
        parser.error("--estimators must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    report = run_experiment(
        args.data_dir,
        args.model_dir,
        args.output_dir,
        estimators=args.estimators,
        folds=args.folds,
    )
    print(json.dumps(_normalize_json(report), indent=2))


if __name__ == "__main__":
    main()
