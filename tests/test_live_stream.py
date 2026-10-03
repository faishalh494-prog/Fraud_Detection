from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException, Request
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from backend.app import (
    TransactionScoreRequest,
    live_events,
    live_status,
    score_transaction,
)
from demo.benchmark_live import percentile, run_benchmark
from demo.live_stream import build_event, stream_events
from src.syndicai_v4.modeling import MODEL_FEATURES
from src.syndicai_v4.service import RiskService
from src.syndicai_v4.transaction_history import TransactionHistory

API_KEY = "test-live-api-key-" + ("k" * 32)


def make_service(history_path: Path) -> RiskService:
    history = TransactionHistory(history_path, reference_max_step=743)

    class FeatureBuilder:
        def build(
            self,
            transaction: dict[str, object],
            *,
            model: str,
            additional_history: pd.DataFrame,
        ) -> dict[str, int | float]:
            features: dict[str, int | float] = {
                name: 0 for name in MODEL_FEATURES[model]
            }
            receiver_count = int(
                (additional_history["nameDest"] == transaction["nameDest"]).sum()
            )
            sender_count = int(
                (additional_history["nameOrig"] == transaction["nameOrig"]).sum()
            )
            features.update(
                {
                    "receiver_txn_count_before": receiver_count,
                    "receiver_total_amount_before": float(
                        additional_history.loc[
                            additional_history["nameDest"] == transaction["nameDest"],
                            "amount",
                        ].sum()
                    ),
                    "receiver_avg_amount_before": (
                        float(
                            additional_history.loc[
                                additional_history["nameDest"] == transaction["nameDest"],
                                "amount",
                            ].mean()
                        )
                        if receiver_count
                        else 0.0
                    ),
                    "receiver_steps_since_last": (
                        int(transaction["step"])
                        - int(
                            additional_history.loc[
                                additional_history["nameDest"] == transaction["nameDest"],
                                "step",
                            ].max()
                        )
                        if receiver_count
                        else -1
                    ),
                    "receiver_is_new": int(receiver_count == 0),
                    "receiver_txn_count_last24_before": receiver_count,
                    "sender_txn_count_before": sender_count,
                    "sender_is_new": int(sender_count == 0),
                }
            )
            return features

    class Model:
        fail = False

        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            if self.fail:
                raise RuntimeError("test inference failure")
            del frame
            return np.array([[0.1, 0.9]])

    service = RiskService.__new__(RiskService)
    service.reference_max_step = 743
    service.transaction_history = history
    service.online_features = FeatureBuilder()
    service.bundle = {"B": Model()}
    service.metrics = {"models": {"B": {"validation": {"threshold": 0.5}}}}
    return service


def live_payload(step: int, event_id: str, *, amount: float = 10.0) -> dict[str, object]:
    return {
        "step": step,
        "type": "TRANSFER",
        "amount": amount,
        "nameOrig": "sender-live",
        "nameDest": "receiver-live",
        "event_id": event_id,
        "model": "B",
    }


def call_score_transaction(
    service: RiskService,
    payload: dict[str, object],
) -> dict[str, object]:
    with patch("backend.app._service", return_value=service):
        return score_transaction(
            TransactionScoreRequest.model_validate(payload),
            Request({"type": "http", "state": {"request_id": "live-test"}}),
        )


class LiveStreamTests(unittest.TestCase):
    def test_demo_event_sequence_is_causal_chronological_and_label_free(self) -> None:
        events = [build_event(i, first_step=744, run_id="test-run") for i in range(8)]

        self.assertEqual([event["step"] for event in events], list(range(744, 752)))
        self.assertEqual(
            [event["type"] for event in events[:4]],
            ["PAYMENT", "PAYMENT", "CASH_OUT", "CASH_OUT"],
        )
        self.assertEqual(events[0]["nameOrig"], events[3]["nameOrig"])
        self.assertEqual(events[0]["nameDest"], events[3]["nameDest"])
        self.assertEqual(len({event["event_id"] for event in events}), len(events))
        self.assertTrue(all("isFraud" not in event for event in events))

    def test_simulator_submits_sequentially_to_the_scoring_endpoint(self) -> None:
        responses = [
            {"risk": {"score": 1}, "evidence_strength": {"status": "New"}, "timings": {"processing_ms": 1}},
            {"risk": {"score": 2}, "evidence_strength": {"status": "Limited history"}, "timings": {"processing_ms": 2}},
            {"risk": {"score": 3}, "evidence_strength": {"status": "Limited history"}, "timings": {"processing_ms": 3}},
        ]
        observed: list[tuple[str, dict[str, object] | None]] = []

        def send(url: str, *, api_key: str, payload=None):
            self.assertEqual(api_key, API_KEY)
            observed.append((url, payload))
            return responses[len(observed) - 1], 5.0

        with (
            patch("demo.live_stream._request", side_effect=send),
            patch("demo.live_stream.time.sleep") as sleep,
        ):
            produced = list(
                stream_events(
                    api_url="http://localhost:8000/",
                    api_key=API_KEY,
                    first_step=744,
                    run_id="controlled",
                    interval=0.01,
                    max_events=3,
                )
            )

        self.assertEqual(len(produced), 3)
        self.assertEqual(
            [item[0]["step"] for item in produced],
            [744, 745, 746],
        )
        self.assertEqual(
            [url for url, _ in observed],
            ["http://localhost:8000/score_transaction"] * 3,
        )
        self.assertEqual([payload["step"] for _, payload in observed], [744, 745, 746])
        self.assertEqual(sleep.call_count, 2)

    def test_live_api_processes_events_persists_state_and_exposes_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "online.sqlite")
            with patch(
                "src.syndicai_v4.service.explain_prediction",
                return_value=[
                    {
                        "feature": "amount",
                        "label": "Transaction amount",
                        "value": 10.0,
                        "contribution": 0.2,
                        "direction": "increases",
                    }
                ],
            ):
                first = call_score_transaction(service, live_payload(744, "live-1"))
                second = call_score_transaction(service, live_payload(745, "live-2"))
                same_step = call_score_transaction(
                    service,
                    live_payload(745, "live-same-step"),
                )
                later = call_score_transaction(service, live_payload(746, "live-3"))

                self.assertEqual(first["evidence_strength"]["status"], "New")
                self.assertEqual(
                    second["features"]["receiver_txn_count_before"],
                    1,
                )
                self.assertEqual(
                    same_step["features"]["receiver_txn_count_before"],
                    1,
                )
                self.assertEqual(
                    later["features"]["receiver_txn_count_before"],
                    3,
                )
                self.assertGreater(first["timings"]["state_update_ms"], 0)

                with self.assertRaises(HTTPException) as duplicate:
                    call_score_transaction(service, live_payload(747, "live-1"))
                self.assertEqual(duplicate.exception.status_code, 409)

                with self.assertRaises(HTTPException) as bad_step:
                    call_score_transaction(service, live_payload(743, "invalid-step"))
                self.assertEqual(bad_step.exception.status_code, 422)

                with patch("backend.app._service", return_value=service):
                    feed = live_events(limit=10)
                    status = live_status()
                self.assertEqual(feed["event_count"], 4)
                self.assertEqual(len(feed["events"]), 4)
                self.assertEqual(status["latest_step"], 746)
                reopened = TransactionHistory(
                    service.transaction_history.path,
                    reference_max_step=743,
                )
                persisted_feed = reopened.recent_live_events(limit=10)
                self.assertEqual(len(persisted_feed), 4)
                self.assertEqual(
                    persisted_feed[0]["event_key"],
                    "live-3",
                )

    def test_failed_scoring_does_not_change_history_or_live_feed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = make_service(Path(directory) / "failed.sqlite")
            service.bundle["B"].fail = True
            with self.assertRaisesRegex(RuntimeError, "test inference failure"):
                call_score_transaction(service, live_payload(744, "failed-event"))
            self.assertEqual(service.live_status()["event_count"], 0)
            self.assertTrue(
                service.transaction_history.history_for(
                    sender="sender-live",
                    receiver="receiver-live",
                    before_step=745,
                ).empty
            )

    def test_live_api_requires_authentication_and_rejects_invalid_payload(self) -> None:
        from backend.app import _api_key_configured, _api_key_matches

        with patch.dict("os.environ", {"SYNDICAI_API_KEY": API_KEY}):
            self.assertTrue(_api_key_configured())
            self.assertTrue(_api_key_matches(API_KEY))
            self.assertFalse(_api_key_matches("invalid"))
        with self.assertRaises(ValidationError):
            TransactionScoreRequest.model_validate(
                {
                    **live_payload(744, "invalid"),
                    "extra_field": "rejected",
                }
            )

    def test_benchmark_path_uses_real_http_result_shape_and_reports_method(self) -> None:
        def fake_request(url: str, *, api_key: str, payload=None):
            self.assertEqual(api_key, API_KEY)
            if url.endswith("/live/status"):
                return {
                    "reference_max_step": 743,
                    "latest_step": None,
                }, 0.1
            assert payload is not None
            return {
                "timings": {
                    "feature_ms": 1.0,
                    "inference_ms": 2.0,
                    "explanation_ms": 3.0,
                    "state_update_ms": 0.5,
                    "processing_ms": 6.5,
                }
            }, 8.0

        with patch("demo.benchmark_live._request", side_effect=fake_request):
            result = run_benchmark(
                api_url="http://localhost:8000",
                api_key=API_KEY,
                sample_count=3,
                run_id="bench-test",
            )

        self.assertEqual(result["sample_count"], 3)
        self.assertGreater(result["events_per_second"], 0)
        self.assertEqual(result["latency_ms"]["feature_ms"]["median"], 1.0)
        self.assertEqual(result["latency_ms"]["end_to_end_ms"]["median"], 8.0)
        self.assertIn("no scoring warm-up", result["methodology"])
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.95), 3.0)


class LiveDashboardTests(unittest.TestCase):
    @staticmethod
    def response_for_url(url: str) -> dict[str, object]:
        if url.endswith("/models"):
            return {
                "models": {
                    name: {
                        "validation": {"threshold": 0.5},
                        "test": {
                            "alerts": 1,
                            "pr_auc": 0.5,
                            "precision": 0.5,
                            "recall": 0.5,
                            "f1": 0.5,
                        },
                    }
                    for name in ("A", "B", "C")
                }
            }
        if url.endswith("/health"):
            return {"status": "ready"}
        if "/alerts?" in url:
            return []
        if url.endswith("/live/status"):
            return {
                "event_count": 1,
                "flagged_event_count": 1,
                "latest_processed_at": "2026-10-03T00:00:00+00:00",
                "latest_step": 744,
                "reference_max_step": 743,
            }
        if url.endswith("/live/events?limit=50"):
            return {
                "event_count": 1,
                "events": [
                    {
                        "event_key": "demo-000004",
                        "processing_state": "Processed",
                        "processed_at": "2026-10-03T00:00:00+00:00",
                        "model": "B",
                        "risk": {
                            "score": 99.0,
                            "score_kind": "model score; not a calibrated probability",
                            "calibrated_probability": None,
                            "review_priority": "High review priority",
                            "review_threshold": 97.69,
                            "flagged_for_review": True,
                        },
                        "transaction": {
                            "step": 744,
                            "type": "CASH_OUT",
                            "amount": 1_000_000.0,
                            "sender": "sender",
                            "receiver": "receiver",
                        },
                        "evidence_strength": {
                            "status": "Limited history",
                            "sender_prior_transactions": 3,
                            "receiver_prior_transactions": 3,
                            "established_history_minimum": 5,
                            "interpretation": "Prior-history coverage only.",
                        },
                        "explanation": {
                            "method": "XGBoost TreeSHAP contributions",
                            "summary": "Score-increasing evidence.",
                            "caveat": "Not proof of fraud.",
                            "reasons": [
                                {
                                    "feature": "amount",
                                    "label": "Transaction amount",
                                    "value": 1_000_000.0,
                                    "direction": "increases",
                                    "contribution": 0.2,
                                }
                            ],
                        },
                        "behavioural_evidence": {"sender_txn_count_before": 3},
                        "timings": {
                            "feature_ms": 1.0,
                            "inference_ms": 2.0,
                            "explanation_ms": 3.0,
                            "state_update_ms": 0.5,
                            "processing_ms": 6.5,
                        },
                    }
                ],
            }
        raise AssertionError(f"Unexpected dashboard request: {url}")

    def test_streamlit_live_monitor_workflow_displays_refreshing_scored_event(self) -> None:
        class FakeResponse:
            def __init__(self, body: bytes):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                del args
                return False

            def read(self) -> bytes:
                return self.body

        def fake_urlopen(request, timeout=60):
            del timeout
            payload = LiveDashboardTests.response_for_url(request.full_url)
            return FakeResponse(json.dumps(payload).encode("utf-8"))

        root = Path(__file__).resolve().parents[1]
        with (
            patch.dict(
                "os.environ",
                {
                    "SYNDICAI_API_KEY": API_KEY,
                    "SYNDICAI_API_URL": "http://syndicai.test",
                },
            ),
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
        ):
            tester = AppTest.from_file(
                str(root / "frontend" / "streamlit_app.py"),
                default_timeout=10,
            ).run()
            self.assertFalse(tester.exception)
            self.assertTrue(
                any("Situational awareness" in item.value for item in tester.markdown)
            )
            tester.radio(key="dashboard_workflow").set_value("LIVE MONITOR").run()

        self.assertFalse(tester.exception)
        self.assertTrue(
            any("Live event monitoring" in item.value for item in tester.markdown)
        )
        self.assertTrue(
            any("demo-000004" in str(item.value) for item in tester.dataframe)
        )
        self.assertTrue(
            any("Automatic refresh every 2 seconds" in item.value for item in tester.caption)
        )


if __name__ == "__main__":
    unittest.main()
