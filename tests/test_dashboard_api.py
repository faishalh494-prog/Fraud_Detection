from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import Request

from backend.app import (
    ScoreRequest,
    TransactionScoreRequest,
    _api_key_configured,
    _api_key_matches,
    _audit_action,
    health,
    score,
    score_transaction,
    transaction_limits,
)


class DashboardApiTests(unittest.TestCase):
    def test_transaction_limits_exposes_reference_boundary(self) -> None:
        class Service:
            reference_max_step = 743

        with patch("backend.app._service", return_value=Service()):
            self.assertEqual(transaction_limits(), {"reference_max_step": 743})

    def test_dashboard_submission_uses_existing_transaction_score_contract(self) -> None:
        class Service:
            def __init__(self) -> None:
                self.received: tuple[dict[str, object], str, str | None] | None = None

            def score_transaction(
                self,
                transaction: dict[str, object],
                model: str,
                *,
                event_id: str | None,
                operating_point: str,
            ) -> dict[str, object]:
                self.received = (transaction, model, event_id, operating_point)
                return {"model": model, "history_updated": True}

        service = Service()
        request = TransactionScoreRequest.model_validate(
            {
                "step": 744,
                "type": "TRANSFER",
                "amount": 20.0,
                "nameOrig": "sender-1",
                "nameDest": "receiver-1",
                "model": "B",
                "event_id": "dashboard-event",
            }
        )

        with patch("backend.app._service", return_value=service):
            result = score_transaction(
                request,
                Request({"type": "http", "state": {"request_id": "unit-test"}}),
            )

        self.assertEqual(result, {"model": "B", "history_updated": True})
        self.assertEqual(
            service.received,
            (
                {
                    "step": 744,
                    "type": "TRANSFER",
                    "amount": 20.0,
                    "nameOrig": "sender-1",
                    "nameDest": "receiver-1",
                },
                "B",
                "dashboard-event",
                "max_f1",
            ),
        )

    def test_historical_row_scoring_route_remains_available(self) -> None:
        class Service:
            def inspect(self, row_index: int, model: str) -> dict[str, object]:
                return {"row_index": row_index, "model": model}

        with patch("backend.app._service", return_value=Service()):
            result = score(
                ScoreRequest(row_index=7, model="B"),
                Request({"type": "http", "state": {"request_id": "unit-test"}}),
            )

        self.assertEqual(result, {"row_index": 7, "model": "B"})

    def test_api_key_requires_configured_long_secret_and_constant_time_match(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_api_key_configured())
            self.assertFalse(_api_key_matches("anything"))
        key = "test-key-" + ("x" * 32)
        with patch.dict("os.environ", {"SYNDICAI_API_KEY": key}):
            self.assertTrue(_api_key_configured())
            self.assertTrue(_api_key_matches(key))
            self.assertFalse(_api_key_matches(key + "wrong"))
        with patch.dict("os.environ", {"SYNDICAI_API_KEY": "short"}):
            self.assertFalse(_api_key_configured())
            self.assertFalse(_api_key_matches("short"))

    def test_health_and_audit_log_do_not_disclose_secret_or_artifact_paths(self) -> None:
        secret = "never-log-this-api-key-" + ("x" * 32)
        with patch.dict("os.environ", {"SYNDICAI_API_KEY": secret}):
            status = health()
        self.assertNotIn("missing_artifacts", status)
        self.assertNotIn("C:\\", str(status))

        request = Request(
            {
                "type": "http",
                "headers": [(b"x-api-key", secret.encode())],
                "state": {"request_id": "audit-test"},
            }
        )
        with self.assertLogs("syndicai.audit", level="INFO") as captured:
            _audit_action(
                request,
                action="score_transaction",
                success=True,
                model="B",
                step=744,
            )
        output = "".join(captured.output)
        self.assertIn("investigation_action", output)
        self.assertIn("score_transaction", output)
        self.assertNotIn(secret, output)


if __name__ == "__main__":
    unittest.main()
