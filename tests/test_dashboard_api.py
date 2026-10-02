from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.app import (
    ScoreRequest,
    TransactionScoreRequest,
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
            ) -> dict[str, object]:
                self.received = (transaction, model, event_id)
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
            result = score_transaction(request)

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
            ),
        )

    def test_historical_row_scoring_route_remains_available(self) -> None:
        class Service:
            def inspect(self, row_index: int, model: str) -> dict[str, object]:
                return {"row_index": row_index, "model": model}

        with patch("backend.app._service", return_value=Service()):
            result = score(ScoreRequest(row_index=7, model="B"))

        self.assertEqual(result, {"row_index": 7, "model": "B"})


if __name__ == "__main__":
    unittest.main()
