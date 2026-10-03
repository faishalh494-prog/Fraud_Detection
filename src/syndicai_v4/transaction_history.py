"""Persistent history for transaction-time behavioural scoring."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd

HISTORY_COLUMNS = ["step", "amount", "nameOrig", "nameDest"]


class DuplicateTransactionError(ValueError):
    """Raised when a supplied event identifier has already been scored."""


class TransactionHistory:
    """Store scored transaction facts without changing the V1 reference data."""

    def __init__(self, path: str | Path, *, reference_max_step: int):
        self.path = Path(path)
        self.reference_max_step = int(reference_max_step)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scored_transactions (
                    event_id TEXT UNIQUE,
                    step INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    nameOrig TEXT NOT NULL,
                    nameDest TEXT NOT NULL,
                    result_json TEXT,
                    processed_at TEXT,
                    flagged_for_review INTEGER,
                    state_update_ms REAL,
                    investigation_status TEXT NOT NULL DEFAULT 'Open',
                    investigation_note TEXT NOT NULL DEFAULT '',
                    investigation_updated_at TEXT
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(scored_transactions)"
                ).fetchall()
            }
            migrations = {
                "result_json": "TEXT",
                "processed_at": "TEXT",
                "flagged_for_review": "INTEGER",
                "state_update_ms": "REAL",
                "investigation_status": "TEXT NOT NULL DEFAULT 'Open'",
                "investigation_note": "TEXT NOT NULL DEFAULT ''",
                "investigation_updated_at": "TEXT",
            }
            for column, sql_type in migrations.items():
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE scored_transactions ADD COLUMN {column} {sql_type}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS scored_transactions_step
                ON scored_transactions(step)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS scored_transactions_sender_step
                ON scored_transactions(nameOrig, step)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS scored_transactions_receiver_step
                ON scored_transactions(nameDest, step)
                """
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _normalize_event_id(event_id: str | None) -> str | None:
        if event_id is None:
            return None
        normalized = event_id.strip()
        if not normalized:
            raise ValueError("event_id must not be blank")
        return normalized

    def ensure_event_is_new(self, event_id: str | None) -> None:
        normalized = self._normalize_event_id(event_id)
        if normalized is None:
            return
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM scored_transactions WHERE event_id = ?",
                (normalized,),
            ).fetchone()
        if existing is not None:
            raise DuplicateTransactionError(
                f"Transaction event_id has already been scored: {normalized}"
            )

    def history_for(
        self,
        *,
        sender: str,
        receiver: str,
        before_step: int,
    ) -> pd.DataFrame:
        """Return relevant online events; feature construction applies strict causality."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT step, amount, nameOrig, nameDest
                FROM scored_transactions
                WHERE step > ?
                  AND step < ?
                  AND (nameOrig = ? OR nameDest = ?)
                ORDER BY step
                """,
                (self.reference_max_step, before_step, sender, receiver),
            ).fetchall()
        return pd.DataFrame(rows, columns=HISTORY_COLUMNS)

    def record_scored_transaction(
        self,
        transaction: dict[str, Any],
        *,
        event_id: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> float:
        """Persist a successfully scored event and its monitor result."""
        normalized = self._normalize_event_id(event_id)
        step = int(transaction["step"])
        if step <= self.reference_max_step:
            raise ValueError(
                "Online history accepts only steps greater than the "
                f"reference maximum step ({self.reference_max_step})"
            )
        processed_at = pd.Timestamp.now(tz="UTC").isoformat()
        result_json = (
            json.dumps(result, separators=(",", ":"), allow_nan=False)
            if result is not None
            else None
        )
        flagged = (
            int(bool(result["risk"]["flagged_for_review"]))
            if result is not None
            else None
        )
        started_at = time.perf_counter()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO scored_transactions
                        (event_id, step, amount, nameOrig, nameDest, result_json,
                         processed_at, flagged_for_review, state_update_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        normalized,
                        step,
                        float(transaction["amount"]),
                        str(transaction["nameOrig"]),
                        str(transaction["nameDest"]),
                        result_json,
                        processed_at,
                        flagged,
                    ),
                )
                state_update_ms = (time.perf_counter() - started_at) * 1000
                connection.execute(
                    "UPDATE scored_transactions SET state_update_ms = ? WHERE rowid = ?",
                    (state_update_ms, cursor.lastrowid),
                )
        except sqlite3.IntegrityError as error:
            if normalized is not None:
                raise DuplicateTransactionError(
                    f"Transaction event_id has already been scored: {normalized}"
                ) from error
            raise
        return state_update_ms

    def live_status(self) -> dict[str, Any]:
        """Summarize successfully scored events available to the live monitor."""
        with self._connect() as connection:
            result = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(flagged_for_review), 0),
                       MAX(processed_at)
                FROM scored_transactions
                WHERE result_json IS NOT NULL
                """
            ).fetchone()
            latest_online_step = connection.execute(
                "SELECT MAX(step) FROM scored_transactions"
            ).fetchone()[0]
            latest_row = connection.execute(
                """
                SELECT rowid, event_id, result_json, processed_at, state_update_ms
                FROM scored_transactions
                WHERE result_json IS NOT NULL
                ORDER BY rowid DESC
                LIMIT 1
                """
            ).fetchone()
        latest_result = (
            json.loads(latest_row[2])
            if latest_row is not None and latest_row[2] is not None
            else {}
        )
        latest_timings = latest_result.get("timings", {})
        return {
            "event_count": int(result[0]),
            "flagged_event_count": int(result[1]),
            "latest_processed_at": result[2],
            "latest_step": int(latest_online_step) if latest_online_step is not None else None,
            "latest_event_key": (
                latest_row[1] or f"online-event-{latest_row[0]}"
                if latest_row is not None
                else None
            ),
            "latest_risk_score": latest_result.get("risk", {}).get("score"),
            "latest_processing_ms": (
                round(
                    float(latest_timings.get("feature_ms", 0))
                    + float(latest_timings.get("inference_ms", 0))
                    + float(latest_timings.get("explanation_ms", 0))
                    + float(latest_row[4] or 0),
                    3,
                )
                if latest_row is not None
                else None
            ),
            "active_model": latest_result.get("model", "B"),
        }

    def recent_live_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return stored scoring results, newest first, for authenticated review."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT rowid, event_id, result_json, processed_at, state_update_ms,
                       investigation_status, investigation_note,
                       investigation_updated_at
                FROM scored_transactions
                WHERE result_json IS NOT NULL
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for (
            row_id,
            event_id,
            result_json,
            processed_at,
            state_update_ms,
            investigation_status,
            investigation_note,
            investigation_updated_at,
        ) in rows:
            result = json.loads(result_json)
            timings = result.setdefault("timings", {})
            timings["state_update_ms"] = round(float(state_update_ms or 0), 3)
            timings["processing_ms"] = round(
                float(timings.get("feature_ms", 0))
                + float(timings.get("inference_ms", 0))
                + float(timings.get("explanation_ms", 0))
                + float(timings["state_update_ms"]),
                3,
            )
            events.append(
                {
                    "event_key": event_id or f"online-event-{row_id}",
                    "processed_at": processed_at,
                    "processing_state": "Processed",
                    "investigation": {
                        "status": investigation_status,
                        "note": investigation_note,
                        "updated_at": investigation_updated_at,
                    },
                    **result,
                }
            )
        return events

    def live_event_details(self, event_key: str) -> dict[str, Any] | None:
        """Load one scored event and only earlier online relationships."""
        row_id = self._event_row_id(event_key)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT rowid, event_id, step, amount, nameOrig, nameDest,
                       result_json, processed_at, state_update_ms,
                       investigation_status, investigation_note,
                       investigation_updated_at
                FROM scored_transactions
                WHERE event_id = ?
                   OR (event_id IS NULL AND rowid = ?)
                LIMIT 1
                """,
                (event_key, row_id),
            ).fetchone()
            if row is None:
                return None
            related_rows = connection.execute(
                """
                SELECT rowid, event_id, step, amount, nameOrig, nameDest,
                       result_json, processed_at, investigation_status
                FROM scored_transactions
                WHERE step > ?
                  AND step < ?
                  AND rowid != ?
                  AND (
                      nameOrig IN (?, ?)
                      OR nameDest IN (?, ?)
                  )
                ORDER BY step DESC, rowid DESC
                LIMIT 100
                """,
                (
                    self.reference_max_step,
                    int(row[2]),
                    int(row[0]),
                    str(row[4]),
                    str(row[5]),
                    str(row[4]),
                    str(row[5]),
                ),
            ).fetchall()

        result = json.loads(row[6]) if row[6] is not None else {}
        timings = result.setdefault("timings", {})
        timings["state_update_ms"] = round(float(row[8] or 0), 3)
        timings["processing_ms"] = round(
            float(timings.get("feature_ms", 0))
            + float(timings.get("inference_ms", 0))
            + float(timings.get("explanation_ms", 0))
            + float(timings["state_update_ms"]),
            3,
        )
        related_events: list[dict[str, Any]] = []
        sender_receivers: dict[str, dict[str, Any]] = {}
        receiver_senders: dict[str, dict[str, Any]] = {}
        relationship_seen = False
        for related in related_rows:
            (
                related_row_id,
                related_event_id,
                related_step,
                related_amount,
                related_sender,
                related_receiver,
                related_result_json,
                related_processed_at,
                related_status,
            ) = related
            related_result = (
                json.loads(related_result_json)
                if related_result_json is not None
                else {}
            )
            related_tx = related_result.get("transaction", {})
            related_risk = related_result.get("risk", {})
            related_events.append(
                {
                    "event_key": related_event_id
                    or f"online-event-{related_row_id}",
                    "step": int(related_step),
                    "processed_at": related_processed_at,
                    "type": related_tx.get("type"),
                    "amount": float(related_amount),
                    "sender": str(related_sender),
                    "receiver": str(related_receiver),
                    "risk_score": related_risk.get("score"),
                    "review_priority": related_risk.get("review_priority"),
                    "flagged_for_review": related_risk.get("flagged_for_review"),
                    "investigation_status": related_status,
                }
            )
            if related_sender == row[4]:
                counterparty = sender_receivers.setdefault(
                    str(related_receiver),
                    {"account": str(related_receiver), "transactions": 0, "amount": 0.0},
                )
                counterparty["transactions"] += 1
                counterparty["amount"] += float(related_amount)
            if related_receiver == row[5]:
                counterparty = receiver_senders.setdefault(
                    str(related_sender),
                    {"account": str(related_sender), "transactions": 0, "amount": 0.0},
                )
                counterparty["transactions"] += 1
                counterparty["amount"] += float(related_amount)
            relationship_seen = relationship_seen or (
                related_sender == row[4] and related_receiver == row[5]
            )

        for counterparties in (sender_receivers, receiver_senders):
            for counterparty in counterparties.values():
                counterparty["amount"] = round(float(counterparty["amount"]), 2)

        return {
            "event_key": row[1] or f"online-event-{row[0]}",
            "processed_at": row[7],
            "processing_state": "Processed",
            "transaction": result.get(
                "transaction",
                {
                    "step": int(row[2]),
                    "amount": float(row[3]),
                    "sender": str(row[4]),
                    "receiver": str(row[5]),
                },
            ),
            "investigation": {
                "status": row[9],
                "note": row[10],
                "updated_at": row[11],
            },
            "related_activity": related_events,
            "online_network_context": {
                "window": "Previously scored online events only",
                "history_before_step": int(row[2]),
                "prior_relationship_seen": relationship_seen,
                "sender_prior_receivers": sorted(
                    sender_receivers.values(),
                    key=lambda item: (item["transactions"], item["amount"]),
                    reverse=True,
                ),
                "receiver_prior_senders": sorted(
                    receiver_senders.values(),
                    key=lambda item: (item["transactions"], item["amount"]),
                    reverse=True,
                ),
            },
            **result,
        }

    def update_live_investigation(
        self,
        event_key: str,
        *,
        status: str,
        note: str,
    ) -> dict[str, str] | None:
        """Persist an analyst's status and note for a scored online event."""
        allowed_statuses = {"Open", "Investigating", "Escalated", "Closed"}
        if status not in allowed_statuses:
            raise ValueError("Unsupported investigation status")
        if len(note) > 2000:
            raise ValueError("Investigation note must not exceed 2000 characters")
        row_id = self._event_row_id(event_key)
        updated_at = pd.Timestamp.now(tz="UTC").isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE scored_transactions
                SET investigation_status = ?,
                    investigation_note = ?,
                    investigation_updated_at = ?
                WHERE event_id = ?
                   OR (event_id IS NULL AND rowid = ?)
                """,
                (status, note, updated_at, event_key, row_id),
            )
            if cursor.rowcount == 0:
                return None
        return {
            "event_key": event_key,
            "status": status,
            "note": note,
            "updated_at": updated_at,
        }

    def get_live_investigation(self, event_key: str) -> dict[str, str] | None:
        row_id = self._event_row_id(event_key)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT event_id, investigation_status, investigation_note,
                       investigation_updated_at
                FROM scored_transactions
                WHERE event_id = ?
                   OR (event_id IS NULL AND rowid = ?)
                LIMIT 1
                """,
                (event_key, row_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "event_key": row[0] or event_key,
            "status": str(row[1]),
            "note": str(row[2]),
            "updated_at": str(row[3] or ""),
        }

    @staticmethod
    def _event_row_id(event_key: str) -> int:
        prefix = "online-event-"
        if event_key.startswith(prefix):
            suffix = event_key.removeprefix(prefix)
            if suffix.isdigit():
                return int(suffix)
        return -1
