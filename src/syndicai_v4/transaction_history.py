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
                    state_update_ms REAL
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
        return {
            "event_count": int(result[0]),
            "flagged_event_count": int(result[1]),
            "latest_processed_at": result[2],
            "latest_step": int(latest_online_step) if latest_online_step is not None else None,
        }

    def recent_live_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return stored scoring results, newest first, for authenticated review."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT rowid, event_id, result_json, processed_at, state_update_ms
                FROM scored_transactions
                WHERE result_json IS NOT NULL
                ORDER BY rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row_id, event_id, result_json, processed_at, state_update_ms in rows:
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
                    **result,
                }
            )
        return events
