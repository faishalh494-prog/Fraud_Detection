"""Persistent history for transaction-time behavioural scoring."""

from __future__ import annotations

import sqlite3
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
                    nameDest TEXT NOT NULL
                )
                """
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
    ) -> None:
        """Persist a successfully scored event for later requests."""
        normalized = self._normalize_event_id(event_id)
        step = int(transaction["step"])
        if step <= self.reference_max_step:
            raise ValueError(
                "Online history accepts only steps greater than the "
                f"reference maximum step ({self.reference_max_step})"
            )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO scored_transactions
                        (event_id, step, amount, nameOrig, nameDest)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        normalized,
                        step,
                        float(transaction["amount"]),
                        str(transaction["nameOrig"]),
                        str(transaction["nameDest"]),
                    ),
                )
        except sqlite3.IntegrityError as error:
            if normalized is not None:
                raise DuplicateTransactionError(
                    f"Transaction event_id has already been scored: {normalized}"
                ) from error
            raise
