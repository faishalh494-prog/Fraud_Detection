"""Local SQLite-backed investigation workflow status."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

INVESTIGATION_STATUSES = {"Open", "Investigating", "Escalated", "Closed"}


class InvestigationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigations (
                    row_index INTEGER PRIMARY KEY,
                    status TEXT NOT NULL CHECK (
                        status IN ('Open', 'Investigating', 'Escalated', 'Closed')
                    ),
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, row_index: int) -> dict[str, str | int]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT row_index, status, note, updated_at FROM investigations WHERE row_index = ?",
                (row_index,),
            ).fetchone()
        if row is None:
            return {"row_index": row_index, "status": "Open", "note": "", "updated_at": ""}
        return {
            "row_index": int(row[0]),
            "status": str(row[1]),
            "note": str(row[2]),
            "updated_at": str(row[3]),
        }

    def update(self, row_index: int, status: str, note: str = "") -> dict[str, str | int]:
        if status not in INVESTIGATION_STATUSES:
            raise ValueError(f"Unsupported investigation status: {status}")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO investigations (row_index, status, note, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(row_index) DO UPDATE SET
                    status = excluded.status,
                    note = excluded.note,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (row_index, status, note),
            )
        return self.get(row_index)
