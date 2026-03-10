"""SQLite connection and retry helpers shared by the app."""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Callable, TypeVar


SQLITE_TIMEOUT_SECONDS = 30
SQLITE_BUSY_TIMEOUT_MS = 30000
SQLITE_WRITE_RETRY_DELAYS = (0.2, 0.5, 1.0)

_T = TypeVar("_T")


def connect_sqlite(db_path: str, *, row_factory=None) -> sqlite3.Connection:
    """Create a SQLite connection with consistent pragmas."""
    conn = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_SECONDS)
    if row_factory is not None:
        conn.row_factory = row_factory

    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def is_sqlite_locked_error(error: Exception) -> bool:
    return isinstance(error, sqlite3.OperationalError) and "database is locked" in str(error).lower()


def run_sqlite_write_with_retry(
    action: Callable[[], _T],
    *,
    logger: logging.Logger | None = None,
    action_name: str = "sqlite_write",
) -> _T:
    """Retry a SQLite write when the database is temporarily locked."""
    attempt = 0
    while True:
        try:
            return action()
        except sqlite3.OperationalError as exc:
            if not is_sqlite_locked_error(exc) or attempt >= len(SQLITE_WRITE_RETRY_DELAYS):
                raise

            delay = SQLITE_WRITE_RETRY_DELAYS[attempt]
            if logger is not None:
                logger.warning(
                    "sqlite_retry action=%s attempt=%s delay=%.1fs error=%s",
                    action_name,
                    attempt + 1,
                    delay,
                    exc,
                )
            time.sleep(delay)
            attempt += 1
