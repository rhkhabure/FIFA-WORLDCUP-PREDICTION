"""Health and monitoring helpers that avoid live provider dependencies."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def sqlite_probe(path: str | Path) -> bool:
    """Return whether a SQLite database can be opened and queried."""
    try:
        with sqlite3.connect(path, timeout=1) as connection:
            connection.execute("SELECT 1").fetchone()
        return True
    except (OSError, sqlite3.Error):
        return False


def readiness_report(
    *,
    football_model_loaded: bool,
    nba_model_loaded: bool,
    football_db_ready: bool,
    nba_db_ready: bool,
) -> dict:
    """Build a safe readiness payload from local application dependencies."""
    checks = {
        "football_model": bool(football_model_loaded),
        "nba_model": bool(nba_model_loaded),
        "football_database": bool(football_db_ready),
        "nba_database": bool(nba_db_ready),
    }
    ready = all(checks.values())
    return {
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "checks": checks,
    }
