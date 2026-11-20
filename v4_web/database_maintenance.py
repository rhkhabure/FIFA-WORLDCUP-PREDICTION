"""Offline database integrity checks and SQLite backup helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def integrity_check(path: str | Path) -> bool:
    """Run SQLite's full integrity check without modifying the source DB."""
    connection = None
    try:
        connection = sqlite3.connect(
            f"file:{Path(path).resolve()}?mode=ro", uri=True
        )
        return connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    except (OSError, sqlite3.Error):
        return False
    finally:
        if connection is not None:
            connection.close()


def backup_database(source: str | Path, destination: str | Path) -> None:
    """Create an integrity-checked SQLite backup using SQLite's backup API."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not integrity_check(source_path):
        raise RuntimeError(f"Source database failed integrity check: {source_path}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        source_db = sqlite3.connect(source_path)
        destination_db = sqlite3.connect(temporary_path)
        try:
            source_db.backup(destination_db)
        finally:
            destination_db.close()
            source_db.close()
        if not integrity_check(temporary_path):
            raise RuntimeError(f"Backup failed integrity check: {temporary_path}")
        temporary_path.replace(destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
