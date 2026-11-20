"""Versioned SQLite schema bookkeeping for application databases."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

Migration = Callable[[sqlite3.Connection], None]


def schema_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def apply_migrations(
    connection: sqlite3.Connection,
    migrations: dict[int, Migration],
) -> int:
    """Apply each pending migration atomically and return the final version."""
    current = schema_version(connection)
    versions = sorted(migrations)
    if versions and current > versions[-1]:
        raise RuntimeError(
            f"Database schema version {current} is newer than supported "
            f"version {versions[-1]}"
        )

    for version in versions:
        if version <= current:
            continue
        migration = migrations[version]
        with connection:
            migration(connection)
            connection.execute(f"PRAGMA user_version = {version}")
        current = version
    return current


def _baseline(_connection: sqlite3.Connection) -> None:
    """Version 1 records the schema created by the existing startup code."""


FOOTBALL_MIGRATIONS: dict[int, Migration] = {1: _baseline}
NBA_MIGRATIONS: dict[int, Migration] = {1: _baseline}
