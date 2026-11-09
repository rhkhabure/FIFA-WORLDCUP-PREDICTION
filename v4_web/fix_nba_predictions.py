"""Read-only audit of NBA prediction snapshot verifiability."""

import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).parent / "predictions.db"

if not DB.is_file():
    sys.exit(f"NBA predictions database not found: {DB}")

with sqlite3.connect(f"{DB.resolve().as_uri()}?mode=ro", uri=True) as conn:
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(predictions)")
    }
    if "pregame_logged_at" not in columns:
        sys.exit(
            "This database predates verified pregame timestamps. "
            "No rows were modified."
        )

    total, verified = conn.execute("""
        SELECT COUNT(*),
               SUM(CASE
                   WHEN pregame_logged_at IS NOT NULL
                    AND game_time_utc IS NOT NULL
                    AND julianday(pregame_logged_at) < julianday(game_time_utc)
                   THEN 1 ELSE 0
               END)
        FROM predictions
        WHERE league='nba'
    """).fetchone()

print(f"NBA prediction rows: {total}")
print(f"Timestamp-verified pregame snapshots: {verified or 0}")
print(f"Unverified legacy rows: {total - (verified or 0)}")
print("Read-only audit complete; no predictions or results were changed.")
