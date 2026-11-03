"""Anonymous, one-vote-per-browser polls for upcoming football matches."""

import hashlib
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def _voter_hash(voter_id: str) -> str | None:
    try:
        normalized = str(uuid.UUID(voter_id))
    except (ValueError, TypeError, AttributeError):
        return None
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def init_fan_votes(db_path: str | Path) -> None:
    with closing(sqlite3.connect(db_path)) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS football_fan_votes (
                match_id TEXT NOT NULL,
                voter_hash TEXT NOT NULL,
                choice TEXT NOT NULL CHECK (choice IN ('H', 'D', 'A')),
                created_at TEXT NOT NULL,
                PRIMARY KEY (match_id, voter_hash)
            )
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_football_fan_votes_match
            ON football_fan_votes(match_id)
        """)
        db.commit()


def _summary(db: sqlite3.Connection, match_id: str, voter_hash: str | None) -> dict:
    counts = {choice: 0 for choice in ("H", "D", "A")}
    rows = db.execute(
        "SELECT choice, COUNT(*) FROM football_fan_votes "
        "WHERE match_id=? GROUP BY choice",
        (match_id,),
    ).fetchall()
    counts.update({choice: count for choice, count in rows})
    total = sum(counts.values())
    percentages = {
        choice: round(counts[choice] / total * 100, 1) if total else 0.0
        for choice in counts
    }
    voted = False
    if voter_hash:
        voted = db.execute(
            "SELECT 1 FROM football_fan_votes WHERE match_id=? AND voter_hash=?",
            (match_id, voter_hash),
        ).fetchone() is not None
    return {
        "total": total,
        "counts": counts,
        "percentages": percentages,
        "has_voted": voted,
    }


def get_vote_summary(
    db_path: str | Path, match_id: str, voter_id: str | None = None
) -> dict:
    voter_hash = _voter_hash(voter_id) if voter_id else None
    with closing(sqlite3.connect(db_path)) as db:
        return _summary(db, match_id, voter_hash)


def cast_vote(
    db_path: str | Path,
    match_id: str,
    choice: str,
    voter_id: str,
    *,
    now: datetime | None = None,
) -> dict:
    if choice not in ("H", "D", "A"):
        return {"accepted": False, "reason": "invalid_choice"}
    voter_hash = _voter_hash(voter_id)
    if not voter_hash:
        return {"accepted": False, "reason": "invalid_voter"}

    now = now or datetime.now(timezone.utc)
    with closing(sqlite3.connect(db_path)) as db:
        match = db.execute(
            "SELECT kickoff_utc, actual_result FROM predictions WHERE match_id=?",
            (match_id,),
        ).fetchone()
        if not match:
            return {"accepted": False, "reason": "unknown_match"}
        kickoff_raw, actual_result = match
        try:
            kickoff = datetime.fromisoformat(
                str(kickoff_raw).replace("Z", "+00:00")
            )
            if kickoff.tzinfo is None:
                return {"accepted": False, "reason": "invalid_kickoff"}
            kickoff = kickoff.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return {"accepted": False, "reason": "invalid_kickoff"}
        if actual_result or kickoff <= now.astimezone(timezone.utc):
            return {"accepted": False, "reason": "poll_closed"}

        cursor = db.execute(
            """
            INSERT OR IGNORE INTO football_fan_votes
                (match_id, voter_hash, choice, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (match_id, voter_hash, choice, now.astimezone(timezone.utc).isoformat()),
        )
        db.commit()
        summary = _summary(db, match_id, voter_hash)
        return {
            "accepted": cursor.rowcount == 1,
            "reason": None if cursor.rowcount == 1 else "already_voted",
            **summary,
        }
