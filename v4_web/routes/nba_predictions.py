"""
nba_predictions.py  —  NBA prediction logging
==============================================
Logs pregame predictions when a game appears on the scoreboard,
and fills in the result when the game goes final.

Called from:
  - nba_api_client.get_today_scoreboard() — after enrichment
  - routes/nba.py nba_live_poll() — when status flips to 'final'

Uses the shared predictions.db (same file as football, league='nba').
All DB writes are fire-and-forget — failures are logged and ignored
so a DB error never crashes a page load.
"""

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

# Module-level DB ref — injected by nba_setup() in routes/nba.py
_db: sqlite3.Connection | None = None


def set_db(conn: sqlite3.Connection) -> None:
    """Called once from nba_setup() after the DB is opened."""
    global _db
    _db = conn


# ── Core write helpers ────────────────────────────────────────────────────────

def _execute(sql: str, params: tuple) -> bool:
    """Run one SQL statement; return True on success."""
    if _db is None:
        return False
    try:
        _db.execute(sql, params)
        _db.commit()
        return True
    except Exception as exc:
        log.warning("NBA predictions DB error: %s", exc)
        return False


def _game_exists(game_id: str) -> bool:
    """Return True if this game_id is already in the predictions table."""
    if _db is None:
        return False
    try:
        row = _db.execute(
            "SELECT id FROM predictions WHERE game_id=? AND league='nba'",
            (game_id,)
        ).fetchone()
        return row is not None
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def log_pregame(game: dict, home_win_prob: float) -> bool:
    """
    Insert a pregame prediction row when a game first appears on the hub.
    Safe to call multiple times — uses INSERT OR IGNORE so duplicates are skipped.

    game dict keys used:
        game_id, home_tricode, away_tricode,
        home_team_id, away_team_id, game_time_utc, status
    """
    if _game_exists(game.get("game_id", "")):
        return False   # already logged

    away_win_prob = round(1.0 - home_win_prob, 4)
    hp            = round(home_win_prob, 4)
    created_at    = (game.get("game_time_utc") or "")[:10]  # YYYY-MM-DD

    return _execute("""
        INSERT OR IGNORE INTO predictions
            (league, game_id, home_team, away_team,
             home_team_id, away_team_id, conference,
             home_win_prob, away_win_prob,
             home_score, away_score, correct, brier, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "nba",
        game.get("game_id", ""),
        game.get("home_tricode", ""),
        game.get("away_tricode", ""),
        game.get("home_team_id"),
        game.get("away_team_id"),
        "Preseason" if str(game.get("game_id","")).startswith("001") else "Regular Season",
        hp,
        away_win_prob,
        None,   # scores filled in by log_result()
        None,
        None,   # correct filled in by log_result()
        None,   # brier filled in by log_result()
        created_at,
    ))


def log_result(game_id: str, home_score: int, away_score: int) -> bool:
    """
    Fill in the final score and compute correct/brier once a game is final.
    Only updates rows that don't already have a result (correct IS NULL).
    """
    if _db is None:
        return False

    try:
        row = _db.execute(
            "SELECT id, home_win_prob FROM predictions "
            "WHERE game_id=? AND league='nba' AND correct IS NULL",
            (game_id,)
        ).fetchone()
    except Exception as exc:
        log.warning("log_result lookup failed: %s", exc)
        return False

    if row is None:
        return False   # not found or already has a result

    pred_id      = row[0]
    home_win_prob = float(row[1])
    home_won     = home_score > away_score
    correct      = 1 if (home_win_prob >= 0.5) == home_won else 0
    brier        = round((home_win_prob - (1 if home_won else 0)) ** 2, 6)

    return _execute("""
        UPDATE predictions
        SET home_score=?, away_score=?, correct=?, brier=?
        WHERE id=?
    """, (home_score, away_score, correct, brier, pred_id))


def bulk_log_pregame(games: list[dict], probs: dict[str, float]) -> int:
    """
    Log multiple games at once. probs = {game_id: home_win_prob}.
    Returns count of newly inserted rows.
    """
    inserted = 0
    for game in games:
        gid = game.get("game_id", "")
        hp  = probs.get(gid, 0.5)
        if log_pregame(game, hp):
            inserted += 1
    return inserted


def get_stats() -> dict:
    """Return prediction accuracy stats for the /nba/history page."""
    if _db is None:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "avg_brier": None}
    try:
        total   = _db.execute(
            "SELECT COUNT(*) FROM predictions WHERE league='nba'"
        ).fetchone()[0]
        correct = _db.execute(
            "SELECT SUM(correct) FROM predictions WHERE league='nba' AND correct IS NOT NULL"
        ).fetchone()[0] or 0
        avg_b   = _db.execute(
            "SELECT AVG(brier) FROM predictions WHERE league='nba' AND brier IS NOT NULL"
        ).fetchone()[0]
        played  = _db.execute(
            "SELECT COUNT(*) FROM predictions WHERE league='nba' AND correct IS NOT NULL"
        ).fetchone()[0]
        return {
            "total":     total,
            "correct":   correct,
            "played":    played,
            "accuracy":  round(correct / played * 100, 1) if played else 0.0,
            "avg_brier": round(avg_b, 4) if avg_b else None,
        }
    except Exception as exc:
        log.warning("get_stats error: %s", exc)
        return {"total": 0, "correct": 0, "accuracy": 0.0, "avg_brier": None}
