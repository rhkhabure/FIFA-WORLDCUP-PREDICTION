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
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from constants import EAT, nba_team

log = logging.getLogger(__name__)

# Module-level DB ref — injected by nba_setup() in routes/nba.py
_db: sqlite3.Connection | None = None


def set_db(conn: sqlite3.Connection | None) -> None:
    """Called once from nba_setup() after the DB is opened."""
    global _db
    _db = conn
    if conn is None:
        return
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(predictions)")
    }
    if "pregame_logged_at" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN pregame_logged_at TEXT")
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nba_prediction_game
        ON predictions(game_id)
        WHERE league='nba' AND game_id IS NOT NULL
    """)
    conn.execute("DROP TRIGGER IF EXISTS nba_pregame_snapshot_immutable")
    conn.execute("""
        CREATE TRIGGER nba_pregame_snapshot_immutable
        BEFORE UPDATE OF league, game_id, home_team, away_team, home_team_id,
            away_team_id, game_time_utc, model_version, home_win_prob,
            away_win_prob, pregame_logged_at
        ON predictions
        WHEN OLD.league = 'nba' AND (
            NEW.league IS NOT OLD.league OR
            NEW.game_id IS NOT OLD.game_id OR
            NEW.home_team IS NOT OLD.home_team OR
            NEW.away_team IS NOT OLD.away_team OR
            NEW.home_team_id IS NOT OLD.home_team_id OR
            NEW.away_team_id IS NOT OLD.away_team_id OR
            NEW.game_time_utc IS NOT OLD.game_time_utc OR
            NEW.model_version IS NOT OLD.model_version OR
            NEW.home_win_prob IS NOT OLD.home_win_prob OR
            NEW.away_win_prob IS NOT OLD.away_win_prob OR
            NEW.pregame_logged_at IS NOT OLD.pregame_logged_at
        ) AND NOT (
            OLD.pregame_logged_at IS NULL
            AND OLD.home_score IS NULL
            AND OLD.away_score IS NULL
            AND julianday(NEW.game_time_utc) > julianday('now')
            AND NEW.pregame_logged_at IS NOT NULL
            AND julianday(NEW.pregame_logged_at) < julianday(NEW.game_time_utc)
        )
        BEGIN
            SELECT RAISE(ABORT, 'NBA pregame snapshots are immutable');
        END
    """)
    conn.commit()


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


def _parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def is_verified_pregame(prediction: dict | None) -> bool:
    """True only when a recorded snapshot timestamp precedes its scheduled tip."""
    if not prediction:
        return False
    logged_at = _parse_utc(prediction.get("pregame_logged_at"))
    tipoff = _parse_utc(prediction.get("game_time_utc"))
    try:
        home_prob = float(prediction.get("home_win_prob"))
        away_prob = float(prediction.get("away_win_prob"))
    except (TypeError, ValueError):
        return False
    return (
        logged_at is not None
        and tipoff is not None
        and logged_at < tipoff
        and math.isfinite(home_prob)
        and math.isfinite(away_prob)
        and 0.0 <= home_prob <= 1.0
        and 0.0 <= away_prob <= 1.0
        and abs(home_prob + away_prob - 1.0) <= 0.00011
    )


def get_prediction(game_id: str) -> dict | None:
    """Return the immutable NBA prediction snapshot for a game."""
    if _db is None:
        return None
    try:
        cursor = _db.execute(
            "SELECT * FROM predictions WHERE game_id=? AND league='nba' "
            "ORDER BY id LIMIT 1",
            (str(game_id),),
        )
        row = cursor.fetchone()
        return dict(zip((column[0] for column in cursor.description), row)) if row else None
    except sqlite3.Error as exc:
        log.warning("NBA prediction lookup failed for %s: %s", game_id, exc)
        return None


def _conference_for_teams(home_team_id, away_team_id) -> str:
    """Assign a game to its home team's conference, falling back to away team."""
    for team_id in (home_team_id, away_team_id):
        try:
            conference = nba_team(int(team_id)).get("conference", "")
        except (TypeError, ValueError):
            continue
        if conference in ("East", "West"):
            return conference
    return ""


def normalize_conferences() -> int:
    """Reclassify legacy NBA rows whose conference is not East or West."""
    if _db is None:
        return 0
    try:
        rows = _db.execute(
            "SELECT id, home_team_id, away_team_id FROM predictions "
            "WHERE league='nba' AND (conference IS NULL OR conference NOT IN ('East', 'West'))"
        ).fetchall()
        updates = [
            (_conference_for_teams(row[1], row[2]), row[0])
            for row in rows
        ]
        updates = [(conference, row_id) for conference, row_id in updates if conference]
        if updates:
            _db.executemany(
                "UPDATE predictions SET conference=? WHERE id=?",
                updates,
            )
            _db.commit()
        if updates:
            log.info("Reclassified %s NBA predictions by team conference", len(updates))
        return len(updates)
    except sqlite3.Error as exc:
        log.warning("NBA conference normalization failed: %s", exc)
        return 0


# ── Public API ────────────────────────────────────────────────────────────────

def log_pregame(
    game: dict,
    home_win_prob: float,
    model_version: str = "nba-elo-static-v0",
    *,
    now: datetime | None = None,
) -> bool:
    """
    Insert a prediction only for an unstarted game, preserving its original snapshot.

    game dict keys used:
        game_id, home_tricode, away_tricode,
        home_team_id, away_team_id, game_time_utc, status
    """
    game_id = str(game.get("game_id", ""))
    if not game_id or game.get("status") != "pre":
        return False

    kickoff = game.get("game_time_utc") or ""
    kickoff_dt = _parse_utc(kickoff)
    captured_at = now or datetime.now(timezone.utc)
    if captured_at.tzinfo is None:
        return False
    captured_at = captured_at.astimezone(timezone.utc)
    if kickoff_dt is None or kickoff_dt <= captured_at:
        return False
    try:
        hp = float(home_win_prob)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(hp) or not 0.0 <= hp <= 1.0:
        return False
    hp = round(hp, 4)
    away_win_prob = round(1.0 - hp, 4)
    created_at = kickoff_dt.astimezone(EAT).date().isoformat()
    pregame_logged_at = captured_at.isoformat()

    home_team_id = game.get("home_team_id")
    away_team_id = game.get("away_team_id")
    conference = _conference_for_teams(home_team_id, away_team_id)

    existing = get_prediction(game_id)
    if existing:
        if is_verified_pregame(existing):
            return False
        if existing.get("home_score") is not None or existing.get("away_score") is not None:
            return False
        return _execute("""
            UPDATE predictions
            SET home_team=?, away_team=?, home_team_id=?, away_team_id=?,
                conference=?, game_time_utc=?, model_version=?, created_at=?,
                home_win_prob=?, away_win_prob=?, pregame_logged_at=?
            WHERE id=? AND league='nba' AND pregame_logged_at IS NULL
        """, (
            game.get("home_tricode", ""),
            game.get("away_tricode", ""),
            home_team_id,
            away_team_id,
            conference,
            kickoff,
            model_version,
            created_at,
            hp,
            away_win_prob,
            pregame_logged_at,
            existing["id"],
        ))

    return _execute("""
        INSERT OR IGNORE INTO predictions
            (league, game_id, home_team, away_team,
             home_team_id, away_team_id, conference, game_time_utc, model_version,
             home_win_prob, away_win_prob, pregame_logged_at,
             home_score, away_score, correct, brier, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "nba",
        game_id,
        game.get("home_tricode", ""),
        game.get("away_tricode", ""),
        home_team_id,
        away_team_id,
        conference,
        kickoff,
        model_version,
        hp,
        away_win_prob,
        pregame_logged_at,
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
            "SELECT id, home_win_prob, away_win_prob, pregame_logged_at, game_time_utc "
            "FROM predictions "
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
    home_won = home_score > away_score
    verified = is_verified_pregame({
        "home_win_prob": row[1],
        "away_win_prob": row[2],
        "pregame_logged_at": row[3],
        "game_time_utc": row[4],
    })
    correct = 1 if (home_win_prob >= 0.5) == home_won else 0
    brier = round((home_win_prob - (1 if home_won else 0)) ** 2, 6)

    return _execute("""
        UPDATE predictions
        SET home_score=?, away_score=?, correct=?, brier=?
        WHERE id=?
    """, (
        home_score,
        away_score,
        correct if verified else None,
        brier if verified else None,
        pred_id,
    ))


def bulk_log_pregame(
    games: list[dict],
    probs: dict[str, float],
    model_version: str = "nba-elo-static-v0",
) -> int:
    """
    Log multiple games at once. probs = {game_id: home_win_prob}.
    Returns count of newly inserted rows.
    """
    inserted = 0
    for game in games:
        gid = game.get("game_id", "")
        hp  = probs.get(gid, 0.5)
        if log_pregame(game, hp, model_version):
            inserted += 1
    return inserted


def get_stats(conference: str = "all") -> dict:
    """Return accuracy stats using only timestamp-verified pregame snapshots."""
    if _db is None:
        return {
            "total": 0, "verified": 0, "unverified": 0, "played": 0,
            "correct": 0, "accuracy": 0.0, "avg_brier": None,
        }
    where = "league='nba'"
    params: tuple = ()
    if conference in ("East", "West"):
        where += " AND conference=?"
        params = (conference,)
    verified_where = (
        f"{where} AND pregame_logged_at IS NOT NULL "
        "AND game_time_utc IS NOT NULL "
        "AND julianday(pregame_logged_at) < julianday(game_time_utc)"
        " AND home_win_prob BETWEEN 0 AND 1"
        " AND away_win_prob BETWEEN 0 AND 1"
    )
    try:
        total   = _db.execute(
            f"SELECT COUNT(*) FROM predictions WHERE {where}", params
        ).fetchone()[0]
        verified = _db.execute(
            f"SELECT COUNT(*) FROM predictions WHERE {verified_where}", params
        ).fetchone()[0]
        played = _db.execute(
            f"SELECT COUNT(*) FROM predictions WHERE {verified_where} "
            "AND correct IS NOT NULL",
            params,
        ).fetchone()[0]
        correct = _db.execute(
            f"SELECT SUM(correct) FROM predictions WHERE {verified_where} "
            "AND correct IS NOT NULL",
            params,
        ).fetchone()[0] or 0
        avg_b   = _db.execute(
            f"SELECT AVG(brier) FROM predictions WHERE {verified_where} "
            "AND brier IS NOT NULL",
            params,
        ).fetchone()[0]
        return {
            "total":     total,
            "correct":   correct,
            "played":    played,
            "verified":  verified,
            "unverified": total - verified,
            "accuracy":  round(correct / played * 100, 1) if played else 0.0,
            "avg_brier": round(avg_b, 4) if avg_b is not None else None,
        }
    except Exception as exc:
        log.warning("get_stats error: %s", exc)
        return {
            "total": 0, "verified": 0, "unverified": 0, "played": 0,
            "correct": 0, "accuracy": 0.0, "avg_brier": None,
        }
