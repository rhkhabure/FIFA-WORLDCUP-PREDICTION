"""
predictions.py  —  V4.2
========================
Prediction logging and evaluation store.

SQLite database at v4_web/data/predictions.db

Two snapshots per match:
  pre_lineup  — DC prior logged as soon as fixture is announced
  post_lineup — DC + lineup-adjusted logged once FotMob lineup confirmed

Background job (run_prediction_job) is called on startup and every
30 minutes. It is fully automatic — no user action required.

Schema
------
predictions (
    match_id        TEXT PRIMARY KEY,
    home_team       TEXT,
    away_team       TEXT,
    kickoff_utc     TEXT,
    season          INTEGER,

    -- Snapshot 1: pre-lineup DC odds (logged when fixture first seen)
    pre_dc_home     REAL,
    pre_dc_draw     REAL,
    pre_dc_away     REAL,
    pre_dc_lam      REAL,
    pre_dc_mu       REAL,
    pre_logged_at   TEXT,

    -- Snapshot 2: post-lineup adjusted odds (logged when lineup confirmed)
    adj_home        REAL,
    adj_draw        REAL,
    adj_away        REAL,
    adj_lam         REAL,
    adj_mu          REAL,
    absent_home     TEXT,   -- JSON list
    absent_away     TEXT,   -- JSON list
    adj_logged_at   TEXT,

    -- Result (filled after match)
    actual_result   TEXT,   -- 'H', 'D', 'A'
    actual_hg       INTEGER,
    actual_ag       INTEGER,
    result_logged_at TEXT,

    -- Evaluation flags
    pre_correct     INTEGER,  -- 1/0/NULL
    adj_correct     INTEGER   -- 1/0/NULL
)
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_DIR  = Path(__file__).resolve().parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "predictions.db"


# ── Database setup ─────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Idempotent."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                match_id         TEXT PRIMARY KEY,
                home_team        TEXT,
                away_team        TEXT,
                kickoff_utc      TEXT,
                season           INTEGER,
                league           TEXT DEFAULT 'pl',

                pre_dc_home      REAL,
                pre_dc_draw      REAL,
                pre_dc_away      REAL,
                pre_dc_lam       REAL,
                pre_dc_mu        REAL,
                pre_logged_at    TEXT,

                adj_home         REAL,
                adj_draw         REAL,
                adj_away         REAL,
                adj_lam          REAL,
                adj_mu           REAL,
                absent_home      TEXT,
                absent_away      TEXT,
                adj_logged_at    TEXT,

                actual_result    TEXT,
                actual_hg        INTEGER,
                actual_ag        INTEGER,
                result_logged_at TEXT,

                pre_correct      INTEGER,
                adj_correct      INTEGER
            )
        """)
        # Add league column to existing DBs (safe no-op if already present)
        try:
            conn.execute("ALTER TABLE predictions ADD COLUMN league TEXT DEFAULT 'pl'")
        except Exception:
            pass
        # Match snapshot table — stores everything needed to render match page
        # Persists across restarts so finished games always display correctly
        conn.execute("""
            CREATE TABLE IF NOT EXISTS match_snapshots (
                match_id        TEXT PRIMARY KEY,
                home_team       TEXT,
                away_team       TEXT,
                kickoff_utc     TEXT,
                h_score         INTEGER DEFAULT 0,
                a_score         INTEGER DEFAULT 0,
                status          TEXT DEFAULT 'Not Started',
                minute          INTEGER DEFAULT 0,
                home_formation  TEXT,
                away_formation  TEXT,
                home_players    TEXT,
                away_players    TEXT,
                absent_home     TEXT,
                absent_away     TEXT,
                home_colour     TEXT,
                away_colour     TEXT,
                fotmob_id       TEXT,
                snapped_at      TEXT
            )
        """)
        conn.commit()


# ── Write functions ────────────────────────────────────────────────────────────

def log_pre_lineup(
    match_id: str,
    home_team: str, away_team: str,
    kickoff_utc: str, season: int,
    dc_home: float, dc_draw: float, dc_away: float,
    dc_lam: float, dc_mu: float,
    league: str = "pl",
):
    """
    Insert snapshot 1 (pre-lineup DC prior).
    Skips if match_id already has a pre_logged_at entry.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT pre_logged_at FROM predictions WHERE match_id=?",
            (match_id,)
        ).fetchone()

        if existing and existing["pre_logged_at"]:
            return  # already logged — never overwrite

        if existing:
            conn.execute("""
                UPDATE predictions SET
                    pre_dc_home=?, pre_dc_draw=?, pre_dc_away=?,
                    pre_dc_lam=?, pre_dc_mu=?, pre_logged_at=?,
                    league=?
                WHERE match_id=?
            """, (dc_home, dc_draw, dc_away, dc_lam, dc_mu, now,
                  league, match_id))
        else:
            conn.execute("""
                INSERT INTO predictions
                    (match_id, home_team, away_team, kickoff_utc, season,
                     league,
                     pre_dc_home, pre_dc_draw, pre_dc_away,
                     pre_dc_lam, pre_dc_mu, pre_logged_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (match_id, home_team, away_team, kickoff_utc, season,
                  league,
                  dc_home, dc_draw, dc_away, dc_lam, dc_mu, now))
        conn.commit()


def log_post_lineup(
    match_id: str,
    adj_home: float, adj_draw: float, adj_away: float,
    adj_lam: float, adj_mu: float,
    absent_home: list[str], absent_away: list[str],
):
    """
    Insert snapshot 2 (post-lineup adjusted odds).
    Skips if match_id already has an adj_logged_at entry.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        existing = conn.execute(
            "SELECT adj_logged_at FROM predictions WHERE match_id=?",
            (match_id,)
        ).fetchone()

        if not existing:
            return  # pre snapshot must exist first
        if existing["adj_logged_at"]:
            return  # already logged — never overwrite

        conn.execute("""
            UPDATE predictions SET
                adj_home=?, adj_draw=?, adj_away=?,
                adj_lam=?, adj_mu=?,
                absent_home=?, absent_away=?,
                adj_logged_at=?
            WHERE match_id=?
        """, (adj_home, adj_draw, adj_away, adj_lam, adj_mu,
              json.dumps(absent_home), json.dumps(absent_away),
              now, match_id))
        conn.commit()


def log_result(
    match_id: str,
    actual_result: str,   # 'H', 'D', 'A'
    home_goals: int,
    away_goals: int,
):
    """
    Fill in actual result and compute correctness flags.
    Skips if result already logged.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE match_id=?", (match_id,)
        ).fetchone()

        if not row or row["actual_result"]:
            return  # not found or already logged

        # Which outcome did each model predict?
        def predicted_outcome(home_pct, draw_pct, away_pct) -> str | None:
            if home_pct is None:
                return None
            probs = {"H": home_pct, "D": draw_pct, "A": away_pct}
            return max(probs, key=probs.get)

        pre_pred = predicted_outcome(
            row["pre_dc_home"], row["pre_dc_draw"], row["pre_dc_away"]
        )
        adj_pred = predicted_outcome(
            row["adj_home"], row["adj_draw"], row["adj_away"]
        )

        pre_correct = int(pre_pred == actual_result) if pre_pred else None
        adj_correct = int(adj_pred == actual_result) if adj_pred else None

        conn.execute("""
            UPDATE predictions SET
                actual_result=?, actual_hg=?, actual_ag=?,
                result_logged_at=?,
                pre_correct=?, adj_correct=?
            WHERE match_id=?
        """, (actual_result, home_goals, away_goals, now,
              pre_correct, adj_correct, match_id))
        conn.commit()


# ── Read functions ─────────────────────────────────────────────────────────────

def get_all_predictions(season: int | None = None) -> list[dict]:
    """Return all predictions, optionally filtered by season."""
    with _conn() as conn:
        if season:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE season=? ORDER BY kickoff_utc DESC",
                (season,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY kickoff_utc DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def save_match_snapshot(
    match_id: str,
    home_team: str, away_team: str,
    kickoff_utc: str,
    h_score: int, a_score: int,
    status: str, minute: int,
    home_formation: str = "", away_formation: str = "",
    home_players: list = None, away_players: list = None,
    absent_home: list = None, absent_away: list = None,
    home_colour: str = "", away_colour: str = "",
    fotmob_id: str = "",
):
    """
    Upsert a match snapshot. Called whenever we have live/finished data.
    Overwrites previous snapshot so the latest state is always stored.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT INTO match_snapshots
                (match_id, home_team, away_team, kickoff_utc,
                 h_score, a_score, status, minute,
                 home_formation, away_formation,
                 home_players, away_players,
                 absent_home, absent_away,
                 home_colour, away_colour,
                 fotmob_id, snapped_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(match_id) DO UPDATE SET
                h_score=excluded.h_score,
                a_score=excluded.a_score,
                status=excluded.status,
                minute=excluded.minute,
                home_formation=excluded.home_formation,
                away_formation=excluded.away_formation,
                home_players=excluded.home_players,
                away_players=excluded.away_players,
                absent_home=excluded.absent_home,
                absent_away=excluded.absent_away,
                home_colour=excluded.home_colour,
                away_colour=excluded.away_colour,
                fotmob_id=excluded.fotmob_id,
                snapped_at=excluded.snapped_at
        """, (
            str(match_id), home_team, away_team, kickoff_utc,
            h_score, a_score, status, minute,
            home_formation, away_formation,
            json.dumps(home_players or []),
            json.dumps(away_players or []),
            json.dumps(absent_home or []),
            json.dumps(absent_away or []),
            home_colour, away_colour,
            fotmob_id, now,
        ))
        conn.commit()


def get_match_snapshot(match_id: str) -> dict | None:
    """
    Retrieve a stored match snapshot. Returns None if not found.
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM match_snapshots WHERE match_id=?",
            (str(match_id),)
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    # Deserialise JSON fields
    for field in ("home_players", "away_players", "absent_home", "absent_away"):
        try:
            d[field] = json.loads(d[field]) if d[field] else []
        except Exception:
            d[field] = []
    return d


def get_accuracy_stats(season: int | None = None) -> dict:
    """
    Return accuracy summary:
      pre_total, pre_correct, pre_accuracy
      adj_total, adj_correct, adj_accuracy
    """
    preds = get_all_predictions(season)
    finished = [p for p in preds if p["actual_result"]]

    pre_scored = [p for p in finished if p["pre_correct"] is not None]
    adj_scored = [p for p in finished if p["adj_correct"] is not None]

    def acc(lst, key):
        if not lst:
            return 0.0
        return sum(p[key] for p in lst) / len(lst) * 100

    return {
        "total_logged"   : len(preds),
        "total_finished" : len(finished),
        "pre_total"      : len(pre_scored),
        "pre_correct"    : sum(p["pre_correct"] for p in pre_scored),
        "pre_accuracy"   : round(acc(pre_scored, "pre_correct"), 1),
        "adj_total"      : len(adj_scored),
        "adj_correct"    : sum(p["adj_correct"] for p in adj_scored),
        "adj_accuracy"   : round(acc(adj_scored, "adj_correct"), 1),
    }
