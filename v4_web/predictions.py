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

DB_DIR  = Path(__file__).parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "predictions.db"


# ── Database setup ─────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                match_id         TEXT PRIMARY KEY,
                home_team        TEXT,
                away_team        TEXT,
                kickoff_utc      TEXT,
                season           INTEGER,

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
        conn.commit()


# ── Write functions ────────────────────────────────────────────────────────────

def log_pre_lineup(
    match_id: str,
    home_team: str, away_team: str,
    kickoff_utc: str, season: int,
    dc_home: float, dc_draw: float, dc_away: float,
    dc_lam: float, dc_mu: float,
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
            # Row exists but pre snapshot not yet filled
            conn.execute("""
                UPDATE predictions SET
                    pre_dc_home=?, pre_dc_draw=?, pre_dc_away=?,
                    pre_dc_lam=?, pre_dc_mu=?, pre_logged_at=?
                WHERE match_id=?
            """, (dc_home, dc_draw, dc_away, dc_lam, dc_mu, now, match_id))
        else:
            # New row
            conn.execute("""
                INSERT INTO predictions
                    (match_id, home_team, away_team, kickoff_utc, season,
                     pre_dc_home, pre_dc_draw, pre_dc_away,
                     pre_dc_lam, pre_dc_mu, pre_logged_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (match_id, home_team, away_team, kickoff_utc, season,
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
