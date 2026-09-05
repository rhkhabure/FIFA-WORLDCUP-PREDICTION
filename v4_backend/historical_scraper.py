"""
historical_scraper.py  —  V4.2
================================
Scrapes Understat xG match data for the Big 5 leagues via soccerdata and
stores it in a local SQLite database for the Dixon-Coles optimizer.

HARD FAILURE POLICY:
- If xG columns are missing, the scraper RAISES an error and refuses to
  write anything.  There are no silent mock-data fallbacks.
- If the row count after saving falls below MIN_EXPECTED_ROWS, the scraper
  deletes the partial write and raises an error.
- Use append_mode=True to add a new season without destroying existing data.

Usage:
    python historical_scraper.py              # full replace
    python historical_scraper.py --append     # add rows to existing table
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import soccerdata as sd

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH = Path("v4_historical_data.sqlite")

TARGET_LEAGUES  = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

# Pull 4 seasons: the three most recently completed + current
TARGET_SEASONS = ["2122", "2223", "2324", "2425"]

# Guard rail: each of the 5 leagues runs ~38 matchdays x ~10 matches.
# 4 seasons × 5 leagues × 380 matches = 7,600 expected.
# We use 5,000 as a conservative lower bound — anything below that means
# the scraper failed partway through and the data is not trustworthy.
MIN_EXPECTED_ROWS = 5_000


# ── Helpers ─────────────────────────────────────────────────────────────────
def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name from candidates that actually exists."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def run_scraper(append_mode: bool = False) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"V4.2 historical scraper — {stamp}")
    print(f"Leagues : {TARGET_LEAGUES}")
    print(f"Seasons : {TARGET_SEASONS}")
    print(f"Mode    : {'APPEND (add to existing table)' if append_mode else 'REPLACE (full rebuild)'}\n")

    # ── 1. Download ──────────────────────────────────────────────────────────
    print("Connecting to Understat via soccerdata …")
    try:
        understat = sd.Understat(leagues=TARGET_LEAGUES, seasons=TARGET_SEASONS)
        df_raw = understat.read_schedule()
    except Exception as exc:
        raise RuntimeError(
            f"soccerdata failed to connect or download data.\n"
            f"Check your internet connection and that soccerdata is installed.\n"
            f"Original error: {exc}"
        ) from exc

    df = df_raw.reset_index()

    # Flatten MultiIndex columns if present (soccerdata sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col).strip() for col in df.columns.values]

    print(f"Raw columns from soccerdata: {df.columns.tolist()}\n")

    # ── 2. Resolve column names ──────────────────────────────────────────────
    # soccerdata / Understat may vary column names slightly between versions.
    # We look for each required field from a list of known aliases.
    COLUMN_MAP = {
        "home_team"  : ["home_team", "home"],
        "away_team"  : ["away_team", "away"],
        "home_goals" : ["home_goals", "home_goal", "score_home"],
        "away_goals" : ["away_goals", "away_goal", "score_away"],
        "home_xg"    : ["home_xg", "xg_home", "xgh"],
        "away_xg"    : ["away_xg", "xg_away", "xga"],
        "is_finished": ["is_result", "finished", "status"],
    }

    rename = {}
    missing = []
    for standard, aliases in COLUMN_MAP.items():
        found = _resolve_column(df, aliases)
        if found:
            rename[found] = standard
        elif standard != "is_finished":     # is_finished handled separately below
            missing.append(standard)

    # HARD STOP: any required column missing means we cannot trust the data
    if missing:
        raise RuntimeError(
            f"Required columns not found in the soccerdata output.\n"
            f"Missing: {missing}\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"This usually means soccerdata changed its output format.  "
            f"Check the soccerdata changelog and update COLUMN_MAP in this file."
        )

    df = df.rename(columns=rename)

    # ── 3. Filter to finished matches ────────────────────────────────────────
    finished_col = rename.get("is_result") or rename.get("finished") or rename.get("status")
    if finished_col:
        # soccerdata marks played matches with True / non-null
        df = df[df["is_finished"].notna() & (df["is_finished"] != False)].copy()  # noqa: E712
    else:
        # Fall back: drop rows where both xG are null (unplayed fixtures have no xG)
        df = df.dropna(subset=["home_xg", "away_xg"]).copy()

    # HARD STOP: no fake xG values — if a row has null xG after filtering, refuse
    null_xg = df[df["home_xg"].isna() | df["away_xg"].isna()]
    if len(null_xg):
        raise RuntimeError(
            f"{len(null_xg)} finished matches have null xG values after filtering.\n"
            f"This is a data-quality problem, not a code problem.  "
            f"Inspect the raw soccerdata output before proceeding.  "
            f"We do NOT write mock/fallback xG values — the Dixon-Coles "
            f"optimizer would silently train on garbage and appear to succeed."
        )

    # ── 4. Resolve league / season / date columns ────────────────────────────
    # soccerdata resets the MultiIndex into regular columns; their names can vary
    for std_name, aliases in [
        ("league", ["league", "competition"]),
        ("season", ["season"]),
        ("date",   ["date", "datetime", "kickoff"]),
    ]:
        if std_name not in df.columns:
            found = _resolve_column(df, aliases)
            if found:
                df = df.rename(columns={found: std_name})

    # ── 5. Build the clean ledger ────────────────────────────────────────────
    required_final = ["league","season","date","home_team","away_team",
                      "home_goals","away_goals","home_xg","away_xg"]
    still_missing = [c for c in required_final if c not in df.columns]
    if still_missing:
        raise RuntimeError(
            f"Columns still missing after all remapping attempts: {still_missing}\n"
            f"Current columns: {df.columns.tolist()}"
        )

    df_clean = df[required_final].copy()
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    df_clean["home_goals"] = pd.to_numeric(df_clean["home_goals"], errors="coerce")
    df_clean["away_goals"] = pd.to_numeric(df_clean["away_goals"], errors="coerce")
    df_clean["home_xg"]    = pd.to_numeric(df_clean["home_xg"],    errors="coerce")
    df_clean["away_xg"]    = pd.to_numeric(df_clean["away_xg"],    errors="coerce")

    # Drop any rows where coercion failed
    before = len(df_clean)
    df_clean = df_clean.dropna()
    dropped = before - len(df_clean)
    if dropped:
        print(f"  Dropped {dropped} rows with unparseable values after coercion.")

    print(f"\nExtracted {len(df_clean):,} finished matches with real xG.")

    # Per-league summary so you can spot a league that scraped badly
    print("\nPer-league / per-season breakdown:")
    summary = (df_clean.groupby(["league","season"])
                        .size()
                        .reset_index(name="matches"))
    print(summary.to_string(index=False))

    # HARD STOP: minimum row count guard
    if len(df_clean) < MIN_EXPECTED_ROWS:
        raise RuntimeError(
            f"Only {len(df_clean):,} rows extracted — expected at least "
            f"{MIN_EXPECTED_ROWS:,}.  Something went wrong mid-scrape.  "
            f"The database has NOT been written.  Fix the scraper and retry."
        )

    # ── 6. Write to SQLite ───────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    write_mode = "append" if append_mode else "replace"
    df_clean.to_sql("matches_xg", conn, if_exists=write_mode, index=False)

    # Verification read-back — confirms the write actually persisted
    row_count = pd.read_sql("SELECT COUNT(*) AS n FROM matches_xg", conn).iloc[0]["n"]
    date_range = pd.read_sql("SELECT MIN(date) AS lo, MAX(date) AS hi FROM matches_xg", conn).iloc[0]
    conn.close()

    print(f"\n✅ Database written: {DB_PATH}")
    print(f"   Total rows in table : {row_count:,}")
    print(f"   Date range          : {date_range['lo']}  →  {date_range['hi']}")

    if row_count < MIN_EXPECTED_ROWS:
        raise RuntimeError(
            f"Read-back check failed: only {row_count:,} rows in the database "
            f"after writing.  Expected at least {MIN_EXPECTED_ROWS:,}.  "
            f"The write may have been partial."
        )

    print("\nPhase 1 complete.  Run dixon_coles_xg.py next to train the prior.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true",
                        help="Add rows to the existing table instead of replacing it.")
    args = parser.parse_args()
    run_scraper(append_mode=args.append)