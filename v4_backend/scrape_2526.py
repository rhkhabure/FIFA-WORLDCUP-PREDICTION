"""
scrape_2526.py  —  V4.2
========================
Appends the 2025/26 season to v4_historical_data.sqlite.
Uses --append mode so the existing 2122-2425 data is preserved.

Run: python scrape_2526.py
"""

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import soccerdata as sd

ROOT       = Path(__file__).parent.parent
DB_PATH    = ROOT / "v4_historical_data.sqlite"
NEW_SEASON = "2526"

TARGET_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1",
]

COLUMN_MAP = {
    "home_team"  : ["home_team", "home"],
    "away_team"  : ["away_team", "away"],
    "home_goals" : ["home_goals", "home_goal", "score_home"],
    "away_goals" : ["away_goals", "away_goal", "score_away"],
    "home_xg"    : ["home_xg", "xg_home", "xgh"],
    "away_xg"    : ["away_xg", "xg_away", "xga"],
    "is_finished": ["is_result", "finished", "status"],
}

def resolve_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Scraping {NEW_SEASON} season — {stamp}")
    print(f"Leagues : {TARGET_LEAGUES}")
    print(f"Database: {DB_PATH}\n")

    # Check existing row count before touching anything
    conn = sqlite3.connect(DB_PATH)
    before = pd.read_sql(
        "SELECT season, COUNT(*) as n FROM matches_xg GROUP BY season ORDER BY season",
        conn
    )
    print("Existing database:")
    print(before.to_string(index=False))
    print()
    conn.close()

    # Scrape
    print(f"Downloading {NEW_SEASON} from Understat via soccerdata...")
    try:
        understat = sd.Understat(leagues=TARGET_LEAGUES, seasons=[NEW_SEASON])
        df_raw = understat.read_schedule()
    except Exception as e:
        raise RuntimeError(f"soccerdata failed: {e}") from e

    df = df_raw.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col).strip() for col in df.columns]

    print(f"Raw columns: {df.columns.tolist()}\n")

    # Resolve columns
    rename = {}
    missing = []
    for standard, aliases in COLUMN_MAP.items():
        found = resolve_column(df, aliases)
        if found:
            rename[found] = standard
        elif standard != "is_finished":
            missing.append(standard)

    if missing:
        raise RuntimeError(
            f"Required columns missing: {missing}\n"
            f"Available: {df.columns.tolist()}"
        )

    df = df.rename(columns=rename)

    # Filter to finished matches only
    if "is_finished" in df.columns:
        df = df[df["is_finished"].notna() & (df["is_finished"] != False)].copy()
    else:
        df = df.dropna(subset=["home_xg", "away_xg"]).copy()

    # Hard stop: no null xG
    null_xg = df[df["home_xg"].isna() | df["away_xg"].isna()]
    if len(null_xg):
        raise RuntimeError(
            f"{len(null_xg)} finished matches have null xG. "
            f"Inspect the raw data before proceeding."
        )

    # Standardise remaining columns
    for std, aliases in [
        ("league", ["league", "competition"]),
        ("season", ["season"]),
        ("date",   ["date", "datetime", "kickoff"]),
    ]:
        if std not in df.columns:
            found = resolve_column(df, aliases)
            if found:
                df = df.rename(columns={found: std})

    required = ["league","season","date","home_team","away_team",
                "home_goals","away_goals","home_xg","away_xg"]
    still_missing = [c for c in required if c not in df.columns]
    if still_missing:
        raise RuntimeError(f"Still missing after remapping: {still_missing}")

    df_clean = df[required].copy()
    df_clean["date"]       = pd.to_datetime(df_clean["date"], errors="coerce")
    df_clean["home_goals"] = pd.to_numeric(df_clean["home_goals"], errors="coerce")
    df_clean["away_goals"] = pd.to_numeric(df_clean["away_goals"], errors="coerce")
    df_clean["home_xg"]    = pd.to_numeric(df_clean["home_xg"],    errors="coerce")
    df_clean["away_xg"]    = pd.to_numeric(df_clean["away_xg"],    errors="coerce")

    before_drop = len(df_clean)
    df_clean = df_clean.dropna()
    dropped = before_drop - len(df_clean)
    if dropped:
        print(f"Dropped {dropped} rows with unparseable values.")

    print(f"Finished {NEW_SEASON} matches with real xG: {len(df_clean):,}")
    print()
    print("Per-league breakdown:")
    summary = df_clean.groupby(["league","season"]).size().reset_index(name="matches")
    print(summary.to_string(index=False))
    print()

    if len(df_clean) < 50:
        raise RuntimeError(
            f"Only {len(df_clean)} rows -- too few for a meaningful holdout. "
            f"The season may not have started yet or data is unavailable."
        )

    # Append to database
    conn = sqlite3.connect(DB_PATH)
    df_clean.to_sql("matches_xg", conn, if_exists="append", index=False)

    # Verify
    after = pd.read_sql(
        "SELECT season, COUNT(*) as n FROM matches_xg GROUP BY season ORDER BY season",
        conn
    )
    date_range = pd.read_sql(
        "SELECT MIN(date) as lo, MAX(date) as hi FROM matches_xg", conn
    ).iloc[0]
    conn.close()

    print("Database after append:")
    print(after.to_string(index=False))
    print(f"\nDate range: {date_range['lo']}  ->  {date_range['hi']}")
    print(f"\n✅ {NEW_SEASON} appended successfully.")
    print("Next: run build_v4_dataset.py with HOLDOUT_SEASON = '2526'")
