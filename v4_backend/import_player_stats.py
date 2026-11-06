"""Import historical Understat player-season statistics into a local SQLite DB.

Run from the repository root:
    python v4_backend/import_player_stats.py
    python v4_backend/import_player_stats.py --from-season 2020 --through-season 2025

The application reads this database locally and does not scrape on page requests.
"""

import argparse
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path

import pandas as pd
import soccerdata as sd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "v4_web" / "football" / "sql" / "football_player_stats.sqlite"
LEAGUES = {
    "pl": "ENG-Premier League",
    "laliga": "ESP-La Liga",
    "bundesliga": "GER-Bundesliga",
    "seriea": "ITA-Serie A",
    "ligue1": "FRA-Ligue 1",
}
STAT_FIELDS = (
    "position", "matches", "minutes", "goals", "assists", "xg", "xa",
    "shots", "key_passes", "yellow_cards", "red_cards", "np_goals",
    "np_xg", "xg_chain", "xg_buildup",
)


def player_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(
        "".join(char for char in ascii_name.casefold() if char.isalnum() or char.isspace())
        .split()
    )


def _to_int(value):
    if pd.isna(value):
        return None
    return int(value)


def _to_float(value):
    if pd.isna(value):
        return None
    return float(value)


def normalize_player_stats(frame: pd.DataFrame, league: str, season: int) -> list[tuple]:
    """Validate a SoccerData Understat frame and normalize it for storage."""
    if frame.empty:
        raise ValueError(f"No player statistics returned for {league} {season}.")
    frame = frame.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            "_".join(str(part) for part in column if str(part))
            for column in frame.columns
        ]
    required = {"player", "player_id", "team", *STAT_FIELDS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Understat player stats missing columns {sorted(missing)}; "
            f"received {list(frame.columns)}"
        )

    season_label = f"{season}-{str(season + 1)[-2:]}"
    normalized = []
    for row in frame.to_dict(orient="records"):
        player_name = str(row.get("player") or "").strip()
        team_name = str(row.get("team") or "").strip()
        player_id = _to_int(row.get("player_id"))
        if not player_name or not team_name or player_id is None:
            continue
        values = (
            league, season, season_label, team_name, player_name,
            player_key(player_name), player_id,
            str(row.get("position") or "").strip(),
            _to_int(row.get("matches")), _to_int(row.get("minutes")),
            _to_int(row.get("goals")), _to_int(row.get("assists")),
            _to_float(row.get("xg")), _to_float(row.get("xa")),
            _to_int(row.get("shots")), _to_int(row.get("key_passes")),
            _to_int(row.get("yellow_cards")), _to_int(row.get("red_cards")),
            _to_int(row.get("np_goals")), _to_float(row.get("np_xg")),
            _to_float(row.get("xg_chain")), _to_float(row.get("xg_buildup")),
        )
        normalized.append(values)
    if not normalized:
        raise ValueError(f"No valid player rows returned for {league} {season}.")
    return normalized


def import_player_stats(
    from_season: int,
    through_season: int,
    league_keys: list[str] | None = None,
    database_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    if from_season > through_season:
        raise ValueError("from_season must not be later than through_season.")
    selected_leagues = league_keys or list(LEAGUES)
    unknown = set(selected_leagues) - set(LEAGUES)
    if unknown:
        raise ValueError(f"Unknown league key(s): {', '.join(sorted(unknown))}")

    rows = []
    seasons = range(from_season, through_season + 1)
    for league_key in selected_leagues:
        source_league = LEAGUES[league_key]
        for season in seasons:
            season_arg = f"{season}/{season + 1}"
            print(f"Reading {source_league} player stats for {season_arg}...")
            reader = sd.Understat(
                leagues=[source_league],
                seasons=[season_arg],
            )
            frame = reader.read_player_season_stats()
            rows.extend(
                normalize_player_stats(frame, league_key, season)
            )

    target = Path(database_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(target)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS player_season_stats (
                league TEXT NOT NULL,
                season INTEGER NOT NULL,
                season_label TEXT NOT NULL,
                team TEXT NOT NULL,
                player TEXT NOT NULL,
                player_key TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                position TEXT,
                appearances INTEGER,
                minutes INTEGER,
                goals INTEGER,
                assists INTEGER,
                xg REAL,
                xa REAL,
                shots INTEGER,
                key_passes INTEGER,
                yellow_cards INTEGER,
                red_cards INTEGER,
                np_goals INTEGER,
                np_xg REAL,
                xg_chain REAL,
                xg_buildup REAL,
                PRIMARY KEY (league, season, player_id)
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO player_season_stats (
                league, season, season_label, team, player, player_key, player_id,
                position,
                appearances, minutes, goals, assists, xg, xa, shots, key_passes,
                yellow_cards, red_cards, np_goals, np_xg, xg_chain, xg_buildup
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(league, season, player_id) DO UPDATE SET
                season_label=excluded.season_label,
                team=excluded.team,
                player=excluded.player,
                player_key=excluded.player_key,
                position=excluded.position,
                appearances=excluded.appearances,
                minutes=excluded.minutes,
                goals=excluded.goals,
                assists=excluded.assists,
                xg=excluded.xg,
                xa=excluded.xa,
                shots=excluded.shots,
                key_passes=excluded.key_passes,
                yellow_cards=excluded.yellow_cards,
                red_cards=excluded.red_cards,
                np_goals=excluded.np_goals,
                np_xg=excluded.np_xg,
                xg_chain=excluded.xg_chain,
                xg_buildup=excluded.xg_buildup
            """,
            rows,
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_player_stats_lookup "
            "ON player_season_stats (league, player, season DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_player_stats_name "
            "ON player_season_stats (player_key, league, season DESC)"
        )
        connection.commit()
    print(f"Stored {len(rows):,} player-season rows in {target}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-season", type=int, default=2014,
                        help="First season start year (default: 2014).")
    parser.add_argument("--through-season", type=int, default=2025,
                        help="Last season start year (default: 2025).")
    parser.add_argument("--league", action="append", choices=LEAGUES,
                        help="League key to import; repeat to select multiple.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH,
                        help="Output SQLite file.")
    args = parser.parse_args()
    import_player_stats(
        args.from_season,
        args.through_season,
        args.league,
        args.database,
    )


if __name__ == "__main__":
    main()
