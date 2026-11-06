"""Read locally imported historical Understat player-season statistics."""

import math
import os
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "football" / "sql" / "football_player_stats.sqlite"
LEAGUE_NAMES = {
    "pl": "Premier League",
    "laliga": "La Liga",
    "bundesliga": "Bundesliga",
    "seriea": "Serie A",
    "ligue1": "Ligue 1",
}


def _player_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(
        "".join(char for char in ascii_name.casefold() if char.isalnum() or char.isspace())
        .split()
    )


def _team_key(value: str) -> str:
    from v4_backend.feature_builder import TEAM_NAME_ALIASES

    return _player_key(TEAM_NAME_ALIASES.get(value, value))


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def get_player_season_stats(
    player_name: str,
    league: str,
    current_team: str = "",
    selected_season: str | None = None,
    database_path: str | Path | None = None,
) -> dict:
    """Return the player's historical rows across supported leagues."""
    if not player_name or league not in LEAGUE_NAMES:
        return {
            "available": False, "archive_available": False,
            "seasons": [], "selected": None,
        }

    configured_path = database_path or os.environ.get(
        "FOOTBALL_PLAYER_STATS_DB_PATH", DEFAULT_DB_PATH
    )
    path = Path(configured_path)
    if not path.is_file():
        return {
            "available": False, "archive_available": False,
            "seasons": [], "selected": None,
        }

    uri = f"file:{quote(str(path.resolve().as_posix()), safe='/:')}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=2)) as connection:
            connection.row_factory = sqlite3.Row
            matches = connection.execute(
                """
                SELECT league, season, season_label, team, player, player_id,
                       player_key, position, appearances, minutes, goals, assists, xg, xa,
                       shots, key_passes, yellow_cards, red_cards, np_goals,
                       np_xg, xg_chain, xg_buildup
                FROM player_season_stats
                WHERE player_key = ? COLLATE NOCASE
                """,
                (name_key := _player_key(player_name),),
            ).fetchall()
    except sqlite3.Error as error:
        print(f"[football player stats] database read failed at {path}: {error}")
        return {
            "available": False, "archive_available": False,
            "seasons": [], "selected": None,
        }

    rows = [dict(row) for row in matches]
    if not rows:
        return {
            "available": False, "archive_available": True,
            "seasons": [], "selected": None,
        }

    current_team_key = _team_key(current_team) if current_team else ""
    player_ids = {row["player_id"] for row in rows if row["player_id"] is not None}
    if current_team_key and player_ids:
        preferred_ids = {
            row["player_id"] for row in rows
            if row["player_id"] is not None and _team_key(row["team"]) == current_team_key
        }
        if preferred_ids:
            player_ids = preferred_ids
    if player_ids:
        candidate_ids = {
            player_id: [row for row in rows if row["player_id"] == player_id]
            for player_id in player_ids
        }
        chosen_id = max(
            candidate_ids,
            key=lambda player_id: max(
                (row["season"] for row in candidate_ids[player_id]), default=0
            ),
        )
        rows = candidate_ids[chosen_id]

    for row in rows:
        minutes = _number(row["minutes"]) or 0
        xg = _number(row["xg"])
        xa = _number(row["xa"])
        row["minutes"] = int(minutes)
        row["xg"] = xg
        row["xa"] = xa
        row["xg_per90"] = round(xg * 90 / minutes, 3) if xg is not None and minutes else None
        row["xa_per90"] = round(xa * 90 / minutes, 3) if xa is not None and minutes else None
        for field in (
            "appearances", "goals", "assists", "shots", "key_passes",
            "yellow_cards", "red_cards", "np_goals", "np_xg",
            "xg_chain", "xg_buildup",
        ):
            row[field] = _number(row[field])
        for field in (
            "appearances", "goals", "assists", "shots", "key_passes",
            "yellow_cards", "red_cards", "np_goals",
        ):
            if row[field] is not None:
                row[field] = int(row[field])

    rows.sort(
        key=lambda row: (
            -row["season"],
            row["league"] != league,
            bool(current_team_key) and _team_key(row["team"]) != current_team_key,
            row["league"],
        )
    )
    for row in rows:
        row["selection_key"] = f"{row['league']}:{row['season']}"
        row["league_name"] = LEAGUE_NAMES.get(row["league"], row["league"])
    available_seasons = {row["selection_key"] for row in rows}
    selected = selected_season if selected_season in available_seasons else rows[0]["selection_key"]
    selected_row = next(row for row in rows if row["selection_key"] == selected)
    return {
        "available": True,
        "archive_available": True,
        "seasons": rows,
        "selected": selected_row,
        "selected_season": selected,
    }
