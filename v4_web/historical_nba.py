"""Read-only queries for NBA history stored in the local SQLite datasets."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

log = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parent
HISTORICAL_DB_PATH = APP_ROOT / "nba" / "sql" / "nba_stats.sqlite"
PLAY_BY_PLAY_DB_PATH = APP_ROOT / "nba" / "sql" / "nba_stats_pbp.sqlite"
HISTORICAL_SEASON_CUTOFF = "2024-25"
MIN_BEST_SHOT_ATTEMPTS = 25
_HISTORICAL_START_YEAR_CUTOFF = 2024


class HistoricalNBADataUnavailableError(RuntimeError):
    """Raised when a required local historical database is unavailable."""


def _configured_path(variable: str, default: Path) -> Path:
    configured = os.environ.get(variable)
    if not configured:
        return default
    path = Path(configured).expanduser()
    return path if path.is_absolute() else APP_ROOT / path


def get_historical_db_path() -> Path:
    """Return the main historical database path without opening the database."""
    return _configured_path("NBA_HISTORY_DB_PATH", HISTORICAL_DB_PATH)


def get_play_by_play_db_path() -> Path:
    """Return the optional, separately downloaded play-by-play database path."""
    return _configured_path("NBA_PBP_DB_PATH", PLAY_BY_PLAY_DB_PATH)


@contextmanager
def _read_only_connection(path: Path) -> Iterator[sqlite3.Connection]:
    if not path.is_file():
        raise HistoricalNBADataUnavailableError(
            f"NBA historical database not found at {path}. "
            "See v4_web/nba/sql/README.md for setup instructions."
        )

    connection: sqlite3.Connection | None = None
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        yield connection
    except sqlite3.Error as exc:
        log.error("NBA historical database query failed for %s: %s", path, exc)
        raise
    finally:
        if connection is not None:
            connection.close()


def _immutable_rows(rows: list[sqlite3.Row]) -> tuple[Mapping[str, object], ...]:
    return tuple(MappingProxyType(dict(row)) for row in rows)


def _public_rows(
    rows: tuple[Mapping[str, object], ...],
) -> list[dict[str, object]]:
    return [dict(row) for row in rows]


def _validate_player_id(player_id: int) -> int:
    if isinstance(player_id, bool) or not isinstance(player_id, int) or player_id <= 0:
        raise ValueError("player_id must be a positive integer")
    return player_id


def _validate_team_id(team_id: int) -> int:
    if isinstance(team_id, bool) or not isinstance(team_id, int) or team_id <= 0:
        raise ValueError("team_id must be a positive integer")
    return team_id


def _validate_season(season_year: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})", season_year)
    if not match or int(match.group(2)) != (int(match.group(1)) + 1) % 100:
        raise ValueError("season_year must use the NBA format YYYY-YY")
    if int(match.group(1)) > _HISTORICAL_START_YEAR_CUTOFF:
        raise ValueError(
            f"{season_year} is API-backed; historical data ends at "
            f"{HISTORICAL_SEASON_CUTOFF}"
        )
    return season_year


@lru_cache(maxsize=256)
def _player_seasons(
    database_path: str, player_id: int
) -> tuple[Mapping[str, object], ...]:
    with _read_only_connection(Path(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT season_year,
                   MAX(player_name) AS player_name,
                   GROUP_CONCAT(DISTINCT team_abbreviation) AS teams,
                   COUNT(DISTINCT game_id) AS games_played,
                   SUM(pts) AS points,
                   SUM(reb) AS rebounds,
                   SUM(ast) AS assists,
                   SUM(stl) AS steals,
                   SUM(blk) AS blocks,
                   SUM(fgm) AS field_goals_made,
                   SUM(fga) AS field_goals_attempted,
                   SUM(fg3m) AS three_pointers_made,
                   SUM(fg3a) AS three_pointers_attempted,
                   SUM(ftm) AS free_throws_made,
                   SUM(fta) AS free_throws_attempted,
                   SUM(min) AS minutes
            FROM player_boxscores
            WHERE player_id = ? AND season_year <= ? AND game_id LIKE '002%'
            GROUP BY season_year
            ORDER BY season_year DESC
            """,
            (player_id, HISTORICAL_SEASON_CUTOFF),
        ).fetchall()
    return _immutable_rows(rows)


def get_player_seasons(player_id: int) -> list[dict[str, object]]:
    """Return per-season box-score totals for a player's historical seasons."""
    player_id = _validate_player_id(player_id)
    return _public_rows(_player_seasons(str(get_historical_db_path()), player_id))


@lru_cache(maxsize=512)
def _player_game_log(
    database_path: str,
    player_id: int,
    season_year: str | None,
    limit: int,
) -> tuple[Mapping[str, object], ...]:
    query = """
        SELECT season_year, game_id, game_date, matchup, team_id,
               team_abbreviation, team_name, is_home, wl, min,
               pts, reb, ast, stl, blk, tov, fgm, fga, fg_pct,
               fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, plus_minus
        FROM player_boxscores
        WHERE player_id = ? AND season_year <= ? AND game_id LIKE '002%'
    """
    params: list[object] = [player_id, HISTORICAL_SEASON_CUTOFF]
    if season_year is not None:
        query += " AND season_year = ?"
        params.append(season_year)
    query += " ORDER BY game_date DESC, game_id DESC LIMIT ?"
    params.append(limit)
    with _read_only_connection(Path(database_path)) as connection:
        rows = connection.execute(query, params).fetchall()
    return _immutable_rows(rows)


def get_player_game_log(
    player_id: int,
    season_year: str | None = None,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Return recent player game rows, excluding API-backed seasons."""
    player_id = _validate_player_id(player_id)
    if season_year is not None:
        season_year = _validate_season(season_year)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer between 1 and 200")
    return _public_rows(
        _player_game_log(
            str(get_historical_db_path()), player_id, season_year, limit
        )
    )


@lru_cache(maxsize=512)
def _player_team_history(
    database_path: str, player_id: int
) -> tuple[Mapping[str, object], ...]:
    with _read_only_connection(Path(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT season_year, team_id, team_abbreviation, team_name,
                   COUNT(DISTINCT game_id) AS games_played,
                   SUM(CASE WHEN wl = 1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN wl = 0 THEN 1 ELSE 0 END) AS losses,
                   SUM(pts) AS points,
                   SUM(reb) AS rebounds,
                   SUM(ast) AS assists
            FROM player_boxscores
            WHERE player_id = ? AND season_year <= ? AND game_id LIKE '002%'
            GROUP BY season_year, team_id, team_abbreviation, team_name
            ORDER BY season_year DESC, games_played DESC
            """,
            (player_id, HISTORICAL_SEASON_CUTOFF),
        ).fetchall()
    return _immutable_rows(rows)


def get_player_team_history(player_id: int) -> list[dict[str, object]]:
    """Return historical teams and per-team player results by season."""
    player_id = _validate_player_id(player_id)
    return _public_rows(
        _player_team_history(str(get_historical_db_path()), player_id)
    )


@lru_cache(maxsize=256)
def _team_seasons(
    database_path: str, team_id: int
) -> tuple[Mapping[str, object], ...]:
    with _read_only_connection(Path(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT season_year, team_abbreviation, team_name,
                   COUNT(DISTINCT game_id) AS games_played,
                   SUM(CASE WHEN wl = 1 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN wl = 0 THEN 1 ELSE 0 END) AS losses,
                   SUM(pts) AS points_for,
                   SUM(pts - plus_minus) AS points_against
            FROM team_boxscores
            WHERE team_id = ? AND season_year <= ? AND game_id LIKE '002%'
            GROUP BY season_year, team_abbreviation, team_name
            ORDER BY season_year DESC
            """,
            (team_id, HISTORICAL_SEASON_CUTOFF),
        ).fetchall()
    return _immutable_rows(rows)


def get_team_seasons(team_id: int) -> list[dict[str, object]]:
    """Return historical season records and scoring totals for a team."""
    team_id = _validate_team_id(team_id)
    return _public_rows(_team_seasons(str(get_historical_db_path()), team_id))


@lru_cache(maxsize=256)
def _player_career_highs(
    database_path: str, player_id: int
) -> tuple[Mapping[str, object], ...]:
    fields = (
        ("Points", "pts"),
        ("Rebounds", "reb"),
        ("Assists", "ast"),
        ("Steals", "stl"),
        ("Blocks", "blk"),
        ("Three-pointers", "fg3m"),
    )
    selects = [
        f"""
        SELECT *
        FROM (
            SELECT '{label}' AS stat, {column} AS value,
                   season_year, game_date, matchup, game_id
            FROM player_boxscores
            WHERE player_id = ? AND season_year <= ?
                  AND game_id LIKE '002%' AND {column} IS NOT NULL
            ORDER BY {column} DESC, game_date ASC, game_id ASC
            LIMIT 1
        )
        """
        for label, column in fields
    ]
    params = tuple(
        value
        for _ in fields
        for value in (player_id, HISTORICAL_SEASON_CUTOFF)
    )
    with _read_only_connection(Path(database_path)) as connection:
        rows = connection.execute(" UNION ALL ".join(selects), params).fetchall()
    return _immutable_rows(rows)


def get_player_career_highs(player_id: int) -> list[dict[str, object]]:
    """Return best historical regular-season single-game box-score marks."""
    player_id = _validate_player_id(player_id)
    return _public_rows(
        _player_career_highs(str(get_historical_db_path()), player_id)
    )


def get_player_history(player_id: int) -> dict[str, list[dict[str, object]]]:
    """Gather the historical player-page datasets from the local database."""
    player_id = _validate_player_id(player_id)
    return {
        "seasons": get_player_seasons(player_id),
        "game_log": get_player_game_log(player_id),
        "team_history": get_player_team_history(player_id),
        "career_highs": get_player_career_highs(player_id),
    }


def _normalize_shot_type(subtype: object) -> str:
    label = " ".join(str(subtype or "").lower().split())
    if "fadeaway" in label or "fade away" in label:
        return "Fadeaway"
    if "putback" in label or "put back" in label:
        return "Putback"
    if "dunk" in label:
        return "Dunk"
    if "layup" in label or "finger roll" in label:
        return "Layup"
    if "hook" in label:
        return "Hook"
    if "tip" in label:
        return "Tip-in"
    if "jump" in label or "pullup" in label or "pull-up" in label:
        return "Jump shot"
    if "bank" in label:
        return "Bank shot"
    return "Other"


@lru_cache(maxsize=256)
def _player_shot_profile(
    database_path: str, player_id: int
) -> tuple[
    int,
    int,
    float | None,
    Mapping[str, object] | None,
    Mapping[str, object] | None,
    tuple[Mapping[str, object], ...],
]:
    with _read_only_connection(Path(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT subtype, action_type, COUNT(*) AS attempts
            FROM play_by_play
            WHERE person_id = ?
                  AND game_id LIKE '002%'
                  AND game_date < '2025-07-01'
                  AND action_type IN ('Made Shot', 'Missed Shot')
                  AND shot_value IN (2, 3)
            GROUP BY subtype, action_type
            """,
            (player_id,),
        ).fetchall()

    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        shot_type = _normalize_shot_type(row["subtype"])
        counts = totals.setdefault(shot_type, {"attempts": 0, "made": 0})
        count = int(row["attempts"])
        counts["attempts"] += count
        if row["action_type"] == "Made Shot":
            counts["made"] += count

    shot_types = [
        {
            "shot_type": label,
            **counts,
            "accuracy": counts["made"] / counts["attempts"],
        }
        for label, counts in sorted(
            totals.items(), key=lambda item: (-item[1]["attempts"], item[0])
        )
        if counts["attempts"] > 0
    ]
    attempts = sum(row["attempts"] for row in shot_types)
    made = sum(row["made"] for row in shot_types)
    favorite = shot_types[0] if shot_types else None
    eligible = [
        row for row in shot_types
        if row["shot_type"] != "Dunk"
        and row["attempts"] >= MIN_BEST_SHOT_ATTEMPTS
    ]
    best = max(
        eligible,
        key=lambda row: (row["accuracy"], row["attempts"], row["shot_type"]),
        default=None,
    )
    return (
        attempts,
        made,
        made / attempts if attempts else None,
        MappingProxyType(favorite) if favorite else None,
        MappingProxyType(best) if best else None,
        tuple(MappingProxyType(row) for row in shot_types),
    )


def get_player_shot_profile(player_id: int) -> dict[str, object]:
    """Return historical regular-season shot types and conversion rates."""
    player_id = _validate_player_id(player_id)
    attempts, made, accuracy, favorite, best, shot_types = _player_shot_profile(
        str(get_play_by_play_db_path()), player_id
    )
    return {
        "attempts": attempts,
        "made": made,
        "accuracy": accuracy,
        "favorite": dict(favorite) if favorite else None,
        "best": dict(best) if best else None,
        "best_min_attempts": MIN_BEST_SHOT_ATTEMPTS,
        "shot_types": _public_rows(shot_types),
    }


def clear_historical_nba_cache() -> None:
    """Clear cached historical query results after the local database changes."""
    _player_seasons.cache_clear()
    _player_game_log.cache_clear()
    _player_team_history.cache_clear()
    _team_seasons.cache_clear()
    _player_career_highs.cache_clear()
    _player_shot_profile.cache_clear()


def historical_nba_cache_info() -> dict[str, object]:
    """Return cache counters for diagnostics."""
    return {
        "player_seasons": _player_seasons.cache_info()._asdict(),
        "player_game_log": _player_game_log.cache_info()._asdict(),
        "player_team_history": _player_team_history.cache_info()._asdict(),
        "team_seasons": _team_seasons.cache_info()._asdict(),
        "player_career_highs": _player_career_highs.cache_info()._asdict(),
        "player_shot_profile": _player_shot_profile.cache_info()._asdict(),
    }
