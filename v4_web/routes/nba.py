"""
routes/nba.py  —  NBA routes for v4_web
========================================
All NBA page and API routes as a FastAPI APIRouter.
Mounted in main.py with prefix="/nba".

Routes:
    GET  /nba                       → nba_hub.html
    GET  /nba/match                 → nba_match.html
    GET  /nba/live/{game_id}        → JSON (live poll, called every 20s)
    GET  /nba/history               → nba_history.html
    GET  /nba/teams                 → nba_teams.html
    GET  /nba/team/{team_id}        → nba_team.html
    GET  /nba/debug                 → JSON (model + cache health)

Usage in main.py:
    from routes.nba import router as nba_router, nba_setup
    app.include_router(nba_router, prefix="/nba")
    # in lifespan:
    nba_setup(nba_model, predictions_db)
"""

import logging
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from constants import (
    NBA_TEAMS,
    NBA_CONFERENCES,
    NBA_TEAM_IDS,
    nba_team,
    nba_primary_colour,
    nba_tricode,
    EAT,
)
from nba_api_client import (
    get_today_scoreboard,
    get_tomorrow_scoreboard,
    get_yesterday_scoreboard,
    get_next_team_game,
    get_scheduled_game,
    get_conference_standings,
    get_live_game,
    get_live_pbp,
    get_box_score,
    get_espn_lineup,
    get_team_roster,
    get_player_profile,
    current_nba_season,
    build_live_snapshot,
    build_snapshot_series,
    to_eat,
    cache_stats,
    _parse_clock,
    _elapsed,
)
from nba_model import NBAModel
from historical_nba import (
    HISTORICAL_SEASON_CUTOFF,
    HistoricalNBADataUnavailableError,
    get_player_history,
    get_player_shot_profile,
    get_team_seasons,
)
from fan_votes import (
    cast_nba_vote,
    get_nba_vote_summary,
    init_nba_fan_votes,
)
from teamdata import _flag_img
import nba_predictions as pred_log

log = logging.getLogger(__name__)

router    = APIRouter()
templates = Jinja2Templates(
    directory=Path(__file__).resolve().parent.parent / "templates"
)

# ── Module-level refs (set by nba_setup() from main.py lifespan) ─────────────
_model: NBAModel | None = None
_db    = None   # predictions SQLite connection (same as football, league='nba')
_db_path: str | None = None


def nba_setup(model: NBAModel, db=None) -> None:
    """Called once from main.py lifespan after model is loaded."""
    global _model, _db, _db_path
    _model = model
    _db    = db
    _db_path = None
    pred_log.set_db(db)   # share the same connection
    if db is not None:
        pred_log.normalize_conferences()
        db_row = db.execute("PRAGMA database_list").fetchone()
        if db_row and db_row[2]:
            _db_path = str(db_row[2])
            init_nba_fan_votes(_db_path)
    try:
        _refresh_nba_elo()
    except Exception:
        log.exception("NBA Elo result update failed during startup")
    log.info("NBA routes ready — model loaded: %s", model.loaded)


def _refresh_nba_elo() -> None:
    if not _model or not _model.loaded or _db is None:
        return
    baseline_ratings = _model.get_base_elo_ratings()
    if not baseline_ratings:
        log.warning("Skipping NBA Elo update because baseline ratings are unavailable")
        return
    from model_updates import refresh_nba_elo

    update = refresh_nba_elo(_db, baseline_ratings)
    _model.set_elo_ratings(update["ratings"], updated=update["accepted"])
    if update.get("changed"):
        log.info(
            "NBA Elo update: %d completed regular-season games, active=%s",
            update["sample_size"], update["accepted"],
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enrich_game(game: dict) -> dict:
    """
    Add team colour, full name, and pregame win probability to a scoreboard
    game dict. Returns the enriched dict.
    """
    ht_id = game.get("home_team_id")
    at_id = game.get("away_team_id")
    ht    = nba_team(ht_id)
    at    = nba_team(at_id)

    pregame = {"home_win_prob": 0.5, "away_win_prob": 0.5,
               "home_elo": 1500, "away_elo": 1500, "elo_diff": 0}

    if _model and _model.loaded and ht_id and at_id:
        pregame = _model.predict_pregame(ht_id, at_id)

    if _model and _model.loaded:
        pred_log.log_pregame(
            game,
            pregame["home_win_prob"],
            _model.elo_rating_version,
        )

    enriched = {
        **game,
        "home_colour":    ht.get("primary", "#6b7280"),
        "away_colour":    at.get("primary", "#6b7280"),
        "home_full":      f"{ht.get('city','')} {ht.get('name','')}".strip(),
        "away_full":      f"{at.get('city','')} {at.get('name','')}".strip(),
        "home_conference": ht.get("conference", ""),
        "away_conference": at.get("conference", ""),
        "home_win_prob":  pregame["home_win_prob"],
        "away_win_prob":  pregame["away_win_prob"],
        "home_elo":       pregame["home_elo"],
        "away_elo":       pregame["away_elo"],
        "elo_diff":       pregame["elo_diff"],
        "tip_off_eat":    to_eat(game.get("game_time_utc", "")),
    }
    return _apply_saved_pregame(enriched)


def _apply_saved_pregame(game: dict) -> dict:
    """Use only the timestamp-verified snapshot as a pregame pick."""
    prediction = pred_log.get_prediction(str(game.get("game_id", "")))
    verified = pred_log.is_verified_pregame(prediction)
    if verified:
        home_prob = float(prediction["home_win_prob"])
        game["home_win_prob"] = home_prob
        game["away_win_prob"] = float(prediction["away_win_prob"])
        game["pregame_model_version"] = prediction.get("model_version")
    game["pregame_verified"] = verified
    game["pregame_logged_at"] = (
        prediction.get("pregame_logged_at") if verified else None
    )
    game["pregame_correct"] = (
        prediction.get("correct")
        if verified and prediction.get("correct") is not None
        else None
    )
    return game


def _is_future_game(game_time_utc: str, now: datetime) -> bool:
    try:
        tip = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return False
    return tip.tzinfo is not None and tip.astimezone(timezone.utc) > now


def _get_nba_history(limit: int = 50, conference: str = "all") -> list[dict]:
    """Pull recent NBA predictions from the shared predictions DB."""
    if _db is None:
        return []
    try:
        cur = _db.cursor()
        if conference not in ("East", "West"):
            cur.execute(
                "SELECT * FROM predictions WHERE league='nba' "
                "ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        else:
            cur.execute(
                "SELECT * FROM predictions WHERE league='nba' "
                "AND conference=? "
                "ORDER BY created_at DESC LIMIT ?",
                (conference, limit)
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        log.warning("NBA history DB error: %s", exc)
        return []


def _get_nba_team_season_record(team_id: int, season: str) -> dict[str, int]:
    """Return the team's actual logged wins and losses for the NBA season."""
    record = {"wins": 0, "losses": 0, "played": 0}
    if _db is None:
        return record
    try:
        season_start = f"{season[:4]}-09-01"
        season_end = f"{int(season[:4]) + 1}-09-01"
        rows = _db.execute(
            "SELECT home_team_id, home_score, away_score "
            "FROM predictions WHERE league='nba' "
            "AND (home_team_id=? OR away_team_id=?) "
            "AND home_score IS NOT NULL AND away_score IS NOT NULL "
            "AND COALESCE(NULLIF(substr(game_time_utc, 1, 10), ''), "
            "substr(created_at, 1, 10)) >= ? "
            "AND COALESCE(NULLIF(substr(game_time_utc, 1, 10), ''), "
            "substr(created_at, 1, 10)) < ?",
            (team_id, team_id, season_start, season_end),
        ).fetchall()
        record["played"] = len(rows)
        record["wins"] = sum(
            (home_score > away_score) == (home_team_id == team_id)
            for home_team_id, home_score, away_score in rows
        )
        record["losses"] = record["played"] - record["wins"]
    except Exception as exc:
        log.warning("NBA team season record error for %s: %s", team_id, exc)
    return record


def _ordinal_label(rank: int) -> str:
    """Format a conference rank as an ordinal label."""
    suffix = "th" if 10 <= rank % 100 <= 20 else {
        1: "st", 2: "nd", 3: "rd"
    }.get(rank % 10, "th")
    return f"{rank}{suffix}"


def _get_team_standing(
    standings: dict[str, list[dict]], team_id: int, conference: str
) -> dict | None:
    """Find a team's row in its conference table."""
    return next(
        (row for row in standings.get(conference, []) if row["team_id"] == team_id),
        None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def nba_hub(request: Request):
    """
    /nba — today's games hub.
    Shows live game strip + full day schedule with pregame win probabilities.
    """
    today_games_raw = get_today_scoreboard()
    tomorrow_games_raw = get_tomorrow_scoreboard()
    showing_tomorrow = not today_games_raw

    # Off-day fallback: keep the primary schedule useful when today is empty.
    games_raw = today_games_raw or tomorrow_games_raw

    games = [_enrich_game(g) for g in games_raw]
    next_day_games = (
        [] if showing_tomorrow
        else [_enrich_game(g) for g in tomorrow_games_raw]
    )
    live  = [g for g in games if g["status"] == "live"]
    pre   = [g for g in games if g["status"] == "pre"]
    final = [g for g in games if g["status"] == "final"]

    results_changed = False
    for game in final:
        if game.get("home_score") is not None and game.get("away_score") is not None:
            results_changed = pred_log.log_result(
                str(game["game_id"]),
                int(game["home_score"]),
                int(game["away_score"]),
            ) or results_changed
            game.update(_apply_saved_pregame(game))
    if results_changed:
        try:
            _refresh_nba_elo()
        except Exception:
            log.exception("NBA Elo result update failed after hub result logging")

    from datetime import timedelta
    d = datetime.now(EAT).date() + (timedelta(days=1) if showing_tomorrow else timedelta())
    hub_date = d.strftime("%A %b ") + str(d.day)

    standings = await run_in_threadpool(get_conference_standings)

    standings = await run_in_threadpool(get_conference_standings)

    return templates.TemplateResponse(
        request=request,
        name="nba_hub.html",
        context={
            "live_games":     live,
            "upcoming_games": pre,
            "final_games":    final,
            "tomorrow_games": next_day_games,
            "all_games":      games,
            "showing_tomorrow": showing_tomorrow,
            "model_info":     _model.info() if _model else {},
            "sport":          "nba",
            "topbar_title":   "NBA · Today",
            "live_count":     len(live),
            "title":          "NBA Hub",
            "standings":      standings,
        },
    )


@router.get("/match", response_class=HTMLResponse)
async def nba_match(
    request: Request,
    game_id: str = "",
    game_date: str = "",
):
    """
    /nba/match?game_id=... — live match page.
    Left panel: pregame analysis.
    Right panel: live win probability chart (Chart.js, polls /nba/live/{id}).
    """
    games = get_today_scoreboard()
    tomorrow_games = get_tomorrow_scoreboard()
    all_games = [*games, *tomorrow_games]

    # If no game_id given, use first live game, else first game today
    if not game_id:
        live = [g for g in games if g["status"] == "live"]
        game_id = (
            live[0]["game_id"] if live else
            games[0]["game_id"] if games else
            tomorrow_games[0]["game_id"] if tomorrow_games else ""
        )

    game = next((g for g in all_games if g["game_id"] == game_id), {})
    if not game and game_id and game_date:
        game = await run_in_threadpool(
            get_scheduled_game,
            game_id,
            game_date=game_date,
        ) or {}
    if not game:
        return templates.TemplateResponse(
            request=request,
            name="nba_match.html",
            context={"error": "Game not found", "game_id": game_id, "sport": "nba",
                     "topbar_title": "NBA · Match", "title": "NBA Match"},
        )

    enriched = _enrich_game(game)
    if (
        game.get("status") == "final"
        and game.get("home_score") is not None
        and game.get("away_score") is not None
    ):
        pred_log.log_result(
            str(game_id),
            int(game["home_score"]),
            int(game["away_score"]),
        )
        enriched = _apply_saved_pregame(enriched)
    ht_id    = game.get("home_team_id")
    at_id    = game.get("away_team_id")
    ht       = nba_team(ht_id)
    at       = nba_team(at_id)

    # Form (last 5 games) — pulled from history DB
    # Returns list of {"won": bool, "opponent": str, "score": str}
    home_form = []
    away_form = []

    return templates.TemplateResponse(
        request=request,
        name="nba_match.html",
        context={
            "game":           enriched,
            "game_id":        game_id,
            "game_date":      game.get("game_time_utc", "")[:10],
            "home_team":      ht,
            "away_team":      at,
            "home_form":      home_form,
            "away_form":      away_form,
            "fixtures":       [_enrich_game(g) for g in all_games],
            "model_info":     _model.info() if _model else {},
            "sport":          "nba",
            # Chart.js score and game-clock poll interval (ms)
            "poll_interval":  20000,
            "topbar_title":   f"{enriched['away_tricode']} @ {enriched['home_tricode']}",
            "live_count":     sum(1 for g in games if g["status"] == "live"),
            "title":          f"{enriched['away_tricode']} @ {enriched['home_tricode']}",
        },
    )


@router.get("/lineup/{game_id}")
async def nba_lineup(
    game_id: str,
    game_time_utc: str = "",
    home_tricode: str = "",
    away_tricode: str = "",
):
    """Return current on-court players, falling back to confirmed starters."""
    box = await run_in_threadpool(get_box_score, game_id)
    used_starters = False

    def is_flagged(value: object) -> bool:
        return value is True or str(value).strip().lower() in ("1", "true", "yes")

    def on_court(team: dict) -> list[dict]:
        nonlocal used_starters
        roster = team.get("players")
        if not isinstance(roster, list):
            roster = []
        active = [
            player for player in roster
            if isinstance(player, dict)
            and is_flagged(
                player.get(
                    "oncourt",
                    player.get("onCourt", player.get("isOnCourt")),
                )
            )
        ]
        if not active and roster:
            active = [
                player for player in roster
                if isinstance(player, dict)
                and is_flagged(
                    player.get("starter", player.get("isStarter"))
                )
            ][:5]
            used_starters = used_starters or bool(active)

        players = []
        for player in active[:5]:
            person_id = str(player.get("personId", ""))
            name = player.get("nameI") or player.get("name") or " ".join(
                filter(None, (player.get("firstName"), player.get("familyName")))
            )
            if not name and person_id.isdigit():
                name = f"Player {person_id}"
            players.append({
                "name": name,
                "jersey": player.get("jerseyNum", ""),
                "position": player.get("position", ""),
                "headshot": (
                    f"https://cdn.nba.com/headshots/nba/latest/260x190/{person_id}.png"
                    if person_id.isdigit() else ""
                ),
            })
        return players

    home_players = on_court(box.get("home", {}))
    away_players = on_court(box.get("away", {}))
    lineup_source = "NBA live" if not used_starters else "NBA starters"

    if (not home_players or not away_players) and all(
        (game_time_utc, home_tricode, away_tricode)
    ):
        fallback = await run_in_threadpool(
            get_espn_lineup,
            game_time_utc,
            home_tricode,
            away_tricode,
        )
        if not home_players and fallback.get("home"):
            home_players = fallback["home"]
            used_starters = True
        if not away_players and fallback.get("away"):
            away_players = fallback["away"]
            used_starters = True
        if fallback.get("available"):
            lineup_source = "ESPN starters"

    return JSONResponse({
        "available": bool(home_players or away_players),
        "used_starters": used_starters,
        "source": lineup_source,
        "home": home_players,
        "away": away_players,
    })


def _find_nba_game(game_id: str, game_date: str = "") -> dict | None:
    if game_date:
        return get_scheduled_game(game_id, game_date=game_date)
    for game in (*get_today_scoreboard(), *get_tomorrow_scoreboard()):
        if str(game.get("game_id", "")) == game_id:
            return game
    return None


@router.get("/api/fan-vote/{game_id}")
async def nba_fan_vote_summary(
    game_id: str,
    game_date: str = "",
    voter_id: str = "",
):
    if not _db_path:
        raise HTTPException(status_code=503, detail="NBA polls are unavailable")
    game = await run_in_threadpool(_find_nba_game, game_id, game_date)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    closed = game.get("status") != "pre"
    if not closed:
        try:
            tip = datetime.fromisoformat(
                game.get("game_time_utc", "").replace("Z", "+00:00")
            )
            closed = tip.tzinfo is None or tip <= datetime.now(tip.tzinfo)
        except (AttributeError, ValueError):
            closed = True
    return await run_in_threadpool(
        get_nba_vote_summary,
        _db_path,
        game_id,
        voter_id or None,
        closed=closed,
    )


@router.post("/api/fan-vote/{game_id}")
async def nba_submit_fan_vote(game_id: str, request: Request):
    if not _db_path:
        raise HTTPException(status_code=503, detail="NBA polls are unavailable")
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Expected a JSON vote payload") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON vote payload")

    game_date = body.get("game_date", "")
    if not isinstance(game_date, str):
        raise HTTPException(status_code=422, detail="Invalid game date")
    game = await run_in_threadpool(_find_nba_game, game_id, game_date)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    result = await run_in_threadpool(
        cast_nba_vote,
        _db_path,
        game_id,
        body.get("choice"),
        body.get("voter_id"),
        game.get("status", ""),
        game.get("game_time_utc", ""),
    )
    if result["reason"] in ("invalid_choice", "invalid_voter"):
        raise HTTPException(status_code=422, detail=result["reason"])
    if result["reason"] in ("invalid_kickoff", "poll_closed"):
        raise HTTPException(status_code=409, detail=result["reason"])
    if result["reason"] == "already_voted":
        raise HTTPException(
            status_code=409,
            detail={"reason": "already_voted", **result},
        )
    return result


@router.get("/history", response_class=HTMLResponse)
async def nba_history(request: Request, conference: str = "all"):
    """
    /nba/history — prediction accuracy tracker.
    """
    rows  = _get_nba_history(limit=100, conference=conference)
    for row in rows:
        row["pregame_verified"] = pred_log.is_verified_pregame(row)
    if conference not in ("all", "East", "West"):
        conference = "all"
    stats = pred_log.get_stats(conference)

    total     = stats["total"]
    correct   = stats["correct"]
    accuracy  = stats["accuracy"]
    avg_brier = stats["avg_brier"]

    return templates.TemplateResponse(
        request=request,
        name="nba_history.html",
        context={
            "rows":        rows,
            "total":       total,
            "correct":     correct,
            "accuracy":    accuracy,
            "avg_brier":   avg_brier,
            "stats":       stats,
            "conference":    conference,
            "sport":         "nba",
            "topbar_title":  "Prediction history",
            "live_count":    0,
            "title":         "NBA History",
        },
    )


@router.get("/teams", response_class=HTMLResponse)
async def nba_teams(request: Request):
    """
    /nba/teams — all 30 teams grouped by conference and division.
    Shows Elo rating and colour badge per team.
    """
    # Build team cards with current Elo
    team_cards = {}
    for tid in NBA_TEAM_IDS:
        t = nba_team(tid)
        elo = _model.get_elo(tid) if _model else 1500.0
        team_cards[tid] = {**t, "elo": round(elo, 0), "team_id": tid}

    return templates.TemplateResponse(
        request=request,
        name="nba_teams.html",
        context={
            "conferences":   NBA_CONFERENCES,
            "team_cards":    team_cards,
            "sport":         "nba",
            "topbar_title":  "Teams",
            "live_count":    0,
            "title":         "NBA Teams",
        },
    )


@router.get("/team/{team_id}", response_class=HTMLResponse)
async def nba_team_profile(request: Request, team_id: int):
    """
    /nba/team/{team_id} — single team profile.
    Season record, Elo trend, recent form, upcoming games.
    """
    if team_id not in NBA_TEAM_IDS:
        raise HTTPException(status_code=404, detail="NBA team not found")

    t   = nba_team(team_id)
    elo = _model.get_elo(team_id) if _model else 1500.0
    season = current_nba_season()
    roster = await run_in_threadpool(get_team_roster, team_id, season)
    standings = await run_in_threadpool(get_conference_standings)
    team_standing = _get_team_standing(standings, team_id, t["conference"])
    historical_team_seasons = []
    team_history_unavailable = False
    try:
        historical_team_seasons = await run_in_threadpool(get_team_seasons, team_id)
    except (HistoricalNBADataUnavailableError, sqlite3.Error) as exc:
        log.error("NBA historical team data unavailable for %s: %s", team_id, exc)
        team_history_unavailable = True

    # Recent games and the current-season record from the prediction history DB.
    recent = []
    team_record = _get_nba_team_season_record(team_id, season)
    if team_standing:
        team_record = {
            "wins": team_standing["wins"],
            "losses": team_standing["losses"],
            "played": team_standing["wins"] + team_standing["losses"],
        }
    if _db:
        try:
            cur = _db.cursor()
            cur.execute(
                "SELECT * FROM predictions WHERE league='nba' "
                "AND (home_team_id=? OR away_team_id=?) "
                "ORDER BY created_at DESC LIMIT 10",
                (team_id, team_id)
            )
            cols   = [d[0] for d in cur.description]
            recent = [dict(zip(cols, row)) for row in cur.fetchall()]
            for game in recent:
                game["pregame_verified"] = pred_log.is_verified_pregame(game)
                if game.get("home_score") is None or game.get("away_score") is None:
                    game["team_won"] = None
                    continue
                home_won = game["home_score"] > game["away_score"]
                game["team_won"] = int(home_won == (game.get("home_team_id") == team_id))
        except Exception as exc:
            log.warning("Team history error: %s", exc)

    # Prefer the next scheduled fixture; a team's latest final must not hide it.
    now_utc = datetime.now(timezone.utc)
    today_and_tomorrow = [*get_today_scoreboard(), *get_tomorrow_scoreboard()]
    team_games = [
        game for game in today_and_tomorrow
        if team_id in (game.get("home_team_id"), game.get("away_team_id"))
    ]
    next_raw = next((
        game for game in sorted(team_games, key=lambda g: g.get("game_time_utc", ""))
        if game.get("status") == "pre"
        and _is_future_game(game.get("game_time_utc", ""), now_utc)
    ), None)
    if not next_raw:
        next_raw = await run_in_threadpool(get_next_team_game, team_id)
    if not next_raw:
        next_raw = next(
            (game for game in team_games if game.get("status") == "live"),
            None,
        )
    next_game = _enrich_game(next_raw) if next_raw else None
    if next_game:
        next_game["game_date"] = str(next_game.get("game_time_utc") or "")[:10]

    return templates.TemplateResponse(
        request=request,
        name="nba_team.html",
        context={
            "team":          {
                **t,
                "elo": round(elo, 0),
                "team_id": team_id,
                "standing_label": _ordinal_label(team_standing["rank"]) if team_standing else None,
            },
            "recent":        recent,
            "team_record":   team_record,
            "upcoming":      [next_game] if next_game else [],
            "standings":     standings,
            "roster":        roster,
            "roster_season": season,
            "historical_seasons": historical_team_seasons,
            "historical_season_cutoff": HISTORICAL_SEASON_CUTOFF,
            "team_history_unavailable": team_history_unavailable,
            "sport":         "nba",
            "topbar_title":  f"{t.get('city','')} {t.get('name','')}",
            "live_count":    0,
            "title":         f"{t.get('tricode','')} · NBA",
        },
    )


def _player_experience(
    seasons: list[dict], history_available: bool
) -> dict[str, object]:
    played = set()
    if history_available:
        for row in seasons:
            season = str(row.get("SEASON_ID") or "")
            try:
                games = float(row.get("GP") or 0)
            except (TypeError, ValueError):
                continue
            if re.fullmatch(r"\d{4}-\d{2}", season) and games > 0:
                played.add(season)

    count = len(played)
    tiers = (
        (1, "Rookie", "rookie"),
        (3, "Early Career", "early"),
        (7, "Established", "established"),
        (11, "Veteran", "veteran"),
        (15, "Long-Tenured", "long"),
        (float("inf"), "Career Veteran", "career"),
    )
    tier = next((label, key) for maximum, label, key in tiers if count <= maximum)
    return {
        "available": history_available and count > 0,
        "seasons_played": count,
        "label": tier[0] if count else None,
        "key": tier[1] if count else None,
    }


def _player_honors(
    awards: list[dict],
    awards_available: bool,
    team_timeline: list[dict] | None = None,
) -> dict[str, object]:
    counted: dict[str, set[str]] = {"championships": set(), "mvps": set()}
    championship_teams: dict[str, dict] = {}
    timeline_by_season = {
        row["season"]: row["teams"] for row in (team_timeline or [])
    }
    for award in awards:
        description = str(award.get("DESCRIPTION") or "").strip().casefold()
        season = str(award.get("SEASON") or "").strip()
        if not season:
            continue
        if description == "nba champion":
            counted["championships"].add(season)
            try:
                team_id = int(award.get("TEAM_ID") or 0)
            except (TypeError, ValueError):
                team_id = 0
            team_id = team_id if team_id in NBA_TEAM_IDS else None
            award_team_name = str(award.get("TEAM") or "").strip()
            normalized_award_team = " ".join(award_team_name.casefold().split())
            if not team_id and normalized_award_team:
                for candidate_id, candidate in NBA_TEAMS.items():
                    candidate_name = " ".join(
                        f"{candidate.get('city', '')} {candidate.get('name', '')}"
                        .casefold()
                        .split()
                    )
                    if normalized_award_team == candidate_name:
                        team_id = candidate_id
                        break
            if not team_id:
                season_teams = timeline_by_season.get(season, [])
                if len(season_teams) == 1:
                    team_id = season_teams[0]["team_id"]
            team = nba_team(team_id) if team_id else {}
            entry = {
                "season": season,
                "team_id": team_id,
                "team_name": (
                    f"{team.get('city', '')} {team.get('name', '')}".strip()
                    if team
                    else award_team_name
                ),
                "tricode": team.get("tricode") or "",
            }
            existing = championship_teams.get(season)
            if not existing or (not existing["team_id"] and team_id):
                championship_teams[season] = entry
        elif description == "nba most valuable player":
            counted["mvps"].add(season)
    return {
        "available": awards_available,
        "championships": len(counted["championships"]),
        "championship_seasons": [
            championship_teams[season]
            for season in sorted(counted["championships"])
        ],
        "mvps": len(counted["mvps"]),
    }


def _player_team_timeline(
    career_seasons: list[dict], historical_teams: list[dict]
) -> dict[str, object]:
    by_season_team: dict[tuple[str, int], dict] = {}
    for row in [*historical_teams, *career_seasons]:
        season = str(row.get("SEASON_ID") or row.get("season_year") or "")
        abbreviation = str(
            row.get("TEAM_ABBREVIATION") or row.get("team_abbreviation") or ""
        ).upper()
        if not re.fullmatch(r"\d{4}-\d{2}", season) or abbreviation in ("", "TOT"):
            continue
        try:
            team_id = int(row.get("TEAM_ID") or row.get("team_id") or 0)
            games = float(row.get("GP") or row.get("games_played") or 0)
        except (TypeError, ValueError):
            continue
        if team_id not in NBA_TEAM_IDS or games <= 0:
            continue
        team = nba_team(team_id)
        by_season_team[(season, team_id)] = {
            "season": season,
            "team_id": team_id,
            "tricode": team.get("tricode") or abbreviation,
            "name": f"{team.get('city', '')} {team.get('name', '')}".strip(),
            "primary": team.get("primary", "#6b7280"),
        }

    grouped: dict[str, list[dict]] = {}
    for row in sorted(
        by_season_team.values(), key=lambda item: (item["season"], item["team_id"])
    ):
        grouped.setdefault(row["season"], []).append(
            {key: value for key, value in row.items() if key != "season"}
        )
    seasons = [
        {"season": season, "teams": teams}
        for season, teams in sorted(grouped.items())
    ]
    return {
        "seasons": seasons,
        "team_ids": {row["team_id"] for row in by_season_team.values()},
    }


def _has_one_franchise_career(
    timeline: dict[str, object], experience: dict[str, object]
) -> bool:
    return (
        bool(experience.get("available"))
        and int(experience.get("seasons_played") or 0) >= 3
        and len(timeline.get("team_ids", set())) == 1
    )


@router.get("/player/{player_id}", response_class=HTMLResponse)
async def nba_player_profile(request: Request, player_id: int):
    """Show NBA bio, honors, career tenure and historical regular seasons."""
    if player_id <= 0:
        raise HTTPException(status_code=404, detail="NBA player not found")

    profile = await run_in_threadpool(get_player_profile, player_id)
    info = profile.get("info", {})
    history_unavailable = False
    try:
        historical = await run_in_threadpool(get_player_history, player_id)
    except (HistoricalNBADataUnavailableError, sqlite3.Error) as exc:
        log.error("NBA player history unavailable for %s: %s", player_id, exc)
        historical = {
            "seasons": [],
            "game_log": [],
            "team_history": [],
            "career_highs": [],
        }
        history_unavailable = True
    shot_history_unavailable = False
    try:
        shot_profile = await run_in_threadpool(get_player_shot_profile, player_id)
    except (HistoricalNBADataUnavailableError, sqlite3.Error) as exc:
        log.error("NBA player shot history unavailable for %s: %s", player_id, exc)
        shot_profile = {}
        shot_history_unavailable = True

    current_season = current_nba_season()
    current_start_year = int(current_season[:4])
    api_seasons = []
    for row in profile.get("seasons", []):
        season_id = str(row.get("SEASON_ID") or row.get("season_year") or "")
        match = re.fullmatch(r"(\d{4})-(\d{2})", season_id)
        if not match:
            continue
        start_year = int(match.group(1))
        if start_year not in (current_start_year - 1, current_start_year):
            continue
        season_id = f"{start_year}-{str(start_year + 1)[-2:]}"
        api_seasons.append({**row, "SEASON_ID": season_id, "_source": "NBA API"})

    historical_seasons = [
        {
            "SEASON_ID": row["season_year"],
            "TEAM_ABBREVIATION": row.get("teams") or "—",
            "GP": row.get("games_played"),
            "PTS": row.get("points"),
            "REB": row.get("rebounds"),
            "AST": row.get("assists"),
            "MIN": round(row["minutes"], 1) if row.get("minutes") is not None else None,
            "_source": "Historical database",
        }
        for row in historical["seasons"]
    ]
    seasons = sorted(
        [*api_seasons, *historical_seasons],
        key=lambda row: row["SEASON_ID"],
        reverse=True,
    )

    info = dict(info)
    if not info and historical["seasons"]:
        info["DISPLAY_FIRST_LAST"] = historical["seasons"][0].get("player_name")
    if info:
        team_id = info.get("TEAM_ID")
        try:
            team_id = int(team_id)
        except (TypeError, ValueError):
            team_id = None
        if team_id not in NBA_TEAM_IDS:
            team_id = None
    else:
        team_id = None

    career_seasons = profile.get("career_seasons", [])
    experience = _player_experience(
        career_seasons, bool(profile.get("career_history_available"))
    )
    team_timeline = _player_team_timeline(
        career_seasons, historical["team_history"]
    )
    honors = _player_honors(
        profile.get("awards", []),
        bool(profile.get("awards_available")),
        team_timeline["seasons"],
    )
    country_flag = _flag_img(str(info.get("COUNTRY") or ""))
    loyalty_badge = _has_one_franchise_career(team_timeline, experience)

    player_name = info.get("DISPLAY_FIRST_LAST") or f"Player {player_id}"
    unavailable = not bool(profile.get("info")) and not bool(historical["seasons"])
    return templates.TemplateResponse(
        request=request,
        name="nba_player.html",
        context={
            "player_id": player_id,
            "player": info,
            "headline": profile.get("headline", {}),
            "seasons": seasons,
            "game_log": historical["game_log"],
            "team_history": historical["team_history"],
            "career_highs": historical["career_highs"],
            "shot_profile": shot_profile,
            "experience": experience,
            "honors": honors,
            "country_flag": country_flag,
            "team_timeline": team_timeline["seasons"],
            "team_timeline_available": bool(team_timeline["seasons"]),
            "loyalty_badge": loyalty_badge,
            "shot_history_unavailable": shot_history_unavailable,
            "historical_season_cutoff": HISTORICAL_SEASON_CUTOFF,
            "history_unavailable": history_unavailable,
            "team": nba_team(team_id) if team_id else {},
            "team_id": team_id,
            "unavailable": unavailable,
            "sport": "nba",
            "topbar_title": player_name,
            "live_count": 0,
            "title": f"{player_name} · NBA",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# API / POLL ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/live/{game_id}")
async def nba_live_poll(game_id: str, game_date: str = ""):
    """
    /nba/live/{game_id} — live poll endpoint.
    Called by Chart.js every 20 seconds from nba_match.html.

    Returns:
        {
            "home_score":     int,
            "away_score":     int,
            "status":         "pre" | "live" | "final",
            "status_text":    str,
            "period":         int,
            "clock":          str,
            "home_win_prob":  float,
            "away_win_prob":  float,
            "elo_prior_weight": float,
            "confidence":     str,
            "possession":     float,
            "home_in_bonus":  int,
            "away_in_bonus":  int,
            "source":         str,
        }
    """
    game = await run_in_threadpool(get_live_game, game_id, game_date)
    if not game:
        return JSONResponse({"error": "game not found"}, status_code=404)

    ht_id = game.get("home_team_id")
    at_id = game.get("away_team_id")

    # Pregame Elo for the snapshot
    home_elo = _model.get_elo(ht_id) if _model else 1500.0
    away_elo = _model.get_elo(at_id) if _model else 1500.0

    status = game.get("status", "pre")

    # Before tip-off there is no game state: period=0 and an empty clock would
    # be read as "Q1, 0:00 left in the quarter". Use the Elo pregame prob instead.
    snapshot = {}
    if status != "pre":
        snapshot = build_live_snapshot(
            game_id,
            home_elo=home_elo,
            away_elo=away_elo,
            is_playoffs=0,
        )

    # Run inference
    if snapshot and _model and _model.loaded:
        pred = _model.predict_live(snapshot)
    else:
        # Fallback: Elo-only pregame prob
        pred = _model.predict_pregame(ht_id, at_id) if _model else {
            "home_win_prob": 0.5, "away_win_prob": 0.5,
            "elo_prior_weight": 1.0, "confidence": "low", "source": "fallback",
        }

    home_score = game.get("home_score", 0)
    away_score = game.get("away_score", 0)
    game_status = game.get("status", "pre")

    # When a game goes final, fill in the result
    if game_status == "final":
        if pred_log.log_result(game_id, home_score, away_score):
            try:
                _refresh_nba_elo()
            except Exception:
                log.exception("NBA Elo result update failed after live result logging")

    return JSONResponse({
        # Scoreboard state
        "home_score":       home_score,
        "away_score":       away_score,
        "status":           game_status,
        "status_text":      game.get("status_text", ""),
        "period":           game.get("period", 0),
        "clock":            game.get("clock", ""),
        # Model output
        "home_win_prob":    pred.get("home_win_prob", 0.5),
        "away_win_prob":    pred.get("away_win_prob", 0.5),
        "elo_prior_weight": pred.get("elo_prior_weight", 1.0),
        "confidence":       pred.get("confidence", "low"),
        # Game state features (for UI indicators)
        "possession":       snapshot.get("possession", 0.5) if snapshot else 0.5,
        "home_in_bonus":    snapshot.get("home_in_bonus", 0) if snapshot else 0,
        "away_in_bonus":    snapshot.get("away_in_bonus", 0) if snapshot else 0,
        "source":           pred.get("source", "unknown"),
        # Seconds since tip-off — x-axis position for the chart (0 before tip)
        "elapsed_sec":      (_elapsed(game.get("period") or 1, _parse_clock(game.get("clock", "")))
                             if game_status != "pre" else 0),
    })


@router.get("/curve/{game_id}")
async def nba_curve(game_id: str, game_date: str = ""):
    """
    /nba/curve/{game_id} — full win-probability history for the chart.
    Called once when nba_match.html loads (and once more at the final buzzer),
    so a game opened mid-way shows how it got there, not an empty chart.

    One point per scoring play, plus a tip-off point at the pregame Elo prob.
    Returns: {"points": [{"t", "hp", "hs", "as", "period", "clock", "desc"}]}
    """
    game = await run_in_threadpool(get_live_game, game_id, game_date)
    if not game:
        return JSONResponse({"error": "game not found"}, status_code=404)

    ht_id = game.get("home_team_id")
    at_id = game.get("away_team_id")
    pre   = (_model.predict_pregame(ht_id, at_id) if _model
             else {"home_win_prob": 0.5, "home_elo": 1500.0, "away_elo": 1500.0})

    points = [{"t": 0, "hp": pre["home_win_prob"], "hs": 0, "as": 0,
               "period": 1, "clock": "12:00", "desc": "Tip-off (pregame Elo)"}]

    if game.get("status") == "pre":
        return JSONResponse({"points": points})

    series = build_snapshot_series(
        game_id, ht_id,
        home_elo=pre["home_elo"], away_elo=pre["away_elo"], is_playoffs=0,
        game=game,
    )
    if series and _model and _model.loaded:
        preds = _model.predict_live_batch([s["snapshot"] for s in series])
        for s, p in zip(series, preds):
            points.append({
                "t":      round(s["elapsed_sec"], 1),
                "hp":     p["home_win_prob"],
                "hs":     s["home_score"],
                "as":     s["away_score"],
                "period": s["period"],
                "clock":  s["clock"],
                "desc":   s["description"],
            })
    elif game.get("status") == "final":
        home_won = game.get("home_score", 0) > game.get("away_score", 0)
        points.append({
            "t": 2880,
            "hp": 1.0 if home_won else 0.0,
            "hs": game.get("home_score", 0),
            "as": game.get("away_score", 0),
            "period": 4,
            "clock": "0:00",
            "desc": "Final result · play-by-play unavailable",
        })

    return JSONResponse({"points": points})


@router.get("/debug")
async def nba_debug():
    """
    /nba/debug — model health + cache stats.
    Not linked in the UI, useful during development.
    """
    return JSONResponse({
        "model":  _model.info() if _model else {"loaded": False},
        "cache":  cache_stats(),
    })
