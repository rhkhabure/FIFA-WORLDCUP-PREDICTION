"""
routes/nba.py  —  NBA routes for v4_web
========================================
All NBA page and API routes as a FastAPI APIRouter.
Mounted in main.py with prefix="/nba".

Routes:
    GET  /nba                       → nba_hub.html
    GET  /nba/match                 → nba_match.html
    GET  /nba/live/{game_id}        → JSON (live poll, called every 30s)
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
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

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
    get_live_game,
    get_live_pbp,
    build_live_snapshot,
    to_eat,
    cache_stats,
)
from nba_model import NBAModel

log = logging.getLogger(__name__)

router    = APIRouter()
templates = Jinja2Templates(
    directory=Path(__file__).resolve().parent.parent / "templates"
)

# ── Module-level refs (set by nba_setup() from main.py lifespan) ─────────────
_model: NBAModel | None = None
_db    = None   # predictions SQLite connection (same as football, league='nba')


def nba_setup(model: NBAModel, db=None) -> None:
    """Called once from main.py lifespan after model is loaded."""
    global _model, _db
    _model = model
    _db    = db
    log.info("NBA routes ready — model loaded: %s", model.loaded)


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

    return {
        **game,
        "home_colour":    ht.get("primary", "#6b7280"),
        "away_colour":    at.get("primary", "#6b7280"),
        "home_full":      f"{ht.get('city','')} {ht.get('name','')}".strip(),
        "away_full":      f"{at.get('city','')} {at.get('name','')}".strip(),
        "home_win_prob":  pregame["home_win_prob"],
        "away_win_prob":  pregame["away_win_prob"],
        "home_elo":       pregame["home_elo"],
        "away_elo":       pregame["away_elo"],
        "elo_diff":       pregame["elo_diff"],
        "tip_off_eat":    to_eat(game.get("game_time_utc", "")),
    }


def _get_nba_history(limit: int = 50, conference: str = "all") -> list[dict]:
    """Pull recent NBA predictions from the shared predictions DB."""
    if _db is None:
        return []
    try:
        cur = _db.cursor()
        if conference == "all":
            cur.execute(
                "SELECT * FROM predictions WHERE league='nba' "
                "ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        else:
            cur.execute(
                "SELECT * FROM predictions WHERE league='nba' "
                "AND conference=? ORDER BY created_at DESC LIMIT ?",
                (conference, limit)
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        log.warning("NBA history DB error: %s", exc)
        return []


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
    games_raw = get_today_scoreboard()
    games     = [_enrich_game(g) for g in games_raw]

    live   = [g for g in games if g["status"] == "live"]
    pre    = [g for g in games if g["status"] == "pre"]
    final  = [g for g in games if g["status"] == "final"]

    return templates.TemplateResponse(
        request=request,
        name="nba_hub.html",
        context={
            "live_games":     live,
            "upcoming_games": pre,
            "final_games":    final,
            "all_games":      games,
            "model_info":     _model.info() if _model else {},
            "sport":          "nba",
            "topbar_title":   "NBA · Today",
            "live_count":     len(live),
            "title":          "NBA Hub",
        },
    )


@router.get("/match", response_class=HTMLResponse)
async def nba_match(request: Request, game_id: str = ""):
    """
    /nba/match?game_id=... — live match page.
    Left panel: pregame analysis.
    Right panel: live win probability chart (Chart.js, polls /nba/live/{id}).
    """
    games = get_today_scoreboard()

    # If no game_id given, use first live game, else first game today
    if not game_id:
        live = [g for g in games if g["status"] == "live"]
        game_id = live[0]["game_id"] if live else (games[0]["game_id"] if games else "")

    game = next((g for g in games if g["game_id"] == game_id), {})
    if not game:
        return templates.TemplateResponse(
            request=request,
            name="nba_match.html",
            context={"error": "Game not found", "game_id": game_id, "sport": "nba"},
        )

    enriched = _enrich_game(game)
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
            "home_team":      ht,
            "away_team":      at,
            "home_form":      home_form,
            "away_form":      away_form,
            "model_info":     _model.info() if _model else {},
            "sport":          "nba",
            # Chart.js poll interval (ms)
            "poll_interval":  30000,
        },
    )


@router.get("/history", response_class=HTMLResponse)
async def nba_history(request: Request, conference: str = "all"):
    """
    /nba/history — prediction accuracy tracker.
    """
    rows = _get_nba_history(limit=100, conference=conference)

    # Compute running accuracy stats
    total    = len(rows)
    correct  = sum(1 for r in rows if r.get("correct"))
    accuracy = round(correct / total * 100, 1) if total else 0.0

    briers = [r["brier"] for r in rows if r.get("brier") is not None]
    avg_brier = round(sum(briers) / len(briers), 4) if briers else None

    return templates.TemplateResponse(
        request=request,
        name="nba_history.html",
        context={
            "rows":        rows,
            "total":       total,
            "correct":     correct,
            "accuracy":    accuracy,
            "avg_brier":   avg_brier,
            "conference":  conference,
            "sport":       "nba",
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
            "conferences": NBA_CONFERENCES,
            "team_cards":  team_cards,
            "sport":       "nba",
        },
    )


@router.get("/team/{team_id}", response_class=HTMLResponse)
async def nba_team_profile(request: Request, team_id: int):
    """
    /nba/team/{team_id} — single team profile.
    Season record, Elo trend, recent form, upcoming games.
    """
    t   = nba_team(team_id)
    elo = _model.get_elo(team_id) if _model else 1500.0

    # Recent games from history DB involving this team
    recent = []
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
        except Exception as exc:
            log.warning("Team history error: %s", exc)

    # Upcoming games from today's schedule
    all_games = get_today_scoreboard()
    upcoming  = [
        _enrich_game(g) for g in all_games
        if g.get("home_team_id") == team_id or g.get("away_team_id") == team_id
    ]

    return templates.TemplateResponse(
        request=request,
        name="nba_team.html",
        context={
            "team":        {**t, "elo": round(elo, 0), "team_id": team_id},
            "recent":      recent,
            "upcoming":    upcoming,
            "sport":       "nba",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# API / POLL ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/live/{game_id}")
async def nba_live_poll(game_id: str):
    """
    /nba/live/{game_id} — live poll endpoint.
    Called by Chart.js every 30 seconds from nba_match.html.

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
    game = get_live_game(game_id)
    if not game:
        return JSONResponse({"error": "game not found"}, status_code=404)

    ht_id = game.get("home_team_id")
    at_id = game.get("away_team_id")

    # Pregame Elo for the snapshot
    home_elo = _model.get_elo(ht_id) if _model else 1500.0
    away_elo = _model.get_elo(at_id) if _model else 1500.0

    # Build feature snapshot from live PBP
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

    return JSONResponse({
        # Scoreboard state
        "home_score":       game.get("home_score", 0),
        "away_score":       game.get("away_score", 0),
        "status":           game.get("status", "pre"),
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
    })


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
