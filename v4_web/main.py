"""
main.py  —  V4.2 Quant Terminal
=================================
FastAPI app for the league football win-probability dashboard.

Architecture (confirmed by Phase 3 validation):
  PRE-GAME  ->  Dixon-Coles bivariate Poisson (v4_priors.json)
                draw_propensity = 0.10
  LIVE      ->  Neural net (football_v4.pth) once minute > 0
                Score + time features dominate mid-match

Data sources:
  Fixtures / scores  ->  football-data.org free tier (X-Auth-Token)
  Upcoming fixtures  ->  FPL API (no key, free, EAT timezone)

Run:  uvicorn main:app --reload --port 8000
"""

# ── Standard library ──────────────────────────────────────────────────────────
import concurrent.futures
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.stats import poisson

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

# ── Local modules — data sources ──────────────────────────────────────────────
from bbs import (
    get_match_detail as bbs_get_match_detail,
    get_today_matches as bbs_get_today_matches,
    get_upcoming_fixtures as bbs_get_upcoming_fixtures,
    has_key as bbs_ok,
)
from football_co_uk import get_team_results_historical, get_pl_standings_historical
from footballdata import (
    get_live_match_data,
    get_last_completed_pl_match,
    get_finished_match,
    find_finished_match_by_teams,
    get_upcoming_fixtures_fd,
    _populate_finished_cache,
    PL_CODE,
)
from fotmob import (
    get_lineup,
    get_live_xg,
    get_fotmob_match_id,
    get_match_status_from_date_cache,
    fotmob_to_fpl_team_name,
    has_key as fotmob_available,
    _lineup_cache,
    _refresh_date_cache_if_stale,
)
from fpl import (
    get_upcoming_fixtures,
    get_team_map,
    warm_cache as fpl_warm,
    _cache as fpl_cache,
    _CACHE_TTLS,
)
from lineup_adjustment import compute_lineup_adjusted_odds, get_absent_key_players
from prediction_job import run_prediction_job
from predictions import get_all_predictions, get_accuracy_stats
from scoreline_matrix import build_scoreline_svg
from simulate import simulate_cl_tournament, simulate_pl_season
from teamdata import (
    get_team_profile,
    get_team_season_results,
    get_next_fixture,
    get_last_n_results,
    get_standings,
)
from thesportsdb import search_player, get_career_history
from timeline import build_match_timeline_svg
from utils import (
    generate_pitch_svg_horizontal,
    generate_pitch_svg_vertical,
    get_theme_for_team,
    get_formation_for_team,
    get_squad_for_team,
    get_crest_url,
    get_crest_proxy_url,
    _CREST_IDS,
    _KITS_PUBLIC,
    TEAM_MANAGERS,
)
from v4_backend.feature_builder import DCStrengthLookup, TEAM_NAME_ALIASES
from routes.match import router as match_router, setup as match_setup, dc_pregame
from constants import EAT, DRAW_PROPENSITY, LEAGUE_FILTERS, LEAGUE_MAP

# ── Constants ─────────────────────────────────────────────────────────────────
# DRAW_PROPENSITY, EAT, LEAGUE_FILTERS, LEAGUE_MAP imported from constants.py
LEAGUE_KEY = "ENG-Premier League"   # default league for backward compat

# ── Paths ─────────────────────────────────────────────────────────────────────
def find_file(filename):
    for p in ROOT.parent.rglob(filename):
        return p
    return None

PRIORS_PATH = find_file("v4_priors.json")
MODEL_PATH  = find_file("football_v4.pth")
SCALER_PATH = find_file("scaler_v4.pkl")

# ── Load DC priors ─────────────────────────────────────────────────────────────
if PRIORS_PATH and PRIORS_PATH.exists():
    with open(PRIORS_PATH) as f:
        priors_db = json.load(f)
    dc_lookup = DCStrengthLookup(PRIORS_PATH)
else:
    priors_db = {}
    dc_lookup = None
    print("WARNING: v4_priors.json not found")

# ── Load neural net ────────────────────────────────────────────────────────────
class FootballWinProbNet(nn.Module):
    def __init__(self, n_features=11, n_classes=3, h1=40, h2=20, dropout=0.30):
        super().__init__()
        self.fc1  = nn.Linear(n_features, h1)
        self.fc2  = nn.Linear(h1, h2)
        self.head = nn.Linear(h2, n_classes)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.ReLU()

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return self.head(x)


nn_model, nn_scaler, nn_T = None, None, 1.0
if MODEL_PATH and MODEL_PATH.exists() and SCALER_PATH and SCALER_PATH.exists():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    nn_model = FootballWinProbNet(**ckpt["arch"])
    nn_model.load_state_dict(ckpt["model_state"])
    nn_model.eval()
    nn_T = ckpt.get("temperature", 1.0)
    with open(SCALER_PATH, "rb") as f:
        nn_scaler = pickle.load(f)
    print(f"Loaded football_v4.pth  T={nn_T:.3f}")
else:
    print("WARNING: football_v4.pth or scaler not found")

import asyncio
from contextlib import asynccontextmanager
from prediction_job import schedule_prediction_job
from predictions import (init_db, log_pre_lineup, log_post_lineup,
                         log_result, get_all_predictions,
                         save_match_snapshot, get_match_snapshot)

@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan — starts background jobs and warms caches on startup."""
    init_db()

    # Warm FPL fixture cache so first page load is instant
    # Runs in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, fpl_warm)

    if priors_db:
        asyncio.create_task(
            schedule_prediction_job(priors_db, LEAGUE_KEY, interval_minutes=30)
        )
        print("[startup] Prediction logging job scheduled (every 30 min)")
    else:
        print("[startup] No priors loaded — prediction job skipped")

    match_setup(
        priors_db, nn_model, nn_scaler, nn_T, dc_lookup,
        templates, TEAM_NAME_ALIASES, LEAGUE_KEY, LEAGUE_CONTEXTS,
    )
    yield
    # Shutdown: nothing to clean up (SQLite handles its own flush)

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Melios Dira Odds", lifespan=lifespan)
app.include_router(match_router)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")

# ── League context helper ──────────────────────────────────────────
LEAGUE_CONTEXTS = {
    "pl":         {"league": "pl",         "league_code": "PL",  "league_name": "Premier League"},
    "laliga":     {"league": "laliga",     "league_code": "LL",  "league_name": "La Liga"},
    "bundesliga": {"league": "bundesliga", "league_code": "BL",  "league_name": "Bundesliga"},
    "seriea":     {"league": "seriea",     "league_code": "SA",  "league_name": "Serie A"},
    "ligue1":     {"league": "ligue1",     "league_code": "L1",  "league_name": "Ligue 1"},
}

def league_ctx(league_key: str = "pl") -> dict:
    """Returns template context dict for the given league."""
    return LEAGUE_CONTEXTS.get(league_key, LEAGUE_CONTEXTS["pl"])


# ── DC pre-game ───────────────────────────────────────────────────────────────

@app.get("/crest/{team_name}")
async def crest_proxy(team_name: str):
    """Proxy crest images to avoid CORS issues in SVG <image> tags."""
    url = get_crest_url(team_name)
    if not url:
        return _Response(status_code=404)
    try:
        req = _urllib_req.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _urllib_req.urlopen(req, timeout=5) as resp:
            data = resp.read()
        return _Response(content=data, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        return _Response(status_code=404)



@app.get("/teams", response_class=HTMLResponse)
async def teams_hub(request: Request):
    """Teams hub — England map with all PL clubs plotted."""
    team_colours = {
        name: get_theme_for_team(name)["primary"]
        for name in _CREST_IDS
    }
    return templates.TemplateResponse(
        request=request, name="teams.html",
        context={
            "request"      : request,
            "team_colours" : team_colours,
            **league_ctx("pl"),
        }
    )


@app.get("/teams/laliga", response_class=HTMLResponse)
async def teams_laliga(request: Request):
    """Spain map — all La Liga clubs with zoom tiers."""
    return templates.TemplateResponse(
        request=request, name="teams_laliga.html",
        context={
            "request": request,
            **league_ctx("laliga"),
        }
    )


@app.get("/team/{team_id}", response_class=HTMLResponse)
async def team_profile(request: Request, team_id: int, season: int = 2025):
    """Team profile page — squad, ratings, form, season results."""

    profile  = get_team_profile(team_id)
    if not profile:
        return HTMLResponse("<h1>Team not found</h1>", status_code=404)

    team_name = profile["name"]

    # Coach: API often returns null -- fall back to our dict
    coach = profile.get("coach") or TEAM_MANAGERS.get(team_name, "Unknown")

    # Theme
    theme       = get_theme_for_team(team_name)
    team_colour = theme["primary"]
    crest_url   = get_crest_proxy_url(team_name) or profile.get("crest", "")

    # Formation + squad
    formation = get_formation_for_team(team_name)
    players   = get_squad_for_team(team_name)

    # Vertical pitch SVG
    pitch_svg = generate_pitch_svg_vertical(
        formation=formation,
        team_color=team_colour,
        players=players,
        team_name=team_name,
    )

    # DC ratings
    dc_alpha = dc_beta = xg_proj = None
    dc_estimated = False
    if priors_db:
        league_data = priors_db.get(LEAGUE_KEY, {})
        teams_data  = league_data.get("teams", {})
        meta        = league_data.get("meta", {})
        all_alpha   = [v["alpha"] for v in teams_data.values()]
        all_beta    = [v["beta"]  for v in teams_data.values()]
        q25_alpha   = float(np.percentile(all_alpha, 25))
        q75_beta    = float(np.percentile(all_beta,  75))
        tkey        = TEAM_NAME_ALIASES.get(team_name, team_name)
        t_params    = teams_data.get(tkey)
        if t_params:
            dc_alpha = t_params["alpha"]
            dc_beta  = t_params["beta"]
            gamma    = get_effective_gamma(meta.get("gamma_home_advantage", 1.25))
            # xG projection = how many goals this team expects at home vs average defence
            avg_beta = float(np.mean(all_beta))
            xg_proj  = round(dc_alpha * avg_beta * gamma, 2)
        else:
            dc_alpha     = q25_alpha
            dc_beta      = q75_beta
            dc_estimated = True

    # Season results
    # Upcoming fixtures strip — filter to this team's games
    # FPL uses short names (e.g. "Man City", "Nott'm Forest") while
    # profile["name"] uses full names ("Manchester City", "Nottingham Forest")
    # Build a set of all name variants for this team to match against
    all_fixtures = get_upcoming_fixtures(max_fixtures=60)
    _fpl_short_map = {
        "Man City"      : "Manchester City",
        "Man United"    : "Manchester United",
        "Nott'm Forest" : "Nottingham Forest",
        "Spurs"         : "Tottenham",
        "Wolves"        : "Wolverhampton Wanderers",
        "Newcastle"     : "Newcastle United",
        "Leeds"         : "Leeds",
        "Leicester"     : "Leicester",
        "Brighton"      : "Brighton",
        "Brentford"     : "Brentford",
        "Bournemouth"   : "Bournemouth",
        "Fulham"        : "Fulham",
        "Everton"       : "Everton",
        "Burnley"       : "Burnley",
        "Sunderland"    : "Sunderland",
        "Ipswich"       : "Ipswich",
        "Coventry"      : "Coventry",
        "Hull"          : "Hull City",
    }
    # All FPL names that map to this team
    _this_team_fpl_names = {team_name} | {
        fpl for fpl, canonical in _fpl_short_map.items()
        if canonical == team_name
    }
    # Also add aliases from feature_builder
    _this_team_fpl_names |= {
        alias for alias, canonical in TEAM_NAME_ALIASES.items()
        if canonical == team_name
    }

    def _fixture_involves_team(f):
        home = f.get("home", "")
        away = f.get("away", "")
        return (
            home in _this_team_fpl_names or
            away in _this_team_fpl_names or
            # Fallback: partial match on first word
            team_name.split()[0].lower() in home.lower() or
            team_name.split()[0].lower() in away.lower()
        )

    team_fixtures = [f for f in all_fixtures if _fixture_involves_team(f)][:5]

    # Season data — use fdco CSVs for historical, football-data.org for recent
    # football-data.org free tier: current + ~2 recent seasons
    # fdco: free CSVs back to 1888, we offer 2014-present
    USE_FDCO = season < 2023

    if USE_FDCO:
        all_results = get_team_results_historical(team_name, season)
        standings   = get_pl_standings_historical(season)
        # Map fdco standings team_id using our _CREST_IDS
        for row in standings:
            row["team_id"] = _CREST_IDS.get(row["team_name"])
    else:
        all_results = get_team_season_results(team_id, season=season)
        standings   = get_standings(season=season)

    last_5       = get_last_n_results(all_results, n=5)
    next_fixture = get_next_fixture(all_results)

    ctx = {
        "request"        : request,
        "profile"        : profile,
        "coach"          : coach,
        "team_colour"    : team_colour,
        "crest_url"      : crest_url,
        "formation"      : formation,
        "pitch_svg"      : pitch_svg,
        "dc_alpha"       : dc_alpha,
        "dc_beta"        : dc_beta,
        "xg_proj"        : xg_proj,
        "dc_estimated"   : dc_estimated,
        "all_results"    : all_results,
        "last_5"         : last_5,
        "next_fixture"   : next_fixture,
        "standings"      : standings,
        "team_id"        : team_id,
        "current_season" : season,
        "team_fixtures"  : team_fixtures,
    }
    return templates.TemplateResponse(
        request=request, name="team.html", context=ctx
    )


@app.get("/admin/run-predictions")
async def admin_run_predictions():
    """Manually trigger the prediction job — useful for initial setup."""
    try:
        await run_prediction_job(priors_db, LEAGUE_KEY)
        stats = get_accuracy_stats()
        return JSONResponse({
            "status": "ok",
            "message": "Prediction job completed",
            "stats": stats,
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/admin/predictions-status")
async def admin_predictions_status():
    """Check what's in the predictions DB."""
    try:
        preds = get_all_predictions()
        stats = get_accuracy_stats()
        return JSONResponse({
            "total": len(preds),
            "stats": stats,
            "recent": [
                {
                    "match_id"   : p["match_id"],
                    "home"       : p["home_team"],
                    "away"       : p["away_team"],
                    "kickoff"    : p["kickoff_utc"],
                    "pre_logged" : bool(p["pre_logged_at"]),
                    "adj_logged" : bool(p["adj_logged_at"]),
                    "result"     : p["actual_result"],
                }
                for p in preds[:20]
            ]
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/admin/cache-status")
async def admin_cache_status():
    """Check FPL cache state — what's cached and how stale it is."""
    now = time.time()
    status = {}
    for key, (data, ts) in _cache.items():
        ttl     = _CACHE_TTLS.get(key, 3600)
        age_s   = int(now - ts)
        expires = max(0, ttl - age_s)
        size    = len(data) if isinstance(data, list) else len(str(data))
        status[key] = {
            "cached"         : True,
            "age_seconds"    : age_s,
            "expires_seconds": expires,
            "ttl_seconds"    : ttl,
            "size"           : size,
        }
    for key in _CACHE_TTLS:
        if key not in status:
            status[key] = {"cached": False}
    return JSONResponse({"cache": status})


@app.get("/player", response_class=HTMLResponse)
async def player_page(request: Request):
    """Player profile — powered by TheSportsDB + DC priors."""
    name    = request.query_params.get("name", "").strip()
    team    = request.query_params.get("team", "").strip()
    league  = request.query_params.get("league", "pl").lower()

    # Resolve team_id for crest proxy
    team_id = request.query_params.get("team_id", "")
    if not team_id and team:
        team_id = str(_CREST_IDS.get(team, ""))

    # ── TheSportsDB: player profile + career history ───────────────────────
    tsdb_player  = None
    career       = []
    try:
        tsdb_player = search_player(name)
        if tsdb_player and tsdb_player.get("id"):
            career = get_career_history(tsdb_player["id"])
    except Exception as e:
        print(f"[player] TheSportsDB error: {e}")

    # ── DC priors for this player's team ──────────────────────────────────
    priors_key = "ESP-La Liga" if league == "laliga" else "ENG-Premier League"
    team_priors = {}
    try:
        league_data = priors_db.get(priors_key, {})
        teams_data  = league_data.get("teams", {})
        meta        = league_data.get("meta", {})
        # Try exact match then fuzzy
        team_key = TEAM_NAME_ALIASES.get(team, team)
        team_priors = teams_data.get(team_key, {})
        if not team_priors:
            team_lower = team.lower()
            for k, v in teams_data.items():
                if k.lower() in team_lower or team_lower in k.lower():
                    team_priors = v
                    break
    except Exception as e:
        print(f"[player] priors lookup error: {e}")

    alpha   = team_priors.get("alpha")
    beta    = team_priors.get("beta")
    gamma   = meta.get("gamma_home_advantage", 1.25) if 'meta' in dir() else 1.25
    try:
        gamma = priors_db.get(priors_key, {}).get("meta", {}).get("gamma_home_advantage", 1.25)
    except Exception:
        gamma = 1.25

    # α bar width: scale 0-2 range to 0-100%
    alpha_pct = round(min(100, (alpha / 2.0) * 100)) if alpha else 0
    beta_pct  = round(min(100, (beta  / 2.0) * 100)) if beta  else 0

    # ── Next match for this team from predictions DB ───────────────────────
    next_match = None
    try:
        now_str = datetime.now(timezone.utc).isoformat()
        season  = datetime.now(timezone.utc).year
        if datetime.now(timezone.utc).month < 8:
            season -= 1
        preds = get_all_predictions(season)
        upcoming = [
            p for p in preds
            if not p.get("actual_result")
            and p.get("kickoff_utc", "") > now_str
            and p.get("league", "pl") == league
            and (team.lower() in p.get("home_team","").lower()
                 or team.lower() in p.get("away_team","").lower())
        ]
        if upcoming:
            upcoming.sort(key=lambda x: x.get("kickoff_utc",""))
            p = upcoming[0]
            # Convert kickoff to EAT display
            ko_display = ""
            try:
                ko_dt = datetime.fromisoformat(p["kickoff_utc"].replace("Z","+00:00"))
                eat   = ko_dt.astimezone(EAT)
                ko_display = eat.strftime("%a %d %b · %H:%M EAT")
            except Exception:
                ko_display = p.get("kickoff_utc","")[:10]
            next_match = {
                "home"       : p.get("home_team",""),
                "away"       : p.get("away_team",""),
                "kickoff_eat": ko_display,
                "match_id"   : p.get("match_id",""),
                "dc_home"    : p.get("pre_dc_home"),
                "dc_draw"    : p.get("pre_dc_draw"),
                "dc_away"    : p.get("pre_dc_away"),
                "league"     : league,
            }
    except Exception as e:
        print(f"[player] next match error: {e}")

    # ── Kit for this team ──────────────────────────────────────────────────
    try:
        kit = _KITS_PUBLIC.get(team, {"fill": "#14b8a6", "stroke": "#fff", "abbr": team[:3].upper()})
    except Exception:
        kit = {"fill": "#14b8a6", "stroke": "#fff", "abbr": team[:3].upper() if team else "---"}

    league_display = "La Liga" if league == "laliga" else "Premier League"

    return templates.TemplateResponse(
        request=request, name="player.html",
        context={
            "request"       : request,
            "player_name"   : tsdb_player["name"] if tsdb_player else name,
            "team_name"     : team,
            "team_id"       : team_id,
            "league"        : league,
            "league_display": league_display,
            "tsdb"          : tsdb_player,
            "career"        : career,
            "alpha"         : round(alpha, 3) if alpha else None,
            "beta"          : round(beta,  3) if beta  else None,
            "gamma"         : round(gamma, 3),
            "alpha_pct"     : alpha_pct,
            "beta_pct"      : beta_pct,
            "kit"           : kit,
            "next_match"    : next_match,
            **league_ctx(league),
        }
    )


@app.get("/tournament", response_class=HTMLResponse)
async def tournament_page(request: Request):
    """Tournament simulator — CL knockout + PL season Monte Carlo."""

    # Build team list per league from priors
    all_teams: dict[str, list[dict]] = {}
    for league_key, league_name in [
        ("ENG-Premier League", "pl"),
        ("ESP-La Liga",        "laliga"),
        ("GER-Bundesliga",     "bundesliga"),
        ("ITA-Serie A",        "seriea"),
        ("FRA-Ligue 1",        "ligue1"),
    ]:
        data  = priors_db.get(league_key, {})
        teams = data.get("teams", {})
        ranked = sorted(teams.items(), key=lambda x: x[1]["alpha"], reverse=True)
        all_teams[league_name] = [
            {
                "name"  : t,
                "alpha" : round(v["alpha"], 3),
                "abbr"  : _KITS_PUBLIC.get(t, {}).get("abbr", t[:3].upper()),
                "fill"  : _KITS_PUBLIC.get(t, {}).get("fill", "#14b8a6"),
                "stroke": _KITS_PUBLIC.get(t, {}).get("stroke", "#fff"),
            }
            for t, v in ranked
        ]

    return templates.TemplateResponse(
        request=request, name="tournament.html",
        context={
            "request"  : request,
            "all_teams": json.dumps(all_teams),
            **league_ctx("pl"),
        }
    )



@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """Prediction history — model accuracy tracker, filterable by league."""

    # League filter from query param: ?league=pl / ?league=laliga / (all)
    league_filter = request.query_params.get("league", "all").lower()

    try:
        season   = datetime.now(timezone.utc).year
        if datetime.now(timezone.utc).month < 8:
            season -= 1

        all_pred = get_all_predictions(season)

        # Apply league filter
        if league_filter == "pl":
            all_pred = [p for p in all_pred if p.get("league","pl") == "pl"]
        elif league_filter == "laliga":
            all_pred = [p for p in all_pred if p.get("league","pl") == "laliga"]

        finished = [p for p in all_pred if p["actual_result"]]
        upcoming = [p for p in all_pred if not p["actual_result"]]

        # Per-league counts for tabs
        all_preds_all = get_all_predictions(season)
        pl_count  = sum(1 for p in all_preds_all if p.get("league","pl") == "pl")
        ll_count  = sum(1 for p in all_preds_all if p.get("league","pl") == "laliga")

        # Accuracy stats
        def _stats(preds):
            fin = [p for p in preds if p["actual_result"]]
            if not fin:
                return {"total_logged": len(preds), "total_finished": 0,
                        "pre_accuracy": 0, "pre_correct": 0, "pre_total": 0,
                        "adj_accuracy": 0, "adj_correct": 0, "adj_total": 0}
            pre_total   = len(fin)
            pre_correct = sum(1 for p in fin if p.get("pre_correct") == 1)
            adj_set     = [p for p in fin if p.get("adj_logged_at")]
            adj_correct = sum(1 for p in adj_set if p.get("adj_correct") == 1)
            return {
                "total_logged"  : len(preds),
                "total_finished": pre_total,
                "pre_accuracy"  : round(pre_correct/pre_total*100,1) if pre_total else 0,
                "pre_correct"   : pre_correct,
                "pre_total"     : pre_total,
                "adj_accuracy"  : round(adj_correct/len(adj_set)*100,1) if adj_set else 0,
                "adj_correct"   : adj_correct,
                "adj_total"     : len(adj_set),
            }

        stats = _stats(all_pred)

        # Per-league stats for tab badges
        pl_preds  = [p for p in all_preds_all if p.get("league","pl") == "pl"]
        ll_preds  = [p for p in all_preds_all if p.get("league","pl") == "laliga"]
        pl_stats  = _stats(pl_preds)
        ll_stats  = _stats(ll_preds)

        # Confidence band breakdown
        bands = {
            "High (>60%)"  : [],
            "Mid (40-60%)" : [],
            "Low (<40%)"   : [],
        }
        for p in finished:
            if p["pre_dc_home"] is None:
                continue
            conf = max(p["pre_dc_home"], p["pre_dc_draw"], p["pre_dc_away"])
            if conf > 60:   bands["High (>60%)"].append(p)
            elif conf >= 40: bands["Mid (40-60%)"].append(p)
            else:            bands["Low (<40%)"].append(p)

        band_stats = {}
        for label, games in bands.items():
            if not games:
                band_stats[label] = {"n": 0, "correct": 0, "pct": 0}
                continue
            correct = sum(1 for g in games if g["pre_correct"])
            band_stats[label] = {
                "n"      : len(games),
                "correct": correct,
                "pct"    : round(correct/len(games)*100, 1),
            }

        draws_actual    = sum(1 for p in finished if p["actual_result"] == "D")
        draws_predicted = sum(
            1 for p in finished
            if p["pre_dc_draw"] and
            p["pre_dc_draw"] > (p["pre_dc_home"] or 0) and
            p["pre_dc_draw"] > (p["pre_dc_away"] or 0)
        )

    except Exception as e:
        print(f"[history] error: {e}")
        stats = pl_stats = ll_stats = None
        finished = upcoming = []
        band_stats = {}
        draws_actual = draws_predicted = 0
        pl_count = ll_count = 0
        league_filter = "all"

    return templates.TemplateResponse(
        request=request, name="history.html",
        context={
            "request"         : request,
            "stats"           : stats,
            "pl_stats"        : pl_stats,
            "ll_stats"        : ll_stats,
            "pl_count"        : pl_count,
            "ll_count"        : ll_count,
            "league_filter"   : league_filter,
            "finished"        : finished,
            "upcoming"        : upcoming[:10],
            "band_stats"      : band_stats,
            "draws_actual"    : draws_actual,
            "draws_predicted" : draws_predicted,
            "season"          : season,
            **league_ctx("pl"),
        }
    )


@app.get("/", response_class=HTMLResponse)
async def hub(request: Request):

    now_eat = datetime.now(timezone(timedelta(hours=3)))
    today   = now_eat.date().isoformat()

    # PL stats from predictions DB
    try:
        pl_stats = get_accuracy_stats(season=2025)
    except Exception:
        pl_stats = None

    # Upcoming PL fixtures for today strip fallback
    try:
        upcoming = fpl_get_upcoming_fixtures(max_fixtures=10)
    except Exception:
        upcoming = []

    # Today's matches across all leagues (PL only for now — expand as leagues added)
    today_matches = []
    try:
        for f in upcoming:
            ko = f.get("kickoff_utc", "")
            if ko and ko[:10] == today:
                # Compute DC odds
                dc = dc_pregame(f["home"], f["away"], "ENG-Premier League")
                today_matches.append({
                    "match_id"     : f["match_id"],
                    "home"         : f["home"],
                    "away"         : f["away"],
                    "kickoff_eat"  : f.get("kickoff_eat", ""),
                    "league_name"  : "Premier League",
                    "league_colour": "#9B59B6",
                    "is_live"      : False,
                    "minute"       : None,
                    "dc_home"      : dc[0] if dc else None,
                    "dc_draw"      : dc[1] if dc else None,
                    "dc_away"      : dc[2] if dc else None,
                })
    except Exception:
        pass

    return templates.TemplateResponse(
        request=request, name="index.html",
        context={
            "request"         : request,
            "league"          : "pl",
            "league_code"     : "PL",
            "pl_stats"        : pl_stats,
            "today_matches"   : today_matches,
            "upcoming_fixtures": upcoming,
            "live_count"      : 0,
            "live_count_pl"   : 0,
        }
    )


@app.get("/league/{league_key}", response_class=HTMLResponse)
async def league_page(request: Request, league_key: str):
    """
    League dashboard router.
    PL      → redirects to /match (fully built).
    La Liga → live dashboard with DC odds and upcoming fixtures.
    Others  → under-construction page.
    """

    if league_key not in LEAGUE_CONTEXTS:
        return RedirectResponse(url="/")

    if league_key == "pl":
        return RedirectResponse(url="/match")

    if league_key == "laliga":
        return RedirectResponse(url="/match?league=laliga")

    # All other leagues — under construction
    LEAGUE_FEATURES = {
        "bundesliga": [
            ("DC prior odds (18 teams)",          90),
            ("Live match scores via FotMob",      60),
            ("Germany team map",                  15),
            ("Lineup-adjusted predictions",       10),
            ("Prediction history logging",        10),
            ("Neural net live model",             10),
        ],
        "seriea": [
            ("DC prior odds (20 teams)",          90),
            ("Live match scores via FotMob",      60),
            ("Italy team map",                    10),
            ("Lineup-adjusted predictions",       10),
            ("Prediction history logging",        10),
            ("Neural net live model",             10),
        ],
        "ligue1": [
            ("DC prior odds (18 teams)",          90),
            ("Live match scores via FotMob",      60),
            ("France team map",                   10),
            ("Lineup-adjusted predictions",       10),
            ("Prediction history logging",        10),
            ("Neural net live model",             10),
        ],
    }
    ctx = LEAGUE_CONTEXTS[league_key]
    return templates.TemplateResponse(
        request=request, name="under_construction.html",
        context={
            "request"    : request,
            "league"     : league_key,
            "league_key" : league_key,
            "league_code": ctx["league_code"],
            "league_name": ctx["league_name"],
            "features"   : LEAGUE_FEATURES.get(league_key, []),
        }
    )


async def laliga_dashboard(request: Request):
    """
    La Liga dashboard — upcoming fixtures with DC odds,
    today's matches, accuracy stats, team map link.
    Fixture source: football-data.org /competitions/PD/matches
    DC odds: ESP-La Liga priors from v4_priors.json
    """

    LL_KEY = "ESP-La Liga"
    EAT    = timezone(timedelta(hours=3))
    today  = datetime.now(EAT).date().isoformat()

    # Upcoming La Liga fixtures
    upcoming_raw = []
    try:
        upcoming_raw = get_upcoming_fixtures_fd(PD_CODE, season=2025, max_fixtures=20)
    except Exception as e:
        print(f"[laliga] fixture fetch error: {e}")

    # Enrich with DC odds
    upcoming = []
    for f in upcoming_raw:
        dc = dc_pregame(f["home"], f["away"], LL_KEY)
        upcoming.append({
            **f,
            "dc_home": dc[0] if dc else None,
            "dc_draw": dc[1] if dc else None,
            "dc_away": dc[2] if dc else None,
            "pred"   : ("H" if dc and dc[0]>dc[1] and dc[0]>dc[2]
                        else "D" if dc and dc[1]>dc[2] else "A") if dc else None,
        })

    # Split today vs future
    today_matches = [f for f in upcoming if f.get("kickoff_utc","")[:10] == today]
    future        = [f for f in upcoming if f.get("kickoff_utc","")[:10] > today]

    # La Liga accuracy from predictions DB (season=2025, league tagged laliga)
    ll_stats = None
    try:
        ll_stats = get_accuracy_stats(season=2025)
    except Exception:
        pass

    return templates.TemplateResponse(
        request=request, name="laliga_dashboard.html",
        context={
            "request"      : request,
            **league_ctx("laliga"),
            "today_matches": today_matches,
            "upcoming"     : future[:15],
            "ll_stats"     : ll_stats,
            "today"        : today,
        }
    )

