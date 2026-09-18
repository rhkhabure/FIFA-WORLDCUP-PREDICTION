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

import json
import pickle
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from scipy.stats import poisson

import uvicorn

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

from footballdata import (get_live_match_data, get_last_completed_pl_match,
                          get_finished_match, find_finished_match_by_teams)
from utils import (generate_pitch_svg_horizontal, get_theme_for_team,
                   get_formation_for_team, get_squad_for_team,
                   get_crest_url, get_crest_proxy_url, _CREST_IDS)
from timeline import build_match_timeline_svg
from scoreline_matrix import build_scoreline_svg
from fpl import get_upcoming_fixtures
from fotmob import (get_lineup, get_live_xg, get_fotmob_match_id,
                    get_match_status_from_date_cache, fotmob_to_fpl_team_name,
                    has_key as fotmob_available)
from lineup_adjustment import compute_lineup_adjusted_odds, get_absent_key_players
from teamdata import get_team_profile, get_team_season_results, get_next_fixture, get_last_n_results, get_standings
from football_co_uk import get_team_results_historical, get_pl_standings_historical
from v4_backend.feature_builder import DCStrengthLookup, TEAM_NAME_ALIASES

# ── Constants ─────────────────────────────────────────────────────────────────
DRAW_PROPENSITY = 0.10
LEAGUE_KEY      = "ENG-Premier League"

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
    import asyncio, concurrent.futures
    from fpl import warm_cache as fpl_warm
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
    yield
    # Shutdown: nothing to clean up (SQLite handles its own flush)

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Melios Dira Odds", lifespan=lifespan)
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
def get_effective_gamma(gamma_calibrated: float, season_start_month: int = 8,
                        season_start_day: int = 21) -> float:
    """
    Decay home advantage with a 60% floor at season start, fully restored by GW6.

    Empirically validated on 2025/26 PL GW1-GW4 (30 games):
      - 60% floor: preserves GW1 home predictions (80% acc, unchanged from flat)
      - Moderate decay: correctly flips borderline GW2 predictions (+1 correct)
      - Full restore by GW6: mid-season predictions fully trust calibrated priors
      - Net result: +1 correct pick overall, log-loss improves at GW4

    Formula: gamma_eff = 1.0 + (gamma_cal - 1.0) * decay
    decay = 0.60 + 0.40 * min(gw / 6, 1.0)
    GW1: decay=0.67 → gamma≈1.157  (67% of full advantage)
    GW3: decay=0.80 → gamma≈1.189
    GW6: decay=1.00 → gamma=calibrated
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    year = now.year if now.month >= season_start_month else now.year - 1
    season_start = datetime(year, season_start_month, season_start_day,
                            tzinfo=timezone.utc)

    days_elapsed = max(0, (now - season_start).days)
    gw_approx    = max(1, days_elapsed // 7 + 1)

    FLOOR   = 0.60
    GW_FULL = 6
    decay   = FLOOR + (1.0 - FLOOR) * min(gw_approx / GW_FULL, 1.0)

    return round(1.0 + (gamma_calibrated - 1.0) * decay, 4)


def dc_pregame(home_team: str, away_team: str, league: str) -> list | None:
    """
    Dixon-Coles bivariate Poisson with draw_propensity correction.
    Returns [p_home%, p_draw%, p_away%] rounded to 1 dp, or None.
    Uses bottom-quartile fallback for teams not in priors (promoted clubs etc.)
    Home advantage decays toward 1.0 at season start, recovering over 10 GWs.
    """
    if not priors_db:
        return None
    league_data = priors_db.get(league)
    if not league_data:
        return None
    teams  = league_data["teams"]
    meta   = league_data["meta"]

    home_key = TEAM_NAME_ALIASES.get(home_team, home_team)
    away_key = TEAM_NAME_ALIASES.get(away_team, away_team)

    all_alpha = [v["alpha"] for v in teams.values()]
    all_beta  = [v["beta"]  for v in teams.values()]
    q25_alpha = float(np.percentile(all_alpha, 25))
    q75_beta  = float(np.percentile(all_beta,  75))

    h = teams.get(home_key, {"alpha": q25_alpha, "beta": q75_beta})
    a = teams.get(away_key, {"alpha": q25_alpha, "beta": q75_beta})

    gamma_cal = meta.get("gamma_home_advantage", 1.25)
    gamma     = get_effective_gamma(gamma_cal)   # decayed for early season
    rho       = meta.get("rho_draw_correction",  0.0)

    lam   = np.clip(h["alpha"] * a["beta"] * gamma, 1e-5, 15.0)
    mu    = np.clip(a["alpha"] * h["beta"],          1e-5, 15.0)
    hp    = poisson.pmf(np.arange(9), lam)
    ap    = poisson.pmf(np.arange(9), mu)
    joint = np.outer(hp, ap)
    joint[0,0] *= max(1.0 - lam*mu*rho, 1e-5)
    joint[1,0] *= max(1.0 + mu*rho,     1e-5)
    joint[0,1] *= max(1.0 + lam*rho,    1e-5)
    joint[1,1] *= max(1.0 - rho,        1e-5)
    joint /= joint.sum()
    ph  = float(np.tril(joint,-1).sum())
    pd_ = float(np.trace(joint))
    pa  = float(np.triu(joint,+1).sum())
    tr  = DRAW_PROPENSITY / 2.0
    ph2 = max(ph - tr, 0.0)
    pa2 = max(pa - tr, 0.0)
    pd2 = pd_ + DRAW_PROPENSITY
    tot = ph2 + pd2 + pa2
    return [round(ph2/tot*100,1), round(pd2/tot*100,1), round(pa2/tot*100,1)]


# ── Neural net live ───────────────────────────────────────────────────────────
def nn_live(home_team, away_team, league, minute, home_score, away_score):
    if nn_model is None or nn_scaler is None or dc_lookup is None:
        return None
    goals_so_far  = home_score + away_score
    lead_changes  = 1 if (home_score > 0 and away_score > 0) else 0
    try:
        feat = dc_lookup.build_feature_row(
            home_team=home_team, away_team=away_team, league=league,
            minute=minute, home_score=home_score, away_score=away_score,
            lead_changes=lead_changes, goals_so_far=goals_so_far,
            is_knockout=0, is_neutral_venue=0,
        )
    except Exception as e:
        print(f"Feature build failed: {e}")
        return None
    X = nn_scaler.transform(np.array([feat], dtype="float32")).astype("float32")
    with torch.no_grad():
        logits = nn_model(torch.tensor(X)).numpy()[0]
    z = logits / nn_T; z -= z.max()
    p = np.exp(z); p /= p.sum()
    return [round(float(p[2])*100,1), round(float(p[1])*100,1), round(float(p[0])*100,1)]


# ── Routes ────────────────────────────────────────────────────────────────────
import urllib.request as _urllib_req
from fastapi.responses import Response as _Response, JSONResponse

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


@app.get("/live/{match_id}")
async def live_poll(match_id: str, request: Request):
    """Live polling — returns score, minute from BBS (La Liga) or FotMob (PL)."""

    # BBS live poll for La Liga matches
    if str(match_id).startswith("bbs_"):
        try:
            from bbs import get_match_detail
            bbs_uuid = str(match_id)[4:]
            detail   = get_match_detail(bbs_uuid)
            if not detail:
                return JSONResponse({"status": "no_data"})
            raw_st = detail.get("status", "Not Started")
            st_map = {"live": "In Play", "finished": "Finished",
                      "final": "Finished", "scheduled": "Not Started",
                      "In Play": "In Play", "Finished": "Finished",
                      "Not Started": "Not Started"}
            match_status = st_map.get(raw_st, raw_st)
            # Estimate minute from kickoff
            live_minute = None
            if match_status == "In Play":
                try:
                    from datetime import datetime, timezone as _tz
                    ko_raw = detail.get("kickoff_utc", "")
                    if ko_raw:
                        ko_dt   = datetime.fromisoformat(ko_raw.replace("Z","+00:00"))
                        elapsed = (datetime.now(_tz.utc) - ko_dt).total_seconds() / 60
                        if elapsed > 60:
                            elapsed -= 15
                        live_minute = max(1, min(90, int(elapsed)))
                    else:
                        live_minute = 50
                except Exception:
                    live_minute = 50
            return JSONResponse({
                "status"      : "ok",
                "match_status": match_status,
                "home_score"  : detail.get("h_score", 0),
                "away_score"  : detail.get("a_score", 0),
                "live_minute" : live_minute,
                "minute_str"  : str(live_minute) if live_minute else "",
            })
        except Exception as e:
            print(f"[live] BBS poll error: {e}")
            return JSONResponse({"status": "no_data"})

    if not fotmob_available():
        return _Response(status_code=204)

    fotmob_id = request.query_params.get("fotmob_id") or match_id

    # Status/score/minute from date cache (free)
    fm = get_match_status_from_date_cache(fotmob_id)

    # xG from separate endpoint (1 credit, cached 5 min)
    xg = get_live_xg(fotmob_id, max_age_seconds=300)

    if not fm and not xg:
        return JSONResponse({"status": "no_data"})

    payload: dict = {"status": "ok"}

    if fm:
        if fm["finished"]:
            match_status = "Finished"
        elif fm["is_halftime"]:
            match_status = "Half Time"
        elif fm["ongoing"]:
            match_status = "In Play"
        else:
            match_status = "Not Started"

        payload.update({
            "match_status" : match_status,
            "live_minute"  : fm["minute_int"] if not fm["finished"] and not fm["is_halftime"] else None,
            "minute_str"   : fm["minute_str"],
            "home_score"   : fm["home_score"],
            "away_score"   : fm["away_score"],
        })

    if xg:
        payload.update({
            "home_xg"    : xg["home_xg"],
            "away_xg"    : xg["away_xg"],
            "home_xg_h1" : xg["home_xg_h1"],
            "away_xg_h1" : xg["away_xg_h1"],
            "home_xg_h2" : xg["home_xg_h2"],
            "away_xg_h2" : xg["away_xg_h2"],
        })

    return JSONResponse(payload)


@app.get("/teams", response_class=HTMLResponse)
async def teams_hub(request: Request):
    """Teams hub — England map with all PL clubs plotted."""
    from utils import get_theme_for_team, _CREST_IDS
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
    from utils import (generate_pitch_svg_vertical, get_theme_for_team,
                       get_formation_for_team, get_squad_for_team,
                       get_crest_proxy_url, TEAM_MANAGERS)
    from v4_backend.feature_builder import TEAM_NAME_ALIASES

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
        from utils import _CREST_IDS as _CID
        for row in standings:
            row["team_id"] = _CID.get(row["team_name"])
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
    from prediction_job import run_prediction_job
    from predictions import get_accuracy_stats
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
    from predictions import get_all_predictions, get_accuracy_stats
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
    from fpl import _cache, _CACHE_TTLS
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
        from utils import _CREST_IDS
        team_id = str(_CREST_IDS.get(team, ""))

    # ── TheSportsDB: player profile + career history ───────────────────────
    from thesportsdb import search_player, get_career_history
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
        from predictions import get_all_predictions
        from datetime import datetime, timezone as _tz
        now_str = datetime.now(_tz.utc).isoformat()
        season  = datetime.now(_tz.utc).year
        if datetime.now(_tz.utc).month < 8:
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
                from datetime import timedelta
                ko_dt = datetime.fromisoformat(p["kickoff_utc"].replace("Z","+00:00"))
                eat   = ko_dt.astimezone(_tz(timedelta(hours=3)))
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
        from utils import _KITS_PUBLIC
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
    import json

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
        from utils import _KITS_PUBLIC
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


@app.post("/api/simulate/cl")
async def api_simulate_cl(request: Request):
    """
    Run CL knockout Monte Carlo simulation.
    Body: {"teams": ["Liverpool", "Barcelona", ...], "league_map": {"Liverpool": "ENG-Premier League", ...}}
    Returns: win_pct, sf_pct, qf_pct, most_likely bracket
    """
    from simulate import simulate_cl_tournament
    body = await request.json()
    teams     = body.get("teams", [])
    league_map= body.get("league_map", {})

    if len(teams) != 16:
        return JSONResponse({"error": f"Need 16 teams, got {len(teams)}"}, status_code=400)

    try:
        result = simulate_cl_tournament(
            teams=teams,
            league_map=league_map,
            priors_db=priors_db,
            n_runs=10_000,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/simulate/pl")
async def api_simulate_pl(request: Request):
    """
    Run PL season Monte Carlo simulation.
    Fetches current standings from fd.org, remaining fixtures from FPL.
    Returns: predicted table with title/top4/relegation probabilities.
    """
    from simulate import simulate_pl_season

    # Get current standings
    current_table = []
    try:
        import ssl, json as _json, urllib.request as _req
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        API_KEY = os.getenv("FOOTBALLDATA_ORG_KEY", "")
        r = _req.Request(
            "https://api.football-data.org/v4/competitions/PL/standings",
            headers={"X-Auth-Token": API_KEY}
        )
        with _req.urlopen(r, timeout=8, context=ctx) as resp:
            data = _json.loads(resp.read())
        for row in data.get("standings", [{}])[0].get("table", []):
            current_table.append({
                "team"  : row["team"]["name"],
                "played": row["playedGames"],
                "won"   : row["won"],
                "drawn" : row["draw"],
                "lost"  : row["lost"],
                "gf"    : row["goalsFor"],
                "ga"    : row["goalsAgainst"],
                "points": row["points"],
            })
    except Exception as e:
        print(f"[simulate/pl] standings error: {e}")
        # Fallback: build from fd.org finished cache
        from footballdata import _populate_finished_cache, PL_CODE
        from collections import defaultdict
        cache = _populate_finished_cache(PL_CODE)
        tbl   = defaultdict(lambda: {"played":0,"won":0,"drawn":0,"lost":0,"gf":0,"ga":0,"points":0})
        for m in cache.values():
            home, away = m.get("home_team",""), m.get("away_team","")
            hg, ag = m.get("h_score"), m.get("a_score")
            if not home or not away or hg is None: continue
            tbl[home]["played"]+=1; tbl[away]["played"]+=1
            tbl[home]["gf"]+=hg;   tbl[home]["ga"]+=ag
            tbl[away]["gf"]+=ag;   tbl[away]["ga"]+=hg
            if hg>ag:  tbl[home]["won"]+=1;   tbl[home]["points"]+=3; tbl[away]["lost"]+=1
            elif hg==ag: tbl[home]["drawn"]+=1; tbl[home]["points"]+=1; tbl[away]["drawn"]+=1; tbl[away]["points"]+=1
            else:      tbl[away]["won"]+=1;   tbl[away]["points"]+=3; tbl[home]["lost"]+=1
        current_table = [{"team":t,**v} for t,v in tbl.items()]

    # Get remaining fixtures from FPL
    remaining = []
    try:
        fpl_fixtures = get_upcoming_fixtures(max_fixtures=500)
        fpl_bootstrap= _get_bootstrap()
        fpl_teams    = {t["id"]: t["name"] for t in fpl_bootstrap.get("teams",[])}
        for f in fpl_fixtures:
            remaining.append({
                "home": f.get("home", ""),
                "away": f.get("away", ""),
            })
    except Exception as e:
        print(f"[simulate/pl] fixtures error: {e}")

    if not current_table:
        return JSONResponse({"error": "Could not load standings"}, status_code=503)

    try:
        result = simulate_pl_season(
            current_table=current_table,
            remaining_fixtures=remaining,
            priors_db=priors_db,
            n_runs=10_000,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    """Prediction history — model accuracy tracker, filterable by league."""
    from predictions import get_accuracy_stats, get_all_predictions
    from datetime import datetime, timezone

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
    from predictions import get_accuracy_stats, get_all_predictions
    from datetime import datetime, timezone, timedelta
    from fpl import get_upcoming_fixtures as fpl_upcoming

    now_eat = datetime.now(timezone(timedelta(hours=3)))
    today   = now_eat.date().isoformat()

    # PL stats from predictions DB
    try:
        pl_stats = get_accuracy_stats(season=2025)
    except Exception:
        pl_stats = None

    # Upcoming PL fixtures for today strip fallback
    try:
        upcoming = fpl_upcoming(max_fixtures=10)
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
    from fastapi.responses import RedirectResponse

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
    from footballdata import get_upcoming_fixtures_fd, PD_CODE
    from datetime import datetime, timezone, timedelta

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
        from predictions import get_accuracy_stats
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


@app.get("/match", response_class=HTMLResponse)
async def match(request: Request):
    match_id = request.query_params.get("match_id")

    # ── League selection ────────────────────────────────────────────
    # URL: /match?match_id=X&league=laliga
    # Defaults to Premier League for all existing links.
    _league_param = request.query_params.get("league", "pl").lower()

    # Map URL param → priors key + competition code
    _LEAGUE_MAP = {
        "pl":         {"priors": "ENG-Premier League", "comp": "PL",
                       "name": "Premier League",   "ctx_key": "pl"},
        "laliga":     {"priors": "ESP-La Liga",        "comp": "PD",
                       "name": "La Liga",           "ctx_key": "laliga"},
        "bundesliga": {"priors": "GER-Bundesliga",     "comp": "BL1",
                       "name": "Bundesliga",        "ctx_key": "bundesliga"},
        "seriea":     {"priors": "ITA-Serie A",        "comp": "SA",
                       "name": "Serie A",           "ctx_key": "seriea"},
        "ligue1":     {"priors": "FRA-Ligue 1",        "comp": "FL1",
                       "name": "Ligue 1",           "ctx_key": "ligue1"},
    }
    _lm = _LEAGUE_MAP.get(_league_param, _LEAGUE_MAP["pl"])
    active_league  = _lm["priors"]   # e.g. "ESP-La Liga"
    active_comp    = _lm["comp"]     # e.g. "PD"
    active_name    = _lm["name"]     # e.g. "La Liga"
    active_ctx_key = _lm["ctx_key"]  # e.g. "laliga"

    if not match_id:
        # Default: redirect to the most recent/current match for this league
        from footballdata import _populate_finished_cache
        from datetime import datetime, timezone, timedelta

        # For La Liga and other non-PL leagues: BBS API first, FotMob fallback
        if active_ctx_key != "pl":
            # Try BBS (Big Balls Sports Data) — primary La Liga live source
            try:
                from bbs import get_today_matches, has_key as bbs_ok
                if bbs_ok():
                    today_matches = get_today_matches(active_ctx_key)
                    if today_matches:
                        # Prefer live match, then most recent kickoff
                        live = [m for m in today_matches if m.get("is_live")]
                        target = live[0] if live else today_matches[0]
                        bbs_id = target.get("bbs_id", "")
                        if bbs_id:
                            return RedirectResponse(
                                url=f"/match?match_id=bbs_{bbs_id}&league={active_ctx_key}",
                                status_code=302
                            )
            except Exception as e:
                print(f"[match] BBS redirect error: {e}")

            # FotMob fallback
            if fotmob_available():
                try:
                    from fotmob import _lineup_cache, _refresh_date_cache_if_stale
                    from datetime import datetime, timezone, timedelta
                    today_str = datetime.now(
                        timezone(timedelta(hours=3))
                    ).strftime("%Y%m%d")
                    _refresh_date_cache_if_stale(today_str)
                    cache_key = f"matches_{today_str}"
                    date_data = _lineup_cache.get(cache_key, {})
                    _LEAGUE_FILTERS = {
                        "laliga"    : ["laliga", "la liga", "primera", "esp"],
                        "bundesliga": ["bundesliga", "germany"],
                        "seriea"    : ["serie a", "italy"],
                        "ligue1"    : ["ligue 1", "france"],
                    }
                    filters = _LEAGUE_FILTERS.get(active_ctx_key, [])
                    for lg in date_data.get("data", {}).get("leagues", []):
                        if any(f in lg.get("name","").lower() for f in filters):
                            matches = lg.get("matches", [])
                            if matches:
                                first_id = matches[0].get("id")
                                if first_id:
                                    return RedirectResponse(
                                        url=f"/match?match_id=fm_{first_id}&league={active_ctx_key}",
                                        status_code=302
                                    )
                            break
                except Exception as e:
                    print(f"[match] FotMob redirect error: {e}")

        # FotMob unavailable or 429 — use fd.org finished cache for any league
        # This gives us real team names and DC odds even without live data
        if active_ctx_key != "pl":
            try:
                fd_cache = _populate_finished_cache(active_comp)
                if fd_cache:
                    last    = list(fd_cache.values())[-1]
                    last_id = last.get("fd_id")
                    if last_id:
                        return RedirectResponse(
                            url=f"/match?match_id={last_id}&league={active_ctx_key}",
                            status_code=302
                        )
            except Exception as e:
                print(f"[match] fd.org fallback redirect error: {e}")

        # PL: try FPL upcoming first, then FotMob today as fallback
        if active_ctx_key == "pl":
            # FPL upcoming (most reliable — gives next GW fixtures)
            try:
                upcoming = get_upcoming_fixtures(max_fixtures=1)
                if upcoming:
                    found_id = upcoming[0].get("match_id")
                    if found_id:
                        return RedirectResponse(
                            url=f"/match?match_id={found_id}",
                            status_code=302
                        )
            except Exception:
                pass

            # FPL failed — try FotMob today for any PL match
            if fotmob_available():
                try:
                    from fotmob import _lineup_cache, _refresh_date_cache_if_stale
                    from datetime import datetime, timezone, timedelta
                    today_str = datetime.now(
                        timezone(timedelta(hours=3))
                    ).strftime("%Y%m%d")
                    _refresh_date_cache_if_stale(today_str)
                    date_data = _lineup_cache.get(f"matches_{today_str}", {})
                    for lg in date_data.get("data", {}).get("leagues", []):
                        lg_name = lg.get("name", "").lower()
                        if "premier league" in lg_name or "england" in lg_name:
                            matches = lg.get("matches", [])
                            if matches:
                                first_id = matches[0].get("id")
                                if first_id:
                                    return RedirectResponse(
                                        url=f"/match?match_id=fm_{first_id}",
                                        status_code=302
                                    )
                            break
                except Exception as e:
                    print(f"[match] FotMob PL fallback error: {e}")

        # Absolute fallback — empty match page
        ctx = {
            "request"        : request,
            "current_league" : active_name,
            "featured"       : {}, "prior": None, "posterior": None,
            "pitch_svg"      : "", "fixtures": [],
            "home_colour"    : "#14b8a6", "away_colour": "#f43f5e",
            "home_formation" : "", "away_formation": "",
            **league_ctx(active_ctx_key),
        }
        return templates.TemplateResponse(
            request=request, name="match.html", context=ctx
        )

    prior, posterior, featured = None, None, {}
    pitch_svg    = ""
    timeline_svg = ""
    matrix_svg   = ""
    lineup_prior = None
    adj_lam      = None
    adj_mu       = None
    absent_home  = []
    absent_away  = []
    lineup_prior_post = None
    home_colour  = "#14b8a6"
    away_colour  = "#f43f5e"
    home_formation = "4-3-3"
    away_formation = "4-3-3"

    # Check if this match_id is in FPL upcoming fixtures (PL only)
    all_upcoming = []
    if active_ctx_key == "pl":
        try:
            all_upcoming = get_upcoming_fixtures(max_fixtures=50)
        except Exception:
            pass
    fpl_match = next(
        (f for f in all_upcoming if str(f.get("match_id")) == str(match_id)),
        None
    )

    # For matches not in upcoming list, try snapshot first
    # This makes finished games display correctly without any API calls
    if not fpl_match and match_id:
        snap = get_match_snapshot(match_id)
        if snap and snap.get("status") in ("Finished", "In Play", "Half Time"):
            home_name   = snap["home_team"]
            away_name   = snap["away_team"]
            h_score     = snap["h_score"]
            a_score     = snap["a_score"]
            status      = snap["status"]
            minute      = snap["minute"]
            home_colour = snap["home_colour"] or get_theme_for_team(home_name)["primary"]
            away_colour = snap["away_colour"] or get_theme_for_team(away_name)["primary"]
            home_players = snap["home_players"] or None
            away_players = snap["away_players"] or None
            home_formation = snap["home_formation"] or get_formation_for_team(home_name)
            away_formation = snap["away_formation"] or get_formation_for_team(away_name)
            fotmob_id   = snap.get("fotmob_id", "")

            featured = {
                "home_name" : home_name, "away_name": away_name,
                "minute"    : minute,    "status"   : status,
                "h_score"   : h_score,   "a_score"  : a_score,
                "h_xg"      : 0.0,       "a_xg"     : 0.0,
                "fixture_id": int(match_id),
                "fotmob_id" : fotmob_id,
            }
            prior = dc_pregame(home_name, away_name, active_league)
            posterior = nn_live(home_name, away_name, active_league,
                               int(minute), h_score, a_score)
            absent_home = snap["absent_home"]
            absent_away = snap["absent_away"]

            pitch_svg = generate_pitch_svg_horizontal(
                home_formation=home_formation, away_formation=away_formation,
                home_color=home_colour, away_color=away_colour,
                home_team=home_name, away_team=away_name,
                home_players=home_players, away_players=away_players,
                home_crest_url=get_crest_proxy_url(home_name),
                away_crest_url=get_crest_proxy_url(away_name),
                h_score=h_score, a_score=a_score, status=status,
            )
            matrix_svg = build_scoreline_svg(
                home_name=home_name, away_name=away_name,
                league=active_league, priors_db=priors_db,
                home_colour=home_colour, away_colour=away_colour,
                max_goals=4, cell_size=38,
            )
            if status == "Finished":
                timeline_svg = build_match_timeline_svg(
                    home_name=home_name, away_name=away_name,
                    league=active_league, h_score=h_score, a_score=a_score,
                    home_colour=home_colour, away_colour=away_colour,
                    dc_lookup=dc_lookup, nn_model=nn_model,
                    nn_scaler=nn_scaler, nn_T=nn_T,
                )
            # Skip all further API calls — snapshot has everything
            snap = None  # clear so we don't enter elif block below

    if fpl_match:
        # Match found in FPL upcoming list
        home_name = fpl_match["home"]
        away_name = fpl_match["away"]
        kickoff   = fpl_match.get("kickoff_utc", "")

        # Check if kickoff has passed — if so, try football-data.org
        # for the actual status (it works for finished matches)
        from datetime import datetime, timezone as _tz
        now_utc   = datetime.now(_tz.utc)
        kicked_off = False
        if kickoff:
            try:
                ko_dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                kicked_off = now_utc > ko_dt
            except Exception:
                pass

        # For finished/live games: football-data.org has the result
        fd_data = None
        if kicked_off:
            fd_data = get_live_match_data(match_id)

        if fd_data and fd_data.get("status") in ("Finished", "In Play", "Half Time"):
            # Game has result — use football-data.org data
            h_score = fd_data.get("h_score", 0)
            a_score = fd_data.get("a_score", 0)
            status  = fd_data.get("status", "Finished")
            minute  = fd_data.get("current_minute", 90)
        else:
            h_score = 0
            a_score = 0
            status  = "Not Started"
            minute  = 0

        featured = {
            "home_name" : home_name,
            "away_name" : away_name,
            "minute"    : minute,
            "status"    : status,
            "h_score"   : h_score,
            "a_score"   : a_score,
            "h_xg"      : 0.0,
            "a_xg"      : 0.0,
            "fixture_id": fpl_match.get("match_id"),
        }
        prior = dc_pregame(home_name, away_name, active_league)
        if status not in ("Not Started",):
            posterior = nn_live(home_name, away_name, active_league,
                               int(minute), h_score, a_score)
        home_theme     = get_theme_for_team(home_name)
        away_theme     = get_theme_for_team(away_name)
        home_colour    = home_theme["primary"]
        away_colour    = away_theme["primary"]

        # Try FotMob for confirmed lineup
        # FotMob IDs differ from FPL codes -- resolve via team name lookup
        fotmob_lineup = None
        fotmob_id = None
        if fotmob_available():
            fotmob_id = get_fotmob_match_id(home_name, away_name)
            if fotmob_id:
                fotmob_lineup = get_lineup(fotmob_id)
                featured["fotmob_id"] = fotmob_id

        # Try FotMob for confirmed lineup
        # FotMob IDs differ from FPL codes -- resolve via team name lookup
        fotmob_lineup = None
        fotmob_id = None
        if fotmob_available():
            fotmob_id = get_fotmob_match_id(home_name, away_name)
            if fotmob_id:
                fotmob_lineup = get_lineup(fotmob_id)
                featured["fotmob_id"] = fotmob_id

        if fotmob_lineup and fotmob_lineup.get("home_players"):
            home_formation = fotmob_lineup["home_formation"]
            away_formation = fotmob_lineup["away_formation"]
            home_players   = fotmob_lineup["home_players"]
            away_players   = fotmob_lineup["away_players"]
        else:
            home_formation = get_formation_for_team(home_name)
            away_formation = get_formation_for_team(away_name)
            home_players   = None
            away_players   = None

        # Get live status from FotMob date cache (zero extra credits)
        # This is the authoritative source for score, minute, half-time, extra time
        if fotmob_id and fotmob_available():
            fm = get_match_status_from_date_cache(fotmob_id, home_name, away_name)
            if fm and fm["started"]:
                if fm["finished"]:
                    featured["status"]  = "Finished"
                    featured["minute"]  = 90
                elif fm["is_halftime"]:
                    featured["status"]  = "Half Time"
                    featured["minute"]  = 45
                elif fm["ongoing"]:
                    featured["status"]  = "In Play"
                    featured["minute"]  = fm["minute_int"]
                featured["h_score"] = fm["home_score"]
                featured["a_score"] = fm["away_score"]
                # Activate neural net
                if featured["status"] != "Not Started":
                    safe_min = int(featured["minute"])
                    posterior = nn_live(
                        home_name, away_name, active_league,
                        safe_min, featured["h_score"], featured["a_score"],
                    )

        # Lineup-adjusted odds (only when confirmed lineups available)
        lineup_prior = None
        adj_lam = adj_mu = None
        absent_home = absent_away = []
        if fotmob_lineup and home_players and priors_db:
            league_data = priors_db.get(active_league, {})
            teams = league_data.get("teams", {})
            meta  = league_data.get("meta", {})
            hk = TEAM_NAME_ALIASES.get(home_name, home_name)
            ak = TEAM_NAME_ALIASES.get(away_name, away_name)
            h_p = teams.get(hk, {})
            a_p = teams.get(ak, {})
            if h_p and a_p:
                gamma    = get_effective_gamma(meta.get("gamma_home_advantage", 1.25))
                rho      = meta.get("rho_draw_correction", 0.0)
                base_lam = float(np.clip(h_p["alpha"] * a_p["beta"] * gamma, 1e-5, 15.0))
                base_mu  = float(np.clip(a_p["alpha"] * h_p["beta"], 1e-5, 15.0))
                lineup_prior, adj_lam, adj_mu = compute_lineup_adjusted_odds(
                    home_name=home_name, away_name=away_name,
                    home_lineup=home_players, away_lineup=away_players,
                    base_lam=base_lam, base_mu=base_mu, rho=rho,
                )
                absent_home = get_absent_key_players(home_name, home_players)
                absent_away = get_absent_key_players(away_name, away_players)
        matrix_svg = build_scoreline_svg(
            home_name=home_name, away_name=away_name,
            league=active_league, priors_db=priors_db,
            home_colour=home_colour, away_colour=away_colour,
            max_goals=4, cell_size=38,
        )
        # Render pitch AFTER FotMob status check so score/status are correct
        _h_score = featured.get("h_score", 0) if featured else 0
        _a_score = featured.get("a_score", 0) if featured else 0
        _status  = featured.get("status", "Not Started") if featured else "Not Started"
        pitch_svg = generate_pitch_svg_horizontal(
            home_formation=home_formation,
            away_formation=away_formation,
            home_color=home_colour,
            away_color=away_colour,
            home_team=home_name,
            away_team=away_name,
            home_players=home_players,
            away_players=away_players,
            home_crest_url=get_crest_proxy_url(home_name),
            away_crest_url=get_crest_proxy_url(away_name),
            h_score=_h_score, a_score=_a_score,
            status=_status,
        )

        # Save snapshot whenever we have live/finished state
        if _status not in ("Not Started", "") and match_id:
            try:
                save_match_snapshot(
                    match_id=str(match_id),
                    home_team=home_name, away_team=away_name,
                    kickoff_utc=fpl_match.get("kickoff_utc", ""),
                    h_score=_h_score, a_score=_a_score,
                    status=_status,
                    minute=int(featured.get("minute", 0) or 0),
                    home_formation=home_formation,
                    away_formation=away_formation,
                    home_players=home_players or [],
                    away_players=away_players or [],
                    absent_home=absent_home or [],
                    absent_away=absent_away or [],
                    home_colour=home_colour,
                    away_colour=away_colour,
                    fotmob_id=str(featured.get("fotmob_id", "") or ""),
                )
            except Exception as _e:
                print(f"[snapshot] save failed: {_e}")

    elif match_id:
        # Match not in FPL upcoming. Strategy:
        # 1. Check if this is a football-data.org match ID (6 digits)
        #    → look up directly in finished match cache
        # 2. Get team names from predictions DB (logged at announcement)
        # 3. Fall back to FotMob date cache for live matches

        from predictions import get_all_predictions
        from footballdata import _populate_finished_cache
        fd_home = fd_away = ""
        live_data = None
        home_players   = None
        away_players   = None
        home_formation = None
        away_formation = None

        # Step 0a: BBS ID (bbs_UUID) — La Liga live matches
        if str(match_id).startswith("bbs_"):
            bbs_uuid = str(match_id)[4:]
            try:
                from bbs import get_match_detail
                detail = get_match_detail(bbs_uuid)
                if detail:
                    fd_home = detail.get("home", "")
                    fd_away = detail.get("away", "")
                    live_data = {
                        "home_team"     : fd_home,
                        "away_team"     : fd_away,
                        "h_score"       : detail.get("h_score", 0),
                        "a_score"       : detail.get("a_score", 0),
                        "status"        : ("Finished" if detail.get("is_finished")
                                          else "In Play" if detail.get("is_live")
                                          else "Not Started"),
                        "current_minute": 90 if detail.get("is_finished") else 0,
                        "referee"       : "",
                        "venue"         : "",
                    }
                    print(f"[bbs] match detail: {fd_home} vs {fd_away} "
                          f"{detail.get('h_score')}-{detail.get('a_score')} "
                          f"{detail.get('status')}")
            except Exception as e:
                print(f"[match] BBS detail error: {e}")

        # Step 0b: FotMob ID (fm_XXXXX) — fallback for other leagues
        raw_fotmob_id = None
        if str(match_id).startswith("fm_"):
            raw_fotmob_id = str(match_id)[3:]  # strip "fm_" prefix
            if fotmob_available():
                try:
                    from fotmob import _lineup_cache, _refresh_date_cache_if_stale
                    from datetime import datetime, timezone, timedelta
                    today_str = datetime.now(
                        timezone(timedelta(hours=3))
                    ).strftime("%Y%m%d")
                    _refresh_date_cache_if_stale(today_str)
                    cache_key = f"matches_{today_str}"
                    date_data = _lineup_cache.get(cache_key, {})
                    leagues   = date_data.get("data", {}).get("leagues", [])
                    for lg in leagues:
                        for m in lg.get("matches", []):
                            if str(m.get("id","")) == raw_fotmob_id:
                                fd_home = m.get("home", {}).get("name", "")
                                fd_away = m.get("away", {}).get("name", "")
                                st      = m.get("status", {}) or {}
                                score   = st.get("scoreStr","0 - 0") or "0 - 0"
                                hg = ag = 0
                                if " - " in score:
                                    parts = score.split(" - ")
                                    try: hg,ag = int(parts[0]),int(parts[1])
                                    except: pass
                                live_data = {
                                    "home_team"     : fd_home,
                                    "away_team"     : fd_away,
                                    "h_score"       : hg,
                                    "a_score"       : ag,
                                    "status"        : ("Finished" if st.get("finished")
                                                      else "In Play" if st.get("ongoing")
                                                      else "Not Started"),
                                    "current_minute": 90 if st.get("finished") else 0,
                                }
                                break
                        if fd_home: break
                except Exception as e:
                    print(f"[match] FotMob fm_ lookup error: {e}")

        # Step 1: fd.org cache lookup — runs for ALL match IDs including fm_ ones
        # When FotMob is 429ing, this provides the team names from last season's
        # La Liga data so the DC charts still render with real teams
        if not fd_home:
            try:
                fd_cache = _populate_finished_cache(active_comp)
                # For fm_ IDs: scan all cached matches for today's teams
                # For numeric IDs: direct lookup by fd_id
                if not str(match_id).startswith("fm_"):
                    if str(match_id) in fd_cache:
                        live_data = fd_cache[str(match_id)]
                        fd_home   = live_data.get("home_team", "")
                        fd_away   = live_data.get("away_team", "")
                else:
                    # FotMob failed — use most recent finished match from fd.org
                    # so charts render with real La Liga teams
                    if fd_cache:
                        last      = list(fd_cache.values())[-1]
                        fd_home   = last.get("home_team", "")
                        fd_away   = last.get("away_team", "")
                        live_data = last
            except Exception:
                pass

        # Step 2: predictions DB (for FPL match codes — 7-digit IDs)
        if not fd_home:
            preds = get_all_predictions()
            pred  = next((p for p in preds
                          if str(p["match_id"]) == str(match_id)), None)
            if pred:
                fd_home = pred["home_team"]
                fd_away = pred["away_team"]

        # Step 3: if still no team names, scan FPL fixtures cache
        if not fd_home:
            # FPL stores 'code' = FPL match code in bootstrap
            from fpl import get_team_map
            import urllib.request as _ur, json as _json
            try:
                req = _ur.Request(
                    "https://fantasy.premierleague.com/api/fixtures/",
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                with _ur.urlopen(req, timeout=8) as resp:
                    all_fx = _json.loads(resp.read())
                team_map = get_team_map()
                for fx in all_fx:
                    if str(fx.get("code")) == str(match_id):
                        fd_home = team_map.get(fx.get("team_h"), "")
                        fd_away = team_map.get(fx.get("team_a"), "")
                        break
            except Exception:
                pass

        home_name = fd_home or "Unknown Home"
        away_name = fd_away or "Unknown Away"

        # Step 3: find finished match data by team name in the correct league
        # Skip if we already have live_data from BBS (don't overwrite live score)
        bbs_live_data = live_data if str(match_id).startswith("bbs_") else None
        if fd_home and fd_away and not bbs_live_data:
            live_data = find_finished_match_by_teams(
                fd_home, fd_away, comp_code=active_comp
            )
            if not live_data or live_data.get("home_team") == "Unknown Home":
                live_data = None

        # Step 4: FotMob date cache for today's live matches
        fotmob_id = None
        fm        = None
        if fotmob_available() and fd_home and fd_away:
            fotmob_id = get_fotmob_match_id(fd_home, fd_away)
            fm = get_match_status_from_date_cache(
                fotmob_id or "0", fd_home, fd_away
            )

        # Build status/score
        # BBS live data takes priority when we have it
        if bbs_live_data:
            h_score = bbs_live_data.get("h_score", 0) or 0
            a_score = bbs_live_data.get("a_score", 0) or 0
            raw_st  = bbs_live_data.get("status", "Not Started")
            status  = {"In Play": "In Play", "Finished": "Finished",
                       "Not Started": "Not Started", "live": "In Play",
                       "finished": "Finished", "final": "Finished",
                       "scheduled": "Not Started"}.get(raw_st, raw_st)
            # Estimate minute from kickoff time — BBS doesn't provide a clock
            if status == "In Play":
                try:
                    from datetime import datetime, timezone as _tz
                    ko_raw = bbs_live_data.get("kickoff_utc", "")
                    if ko_raw:
                        ko_dt  = datetime.fromisoformat(ko_raw.replace("Z","+00:00"))
                        elapsed = (datetime.now(_tz.utc) - ko_dt).total_seconds() / 60
                        # Account for 15-min halftime break after 45 min
                        if elapsed > 60:
                            elapsed -= 15
                        minute = max(1, min(90, int(elapsed)))
                    else:
                        minute = 50  # safe midpoint fallback
                except Exception:
                    minute = 50
            elif status == "Finished":
                minute = 90
            else:
                minute = 0
        elif live_data and live_data.get("status") == "Finished":
            h_score = live_data.get("h_score", 0) or 0
            a_score = live_data.get("a_score", 0) or 0
            status  = "Finished"
            minute  = 90
        elif fm and fm["started"]:
            h_score = fm["home_score"]
            a_score = fm["away_score"]
            if fm["finished"]:
                status = "Finished"; minute = 90
            elif fm["is_halftime"]:
                status = "Half Time"; minute = 45
            elif fm["ongoing"]:
                status = "In Play"; minute = fm["minute_int"]
            else:
                status = "Not Started"; minute = 0
        elif live_data:
            h_score = live_data.get("h_score", 0)
            a_score = live_data.get("a_score", 0)
            status  = live_data.get("status", "Unknown")
            minute  = live_data.get("current_minute", 0)
        else:
            h_score = a_score = 0
            status  = "Not Started"
            minute  = 0

        if home_name not in (None, "Unknown Home", ""):
            # Handle bbs_ and fm_ prefixes in match_id
            try:
                _clean_id = str(match_id).replace("fm_","").replace("bbs_","")
                # UUIDs can't convert to int — use 0
                fixture_id_int = int(_clean_id) if _clean_id.isdigit() else 0
            except (ValueError, TypeError):
                fixture_id_int = 0

            featured = {
                "home_name" : home_name,
                "away_name" : away_name,
                "minute"    : minute,
                "status"    : status,
                "h_score"   : h_score,
                "a_score"   : a_score,
                "h_xg"      : 0.0,
                "a_xg"      : 0.0,
                "fixture_id": fixture_id_int,
                "fotmob_id" : fotmob_id or "",
                "venue"     : live_data.get("venue", "") if live_data else "",
                "referee"   : live_data.get("referee", "") if live_data else "",
            }

            prior = dc_pregame(home_name, away_name, active_league)
            home_theme     = get_theme_for_team(home_name)
            away_theme     = get_theme_for_team(away_name)
            home_colour    = home_theme["primary"]
            away_colour    = away_theme["primary"]
            home_formation = get_formation_for_team(home_name)
            away_formation = get_formation_for_team(away_name)

            if status not in ("Not Started", "Unknown", ""):
                safe_min = int(minute) if minute else 0
                posterior = nn_live(
                    home_name, away_name, active_league,
                    safe_min, h_score, a_score,
                )

        # Always try football-data.org — works for FINISHED matches
        # Returns 400 for live/upcoming, so errors are expected during games
        if not live_data:
            live_data = get_live_match_data(match_id)
            if live_data and not fd_home:
                fd_home   = live_data.get("home_team", "")
                fd_away   = live_data.get("away_team", "")
                home_name = fd_home or home_name
                away_name = fd_away or away_name

        # For finished games: football-data.org is authoritative
        if live_data and live_data.get("status") == "Finished":
            h_score = live_data.get("h_score", 0) or 0
            a_score = live_data.get("a_score", 0) or 0
            status  = "Finished"
            minute  = 90
            # Override fm result with definitive football-data.org data
            featured["h_score"] = h_score
            featured["a_score"] = a_score
            featured["status"]  = status
            featured["minute"]  = minute

        # Lineup from predictions DB (has cached lineup from match day)
        # Fall back to FotMob if available, then DEFAULT_SQUADS
        if not home_players and fotmob_id and fotmob_available():
            fotmob_lineup_post = get_lineup(fotmob_id)
            if fotmob_lineup_post and fotmob_lineup_post.get("home_players"):
                home_formation = fotmob_lineup_post["home_formation"]
                away_formation = fotmob_lineup_post["away_formation"]
                home_players   = fotmob_lineup_post["home_players"]
                away_players   = fotmob_lineup_post["away_players"]
                featured["home_name"] = home_name
                featured["away_name"] = away_name

        # Also compute lineup-adjusted odds if we have lineups
        lineup_prior_post = None
        if home_players and away_players and priors_db:
            league_data_p = priors_db.get(active_league, {})
            teams_p = league_data_p.get("teams", {})
            meta_p  = league_data_p.get("meta", {})
            hk_p = TEAM_NAME_ALIASES.get(home_name, home_name)
            ak_p = TEAM_NAME_ALIASES.get(away_name, away_name)
            h_p_p = teams_p.get(hk_p, {})
            a_p_p = teams_p.get(ak_p, {})
            if h_p_p and a_p_p:
                gamma_p = get_effective_gamma(meta_p.get("gamma_home_advantage", 1.25))
                rho_p   = meta_p.get("rho_draw_correction", 0.0)
                bl_p    = float(np.clip(h_p_p["alpha"]*a_p_p["beta"]*gamma_p,1e-5,15.0))
                bm_p    = float(np.clip(a_p_p["alpha"]*h_p_p["beta"],1e-5,15.0))
                try:
                    lineup_prior_post, _, _ = compute_lineup_adjusted_odds(
                        home_name=home_name, away_name=away_name,
                        home_lineup=home_players, away_lineup=away_players,
                        base_lam=bl_p, base_mu=bm_p, rho=rho_p,
                    )
                except Exception:
                    lineup_prior_post = None

        # Pitch SVG — always generate when we have team names
        # home_players/away_players may be None for historical matches
        # (pitch shows formation dots without player names)
        if featured and home_name not in ("Unknown Home", "", None):
            # Colours must be set before pitch generation
            home_colour    = home_colour or get_theme_for_team(home_name)["primary"]
            away_colour    = away_colour or get_theme_for_team(away_name)["primary"]
            home_formation = home_formation or get_formation_for_team(home_name)
            away_formation = away_formation or get_formation_for_team(away_name)

            pitch_svg = generate_pitch_svg_horizontal(
                home_formation=home_formation,
                away_formation=away_formation,
                home_color=home_colour,
                away_color=away_colour,
                home_team=home_name,
                away_team=away_name,
                home_players=home_players,
                away_players=away_players,
                home_crest_url=get_crest_proxy_url(home_name),
                away_crest_url=get_crest_proxy_url(away_name),
                h_score=h_score, a_score=a_score, status=status,
            )

            safe_minute_int = 0
            try:
                safe_minute_int = int(str(minute).replace("'","").strip())
            except (ValueError, TypeError):
                pass
            try:
                safe_minute = int(str(minute).replace("'","").strip())
            except (ValueError, TypeError):
                pass

            if status not in ("Not Started","","") and safe_minute > 0:
                posterior = nn_live(home_name, away_name, active_league,
                                    safe_minute, h_score, a_score)

            # Scoreline probability matrix (always shown -- pre-game prediction)
            matrix_svg = build_scoreline_svg(
                home_name=home_name, away_name=away_name,
                league=active_league, priors_db=priors_db,
                home_colour=home_colour, away_colour=away_colour,
                max_goals=4, cell_size=38,
            )

            # Win probability timeline
            timeline_svg = ""
            if status in ("Finished", "FT") and (h_score + a_score) > 0:
                try:
                    timeline_svg = build_match_timeline_svg(
                        home_name=home_name,
                        away_name=away_name,
                        league=active_league,
                        h_score=h_score,
                        a_score=a_score,
                        home_colour=home_colour,
                        away_colour=away_colour,
                        dc_lookup=dc_lookup,
                        nn_model=nn_model,
                        nn_scaler=nn_scaler,
                        nn_T=nn_T,
                    )
                except Exception as e:
                    print(f"[match] timeline_svg error: {e}")
                    timeline_svg = ""

    # Team ratings for the right panel
    home_alpha = home_beta = away_alpha = away_beta = None
    home_xg_proj = away_xg_proj = 0.0
    league_gamma = league_rho = None
    home_alpha_estimated = away_alpha_estimated = False

    if featured and priors_db:
        league_data = priors_db.get(active_league, {})
        teams = league_data.get("teams", {})
        meta  = league_data.get("meta", {})
        hk = TEAM_NAME_ALIASES.get(home_name if featured else "", "")
        ak = TEAM_NAME_ALIASES.get(away_name if featured else "", "")
        if not hk: hk = home_name if featured else ""
        if not ak: ak = away_name if featured else ""

        # Bottom-quartile fallback for promoted/cup sides not in priors
        all_alpha = [v["alpha"] for v in teams.values()]
        all_beta  = [v["beta"]  for v in teams.values()]
        q25_alpha = float(np.percentile(all_alpha, 25))
        q75_beta  = float(np.percentile(all_beta,  75))

        h_params = teams.get(hk)
        a_params = teams.get(ak)

        if h_params:
            home_alpha = h_params.get("alpha")
            home_beta  = h_params.get("beta")
        else:
            home_alpha = q25_alpha
            home_beta  = q75_beta
            home_alpha_estimated = True

        if a_params:
            away_alpha = a_params.get("alpha")
            away_beta  = a_params.get("beta")
        else:
            away_alpha = q25_alpha
            away_beta  = q75_beta
            away_alpha_estimated = True

        if home_alpha and away_beta:
            gamma = get_effective_gamma(meta.get("gamma_home_advantage", 1.25))
            home_xg_proj = round(home_alpha * away_beta * gamma, 2)
        if away_alpha and home_beta:
            away_xg_proj = round(away_alpha * home_beta, 2)
        league_gamma = meta.get("gamma_home_advantage")
        league_rho   = meta.get("rho_draw_correction")

    # Fixture strip — source depends on league
    fixtures = []
    if active_ctx_key == "pl":
        try:
            fixtures = get_upcoming_fixtures(max_fixtures=10)
        except Exception:
            pass
    else:
        # Non-PL: BBS primary, FotMob fallback, fd.org last resort
        # BBS: get today's matches + upcoming
        try:
            from bbs import get_today_matches, get_upcoming_fixtures as bbs_upcoming
            from bbs import has_key as bbs_ok
            if bbs_ok():
                # Today's matches first (shows live scores)
                today = get_today_matches(active_ctx_key)
                for m in today:
                    sc = ""
                    if m.get("is_live"):
                        sc = f"🔴 {m['h_score']}-{m['a_score']}"
                    elif m.get("is_finished"):
                        sc = f"FT {m['h_score']}-{m['a_score']}"
                    else:
                        sc = m.get("kickoff_eat", "Today")
                    fixtures.append({
                        "match_id"   : f"bbs_{m['bbs_id']}",
                        "home"       : m["home"],
                        "away"       : m["away"],
                        "kickoff_eat": sc,
                    })
                # Upcoming fixtures after today
                upcoming = bbs_upcoming(active_ctx_key, max_fixtures=8)
                for m in upcoming:
                    fixtures.append({
                        "match_id"   : f"bbs_{m['bbs_id']}",
                        "home"       : m["home"],
                        "away"       : m["away"],
                        "kickoff_eat": m["kickoff_eat"],
                    })
        except Exception as e:
            print(f"[match] BBS fixture strip error: {e}")
        # Clear stale failure flag so we retry after rate limit clears
        try:
            from fotmob import _lineup_cache, _refresh_date_cache_if_stale
            from datetime import datetime, timezone, timedelta
            _EAT = timezone(timedelta(hours=3))
            today_str = datetime.now(_EAT).strftime("%Y%m%d")
            failed_key = f"matches_{today_str}_failed"
            failed_ts  = f"matches_{today_str}_failed_ts"
            fail_time  = _lineup_cache.get(failed_ts, 0)
            if _lineup_cache.get(failed_key) and (_time.time() - fail_time) > 300:
                _lineup_cache.pop(failed_key, None)
                _lineup_cache.pop(failed_ts, None)
                print(f"[match] FotMob backoff expired — retrying")

            _refresh_date_cache_if_stale(today_str)
            cache_key = f"matches_{today_str}"
            date_data = _lineup_cache.get(cache_key, {})
            _LEAGUE_FILTERS = {
                "laliga"    : ["laliga", "la liga", "primera", "esp"],
                "bundesliga": ["bundesliga", "germany"],
                "seriea"    : ["serie a", "italy"],
                "ligue1"    : ["ligue 1", "france"],
            }
            filters = _LEAGUE_FILTERS.get(active_ctx_key, [])
            for lg in date_data.get("data", {}).get("leagues", []):
                lg_name = lg.get("name", "").lower()
                if any(f in lg_name for f in filters):
                    for m in lg.get("matches", []):
                        st      = m.get("status", {}) or {}
                        score   = st.get("scoreStr", "")
                        is_fin  = st.get("finished", False)
                        is_live = st.get("ongoing",  False)
                        utc_raw = st.get("utcTime", "")
                        kickoff_display = ""
                        if utc_raw:
                            try:
                                ko = datetime.fromisoformat(
                                    utc_raw.replace("Z", "+00:00")
                                ).astimezone(_EAT)
                                kickoff_display = ko.strftime("%H:%M EAT")
                            except Exception:
                                kickoff_display = utc_raw[:5]
                        label = (f"FT {score}" if is_fin
                                 else f"🔴 {score}" if is_live
                                 else kickoff_display or "Today")
                        fixtures.append({
                            "match_id"   : f"fm_{m.get('id','')}",
                            "home"       : m.get("home", {}).get("name", ""),
                            "away"       : m.get("away", {}).get("name", ""),
                            "kickoff_eat": label,
                        })
                    break
        except Exception as e:
            print(f"[match] fixture strip FotMob error: {e}")

        # FotMob empty (still 429) — fall back to fd.org PD finished cache
        if not fixtures:
            try:
                from footballdata import _populate_finished_cache
                fd_cache = _populate_finished_cache(active_comp)
                # Take the last 10 finished matches — most recent first
                recent = list(fd_cache.values())[-10:]
                recent.reverse()
                for m in recent:
                    h  = m.get("home_team", "") or m.get("home", "")
                    a  = m.get("away_team", "") or m.get("away", "")
                    hg = m.get("h_score",   m.get("home_score", ""))
                    ag = m.get("a_score",   m.get("away_score", ""))
                    score_str = f"{hg}-{ag}" if hg != "" else ""
                    fixtures.append({
                        "match_id"   : str(m.get("fd_id", m.get("fixture_id", ""))),
                        "home"       : h,
                        "away"       : a,
                        "kickoff_eat": f"FT {score_str}" if score_str else "Finished",
                    })
            except Exception as e:
                print(f"[match] fixture strip fd.org fallback error: {e}")

    ctx = {
        "request"          : request,
        "current_league"   : active_name,
        **league_ctx(active_ctx_key),
        "featured"         : featured,
        "prior"            : prior,
        "posterior"        : posterior,
        "pitch_svg"        : pitch_svg,
        "timeline_svg"     : timeline_svg,
        "matrix_svg"       : matrix_svg,
        "fixtures"         : fixtures,
        "home_colour"      : home_colour,
        "away_colour"      : away_colour,
        "home_formation"   : home_formation,
        "away_formation"   : away_formation,
        "home_alpha"              : home_alpha,
        "home_beta"               : home_beta,
        "away_alpha"              : away_alpha,
        "away_beta"               : away_beta,
        "home_xg_proj"            : home_xg_proj,
        "away_xg_proj"            : away_xg_proj,
        "league_gamma"            : league_gamma,
        "league_rho"              : league_rho,
        "home_alpha_estimated"    : home_alpha_estimated,
        "away_alpha_estimated"    : away_alpha_estimated,
        "fotmob_available"        : fotmob_available(),
        "lineup_prior"            : lineup_prior or lineup_prior_post,
        "adj_lam"                 : adj_lam,
        "adj_mu"                  : adj_mu,
        "absent_home"             : absent_home,
        "absent_away"             : absent_away,
        "home_team_id"            : _CREST_IDS.get(home_name if featured else "", 0),
        "away_team_id"            : _CREST_IDS.get(away_name if featured else "", 0),
    }
    # DEBUG — remove after confirming La Liga data flow
    if active_ctx_key != "pl":
        print(f"[DEBUG-LALIGA] match_id={match_id} featured={bool(featured)} "
              f"home={featured.get('home_name','') if featured else ''} "
              f"away={featured.get('away_name','') if featured else ''} "
              f"prior={prior} pitch={bool(pitch_svg)} matrix={bool(matrix_svg)} "
              f"fixtures={len(fixtures)} home_alpha={home_alpha}")

    return templates.TemplateResponse(request=request, name="match.html", context=ctx)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
