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

from footballdata import get_live_match_data, get_last_completed_pl_match
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
from predictions import init_db

@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan — starts background prediction job on startup."""
    init_db()
    if priors_db:
        asyncio.create_task(
            schedule_prediction_job(priors_db, LEAGUE_KEY, interval_minutes=30)
        )
        print("[startup] Prediction logging job scheduled (every 30 min)")
    else:
        print("[startup] No priors loaded — prediction job skipped")
    yield
    # Shutdown: nothing to clean up (SQLite handles its own flush)

app = FastAPI(title="V4 Quant Terminal", lifespan=lifespan)
templates = Jinja2Templates(directory=ROOT / "templates")


# ── DC pre-game ───────────────────────────────────────────────────────────────
def dc_pregame(home_team: str, away_team: str, league: str) -> list | None:
    """
    Dixon-Coles bivariate Poisson with draw_propensity correction.
    Returns [p_home%, p_draw%, p_away%] rounded to 1 dp, or None.
    Uses bottom-quartile fallback for teams not in priors (promoted clubs etc.)
    """
    if not priors_db:
        return None
    league_data = priors_db.get(league)
    if not league_data:
        return None
    teams  = league_data["teams"]
    meta   = league_data["meta"]

    # Apply alias table -- FPL uses different names than Understat
    home_key = TEAM_NAME_ALIASES.get(home_team, home_team)
    away_key = TEAM_NAME_ALIASES.get(away_team, away_team)

    # Bottom-quartile fallback for teams not in priors (promoted/cup sides)
    all_alpha = [v["alpha"] for v in teams.values()]
    all_beta  = [v["beta"]  for v in teams.values()]
    q25_alpha = float(np.percentile(all_alpha, 25))
    q75_beta  = float(np.percentile(all_beta,  75))

    h = teams.get(home_key, {"alpha": q25_alpha, "beta": q75_beta})
    a = teams.get(away_key, {"alpha": q25_alpha, "beta": q75_beta})
    gamma = meta.get("gamma_home_advantage", 1.25)
    rho   = meta.get("rho_draw_correction",  0.0)
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
    """Live polling — returns xG, score, minute from FotMob. Zero extra credits for status."""
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
        context={"request": request, "team_colours": team_colours}
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
            gamma    = meta.get("gamma_home_advantage", 1.25)
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


@app.get("/", response_class=HTMLResponse)
async def hub(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


@app.get("/match", response_class=HTMLResponse)
async def match(request: Request):
    match_id = request.query_params.get("match_id")

    if not match_id:
        # Default: redirect to the next upcoming PL fixture
        upcoming = get_upcoming_fixtures(max_fixtures=1)
        if upcoming:
            found_id = upcoming[0].get("match_id")
            if found_id:
                return RedirectResponse(url=f"/match?match_id={found_id}", status_code=302)
        # Final fallback: last completed match
        found_id = get_last_completed_pl_match()
        if found_id:
            return RedirectResponse(url=f"/match?match_id={found_id}", status_code=302)
        ctx = {"request": request, "current_league": "Premier League",
               "featured": {}, "prior": None, "posterior": None,
               "pitch_svg": "", "fixtures": [],
               "home_colour": "#14b8a6", "away_colour": "#f43f5e",
               "home_formation": "", "away_formation": ""}
        return templates.TemplateResponse(request=request, name="match.html", context=ctx)

    prior, posterior, featured = None, None, {}
    pitch_svg    = ""
    timeline_svg = ""
    matrix_svg   = ""
    lineup_prior = None
    adj_lam      = None
    adj_mu       = None
    absent_home  = []
    absent_away  = []
    home_colour  = "#14b8a6"
    away_colour  = "#f43f5e"
    home_formation = "4-3-3"
    away_formation = "4-3-3"

    # Check if this match_id is in the FPL upcoming fixtures first.
    # football-data.org free tier returns 400 for upcoming matches --
    # it only serves individual match detail for finished games.
    # For upcoming matches, we use FPL team names directly.
    all_upcoming = get_upcoming_fixtures(max_fixtures=50)
    fpl_match = next(
        (f for f in all_upcoming if str(f.get("match_id")) == str(match_id)),
        None
    )

    if fpl_match:
        # Upcoming match -- FPL has team names, no score data yet
        home_name = fpl_match["home"]
        away_name = fpl_match["away"]
        featured = {
            "home_name" : home_name,
            "away_name" : away_name,
            "minute"    : 0,
            "status"    : "Not Started",
            "h_score"   : 0,
            "a_score"   : 0,
            "h_xg"      : 0.0,
            "a_xg"      : 0.0,
            "fixture_id": fpl_match.get("match_id"),
        }
        prior = dc_pregame(home_name, away_name, LEAGUE_KEY)
        # No posterior -- match hasn't started
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
                        home_name, away_name, LEAGUE_KEY,
                        safe_min, featured["h_score"], featured["a_score"],
                    )

        # Lineup-adjusted odds (only when confirmed lineups available)
        lineup_prior = None
        adj_lam = adj_mu = None
        absent_home = absent_away = []
        if fotmob_lineup and home_players and priors_db:
            league_data = priors_db.get(LEAGUE_KEY, {})
            teams = league_data.get("teams", {})
            meta  = league_data.get("meta", {})
            hk = TEAM_NAME_ALIASES.get(home_name, home_name)
            ak = TEAM_NAME_ALIASES.get(away_name, away_name)
            h_p = teams.get(hk, {})
            a_p = teams.get(ak, {})
            if h_p and a_p:
                gamma    = meta.get("gamma_home_advantage", 1.25)
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
            league=LEAGUE_KEY, priors_db=priors_db,
            home_colour=home_colour, away_colour=away_colour,
            max_goals=4, cell_size=38,
        )
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
            h_score=0, a_score=0,
            status="Not Started",
        )

    elif match_id:
        # For matches not in FPL upcoming (already kicked off):
        # 1. Get team names from football-data.org (works for any match status)
        # 2. Use team names to find the match in FotMob date cache
        # 3. Build featured from FotMob live data

        # Get basic match info from football-data.org
        live_data  = get_live_match_data(match_id)
        fd_home    = live_data.get("home_team", "") if live_data else ""
        fd_away    = live_data.get("away_team", "") if live_data else ""
        home_name  = fd_home or "Unknown Home"
        away_name  = fd_away or "Unknown Away"

        # Try FotMob date cache with team name matching (handles ID mismatch)
        fotmob_id   = None
        fotmob_live = None
        fm          = None

        if fotmob_available() and fd_home and fd_away:
            # get_fotmob_match_id resolves via name matching
            fotmob_id = get_fotmob_match_id(fd_home, fd_away)
            if fotmob_id:
                fm = get_match_status_from_date_cache(
                    fotmob_id, fd_home, fd_away
                )

        # Also try by name directly in date cache even without FotMob ID
        if not fm and fd_home and fd_away:
            fm = get_match_status_from_date_cache("", fd_home, fd_away)

        # Build scores and status
        if fm and fm["started"]:
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
            status  = live_data.get("status", "")
            minute  = live_data.get("current_minute", 0)
        else:
            h_score = a_score = 0
            status  = "Unknown"
            minute  = 0

        if home_name not in (None, "Unknown Home", ""):
            featured = {
                "home_name" : home_name,
                "away_name" : away_name,
                "minute"    : minute,
                "status"    : status,
                "h_score"   : h_score,
                "a_score"   : a_score,
                "h_xg"      : 0.0,
                "a_xg"      : 0.0,
                "fixture_id": int(match_id),
                "fotmob_id" : fotmob_id or "",
                "venue"     : live_data.get("venue", "") if live_data else "",
                "referee"   : live_data.get("referee", "") if live_data else "",
            }

            prior = dc_pregame(home_name, away_name, LEAGUE_KEY)
            home_theme     = get_theme_for_team(home_name)
            away_theme     = get_theme_for_team(away_name)
            home_colour    = home_theme["primary"]
            away_colour    = away_theme["primary"]
            home_formation = get_formation_for_team(home_name)
            away_formation = get_formation_for_team(away_name)

            if status not in ("Not Started", "Unknown", ""):
                safe_min = int(minute) if minute else 0
                posterior = nn_live(
                    home_name, away_name, LEAGUE_KEY,
                    safe_min, h_score, a_score,
                )

            # Try lineup from FotMob
            if fotmob_id and fotmob_available():
                fotmob_lineup = get_lineup(fotmob_id)
                if fotmob_lineup and fotmob_lineup.get("home_players"):
                    home_formation = fotmob_lineup["home_formation"]
                    away_formation = fotmob_lineup["away_formation"]
                    home_players   = fotmob_lineup["home_players"]
                    away_players   = fotmob_lineup["away_players"]

            pitch_svg = generate_pitch_svg_horizontal(
                home_formation=home_formation, away_formation=away_formation,
                home_color=home_colour, away_color=away_colour,
                home_team=home_name, away_team=away_name,
                home_players=home_players, away_players=away_players,
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
                posterior = nn_live(home_name, away_name, LEAGUE_KEY,
                                    safe_minute, h_score, a_score)

            # Colours and formation -- must come BEFORE timeline and pitch
            home_theme     = get_theme_for_team(home_name)
            away_theme     = get_theme_for_team(away_name)
            home_colour    = home_theme["primary"]
            away_colour    = away_theme["primary"]
            home_formation = get_formation_for_team(home_name)
            away_formation = get_formation_for_team(away_name)

            # Scoreline probability matrix (always shown -- pre-game prediction)
            matrix_svg = build_scoreline_svg(
                home_name=home_name, away_name=away_name,
                league=LEAGUE_KEY, priors_db=priors_db,
                home_colour=home_colour, away_colour=away_colour,
                max_goals=4, cell_size=38,
            )

            # Win probability timeline
            timeline_svg = ""
            if status in ("Finished", "FT") and (h_score + a_score) > 0:
                timeline_svg = build_match_timeline_svg(
                    home_name=home_name,
                    away_name=away_name,
                    league=LEAGUE_KEY,
                    h_score=h_score,
                    a_score=a_score,
                    home_colour=home_colour,
                    away_colour=away_colour,
                    dc_lookup=dc_lookup,
                    nn_model=nn_model,
                    nn_scaler=nn_scaler,
                    nn_T=nn_T,
                )

            # Pitch
            pitch_svg = generate_pitch_svg_horizontal(
                home_formation=home_formation,
                away_formation=away_formation,
                home_color=home_colour,
                away_color=away_colour,
                home_team=home_name,
                away_team=away_name,
                home_crest_url=get_crest_proxy_url(home_name) or get_crest_url(
                    home_name, live_data.get("home_crest", "")
                ),
                away_crest_url=get_crest_proxy_url(away_name) or get_crest_url(
                    away_name, live_data.get("away_crest", "")
                ),
                h_score=h_score,
                a_score=a_score,
                status=status,
            )

    # Team ratings for the right panel
    home_alpha = home_beta = away_alpha = away_beta = None
    home_xg_proj = away_xg_proj = 0.0
    league_gamma = league_rho = None
    home_alpha_estimated = away_alpha_estimated = False

    if featured and priors_db:
        league_data = priors_db.get(LEAGUE_KEY, {})
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
            gamma = meta.get("gamma_home_advantage", 1.25)
            home_xg_proj = round(home_alpha * away_beta * gamma, 2)
        if away_alpha and home_beta:
            away_xg_proj = round(away_alpha * home_beta, 2)
        league_gamma = meta.get("gamma_home_advantage")
        league_rho   = meta.get("rho_draw_correction")

    # Upcoming fixtures from FPL (no key, free, EAT times)
    fixtures = get_upcoming_fixtures(max_fixtures=10)

    ctx = {
        "request"          : request,
        "current_league"   : "Premier League",
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
        "lineup_prior"            : lineup_prior,
        "adj_lam"                 : adj_lam,
        "adj_mu"                  : adj_mu,
        "absent_home"             : absent_home,
        "absent_away"             : absent_away,
        "home_team_id"            : _CREST_IDS.get(home_name if featured else "", 0),
        "away_team_id"            : _CREST_IDS.get(away_name if featured else "", 0),
    }
    return templates.TemplateResponse(request=request, name="match.html", context=ctx)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
