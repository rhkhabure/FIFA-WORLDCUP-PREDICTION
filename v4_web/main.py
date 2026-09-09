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
                   get_formation_for_team, get_squad_for_team)
from fpl import get_upcoming_fixtures
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

app = FastAPI(title="V4 Quant Terminal")
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
@app.get("/", response_class=HTMLResponse)
async def hub(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


@app.get("/match", response_class=HTMLResponse)
async def match(request: Request):
    match_id = request.query_params.get("match_id")

    if not match_id:
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
    pitch_svg = ""
    home_colour = "#14b8a6"
    away_colour = "#f43f5e"
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
        home_formation = get_formation_for_team(home_name)
        away_formation = get_formation_for_team(away_name)
        pitch_svg = generate_pitch_svg_horizontal(
            home_formation=home_formation,
            away_formation=away_formation,
            home_color=home_colour,
            away_color=away_colour,
            home_team=home_name,
            away_team=away_name,
        )

    elif match_id:
        live_data = get_live_match_data(match_id)
        if live_data and live_data.get("home_team") not in (None, "Unknown Home"):
            home_name = live_data["home_team"]
            away_name = live_data["away_team"]
            minute    = live_data["current_minute"]
            h_score   = live_data.get("h_score", 0)
            a_score   = live_data.get("a_score", 0)
            status    = live_data["status"]

            featured = {
                "home_name" : home_name,
                "away_name" : away_name,
                "minute"    : minute,
                "status"    : status,
                "h_score"   : h_score,
                "a_score"   : a_score,
                "h_xg"      : live_data["live_xg"]["home"],
                "a_xg"      : live_data["live_xg"]["away"],
                "fixture_id": int(match_id),
            }

            prior = dc_pregame(home_name, away_name, LEAGUE_KEY)

            safe_minute = 0
            try:
                safe_minute = int(str(minute).replace("'","").strip())
            except (ValueError, TypeError):
                pass

            if status not in ("Not Started","","") and safe_minute > 0:
                posterior = nn_live(home_name, away_name, LEAGUE_KEY,
                                    safe_minute, h_score, a_score)

            # Pitch -- use real kit colours and typical formation
            home_theme     = get_theme_for_team(home_name)
            away_theme     = get_theme_for_team(away_name)
            home_colour    = home_theme["primary"]
            away_colour    = away_theme["primary"]
            home_formation = get_formation_for_team(home_name)
            away_formation = get_formation_for_team(away_name)
            pitch_svg = generate_pitch_svg_horizontal(
                home_formation=home_formation,
                away_formation=away_formation,
                home_color=home_colour,
                away_color=away_colour,
                home_team=home_name,
                away_team=away_name,
            )

    # Upcoming fixtures from FPL (no key, free, EAT times)
    fixtures = get_upcoming_fixtures(max_fixtures=10)

    ctx = {
        "request"        : request,
        "current_league" : "Premier League",
        "featured"       : featured,
        "prior"          : prior,
        "posterior"      : posterior,
        "pitch_svg"      : pitch_svg,
        "fixtures"       : fixtures,
        "home_colour"    : home_colour,
        "away_colour"    : away_colour,
        "home_formation" : home_formation,
        "away_formation" : away_formation,
    }
    return templates.TemplateResponse(request=request, name="match.html", context=ctx)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
