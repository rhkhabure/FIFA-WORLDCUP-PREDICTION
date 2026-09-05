import json
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v4_backend.in_play_posterior import generate_live_in_play_odds
from utils import generate_pitch_svg_horizontal, generate_pitch_svg_vertical, TEAM_THEMES, LEAGUE_TEAMS, TEAM_MANAGERS, get_theme_for_team
from footballdata import get_live_match_data

app = FastAPI(title="V4 Quant Terminal")
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

TOURNAMENTS = {
    "Premier League": 17,
    "La Liga": 8,
    "Serie A": 23,
    "Bundesliga": 35,
    "Ligue 1": 34
}

def find_file(filename):
    for p in Path(__file__).resolve().parent.parent.rglob(filename):
        return p
    return None

PRIORS_PATH = find_file("v4_priors.json")
if PRIORS_PATH and PRIORS_PATH.exists():
    with open(PRIORS_PATH, "r") as f:
        priors_db = json.load(f)
else:
    priors_db = {}

# Keep this for the Prior bar graph calculations ONLY
def calculate_prior_probs(alpha_h, beta_a, alpha_a, beta_h):
    # Using simplified fallback logic if bivariate_poisson not fully imported/running here
    # To be mathematically strict, we should import bivariate_poisson.
    pass

# We will import the actual poisson calculator so we are not mocking the math
from v4_backend.bivariate_poisson import generate_match_probabilities

@app.get("/", response_class=HTMLResponse)
async def hub(request: Request):
    # For now, wipe the index page and redirect or show a blank WIP
    # Since the instruction was "we are wiping clean all pages"
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.get("/match", response_class=HTMLResponse)
async def match(request: Request):
    match_id = request.query_params.get("match_id")
    
    # 1. Fetch live match data if we have an ID
    if match_id:
        live_data = get_live_match_data(match_id)
    else:
        live_data = None
        
    prior = None
    likelihood = None
    posterior = None
    featured = {}

    if live_data and live_data.get("home_team") != "Unknown Home":
        home_name = live_data["home_team"]
        away_name = live_data["away_team"]
        minute = live_data["current_minute"]
        h_score = live_data["home_score"]
        a_score = live_data["away_score"]
        h_xg = live_data["live_xg"]["home"]
        a_xg = live_data["live_xg"]["away"]
        
        # Grab priors from DB
        league_priors = priors_db.get("ENG-Premier League", {}).get("teams", {})
        if not league_priors:
            league_priors = priors_db.get("Premier League", {}).get("teams", {})
            
        if home_name in league_priors and away_name in league_priors:
            h_alpha = league_priors[home_name].get('alpha', 1.0)
            h_beta = league_priors[home_name].get('beta', 1.0)
            a_alpha = league_priors[away_name].get('alpha', 1.0)
            a_beta = league_priors[away_name].get('beta', 1.0)

            # Prior Calculation strictly from Model
            probs, _, _ = generate_match_probabilities(h_alpha, a_beta, a_alpha, h_beta, gamma=1.0, rho=0.0)
            prior = [round(probs["home_win"]*100, 1), round(probs["draw"]*100, 1), round(probs["away_win"]*100, 1)]
            
            # Likelihood (No lineup API yet -> None)
            likelihood = None
            
            # Posterior Calculation strictly from Model
            if isinstance(minute, int):
                safe_min = minute
            else:
                try: safe_min = int(str(minute).replace("'", ""))
                except: safe_min = 90
                
            posterior_res = generate_live_in_play_odds(
                current_minute=safe_min,
                home_score=h_score,
                away_score=a_score,
                live_xg_h=h_xg,
                live_xg_a=a_xg,
                alpha_h_adj=h_alpha,
                beta_a_adj=a_beta,
                alpha_a_adj=a_alpha,
                beta_h_adj=h_beta,
                gamma=1.0, 
                rho=0.0
            )
            
            pos_probs = posterior_res["live_probabilities"]
            posterior = [round(pos_probs["1"]*100, 1), round(pos_probs["X"]*100, 1), round(pos_probs["2"]*100, 1)]
        
        featured = {
            "home_name": home_name,
            "away_name": away_name,
            "minute": minute,
            "status": live_data["status"]
        }

    ctx = {
        "request": request,
        "current_league": "Premier League",
        "featured": featured,
        "prior": prior,
        "likelihood": likelihood,
        "posterior": posterior
    }
    return templates.TemplateResponse(request=request, name="match.html", context=ctx)

@app.get("/player", response_class=HTMLResponse)
async def player(request: Request):
    name = request.query_params.get("name", "Player")
    team = request.query_params.get("team", "Unknown")
    html = f"<div style='background:#0b0f19; color:white; height:100vh; font-family:monospace; padding:2rem;'><h1>PLAYER LOG: {name} ({team})</h1></div>"
    return HTMLResponse(html)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
