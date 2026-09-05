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
from footballdata import get_live_match_data

app = FastAPI(title="V4 Quant Terminal")
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

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

from v4_backend.bivariate_poisson import generate_match_probabilities

@app.get("/", response_class=HTMLResponse)
async def hub(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.get("/match", response_class=HTMLResponse)
async def match(request: Request):
    match_id = request.query_params.get("match_id")
    home_override = request.query_params.get("home")
    away_override = request.query_params.get("away")
    
    # Base states
    prior = None
    likelihood = None
    posterior = None
    featured = {}

    home_name = None
    away_name = None

    # Fetch live match data if we have an ID
    if match_id:
        live_data = get_live_match_data(match_id)
        if live_data and live_data.get("home_team") != "Unknown Home":
            home_name = live_data["home_team"]
            away_name = live_data["away_team"]
            minute = live_data["current_minute"]
            h_score = live_data["home_score"]
            a_score = live_data["away_score"]
            h_xg = live_data["live_xg"]["home"]
            a_xg = live_data["live_xg"]["away"]
            status = live_data["status"]
            featured = {
                "home_name": home_name,
                "away_name": away_name,
                "minute": minute,
                "status": status,
                "h_score": h_score,
                "a_score": a_score,
                "h_xg": h_xg,
                "a_xg": a_xg
            }
    elif home_override and away_override:
        # Allow testing the Prior Model without a live API match_id
        home_name = home_override
        away_name = away_override
        featured = {
            "home_name": home_name,
            "away_name": away_name,
            "minute": 0,
            "status": "Not Started"
        }

    # If we have teams to evaluate, run the Prior math engine
    if home_name and away_name:
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
            # ONLY RUN IF WE HAVE A LIVE MATCH WITH ACTUAL DATA
            if featured.get("status") and featured.get("status") not in ["Not Started"]:
                minute = featured["minute"]
                if isinstance(minute, int): safe_min = minute
                else:
                    try: safe_min = int(str(minute).replace("'", ""))
                    except: safe_min = 90
                    
                posterior_res = generate_live_in_play_odds(
                    current_minute=safe_min,
                    home_score=featured["h_score"],
                    away_score=featured["a_score"],
                    live_xg_h=featured["h_xg"],
                    live_xg_a=featured["a_xg"],
                    alpha_h_adj=h_alpha,
                    beta_a_adj=a_beta,
                    alpha_a_adj=a_alpha,
                    beta_h_adj=h_beta,
                    gamma=1.0, 
                    rho=0.0
                )
                
                pos_probs = posterior_res["live_probabilities"]
                posterior = [round(pos_probs["1"]*100, 1), round(pos_probs["X"]*100, 1), round(pos_probs["2"]*100, 1)]

    ctx = {
        "request": request,
        "current_league": "Premier League",
        "featured": featured,
        "prior": prior,
        "likelihood": likelihood,
        "posterior": posterior
    }
    return templates.TemplateResponse(request=request, name="match.html", context=ctx)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
