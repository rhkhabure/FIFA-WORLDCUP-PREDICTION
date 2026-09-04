import json
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
import sys

# Ensure import works
sys.path.append(str(Path(__file__).resolve().parent))
from utils import generate_pitch_svg_horizontal, generate_pitch_svg_vertical, TEAM_THEMES, LEAGUE_TEAMS, TEAM_MANAGERS, get_theme_for_team

app = FastAPI(title="V4 Quant Terminal")
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

TOURNAMENTS = {
    "Premier League": 17,
    "La Liga": 8,
    "Serie A": 23,
    "Bundesliga": 35,
    "Ligue 1": 34
}

# Locate prior db dynamically
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

# Simple fallback probability generator if models aren't loaded
def mock_probs(alpha_h, beta_a, alpha_a, beta_h):
    # Dummy calculation for display
    h_pow = alpha_h * beta_a * 1.15
    a_pow = alpha_a * beta_h
    total = h_pow + a_pow + 0.8
    return h_pow/total, 0.8/total, a_pow/total

@app.get("/", response_class=HTMLResponse)
async def hub(request: Request):
    
    t_id = request.query_params.get("tournament_id")
    if t_id:
        try:
            t_id = int(t_id)
        except ValueError:
            t_id = 17
    else:
        t_id = 17
        
    current_league_name = next((name for name, i in TOURNAMENTS.items() if i == t_id), "Premier League")
    
    # Grab teams from the priors db
    league_key = f"ENG-{current_league_name}" if current_league_name == "Premier League" else \
                 f"ESP-{current_league_name}" if current_league_name == "La Liga" else \
                 f"ITA-{current_league_name}" if current_league_name == "Serie A" else \
                 f"GER-{current_league_name}" if current_league_name == "Bundesliga" else \
                 f"FRA-{current_league_name}"
    
    league_priors = priors_db.get(league_key, {}).get("teams", {})
    if not league_priors:
        league_priors = priors_db.get(current_league_name, {}).get("teams", {})
        
    teams = list(league_priors.keys())
    if not teams:
        teams = ["Team A", "Team B", "Team C", "Team D", "Team E"] # Fallback

    # Build mock standings
    standings = []
    for idx, team in enumerate(teams[:10]):
        standings.append({
            "name": team,
            "played": 38,
            "xg_diff": round(np.random.uniform(-15, 25), 1),
            "points": 80 - (idx * 4)
        })
        
    home_name = teams[0]
    away_name = teams[1] if len(teams) > 1 else "Unknown"
    
    h_alpha = league_priors.get(home_name, {}).get('alpha', 1.1)
    a_beta = league_priors.get(away_name, {}).get('beta', 1.0)
    a_alpha = league_priors.get(away_name, {}).get('alpha', 0.9)
    h_beta = league_priors.get(home_name, {}).get('beta', 0.9)
    
    p1, px, p2 = mock_probs(h_alpha, a_beta, a_alpha, h_beta)
    
    featured = {
        "home_name": home_name,
        "away_name": away_name,
        "home_score": 2,
        "away_score": 1,
        "minute": 68,
        "prob_1": p1,
        "prob_x": px,
        "prob_2": p2,
        "svg_pitch": generate_pitch_svg_horizontal("4-3-3", "4-2-3-1")
    }
    
    ctx = {
        "request": request,
        "tournaments": TOURNAMENTS,
        "current_tournament": t_id,
        "current_league": current_league_name,
        "featured": featured,
        "standings": standings
    }
    
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)

@app.get("/match", response_class=HTMLResponse)
async def match(request: Request):
    
    t_id = request.query_params.get("tournament_id")
    if t_id:
        try:
            t_id = int(t_id)
        except ValueError:
            t_id = 17
    else:
        t_id = 17
        
    current_league_name = next((name for name, i in TOURNAMENTS.items() if i == t_id), "Premier League")
    
    league_key = f"ENG-{current_league_name}" if current_league_name == "Premier League" else \
                 f"ESP-{current_league_name}" if current_league_name == "La Liga" else \
                 f"ITA-{current_league_name}" if current_league_name == "Serie A" else \
                 f"GER-{current_league_name}" if current_league_name == "Bundesliga" else \
                 f"FRA-{current_league_name}"
    
    league_priors = priors_db.get(league_key, {}).get("teams", {})
    if not league_priors:
        league_priors = priors_db.get(current_league_name, {}).get("teams", {})
        
    teams = list(league_priors.keys())
    if not teams:
        teams = ["Team A", "Team B", "Team C", "Team D", "Team E"]
        
    home_name = teams[0]
    away_name = teams[1] if len(teams) > 1 else "Unknown"
    
    h_alpha = league_priors.get(home_name, {}).get('alpha', 1.1)
    a_beta = league_priors.get(away_name, {}).get('beta', 1.0)
    a_alpha = league_priors.get(away_name, {}).get('alpha', 0.9)
    h_beta = league_priors.get(home_name, {}).get('beta', 0.9)
    
    p1, px, p2 = mock_probs(h_alpha, a_beta, a_alpha, h_beta)
    
    featured = {
        "home_name": home_name,
        "away_name": away_name,
        "home_score": 2,
        "away_score": 1,
        "minute": 68,
        "prob_1": p1,
        "prob_x": px,
        "prob_2": p2,
        "svg_pitch": generate_pitch_svg_horizontal("4-3-3", "4-2-3-1")
    }
    
    # Base dummy context for template to render without crashing
    ctx = {
        "request": request,
        "active_page": "match",
        "current_league": current_league_name,
        "current_team": "Default",
        "available_teams": teams,
        "leagues": list(TOURNAMENTS.keys()),
        "tournaments": TOURNAMENTS,
        "current_tournament": t_id,
        "match_home": home_name,
        "match_away": away_name,
        "match_score": "2 - 1",
        "match_status": "68'",
        "prob_home": p1,
        "prob_draw": px,
        "prob_away": p2,
        "fair_odds": {'1': round(1/p1,2), 'X': round(1/px,2), '2': round(1/p2,2)},
        "pitch_svg": featured["svg_pitch"]
    }
    
    return templates.TemplateResponse(request=request, name="match.html", context=ctx)

@app.get("/team", response_class=HTMLResponse)
async def team(request: Request):
    
    t_id = request.query_params.get("tournament_id", 17)
    current_league_name = next((name for name, i in TOURNAMENTS.items() if i == int(t_id)), "Premier League")
    
    team_name = request.query_params.get("team_name", "Arsenal")
    
    # We will mock the team context to fulfill the team.html requirement
    ctx = {
        "request": request,
        "active_page": "team",
        "current_league": current_league_name,
        "current_team": team_name,
        "available_teams": ["Arsenal", "Aston Villa", "Bournemouth"],
        "leagues": list(TOURNAMENTS.keys()),
        "tournaments": TOURNAMENTS,
        "current_tournament": int(t_id),
        "manager": "Mikel Arteta",
        "prior_alpha": 1.42,
        "prior_beta": 0.78,
        "match_history": [
            {"result": "W 2-0", "opponent": "Chelsea", "xg": "2.1 - 0.8"},
            {"result": "D 1-1", "opponent": "Tottenham", "xg": "1.5 - 1.5"}
        ],
        "roster_delta": {"att": 0.936, "def": 1.064},
        "svg_pitch": "<svg width='100%' height='200px' style='background:#111827'></svg>"
    }
    
    return templates.TemplateResponse(request=request, name="team.html", context=ctx)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
