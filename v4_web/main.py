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
    try:
        t_id = int(t_id) if t_id else 17
    except ValueError:
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
        teams = LEAGUE_TEAMS.get(current_league_name, ["Arsenal", "Aston Villa", "Liverpool", "Manchester City"])

    standings = []
    for idx, team in enumerate(teams[:10]):
        standings.append({
            "name": team,
            "played": 38,
            "xg_diff": round(float(np.random.uniform(-15, 25)), 1),
            "points": 88 - (idx * 4)
        })
        
    home_name = teams[0]
    away_name = teams[1] if len(teams) > 1 else "Aston Villa"
    
    h_theme = get_theme_for_team(home_name)
    a_theme = get_theme_for_team(away_name)
    
    h_alpha = league_priors.get(home_name, {}).get('alpha', 1.42)
    a_beta = league_priors.get(away_name, {}).get('beta', 0.88)
    a_alpha = league_priors.get(away_name, {}).get('alpha', 1.05)
    h_beta = league_priors.get(home_name, {}).get('beta', 0.78)
    
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
        # PASS TEAMS AND COLORS DIRECTLY
        "svg_pitch": generate_pitch_svg_horizontal(
            home_formation="4-3-3",
            away_formation="4-2-3-1",
            home_color=h_theme["primary"],
            away_color=a_theme["primary"],
            home_team=home_name,
            away_team=away_name
        )
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
    try:
        t_id = int(t_id) if t_id else 17
    except ValueError:
        t_id = 17
        
    current_league_name = next((name for name, i in TOURNAMENTS.items() if i == t_id), "Premier League")
    teams = LEAGUE_TEAMS.get(current_league_name, ["Arsenal", "Aston Villa"])
    
    home_name = teams[0]
    away_name = teams[1] if len(teams) > 1 else "Aston Villa"
    
    h_theme = get_theme_for_team(home_name)
    a_theme = get_theme_for_team(away_name)
    
    p1, px, p2 = mock_probs(1.42, 0.88, 1.05, 0.78)
    
    # 4x4 Remainder Matrix for the UI Grid
    sample_matrix = [
        [0.15, 0.12, 0.05, 0.01],
        [0.18, 0.14, 0.06, 0.02],
        [0.08, 0.07, 0.04, 0.01],
        [0.03, 0.02, 0.01, 0.01]
    ]
    
    ctx = {
        "request": request,
        "active_page": "match",
        "current_league": current_league_name,
        "current_team": home_name,
        "available_teams": teams,
        "leagues": list(TOURNAMENTS.keys()),
        "tournaments": TOURNAMENTS,
        "current_tournament": t_id,
        "match": {
            "home_name": home_name,
            "away_name": away_name,
            "home_score": 2,
            "away_score": 1,
            "minute": 68,
            "stadium": "Emirates Stadium",
            "referee": "Michael Oliver",
            "svg_pitch": generate_pitch_svg_horizontal(
                home_formation="4-3-3",
                away_formation="4-2-3-1",
                home_color=h_theme["primary"],
                away_color=a_theme["primary"],
                home_team=home_name,
                away_team=away_name
            )
        },
        "probs": {"home_win": p1, "draw": px, "away_win": p2},
        "fair_odds": {'1': round(1/p1, 2), 'X': round(1/px, 2), '2': round(1/p2, 2)},
        "matrix": sample_matrix,
        "events": [
            {"time": "15", "detail": "Goal (Arsenal) - Saka shot on target (0.78 xG)"},
            {"time": "42", "detail": "Yellow Card (Aston Villa) - Cash"},
            {"time": "54", "detail": "Goal (Aston Villa) - Watkins (0.42 xG)"},
            {"time": "65", "detail": "Goal (Arsenal) - Havertz (0.64 xG)"}
        ]
    }
    return templates.TemplateResponse(request=request, name="match.html", context=ctx)


@app.get("/team", response_class=HTMLResponse)
async def team(request: Request):
    t_id = request.query_params.get("tournament_id", 17)
    try:
        t_id = int(t_id)
    except ValueError:
        t_id = 17
        
    current_league_name = next((name for name, i in TOURNAMENTS.items() if i == t_id), "Premier League")
    team_name = request.query_params.get("team_name", "Arsenal")
    
    theme = get_theme_for_team(team_name)
    manager = TEAM_MANAGERS.get(team_name, "Head Coach")
    
    # Generate the actual vertical pitch SVG
    pitch_svg = generate_pitch_svg_vertical(
        formation="4-3-3",
        team_color=theme["primary"],
        team_name=team_name
    )
    
    ctx = {
        "request": request,
        "active_page": "team",
        "current_league": current_league_name,
        "current_team": team_name,
        "team_color": theme["primary"],
        "available_teams": LEAGUE_TEAMS.get(current_league_name, ["Arsenal", "Aston Villa"]),
        "leagues": list(TOURNAMENTS.keys()),
        "tournaments": TOURNAMENTS,
        "current_tournament": t_id,
        "manager": manager,
        "prior_alpha": 1.42,
        "prior_beta": 0.78,
        "roster_delta": {"att": 0.936, "def": 1.064},
        "match_history": [
            {"result": "W 2-1", "opponent": "Aston Villa", "xg": "1.88 - 0.92"},
            {"result": "D 1-1", "opponent": "Manchester City", "xg": "0.95 - 1.20"},
            {"result": "W 2-0", "opponent": "Chelsea", "xg": "2.10 - 0.80"},
            {"result": "L 0-1", "opponent": "Liverpool", "xg": "0.85 - 1.40"}
        ],
        # KEY MATCHES EXPECTATION IN team.html
        "pitch_svg": pitch_svg
    }
    return templates.TemplateResponse(request=request, name="team.html", context=ctx)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
