import json
import numpy as np
import random
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
import sys

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

def mock_probs(alpha_h, beta_a, alpha_a, beta_h):
    h_pow = alpha_h * beta_a * 1.15
    a_pow = alpha_a * beta_h
    total = h_pow + a_pow + 0.8
    return h_pow/total, 0.8/total, a_pow/total

@app.get("/", response_class=HTMLResponse)
async def hub(request: Request):
    t_id = request.query_params.get("tournament_id", 17)
    try: t_id = int(t_id)
    except: t_id = 17
        
    current_league_name = next((name for name, i in TOURNAMENTS.items() if i == t_id), "Premier League")
    teams = LEAGUE_TEAMS.get(current_league_name, ["Arsenal", "Manchester City", "Liverpool", "Aston Villa", "Chelsea", "Tottenham Hotspur"])

    # 1. Generate Scatter Plot Data
    scatter_data = []
    for t in teams:
        a = random.uniform(0.7, 1.4)
        b = random.uniform(0.7, 1.4)
        scatter_data.append({
            "team": t, 
            "short": t[:3].upper(), 
            "alpha": round(a, 2), 
            "beta": round(b, 2)
        })

    # 2. Generate Quant Fixtures
    fixtures = []
    for i in range(0, min(len(teams), 10), 2):
        if i+1 < len(teams):
            h, a = teams[i], teams[i+1]
            ha, hb = random.uniform(1.0, 1.4), random.uniform(0.7, 1.1)
            aa, ab = random.uniform(0.9, 1.3), random.uniform(0.8, 1.2)
            p1, px, p2 = mock_probs(ha, ab, aa, hb)
            
            fixtures.append({
                "home": h, "away": a,
                "h_alpha": ha, "h_beta": hb,
                "a_alpha": aa, "a_beta": ab,
                "prob_1": round(p1*100, 1), "prob_x": round(px*100, 1), "prob_2": round(p2*100, 1),
                "odds_1": round(1/p1, 2) if p1>0 else 0,
                "odds_x": round(1/px, 2) if px>0 else 0,
                "odds_2": round(1/p2, 2) if p2>0 else 0,
                "ev": "HOME +EV" if p1 > 0.45 else ("AWAY +EV" if p2 > 0.4 else "NO EDGE")
            })

    # 3. Generate Compact Standings Table
    standings = []
    for idx, t in enumerate(teams[:10]):
        xg = random.uniform(30, 60)
        xga = random.uniform(20, 50)
        standings.append({
            "name": t, "played": 25,
            "xg": round(xg, 1), "xga": round(xga, 1),
            "xgd_90": (xg - xga)/25,
            "points": 50 - idx*3
        })
    
    ctx = {
        "request": request,
        "current_league": current_league_name,
        "fixtures": fixtures,
        "scatter_data": scatter_data,
        "standings": standings
    }
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)


@app.get("/match", response_class=HTMLResponse)
async def match(request: Request):
    home_name = request.query_params.get("home", "Arsenal")
    away_name = request.query_params.get("away", "Manchester City")
    
    h_theme = get_theme_for_team(home_name)
    a_theme = get_theme_for_team(away_name)
    
    # Base Probabilities (Prior, Likelihood, Posterior)
    p1, px, p2 = mock_probs(1.42, 0.88, 1.05, 0.78)
    
    prior = [round(p1*100, 1), round(px*100, 1), round(p2*100, 1)]
    likelihood = [max(0, prior[0]-5), prior[1]+1, min(100, prior[2]+4)] # example shift
    posterior = [max(0, likelihood[0]-6), likelihood[1]+2, min(100, likelihood[2]+4)] # example live shift
    
    # Dynamic Match Momentum Path
    pts = []
    x_step = 100 / 20
    x, y = 0, 25
    for i in range(21):
        pts.append(f"{x},{y}")
        x += x_step
        y += random.uniform(-5, 5)
        y = max(5, min(45, y))
    momentum_path = "M" + " L".join(pts)
    
    # Dynamic Remainder Matrix
    matrix = []
    for h in range(4):
        row = []
        for a in range(4):
            base = 0.05
            if h > a: base += p1 * 0.15
            elif a > h: base += p2 * 0.15
            else: base += px * 0.15
            base = base / (h+a+1)
            row.append(base)
        matrix.append(row)
        
    total = sum(sum(r) for r in matrix)
    matrix = [[round((c/total)*100, 0) for c in r] for r in matrix]

    # Dynamic xG Events Log
    events = [
        {"time": "15:12", "detail": f"{home_name} - Shot on Target, {round(random.uniform(0.1, 0.8),2)} xG"},
        {"time": "32:45", "detail": f"{away_name} - Blocked Shot, {round(random.uniform(0.1, 0.4),2)} xG"},
        {"time": "42:10", "detail": f"{home_name} - Goal, {round(random.uniform(0.5, 0.9),2)} xG"},
        {"time": "55:05", "detail": f"{away_name} - Missed Header, {round(random.uniform(0.2, 0.6),2)} xG"},
        {"time": "68:45", "detail": f"{away_name} - Shot on Target, {round(random.uniform(0.3, 0.9),2)} xG"}
    ]

    ctx = {
        "request": request,
        "featured": {
            "home_name": home_name,
            "away_name": away_name,
            "home_score": 1,
            "away_score": 2,
            "minute": 68,
            "h_alpha": 1.65, "h_beta": 0.92,
            "a_alpha": 1.88, "a_beta": 0.75,
            "svg_pitch": generate_pitch_svg_horizontal(
                home_formation="4-3-3",
                away_formation="4-2-3-1",
                home_color=h_theme["primary"],
                away_color=a_theme["primary"],
                home_team=home_name,
                away_team=away_name
            )
        },
        "prior": prior,
        "likelihood": likelihood,
        "posterior": posterior,
        "momentum_path": momentum_path,
        "matrix": matrix,
        "events": events
    }
    return templates.TemplateResponse(request=request, name="match.html", context=ctx)


@app.get("/player", response_class=HTMLResponse)
async def player(request: Request):
    name = request.query_params.get("name", "Player")
    team = request.query_params.get("team", "Unknown")
    html = f"""
    <div style='background:#0b0f19; color:white; height:100vh; font-family:monospace; padding:2rem;'>
        <h1 style='color:#14b8a6;'>PLAYER LOG: {name} ({team})</h1>
        <p>Quantitative Profile Syncing...</p>
        <br><br>
        <a href='javascript:history.back()' style='color:#38bdf8; text-decoration:none;'>&larr; Return to Active Dashboard</a>
    </div>
    """
    return HTMLResponse(html)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
