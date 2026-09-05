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
from footballdata import get_live_match_data
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v4_backend.in_play_posterior import generate_live_in_play_odds
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

    # 1. Generate Scatter Plot Data (Using loaded Priors)
    scatter_data = []
    league_priors = priors_db.get(current_league_name, {}).get("teams", {})
    if not league_priors:
        league_priors = priors_db.get(f"ENG-{current_league_name}", {}).get("teams", {})

    for t in teams:
        a = league_priors.get(t, {}).get("alpha", random.uniform(0.7, 1.4))
        b = league_priors.get(t, {}).get("beta", random.uniform(0.7, 1.4))
        scatter_data.append({
            "team": t, 
            "short": t[:3].upper(), 
            "alpha": round(a, 2), 
            "beta": round(b, 2)
        })

    # 2. Fetch real live fixtures
    from footballdata import fetch_api
    today_data = fetch_api("fixtures/today")
    match_list = []
    if today_data and today_data.get("success") and "data" in today_data:
        d = today_data["data"]
        if isinstance(d, list): match_list = d
        elif isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, list): match_list.extend(v)
                elif isinstance(v, dict) and "matches" in v: match_list.extend(v["matches"])

    fixtures = []
    if not match_list:
        # Fallback to simulated fixtures if empty
        for i in range(0, min(len(teams), 10), 2):
            if i+1 < len(teams):
                h, a = teams[i], teams[i+1]
                ha, hb = random.uniform(1.0, 1.4), random.uniform(0.7, 1.1)
                aa, ab = random.uniform(0.9, 1.3), random.uniform(0.8, 1.2)
                p1, px, p2 = mock_probs(ha, ab, aa, hb)
                fixtures.append({
                    "match_id": None, "home": h, "away": a,
                    "h_alpha": ha, "h_beta": hb, "a_alpha": aa, "a_beta": ab,
                    "prob_1": round(p1*100, 1), "prob_x": round(px*100, 1), "prob_2": round(p2*100, 1),
                    "odds_1": round(1/p1, 2) if p1>0 else 0, "odds_x": round(1/px, 2) if px>0 else 0, "odds_2": round(1/p2, 2) if p2>0 else 0,
                    "ev": "HOME +EV" if p1 > 0.45 else ("AWAY +EV" if p2 > 0.4 else "NO EDGE")
                })
    else:
        for m in match_list[:10]:
            h = m.get("home_team", {}).get("team_name", "Home")
            a = m.get("away_team", {}).get("team_name", "Away")
            mid = m.get("match_id")
            
            ha = league_priors.get(h, {}).get("alpha", 1.1)
            hb = league_priors.get(h, {}).get("beta", 0.9)
            aa = league_priors.get(a, {}).get("alpha", 1.0)
            ab = league_priors.get(a, {}).get("beta", 1.0)
            
            p1, px, p2 = mock_probs(ha, ab, aa, hb)
            fixtures.append({
                "match_id": mid, "home": h, "away": a,
                "h_alpha": ha, "h_beta": hb, "a_alpha": aa, "a_beta": ab,
                "prob_1": round(p1*100, 1), "prob_x": round(px*100, 1), "prob_2": round(p2*100, 1),
                "odds_1": round(1/p1, 2) if p1>0 else 0, "odds_x": round(1/px, 2) if px>0 else 0, "odds_2": round(1/p2, 2) if p2>0 else 0,
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
    match_id = request.query_params.get("match_id")
    home_name_fallback = request.query_params.get("home", "Arsenal")
    away_name_fallback = request.query_params.get("away", "Manchester City")
    
    # 1. Fetch live match data if we have an ID, else mock it
    if match_id:
        live_data = get_live_match_data(match_id)
        home_name = live_data["home_team"] or home_name_fallback
        away_name = live_data["away_team"] or away_name_fallback
        minute = live_data["current_minute"]
        h_score = live_data["home_score"]
        a_score = live_data["away_score"]
        h_xg = live_data["live_xg"]["home"]
        a_xg = live_data["live_xg"]["away"]
        events = live_data["events"]
    else:
        # Fallback to simulated defaults if no match_id provided
        home_name = home_name_fallback
        away_name = away_name_fallback
        minute = 68
        h_score, a_score = 1, 2
        h_xg, a_xg = 0.8, 1.4
        events = [{"time": "15'", "detail": "Test Event - No match ID provided"}]

    h_theme = get_theme_for_team(home_name)
    a_theme = get_theme_for_team(away_name)
    
    # Grab priors from DB
    league_priors = priors_db.get("Premier League", {}).get("teams", {})
    h_alpha = league_priors.get(home_name, {}).get('alpha', 1.42)
    a_beta = league_priors.get(away_name, {}).get('beta', 0.88)
    a_alpha = league_priors.get(away_name, {}).get('alpha', 1.05)
    h_beta = league_priors.get(home_name, {}).get('beta', 0.78)

    # Base Probabilities (Prior via mock fallback for UI)
    p1, px, p2 = mock_probs(h_alpha, a_beta, a_alpha, h_beta)
    prior = [round(p1*100, 1), round(px*100, 1), round(p2*100, 1)]
    likelihood = [max(0, prior[0]-5), prior[1]+1, min(100, prior[2]+4)]
    
    # 🔥 LIVE POSTERIOR LOGIC CALLED 🔥
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
        gamma=1.0, # static gamma assumption
        rho=0.0
    )
    
    pos_probs = posterior_res["live_probabilities"]
    posterior = [round(pos_probs["1"]*100, 1), round(pos_probs["X"]*100, 1), round(pos_probs["2"]*100, 1)]

    # Dynamic Match Momentum Path (Fake chart based on xG differential)
    pts = []
    x_step = 100 / 20
    x, y = 0, 25
    for i in range(21):
        pts.append(f"{x},{y}")
        x += x_step
        y += random.uniform(-5, 5) + (h_xg - a_xg)
        y = max(5, min(45, y))
    momentum_path = "M" + " L".join(pts)
    
    # Dynamic Remainder Matrix
    matrix = []
    for h in range(4):
        row = []
        for a in range(4):
            base = 0.05
            if h > a: base += pos_probs["1"] * 0.15
            elif a > h: base += pos_probs["2"] * 0.15
            else: base += pos_probs["X"] * 0.15
            base = base / (h+a+1)
            row.append(base)
        matrix.append(row)
        
    total = sum(sum(r) for r in matrix)
    matrix = [[round((c/total)*100, 0) for c in r] for r in matrix]

    ctx = {
        "request": request,
        "current_league": "Premier League",
        "featured": {
            "home_name": home_name,
            "away_name": away_name,
            "home_score": h_score,
            "away_score": a_score,
            "minute": minute,
            "h_alpha": round(h_alpha, 2), "h_beta": round(h_beta, 2),
            "a_alpha": round(a_alpha, 2), "a_beta": round(a_beta, 2),
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
        "events": events[-10:] # Keep last 10 events for UI
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
