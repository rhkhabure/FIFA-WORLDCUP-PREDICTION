from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from utils import TEAM_THEMES, LEAGUE_TEAMS, generate_pitch_svg

app = FastAPI(title="V3 Universal Football Model")
templates = Jinja2Templates(directory="templates")

def get_common_context(request: Request):
    league = request.query_params.get("league", "Premier League")
    # ensure safe fallback
    if league not in LEAGUE_TEAMS:
        league = "Premier League"
    
    available_teams = LEAGUE_TEAMS[league]
    team = request.query_params.get("team", "Default")
    
    if team not in available_teams and team != "Default":
        team = "Default"
        
    # Snap to the first team if "Default" is selected but we want colors to pop
    if team == "Default" and available_teams:
        # Actually let user select Default if they want, but fallback safely
        pass

    theme = TEAM_THEMES.get(team, TEAM_THEMES["Default"])
    
    return {
        "request": request,
        "current_league": league,
        "current_team": team,
        "available_teams": available_teams,
        "leagues": list(LEAGUE_TEAMS.keys()),
        "theme": theme
    }

@app.get("/", response_class=HTMLResponse)
async def hub(request: Request):
    ctx = get_common_context(request)
    ctx["active_page"] = "hub"
    
    # Determine "Match of the week"
    league = ctx["current_league"]
    avail = ctx["available_teams"]
    if league == "Premier League":
        home, away = "Manchester City", "Arsenal"
    elif league == "La Liga":
        home, away = "Real Madrid", "Barcelona"
    else:
        home, away = avail[0], avail[-1] if len(avail) > 1 else "Default"
        
    ctx["match_home"] = home
    ctx["match_away"] = away
    
    h_color = TEAM_THEMES.get(home, TEAM_THEMES["Default"])["primary"]
    a_color = TEAM_THEMES.get(away, TEAM_THEMES["Default"])["primary"]
    
    ctx["pitch_svg"] = generate_pitch_svg("4-3-3", "4-2-3-1", h_color, a_color)
    
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)

@app.get("/match", response_class=HTMLResponse)
async def match(request: Request):
    ctx = get_common_context(request)
    ctx["active_page"] = "match"
    
    league = ctx["current_league"]
    avail = ctx["available_teams"]
    if league == "Premier League":
        home, away = "Manchester City", "Arsenal"
    elif league == "La Liga":
        home, away = "Real Madrid", "Barcelona"
    else:
        home, away = avail[0], avail[-1] if len(avail) > 1 else "Default"
        
    ctx["match_home"] = home
    ctx["match_away"] = away
    
    h_color = TEAM_THEMES.get(home, TEAM_THEMES["Default"])["primary"]
    a_color = TEAM_THEMES.get(away, TEAM_THEMES["Default"])["primary"]
    
    home_players = ["Ederson", "Walker", "Dias", "Akanji", "Gvardiol", "Rodri", "De Bruyne", "Silva", "Foden", "Haaland", "Grealish"]
    away_players = ["Raya", "White", "Saliba", "Gabriel", "Zinchenko", "Rice", "Odegaard", "Saka", "Martinelli", "Jesus", "Trossard"]

    ctx["pitch_svg"] = generate_pitch_svg("4-3-3", "4-2-3-1", h_color, a_color, home_players, away_players)
    
    return templates.TemplateResponse(request=request, name="match.html", context=ctx)

@app.get("/team", response_class=HTMLResponse)
async def team(request: Request):
    ctx = get_common_context(request)
    ctx["active_page"] = "team"
    
    t_color = ctx["theme"]["primary"]
    
    # Render half pitch for just the home team standard XI
    ctx["pitch_svg"] = generate_pitch_svg("4-3-3", "0-0", t_color, "#000000")
    
    return templates.TemplateResponse(request=request, name="team.html", context=ctx)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
