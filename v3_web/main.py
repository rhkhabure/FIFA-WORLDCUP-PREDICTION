from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from utils import TEAM_THEMES, LEAGUE_TEAMS, TEAM_MANAGERS, get_theme_for_team, generate_pitch_svg_horizontal, generate_pitch_svg_vertical

app = FastAPI(title="V3 Universal Football Model")
templates = Jinja2Templates(directory="templates")

def get_common_context(request: Request):
    league = request.query_params.get("league", "Premier League")
    if league not in LEAGUE_TEAMS:
        league = "Premier League"
    
    available_teams = LEAGUE_TEAMS[league]
    team = request.query_params.get("team", "Default")
    
    if team not in available_teams and team != "Default":
        team = "Default"

    theme = get_theme_for_team(team)
    
    return {
        "request": request,
        "current_league": league,
        "current_team": team,
        "available_teams": available_teams,
        "leagues": list(LEAGUE_TEAMS.keys()),
        "theme": theme
    }

def get_real_players(team):
    # Expanded real player list for UI testing
    rosters = {
        "Liverpool": ["Alisson", "Alexander-Arnold", "Konate", "Van Dijk", "Robertson", "Mac Allister", "Szoboszlai", "Diaz", "Salah", "Nunez", "Gakpo"],
        "Manchester City": ["Ederson", "Walker", "Dias", "Akanji", "Gvardiol", "Rodri", "De Bruyne", "Silva", "Foden", "Haaland", "Grealish"],
        "Arsenal": ["Raya", "White", "Saliba", "Gabriel", "Zinchenko", "Rice", "Odegaard", "Saka", "Martinelli", "Jesus", "Trossard"],
        "Real Madrid": ["Courtois", "Carvajal", "Militao", "Rudiger", "Mendy", "Tchouameni", "Valverde", "Bellingham", "Vinicius", "Rodrygo", "Mbappe"],
        "Barcelona": ["Ter Stegen", "Kounde", "Araujo", "Christensen", "Balde", "De Jong", "Pedri", "Gundogan", "Raphinha", "Lewandowski", "Yamal"],
        "Bayern Munich": ["Neuer", "Mazraoui", "Upamecano", "Kim", "Davies", "Kimmich", "Goretzka", "Sane", "Musiala", "Coman", "Kane"]
    }
    return rosters.get(team, [f"{team[:3]}{i}" for i in range(11)])


@app.get("/", response_class=HTMLResponse)
async def hub(request: Request):
    ctx = get_common_context(request)
    ctx["active_page"] = "hub"
    
    league = ctx["current_league"]
    avail = ctx["available_teams"]
    
    # Pick a dynamic feature match based on the league
    if league == "Premier League":
        home, away = "Manchester City", "Arsenal"
    elif league == "La Liga":
        home, away = "Real Madrid", "Barcelona"
    elif league == "Serie A":
        home, away = "Inter Milan", "Juventus"
    elif league == "Bundesliga":
        home, away = "Bayern Munich", "Bayer Leverkusen"
    elif league == "Ligue 1":
        home, away = "Paris SG", "Marseille"
    else:
        home, away = avail[0], avail[1]
        
    ctx["match_home"] = home
    ctx["match_away"] = away
    
    h_color = get_theme_for_team(home)["primary"]
    a_color = get_theme_for_team(away)["primary"]
    
    home_players = get_real_players(home)
    away_players = get_real_players(away)
    
    ctx["pitch_svg"] = generate_pitch_svg_horizontal(
        home_formation="4-3-3", 
        away_formation="4-2-3-1", 
        home_color=h_color, 
        away_color=a_color,
        home_players=home_players,
        away_players=away_players
    )
    
    return templates.TemplateResponse(request=request, name="index.html", context=ctx)

@app.get("/match", response_class=HTMLResponse)
async def match(request: Request):
    ctx = get_common_context(request)
    ctx["active_page"] = "match"
    
    league = ctx["current_league"]
    avail = ctx["available_teams"]
    
    # Pick a dynamic feature match based on the league
    if league == "Premier League":
        home, away = "Manchester City", "Arsenal"
    elif league == "La Liga":
        home, away = "Real Madrid", "Barcelona"
    elif league == "Serie A":
        home, away = "Inter Milan", "Juventus"
    elif league == "Bundesliga":
        home, away = "Bayern Munich", "Bayer Leverkusen"
    elif league == "Ligue 1":
        home, away = "Paris SG", "Marseille"
    else:
        home, away = avail[0], avail[1]
        
    ctx["match_home"] = home
    ctx["match_away"] = away
    
    ctx["home_manager"] = TEAM_MANAGERS.get(home, "Head Coach")
    ctx["away_manager"] = TEAM_MANAGERS.get(away, "Head Coach")
    
    h_color = get_theme_for_team(home)["primary"]
    a_color = get_theme_for_team(away)["primary"]
    
    home_players = get_real_players(home)
    away_players = get_real_players(away)

    ctx["pitch_svg"] = generate_pitch_svg_horizontal(
        home_formation="4-3-3", 
        away_formation="4-2-3-1", 
        home_color=h_color, 
        away_color=a_color, 
        home_players=home_players, 
        away_players=away_players
    )
    
    return templates.TemplateResponse(request=request, name="match.html", context=ctx)

@app.get("/team", response_class=HTMLResponse)
async def team(request: Request):
    ctx = get_common_context(request)
    ctx["active_page"] = "team"
    
    team_name = ctx["current_team"]
    if team_name == "Default" and len(ctx["available_teams"]) > 0:
        # If user navigates to team profile but is on Default, fallback gracefully
        team_name = ctx["available_teams"][0]
        ctx["current_team"] = team_name
        ctx["theme"] = get_theme_for_team(team_name)
    
    t_color = ctx["theme"]["primary"]
    home_players = get_real_players(team_name)
    
    ctx["manager"] = TEAM_MANAGERS.get(team_name, "Head Coach")
    
    # Team Profile gets the vertical half-pitch
    ctx["pitch_svg"] = generate_pitch_svg_vertical(
        formation="4-3-3", 
        team_color=t_color, 
        players=home_players
    )
    
    return templates.TemplateResponse(request=request, name="team.html", context=ctx)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
