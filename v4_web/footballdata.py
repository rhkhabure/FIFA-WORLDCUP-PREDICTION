import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

# Load key from .env (in the parent dir)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
API_KEY = os.getenv("FOOTBALLDATA_API_KEY", "")
BASE_URL = "https://footballdata.io/api/v1"

def fetch_api(endpoint):
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Footballdata API Error fetching {endpoint}: {e}")
        return {}

def get_live_match_data(match_id):
    """
    Fetches base match info, stats, and events, then parses it into V4 model format.
    """
    match_data = fetch_api(f"matches/{match_id}")
    stats_data = fetch_api(f"matches/{match_id}/stats")
    events_data = fetch_api(f"matches/{match_id}/events")
    
    match_info = match_data.get("data", {}).get("match", {})
    if not match_info:
        # Sometimes it's nested in stats_data instead if match_data fails
        match_info = stats_data.get("data", {}).get("match", {})

    home_team = match_info.get("home_team", {}).get("team_name", "Unknown Home")
    away_team = match_info.get("away_team", {}).get("team_name", "Unknown Away")
    
    # Authoritative Goals
    home_score = int(match_info.get("home_score", 0) or match_info.get("goals_home", 0) or 0)
    away_score = int(match_info.get("away_score", 0) or match_info.get("goals_away", 0) or 0)
    
    # Parse Events for Red Cards & timeline log
    red_cards = {"home": 0, "away": 0}
    parsed_events = []
    
    events = events_data.get("data", {}).get("events", []) if events_data else []
    for event in events:
        side = event.get("team_side") # 'home' or 'away'
        etype = str(event.get("event_type", "")).lower()
        minute = event.get("minute", "?")
        detail = event.get("detail", etype.replace("_", " ").title())
        player = event.get("player", {}).get("player_name", "Unknown")
        team_name = home_team if side == "home" else away_team
        
        parsed_events.append({"time": f"{minute}'", "detail": f"{player} ({team_name}) - {detail}"})
        
        if etype == "red_card":
            if side == "home": red_cards["home"] += 1
            elif side == "away": red_cards["away"] += 1

    # Parse Stats (Live xG, etc.)
    live_xg = {"home": 0.0, "away": 0.0}
    stats_dict = stats_data.get("data", {}).get("stats", {})
    
    if isinstance(stats_dict, dict) and "xg" in stats_dict:
        live_xg["home"] = float(stats_dict["xg"].get("home", 0.0))
        live_xg["away"] = float(stats_dict["xg"].get("away", 0.0))
        
    status = match_info.get("status", "")
    current_minute = 90 if status in ["complete", "Finished", "FT"] else match_info.get("minute", "Live")
    if current_minute is None: current_minute = 0
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "current_minute": current_minute,
        "red_cards": red_cards,
        "live_xg": live_xg,
        "events": parsed_events,
        "status": status
    }
