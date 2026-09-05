import os
import json
import urllib.request
import urllib.error

# Since dotenv isn't in requirements.txt and this environment might not have it installed natively,
# we'll read the .env file manually.
def get_api_key():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    try:
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith("FOOTBALLDATA_API_KEY="):
                    return line.strip().split("=")[1]
    except Exception:
        pass
    return ""

API_KEY = get_api_key()
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

def get_last_completed_pl_match():
    """
    Fetches the most recently completed match in the Premier League (league_id = 15).
    Useful for populating the Match Dashboard when no active ID is supplied.
    """
    res = fetch_api("leagues/15/matches")
    if res and res.get("success"):
        data = res.get("data", [])
        if isinstance(data, list):
            # Sort or filter for finished matches. Usually the array is chronological or reverse chronological.
            finished = [m for m in data if m.get("status") in ["complete", "Finished", "FT"]]
            if finished:
                return finished[-1].get("match_id") # Assuming chronological, [-1] is the most recent.
                
    # Fallback to a hardcoded known match if API fails to retrieve the list
    return 780100645

def get_live_match_data(match_id):
    """
    Fetches base match info, stats, and events, then parses it into V4 model format.
    """
    match_data = fetch_api(f"matches/{match_id}")
    stats_data = fetch_api(f"matches/{match_id}/stats")
    events_data = fetch_api(f"matches/{match_id}/events")
    
    match_info = match_data.get("data", {}).get("match", {})
    if not match_info:
        match_info = stats_data.get("data", {}).get("match", {})

    home_team = match_info.get("home_team", {}).get("team_name", "Unknown Home")
    away_team = match_info.get("away_team", {}).get("team_name", "Unknown Away")
    
    home_score = int(match_info.get("home_score", 0) or match_info.get("goals_home", 0) or 0)
    away_score = int(match_info.get("away_score", 0) or match_info.get("goals_away", 0) or 0)
    
    red_cards = {"home": 0, "away": 0}
    parsed_events = []
    
    events = events_data.get("data", {}).get("events", []) if events_data else []
    for event in events:
        side = event.get("team_side") 
        etype = str(event.get("event_type", "")).lower()
        minute = event.get("minute", "?")
        detail = event.get("detail", etype.replace("_", " ").title())
        player = event.get("player", {}).get("player_name", "Unknown")
        team_name = home_team if side == "home" else away_team
        
        parsed_events.append({"time": f"{minute}'", "detail": f"{player} ({team_name}) - {detail}"})
        
        if etype == "red_card":
            if side == "home": red_cards["home"] += 1
            elif side == "away": red_cards["away"] += 1

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
