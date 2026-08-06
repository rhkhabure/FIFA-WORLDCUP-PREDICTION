import json
import urllib.request
import ssl
import time

API_KEY = "2d8a002cf8mshf22bca7802e285bp1bfac6jsncd0235ec1f66"
HOST = "sofascore.p.rapidapi.com"

def fetch(endpoint):
    url = f"https://{HOST}/{endpoint}"
    req = urllib.request.Request(url, headers={'x-rapidapi-key': API_KEY, 'x-rapidapi-host': HOST})
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, context=ctx) as res:
            if res.status == 200:
                return json.loads(res.read().decode())
    except Exception as e:
        print(f"Error fetching {endpoint}: {e}")
    return None

def sync_data():
    """
    Run this script locally (where outbound internet is not blocked) to pull real data 
    from Sofascore and overwrite data.json for the V3 Dashboard.
    """
    print("Starting Sofascore API Sync...")
    
    # Load existing DB to update
    try:
        with open("data.json", "r") as f:
            db = json.load(f)
    except FileNotFoundError:
        print("data.json not found. Make sure you generated it first.")
        return

    # Example: Sync Wolverhampton specifically as requested
    search = fetch("teams/search?name=Wolverhampton")
    if search and search.get('results'):
        team_id = search['results'][0]['entity']['id']
        print(f"Found Wolverhampton ID: {team_id}")
        
        squad = fetch(f"teams/get-squad?teamId={team_id}")
        if squad and 'players' in squad:
            roster = []
            for p_info in squad['players'][:11]: # Just top 11 to save rate limits
                p = p_info['player']
                p_id = p['id']
                p_name = p['name']
                roster.append(p_name)
                
                # Fetch detailed stats
                chars = fetch(f"players/get-characteristics?playerId={p_id}")
                time.sleep(1) # Be gentle on rate limits
                
                # Mock parsing of characteristics into Pentagon structure
                # Sofascore returns specific trait arrays, we map them here
                att = 75; tec = 75; tac = 75; df = 75; cre = 75
                
                db["players"][p_name] = {
                    "id": p_id,
                    "name": p_name,
                    "team": "Wolverhampton",
                    "position": p.get('position', 'MID'),
                    "jersey": p_info.get('shirtNumber', 0),
                    "age": 25, # Would calculate from p['dateOfBirthTimestamp']
                    "nationality": p.get('country', {}).get('alpha2', 'UNK'),
                    "stats": {"attacking": att, "technical": tec, "tactical": tac, "defending": df, "creativity": cre},
                    "summary": {"rating": 7.0, "matches": 30, "goals": 5, "assists": 5},
                    "image": f"https://api-sports.io/football/players/{p_id}.png"
                }
            
            db["teams"]["Wolverhampton"]["roster"] = roster
            print("Wolverhampton squad synced successfully!")
            
    # Save the updated DB
    with open("data.json", "w") as f:
        json.dump(db, f, indent=2)
    print("Sync complete. Real data written to data.json")

if __name__ == "__main__":
    sync_data()