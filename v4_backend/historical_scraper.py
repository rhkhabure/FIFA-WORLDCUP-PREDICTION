import sqlite3
import requests
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load API Key
load_dotenv(Path.cwd().parent / ".env")
API_KEY = os.getenv("SOFASCORE_API_KEY", "2d8a002cf8mshf22bca7802e285bp1bfac6jsncd0235ec1f66")
HOST = "sofascore.p.rapidapi.com"

# Big 5 Tournaments in Sofascore
TOURNAMENTS = {
    "Premier League": 17,
    "La Liga": 8,
    "Serie A": 23,
    "Bundesliga": 35,
    "Ligue 1": 34
}

# Number of seasons to scrape back (e.g., 3 means 23/24, 22/23, 21/22)
TARGET_SEASONS = 3

DB_PATH = Path("v4_historical_data.sqlite")

def init_db():
    """Initializes the SQLite database with the exact ledger schema needed for V4."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY,
            tournament_id INTEGER,
            season_id INTEGER,
            start_timestamp INTEGER,
            home_team_id INTEGER,
            home_team_name TEXT,
            away_team_id INTEGER,
            away_team_name TEXT,
            home_goals INTEGER,
            away_goals INTEGER,
            home_xg REAL,
            away_xg REAL,
            status TEXT
        )
    ''')
    conn.commit()
    return conn

def fetch_api(endpoint):
    """Fetches data from RapidAPI, handling rate limits via simple sleep."""
    url = f"https://{HOST}/{endpoint}"
    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': HOST
    }
    
    # We must pace ourselves to not blow past the free tier limits instantly
    time.sleep(1.5) 
    
    try:
        import urllib3
        urllib3.disable_warnings()
        response = requests.get(url, headers=headers, verify=False)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print("⚠️ RATE LIMIT REACHED! Pausing for 60 seconds...")
            time.sleep(60)
            return fetch_api(endpoint) # Retry
        else:
            print(f"API Error: {response.status_code} on {endpoint}")
    except Exception as e:
        print(f"Exception fetching {endpoint}: {e}")
    return None

def extract_xg_from_statistics(match_id):
    """Hits the /get-statistics endpoint and extracts expected goals (xG)."""
    stats_data = fetch_api(f"matches/get-statistics?matchId={match_id}")
    home_xg, away_xg = 0.0, 0.0
    
    if not stats_data or 'statistics' not in stats_data:
        return None, None
        
    for period in stats_data['statistics']:
        if period.get('period') == 'ALL':
            for group in period.get('groups', []):
                for item in group.get('statisticsItems', []):
                    if item.get('name') == 'Expected goals':
                        # API returns strings like "1.24"
                        try:
                            home_xg = float(item.get('home', 0))
                            away_xg = float(item.get('away', 0))
                        except ValueError:
                            pass
                        return home_xg, away_xg
    return None, None

def run_scraper():
    conn = init_db()
    cursor = conn.cursor()
    
    for league_name, t_id in TOURNAMENTS.items():
        print(f"\\n🏆 Processing {league_name} (ID: {t_id})")
        
        # 1. Get the last N seasons
        seasons_data = fetch_api(f"tournaments/get-seasons?tournamentId={t_id}")
        if not seasons_data or 'seasons' not in seasons_data:
            continue
            
        # Typically the first one is the active season, we want the historical ones.
        # So we skip index 0 (current) and grab the next TARGET_SEASONS
        target_season_ids = [s['id'] for s in seasons_data['seasons'][1:TARGET_SEASONS+1]]
        
        for s_id in target_season_ids:
            print(f"  📅 Fetching Season {s_id}")
            
            # Loop through pages of events. Usually a 380-game season is split across ~4-5 pages.
            page = 0
            while True:
                events_data = fetch_api(f"tournaments/get-events?tournamentId={t_id}&seasonId={s_id}&page={page}")
                if not events_data or 'events' not in events_data or len(events_data['events']) == 0:
                    break # No more pages
                
                events = events_data['events']
                for match in events:
                    if match.get('status', {}).get('type') != 'finished':
                        continue # Skip abandoned/postponed
                        
                    m_id = match['id']
                    
                    # CHECKPOINTING: Skip if we already scraped this match
                    cursor.execute("SELECT 1 FROM matches WHERE match_id = ?", (m_id,))
                    if cursor.fetchone():
                        continue 
                        
                    # Extract basic match details
                    start_ts = match.get('startTimestamp', 0)
                    h_team_id = match['homeTeam']['id']
                    h_team_name = match['homeTeam']['name']
                    a_team_id = match['awayTeam']['id']
                    a_team_name = match['awayTeam']['name']
                    
                    h_goals = match['homeScore'].get('current', 0)
                    a_goals = match['awayScore'].get('current', 0)
                    
                    # 2. Extract xG (This costs 1 API call per match!)
                    home_xg, away_xg = extract_xg_from_statistics(m_id)
                    
                    if home_xg is not None and away_xg is not None:
                        # Save to database
                        cursor.execute('''
                            INSERT INTO matches 
                            (match_id, tournament_id, season_id, start_timestamp, 
                             home_team_id, home_team_name, away_team_id, away_team_name, 
                             home_goals, away_goals, home_xg, away_xg, status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (m_id, t_id, s_id, start_ts, h_team_id, h_team_name, a_team_id, a_team_name, h_goals, a_goals, home_xg, away_xg, "scraped"))
                        
                        conn.commit()
                        print(f"      ✅ Saved: {h_team_name} {h_goals}-{a_goals} {a_team_name} (xG: {home_xg}-{away_xg})")
                    else:
                        print(f"      ⚠️ No xG data for match {m_id}")
                
                page += 1 # Next page of matches
                
    conn.close()
    print("\\n🎉 Scraping sequence finished or paused.")

if __name__ == "__main__":
    print("Run this locally to build the Historical v4_historical_data.sqlite database.")
    # uncomment to run:
    # run_scraper()