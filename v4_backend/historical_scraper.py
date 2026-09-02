import sqlite3
import pandas as pd
import soccerdata as sd
from pathlib import Path

DB_PATH = Path("v4_historical_data.sqlite")

# Soccerdata uses specific naming conventions for leagues
TARGET_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "GER-Bundesliga",
    "FRA-Ligue 1"
]

# Last 3 completed seasons + the current ongoing season
TARGET_SEASONS = ["2122", "2223", "2324", "2425"]

def init_db():
    """Initializes the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    return conn

def run_scraper():
    print(f"🚀 Initializing soccerdata scraper for FBref...")
    print(f"Leagues: {TARGET_LEAGUES}")
    print(f"Seasons: {TARGET_SEASONS}\\n")
    
    # Initialize the FBref scraper
    # This automatically handles rate limiting, session management, and caching
    try:
        fbref = sd.FBref(leagues=TARGET_LEAGUES, seasons=TARGET_SEASONS)
    except Exception as e:
        print(f"❌ Failed to initialize soccerdata: {e}")
        return

    print("⏳ Downloading schedule and match logs (This may take a moment on the first run to build the cache)...")
    
    try:
        # Pulls the entire fixture list, including xG, for all defined leagues and seasons in one go
        df_schedule = fbref.read_schedule()
        
        # Flatten the MultiIndex (league, season, game)
        df = df_schedule.reset_index()
        
        # FBref schedule data columns typically look like:
        # ['league', 'season', 'game', 'date', 'time', 'home_team', 'home_xg', 'score', 'away_xg', 'away_team', 'attendance', 'venue', 'referee', 'match_report', 'notes']
        
        # Filter out matches that haven't happened yet (they won't have a score or xG)
        df_finished = df.dropna(subset=['score', 'home_xg', 'away_xg']).copy()
        
        # Split the 'score' column (e.g., "2–1") into home and away goals
        # Note: FBref uses an en-dash '–', not a standard hyphen '-'
        df_finished[['home_goals', 'away_goals']] = df_finished['score'].str.split('–|-', expand=True).astype(float)
        
        # Standardize column names for our V4 model
        df_clean = df_finished[[
            'league', 'season', 'date', 
            'home_team', 'away_team', 
            'home_goals', 'away_goals', 
            'home_xg', 'away_xg'
        ]].copy()
        
        # Convert date to datetime
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        print(f"✅ Successfully extracted {len(df_clean)} historical matches with continuous xG!")
        
        # Save straight to SQLite
        conn = init_db()
        print("💾 Saving to database...")
        df_clean.to_sql('matches_xg', conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"🎉 Complete! Historical ledger saved to {DB_PATH}")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")

if __name__ == "__main__":
    print("Running soccerdata scraper to bulk-download FBref xG data into SQLite...")
    run_scraper()
