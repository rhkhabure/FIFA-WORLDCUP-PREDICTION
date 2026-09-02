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
        
        # Determine actual column names (soccerdata sometimes names them 'xG' and 'xG.1' or 'home_xg' and 'away_xg')
        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower == 'xg':
                col_map[col] = 'home_xg'
            elif col_lower in ['xg.1', 'xg_1', 'away_xg']:
                col_map[col] = 'away_xg'
            elif col_lower in ['home_team', 'home team']:
                col_map[col] = 'home_team'
            elif col_lower in ['away_team', 'away team']:
                col_map[col] = 'away_team'
            elif col_lower == 'score':
                col_map[col] = 'score'
                
        df = df.rename(columns=col_map)
        
        # Fallback if the renaming didn't catch it
        if 'home_xg' not in df.columns and 'xG' in df_schedule.columns:
            # MultiIndex columns sometimes cause issues, let's just make sure we grab them
            pass
            
        print("Available columns parsed:", df.columns.tolist())
        
        # Filter out matches that haven't happened yet
        df_finished = df.dropna(subset=['score']).copy()
        
        if 'home_xg' not in df_finished.columns or 'away_xg' not in df_finished.columns:
            print("❌ Warning: xG columns missing from dataset. Check if the seasons selected have xG data available in FBref.")
            return

        # Split the 'score' column (e.g., "2–1") into home and away goals
        # Note: FBref uses an en-dash '–' or standard hyphen '-'
        df_finished[['home_goals', 'away_goals']] = df_finished['score'].astype(str).str.split('–|-', expand=True).astype(float)
        
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
