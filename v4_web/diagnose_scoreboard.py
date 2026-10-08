"""
diagnose_scoreboard.py  —  ScoreboardV3 column verification
Run from v4_web/ with the nba_ml conda env:
    conda activate nba_ml
    python diagnose_scoreboard.py
"""
from datetime import date, timedelta
from nba_api.stats.endpoints import scoreboardv3
import time

def parse_v3(board):
    gh = board.game_header.get_data_frame()
    ls = board.line_score.get_data_frame()
    games = []
    for _, row in gh.iterrows():
        gid = str(row.get("gameId",""))
        gl  = ls[ls["gameId"]==gid]
        away_r = gl.iloc[0] if len(gl)>=1 else {}
        home_r = gl.iloc[1] if len(gl)>=2 else {}
        def g(r,c,d=""): 
            try: v=r[c]; return d if v!=v else v
            except: return d
        games.append({
            "game_id":     gid,
            "game_code":   row.get("gameCode",""),
            "status_text": row.get("gameStatusText",""),
            "game_time_utc": row.get("gameTimeUTC",""),
            "home_tricode": g(home_r,"teamTricode"),
            "away_tricode": g(away_r,"teamTricode"),
            "home_id":     g(home_r,"teamId",0),
            "away_id":     g(away_r,"teamId",0),
            "home_score":  g(home_r,"score",0),
            "away_score":  g(away_r,"score",0),
            "home_wins":   g(home_r,"wins",0),
            "home_losses": g(home_r,"losses",0),
        })
    return games

print("=== Tomorrow's games ===")
tomorrow = date.today() + timedelta(days=1)
ds = f"{tomorrow.month:02d}/{tomorrow.day:02d}/{tomorrow.year}"
time.sleep(0.6)
try:
    games = parse_v3(scoreboardv3.ScoreboardV3(game_date=ds, league_id="00"))
    for g in games:
        print(f"  {g['game_id']}  {g['away_tricode']}({g['away_id']}) @ {g['home_tricode']}({g['home_id']})  {g['status_text']}  UTC:{g['game_time_utc']}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Recent preseason (Oct 3-7) ===")
today = date.today()
for i in range(5, 0, -1):
    d  = today - timedelta(days=i)
    ds = f"{d.month:02d}/{d.day:02d}/{d.year}"
    time.sleep(0.6)
    try:
        games = parse_v3(scoreboardv3.ScoreboardV3(game_date=ds, league_id="00"))
        if not games:
            print(f"  {ds}: no games")
        for g in games:
            print(f"  {ds}  {g['game_id']}  {g['away_tricode']}({g['away_score']}) @ {g['home_tricode']}({g['home_score']})  [{g['status_text']}]")
    except Exception as e:
        print(f"  {ds}: {e}")
