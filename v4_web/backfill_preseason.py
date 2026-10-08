"""
backfill_preseason.py
=====================
Fetches all completed preseason games, runs our Elo pregame prediction
on each one, and inserts results into predictions.db.

Run once from v4_web/ with the nba_ml conda env:
    conda activate nba_ml
    python backfill_preseason.py

Safe to run multiple times — uses INSERT OR IGNORE on game_id.
"""
import sqlite3, sys, time, json, pickle
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent

# ── Load model artefacts ──────────────────────────────────────────────────────
elo_path = ROOT / "nba" / "model" / "elo_ratings_v3.json"
if not elo_path.exists():
    sys.exit(f"ERROR: {elo_path} not found — copy from V3/model/")

with open(elo_path) as f:
    elo_ratings = {str(k): float(v) for k, v in json.load(f).items()}

HOME_ADVANTAGE = 100
ELO_START = 1500

def get_elo(team_id):
    return elo_ratings.get(str(team_id), float(ELO_START))

def pregame_prob(home_id, away_id):
    h = get_elo(home_id) + HOME_ADVANTAGE
    a = get_elo(away_id)
    return 1.0 / (1.0 + 10.0 ** ((a - h) / 400.0))

# ── DB ────────────────────────────────────────────────────────────────────────
db_path = ROOT / "predictions.db"
conn = sqlite3.connect(str(db_path))
conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        league        TEXT    NOT NULL DEFAULT 'football',
        game_id       TEXT    UNIQUE,
        home_team     TEXT,
        away_team     TEXT,
        home_team_id  INTEGER,
        away_team_id  INTEGER,
        conference    TEXT,
        home_win_prob REAL,
        away_win_prob REAL,
        home_score    INTEGER,
        away_score    INTEGER,
        correct       INTEGER,
        brier         REAL,
        created_at    TEXT    DEFAULT (datetime('now'))
    )
""")
conn.commit()

# ── Fetch preseason games via ScoreboardV3 ────────────────────────────────────
from nba_api.stats.endpoints import scoreboardv3

def parse_v3(board):
    gh = board.game_header.get_data_frame()
    ls = board.line_score.get_data_frame()
    games = []
    for _, row in gh.iterrows():
        gid = str(row.get("gameId",""))
        gl  = ls[ls["gameId"]==gid]
        if len(gl) < 2:
            continue
        away_r, home_r = gl.iloc[0], gl.iloc[1]
        def g(r,c,d=0):
            try: v=r[c]; return d if v!=v else v
            except: return d
        status = int(g(row,"gameStatus",1) if hasattr(row,"__getitem__") else row.get("gameStatus",1))
        if status != 3:   # only final games
            continue
        games.append({
            "game_id":      gid,
            "home_team_id": int(g(home_r,"teamId",0)),
            "away_team_id": int(g(away_r,"teamId",0)),
            "home_tricode": str(g(home_r,"teamTricode","")),
            "away_tricode": str(g(away_r,"teamTricode","")),
            "home_score":   int(g(home_r,"score",0)),
            "away_score":   int(g(away_r,"score",0)),
            "game_time_utc":str(row.get("gameTimeUTC","") or ""),
        })
    return games

# Scan preseason window: Oct 1 → yesterday
today     = date.today()
start     = date(today.year, 10, 1)
inserted  = 0
skipped   = 0

print(f"Scanning preseason games {start} → {today - timedelta(days=1)} ...")
print(f"DB: {db_path}")
print(f"Elo ratings loaded: {len(elo_ratings)} teams\n")

d = start
while d < today:
    ds = f"{d.month:02d}/{d.day:02d}/{d.year}"
    time.sleep(0.7)   # polite rate limit
    try:
        board = scoreboardv3.ScoreboardV3(game_date=ds, league_id="00")
        games = parse_v3(board)
        if not games:
            d += timedelta(days=1)
            continue
        for g in games:
            hid  = g["home_team_id"]
            aid  = g["away_team_id"]
            hp   = pregame_prob(hid, aid)
            hs   = g["home_score"]
            as_  = g["away_score"]
            won  = hs > as_
            correct = 1 if (hp >= 0.5) == won else 0
            # Brier = (prob - outcome)^2
            brier = (hp - (1 if won else 0)) ** 2

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO predictions
                    (league, game_id, home_team, away_team, home_team_id, away_team_id,
                     conference, home_win_prob, away_win_prob,
                     home_score, away_score, correct, brier, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    "nba", g["game_id"],
                    g["home_tricode"], g["away_tricode"],
                    hid, aid,
                    "Preseason",
                    round(hp, 4), round(1-hp, 4),
                    hs, as_,
                    correct,
                    round(brier, 6),
                    g["game_time_utc"][:10] if g["game_time_utc"] else ds,
                ))
                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    result = "✓" if correct else "✗"
                    print(f"  {ds}  {g['away_tricode']}({as_}) @ {g['home_tricode']}({hs})"
                          f"  pred={hp:.1%}  {result}  Brier={brier:.4f}")
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  DB insert error: {e}")

        conn.commit()
    except Exception as e:
        print(f"  {ds}: fetch error — {e}")

    d += timedelta(days=1)

# Summary
total = conn.execute("SELECT COUNT(*) FROM predictions WHERE league='nba'").fetchone()[0]
right = conn.execute("SELECT SUM(correct) FROM predictions WHERE league='nba'").fetchone()[0] or 0
avg_b = conn.execute("SELECT AVG(brier) FROM predictions WHERE league='nba'").fetchone()[0] or 0

print(f"\n{'='*50}")
print(f"Inserted : {inserted} new predictions")
print(f"Skipped  : {skipped} already in DB")
print(f"Total NBA: {total} predictions")
print(f"Accuracy : {right/total*100:.1f}% ({right}/{total})" if total else "Accuracy: —")
print(f"Avg Brier: {avg_b:.4f}" if total else "Avg Brier: —")
conn.close()
