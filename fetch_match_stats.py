"""
fetch_match_stats.py
=====================
Run this ONCE (and again any time new matches finish) to build a local
cache of stats for every completed match. The History page reads this
cache — it never fetches stats itself, keeping the dashboard simple.

Usage:
    python fetch_match_stats.py

What's in the cache right now: goal scorers with names and minutes, pulled
straight from the same free worldcup26.ir feed the dashboard already uses.
No new API, no rate limits, no key needed.

What's NOT in it yet: shots, possession, cards, ratings — that would need
API-Football (or similar), which has its own free-tier limits. This script
is the natural place to add that later without touching the dashboard code
at all — just add another field to each match's saved record below.
"""

import json
from pathlib import Path

import common as c

def main():
    print("Fetching games and teams from worldcup26.ir...")
    games = c.fetch_wc26("games")
    teams = c.fetch_wc26("teams")
    team_lookup = c.build_team_lookup(teams)

    finished = [g for g in games if str(g.get("finished", "")).upper() == "TRUE"]
    print(f"Found {len(finished)} finished matches.")

    c.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    if c.STATS_CACHE_PATH.exists():
        cache = json.loads(c.STATS_CACHE_PATH.read_text())
        print(f"Loaded existing cache with {len(cache)} matches.")

    added = 0
    for g in finished:
        mid = str(g["id"])
        if mid in cache:
            continue  # already cached, skip — safe to re-run any time

        home_code = team_lookup.get(g["home_team_id"], {}).get("fifa_code", "UNK")
        away_code = team_lookup.get(g["away_team_id"], {}).get("fifa_code", "UNK")

        cache[mid] = {
            "home_team": home_code,
            "away_team": away_code,
            "home_score": c.safe_int(g.get("home_score")),
            "away_score": c.safe_int(g.get("away_score")),
            "home_scorers": c.parse_scorers_named(g.get("home_scorers")),
            "away_scorers": c.parse_scorers_named(g.get("away_scorers")),
            "date": g.get("local_date"),
            "stage": g.get("type"),
            # Future fields go here once a stats API is wired up:
            # "home_shots": ..., "away_possession_pct": ..., etc.
        }
        added += 1

    c.STATS_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"Added {added} new matches. Cache now has {len(cache)} total.")
    print(f"Saved to: {c.STATS_CACHE_PATH}")


if __name__ == "__main__":
    main()
