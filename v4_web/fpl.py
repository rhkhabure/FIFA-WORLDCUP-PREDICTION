"""
fpl.py  —  V4.2
================
FPL (Fantasy Premier League) API client.

No API key required. Provides:
  - Upcoming fixtures for the current/next gameweek
  - Team ID <-> name mapping
  - Kickoff times converted to EAT (East Africa Time, UTC+3)

The FPL bootstrap is cached in memory for 5 minutes so the fixtures
widget doesn't make a new HTTP request on every page load.

Base URL: https://fantasy.premierleague.com/api/
"""

import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

BASE = "https://fantasy.premierleague.com/api"
EAT  = timezone(timedelta(hours=3))   # East Africa Time = UTC+3
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; V4-Dashboard/1.0)"}

# Simple in-memory cache: (data, timestamp)
_cache: dict = {}
CACHE_TTL = 300   # 5 minutes


def _fetch(endpoint: str) -> dict | list:
    url = f"{BASE}/{endpoint.lstrip('/')}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"FPL HTTP {e.code} on {endpoint}: {e.reason}")
    except Exception as e:
        print(f"FPL error on {endpoint}: {e}")
    return {}


def _cached(key: str, endpoint: str) -> dict | list:
    now = time.time()
    if key in _cache and now - _cache[key][1] < CACHE_TTL:
        return _cache[key][0]
    data = _fetch(endpoint)
    if data:
        _cache[key] = (data, now)
    return data


# ── Public API ─────────────────────────────────────────────────────────────────

def get_team_map() -> dict[int, str]:
    """Returns {fpl_team_id: team_name} for all 20 PL teams."""
    bootstrap = _cached("bootstrap", "bootstrap-static/")
    teams = bootstrap.get("teams", []) if isinstance(bootstrap, dict) else []
    return {t["id"]: t["name"] for t in teams}


def get_current_gameweek() -> int | None:
    """Returns the ID of the current or next gameweek."""
    bootstrap = _cached("bootstrap", "bootstrap-static/")
    if not isinstance(bootstrap, dict):
        return None
    for event in bootstrap.get("events", []):
        if event.get("is_current") or event.get("is_next"):
            return event["id"]
    return None


def get_upcoming_fixtures(max_fixtures: int = 10) -> list[dict]:
    """
    Returns upcoming (not yet finished) PL fixtures, up to max_fixtures,
    sorted by kickoff time.

    Each dict:
      fixture_id  : int
      home        : str  (team name)
      away        : str  (team name)
      kickoff_utc : str  (ISO 8601 UTC)
      kickoff_eat : str  (formatted in EAT for display)
      gameweek    : int
      started     : bool
      finished    : bool
    """
    team_map   = get_team_map()
    if not team_map:
        return []

    all_fixtures = _cached("fixtures", "fixtures/")
    if not isinstance(all_fixtures, list):
        return []

    upcoming = [f for f in all_fixtures if not f.get("finished", True)]
    upcoming.sort(key=lambda f: f.get("kickoff_time") or "")

    result = []
    for f in upcoming[:max_fixtures]:
        home_name = team_map.get(f.get("team_h"), "Unknown")
        away_name = team_map.get(f.get("team_a"), "Unknown")
        ko_raw    = f.get("kickoff_time")

        # Convert kickoff to EAT for display
        if ko_raw:
            try:
                ko_utc = datetime.fromisoformat(ko_raw.replace("Z", "+00:00"))
                ko_eat = ko_utc.astimezone(EAT)
                ko_display = ko_eat.strftime("%a %d %b · %H:%M EAT")
            except Exception:
                ko_display = ko_raw
        else:
            ko_display = "TBC"

        result.append({
            "fixture_id"   : f.get("id"),          # FPL internal ID (for highlighting)
            "match_id"     : f.get("code"),          # football-data.org match code
            "home"         : home_name,
            "away"         : away_name,
            "kickoff_utc"  : ko_raw or "",
            "kickoff_eat"  : ko_display,
            "gameweek"     : f.get("event"),
            "started"      : f.get("started", False),
            "finished"     : f.get("finished", False),
        })

    return result


def get_next_fixture_id() -> int | None:
    """Returns the FPL fixture_id of the next unstarted match."""
    upcoming = get_upcoming_fixtures(max_fixtures=1)
    return upcoming[0]["fixture_id"] if upcoming else None


def fpl_fixture_to_football_data_id(fpl_fixture_id: int) -> int | None:
    """
    FPL fixtures have a 'code' field which matches the football-data.org
    match code. Use this to bridge between the two APIs when needed.
    """
    all_fixtures = _cached("fixtures", "fixtures/")
    if not isinstance(all_fixtures, list):
        return None
    for f in all_fixtures:
        if f.get("id") == fpl_fixture_id:
            return f.get("code")
    return None
