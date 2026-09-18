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
import ssl
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

BASE = "https://fantasy.premierleague.com/api"
EAT  = timezone(timedelta(hours=3))   # East Africa Time = UTC+3
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; V4-Dashboard/1.0)"}

# Unverified SSL context — required on Windows where the system certificate
# store may not include the FPL CDN root CA. Safe: public read-only API.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

# In-memory cache: key → (data, timestamp)
# Different TTLs per data type:
#   bootstrap  — team names, GW info: 1 hour (changes rarely mid-season)
#   fixtures   — full fixture list:   6 hours (only updates Thursday)
#   live       — live GW scores:      60 seconds (changes during matches)
_cache: dict = {}
_CACHE_TTLS: dict[str, int] = {
    "bootstrap": 3600,    # 1 hour
    "fixtures":  21600,   # 6 hours
    "live":      60,      # 1 minute
}
CACHE_TTL = 3600   # default fallback (1 hour)


def _fetch(endpoint: str) -> dict | list:
    url = f"{BASE}/{endpoint.lstrip('/')}"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"FPL HTTP {e.code} on {endpoint}: {e.reason}")
    except Exception as e:
        print(f"FPL error on {endpoint}: {e}")
    return {}


def _cached(key: str, endpoint: str) -> dict | list:
    ttl = _CACHE_TTLS.get(key, CACHE_TTL)
    now = time.time()
    # Return fresh cache if within TTL
    if key in _cache and now - _cache[key][1] < ttl:
        return _cache[key][0]
    # Attempt fetch
    data = _fetch(endpoint)
    # Only cache if response is meaningfully non-empty
    # Empty dict {} or [] means the API failed — use stale cache instead
    is_valid = (isinstance(data, list) and len(data) > 0) or \
               (isinstance(data, dict) and len(data) > 0)
    if is_valid:
        _cache[key] = (data, now)
        return data
    elif key in _cache:
        # Serve stale data rather than empty on fetch failure
        print(f"[fpl] fetch failed for {key} — serving stale cache")
        return _cache[key][0]
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


def warm_cache() -> None:
    """
    Pre-fetch bootstrap and fixtures into cache on server startup.
    Only caches non-empty successful responses.
    """
    try:
        data = _cached("bootstrap", "bootstrap-static/")
        teams = data.get("teams", []) if isinstance(data, dict) else []
        print(f"[fpl] bootstrap cached ({len(teams)} teams)")
    except Exception as e:
        print(f"[fpl] bootstrap warm failed: {e}")
    try:
        data = _cached("fixtures", "fixtures/")
        n = len(data) if isinstance(data, list) else 0
        upcoming = [f for f in (data if isinstance(data, list) else [])
                    if not f.get("finished", True)]
        print(f"[fpl] fixtures cached ({n} total, {len(upcoming)} upcoming, TTL 6h)")
    except Exception as e:
        print(f"[fpl] fixtures warm failed: {e}")
