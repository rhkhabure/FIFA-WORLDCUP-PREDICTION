"""
fotmob.py  —  V4.2
====================
FotMob API client via parse.bot wrapper.

Provides:
  - Confirmed starting lineups + formations (replaces DEFAULT_SQUADS)
  - Real-time xG per team (replaces projected xG from DC model)
  - Live match events (goals with real minutes, cards, subs)
  - Live score + minute

Credit budget (free tier: 200/month):
  - get_match_lineup  : 1 credit — called ONCE per match, cached
  - get_match_xg      : 1 credit — called every poll_interval minutes
  - get_matches_by_date: 3 credits — called once per day (future use)

Caching strategy:
  - Lineups: cached in memory for the match duration (never re-fetched)
  - Live xG: cached for poll_interval seconds, refreshed on poll
  - Cache is keyed by match_id so switching fixtures doesn't bleed

Environment: PARSE_BOT_KEY in project root .env
API base: https://api.parse.bot/scraper/645b8e03-271d-4c85-97e7-35d5733a2d78
"""

import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

PARSE_BASE = "https://api.parse.bot/scraper/645b8e03-271d-4c85-97e7-35d5733a2d78"

# In-memory caches
_lineup_cache: dict[str, dict] = {}          # match_id -> lineup data
_xg_cache:     dict[str, tuple] = {}         # match_id -> (data, timestamp)
XG_CACHE_TTL = 60  # seconds -- actual TTL controlled by poll_interval in JS


def _get_api_key() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        env_path = os.path.join(here, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("PARSE_BOT_KEY="):
                        return line.split("=", 1)[1].strip()
        here = os.path.dirname(here)
    return os.environ.get("PARSE_BOT_KEY", "")


API_KEY = _get_api_key()


def _fetch(endpoint: str, params: dict | None = None) -> dict:
    """Single GET request to parse.bot FotMob API."""
    url = f"{PARSE_BASE}/{endpoint.lstrip('/')}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    req = urllib.request.Request(
        url,
        headers={
            "X-API-Key": API_KEY,
            "Accept"   : "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"FotMob HTTP {e.code} on {endpoint}: {e.reason}")
    except Exception as e:
        print(f"FotMob error on {endpoint}: {e}")
    return {}


# ── Public API ─────────────────────────────────────────────────────────────────

def has_key() -> bool:
    """True if a PARSE_BOT_KEY is configured."""
    return bool(API_KEY)


def get_lineup(match_id: str | int) -> dict | None:
    """
    Fetch confirmed lineup for a match. Cached for the match duration.

    Returns dict with:
      home_formation : str   e.g. "4-2-3-1"
      away_formation : str
      home_players   : list[str]  11 starters in position order
      away_players   : list[str]
      home_subs      : list[str]  unused substitutes
      away_subs      : list[str]
      match_status   : str
      live_minute    : int | None
      score          : {"home": int, "away": int}
    """
    key = str(match_id)
    if key in _lineup_cache:
        return _lineup_cache[key]

    if not API_KEY:
        return None

    data = _fetch("get_match_lineup", {"match_id": match_id})
    if data.get("status") != "success":
        return None

    d = data.get("data", {})

    def extract_players(team_data: dict) -> tuple[list[str], list[str], str]:
        players_raw = team_data.get("players", [])
        formation   = team_data.get("formation", "4-3-3") or "4-3-3"
        starters = [
            p["name"] for p in players_raw
            if p.get("is_starter") and p.get("position") != "Coach"
        ]
        subs = [
            p["name"] for p in players_raw
            if not p.get("is_starter") and p.get("position") != "Coach"
        ]
        return starters, subs, formation

    h_starters, h_subs, h_form = extract_players(d.get("home_team", {}))
    a_starters, a_subs, a_form = extract_players(d.get("away_team", {}))

    result = {
        "home_formation": h_form,
        "away_formation": a_form,
        "home_players"  : h_starters,
        "away_players"  : a_starters,
        "home_subs"     : h_subs,
        "away_subs"     : a_subs,
        "match_status"  : d.get("match_status", ""),
        "live_minute"   : d.get("live_minute"),
        "score"         : d.get("score", {"home": 0, "away": 0}),
    }

    # Cache lineup -- it doesn't change during the match
    _lineup_cache[key] = result
    return result


def get_live_xg(match_id: str | int, max_age_seconds: int = 300) -> dict | None:
    """
    Fetch live xG for a match. Cached for max_age_seconds.

    Returns dict with:
      home_xg      : float   total xG
      away_xg      : float
      home_xg_h1   : float   first half only
      away_xg_h1   : float
      home_xg_h2   : float   second half only
      away_xg_h2   : float
    Or None if unavailable / match hasn't started.
    """
    key = str(match_id)
    if key in _xg_cache:
        cached_data, cached_time = _xg_cache[key]
        if time.time() - cached_time < max_age_seconds:
            return cached_data

    if not API_KEY:
        return None

    data = _fetch("get_match_xg", {"match_id": match_id})
    if data.get("status") != "success":
        return None

    d = data.get("data", {})
    home = d.get("home", {}) or {}
    away = d.get("away", {}) or {}

    def safe_float(v) -> float:
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    result = {
        "home_xg"   : safe_float(home.get("xg")),
        "away_xg"   : safe_float(away.get("xg")),
        "home_xg_h1": safe_float(home.get("first_half")),
        "away_xg_h1": safe_float(away.get("first_half")),
        "home_xg_h2": safe_float(home.get("second_half")),
        "away_xg_h2": safe_float(away.get("second_half")),
    }

    _xg_cache[key] = (result, time.time())
    return result


def clear_cache(match_id: str | int | None = None):
    """
    Clear cached data. Pass match_id to clear one match,
    or None to clear everything (e.g. on server restart).
    """
    if match_id is None:
        _lineup_cache.clear()
        _xg_cache.clear()
    else:
        key = str(match_id)
        _lineup_cache.pop(key, None)
        _xg_cache.pop(key, None)


def get_fotmob_match_id(home_name: str, away_name: str, date_str: str | None = None) -> str | None:
    """
    Look up the FotMob match ID for a given fixture by team names.
    date_str: YYYYMMDD format, defaults to today (UTC).

    FotMob IDs (e.g. 5795445) differ from FPL codes (e.g. 2645229).
    This resolves the correct ID from the matches-by-date endpoint.
    Costs 3 credits -- result is cached per date.
    """
    if not API_KEY:
        return None

    if date_str is None:
        from datetime import datetime, timezone
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    cache_key = f"matches_{date_str}"
    if cache_key not in _lineup_cache:
        data = _fetch("get_matches_by_date", {"date": date_str})
        if data.get("status") == "success":
            _lineup_cache[cache_key] = data
        else:
            return None
    else:
        data = _lineup_cache[cache_key]

    # Normalise names for matching
    def norm(s: str) -> str:
        return s.lower().replace("fc", "").replace("afc", "").strip()

    home_n = norm(fotmob_to_fpl_team_name(home_name) or home_name)
    away_n = norm(fotmob_to_fpl_team_name(away_name) or away_name)

    leagues = data.get("data", {}).get("leagues", [])
    for league in leagues:
        for match in league.get("matches", []):
            mh = norm(match.get("home", {}).get("name", ""))
            ma = norm(match.get("away", {}).get("name", ""))
            if (home_n in mh or mh in home_n) and (away_n in ma or ma in away_n):
                return str(match.get("id"))
    return None
    """
    FotMob uses full official club names. Map to our internal short names
    that match TEAM_NAME_ALIASES and DEFAULT_SQUADS.
    """
    _MAP = {
        "AFC Bournemouth"          : "Bournemouth",
        "Tottenham Hotspur"        : "Tottenham",
        "Nottingham Forest"        : "Nottingham Forest",
        "Brighton & Hove Albion"   : "Brighton",
        "Wolverhampton Wanderers"  : "Wolverhampton Wanderers",
        "West Ham United"          : "West Ham",
        "Newcastle United"         : "Newcastle United",
        "Manchester City"          : "Manchester City",
        "Manchester United"        : "Manchester United",
        "Ipswich Town"             : "Ipswich",
        "Leicester City"           : "Leicester",
        "Leeds United"             : "Leeds",
        "Sheffield United"         : "Sheffield United",
        "Aston Villa"              : "Aston Villa",
        "Crystal Palace"           : "Crystal Palace",
        "Hull City"                : "Hull City",
        "Coventry City"            : "Coventry",
        "Sunderland"               : "Sunderland",
    }
    return _MAP.get(fotmob_name, fotmob_name)
