"""
footballdata.py  —  V4.2
=========================
Live match data from API-Football v3 (api-football.com / api-sports.io).

Free tier: 100 requests/day, all endpoints, Premier League included.
Auth header: x-apisports-key  (NOT Bearer token like footballdata.io)

Previously used footballdata.io which gave 403 on every endpoint for the
free plan. This replaces it entirely with the API-Football v3 endpoints
which are confirmed to work on the free tier.

Environment variable: API_FOOTBALL_KEY in project root .env
"""

import json
import os
import urllib.request
import urllib.error

BASE_URL   = "https://v3.football.api-sports.io"
PL_LEAGUE  = 39       # Premier League league ID in API-Football
PL_SEASON  = 2025     # 2025/26 season is stored as 2025 in API-Football


def _get_api_key() -> str:
    # Walk up from this file's location to find the .env in the project root
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):   # search up to 4 levels up
        env_path = os.path.join(here, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("API_FOOTBALL_KEY="):
                        return line.split("=", 1)[1].strip()
        here = os.path.dirname(here)
    return os.environ.get("API_FOOTBALL_KEY", "")


API_KEY = _get_api_key()


def _fetch(endpoint: str, params: dict | None = None) -> dict:
    """
    GET request to API-Football v3.
    Returns the parsed JSON or {} on any error.
    """
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"

    req = urllib.request.Request(
        url,
        headers={
            "x-apisports-key": API_KEY,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"API-Football HTTP {e.code} on /{endpoint}: {e.reason}")
    except Exception as e:
        print(f"API-Football error on /{endpoint}: {e}")
    return {}


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_last_completed_pl_match() -> int | None:
    """
    Returns the fixture ID of the most recently completed Premier League match.
    Used as a fallback when no match_id is supplied to the /match route.
    """
    data = _fetch("fixtures", {"league": PL_LEAGUE, "season": PL_SEASON,
                               "last": 1})
    for fix in data.get("response", []):
        return fix["fixture"]["id"]
    return None


def get_today_pl_fixtures() -> list[dict]:
    """
    Returns all Premier League fixtures scheduled for today (or live now).
    Each dict has: fixture_id, home, away, status, minute, h_score, a_score.
    """
    data = _fetch("fixtures", {"league": PL_LEAGUE, "season": PL_SEASON,
                               "live": "all"})
    results = []
    for fix in data.get("response", []):
        results.append(_parse_fixture(fix))
    return results


def get_live_match_data(fixture_id: int | str) -> dict:
    """
    Full match detail for one fixture: teams, score, minute, events, xG.
    Returns a dict with keys the rest of main.py expects.
    """
    data = _fetch("fixtures", {
        "id": fixture_id,
        "statistics": "true",
    })

    response = data.get("response", [])
    if not response:
        return _empty_match()

    fix = response[0]
    parsed = _parse_fixture(fix)

    # Events (goals, cards, subs)
    events_data = _fetch("fixtures/events", {"fixture": fixture_id})
    parsed["events"] = _parse_events(events_data, parsed["home_team"],
                                     parsed["away_team"])

    # xG from statistics block (may not be present for all matches)
    stats = fix.get("statistics", []) or []
    parsed["live_xg"] = _parse_xg(stats)

    return parsed


# ── Internal parsers ───────────────────────────────────────────────────────────

def _parse_fixture(fix: dict) -> dict:
    fixture  = fix.get("fixture", {})
    teams    = fix.get("teams", {})
    goals    = fix.get("goals", {})
    score    = fix.get("score", {})
    status   = fixture.get("status", {})

    home_name  = teams.get("home", {}).get("name", "Unknown Home")
    away_name  = teams.get("away", {}).get("name", "Unknown Away")
    h_score    = goals.get("home") or 0
    a_score    = goals.get("away") or 0
    status_str = status.get("long", "")
    minute     = status.get("elapsed") or 0

    # Normalise status to something the template can check
    if status_str in ("Match Finished", "Full Time"):
        status_str = "Finished"
    elif status_str in ("Not Started", "Time to be Defined"):
        status_str = "Not Started"

    return {
        "fixture_id"    : fixture.get("id"),
        "home_team"     : home_name,
        "away_team"     : away_name,
        "h_score"       : int(h_score),
        "a_score"       : int(a_score),
        "home_score"    : int(h_score),   # alias for main.py compat
        "away_score"    : int(a_score),   # alias for main.py compat
        "status"        : status_str,
        "current_minute": minute,
        "minute"        : minute,
        "live_xg"       : {"home": 0.0, "away": 0.0},
        "events"        : [],
        "red_cards"     : {"home": 0, "away": 0},
    }


def _parse_events(data: dict, home_name: str, away_name: str) -> list[dict]:
    events = []
    red_cards = {"home": 0, "away": 0}
    for ev in data.get("response", []):
        time    = ev.get("time", {}).get("elapsed", "?")
        etype   = ev.get("type", "").lower()
        detail  = ev.get("detail", etype.replace("_", " ").title())
        player  = ev.get("player", {}).get("name", "Unknown")
        team    = ev.get("team", {}).get("name", "")
        side    = "home" if team == home_name else "away"

        events.append({"time": f"{time}'", "detail": f"{player} ({team}) - {detail}"})
        if etype == "card" and "red" in detail.lower():
            red_cards[side] += 1

    return events


def _parse_xg(stats: list) -> dict:
    xg = {"home": 0.0, "away": 0.0}
    for team_stat in stats:
        side = "home" if team_stat.get("team", {}).get("id") else "away"
        for stat in team_stat.get("statistics", []):
            if stat.get("type") == "expected_goals":
                try:
                    xg[side] = float(stat.get("value") or 0)
                except (TypeError, ValueError):
                    pass
    return xg


def _empty_match() -> dict:
    return {
        "fixture_id": None, "home_team": "Unknown Home",
        "away_team": "Unknown Away", "h_score": 0, "a_score": 0,
        "home_score": 0, "away_score": 0, "status": "",
        "current_minute": 0, "minute": 0,
        "live_xg": {"home": 0.0, "away": 0.0},
        "events": [], "red_cards": {"home": 0, "away": 0},
    }
