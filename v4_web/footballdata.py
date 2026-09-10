"""
footballdata.py  —  V4.2
=========================
Live match data from football-data.org v4 API.

Free tier: 10 req/min, covers Premier League (PL) current season,
           live scores, match events, team names.
Auth header: X-Auth-Token

Environment variable: FOOTBALLDATA_ORG_KEY in project root .env
(despite the name, this is a football-data.org key -- format: 32-char hex)

Base URL: https://api.football-data.org/v4
Premier League competition code: PL
"""

import json
import os
import urllib.request
import urllib.error

BASE_URL  = "https://api.football-data.org/v4"
PL_CODE   = "PL"


def _get_api_key() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        env_path = os.path.join(here, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FOOTBALLDATA_ORG_KEY="):
                        return line.split("=", 1)[1].strip()
        here = os.path.dirname(here)
    return os.environ.get("FOOTBALLDATA_ORG_KEY", "")


API_KEY = _get_api_key()


def _fetch(endpoint: str, params: dict | None = None) -> dict:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    req = urllib.request.Request(
        url,
        headers={
            "X-Auth-Token": API_KEY,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"football-data.org HTTP {e.code} on /{endpoint}: {e.reason}")
    except Exception as e:
        print(f"football-data.org error on /{endpoint}: {e}")
    return {}


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_last_completed_pl_match() -> int | None:
    """
    Returns the match ID of the most recently completed Premier League
    2025/26 match. Must pass season=2025 explicitly -- without it,
    football-data.org returns FA Cup matches labelled as PL.
    """
    data = _fetch(f"competitions/{PL_CODE}/matches",
                  {"status": "FINISHED", "season": 2025})
    matches = data.get("matches", [])
    if matches:
        # Matches come back in chronological order -- last is most recent
        return matches[-1]["id"]
    return None


def get_today_pl_fixtures() -> list[dict]:
    """All live PL matches right now."""
    data = _fetch(f"competitions/{PL_CODE}/matches", {"status": "LIVE"})
    return [_parse_match(m) for m in data.get("matches", [])]


def get_live_match_data(match_id: int | str) -> dict:
    """Full match detail for one match ID."""
    data = _fetch(f"matches/{match_id}")
    if not data or "id" not in data:
        return _empty_match()
    parsed = _parse_match(data)
    parsed["events"] = _parse_goals(data)
    return parsed


# ── Internal parsers ───────────────────────────────────────────────────────────

def _clean_name(name: str) -> str:
    """
    Normalise football-data.org team names to match what the DC priors
    store (Understat spelling). Two transformations needed:

    1. Strip common suffixes: 'Arsenal FC' -> 'Arsenal'
    2. Strip AFC prefix:      'AFC Bournemouth' -> 'Bournemouth'

    Remaining mismatches (Tottenham Hotspur -> Tottenham, Brighton &
    Hove Albion -> Brighton) are handled by TEAM_NAME_ALIASES in
    feature_builder.py.
    """
    # Handle 'AFC Bournemouth' style (prefix, not suffix)
    if name.startswith("AFC "):
        name = name[4:]

    # Strip common suffixes
    for suffix in [" FC", " AFC", " City FC", " United FC", " Town FC",
                   " Wanderers FC", " Rovers FC", " Athletic FC",
                   " & Hove Albion FC", " & Hove Albion"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name.strip()


def _normalise_status(raw: str) -> str:
    return {
        "FINISHED" : "Finished",
        "IN_PLAY"  : "In Play",
        "PAUSED"   : "Half Time",
        "SCHEDULED": "Not Started",
        "TIMED"    : "Not Started",
        "POSTPONED": "Postponed",
        "CANCELLED": "Cancelled",
    }.get(raw, raw)


def _derive_minute(status_raw: str) -> int:
    """
    football-data.org free tier doesn't expose current elapsed minute.
    Approximate so the neural net gets a sensible time feature.
    """
    return {"IN_PLAY": 70, "PAUSED": 45, "FINISHED": 90}.get(status_raw, 0)


def _parse_match(m: dict) -> dict:
    home_name = _clean_name(m.get("homeTeam", {}).get("name", "Unknown Home"))
    away_name = _clean_name(m.get("awayTeam", {}).get("name", "Unknown Away"))
    score     = m.get("score", {})
    ft        = score.get("fullTime", {})
    ht        = score.get("halfTime", {})
    h_score   = ft.get("home") or 0
    a_score   = ft.get("away") or 0
    h_ht      = ht.get("home")
    a_ht      = ht.get("away")
    status_raw = m.get("status", "")
    minute     = _derive_minute(status_raw)

    # Odds (bookmaker market prices -- interesting context for the model)
    odds      = m.get("odds", {}) or {}
    odds_home = odds.get("homeWin")
    odds_draw = odds.get("draw")
    odds_away = odds.get("awayWin")

    # Venue and crests
    venue      = m.get("venue", "") or ""
    home_crest = m.get("homeTeam", {}).get("crest", "") or ""
    away_crest = m.get("awayTeam", {}).get("crest", "") or ""

    # Referee
    referees   = m.get("referees", []) or []
    referee    = referees[0].get("name", "") if referees else ""

    return {
        "fixture_id"    : m.get("id"),
        "home_team"     : home_name,
        "away_team"     : away_name,
        "h_score"       : int(h_score),
        "a_score"       : int(a_score),
        "home_score"    : int(h_score),
        "away_score"    : int(a_score),
        "h_ht"          : int(h_ht) if h_ht is not None else None,
        "a_ht"          : int(a_ht) if a_ht is not None else None,
        "status"        : _normalise_status(status_raw),
        "current_minute": minute,
        "minute"        : minute,
        "live_xg"       : {"home": 0.0, "away": 0.0},
        "events"        : [],
        "red_cards"     : {"home": 0, "away": 0},
        "venue"         : venue,
        "home_crest"    : home_crest,
        "away_crest"    : away_crest,
        "odds_home"     : float(odds_home) if odds_home else None,
        "odds_draw"     : float(odds_draw) if odds_draw else None,
        "odds_away"     : float(odds_away) if odds_away else None,
        "referee"       : referee,
    }


def _parse_goals(m: dict) -> list[dict]:
    events = []
    for g in m.get("goals", []):
        minute = g.get("minute", "?")
        scorer = g.get("scorer", {}).get("name", "Unknown")
        team   = _clean_name(g.get("team", {}).get("name", ""))
        gtype  = g.get("type", "REGULAR")
        events.append({"time": f"{minute}'",
                        "detail": f"{scorer} ({team}) - {gtype}"})
    return events


def _empty_match() -> dict:
    return {
        "fixture_id": None, "home_team": "Unknown Home",
        "away_team": "Unknown Away", "h_score": 0, "a_score": 0,
        "home_score": 0, "away_score": 0, "status": "",
        "current_minute": 0, "minute": 0,
        "live_xg": {"home": 0.0, "away": 0.0},
        "events": [], "red_cards": {"home": 0, "away": 0},
    }
