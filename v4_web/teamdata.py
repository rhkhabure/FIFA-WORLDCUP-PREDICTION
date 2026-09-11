"""
teamdata.py  —  V4.2
======================
Team profile data layer.

Pulls from:
  - football-data.org free tier: squad, venue, season results
  - utils.py: TEAM_MANAGERS fallback, crest proxy URL
  - v4_priors.json: DC alpha/beta ratings

All functions return plain dicts ready for Jinja2 templates.
"""

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

EAT = timezone(timedelta(hours=3))


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


API_KEY  = _get_api_key()
BASE_URL = "https://api.football-data.org/v4"


def _fetch(endpoint: str, params: dict | None = None) -> dict:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    req = urllib.request.Request(
        url, headers={"X-Auth-Token": API_KEY}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"teamdata HTTP {e.code} on {endpoint}: {e.reason}")
    except Exception as e:
        print(f"teamdata error on {endpoint}: {e}")
    return {}


def _clean_name(raw: str) -> str:
    """Strip ' FC', ' AFC' suffixes for display."""
    for suffix in [" FC", " AFC", " F.C.", " A.F.C."]:
        if raw.endswith(suffix):
            raw = raw[:-len(suffix)]
    return raw.strip()


def _fmt_date(utc_str: str) -> str:
    """Format UTC ISO string to 'Sat 12 Sep' in EAT."""
    if not utc_str:
        return ""
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        return dt.astimezone(EAT).strftime("%a %d %b")
    except Exception:
        return utc_str[:10]


def get_team_profile(team_id: int) -> dict:
    """
    Returns full team profile dict:
      name, short_name, crest, venue, founded, colours,
      coach, squad (list of player dicts),
      competitions (list of league names currently in)
    """
    data = _fetch(f"teams/{team_id}")
    if not data:
        return {}

    # Coach -- football-data.org often returns null; caller falls back to TEAM_MANAGERS
    coach_raw = data.get("coach", {}) or {}
    coach_name = coach_raw.get("name") or ""

    # Squad -- group by position
    squad_raw = data.get("squad", []) or []
    positions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    squad: dict[str, list] = {pos: [] for pos in positions}
    for p in squad_raw:
        pos = p.get("position", "Forward")
        if pos not in squad:
            pos = "Forward"
        dob = p.get("dateOfBirth", "")
        age = ""
        if dob:
            try:
                born = datetime.fromisoformat(dob)
                age = str((datetime.now() - born).days // 365)
            except Exception:
                pass
        squad[pos].append({
            "id"         : p.get("id"),
            "name"       : p.get("name", ""),
            "nationality": p.get("nationality", ""),
            "age"        : age,
        })

    competitions = [
        c.get("name", "") for c in (data.get("runningCompetitions") or [])
    ]

    return {
        "id"          : data.get("id"),
        "name"        : _clean_name(data.get("name", "")),
        "short_name"  : data.get("shortName", ""),
        "tla"         : data.get("tla", ""),
        "crest"       : data.get("crest", ""),
        "venue"       : data.get("venue", ""),
        "founded"     : data.get("founded"),
        "colours"     : data.get("clubColors", ""),
        "coach"       : coach_name,
        "squad"       : squad,
        "competitions": competitions,
        "address"     : data.get("address", ""),
        "website"     : data.get("website", ""),
    }


def get_team_season_results(team_id: int, season: int = 2025,
                             limit: int = 38) -> list[dict]:
    """
    Returns list of season results for a team, most recent first.
    Each dict:
      date, opponent, home_away, h_score, a_score,
      result (W/D/L from team perspective),
      match_id, matchday, status
    """
    data = _fetch(f"teams/{team_id}/matches",
                  {"season": season, "limit": limit})
    matches_raw = data.get("matches", [])
    if not matches_raw:
        return []

    results = []
    for m in matches_raw:
        home = _clean_name(m.get("homeTeam", {}).get("name", ""))
        away = _clean_name(m.get("awayTeam", {}).get("name", ""))
        ft   = (m.get("score", {}) or {}).get("fullTime", {}) or {}
        hs   = ft.get("home")
        as_  = ft.get("away")
        status = m.get("status", "")

        is_home = (m.get("homeTeam", {}).get("id") == team_id)
        opponent = away if is_home else home
        home_away = "H" if is_home else "A"

        # Result from team perspective
        result = ""
        if status == "FINISHED" and hs is not None and as_ is not None:
            team_goals = hs if is_home else as_
            opp_goals  = as_ if is_home else hs
            if team_goals > opp_goals:
                result = "W"
            elif team_goals == opp_goals:
                result = "D"
            else:
                result = "L"

        results.append({
            "match_id"  : m.get("id"),
            "date"      : _fmt_date(m.get("utcDate", "")),
            "utc_date"  : m.get("utcDate", ""),
            "matchday"  : m.get("matchday"),
            "opponent"  : opponent,
            "home_away" : home_away,
            "h_score"   : hs,
            "a_score"   : as_,
            "result"    : result,
            "status"    : status,
            "is_home"   : is_home,
        })

    # Most recent first
    results.sort(key=lambda x: x["utc_date"] or "", reverse=True)
    return results


def get_next_fixture(results: list[dict]) -> dict | None:
    """Extract the next unplayed fixture from the results list."""
    for r in reversed(results):   # results are newest-first, so reverse
        if r["status"] in ("SCHEDULED", "TIMED"):
            return r
    return None


def get_last_n_results(results: list[dict], n: int = 5) -> list[dict]:
    """Return the last N finished results."""
    finished = [r for r in results if r["result"] in ("W", "D", "L")]
    return finished[:n]


def get_form_string(results: list[dict], n: int = 5) -> str:
    """Return form string like 'WDWLW' from last N finished results."""
    last = get_last_n_results(results, n)
    return "".join(r["result"] for r in last)
