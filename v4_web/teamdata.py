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
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

EAT = timezone(timedelta(hours=3))

# Simple in-memory cache to avoid rate limit (free tier: 10 req/min)
_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes


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
    # Cache check
    if url in _cache:
        data, ts = _cache[url]
        if time.time() - ts < _CACHE_TTL:
            return data
    req = urllib.request.Request(
        url, headers={"X-Auth-Token": API_KEY}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        _cache[url] = (data, time.time())
        return data
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


# Nationality string → flag emoji
# Regional indicator letters: A=U+1F1E6, so 'GB' → 🇬🇧
_NAT_TO_CODE: dict[str, str] = {
    "Afghanistan":"AF","Albania":"AL","Algeria":"DZ","Angola":"AO",
    "Argentina":"AR","Armenia":"AM","Australia":"AU","Austria":"AT",
    "Azerbaijan":"AZ","Bahrain":"BH","Belgium":"BE","Bolivia":"BO",
    "Bosnia and Herzegovina":"BA","Bosnia-Herzegovina":"BA",
    "Brazil":"BR","Bulgaria":"BG","Burkina Faso":"BF","Cameroon":"CM",
    "Canada":"CA","Chile":"CL","China PR":"CN","China":"CN",
    "Colombia":"CO","Congo":"CG","Congo DR":"CD","Costa Rica":"CR",
    "Cote d'Ivoire":"CI","Croatia":"HR","Cuba":"CU","Cyprus":"CY",
    "Czech Republic":"CZ","Czechia":"CZ","Denmark":"DK","DR Congo":"CD",
    "Ecuador":"EC","Egypt":"EG","El Salvador":"SV","England":"GB-ENG",
    "Estonia":"EE","Ethiopia":"ET","Finland":"FI","France":"FR",
    "Gabon":"GA","Gambia":"GM","Georgia":"GE","Germany":"DE",
    "Ghana":"GH","Greece":"GR","Guinea":"GN","Guinea-Bissau":"GW",
    "Honduras":"HN","Hungary":"HU","Iceland":"IS","India":"IN",
    "Indonesia":"ID","Iran":"IR","Iraq":"IQ","Ireland":"IE",
    "Israel":"IL","Italy":"IT","Jamaica":"JM","Japan":"JP",
    "Jordan":"JO","Kazakhstan":"KZ","Kenya":"KE","Kosovo":"XK",
    "Latvia":"LV","Lebanon":"LB","Liberia":"LR","Libya":"LY",
    "Lithuania":"LT","Luxembourg":"LU","Mali":"ML","Malta":"MT",
    "Mauritania":"MR","Mexico":"MX","Moldova":"MD","Montenegro":"ME",
    "Morocco":"MA","Mozambique":"MZ","Netherlands":"NL","New Zealand":"NZ",
    "Nigeria":"NG","North Korea":"KP","North Macedonia":"MK",
    "Northern Ireland":"GB-NIR","Norway":"NO","Panama":"PA","Paraguay":"PY",
    "Peru":"PE","Philippines":"PH","Poland":"PL","Portugal":"PT",
    "Qatar":"QA","Republic of Ireland":"IE","Romania":"RO","Russia":"RU",
    "Saudi Arabia":"SA","Scotland":"GB-SCT","Senegal":"SN","Serbia":"RS",
    "Sierra Leone":"SL","Slovakia":"SK","Slovenia":"SI","Somalia":"SO",
    "South Africa":"ZA","South Korea":"KR","Spain":"ES","Basque Country":"ES","Catalonia":"ES","Sudan":"SD",
    "Sweden":"SE","Switzerland":"CH","Syria":"SY","Tanzania":"TZ",
    "Togo":"TG","Trinidad and Tobago":"TT","Tunisia":"TN","Turkey":"TR",
    "Uganda":"UG","Ukraine":"UA","United States":"US","Uruguay":"UY",
    "Uzbekistan":"UZ","Venezuela":"VE","Wales":"GB-WAL","Zambia":"ZM",
    "Zimbabwe":"ZW",
}

def _flag_img(nationality: str) -> str:
    """
    Returns an HTML img tag for the country flag using flagcdn.com.
    Free CDN, no key, reliable cross-platform rendering.
    Returns '' if nationality unknown.
    """
    if not nationality:
        return ""
    nationality = nationality.strip()
    code = _NAT_TO_CODE.get(nationality, "") or _NAT_TO_CODE.get(nationality.title(), "")
    if not code:
        return ""
    # Normalise to two-letter lowercase for flagcdn.com
    # GB subdivisions all use 'gb' on flagcdn
    iso = code.split("-")[0].lower()
    return (f'<img src="https://flagcdn.com/16x12/{iso}.png" '
            f'width="16" height="12" alt="{nationality}" '
            f'class="inline-block rounded-sm" '
            f'style="vertical-align:middle;margin-right:3px">')


def get_standings(season: int = 2025) -> list[dict]:
    """
    Returns PL standings table for the given season.
    Each dict:
      position, team_id, team_name, crest,
      played, won, draw, lost, gf, ga, gd, points, form
    Returns [] on failure.
    """
    data = _fetch("competitions/PL/standings", {"season": season})
    if not data:
        return []
    try:
        table = data["standings"][0]["table"]
    except (KeyError, IndexError):
        return []

    rows = []
    for row in table:
        team = row.get("team", {})
        rows.append({
            "position": row.get("position"),
            "team_id" : team.get("id"),
            "team_name": _clean_name(team.get("name", "")),
            "crest"   : team.get("crest", ""),
            "played"  : row.get("playedGames", 0),
            "won"     : row.get("won", 0),
            "draw"    : row.get("draw", 0),
            "lost"    : row.get("lost", 0),
            "gf"      : row.get("goalsFor", 0),
            "ga"      : row.get("goalsAgainst", 0),
            "gd"      : row.get("goalDifference", 0),
            "points"  : row.get("points", 0),
            "form"    : row.get("form", ""),
        })
    return rows


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

    coach_raw = data.get("coach", {}) or {}
    coach_name = coach_raw.get("name") or ""

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
            "flag"       : _flag_img(p.get("nationality", "")),
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
            "flag"       : _flag_img(p.get("nationality", "")),
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
