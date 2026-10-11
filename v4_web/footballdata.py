"""
footballdata.py  —  V4.2
=========================
Live match data from football-data.org v4 API.

Free tier: 10 req/min, covers Premier League (PL) and La Liga (PD)
           current season, live scores, match events, team names.
Auth header: X-Auth-Token

Environment variable: FOOTBALLDATA_ORG_KEY in project root .env
(despite the name, this is a football-data.org key -- format: 32-char hex)

Base URL: https://api.football-data.org/v4
Competition codes: PL = Premier League, PD = La Liga
"""

import json
import os
import re
import time as _time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

BASE_URL  = "https://api.football-data.org/v4"
PL_CODE   = "PL"
PD_CODE   = "PD"   # La Liga

# Map competition code → priors key
COMP_TO_PRIORS = {
    "PL": "ENG-Premier League",
    "PD": "ESP-La Liga",
}


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


# ── Per-competition finished-match cache ──────────────────────────
# _finished_cache[comp_code] = (dict of parsed matches, timestamp)
_finished_cache: dict[str, tuple[dict, float]] = {}
_FINISHED_TTL = 300   # 5 minutes
_competition_teams_cache: dict[tuple[str, int], tuple[list[dict], float]] = {}
_TEAM_TTL = 3600
_team_matches_cache: dict[tuple[int, str, int], tuple[list[dict], float]] = {}
_competition_standings_cache: dict[tuple[str, int], tuple[list[dict], float]] = {}
_team_detail_cache: dict[int, tuple[dict, float]] = {}


def _current_season() -> int:
    """
    Return the start year used by fd.org for the current football season.
    """
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _populate_finished_cache(comp_code: str) -> dict:
    """Fill or refresh the finished-match cache for a competition."""
    cached = _finished_cache.get(comp_code)
    if cached and _time.time() - cached[1] < _FINISHED_TTL:
        return cached[0]

    season = _current_season()
    data   = _fetch(f"competitions/{comp_code}/matches",
                    {"status": "FINISHED", "season": season})
    matches_by_id = {}
    for m in data.get("matches", []):
        parsed          = _parse_match(m)
        parsed["fd_id"] = m.get("id")
        matches_by_id[str(m.get("id"))] = parsed
    _finished_cache[comp_code] = (matches_by_id, _time.time())
    print(f"[footballdata] {comp_code}: cached {len(matches_by_id)} finished matches "
          f"(season={season})")
    return matches_by_id


# ── Public helpers ─────────────────────────────────────────────────

def get_live_match_data(match_id: int | str) -> dict:
    """Full match detail for one match ID."""
    data = _fetch(f"matches/{match_id}")
    if not data or "id" not in data:
        return _empty_match()
    parsed = _parse_match(data)
    parsed["events"] = _parse_goals(data)
    return parsed


def get_finished_match(match_id: int | str,
                       comp_code: str = PL_CODE) -> dict:
    """
    Get a finished match by football-data.org match ID.
    Scans the competition's finished matches (cached 5min).
    """
    cache = _populate_finished_cache(comp_code)
    return cache.get(str(match_id), _empty_match())


def get_competition_teams_fd(
    comp_code: str, season: int | None = None
) -> list[dict]:
    """Return cached club metadata for a competition season."""
    if season is None:
        season = _current_season()
    cache_key = (comp_code, season)
    cached = _competition_teams_cache.get(cache_key)
    if cached and _time.time() - cached[1] < _TEAM_TTL:
        return cached[0]

    data = _fetch(
        f"competitions/{comp_code}/teams",
        {"season": season},
    )
    from v4_backend.feature_builder import TEAM_NAME_ALIASES

    teams = []
    for raw in data.get("teams", []):
        source_name = _clean_name(raw.get("name", ""))
        team = {
            "id": raw.get("id"),
            "name": TEAM_NAME_ALIASES.get(source_name, source_name),
            "short_name": raw.get("shortName", ""),
            "tla": raw.get("tla", ""),
            "crest": raw.get("crest", ""),
            "venue": raw.get("venue", ""),
            "founded": raw.get("founded"),
            "colours": raw.get("clubColors", ""),
            "coach": (raw.get("coach") or {}).get("name", ""),
            "squad": raw.get("squad", []),
        }
        if team["id"] and team["name"]:
            teams.append(team)
    _competition_teams_cache[cache_key] = (teams, _time.time())
    return teams


def get_team_details_fd(team_id: int) -> dict:
    """Fetch cached detailed club metadata, including the provider squad."""
    cached = _team_detail_cache.get(team_id)
    if cached and _time.time() - cached[1] < _TEAM_TTL:
        return cached[0]
    data = _fetch(f"teams/{team_id}")
    details = {
        "coach": (data.get("coach") or {}).get("name", ""),
        "squad": data.get("squad", []) or [],
    }
    _team_detail_cache[team_id] = (details, _time.time())
    return details


def get_team_matches_fd(
    team_id: int, comp_code: str, season: int | None = None
) -> list[dict]:
    """Return cached season matches for a club in a competition."""
    if season is None:
        season = _current_season()
    cache_key = (team_id, comp_code, season)
    cached = _team_matches_cache.get(cache_key)
    if cached and _time.time() - cached[1] < _TEAM_TTL:
        return cached[0]

    data = _fetch(
        f"teams/{team_id}/matches",
        {"competitions": comp_code, "season": season, "limit": 100},
    )
    from v4_backend.feature_builder import TEAM_NAME_ALIASES

    EAT = timezone(timedelta(hours=3))
    matches = []
    for raw in data.get("matches", []):
        home_raw = _clean_name(raw.get("homeTeam", {}).get("name", "Unknown"))
        away_raw = _clean_name(raw.get("awayTeam", {}).get("name", "Unknown"))
        home = TEAM_NAME_ALIASES.get(home_raw, home_raw)
        away = TEAM_NAME_ALIASES.get(away_raw, away_raw)
        is_home = raw.get("homeTeam", {}).get("id") == team_id
        kickoff = raw.get("utcDate", "")
        try:
            local_kickoff = datetime.fromisoformat(
                kickoff.replace("Z", "+00:00")
            ).astimezone(EAT)
            date = local_kickoff.strftime("%d %b %Y")
            kickoff_eat = local_kickoff.strftime("%a %d %b · %H:%M")
        except ValueError:
            date = kickoff[:10]
            kickoff_eat = date

        score = (raw.get("score", {}) or {}).get("fullTime", {}) or {}
        home_score = score.get("home")
        away_score = score.get("away")
        status = raw.get("status", "")
        result = ""
        if status == "FINISHED" and home_score is not None and away_score is not None:
            team_score = home_score if is_home else away_score
            opponent_score = away_score if is_home else home_score
            result = (
                "W" if team_score > opponent_score
                else "D" if team_score == opponent_score
                else "L"
            )
        matches.append({
            "match_id": raw.get("id"),
            "date": date,
            "utc_date": kickoff,
            "kickoff_eat": kickoff_eat,
            "matchday": raw.get("matchday"),
            "home": home,
            "away": away,
            "opponent": away if is_home else home,
            "home_away": "H" if is_home else "A",
            "is_home": is_home,
            "h_score": home_score,
            "a_score": away_score,
            "result": result,
            "status": status,
        })
    matches.sort(key=lambda match: match["utc_date"], reverse=True)
    _team_matches_cache[cache_key] = (matches, _time.time())
    return matches


def get_competition_standings_fd(
    comp_code: str, season: int | None = None
) -> list[dict]:
    """Return the competition table in the shape expected by team pages."""
    if season is None:
        season = _current_season()
    cache_key = (comp_code, season)
    cached = _competition_standings_cache.get(cache_key)
    if cached and _time.time() - cached[1] < _TEAM_TTL:
        return cached[0]
    data = _fetch(
        f"competitions/{comp_code}/standings",
        {"season": season},
    )
    from v4_backend.feature_builder import TEAM_NAME_ALIASES

    try:
        table = data["standings"][0]["table"]
    except (KeyError, IndexError):
        table = []
    standings = []
    for row in table:
        team = row.get("team", {})
        source_name = _clean_name(team.get("name", ""))
        standings.append({
            "position": row.get("position"),
            "team_id": team.get("id"),
            "team_name": TEAM_NAME_ALIASES.get(source_name, source_name),
            "crest": team.get("crest", ""),
            "played": row.get("playedGames", 0),
            "won": row.get("won", 0),
            "draw": row.get("draw", 0),
            "lost": row.get("lost", 0),
            "gf": row.get("goalsFor", 0),
            "ga": row.get("goalsAgainst", 0),
            "gd": row.get("goalDifference", 0),
            "points": row.get("points", 0),
            "form": row.get("form", ""),
        })
    _competition_standings_cache[cache_key] = (standings, _time.time())
    return standings


def get_last_completed_pl_match() -> int | None:
    """Most recently completed PL match ID."""
    cache = _populate_finished_cache(PL_CODE)
    if cache:
        return int(list(cache.keys())[-1])
    return None


def find_finished_match_by_teams(home_name: str, away_name: str,
                                 comp_code: str = PL_CODE) -> dict:
    """
    Find a finished match by team names for any supported league.
    Name comparison is fuzzy — strips common words before comparing.
    """
    cache = _populate_finished_cache(comp_code)

    def norm(s: str) -> str:
        s = _clean_name(s).casefold()
        s = re.sub(r"\b(?:fc|afc|cf|sad|sc|ac|bc|bv|sv|if|fk|sk|mk|rfc)\b", "", s)
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        aliases = {
            "man city": "manchester city",
            "man utd": "manchester united",
            "man united": "manchester united",
            "nott m forest": "nottingham forest",
            "notts forest": "nottingham forest",
            "spurs": "tottenham hotspur",
            "tottenham": "tottenham hotspur",
            "newcastle": "newcastle united",
            "leeds": "leeds united",
            "wolves": "wolverhampton wanderers",
            "brighton": "brighton hove albion",
            "west ham": "west ham united",
            "bournemouth": "bournemouth",
        }
        return aliases.get(s, s)

    h_n, a_n = norm(home_name), norm(away_name)

    for parsed in cache.values():
        if (norm(parsed.get("home_team", "")) == h_n and
                norm(parsed.get("away_team", "")) == a_n):
            return parsed
    return _empty_match()


def get_upcoming_fixtures_fd(comp_code: str = PD_CODE,
                              season: int | None = None,
                              max_fixtures: int = 20) -> list[dict]:
    """
    Upcoming (scheduled) fixtures for a competition from football-data.org.
    Used for La Liga (and any non-FPL league) fixture strips.
    Returns list of dicts with: match_id, home, away,
    kickoff_utc, kickoff_eat, gameweek.
    """
    from datetime import datetime, timezone, timedelta
    EAT = timezone(timedelta(hours=3))

    if season is None:
        season = _current_season()
    data = _fetch(f"competitions/{comp_code}/matches",
                  {"status": "SCHEDULED", "season": season})
    matches = data.get("matches", [])
    matches.sort(key=lambda m: m.get("utcDate", ""))

    result = []
    for m in matches[:max_fixtures]:
        home = _clean_name(m.get("homeTeam", {}).get("name", "Unknown"))
        away = _clean_name(m.get("awayTeam", {}).get("name", "Unknown"))
        ko_raw = m.get("utcDate", "")
        ko_eat = ""
        local_date = ""
        if ko_raw:
            try:
                ko_utc = datetime.fromisoformat(ko_raw.replace("Z", "+00:00"))
                ko_local = ko_utc.astimezone(EAT)
                ko_eat = ko_local.strftime("%a %d %b · %H:%M")
                local_date = ko_local.date().isoformat()
            except Exception:
                ko_eat = ko_raw[:10]
        result.append({
            "match_id"   : str(m.get("id", "")),
            "home"       : home,
            "away"       : away,
            "kickoff_utc": ko_raw,
            "kickoff_eat": ko_eat,
            "local_date" : local_date,
            "gameweek"   : m.get("matchday"),
        })
    return result


def get_fixtures_by_date_fd(comp_code: str, date_from: str,
                            date_to: str,
                            season: int | None = None) -> list[dict]:
    """Return all fixtures in an inclusive date range, including their scores."""
    from datetime import datetime, timezone, timedelta
    EAT = timezone(timedelta(hours=3))

    if season is None:
        season = _current_season()
    data = _fetch(f"competitions/{comp_code}/matches", {
        "dateFrom": date_from,
        "dateTo": date_to,
        "season": season,
    })

    result = []
    for match in data.get("matches", []):
        kickoff_utc = match.get("utcDate", "")
        if not kickoff_utc:
            continue
        try:
            kickoff = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
            kickoff_eat = kickoff.astimezone(EAT)
        except ValueError:
            continue

        score = match.get("score", {})
        full_time = score.get("fullTime", {})
        current = score.get("regularTime", {})
        if match.get("status") == "FINISHED":
            home_score = full_time.get("home")
            away_score = full_time.get("away")
        else:
            home_score = current.get("home")
            away_score = current.get("away")
        if home_score is None:
            home_score = full_time.get("home")
        if away_score is None:
            away_score = full_time.get("away")

        result.append({
            "match_id": str(match.get("id", "")),
            "home": _clean_name(match.get("homeTeam", {}).get("name", "Unknown")),
            "away": _clean_name(match.get("awayTeam", {}).get("name", "Unknown")),
            "kickoff_utc": kickoff_utc,
            "kickoff_eat": kickoff_eat.strftime("%a %d %b · %H:%M"),
            "local_date": kickoff_eat.date().isoformat(),
            "gameweek": match.get("matchday"),
            "status": _normalise_status(match.get("status", "")),
            "home_score": home_score,
            "away_score": away_score,
        })

    result.sort(key=lambda fixture: fixture["kickoff_utc"])
    return result


# ── Internal parsers ───────────────────────────────────────────────

# La Liga team name aliases: football-data.org name → our priors key
_LL_ALIASES: dict[str, str] = {
    "Athletic Club"          : "Athletic Club",
    "Club Atletico de Madrid": "Atletico Madrid",
    "Atletico de Madrid"     : "Atletico Madrid",
    "Real Betis Balompie"    : "Real Betis",
    "Deportivo Alaves"       : "Alaves",
    "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Girona FC"              : "Girona",
    "RCD Espanyol de Barcelona": "Espanyol",
    "RCD Mallorca"           : "Mallorca",
    "Real Valladolid CF"     : "Real Valladolid",
    "UD Las Palmas"          : "Las Palmas",
    "Real Oviedo"            : "Oviedo",
    "Elche CF"               : "Elche",
    "UD Levante"             : "Levante",
    "Getafe CF"              : "Getafe",
}


def _clean_name(name: str) -> str:
    """
    Normalise football-data.org team names.
    Works for both PL and La Liga.
    """
    # La Liga alias lookup first
    if name in _LL_ALIASES:
        return _LL_ALIASES[name]

    # Strip AFC prefix
    if name.startswith("AFC "):
        name = name[4:]

    # Strip common suffixes
    for suffix in [" FC", " AFC", " CF", " SAD",
                   " City FC", " United FC", " Town FC",
                   " Wanderers FC", " Rovers FC",
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
    return {"IN_PLAY": 70, "PAUSED": 45, "FINISHED": 90}.get(status_raw, 0)


def _parse_match(m: dict) -> dict:
    home_name  = _clean_name(m.get("homeTeam", {}).get("name", "Unknown Home"))
    away_name  = _clean_name(m.get("awayTeam", {}).get("name", "Unknown Away"))
    score      = m.get("score", {})
    ft         = score.get("fullTime", {})
    ht         = score.get("halfTime", {})
    current    = score.get("regularTime", {})
    if m.get("status") == "FINISHED":
        h_score = ft.get("home")
        a_score = ft.get("away")
    else:
        h_score = current.get("home")
        a_score = current.get("away")
    h_score    = h_score if h_score is not None else ft.get("home") or 0
    a_score    = a_score if a_score is not None else ft.get("away") or 0
    h_ht       = ht.get("home")
    a_ht       = ht.get("away")
    status_raw = m.get("status", "")
    minute     = _derive_minute(status_raw)
    odds       = m.get("odds", {}) or {}
    venue      = m.get("venue", "") or ""
    home_crest = m.get("homeTeam", {}).get("crest", "") or ""
    away_crest = m.get("awayTeam", {}).get("crest", "") or ""
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
        "odds_home"     : float(odds["homeWin"]) if odds.get("homeWin") else None,
        "odds_draw"     : float(odds["draw"])    if odds.get("draw")    else None,
        "odds_away"     : float(odds["awayWin"]) if odds.get("awayWin") else None,
        "referee"       : referee,
    }


def _parse_goals(m: dict) -> list[dict]:
    events = []
    for g in m.get("goals", []):
        minute = g.get("minute", "?")
        scorer = g.get("scorer", {}).get("name", "Unknown")
        team   = _clean_name(g.get("team", {}).get("name", ""))
        gtype  = g.get("type", "REGULAR")
        events.append({"time"  : f"{minute}'",
                        "detail": f"{scorer} ({team}) — {gtype}"})
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
