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
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from constants import EAT, LEAGUE_FILTERS

PARSE_BASE = "https://api.parse.bot/scraper/645b8e03-271d-4c85-97e7-35d5733a2d78"

# In-memory caches
_lineup_cache: dict[str, dict] = {}          # match_id -> lineup data
_xg_cache:     dict[str, tuple] = {}         # match_id -> (data, timestamp)
XG_CACHE_TTL = 60  # seconds -- actual TTL controlled by poll_interval in JS
DATE_CACHE_TTL = 24 * 60 * 60
MATCH_DETAIL_CACHE_TTL = 10 * 60
LINEUP_RETRY_TTL = 15 * 60
MATCH_WINDOW_AFTER_KICKOFF = 4 * 60 * 60
_api_cache_lock = threading.RLock()


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


def _persisted_fetch(
    endpoint: str,
    params: dict,
    cache_key: str,
    ttl_seconds: int,
) -> dict:
    """Fetch once per TTL and retain the response across application restarts."""
    from predictions import get_provider_api_cache, save_provider_api_cache

    with _api_cache_lock:
        cached = get_provider_api_cache(cache_key)
        failure_key = f"{cache_key}:failure"
        recent_failure = get_provider_api_cache(failure_key)
        now = time.time()
        if recent_failure and now - recent_failure["fetched_at"] < min(ttl_seconds, 300):
            return cached["payload"] if cached else recent_failure["payload"]
        if cached:
            if (
                cached["payload"].get("status") == "success"
                and now - cached["fetched_at"] < ttl_seconds
            ):
                return cached["payload"]
        payload = _fetch(endpoint, params)
        if payload.get("status") == "success":
            save_provider_api_cache(cache_key, now, payload)
        else:
            save_provider_api_cache(failure_key, now, payload)
            return cached["payload"] if cached else payload
        return payload


def _parse_kickoff(kickoff_utc: str | None):
    if not kickoff_utc:
        return None
    try:
        kickoff = datetime.fromisoformat(kickoff_utc.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return None
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    return kickoff.astimezone(timezone.utc)


def _match_window_is_active(kickoff_utc: str | None) -> bool:
    """Allow live endpoints only from kickoff through four hours afterward."""
    kickoff = _parse_kickoff(kickoff_utc)
    if not kickoff:
        return False
    now = datetime.now(timezone.utc)
    return kickoff <= now <= kickoff + timedelta(seconds=MATCH_WINDOW_AFTER_KICKOFF)


def _match_date_has_active_fixture(date_str: str) -> bool:
    """Use logged kickoff times as the authority for whether today's feed is needed."""
    from predictions import get_all_predictions

    now = datetime.now(timezone.utc)
    try:
        predictions = get_all_predictions()
    except Exception as e:
        print(f"[fotmob] scheduled fixture lookup failed: {e}")
        return False
    for prediction in predictions:
        if prediction.get("actual_result"):
            continue
        kickoff = _parse_kickoff(prediction.get("kickoff_utc"))
        if not kickoff or kickoff.strftime("%Y%m%d") != date_str:
            continue
        if kickoff <= now <= kickoff + timedelta(seconds=MATCH_WINDOW_AFTER_KICKOFF):
            return True
    return False


# ── Public API ─────────────────────────────────────────────────────────────────

def has_key() -> bool:
    """True if a PARSE_BOT_KEY is configured."""
    return bool(API_KEY)


def get_lineup(
    match_id: str | int, kickoff_utc: str | None = None
) -> dict | None:
    """
    Fetch confirmed lineup for a match.

    Lineup requests are match-window-gated. Confirmed lineups persist; empty
    responses are retried after a short cooldown.
    """
    key = str(match_id)
    if not API_KEY or not _match_window_is_active(kickoff_utc):
        return None

    ts_key = f"{key}_ts"
    if key in _lineup_cache:
        cached = _lineup_cache[key]
        cached_status = (cached.get("match_status") or "").lower()
        if cached_status in ("scheduled", "timed", ""):
            # Re-fetch if older than 5 minutes — game may have started
            if time.time() - _lineup_cache.get(ts_key, 0) < 300:
                return cached
            # else fall through to re-fetch
        else:
            return cached  # live/finished — lineup is fixed

    from predictions import get_provider_api_cache
    cache_key = f"fotmob:lineup:{key}"
    cached_response = get_provider_api_cache(cache_key)
    if cached_response:
        raw = cached_response["payload"]
        ttl = 365 * 24 * 60 * 60 if _raw_has_lineup(raw) else LINEUP_RETRY_TTL
        if time.time() - cached_response["fetched_at"] < ttl:
            data = raw
        else:
            data = _persisted_fetch(
                "get_match_lineup", {"match_id": match_id}, cache_key, ttl
            )
    else:
        data = _persisted_fetch(
            "get_match_lineup",
            {"match_id": match_id},
            cache_key,
            LINEUP_RETRY_TTL,
        )
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

    # Cache lineup with timestamp
    _lineup_cache[key] = result
    _lineup_cache[f"{key}_ts"] = time.time()
    return result


def _raw_has_lineup(data: dict) -> bool:
    match = data.get("data", {})
    return bool(
        (match.get("home_team", {}).get("players") or [])
        or (match.get("away_team", {}).get("players") or [])
    )


def get_match_details(
    match_id: str | int, kickoff_utc: str | None = None
) -> dict | None:
    """
    Fetch match status and events, refreshing no more than every 10 minutes.
    """
    if not API_KEY:
        return None

    key = f"details_{match_id}"
    cache_key = f"fotmob:details:{match_id}"
    from predictions import get_provider_api_cache

    cached = get_provider_api_cache(cache_key)
    if _match_window_is_active(kickoff_utc):
        data = _persisted_fetch(
            "get_match_details",
            {"match_id": match_id},
            cache_key,
            MATCH_DETAIL_CACHE_TTL,
        )
    else:
        data = cached["payload"] if cached else None
    if not data or data.get("status") != "success":
        return None

    d = data.get("data", {})
    result = {
        "status"      : d.get("match_status", ""),
        "live_minute" : d.get("live_minute"),
        "score"       : d.get("score", {"home": 0, "away": 0}),
        "events"      : d.get("events", []),
    }
    _xg_cache[key] = (result, time.time())
    return result


def _refresh_date_cache_if_stale(date_str: str):
    """Fetch a date feed once daily, only during a logged fixture's match window."""
    cache_key  = f"matches_{date_str}"
    failed_key = f"{cache_key}_failed"
    failed_ts  = f"{failed_key}_ts"

    if _lineup_cache.get(failed_key):
        fail_time = _lineup_cache.get(failed_ts, 0)
        if time.time() - fail_time < 300:
            return
        _lineup_cache.pop(failed_key, None)
        _lineup_cache.pop(failed_ts, None)

    from predictions import get_provider_api_cache
    cached = get_provider_api_cache(f"fotmob:{cache_key}")
    if cached:
        if time.time() - cached["fetched_at"] < DATE_CACHE_TTL:
            _lineup_cache[cache_key] = cached["payload"]
            _lineup_cache[f"{cache_key}_ts"] = cached["fetched_at"]
            return
    _lineup_cache.pop(cache_key, None)

    if not _match_date_has_active_fixture(date_str):
        return

    data = _persisted_fetch(
        "get_matches_by_date",
        {"date": date_str},
        f"fotmob:{cache_key}",
        DATE_CACHE_TTL,
    )
    if data.get("status") == "success":
        _lineup_cache[cache_key] = data
        _lineup_cache[f"{cache_key}_ts"] = time.time()
    else:
        _lineup_cache[failed_key] = True
        _lineup_cache[failed_ts]  = time.time()


def get_match_status_from_date_cache(match_id: str | int,
                                      home_name: str = "",
                                      away_name: str = "") -> dict | None:
    """
    Extract live status for a match from the already-cached get_matches_by_date
    response. Zero extra credits — uses the cached date response.

    Returns dict:
      started    : bool
      finished   : bool
      ongoing    : bool
      is_halftime: bool
      home_score : int
      away_score : int
      minute_str : str   e.g. "48", "HT", "FT"
      minute_int : int   numeric minute (45 for HT, 90 for FT, actual for live)
    Or None if match not found in cache.
    """
    from datetime import datetime, timezone
    date_str  = datetime.now(timezone.utc).strftime("%Y%m%d")
    cache_key = f"matches_{date_str}"

    cached_at = _lineup_cache.get(f"{cache_key}_ts")
    if (
        cache_key in _lineup_cache
        and isinstance(cached_at, (int, float))
        and time.time() - cached_at >= DATE_CACHE_TTL
    ):
        _lineup_cache.pop(cache_key, None)
    if cache_key not in _lineup_cache:
        from predictions import get_provider_api_cache
        cached = get_provider_api_cache(f"fotmob:{cache_key}")
        if cached and time.time() - cached["fetched_at"] < DATE_CACHE_TTL:
            _lineup_cache[cache_key] = cached["payload"]
            _lineup_cache[f"{cache_key}_ts"] = cached["fetched_at"]
    if cache_key not in _lineup_cache:
        return None

    data    = _lineup_cache[cache_key]
    mid_str = str(match_id)

    # Normalise team name for fuzzy matching
    def _norm(s: str) -> str:
        s = s.lower()
        for w in ["fc", "afc", "city", "united", "hotspur", "wanderers",
                  "& hove albion", "town", "forest"]:
            s = s.replace(w, "").strip()
        return s.strip()

    home_n = _norm(home_name) if home_name else ""
    away_n = _norm(away_name) if away_name else ""

    leagues = data.get("data", {}).get("leagues", [])
    for league in leagues:
        for match in league.get("matches", []):
            fm_id = str(match.get("id", ""))
            mh    = _norm(match.get("home", {}).get("name", ""))
            ma    = _norm(match.get("away", {}).get("name", ""))

            id_match   = fm_id == mid_str and mid_str not in ("", "0")
            name_match = (home_n and away_n and
                          (home_n in mh or mh in home_n) and
                          (away_n in ma or ma in away_n))

            if not id_match and not name_match:
                continue

            st       = match.get("status", {}) or {}
            started  = bool(st.get("started", False))
            finished = bool(st.get("finished", False))
            ongoing  = bool(st.get("ongoing", False))

            # Parse score from "1 - 2" string
            score_str  = st.get("scoreStr", "0 - 0") or "0 - 0"
            home_score = away_score = 0
            if " - " in score_str:
                parts = score_str.split(" - ")
                try:
                    home_score = int(parts[0].strip())
                    away_score = int(parts[1].strip())
                except (ValueError, IndexError):
                    pass

            # Parse minute
            live_time  = st.get("liveTime", {}) or {}
            # Finished matches use "reason" dict
            reason     = st.get("reason", {}) or {}
            minute_str = ""
            minute_int = 0
            is_halftime = False

            if finished:
                minute_str  = "FT"
                minute_int  = 90
            elif live_time:
                short_key   = live_time.get("shortKey", "")
                short       = live_time.get("short", "")
                # Strip invisible unicode characters from minute string
                short_clean = "".join(
                    c for c in short if c.isprintable() and ord(c) < 0x200b
                ).strip().rstrip("'").strip()

                if short_key == "halftime_short" or short_clean == "HT":
                    is_halftime = True
                    minute_str  = "HT"
                    minute_int  = 45
                else:
                    minute_str = short_clean
                    try:
                        minute_int = int(short_clean)
                    except ValueError:
                        minute_int = live_time.get("basePeriod", 45)

            return {
                "started"    : started,
                "finished"   : finished,
                "ongoing"    : ongoing,
                "is_halftime": is_halftime,
                "home_score" : home_score,
                "away_score" : away_score,
                "minute_str" : minute_str,
                "minute_int" : minute_int,
            }

    return None  # not in cache


def get_live_xg(
    match_id: str | int,
    max_age_seconds: int = 300,
    kickoff_utc: str | None = None,
) -> dict | None:
    """
    Fetch xG for an active fixture, with a persistent max_age_seconds cooldown.

    Returns dict with:
      home_xg, away_xg,
      home_xg_h1, away_xg_h1,
      home_xg_h2, away_xg_h2
    Or None if unavailable / match hasn't started.
    """
    return get_live_xg_for_fixture(match_id, kickoff_utc, max_age_seconds)


def get_live_xg_for_fixture(
    match_id: str | int,
    kickoff_utc: str | None,
    max_age_seconds: int = 300,
) -> dict | None:
    """Fetch xG only during the fixture window and persist its cooldown."""
    key = f"xg_{match_id}"
    cache_key = f"fotmob:xg:{match_id}"
    from predictions import get_provider_api_cache

    cached = get_provider_api_cache(cache_key)
    if cached and time.time() - cached["fetched_at"] < max_age_seconds:
        result = _parse_xg_response(cached["payload"])
        if result:
            _xg_cache[key] = (result, cached["fetched_at"])
        return result
    if not API_KEY or not _match_window_is_active(kickoff_utc):
        return None

    data = _persisted_fetch(
        "get_match_xg",
        {"match_id": match_id},
        cache_key,
        max_age_seconds,
    )
    result = _parse_xg_response(data)
    if not result:
        return None
    _xg_cache[key] = (result, time.time())
    return result


def _parse_xg_response(data: dict) -> dict | None:
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


def fotmob_to_fpl_team_name(fotmob_name: str) -> str:
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
    mapped_name = _MAP.get(fotmob_name, fotmob_name)
    from v4_backend.feature_builder import TEAM_NAME_ALIASES

    return TEAM_NAME_ALIASES.get(mapped_name, mapped_name)


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

    cache_key  = f"matches_{date_str}"
    failed_key = f"{cache_key}_failed"

    _refresh_date_cache_if_stale(date_str)
    if cache_key not in _lineup_cache:
        return None
    data = _lineup_cache[cache_key]

    def norm(s: str) -> str:
        return (
            fotmob_to_fpl_team_name(s)
            .lower()
            .replace("fc", "")
            .replace("afc", "")
            .strip()
        )

    home_n = norm(home_name)
    away_n = norm(away_name)

    leagues = data.get("data", {}).get("leagues", [])
    for league in leagues:
        for match in league.get("matches", []):
            mh = norm(match.get("home", {}).get("name", ""))
            ma = norm(match.get("away", {}).get("name", ""))
            if (home_n in mh or mh in home_n) and (away_n in ma or ma in away_n):
                return str(match.get("id"))
    return None


def get_matches_by_date_for_league(date_str: str, league_key: str) -> list[dict]:
    """
    Return all FotMob matches for a given league on a given date.

    date_str  : YYYYMMDD e.g. "20260916"
    league_key: our internal key e.g. "laliga", "bundesliga"

    Returns list of raw FotMob match dicts with id, home, away, status.
    """
    # Map our league key → FotMob league name fragments to match on
    filters = LEAGUE_FILTERS.get(league_key, [])

    # Use the persisted daily date cache; never bypass match-day gating.
    _refresh_date_cache_if_stale(date_str)
    data = _lineup_cache.get(f"matches_{date_str}") or {}

    leagues = data.get("data", {}).get("leagues", [])
    matches = []
    for lg in leagues:
        name = lg.get("name", "").lower()
        if any(f in name for f in filters):
            for m in lg.get("matches", []):
                matches.append({
                    "id"    : m.get("id"),
                    "home"  : m.get("home", {}).get("name", ""),
                    "away"  : m.get("away", {}).get("name", ""),
                    "status": m.get("status", {}),
                })
    return matches