"""
bbs.py  —  V4.2
================
Big Balls Sports Data API client for La Liga (and other football leagues).
https://bigballsdata.com/la-liga-api

Free tier: 1,000 req/day
Auth: Bearer token in Authorization header
Base: https://api.bigballsdata.com/v1

Key endpoints used:
  GET /v1/matches?sport=football&league=laliga      — upcoming fixtures
  GET /v1/scores?sport=football&league=laliga       — live/recent scores (match_ids + scores)
  GET /v1/matches/{id}?sport=football               — single match detail (live status, score, teams)

Strategy:
  1. /v1/scores  → find match_ids with activity (low cost, 1 call)
  2. /v1/matches/{id} → get team names + accurate live status per match
  3. /v1/matches  → upcoming fixture strip (cached 10 min)

Rate budget (1000/day):
  - Scores poll: every 30s during matches = 180 calls/match × 2 matches = 360
  - Match detail: called per navigation click = ~20 calls
  - Fixtures: cached 10 min = ~6 calls/hour
  Total: ~400 calls on a matchday. Comfortable.

Environment variable: BBS_API_KEY in .env
"""

import json
import os
import ssl
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE_URL = "https://api.bigballsdata.com/v1"
EAT      = timezone(timedelta(hours=3))

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

# In-memory cache
_cache: dict = {}

# TTLs in seconds
_TTLS = {
    "fixtures"  : 600,   # 10 minutes — schedule rarely changes
    "scores"    : 30,    # 30 seconds — live scores
    "match"     : 30,    # 30 seconds — individual match detail
}


def _get_key() -> str:
    here = Path(__file__).resolve().parent
    for _ in range(4):
        env = here / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("BBS_API_KEY="):
                    return line.split("=", 1)[1].strip()
        here = here.parent
    return os.environ.get("BBS_API_KEY", "")


API_KEY = _get_key()


def has_key() -> bool:
    return bool(API_KEY)


def _fetch(endpoint: str) -> dict:
    """Make one authenticated GET request. Returns parsed JSON or {}."""
    if not API_KEY:
        return {}
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {API_KEY}",
        "Accept"       : "application/json",
        "User-Agent"   : "V4-Dashboard/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"[bbs] HTTP {e.code} on /{endpoint}: {e.reason}")
    except Exception as e:
        print(f"[bbs] error on /{endpoint}: {e}")
    return {}


def _cached(key: str, endpoint: str, ttl: int) -> dict | list:
    """Fetch with in-memory TTL cache."""
    now = time.time()
    if key in _cache and now - _cache[key][1] < ttl:
        return _cache[key][0]
    data = _fetch(endpoint)
    if data and not data.get("error"):
        _cache[key] = (data, now)
    elif key in _cache:
        print(f"[bbs] fetch failed for {key} — serving stale")
        return _cache[key][0]
    return data


# ── Name normalisation ─────────────────────────────────────────────
# BBS uses full Spanish names. Map to our priors keys.
_BBS_ALIASES = {
    "RC Deportivo La Coruña"        : "Deportivo La Coruna",
    "Real Racing Club de Santander" : "Racing Santander",
    "Alavés"                        : "Alaves",
    "Atlético Madrid"               : "Atletico Madrid",
    "Athletic Club"                 : "Athletic Club",
    "Sevilla FC"                    : "Sevilla",
    "Real Betis"                    : "Real Betis",
    "Real Sociedad"                 : "Real Sociedad",
    "Celta Vigo"                    : "Celta Vigo",
    "Rayo Vallecano"                : "Rayo Vallecano",
    "Villarreal"                    : "Villarreal",
    "Osasuna"                       : "Osasuna",
    "Getafe"                        : "Getafe",
    "Levante"                       : "Levante",
    "Valencia"                      : "Valencia",
    "Elche"                         : "Elche",
    "Espanyol"                      : "Espanyol",
    "Málaga"                        : "Malaga",
    "Real Madrid"                   : "Real Madrid",
    "Barcelona"                     : "Barcelona",
}


def _norm(name: str) -> str:
    return _BBS_ALIASES.get(name, name)


# ── Public API ─────────────────────────────────────────────────────

def get_today_matches(league: str = "laliga") -> list[dict]:
    """
    Get today's matches for a league with live scores.

    Strategy:
    1. Call /v1/scores to find match_ids with score activity (1 call)
    2. Call /v1/matches?... to get full fixture list with team names
    3. Merge scores into fixtures, filter to today

    Returns list of dicts:
      bbs_id, home, away, kickoff_utc, kickoff_eat,
      status, h_score, a_score, is_live, is_finished
    """
    if not API_KEY:
        return []

    today_utc = datetime.now(timezone.utc).date()

    # Step 1: Get scores (active match_ids + scores)
    scores_data = _cached(
        f"scores_{league}",
        f"scores?sport=football&league={league}",
        _TTLS["scores"]
    )
    score_map = {}  # match_id → {h_score, a_score, updated_at}
    for s in scores_data.get("data", {}).get("scores", {}).get("value", []):
        mid = s.get("match_id", "")
        if mid:
            score_map[mid] = {
                "h_score"   : s.get("home", 0),
                "a_score"   : s.get("away", 0),
                "updated_at": s.get("updated_at", ""),
            }

    # Step 2: Get fixture list (cached 10 min)
    fixtures_data = _cached(
        f"fixtures_{league}",
        f"matches?sport=football&league={league}",
        _TTLS["fixtures"]
    )
    all_matches = fixtures_data.get("data", [])
    if not isinstance(all_matches, list):
        all_matches = []

    # Step 3: Filter to today + enrich with scores
    result = []
    for m in all_matches:
        ko_raw = m.get("kickoff_utc", "")
        if not ko_raw:
            continue
        try:
            ko_dt  = datetime.fromisoformat(ko_raw.replace("Z", "+00:00"))
            ko_date = ko_dt.astimezone(timezone.utc).date()
        except Exception:
            continue

        if ko_date != today_utc:
            continue

        mid      = m.get("id", "")
        home_raw = m.get("home", {}).get("name", "")
        away_raw = m.get("away", {}).get("name", "")
        sc       = score_map.get(mid, {})
        h_score  = sc.get("h_score", m.get("score", {}) and m["score"].get("home", 0) or 0)
        a_score  = sc.get("a_score", m.get("score", {}) and m["score"].get("away", 0) or 0)
        status   = m.get("status", "scheduled")

        # Enrich with individual match detail if it has score activity
        if mid in score_map and status in ("scheduled", None):
            detail = get_match_detail(mid)
            if detail:
                status  = detail.get("status", status)
                h_score = detail.get("h_score", h_score)
                a_score = detail.get("a_score", a_score)

        ko_eat = ko_dt.astimezone(EAT).strftime("%H:%M EAT")

        result.append({
            "bbs_id"     : mid,
            "home"       : _norm(home_raw),
            "away"       : _norm(away_raw),
            "home_raw"   : home_raw,
            "away_raw"   : away_raw,
            "kickoff_utc": ko_raw,
            "kickoff_eat": ko_eat,
            "status"     : status,
            "h_score"    : h_score,
            "a_score"    : a_score,
            "is_live"    : status == "live",
            "is_finished": status in ("finished", "final"),
            "has_odds"   : m.get("has_odds", False),
        })

    return sorted(result, key=lambda x: x["kickoff_utc"])


def get_match_detail(bbs_id: str) -> dict | None:
    """
    Get full detail for a single match by BBS UUID.
    Returns normalised dict or None.
    Cached 30 seconds.
    """
    if not bbs_id or not API_KEY:
        return None

    cache_key = f"match_{bbs_id}"
    data = _cached(
        cache_key,
        f"matches/{bbs_id}?sport=football",
        _TTLS["match"]
    )
    m = data.get("data", {})
    if not m:
        return None

    score    = m.get("score") or {}
    h_score  = score.get("home", 0) or 0
    a_score  = score.get("away", 0) or 0
    ko_raw   = m.get("kickoff_utc", "")
    ko_eat   = ""
    if ko_raw:
        try:
            ko_eat = datetime.fromisoformat(
                ko_raw.replace("Z", "+00:00")
            ).astimezone(EAT).strftime("%a %d %b · %H:%M EAT")
        except Exception:
            pass

    home_raw = m.get("home", {}).get("name", "")
    away_raw = m.get("away", {}).get("name", "")
    status   = m.get("status", "scheduled")

    return {
        "bbs_id"      : m.get("id", bbs_id),
        "home"        : _norm(home_raw),
        "away"        : _norm(away_raw),
        "home_raw"    : home_raw,
        "away_raw"    : away_raw,
        "kickoff_utc" : ko_raw,
        "kickoff_eat" : ko_eat,
        "status"      : status,
        "h_score"     : h_score,
        "a_score"     : a_score,
        "is_live"     : status == "live",
        "is_finished" : status in ("finished", "final"),
        "has_odds"    : m.get("has_odds", False),
        "broadcast"   : m.get("broadcast", ""),
    }


def get_upcoming_fixtures(league: str = "laliga",
                           max_fixtures: int = 10) -> list[dict]:
    """
    Get upcoming (not yet played) fixtures for the fixture strip.
    Cached 10 minutes. Returns newest-to-oldest kickoff.
    """
    if not API_KEY:
        return []

    data = _cached(
        f"fixtures_{league}",
        f"matches?sport=football&league={league}",
        _TTLS["fixtures"]
    )
    matches = data.get("data", [])
    if not isinstance(matches, list):
        return []

    now_utc = datetime.now(timezone.utc)
    upcoming = []
    for m in matches:
        ko_raw = m.get("kickoff_utc", "")
        if not ko_raw:
            continue
        try:
            ko_dt = datetime.fromisoformat(ko_raw.replace("Z", "+00:00"))
        except Exception:
            continue
        if ko_dt <= now_utc:
            continue
        ko_eat = ko_dt.astimezone(EAT).strftime("%a %d %b · %H:%M EAT")
        upcoming.append({
            "bbs_id"     : m.get("id", ""),
            "home"       : _norm(m.get("home", {}).get("name", "")),
            "away"       : _norm(m.get("away", {}).get("name", "")),
            "kickoff_utc": ko_raw,
            "kickoff_eat": ko_eat,
            "status"     : m.get("status", "scheduled"),
        })

    upcoming.sort(key=lambda x: x["kickoff_utc"])
    return upcoming[:max_fixtures]


def find_today_match_by_teams(home_name: str,
                               away_name: str,
                               league: str = "laliga") -> dict | None:
    """
    Find today's match by team names (fuzzy).
    Used by match route to get live status.
    """
    matches = get_today_matches(league)

    def norm(s: str) -> str:
        return s.lower().replace("fc", "").replace("cf", "").strip()

    h_n = norm(home_name)
    a_n = norm(away_name)

    for m in matches:
        if (h_n in norm(m["home"]) or norm(m["home"]) in h_n) and \
           (a_n in norm(m["away"]) or norm(m["away"]) in a_n):
            return m
    return None
