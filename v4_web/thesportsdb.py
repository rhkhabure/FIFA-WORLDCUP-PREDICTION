"""
thesportsdb.py  —  V4.2
========================
TheSportsDB free API client for player profiles and career history.
https://www.thesportsdb.com/api.php

Free tier: test key "3", no sign-up required, ~30 req/min
Base URL : https://www.thesportsdb.com/api/v1/json/3

Endpoints used:
  GET /searchplayers.php?p={name}          — search by player name
  GET /lookupformerteams.php?id={player_id} — career history + team badges

No API key required for free tier.
"""

import json
import ssl
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

_cache: dict = {}
_PLAYER_TTL  = 3600   # player profiles — 1 hour
_HISTORY_TTL = 3600   # career history   — 1 hour


def _fetch(endpoint: str) -> dict:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "V4-Dashboard/1.0",
        "Accept"    : "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=8, context=_SSL_CTX) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"[tsdb] HTTP {e.code} on /{endpoint}")
    except Exception as e:
        print(f"[tsdb] error on /{endpoint}: {e}")
    return {}


def _cached(key: str, endpoint: str, ttl: int) -> dict:
    now = time.time()
    if key in _cache and now - _cache[key][1] < ttl:
        return _cache[key][0]
    data = _fetch(endpoint)
    if data:
        _cache[key] = (data, now)
    elif key in _cache:
        return _cache[key][0]
    return data


def _age(dob_str: str) -> int | None:
    """Return age in years from ISO date string e.g. '2001-09-05'."""
    try:
        dob  = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.now(timezone.utc)
        return today.year - dob.year - (
            (today.month, today.day) < (dob.month, dob.day)
        )
    except Exception:
        return None


def _fmt_dob(dob_str: str) -> str:
    """Format '2001-09-05' → '5 Sep 2001'."""
    try:
        return datetime.strptime(dob_str, "%Y-%m-%d").strftime("%-d %b %Y")
    except Exception:
        try:
            return datetime.strptime(dob_str, "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")
        except Exception:
            return dob_str


def search_player(name: str) -> dict | None:
    """
    Search TheSportsDB for a player by name.
    Returns normalised player dict or None.

    Returned fields:
      id, name, team, nationality, position, dob, age_str,
      status, photo_url (cutout), thumb_url, gender
    """
    if not name or len(name.strip()) < 2:
        return None

    key  = f"search_{name.lower().strip()}"
    data = _cached(key, f"searchplayers.php?p={urllib.parse.quote(name)}", _PLAYER_TTL)
    players = data.get("player", [])
    if not players:
        return None

    # Pick the best match — highest relevance (first result from API)
    p = players[0]

    dob = p.get("dateBorn", "") or ""
    return {
        "id"          : p.get("idPlayer", ""),
        "name"        : p.get("strPlayer", name),
        "team"        : p.get("strTeam", ""),
        "nationality" : p.get("strNationality", ""),
        "position"    : p.get("strPosition", ""),
        "dob"         : dob,
        "dob_display" : _fmt_dob(dob) if dob else "",
        "age"         : _age(dob) if dob else None,
        "status"      : p.get("strStatus", ""),
        "photo_url"   : p.get("strCutout", "") or "",
        "thumb_url"   : p.get("strThumb", "") or "",
    }


def get_career_history(player_id: str) -> list[dict]:
    """
    Fetch career history for a player by TheSportsDB player ID.
    Returns list of clubs, newest first (current club at top).

    Each entry:
      team_name, badge_url, joined, departed,
      appearances, goals, move_type, is_current
    """
    if not player_id:
        return []

    key  = f"history_{player_id}"
    data = _cached(key, f"lookupformerteams.php?id={player_id}", _HISTORY_TTL)
    entries = data.get("formerteams", [])
    if not entries or not isinstance(entries, list):
        return []

    result = []
    for e in entries:
        departed  = e.get("strDeparted", "") or ""
        is_current = departed == ""
        apps  = e.get("intAppearances")
        goals = e.get("intGoals")
        result.append({
            "team_name"  : e.get("strFormerTeam", ""),
            "badge_url"  : e.get("strBadge", "") or "",
            "joined"     : e.get("strJoined", "") or "",
            "departed"   : departed,
            "appearances": int(apps)  if apps  else None,
            "goals"      : int(goals) if goals else None,
            "move_type"  : e.get("strMoveType", "") or "",
            "is_current" : is_current,
        })

    # Sort: current clubs first, then by joined year descending
    def sort_key(e):
        try:
            yr = int(e["joined"]) if e["joined"] else 0
        except Exception:
            yr = 0
        return (0 if e["is_current"] else 1, -yr)

    return sorted(result, key=sort_key)


# Lazy import fix for urllib.parse
import urllib.parse
