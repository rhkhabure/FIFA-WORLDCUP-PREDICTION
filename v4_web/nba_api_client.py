"""
nba_api_client.py  —  NBA live data layer for v4_web
=====================================================
Wraps nba_api live endpoints with in-memory caching and stale-serve
fallback, following the same pattern as footballdata.py and bbs.py.

Exposes:
    get_today_scoreboard()          -> list[dict]   all games today
    get_live_game(game_id)          -> dict          one game's live state
    get_live_pbp(game_id)           -> list[dict]   play-by-play actions
    get_box_score(game_id)          -> dict          team + player stats
    build_live_snapshot(game_id)    -> dict          feature dict for nba_model

The nba_api library is rate-limited by stats.nba.com. All calls are
cached (TTL 25s for live data, 300s for static) and serve stale data
on any API failure so the dashboard never crashes mid-game.

Drop this file in v4_web/.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any

log = logging.getLogger(__name__)

# ── TTL constants (seconds) ───────────────────────────────────────────────────
TTL_LIVE    = 25    # scoreboard + PBP during live games
TTL_STATIC  = 300   # team metadata, box score halftime etc.
TTL_PREGAME = 60    # scoreboard before tip-off

# ── In-memory cache: {cache_key: (data, expires_at)} ─────────────────────────
_cache: dict[str, tuple[Any, float]] = {}


def _get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry and time.monotonic() < entry[1]:
        return entry[0]
    return None


def _set(key: str, data: Any, ttl: int) -> None:
    _cache[key] = (data, time.monotonic() + ttl)


def _get_stale(key: str) -> Any | None:
    """Return cached data regardless of TTL (serve-stale-on-error)."""
    entry = _cache.get(key)
    return entry[0] if entry else None


# ── nba_api imports (lazy so the module loads even without nba_api) ───────────
def _scoreboard():
    from nba_api.live.nba.endpoints import scoreboard as sb
    return sb.ScoreBoard()


def _playbyplay(game_id: str):
    from nba_api.live.nba.endpoints import playbyplay as pbp
    return pbp.PlayByPlay(game_id=game_id)


def _boxscore(game_id: str):
    from nba_api.live.nba.endpoints import boxscore as bs
    return bs.BoxScore(game_id=game_id)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Scoreboard (today's games)
# ─────────────────────────────────────────────────────────────────────────────

def get_today_scoreboard() -> list[dict]:
    """
    Returns a list of game dicts for today's NBA schedule.

    Each dict contains:
        game_id         str     e.g. "0022500042"
        home_team_id    int
        away_team_id    int
        home_tricode    str     e.g. "BOS"
        away_tricode    str     e.g. "LAL"
        home_name       str     e.g. "Celtics"
        away_name       str     e.g. "Lakers"
        home_city       str
        away_city       str
        home_score      int
        away_score      int
        status          str     "pre" | "live" | "final"
        status_text     str     e.g. "Q3 4:22" or "7:30 PM ET"
        period          int     0 = pre, 1-4 = quarters, 5+ = OT
        clock           str     e.g. "PT04M22.00S"
        game_time_utc   str     ISO 8601
    """
    key = "scoreboard_today"
    cached = _get(key)
    if cached is not None:
        return cached

    try:
        board = _scoreboard()
        raw_games = board.games.get_dict()
        games = []

        for g in raw_games:
            period      = g.get("period", 0)
            game_status = g.get("gameStatus", 1)  # 1=pre, 2=live, 3=final

            if game_status == 1:
                status = "pre"
            elif game_status == 2:
                status = "live"
            else:
                status = "final"

            # Build a readable status text
            clock = g.get("gameClock", "")
            if status == "live":
                period_label = f"OT{period - 4}" if period > 4 else f"Q{period}"
                clock_display = _format_clock(clock)
                status_text = f"{period_label} {clock_display}"
            elif status == "pre":
                status_text = g.get("gameStatusText", "")
            else:
                status_text = "Final"

            ht = g.get("homeTeam", {})
            at = g.get("awayTeam", {})

            games.append({
                "game_id":       g.get("gameId", ""),
                "home_team_id":  ht.get("teamId"),
                "away_team_id":  at.get("teamId"),
                "home_tricode":  ht.get("teamTricode", ""),
                "away_tricode":  at.get("teamTricode", ""),
                "home_name":     ht.get("teamName", ""),
                "away_name":     at.get("teamName", ""),
                "home_city":     ht.get("teamCity", ""),
                "away_city":     at.get("teamCity", ""),
                "home_score":    int(ht.get("score", 0) or 0),
                "away_score":    int(at.get("score", 0) or 0),
                "status":        status,
                "status_text":   status_text,
                "period":        period,
                "clock":         clock,
                "game_time_utc": g.get("gameTimeUTC", ""),
            })

        # Sort: live first, then pre by tip-off time, then final
        order = {"live": 0, "pre": 1, "final": 2}
        games.sort(key=lambda g: (order[g["status"]], g["game_time_utc"]))

        # Cache: shorter TTL if any game is live
        ttl = TTL_LIVE if any(g["status"] == "live" for g in games) else TTL_PREGAME
        _set(key, games, ttl)
        return games

    except Exception as exc:
        log.warning("Scoreboard fetch failed: %s", exc)
        stale = _get_stale(key)
        return stale if stale is not None else []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Live game state (single game)
# ─────────────────────────────────────────────────────────────────────────────

def get_live_game(game_id: str) -> dict:
    """
    Returns the current live state of one game.
    Pulls from today's scoreboard (already cached) to avoid a second call.
    """
    games = get_today_scoreboard()
    for g in games:
        if g["game_id"] == game_id:
            return g
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Play-by-play
# ─────────────────────────────────────────────────────────────────────────────

def get_live_pbp(game_id: str) -> list[dict]:
    """
    Returns the raw play-by-play action list for a live game.
    Each action is a dict with keys: actionNumber, clock, period,
    teamId, actionType, subType, scoreHome, scoreAway, description, etc.
    """
    key = f"pbp_{game_id}"
    cached = _get(key)
    if cached is not None:
        return cached

    try:
        pbp  = _playbyplay(game_id)
        acts = pbp.actions.get_dict()
        _set(key, acts, TTL_LIVE)
        return acts
    except Exception as exc:
        log.warning("PBP fetch failed for %s: %s", game_id, exc)
        stale = _get_stale(key)
        return stale if stale is not None else []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Box score
# ─────────────────────────────────────────────────────────────────────────────

def get_box_score(game_id: str) -> dict:
    """
    Returns team and player stats for a game.
    Cached for TTL_STATIC (300s) since box score changes slowly.
    """
    key = f"box_{game_id}"
    cached = _get(key)
    if cached is not None:
        return cached

    try:
        box  = _boxscore(game_id)
        data = {
            "home": box.home_team.get_dict(),
            "away": box.away_team.get_dict(),
        }
        _set(key, data, TTL_STATIC)
        return data
    except Exception as exc:
        log.warning("BoxScore fetch failed for %s: %s", game_id, exc)
        stale = _get_stale(key)
        return stale if stale is not None else {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Live snapshot builder
# Converts raw PBP + game state into a feature dict for nba_model.predict_live
# ─────────────────────────────────────────────────────────────────────────────

def build_live_snapshot(
    game_id: str,
    home_elo: float = 1500.0,
    away_elo: float = 1500.0,
    is_playoffs: int = 0,
    home_avail_delta: float = 0.0,
    away_avail_delta: float = 0.0,
) -> dict:
    """
    Builds the feature snapshot for nba_model.predict_live() from live data.

    This is the glue between nba_api_client and nba_model — it fetches
    the current game state, processes the PBP, and returns a dict with
    all 16 features pre-computed and ready for inference.

    Returns {} if no live data is available.
    """
    game = get_live_game(game_id)
    if not game:
        return {}

    pbp = get_live_pbp(game_id)
    if not pbp:
        # Return a partial snapshot using only scoreboard data
        return _snapshot_from_scoreboard(game, home_elo, away_elo,
                                         is_playoffs, home_avail_delta,
                                         away_avail_delta)

    return _snapshot_from_pbp(game, pbp, home_elo, away_elo,
                               is_playoffs, home_avail_delta, away_avail_delta)


def _snapshot_from_scoreboard(
    game: dict,
    home_elo: float,
    away_elo: float,
    is_playoffs: int,
    home_avail_delta: float,
    away_avail_delta: float,
) -> dict:
    """Minimal snapshot using scoreboard data only (no PBP)."""
    period     = game.get("period", 1) or 1
    clock      = game.get("clock", "")
    clock_sec  = _parse_clock(clock)
    time_rem   = _time_remaining(period, clock_sec)
    score_diff = game.get("home_score", 0) - game.get("away_score", 0)
    period_len = 300.0 if period >= 5 else 720.0
    qte        = 1.0 - min(clock_sec / period_len, 1.0) if period_len > 0 else 0.0

    return {
        "score_diff":              score_diff,
        "time_remaining_sec":      time_rem,
        "quarter":                 period,
        "quarter_time_elapsed_pct": qte,
        "home_elo":                home_elo,
        "away_elo":                away_elo,
        "is_playoffs":             is_playoffs,
        "is_overtime":             int(period >= 5),
        "lead_changes_norm":       0.0,   # unknown without PBP
        "possession":              0.5,   # unknown without PBP
        "home_in_bonus":           0,
        "away_in_bonus":           0,
        "home_avail_delta":        home_avail_delta,
        "away_avail_delta":        away_avail_delta,
    }


def _snapshot_from_pbp(
    game: dict,
    pbp: list[dict],
    home_elo: float,
    away_elo: float,
    is_playoffs: int,
    home_avail_delta: float,
    away_avail_delta: float,
) -> dict:
    """
    Full snapshot computed from the play-by-play action list.
    Mirrors the two-pass logic from Phase 1 phase1_data_pipeline.py.
    """
    home_team_id = game.get("home_team_id")
    period       = game.get("period", 1) or 1
    clock        = game.get("clock", "")
    clock_sec    = _parse_clock(clock)
    time_rem     = _time_remaining(period, clock_sec)
    home_score   = game.get("home_score", 0)
    away_score   = game.get("away_score", 0)
    score_diff   = home_score - away_score

    period_len   = 300.0 if period >= 5 else 720.0
    qte          = 1.0 - min(clock_sec / period_len, 1.0)

    # ── Accumulate state from full PBP (two-pass mirrors Phase 1) ────────────
    home_fouls  = 0
    away_fouls  = 0
    prev_per    = None
    home_bonus  = 0
    away_bonus  = 0
    possession  = 0.5      # default neutral
    lead_changes = 0
    prev_leader  = 0
    play_count   = 0

    for action in pbp:
        atype = str(action.get("actionType", ""))
        sub   = str(action.get("subType", ""))
        tid   = action.get("teamId")
        per   = int(action.get("period", period))

        # Reset fouls on new quarter
        if per != prev_per:
            home_fouls = 0
            away_fouls = 0
            prev_per   = per

        # Foul accumulation
        if atype == "Foul" and "Offensive Foul Turnover" not in sub:
            if tid == home_team_id:
                home_fouls += 1
            else:
                away_fouls += 1

        # Bonus: home in bonus when away has ≥5 fouls (and vice versa)
        home_bonus = int(away_fouls >= 5)
        away_bonus = int(home_fouls >= 5)

        # Possession: last scored play = team that had the ball
        if atype in ("Made Shot", "Free Throw") and action.get("scoreHome"):
            possession = 1.0 if tid == home_team_id else 0.0

        # Lead changes
        h = int(action.get("scoreHome") or 0)
        a = int(action.get("scoreAway") or 0)
        if h or a:
            play_count += 1
            diff   = h - a
            leader = 1 if diff > 0 else (-1 if diff < 0 else 0)
            if leader != 0 and leader != prev_leader and prev_leader != 0:
                lead_changes += 1
            prev_leader = leader

    lead_changes_norm = lead_changes / max(play_count, 1)

    return {
        "score_diff":               score_diff,
        "time_remaining_sec":       time_rem,
        "quarter":                  period,
        "quarter_time_elapsed_pct": qte,
        "home_elo":                 home_elo,
        "away_elo":                 away_elo,
        "is_playoffs":              is_playoffs,
        "is_overtime":              int(period >= 5),
        "lead_changes_norm":        lead_changes_norm,
        "possession":               possession,
        "home_in_bonus":            home_bonus,
        "away_in_bonus":            away_bonus,
        "home_avail_delta":         home_avail_delta,
        "away_avail_delta":         away_avail_delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Clock and time helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_clock(clock: str) -> float:
    """
    Parse NBA clock strings to seconds remaining in the current period.
    Handles ISO 8601 format 'PT04M22.00S' and fallback 'MM:SS'.
    Returns 0.0 on any parse failure.
    """
    if not clock:
        return 0.0
    import re
    # ISO 8601: PT04M22.00S
    m = re.match(r"PT(\d+)M([\d.]+)S", clock, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    # MM:SS fallback
    m = re.match(r"^(\d{1,2}):(\d{2})$", clock)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(clock)
    except ValueError:
        return 0.0


def _time_remaining(period: int, clock_sec: float) -> float:
    """
    Total seconds left until end of regulation.
    Regulation = 4 × 720s = 2880s. OT returns 0.
    """
    if period <= 4:
        return clock_sec + max(0, 4 - period) * 720.0
    return 0.0


def _format_clock(clock: str) -> str:
    """Convert PT04M22.00S → 4:22 for display."""
    secs = _parse_clock(clock)
    mins = int(secs // 60)
    sec  = int(secs % 60)
    return f"{mins}:{sec:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EAT conversion helper
# ─────────────────────────────────────────────────────────────────────────────

def to_eat(utc_str: str) -> str:
    """
    Convert a UTC ISO 8601 game time string to EAT (UTC+3) display format.
    e.g. "2025-10-22T00:30:00Z" → "03:30 EAT"
    Returns empty string on parse failure.
    """
    if not utc_str:
        return ""
    try:
        # Handle both "Z" suffix and "+00:00"
        utc_str = utc_str.replace("Z", "+00:00")
        dt_utc  = datetime.fromisoformat(utc_str)
        dt_eat  = dt_utc.astimezone(timezone(timedelta(hours=3)))
        return dt_eat.strftime("%-H:%M EAT")
    except Exception:
        return utc_str


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Cache management
# ─────────────────────────────────────────────────────────────────────────────

def clear_cache(game_id: str | None = None) -> None:
    """
    Clear cache entries. If game_id given, clears only that game's entries.
    Call from the live poll endpoint after a game goes final.
    """
    global _cache
    if game_id:
        keys = [k for k in _cache if game_id in k]
        for k in keys:
            del _cache[k]
    else:
        _cache.clear()


def cache_stats() -> dict:
    """Return cache health info for /nba/debug endpoint."""
    now = time.monotonic()
    live  = sum(1 for _, exp in _cache.values() if exp > now)
    stale = len(_cache) - live
    return {"total": len(_cache), "live": live, "stale": stale}
