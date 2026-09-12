"""
prediction_job.py  —  V4.2
===========================
Background prediction logging job.

Called on FastAPI startup and every 30 minutes via asyncio.

What it does each cycle:
  1. Fetch all upcoming PL fixtures from FPL API
  2. For each fixture:
       a. If not yet in DB → compute DC prior → log snapshot 1
       b. If snapshot 1 exists but not snapshot 2 → try FotMob lineup
          → if lineup confirmed → compute adjusted odds → log snapshot 2
  3. Fetch recent finished PL matches → fill in actual results

Runs in the background — never blocks the web server.
Credits used: get_matches_by_date (3) once per day, cached.
              get_match_lineup (1) per new confirmed lineup only.
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Lazy imports to avoid circular at module level
def _imports():
    from fpl import get_upcoming_fixtures
    from footballdata import get_live_match_data
    from fotmob import get_fotmob_match_id, get_lineup, has_key as fotmob_ok
    from lineup_adjustment import compute_lineup_adjusted_odds, get_absent_key_players
    from predictions import (init_db, log_pre_lineup, log_post_lineup,
                             log_result, get_all_predictions)
    return (get_upcoming_fixtures, get_live_match_data,
            get_fotmob_match_id, get_lineup, fotmob_ok,
            compute_lineup_adjusted_odds, get_absent_key_players,
            init_db, log_pre_lineup, log_post_lineup,
            log_result, get_all_predictions)


def _dc_probs(home_name: str, away_name: str, priors_db: dict,
              league_key: str) -> tuple | None:
    """
    Compute DC pre-game probabilities for a fixture.
    Returns (home%, draw%, away%, lam, mu) or None if teams not in priors.
    """
    from v4_backend.feature_builder import TEAM_NAME_ALIASES
    from scipy.stats import poisson

    league_data = priors_db.get(league_key, {})
    teams       = league_data.get("teams", {})
    meta        = league_data.get("meta", {})

    hk = TEAM_NAME_ALIASES.get(home_name, home_name)
    ak = TEAM_NAME_ALIASES.get(away_name, away_name)
    h  = teams.get(hk, {})
    a  = teams.get(ak, {})

    # Fallback: bottom-quartile estimates for promoted teams
    all_alpha = [v["alpha"] for v in teams.values()]
    all_beta  = [v["beta"]  for v in teams.values()]
    q25a = float(np.percentile(all_alpha, 25))
    q75b = float(np.percentile(all_beta,  75))

    h_alpha = h.get("alpha", q25a)
    h_beta  = h.get("beta",  q75b)
    a_alpha = a.get("alpha", q25a)
    a_beta  = a.get("beta",  q75b)

    gamma = meta.get("gamma_home_advantage", 1.25)
    rho   = meta.get("rho_draw_correction",  0.0)
    dp    = 0.10   # draw propensity

    lam = float(np.clip(h_alpha * a_beta * gamma, 1e-5, 15.0))
    mu  = float(np.clip(a_alpha * h_beta,          1e-5, 15.0))

    N  = 9
    hp = poisson.pmf(np.arange(N), lam)
    ap = poisson.pmf(np.arange(N), mu)
    j  = np.outer(hp, ap)

    j[0,0] *= max(1.0 - lam * mu * rho, 1e-5)
    j[1,0] *= max(1.0 + mu * rho,       1e-5)
    j[0,1] *= max(1.0 + lam * rho,      1e-5)
    j[1,1] *= max(1.0 - rho,            1e-5)
    j /= j.sum()

    ph  = float(np.tril(j,-1).sum())
    pd_ = float(np.trace(j))
    pa  = float(np.triu(j,+1).sum())

    tr  = dp / 2.0
    ph2 = max(ph - tr, 0.0)
    pa2 = max(pa - tr, 0.0)
    pd2 = pd_ + dp
    tot = ph2 + pd2 + pa2

    return (
        round(ph2/tot*100,1),
        round(pd2/tot*100,1),
        round(pa2/tot*100,1),
        round(lam,3),
        round(mu,3),
    )


async def run_prediction_job(priors_db: dict, league_key: str):
    """
    Main job cycle. Safe to call repeatedly — all writes are idempotent.
    """
    (get_upcoming_fixtures, get_live_match_data,
     get_fotmob_match_id, get_lineup, fotmob_ok,
     compute_lineup_adjusted_odds, get_absent_key_players,
     init_db, log_pre_lineup, log_post_lineup,
     log_result, get_all_predictions) = _imports()

    init_db()
    now_utc = datetime.now(timezone.utc)
    season  = now_utc.year if now_utc.month >= 8 else now_utc.year - 1

    print(f"[prediction_job] cycle start {now_utc.strftime('%H:%M:%S UTC')}")

    # ── Step 1: Upcoming fixtures → snapshot 1 ─────────────────────────────
    try:
        upcoming = get_upcoming_fixtures(max_fixtures=50)
    except Exception as e:
        print(f"[prediction_job] FPL fetch error: {e}")
        upcoming = []

    for f in upcoming:
        mid      = str(f.get("match_id", ""))
        home     = f.get("home", "")
        away     = f.get("away", "")
        kickoff  = f.get("kickoff_utc", f.get("kickoff_eat", ""))

        if not mid or not home or not away:
            continue

        # Compute DC prior for this fixture
        try:
            result = _dc_probs(home, away, priors_db, league_key)
        except Exception as e:
            print(f"[prediction_job] DC error {home} v {away}: {e}")
            continue

        if not result:
            continue

        ph, pd, pa, lam, mu = result

        # Log snapshot 1 (no-op if already logged)
        try:
            log_pre_lineup(
                match_id=mid, home_team=home, away_team=away,
                kickoff_utc=str(kickoff), season=season,
                dc_home=ph, dc_draw=pd, dc_away=pa,
                dc_lam=lam, dc_mu=mu,
            )
        except Exception as e:
            print(f"[prediction_job] log_pre error {mid}: {e}")

    # ── Step 2: Check for confirmed lineups → snapshot 2 ───────────────────
    if fotmob_ok():
        existing = get_all_predictions(season)
        needs_lineup = [
            p for p in existing
            if p["pre_logged_at"] and not p["adj_logged_at"]
            and not p["actual_result"]  # not finished yet
        ]

        for p in needs_lineup:
            mid   = p["match_id"]
            home  = p["home_team"]
            away  = p["away_team"]

            try:
                today_str = now_utc.strftime("%Y%m%d")
                fotmob_id = get_fotmob_match_id(home, away, today_str)
                if not fotmob_id:
                    continue
                lineup = get_lineup(fotmob_id)
                if not lineup or not lineup.get("home_players"):
                    continue

                # Has a confirmed lineup — compute adjusted odds
                adj, adj_lam, adj_mu = compute_lineup_adjusted_odds(
                    home_name=home, away_name=away,
                    home_lineup=lineup["home_players"],
                    away_lineup=lineup["away_players"],
                    base_lam=p["pre_dc_lam"],
                    base_mu=p["pre_dc_mu"],
                )
                absent_h = get_absent_key_players(home, lineup["home_players"])
                absent_a = get_absent_key_players(away, lineup["away_players"])

                log_post_lineup(
                    match_id=mid,
                    adj_home=adj[0], adj_draw=adj[1], adj_away=adj[2],
                    adj_lam=adj_lam, adj_mu=adj_mu,
                    absent_home=absent_h, absent_away=absent_a,
                )
                print(f"[prediction_job] lineup logged: {home} v {away}")

            except Exception as e:
                print(f"[prediction_job] lineup error {mid}: {e}")

    # ── Step 3: Log results for finished matches ────────────────────────────
    # Only check matches where kickoff has passed (avoid hammering API)
    existing = get_all_predictions(season)
    unresolved = [
        p for p in existing
        if p["pre_logged_at"] and not p["actual_result"]
        and p["kickoff_utc"]   # must have a kickoff time
        and p["kickoff_utc"] < now_utc.isoformat()  # kickoff must be in the past
    ]

    checked = 0
    MAX_RESULTS_PER_CYCLE = 5  # stay well within 10 req/min limit
    for p in unresolved[:MAX_RESULTS_PER_CYCLE]:
        mid = p["match_id"]
        try:
            # Add small delay to avoid rate limiting (10 req/min = 1 per 6s)
            if checked > 0:
                await asyncio.sleep(7)

            match_data = get_live_match_data(mid)
            if not match_data:
                continue
            status = match_data.get("status", "")
            if status not in ("Finished", "FINISHED", "FT"):
                continue
            score = match_data.get("score", {}) or {}
            ft    = score.get("fullTime", {}) or {}
            hg    = ft.get("home")
            ag    = ft.get("away")
            # Also check top-level scores from _parse_match
            if hg is None:
                hg = match_data.get("h_score")
            if ag is None:
                ag = match_data.get("a_score")
            if hg is None or ag is None:
                continue

            if hg > ag:
                result = "H"
            elif hg == ag:
                result = "D"
            else:
                result = "A"

            log_result(mid, result, int(hg), int(ag))
            print(f"[prediction_job] result logged: "
                  f"{p['home_team']} {hg}-{ag} {p['away_team']}")
            checked += 1

        except Exception as e:
            print(f"[prediction_job] result error {mid}: {e}")

    print(f"[prediction_job] cycle done — "
          f"{len(upcoming)} upcoming, "
          f"{len(unresolved)} awaiting results, "
          f"{checked} results logged this cycle")


async def schedule_prediction_job(priors_db: dict, league_key: str,
                                   interval_minutes: int = 30):
    """
    Runs prediction job on startup then every interval_minutes.
    Call once from FastAPI lifespan or startup event.
    """
    while True:
        try:
            await run_prediction_job(priors_db, league_key)
        except Exception as e:
            print(f"[prediction_job] unhandled error: {e}")
        await asyncio.sleep(interval_minutes * 60)
