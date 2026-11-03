"""
prediction_job.py  —  V4.2
===========================
Background prediction logging job.

Called on FastAPI startup and every 30 minutes via asyncio.

What it does each cycle:
  1. Fetch upcoming PL, La Liga, Serie A, and Bundesliga fixtures
  2. For each fixture:
       a. If not yet in DB → compute DC prior → log snapshot 1
       b. If snapshot 1 exists but not snapshot 2 → try FotMob lineup
          → if lineup confirmed → compute adjusted odds → log snapshot 2
  3. Fetch recent finished matches → fill in actual results

Runs in the background — never blocks the web server.
Credits used: get_matches_by_date (3) once per day, cached.
              get_match_lineup (1) per new confirmed lineup only.
Serie A history currently records pre-match priors and final results; lineup
adjustments remain disabled until that league's lineup path is verified.
"""

import asyncio
import json
import numpy as np
import time
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
    from routes.match import get_effective_gamma
    from model_updates import get_football_match_factors
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

    gamma = get_effective_gamma(meta.get("gamma_home_advantage", 1.25))
    rho   = meta.get("rho_draw_correction",  0.0)
    dp    = 0.10   # draw propensity

    from constants import MATCH_LEAGUES
    rating_league = next(
        (cfg["ctx_key"] for cfg in MATCH_LEAGUES.values()
         if cfg["priors"] == league_key),
        league_key,
    )
    rating_lam_factor, rating_mu_factor = get_football_match_factors(
        rating_league, hk, ak
    )
    lam = float(np.clip(
        h_alpha * a_beta * gamma * rating_lam_factor, 1e-5, 15.0
    ))
    mu  = float(np.clip(
        a_alpha * h_beta * rating_mu_factor, 1e-5, 15.0
    ))

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
        rating_lam_factor,
        rating_mu_factor,
    )


async def run_prediction_job(priors_db: dict, league_key: str):
    """
    Main job cycle. Safe to call repeatedly — all writes are idempotent.
    Logs predictions for PL (FPL), La Liga (BBS), Serie A, and Bundesliga.
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

    # ── Step 1a: PL upcoming fixtures → snapshot 1 ────────────────────────
    try:
        upcoming = get_upcoming_fixtures(max_fixtures=50)
    except Exception as e:
        print(f"[prediction_job] FPL fetch error: {e}")
        upcoming = []

    for f in upcoming:
        mid     = str(f.get("match_id", ""))
        home    = f.get("home", "")
        away    = f.get("away", "")
        kickoff = f.get("kickoff_utc", f.get("kickoff_eat", ""))
        if not mid or not home or not away:
            continue
        try:
            result = _dc_probs(home, away, priors_db, "ENG-Premier League")
        except Exception as e:
            print(f"[prediction_job] DC error {home} v {away}: {e}")
            continue
        if not result:
            continue
        ph, pd, pa, lam, mu, lam_factor, mu_factor = result
        try:
            log_pre_lineup(
                match_id=mid, home_team=home, away_team=away,
                kickoff_utc=str(kickoff), season=season,
                dc_home=ph, dc_draw=pd, dc_away=pa,
                dc_lam=lam, dc_mu=mu,
                league="pl",
                rating_lam_factor=lam_factor,
                rating_mu_factor=mu_factor,
                rating_version="dc-rating-v1" if (
                    lam_factor != 1.0 or mu_factor != 1.0
                ) else "dc-static-v0",
            )
        except Exception as e:
            print(f"[prediction_job] log_pre PL error {mid}: {e}")

    # ── Step 1b: La Liga upcoming fixtures → snapshot 1 via BBS ───────────
    try:
        from bbs import get_upcoming_fixtures as bbs_fixtures, has_key as bbs_ok
        if bbs_ok():
            ll_upcoming = bbs_fixtures("laliga", max_fixtures=50)
            ll_logged = 0
            for f in ll_upcoming:
                mid     = f"ll_{f.get('bbs_id','')}"
                home    = f.get("home", "")
                away    = f.get("away", "")
                kickoff = f.get("kickoff_utc", "")
                if not mid or not home or not away:
                    continue
                try:
                    result = _dc_probs(home, away, priors_db, "ESP-La Liga")
                except Exception:
                    continue
                if not result:
                    continue
                ph, pd, pa, lam, mu, lam_factor, mu_factor = result
                try:
                    log_pre_lineup(
                        match_id=mid, home_team=home, away_team=away,
                        kickoff_utc=str(kickoff), season=season,
                        dc_home=ph, dc_draw=pd, dc_away=pa,
                        dc_lam=lam, dc_mu=mu,
                        league="laliga",
                        rating_lam_factor=lam_factor,
                        rating_mu_factor=mu_factor,
                        rating_version="dc-rating-v1" if (
                            lam_factor != 1.0 or mu_factor != 1.0
                        ) else "dc-static-v0",
                    )
                    ll_logged += 1
                except Exception as e:
                    print(f"[prediction_job] log_pre LL error {mid}: {e}")
            if ll_logged:
                print(f"[prediction_job] La Liga: {ll_logged} fixtures logged")
    except Exception as e:
        print(f"[prediction_job] BBS fixtures error: {e}")

    # ── Step 1c: Serie A and Bundesliga fixtures → snapshot 1 ─────────────
    try:
        from constants import MATCH_LEAGUES
        from footballdata import get_upcoming_fixtures_fd

        for league in ("seriea", "bundesliga"):
            league_cfg = MATCH_LEAGUES[league]
            try:
                league_fixtures = get_upcoming_fixtures_fd(
                    league_cfg["comp"], max_fixtures=50
                )
            except Exception as e:
                print(
                    f"[prediction_job] {league_cfg['name']} fixture fetch error: {e}"
                )
                continue
            league_logged = 0
            for fixture in league_fixtures:
                match_id = str(fixture.get("match_id", ""))
                home = fixture.get("home", "")
                away = fixture.get("away", "")
                kickoff = fixture.get("kickoff_utc", "")
                if not match_id or not home or not away or not kickoff:
                    continue
                try:
                    result = _dc_probs(home, away, priors_db, league_cfg["priors"])
                except Exception as e:
                    print(
                        f"[prediction_job] {league_cfg['name']} DC error "
                        f"{home} v {away}: {e}"
                    )
                    continue
                if not result:
                    continue
                ph, pd, pa, lam, mu, lam_factor, mu_factor = result
                try:
                    log_pre_lineup(
                        match_id=match_id, home_team=home, away_team=away,
                        kickoff_utc=str(kickoff), season=season,
                        dc_home=ph, dc_draw=pd, dc_away=pa,
                        dc_lam=lam, dc_mu=mu,
                        league=league,
                        rating_lam_factor=lam_factor,
                        rating_mu_factor=mu_factor,
                        rating_version="dc-rating-v1" if (
                            lam_factor != 1.0 or mu_factor != 1.0
                        ) else "dc-static-v0",
                    )
                    league_logged += 1
                except Exception as e:
                    print(
                        f"[prediction_job] log_pre {league_cfg['name']} "
                        f"error {match_id}: {e}"
                    )
            if league_logged:
                print(
                    f"[prediction_job] {league_cfg['name']}: "
                    f"{league_logged} fixtures logged"
                )
    except Exception as e:
        print(f"[prediction_job] Serie A/Bundesliga fixtures error: {e}")

    # ── Step 2: Check for confirmed lineups → snapshot 2 ───────────────────
    if fotmob_ok():
        existing = get_all_predictions(season)
        needs_lineup = [
            p for p in existing
            if p["pre_logged_at"] and not p["adj_logged_at"]
            and not p["actual_result"]  # not finished yet
            and p.get("league", "pl") != "seriea"
        ]

        # Pre-warm FotMob date cache ONCE before the loop.
        # get_fotmob_match_id() calls get_matches_by_date internally —
        # without this, it fires one API call per fixture (50+ calls = 429).
        # _refresh_date_cache_if_stale() fetches only if cache is stale,
        # so this is at most 1 credit per cycle, not 50.
        if needs_lineup:
            try:
                from fotmob import _refresh_date_cache_if_stale
                today_str = now_utc.strftime("%Y%m%d")
                _refresh_date_cache_if_stale(today_str)
            except Exception as e:
                print(f"[prediction_job] FotMob cache warm error: {e}")

        for p in needs_lineup:
            mid   = p["match_id"]
            home  = p["home_team"]
            away  = p["away_team"]

            try:
                today_str = now_utc.strftime("%Y%m%d")
                fotmob_id = get_fotmob_match_id(home, away, today_str)
                if not fotmob_id:
                    continue
                lineup = get_lineup(fotmob_id, p.get("kickoff_utc"))
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

    # ── Step 3: Log results for finished matches ───────────────────────────
    from footballdata import find_finished_match_by_teams

    existing   = get_all_predictions(season)
    # Only check matches that kicked off at least 2 hours ago
    # (90 min game + ~30 min for fd.org to update scores on free tier)
    cutoff_utc = now_utc - timedelta(hours=2)

    def kicked_off_before_cutoff(kickoff_utc: str) -> bool:
        try:
            kickoff = datetime.fromisoformat(
                kickoff_utc.replace("Z", "+00:00")
            )
            if kickoff.tzinfo is None:
                kickoff = kickoff.replace(tzinfo=timezone.utc)
            return kickoff < cutoff_utc
        except (AttributeError, TypeError, ValueError):
            return False

    unresolved = sorted(
        [
            p for p in existing
            if p["pre_logged_at"] and not p["actual_result"]
            and p["kickoff_utc"]
            and kicked_off_before_cutoff(p["kickoff_utc"])
        ],
        key=lambda x: x["kickoff_utc"]   # oldest first so backlog clears in order
    )

    checked = 0
    MAX_RESULTS_PER_CYCLE = 11  # enough to clear a full GW in one pass
    for p in unresolved[:MAX_RESULTS_PER_CYCLE]:
        mid    = p["match_id"]
        home   = p["home_team"]
        away   = p["away_team"]
        league = p.get("league", "pl")
        try:
            if checked > 0:
                await asyncio.sleep(3)

            # La Liga: use BBS match detail
            if league == "laliga" and mid.startswith("ll_"):
                try:
                    from bbs import get_match_detail, has_key as bbs_ok
                    if not bbs_ok():
                        continue
                    bbs_uuid = mid[3:]  # strip "ll_" prefix
                    detail   = get_match_detail(bbs_uuid)
                    if not detail or not detail.get("is_finished"):
                        continue
                    hg = detail.get("h_score")
                    ag = detail.get("a_score")
                    if hg is None or ag is None:
                        continue
                    result = "H" if hg > ag else ("D" if hg == ag else "A")
                    log_result(mid, result, int(hg), int(ag))
                    try:
                        from predictions import save_match_snapshot
                        from utils import get_theme_for_team
                        save_match_snapshot(
                            match_id=mid,
                            home_team=home, away_team=away,
                            kickoff_utc=p["kickoff_utc"],
                            h_score=int(hg), a_score=int(ag),
                            status="Finished", minute=90,
                            home_colour=get_theme_for_team(home).get(
                                "primary", "#14b8a6"
                            ),
                            away_colour=get_theme_for_team(away).get(
                                "primary", "#f43f5e"
                            ),
                        )
                    except Exception as snap_e:
                        print(f"[prediction_job] snapshot error: {snap_e}")
                    print(f"[prediction_job] LL result: {home} {hg}-{ag} {away} → {result}")
                    checked += 1
                    continue
                except Exception as e:
                    print(f"[prediction_job] BBS result error {mid}: {e}")
                    continue

            # football-data.org finished cache, scoped to the prediction's league.
            from constants import MATCH_LEAGUES
            comp_code = MATCH_LEAGUES.get(league, MATCH_LEAGUES["pl"])["comp"]
            match_data = find_finished_match_by_teams(
                home, away, comp_code=comp_code
            )
            if not match_data or match_data.get("status") != "Finished":
                continue

            hg = match_data.get("h_score")
            ag = match_data.get("a_score")
            if hg is None or ag is None:
                continue

            result = "H" if hg > ag else ("D" if hg == ag else "A")
            log_result(mid, result, int(hg), int(ag))
            print(
                f"[prediction_job] {league} result: "
                f"{home} {hg}-{ag} {away} -> {result}"
            )

            try:
                from predictions import save_match_snapshot
                from utils import get_theme_for_team
                home_colour = get_theme_for_team(home).get("primary", "#14b8a6")
                away_colour = get_theme_for_team(away).get("primary", "#f43f5e")
                save_match_snapshot(
                    match_id=mid,
                    home_team=home, away_team=away,
                    kickoff_utc=p["kickoff_utc"],
                    h_score=int(hg), a_score=int(ag),
                    status="Finished", minute=90,
                    home_colour=home_colour,
                    away_colour=away_colour,
                )
                print(f"[prediction_job] snapshot saved: {home} vs {away}")
            except Exception as snap_e:
                print(f"[prediction_job] snapshot error: {snap_e}")

            checked += 1

        except Exception as e:
            print(f"[prediction_job] result error {mid}: {e}")

    try:
        from model_updates import refresh_football_ratings
        from predictions import DB_PATH
        update = refresh_football_ratings(DB_PATH, priors_db)
        accepted = [
            item["league"] for item in update.get("evaluations", [])
            if item["accepted"]
        ]
        if accepted:
            print(
                "[prediction_job] validated football rating updates: "
                + ", ".join(accepted)
            )
    except Exception as e:
        print(f"[prediction_job] football rating update failed: {e}")

    print(f"[prediction_job] cycle done — "
          f"{len(upcoming)} upcoming, "
          f"{len(unresolved)} awaiting results, "
          f"{checked} results logged this cycle")


def run_prediction_job_sync(priors_db: dict, league_key: str) -> None:
    """Run one async prediction cycle in a worker thread with its own loop."""
    asyncio.run(run_prediction_job(priors_db, league_key))


async def schedule_prediction_job(priors_db: dict, league_key: str,
                                   interval_minutes: int = 30):
    """
    Runs prediction job on startup then every interval_minutes.
    Call once from FastAPI lifespan or startup event.
    """
    while True:
        started = time.perf_counter()
        try:
            await asyncio.to_thread(
                run_prediction_job_sync, priors_db, league_key
            )
        except Exception as e:
            print(f"[prediction_job] unhandled error: {e}")
        print(
            "[prediction_job] background cycle finished in "
            f"{time.perf_counter() - started:.2f}s"
        )
        await asyncio.sleep(interval_minutes * 60)
