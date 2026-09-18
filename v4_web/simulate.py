"""
simulate.py  —  V4.2
=====================
Monte Carlo simulation engine for:
  1. CL-style knockout tournament (16 teams, no home advantage)
  2. PL season completion (from current standings + remaining fixtures)

Both use the Dixon-Coles bivariate Poisson priors already computed.
No external API calls — all math runs in-process.

CL knockout: ~0.3s for 10,000 runs (16 teams, 4 rounds)
PL season  : ~1.5s for 10,000 runs (20 teams, ~30 GWs remaining)
"""

import numpy as np
from scipy.stats import poisson
from collections import defaultdict

# ── Core DC match sampler ─────────────────────────────────────────────────────

def _lam_mu(home: str, away: str, priors_db: dict, league: str,
             gamma: float = 1.0) -> tuple[float, float]:
    """
    Compute Dixon-Coles λ (home expected goals) and μ (away expected goals).
    gamma=1.0 means no home advantage (used for CL neutral venue).
    """
    league_data = priors_db.get(league, {})
    teams       = league_data.get("teams", {})
    meta        = league_data.get("meta", {})
    rho         = meta.get("rho_draw_correction", 0.0)

    q25_alpha = float(np.percentile([v["alpha"] for v in teams.values()], 25)) if teams else 1.0
    q75_beta  = float(np.percentile([v["beta"]  for v in teams.values()], 75)) if teams else 1.0

    h = teams.get(home, {"alpha": q25_alpha, "beta": q75_beta})
    a = teams.get(away, {"alpha": q25_alpha, "beta": q75_beta})

    lam = float(np.clip(h["alpha"] * a["beta"] * gamma, 1e-5, 15.0))
    mu  = float(np.clip(a["alpha"] * h["beta"],         1e-5, 15.0))
    return lam, mu, rho


def _match_probs(home: str, away: str, priors_db: dict, league: str,
                 gamma: float = 1.0) -> tuple[float, float, float]:
    """Return (p_home, p_draw, p_away) from DC bivariate Poisson."""
    lam, mu, rho = _lam_mu(home, away, priors_db, league, gamma)
    hp    = poisson.pmf(np.arange(9), lam)
    ap    = poisson.pmf(np.arange(9), mu)
    joint = np.outer(hp, ap)
    joint[0,0] *= max(1.0 - lam*mu*rho, 1e-5)
    joint[1,0] *= max(1.0 + mu*rho,     1e-5)
    joint[0,1] *= max(1.0 + lam*rho,    1e-5)
    joint[1,1] *= max(1.0 - rho,        1e-5)
    joint /= joint.sum()
    ph = float(np.tril(joint, -1).sum())
    pd = float(np.trace(joint))
    pa = float(np.triu(joint, +1).sum())
    return ph, pd, pa


def _sample_score(lam: float, mu: float, rng: np.random.Generator) -> tuple[int, int]:
    """Sample a match scoreline from independent Poisson(λ), Poisson(μ)."""
    return int(rng.poisson(lam)), int(rng.poisson(mu))


def _ko_winner(team_a: str, team_b: str,
                priors_db: dict, league_map: dict,
                rng: np.random.Generator) -> str:
    """
    Simulate a single knockout match (neutral venue, no home adv).
    If draw → 50/50 penalty shootout.
    league_map: {team_name: league_key}
    """
    # Use the stronger team's league for priors (or team_a's)
    league = league_map.get(team_a, "ENG-Premier League")
    ph, pd, pa = _match_probs(team_a, team_b, priors_db, league, gamma=1.0)
    r = rng.random()
    if r < ph:
        return team_a
    elif r < ph + pd:
        # Draw — penalty shootout 50/50
        return team_a if rng.random() < 0.5 else team_b
    else:
        return team_b


# ── CL KNOCKOUT TOURNAMENT ────────────────────────────────────────────────────

def simulate_cl_tournament(
    teams: list[str],
    league_map: dict[str, str],
    priors_db: dict,
    n_runs: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Run n_runs Monte Carlo simulations of a 16-team knockout tournament.
    Draw is random each run (no fixed bracket seeding).

    Args:
        teams      : list of 16 team names
        league_map : {team_name: league_key} for DC prior lookup
        priors_db  : full priors dict
        n_runs     : number of simulations
        seed       : RNG seed for reproducibility

    Returns dict:
        win_counts    : {team: count}        — number of times won
        sf_counts     : {team: count}        — semi-final appearances
        qf_counts     : {team: count}        — quarter-final appearances
        win_pct       : {team: float}        — win probability %
        most_likely   : list of match dicts  — most common bracket path
        n_runs        : int
    """
    n = len(teams)
    if n != 16:
        raise ValueError(f"Need exactly 16 teams, got {n}")

    rng = np.random.default_rng(seed)
    win_counts = defaultdict(int)
    qf_counts  = defaultdict(int)
    sf_counts  = defaultdict(int)

    # Track a single "demo" run for the bracket display (first run)
    demo_bracket = {"r16": [], "qf": [], "sf": [], "final": [], "winner": ""}

    for run_idx in range(n_runs):
        # Random draw each simulation
        draw = rng.permutation(teams).tolist()

        # Round of 16 (8 matches)
        r16_winners = []
        r16_matches = []
        for i in range(0, 16, 2):
            a, b = draw[i], draw[i+1]
            w = _ko_winner(a, b, priors_db, league_map, rng)
            r16_winners.append(w)
            r16_matches.append({"home": a, "away": b, "winner": w})

        # Quarter finals (4 matches)
        qf_winners = []
        qf_matches = []
        for i in range(0, 8, 2):
            a, b = r16_winners[i], r16_winners[i+1]
            w = _ko_winner(a, b, priors_db, league_map, rng)
            qf_winners.append(w)
            qf_matches.append({"home": a, "away": b, "winner": w})
            qf_counts[a] += 1
            qf_counts[b] += 1

        # Semi finals (2 matches)
        sf_winners = []
        sf_matches = []
        for i in range(0, 4, 2):
            a, b = qf_winners[i], qf_winners[i+1]
            w = _ko_winner(a, b, priors_db, league_map, rng)
            sf_winners.append(w)
            sf_matches.append({"home": a, "away": b, "winner": w})
            sf_counts[a] += 1
            sf_counts[b] += 1

        # Final
        finalist_a, finalist_b = sf_winners[0], sf_winners[1]
        champion = _ko_winner(finalist_a, finalist_b, priors_db, league_map, rng)
        win_counts[champion] += 1

        if run_idx == 0:
            demo_bracket = {
                "r16"   : r16_matches,
                "qf"    : qf_matches,
                "sf"    : sf_matches,
                "final" : [{"home": finalist_a, "away": finalist_b, "winner": champion}],
                "winner": champion,
            }

    # Build ranked results
    win_pct = {
        t: round(win_counts.get(t, 0) / n_runs * 100, 1)
        for t in teams
    }
    ranked = sorted(win_pct.items(), key=lambda x: x[1], reverse=True)

    return {
        "win_pct"    : dict(ranked),
        "win_counts" : dict(win_counts),
        "qf_pct"     : {t: round(qf_counts.get(t,0)/n_runs*100,1) for t in teams},
        "sf_pct"     : {t: round(sf_counts.get(t,0)/n_runs*100,1) for t in teams},
        "most_likely": demo_bracket,
        "n_runs"     : n_runs,
    }


# ── PL SEASON SIMULATOR ───────────────────────────────────────────────────────

def simulate_pl_season(
    current_table: list[dict],
    remaining_fixtures: list[dict],
    priors_db: dict,
    league: str = "ENG-Premier League",
    n_runs: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo simulation of remaining PL season fixtures.

    current_table: list of dicts with keys:
        team, played, won, drawn, lost, gf, ga, points

    remaining_fixtures: list of dicts with keys:
        home, away

    Returns:
        table_median  : final predicted table (median points, median GD)
        title_pct     : {team: %}  — probability of finishing 1st
        top4_pct      : {team: %}  — probability of finishing top 4
        relegated_pct : {team: %}  — probability of finishing 18th-20th
        positions_pct : {team: {pos: %}}  — full position probability matrix
        n_runs        : int
    """
    teams = [row["team"] for row in current_table]
    n_teams = len(teams)

    rng = np.random.default_rng(seed)

    # Precompute λ, μ for every remaining fixture
    fixture_params = []
    meta  = priors_db.get(league, {}).get("meta", {})
    gamma = meta.get("gamma_home_advantage", 1.25)
    for fix in remaining_fixtures:
        home, away = fix["home"], fix["away"]
        lam, mu, _ = _lam_mu(home, away, priors_db, league, gamma)
        fixture_params.append((home, away, lam, mu))

    # Starting points and GD from current table
    base_pts = {row["team"]: row["points"]       for row in current_table}
    base_gd  = {row["team"]: row["gf"]-row["ga"] for row in current_table}

    # Accumulate finish positions across all runs
    pos_counts   = {t: defaultdict(int) for t in teams}
    title_counts = defaultdict(int)
    top4_counts  = defaultdict(int)
    rel_counts   = defaultdict(int)
    pts_runs     = {t: [] for t in teams}
    gd_runs      = {t: [] for t in teams}

    for _ in range(n_runs):
        pts = dict(base_pts)
        gd  = dict(base_gd)

        for home, away, lam, mu in fixture_params:
            hg = int(rng.poisson(lam))
            ag = int(rng.poisson(mu))
            gd[home] += hg - ag
            gd[away] += ag - hg
            if hg > ag:
                pts[home] += 3
            elif hg == ag:
                pts[home] += 1
                pts[away] += 1
            else:
                pts[away] += 3

        # Sort by points then GD (tiebreaker)
        ranked = sorted(teams, key=lambda t: (pts[t], gd[t]), reverse=True)
        for pos, team in enumerate(ranked, 1):
            pos_counts[team][pos] += 1
            pts_runs[team].append(pts[team])
            gd_runs[team].append(gd[team])
        title_counts[ranked[0]]    += 1
        for t in ranked[:4]:  top4_counts[t] += 1
        for t in ranked[17:]: rel_counts[t]  += 1

    # Median final table
    median_table = []
    for team in teams:
        median_table.append({
            "team"       : team,
            "pts_median" : int(np.median(pts_runs[team])),
            "pts_p10"    : int(np.percentile(pts_runs[team], 10)),
            "pts_p90"    : int(np.percentile(pts_runs[team], 90)),
            "gd_median"  : int(np.median(gd_runs[team])),
            "title_pct"  : round(title_counts[team] / n_runs * 100, 1),
            "top4_pct"   : round(top4_counts[team]  / n_runs * 100, 1),
            "rel_pct"    : round(rel_counts[team]   / n_runs * 100, 1),
            "pos_median" : int(np.median([p for p, c in pos_counts[team].items()
                                          for _ in range(c)])),
        })

    median_table.sort(key=lambda x: (x["pts_median"], x["gd_median"]), reverse=True)

    return {
        "table"    : median_table,
        "n_runs"   : n_runs,
        "fixtures" : len(remaining_fixtures),
    }
