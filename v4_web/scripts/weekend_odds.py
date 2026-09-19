"""
weekend_odds.py  —  V4.2
==========================
Prints pre-game odds for upcoming Premier League fixtures
using the Dixon-Coles model.

Run from project root:
  python weekend_odds.py

Output: clean table of fixtures with model probabilities
and implied decimal odds for each outcome.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
from scipy.stats import poisson

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from v4_backend.feature_builder import TEAM_NAME_ALIASES
from fpl import get_upcoming_fixtures   # uses FPL API, no key needed

PRIORS_PATH = ROOT / "v4_backend" / "v4_priors.json"
LEAGUE      = "ENG-Premier League"
DRAW_PROP   = 0.10
EAT         = timezone(timedelta(hours=3))


def load_priors():
    with open(PRIORS_PATH) as f:
        return json.load(f)


def dc_odds(home: str, away: str, priors: dict) -> dict | None:
    league_data = priors.get(LEAGUE)
    if not league_data:
        return None
    teams  = league_data["teams"]
    meta   = league_data["meta"]
    gamma  = meta["gamma_home_advantage"]
    rho    = meta["rho_draw_correction"]

    hk = TEAM_NAME_ALIASES.get(home, home)
    ak = TEAM_NAME_ALIASES.get(away, away)

    all_alpha = [v["alpha"] for v in teams.values()]
    all_beta  = [v["beta"]  for v in teams.values()]
    q25a = float(np.percentile(all_alpha, 25))
    q75b = float(np.percentile(all_beta,  75))

    h = teams.get(hk, {"alpha": q25a, "beta": q75b})
    a = teams.get(ak, {"alpha": q25a, "beta": q75b})

    lam = float(np.clip(h["alpha"] * a["beta"] * gamma, 1e-5, 15.0))
    mu  = float(np.clip(a["alpha"] * h["beta"],          1e-5, 15.0))

    hp = poisson.pmf(np.arange(9), lam)
    ap = poisson.pmf(np.arange(9), mu)
    j  = np.outer(hp, ap)
    j[0,0] *= max(1.0 - lam*mu*rho, 1e-5)
    j[1,0] *= max(1.0 + mu*rho,     1e-5)
    j[0,1] *= max(1.0 + lam*rho,    1e-5)
    j[1,1] *= max(1.0 - rho,        1e-5)
    j /= j.sum()

    ph = float(np.tril(j,-1).sum())
    pd = float(np.trace(j))
    pa = float(np.triu(j,+1).sum())

    tr = DRAW_PROP / 2.0
    ph2 = max(ph - tr, 0.0)
    pa2 = max(pa - tr, 0.0)
    pd2 = pd + DRAW_PROP
    tot = ph2 + pd2 + pa2

    ph2 /= tot; pd2 /= tot; pa2 /= tot

    # Top 3 scorelines
    scores = sorted(
        [(float(j[i,k]), i, k) for i in range(min(6,9)) for k in range(min(6,9))],
        reverse=True
    )[:3]

    return {
        "ph": ph2, "pd": pd2, "pa": pa2,
        "lam": lam, "mu": mu,
        "top_scores": scores,
        "home_fallback": hk not in teams,
        "away_fallback": ak not in teams,
    }


def main():
    priors   = load_priors()
    fixtures = get_upcoming_fixtures(max_fixtures=20)

    if not fixtures:
        print("No upcoming fixtures found from FPL API.")
        return

    print()
    print("=" * 72)
    print("  PREMIER LEAGUE — MODEL ODDS  (Dixon-Coles, dp=0.10)")
    print(f"  Generated: {datetime.now(EAT).strftime('%a %d %b %Y %H:%M EAT')}")
    print("=" * 72)
    print()

    for f in fixtures:
        home = f["home"]
        away = f["away"]
        ko   = f["kickoff_eat"]
        gw   = f["gameweek"]

        odds = dc_odds(home, away, priors)
        if not odds:
            continue

        ph, pd, pa = odds["ph"], odds["pd"], odds["pa"]

        # Implied decimal odds (1/prob, rounded to 2dp)
        dh = round(1/ph, 2) if ph > 0 else "—"
        dd = round(1/pd, 2) if pd > 0 else "—"
        da = round(1/pa, 2) if pa > 0 else "—"

        fallback_note = ""
        if odds["home_fallback"] or odds["away_fallback"]:
            fallback_note = " ⚠ (fallback strength)"

        print(f"  GW{gw}  {ko}")
        print(f"  {home:<22} vs  {away}")
        print(f"  xG:  {odds['lam']:.2f} – {odds['mu']:.2f}")
        print(f"  {'Home':>6}   {'Draw':>6}   {'Away':>6}")
        print(f"  {ph:>5.1%}   {pd:>5.1%}   {pa:>5.1%}   (probability)")
        print(f"  {dh:>6}   {dd:>6}   {da:>6}   (decimal odds){fallback_note}")

        top = odds["top_scores"]
        top_str = "  ".join(f"{i}-{k} ({p:.1%})" for p, i, k in top)
        print(f"  Top scorelines: {top_str}")
        print()

    print("=" * 72)
    print("  ⚠  These are model probabilities, not betting advice.")
    print("  ⚠  Fallback strength (⚠) = team not in training priors.")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
