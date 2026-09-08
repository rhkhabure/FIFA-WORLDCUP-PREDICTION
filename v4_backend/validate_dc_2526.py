"""
validate_dc_2526.py
====================
Validates the Dixon-Coles prior ALONE on the 2526 holdout.
No neural net involved -- pure bivariate Poisson pre-game predictions.

This answers the key question: does Dixon-Coles alone beat the neural
net on the 2526 holdout the same way it did on 2425?

If yes -> the neural net's value is specifically in LIVE in-game
          updating, not pre-game prediction. Ship accordingly.
If no  -> the normalization issue is the real problem, fix and retrain.

Run: python validate_dc_2526.py
"""

import json, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT        = Path(__file__).parent
DB_PATH     = ROOT.parent / "v4_historical_data.sqlite"
PRIORS_PATH = ROOT / "v4_priors.json"
HOLDOUT     = "2526"
DRAW_PROP   = 0.10   # same value confirmed in Phase 2

with open(PRIORS_PATH) as f:
    priors = json.load(f)

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(
    f"SELECT * FROM matches_xg WHERE season='{HOLDOUT}'",
    conn, parse_dates=["date"],
)
conn.close()

print(f"Holdout: {HOLDOUT}  ({len(df):,} matches)")
print(f"draw_propensity: {DRAW_PROP}")
print()


def predict(home, away, league_data):
    teams = league_data["teams"]
    meta  = league_data["meta"]
    gamma = meta["gamma_home_advantage"]
    rho   = meta["rho_draw_correction"]
    all_a = [v["alpha"] for v in teams.values()]
    all_b = [v["beta"]  for v in teams.values()]
    q25a  = float(np.percentile(all_a, 25))
    q75b  = float(np.percentile(all_b, 75))
    h = teams.get(home, {"alpha": q25a, "beta": q75b})
    a = teams.get(away, {"alpha": q25a, "beta": q75b})
    lam = np.clip(h["alpha"] * a["beta"] * gamma, 1e-5, 15.0)
    mu  = np.clip(a["alpha"] * h["beta"],          1e-5, 15.0)
    hp  = poisson.pmf(np.arange(9), lam)
    ap  = poisson.pmf(np.arange(9), mu)
    j   = np.outer(hp, ap)
    j[0,0] *= max(1.0 - lam*mu*rho, 1e-5)
    j[1,0] *= max(1.0 + mu*rho,     1e-5)
    j[0,1] *= max(1.0 + lam*rho,    1e-5)
    j[1,1] *= max(1.0 - rho,        1e-5)
    j /= j.sum()
    ph = float(np.tril(j,-1).sum())
    pd_ = float(np.trace(j))
    pa = float(np.triu(j,+1).sum())
    tr = DRAW_PROP / 2
    ph2 = max(ph - tr, 0); pa2 = max(pa - tr, 0); pd2 = pd_ + DRAW_PROP
    tot = ph2 + pd2 + pa2
    return ph2/tot, pd2/tot, pa2/tot


rows = []
fallbacks = 0
for _, r in df.iterrows():
    league = r["league"]
    if league not in priors:
        continue
    if (r["home_team"] not in priors[league]["teams"] or
            r["away_team"] not in priors[league]["teams"]):
        fallbacks += 1
    ph, pd_, pa = predict(r["home_team"], r["away_team"], priors[league])
    hg, ag = int(r["home_goals"]), int(r["away_goals"])
    actual = "home" if hg > ag else ("away" if ag > hg else "draw")
    probs  = {"home": ph, "draw": pd_, "away": pa}
    pred   = max(probs, key=probs.get)
    rows.append({"actual": actual, "predicted": pred,
                 "confidence": probs[pred], "league": league})

df_r = pd.DataFrame(rows)
print(f"Graded {len(df_r):,} matches ({fallbacks} fallbacks)\n")

overall   = (df_r["actual"] == df_r["predicted"]).mean()
baseline  = (df_r["actual"] == "home").mean()

print("=" * 56)
print("RESULTS — Dixon-Coles only, 2526 holdout")
print("=" * 56)
print(f"Overall accuracy  : {overall:.3f}")
print(f"Always-home base  : {baseline:.3f}")
print(f"V4 neural net     : 0.493  (same holdout)")
print(f"Beat neural net?  : {'YES' if overall > 0.493 else 'no'}")
print()

print("Per-outcome recall:")
ct = pd.crosstab(df_r["actual"], df_r["predicted"])
print(ct)
print()
for outcome in ["home", "draw", "away"]:
    n_real   = (df_r["actual"] == outcome).sum()
    n_caught = ((df_r["actual"] == outcome) &
                (df_r["predicted"] == outcome)).sum()
    n_pred   = (df_r["predicted"] == outcome).sum()
    rec  = n_caught / max(n_real, 1)
    prec = n_caught / max(n_pred, 1)
    print(f"  {outcome:<5} real={n_real:>4} caught={n_caught:>4} "
          f"recall={rec:.3f} precision={prec:.3f}")

draw_recall = ((df_r["actual"]=="draw") &
               (df_r["predicted"]=="draw")).sum() / \
               max((df_r["actual"]=="draw").sum(), 1)
print()

print("Calibration:")
bins   = [0.33, 0.45, 0.55, 0.65, 0.75, 0.85, 1.01]
labels = ["33-45%","45-55%","55-65%","65-75%","75-85%","85-100%"]
df_r["bucket"] = pd.cut(df_r["confidence"], bins=bins, labels=labels, right=False)
for label in labels:
    sub = df_r[df_r["bucket"]==label]
    if len(sub):
        print(f"  {label:<10} {len(sub):>6}   {sub['correct' if 'correct' in sub else (sub['actual']==sub['predicted']).rename('correct')].mean():.3f}"
              if False else
              f"  {label:<10} {len(sub):>6}   {(sub['actual']==sub['predicted']).mean():.3f}")
print()

print("=" * 56)
print("KEY QUESTION")
print("=" * 56)
if overall > 0.493:
    print(f"Dixon-Coles ({overall:.3f}) BEATS V4 neural net (0.493)")
    print("Conclusion: neural net adds no value pre-game on 2526.")
    print("The neural net's real value is LIVE (score+time features")
    print("dominate pre-game strength features mid-match).")
    print("Recommendation: use DC for pre-game, neural net for live.")
else:
    print(f"Dixon-Coles ({overall:.3f}) does NOT beat V4 neural net (0.493)")
    print("Investigate further before deciding architecture.")

print(f"\nDraw recall DC: {draw_recall:.3f}  |  Neural net: 0.070")
