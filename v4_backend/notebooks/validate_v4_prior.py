"""
validate_v4_prior_fixed.py  —  V4.2 Phase 2 (with both fixes applied)
=======================================================================
Tests the Dixon-Coles prior on the held-out 2024/25 season with two
fixes applied over the original validation:

  FIX 1 — draw propensity parameter (draw_propensity = 0.15)
           Redistributes 15% of probability mass from home+away equally
           to draw, selectively making close matches predict "draw"
           while leaving clear home/away favourites unchanged.
           Value chosen by grid search on training data only.

  FIX 2 — bottom-quartile fallback for unknown teams
           Newly promoted clubs not in the priors now use the
           bottom-quartile alpha/beta for their league, not the mean.
           Rationale: promoted clubs are systematically weaker than
           average -- the mean is a demonstrably wrong prior for them.

Run: python validate_v4_prior_fixed.py
"""

import json, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

DB_PATH          = Path("v4_historical_data.sqlite")
PRIORS_PATH      = Path("v4_backend/v4_priors.json")
HOLDOUT          = "2425"
DRAW_PROPENSITY  = 0.15   # tuned by grid search on training data
MAX_GOALS        = 8

# ── Load ──────────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df_holdout = pd.read_sql(
    f"SELECT * FROM matches_xg WHERE season = '{HOLDOUT}'",
    conn, parse_dates=["date"],
)
conn.close()

with open(PRIORS_PATH) as f:
    priors = json.load(f)

print(f"Holdout matches : {len(df_holdout):,}  (season {HOLDOUT})")
print(f"draw_propensity : {DRAW_PROPENSITY}  (grid-searched on training data)")
print(f"Unknown-team    : bottom-quartile alpha/beta fallback (not league mean)")
print()


# ── Prediction function (both fixes applied) ──────────────────────────────────
def predict_match(home_team, away_team, league_data):
    teams = league_data["teams"]
    meta  = league_data["meta"]
    gamma = meta["gamma_home_advantage"]
    rho   = meta["rho_draw_correction"]

    # FIX 2: bottom-quartile fallback instead of league mean
    all_alpha = [v["alpha"] for v in teams.values()]
    all_beta  = [v["beta"]  for v in teams.values()]
    q25_alpha = float(np.percentile(all_alpha, 25))
    q75_beta  = float(np.percentile(all_beta,  75))

    h = teams.get(home_team, {"alpha": q25_alpha, "beta": q75_beta})
    a = teams.get(away_team, {"alpha": q25_alpha, "beta": q75_beta})

    lam = np.clip(h["alpha"] * a["beta"] * gamma, 1e-5, 15.0)
    mu  = np.clip(a["alpha"] * h["beta"],          1e-5, 15.0)

    hp = poisson.pmf(np.arange(MAX_GOALS + 1), lam)
    ap = poisson.pmf(np.arange(MAX_GOALS + 1), mu)
    joint = np.outer(hp, ap)
    joint[0, 0] *= max(1.0 - lam * mu * rho, 1e-5)
    joint[1, 0] *= max(1.0 + mu * rho,       1e-5)
    joint[0, 1] *= max(1.0 + lam * rho,      1e-5)
    joint[1, 1] *= max(1.0 - rho,            1e-5)
    joint /= joint.sum()

    p_home_raw = float(np.tril(joint, -1).sum())
    p_draw_raw = float(np.trace(joint))
    p_away_raw = float(np.triu(joint, +1).sum())

    # FIX 1: draw propensity -- redistribute from home+away to draw
    transfer = DRAW_PROPENSITY / 2.0
    p_home = max(p_home_raw - transfer, 0.0)
    p_away = max(p_away_raw - transfer, 0.0)
    p_draw = p_draw_raw + DRAW_PROPENSITY
    total  = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


# ── Grade every holdout match ─────────────────────────────────────────────────
rows = []
fallback_count = 0

for _, match in df_holdout.iterrows():
    league = match["league"]
    if league not in priors:
        continue

    league_data = priors[league]
    if (match["home_team"] not in league_data["teams"] or
            match["away_team"] not in league_data["teams"]):
        fallback_count += 1

    p_home, p_draw, p_away = predict_match(
        match["home_team"], match["away_team"], league_data
    )

    hg, ag  = int(match["home_goals"]), int(match["away_goals"])
    actual  = "home" if hg > ag else ("away" if ag > hg else "draw")
    probs   = {"home": p_home, "draw": p_draw, "away": p_away}
    predicted   = max(probs, key=probs.get)
    confidence  = probs[predicted]

    rows.append({
        "league":     league,
        "home":       match["home_team"],
        "away":       match["away_team"],
        "p_home":     p_home,
        "p_draw":     p_draw,
        "p_away":     p_away,
        "actual":     actual,
        "predicted":  predicted,
        "confidence": confidence,
        "correct":    predicted == actual,
    })

df = pd.DataFrame(rows)
print(f"Graded {len(df):,} holdout matches  "
      f"({fallback_count} used bottom-quartile fallback for unknown teams)\n")


# ── Test 1: Overall 3-way accuracy ────────────────────────────────────────────
overall_acc  = df["correct"].mean()
baseline_acc = (df["actual"] == "home").mean()

print("=" * 60)
print("TEST 1 — 3-WAY ACCURACY (home / draw / away)")
print("=" * 60)
print(f"Overall accuracy         : {overall_acc:.3f}")
print(f"Always-home baseline     : {baseline_acc:.3f}")
print(f"V2 neural net (known)    : 0.556")
print(f"Original V4 (no fixes)   : 0.506")
print(f"Beat V2?                 : {'YES' if overall_acc > 0.556 else 'not yet'}")
print(f"Improved over no-fix V4? : {'YES' if overall_acc > 0.506 else 'no'}")
print()

print("Per-league breakdown:")
lg = df.groupby("league")["correct"].agg(["mean","count"]).rename(
    columns={"mean": "accuracy", "count": "matches"})
home_base = (df[df["actual"]=="home"].groupby("league").size() /
             df.groupby("league").size()).rename("home_baseline")
print(pd.concat([lg, home_base], axis=1).round(3).to_string())
print()


# ── Test 2: Per-outcome recall ────────────────────────────────────────────────
print("=" * 60)
print("TEST 2 — PER-OUTCOME RECALL")
print("=" * 60)
print(pd.crosstab(df["actual"], df["predicted"]))
print()

for outcome in ["home", "draw", "away"]:
    n_real  = (df["actual"] == outcome).sum()
    caught  = ((df["actual"] == outcome) & (df["predicted"] == outcome)).sum()
    n_pred  = (df["predicted"] == outcome).sum()
    prec    = caught / max(n_pred, 1)
    rec     = caught / max(n_real, 1)
    print(f"  {outcome:<5}  real={n_real:>4}  caught={caught:>4}  "
          f"recall={rec:.3f}  precision={prec:.3f}")

print()
draw_recall = ((df["actual"]=="draw") & (df["predicted"]=="draw")).sum() / \
              max((df["actual"]=="draw").sum(), 1)
print(f"Key: V2 draw recall was 0.000 -- anything above is progress.")
print(f"     Original V4 draw recall was 0.000")
print(f"     Fixed V4 draw recall: {draw_recall:.3f}")


# ── Test 3: Calibration ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 3 — CALIBRATION")
print("=" * 60)
bins   = [0.33, 0.45, 0.55, 0.65, 0.75, 0.85, 1.01]
labels = ["33-45%", "45-55%", "55-65%", "65-75%", "75-85%", "85-100%"]
df["bucket"] = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
print(f"{'Bucket':<12}{'Predictions':>14}{'Accuracy':>12}")
for label in labels:
    sub = df[df["bucket"] == label]
    if len(sub):
        print(f"  {label:<10}{len(sub):>14}    {sub['correct'].mean():.3f}")
print()
print("Original calibration staircase was already strong (0.379→0.843).")
print("Expect some shift now that draw predictions are redistributing confidence.")


# ── Verdict ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PHASE 2 VERDICT — BOTH FIXES APPLIED")
print("=" * 60)
print(f"Overall accuracy : {overall_acc:.3f}  "
      f"(was 0.506 unfixed, V2 was 0.556, baseline {baseline_acc:.3f})")
print(f"Draw recall      : {draw_recall:.3f}  "
      f"(was 0.000 unfixed, V2 was 0.000)")

if overall_acc > 0.556 and draw_recall > 0.10:
    verdict = "CLEAR IMPROVEMENT over V2 on both metrics -- proceed to Phase 3"
elif overall_acc > 0.506 and draw_recall > 0.0:
    verdict = "IMPROVED over unfixed V4, draw recall now positive -- proceed to Phase 3"
elif overall_acc > baseline_acc and draw_recall > 0.0:
    verdict = "Beats baseline, draw recall improved -- acceptable for Phase 3"
elif draw_recall > 0.0:
    verdict = "Draw recall fixed but accuracy regressed -- investigate draw_propensity value"
else:
    verdict = "Draw recall still zero -- something went wrong with Fix 1"

print(f"Verdict          : {verdict}")
