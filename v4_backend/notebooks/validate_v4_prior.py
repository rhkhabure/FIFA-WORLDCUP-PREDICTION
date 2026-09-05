"""
validate_v4_prior.py  —  V4.2 Phase 2
========================================
Tests the Dixon-Coles prior on the held-out 2024/25 season.
This season was NEVER seen by the optimizer during training.

Grades the bivariate Poisson pre-game prediction three ways:
  1. Overall 3-way accuracy (home win / draw / away win)
  2. Per-outcome recall -- catches the draw problem directly
  3. Calibration by confidence bucket -- does higher confidence mean
     more often right?

Compares against two baselines:
  - Always predict home win (the dumbest useful baseline)
  - The V2 neural net's known group-stage accuracy of 0.556
    (the number to beat -- this is the whole point of Phase 2)

Run with: python validate_v4_prior.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

DB_PATH     = Path("v4_historical_data.sqlite")
PRIORS_PATH = Path("v4_backend/v4_priors.json")
HOLDOUT     = "2425"
MAX_GOALS   = 8   # truncation point for the bivariate Poisson grid

# ── Load ──────────────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df_holdout = pd.read_sql(
    f"SELECT * FROM matches_xg WHERE season = '{HOLDOUT}'",
    conn,
    parse_dates=["date"],
)
conn.close()

with open(PRIORS_PATH) as f:
    priors = json.load(f)

print(f"Holdout matches: {len(df_holdout):,}  (season {HOLDOUT})")
print(f"Leagues        : {sorted(df_holdout['league'].unique())}\n")


# ── Bivariate Poisson prediction ──────────────────────────────────────────────
def predict_match(home_team, away_team, league_data):
    """
    Returns (p_home, p_draw, p_away) using the fitted Dixon-Coles parameters.
    Handles teams not in the priors (promoted/newly added clubs) by using
    the league mean (alpha=1.0, beta=1.0) as a fallback, which is explicitly
    logged so we know how often this happens.
    """
    teams  = league_data["teams"]
    meta   = league_data["meta"]
    gamma  = meta["gamma_home_advantage"]
    rho    = meta["rho_draw_correction"]

    h = teams.get(home_team, {"alpha": 1.0, "beta": 1.0})
    a = teams.get(away_team, {"alpha": 1.0, "beta": 1.0})

    lam = np.clip(h["alpha"] * a["beta"] * gamma, 1e-5, 15.0)  # home expected goals
    mu  = np.clip(a["alpha"] * h["beta"],          1e-5, 15.0)  # away expected goals

    # Build the joint probability matrix P(home scores i, away scores j)
    home_probs = poisson.pmf(np.arange(MAX_GOALS + 1), lam)
    away_probs = poisson.pmf(np.arange(MAX_GOALS + 1), mu)
    joint = np.outer(home_probs, away_probs)  # shape (MAX_GOALS+1, MAX_GOALS+1)

    # Dixon-Coles low-score correction -- same rho_correction as the
    # bivariate_poisson.py module, inlined here to avoid import dependency
    joint[0, 0] *= max(1.0 - lam * mu * rho, 1e-5)
    joint[1, 0] *= max(1.0 + mu * rho,       1e-5)
    joint[0, 1] *= max(1.0 + lam * rho,      1e-5)
    joint[1, 1] *= max(1.0 - rho,            1e-5)

    joint /= joint.sum()  # re-normalise after correction

    # Sum the three outcome regions of the matrix
    p_home = float(np.tril(joint, -1).sum())  # home scored more (below diagonal)
    p_away = float(np.triu(joint, +1).sum())  # away scored more (above diagonal)
    p_draw = float(np.trace(joint))           # tied (on diagonal)

    return p_home, p_draw, p_away


# ── Grade every holdout match ─────────────────────────────────────────────────
rows = []
not_in_priors = []

for _, match in df_holdout.iterrows():
    league = match["league"]
    if league not in priors:
        continue

    p_home, p_draw, p_away = predict_match(
        match["home_team"], match["away_team"], priors[league]
    )

    # Track teams not in the prior (promoted clubs etc.)
    league_teams = priors[league]["teams"]
    if match["home_team"] not in league_teams or match["away_team"] not in league_teams:
        not_in_priors.append(
            f"{match['home_team']} vs {match['away_team']} ({league})"
        )

    # Actual outcome
    hg, ag = int(match["home_goals"]), int(match["away_goals"])
    actual = "home" if hg > ag else ("away" if ag > hg else "draw")

    # Model's top pick
    probs     = {"home": p_home, "draw": p_draw, "away": p_away}
    predicted = max(probs, key=probs.get)
    confidence = probs[predicted]

    rows.append({
        "league"    : league,
        "home"      : match["home_team"],
        "away"      : match["away_team"],
        "p_home"    : p_home,
        "p_draw"    : p_draw,
        "p_away"    : p_away,
        "actual"    : actual,
        "predicted" : predicted,
        "confidence": confidence,
        "correct"   : predicted == actual,
    })

df = pd.DataFrame(rows)
print(f"Graded {len(df):,} holdout matches.")
if not_in_priors:
    print(f"  ⚠ {len(not_in_priors)} matches had at least one team not in priors "
          f"(promoted/new -- used league mean as fallback):")
    for m in not_in_priors[:5]:
        print(f"    {m}")
    if len(not_in_priors) > 5:
        print(f"    ... and {len(not_in_priors) - 5} more")
print()


# ── Test 1: Overall and per-league 3-way accuracy ────────────────────────────
overall_acc  = df["correct"].mean()
baseline_acc = (df["actual"] == "home").mean()

print("=" * 56)
print("TEST 1 — 3-WAY ACCURACY (home / draw / away)")
print("=" * 56)
print(f"Overall accuracy       : {overall_acc:.3f}")
print(f"Always-home baseline   : {baseline_acc:.3f}")
print(f"V2 neural net (known)  : 0.556")
print(f"Beat V2?               : {'YES' if overall_acc > 0.556 else 'not yet'}")
print()

print("Per-league breakdown:")
league_summary = (df.groupby("league")["correct"]
                    .agg(["mean","count"])
                    .rename(columns={"mean":"accuracy","count":"matches"}))
league_summary["home_baseline"] = (
    df[df["actual"] == "home"].groupby("league").size() /
    df.groupby("league").size()
)
print(league_summary.round(3).to_string())
print()


# ── Test 2: Per-outcome recall ────────────────────────────────────────────────
print("=" * 56)
print("TEST 2 — PER-OUTCOME RECALL")
print("=" * 56)
print("(Rows = truth, columns = what the model predicted)")
print()
crosstab = pd.crosstab(df["actual"], df["predicted"])
print(crosstab)
print()

for outcome in ["home", "draw", "away"]:
    n_real      = (df["actual"] == outcome).sum()
    n_caught    = ((df["actual"] == outcome) & (df["predicted"] == outcome)).sum()
    n_predicted = (df["predicted"] == outcome).sum()
    precision   = n_caught / max(n_predicted, 1)
    recall      = n_caught / max(n_real, 1)
    print(f"  {outcome:<5}  real={n_real:>4}  caught={n_caught:>4}  "
          f"recall={recall:.3f}  precision={precision:.3f}")
print()
print("Key comparison -- V2 draw recall was: 0.000 (never predicted a draw)")
print("V4 draw recall target: anything above 0.000 is an improvement")


# ── Test 3: Calibration by confidence bucket ─────────────────────────────────
print()
print("=" * 56)
print("TEST 3 — CALIBRATION (does higher confidence = more often right?)")
print("=" * 56)
bins   = [0.33, 0.45, 0.55, 0.65, 0.75, 0.85, 1.01]
labels = ["33-45%", "45-55%", "55-65%", "65-75%", "75-85%", "85-100%"]
df["bucket"] = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
print(f"{'Bucket':<12}{'Predictions':>14}{'Accuracy':>12}")
for label in labels:
    sub = df[df["bucket"] == label]
    if len(sub):
        print(f"  {label:<10}{len(sub):>14}    {sub['correct'].mean():.3f}")
print()
print("Reading it: accuracy should rise as we go down this table.")
print("Flat or falling = the model is overconfident in that range.")


# ── Summary verdict ───────────────────────────────────────────────────────────
print()
print("=" * 56)
print("PHASE 2 VERDICT")
print("=" * 56)
draw_recall = ((df["actual"] == "draw") & (df["predicted"] == "draw")).sum() / \
              max((df["actual"] == "draw").sum(), 1)
print(f"Overall accuracy : {overall_acc:.3f}  (V2 was 0.556, baseline {baseline_acc:.3f})")
print(f"Draw recall      : {draw_recall:.3f}  (V2 was 0.000 -- any positive number is progress)")

if overall_acc > 0.556 and draw_recall > 0.0:
    verdict = "CLEAR IMPROVEMENT over V2 -- proceed to Phase 3 (wire into neural net)"
elif overall_acc > baseline_acc and draw_recall > 0.0:
    verdict = "BEATS BASELINE, draw recall improved -- proceed with caution"
elif overall_acc > baseline_acc:
    verdict = "BEATS BASELINE but draw recall still zero -- investigate before proceeding"
else:
    verdict = "DOES NOT BEAT BASELINE -- something is wrong, do not proceed"

print(f"Verdict          : {verdict}")