"""
validate_v4_neural.py  —  V4.2 Phase 3 Step 3
===============================================
Tests football_v4.pth on the held-out 2024/25 season.
This season was NEVER seen during training -- not by the Dixon-Coles
optimizer (train_v4_prior.py) and not by the neural net (train_v4_neural.ipynb).

The same four-test structure used throughout this project:
  1. Overall 3-way accuracy (home / draw / away)
  2. Per-outcome recall -- draw handling is the key metric to watch
  3. Calibration by confidence bucket
  4. Side-by-side comparison against every prior model version

Feature construction uses DCStrengthLookup exactly as the training data
was built -- same league, same normalization, same fallback logic.
Consistency between training and inference is critical; any mismatch
here would silently corrupt the predictions.

Run: python validate_v4_neural.py
"""

import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, log_loss,
                              precision_score, recall_score)

# ── Paths  ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent  # this is v4_backend/
DB_PATH = ROOT.parent / "v4_historical_data.sqlite"
MODELS_DIR  = ROOT / "notebooks" / "v4_backend" / "models"
PRIORS_PATH = ROOT / "v4_priors.json"
RESULTS_DIR = ROOT / "notebooks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_SEASON = "2425"

# ── Imports from the project ──────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from feature_builder import DCStrengthLookup   # noqa: E402

FEATURE_COLS = [
    "goal_diff", "minute_norm", "is_second_half",
    "home_rank_norm", "away_rank_norm", "rank_diff",
    "is_knockout", "lead_changes_norm",
    "is_neutral_venue", "score_state", "strength_x_time",
]
N_FEATURES, N_CLASSES = len(FEATURE_COLS), 3


# ── Model definition (must match training exactly) ────────────────────────────
class FootballWinProbNet(nn.Module):
    def __init__(self, n_features=11, n_classes=3, h1=40, h2=20, dropout=0.30):
        super().__init__()
        self.fc1  = nn.Linear(n_features, h1)
        self.fc2  = nn.Linear(h1, h2)
        self.head = nn.Linear(h2, n_classes)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.ReLU()

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return self.head(x)


# ── Load model ────────────────────────────────────────────────────────────────
ckpt   = torch.load(MODELS_DIR / "football_v4.pth",
                    map_location="cpu", weights_only=False)
model  = FootballWinProbNet(**ckpt["arch"])
model.load_state_dict(ckpt["model_state"])
model.eval()
T_best = ckpt["temperature"]

with open(MODELS_DIR / "scaler_v4.pkl", "rb") as f:
    scaler = pickle.load(f)

print(f"Loaded football_v4.pth")
print(f"Architecture : {ckpt['arch']}")
print(f"Temperature  : {T_best:.3f}")
print(f"Strength src : {ckpt.get('strength_source', 'unknown')}")
print(f"Warm-started : {ckpt.get('warm_started', False)}")
print()

# ── Load holdout matches ──────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df_holdout = pd.read_sql(
    f"SELECT * FROM matches_xg WHERE season = '{HOLDOUT_SEASON}'",
    conn, parse_dates=["date"],
)
conn.close()

print(f"Holdout season {HOLDOUT_SEASON}: {len(df_holdout):,} matches")
print(f"Leagues: {sorted(df_holdout['league'].unique())}\n")

# ── Load DC strength lookup (same as used in training) ────────────────────────
dc = DCStrengthLookup(PRIORS_PATH)


# ── Build the PRE-GAME feature row for each holdout match ─────────────────────
# We evaluate at minute=0 with score 0-0 -- the pure pre-game prediction.
# This is the same "minute 0" point the History page line chart always starts
# from, and the same point used in the Dixon-Coles validation earlier.
# It's the fairest test: the model has only team strength to work with,
# no score information to make the prediction trivially easy.
def build_pregame_row(home_team, away_team, league):
    return dc.build_feature_row(
        home_team=home_team, away_team=away_team, league=league,
        minute=0, home_score=0, away_score=0,
        lead_changes=0, goals_so_far=0,
        is_knockout=0, is_neutral_venue=0,
    )


rows = []
fallbacks = 0
for _, match in df_holdout.iterrows():
    if dc.was_fallback(match["home_team"], match["league"]) or \
       dc.was_fallback(match["away_team"], match["league"]):
        fallbacks += 1

    feat = build_pregame_row(
        match["home_team"], match["away_team"], match["league"]
    )
    hg, ag = int(match["home_goals"]), int(match["away_goals"])
    actual = "home" if hg > ag else ("away" if ag > hg else "draw")
    rows.append({"league": match["league"], "actual": actual,
                 "features": feat})

print(f"Built pre-game features for {len(rows):,} matches "
      f"({fallbacks} used fallback strength)\n")

# ── Run inference ─────────────────────────────────────────────────────────────
X = np.array([r["features"] for r in rows], dtype="float32")
X_scaled = scaler.transform(X).astype("float32")

with torch.no_grad():
    logits = model(torch.tensor(X_scaled)).numpy()

def probs_at_T(logits, T):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)

probs = probs_at_T(logits, T_best)   # shape (N, 3): [p_away, p_draw, p_home]
names = ["away", "draw", "home"]

actuals    = np.array([r["actual"] for r in rows])
leagues    = np.array([r["league"] for r in rows])
predicted  = np.array([names[i] for i in probs.argmax(axis=1)])
confidence = probs.max(axis=1)
correct    = (actuals == predicted)

# ── TEST 1: Overall and per-league accuracy ───────────────────────────────────
overall_acc  = correct.mean()
baseline_acc = (actuals == "home").mean()

print("=" * 62)
print("TEST 1 — 3-WAY ACCURACY (home / draw / away)")
print("=" * 62)
print(f"Overall accuracy            : {overall_acc:.3f}")
print(f"Always-home baseline        : {baseline_acc:.3f}")
print(f"Dixon-Coles prior alone     : 0.498  (Phase 2 holdout)")
print(f"V2 neural net (World Cup)   : 0.556  (different distribution)")
print(f"V4 internal test set        : 0.696  (same training seasons)")
print(f"Beat Dixon-Coles alone?     : {'YES' if overall_acc > 0.498 else 'no'}")
print(f"Beat V2 World Cup?          : {'YES' if overall_acc > 0.556 else 'no'}")
print()

print("Per-league breakdown:")
for league in sorted(set(leagues)):
    mask = leagues == league
    lacc = correct[mask].mean()
    lb   = (actuals[mask] == "home").mean()
    print(f"  {league:<22}  acc {lacc:.3f}  baseline {lb:.3f}"
          f"  {'✓' if lacc > lb else '✗'}")
print()


# ── TEST 2: Per-outcome recall ────────────────────────────────────────────────
print("=" * 62)
print("TEST 2 — PER-OUTCOME RECALL")
print("=" * 62)
ct = pd.crosstab(pd.Series(actuals, name="actual"),
                 pd.Series(predicted, name="predicted"))
print(ct)
print()

draw_recall = precision_score(actuals, predicted,
                               labels=["draw"], average="macro",
                               zero_division=0)
# Use the correct formulation: recall = caught / real
for outcome in ["home", "draw", "away"]:
    n_real   = (actuals == outcome).sum()
    n_caught = ((actuals == outcome) & (predicted == outcome)).sum()
    n_pred   = (predicted == outcome).sum()
    rec  = n_caught / max(n_real, 1)
    prec = n_caught / max(n_pred, 1)
    print(f"  {outcome:<5}  real={n_real:>4}  caught={n_caught:>4}  "
          f"recall={rec:.3f}  precision={prec:.3f}")

draw_recall = ((actuals == "draw") & (predicted == "draw")).sum() / \
               max((actuals == "draw").sum(), 1)
print()
print("Draw recall history across all versions:")
print("  V2 neural net (World Cup)        : 0.000")
print("  Dixon-Coles prior, unfixed        : 0.000")
print("  Dixon-Coles prior, fixed (dp=0.10): 0.244")
print(f"  V4 neural net (this run)          : {draw_recall:.3f}")


# ── TEST 3: Calibration ───────────────────────────────────────────────────────
print()
print("=" * 62)
print("TEST 3 — CALIBRATION")
print("=" * 62)
bins   = [0.33, 0.45, 0.55, 0.65, 0.75, 0.85, 1.01]
labels = ["33-45%", "45-55%", "55-65%", "65-75%", "75-85%", "85-100%"]
bucket = pd.cut(confidence, bins=bins, labels=labels, right=False)
print(f"{'Bucket':<12}{'Predictions':>14}{'Accuracy':>12}")
for label in labels:
    mask = bucket == label
    if mask.sum():
        print(f"  {label:<10}{mask.sum():>14}    {correct[mask].mean():.3f}")
print()
print("Accuracy should rise monotonically. Flat/falling = overconfident.")


# ── TEST 4: Full comparison table ─────────────────────────────────────────────
ll = log_loss(
    pd.get_dummies(actuals, columns=["away", "draw", "home"]).values
    if False else actuals,
    probs,
    labels=["away", "draw", "home"],
)

print()
print("=" * 62)
print("TEST 4 — FULL VERSION COMPARISON (on this same holdout)")
print("=" * 62)
print(f"{'Version':<38}{'Accuracy':>10}{'Draw recall':>13}")
print("-" * 62)
print(f"  {'Always-home baseline':<36}{baseline_acc:>10.3f}{'n/a':>13}")
print(f"  {'Dixon-Coles prior, unfixed':<36}{'0.506':>10}{'0.000':>13}")
print(f"  {'Dixon-Coles prior, fixed (dp=0.10)':<36}{'0.498':>10}{'0.244':>13}")
print(f"  {'V4 neural net (this run)':<36}{overall_acc:>10.3f}{draw_recall:>13.3f}")
print()
print(f"Log-loss (lower = better calibration): {ll:.4f}")


# ── Verdict and permanent log ─────────────────────────────────────────────────
print()
print("=" * 62)
print("PHASE 3 VERDICT")
print("=" * 62)

if overall_acc > 0.556 and draw_recall > 0.10:
    verdict = "STRONG IMPROVEMENT -- beats V2 World Cup AND draws improved"
elif overall_acc > 0.498 and draw_recall > 0.10:
    verdict = "IMPROVED over Dixon-Coles alone, draws improved"
elif overall_acc > 0.498:
    verdict = "Beats Dixon-Coles alone but draw recall still weak"
elif draw_recall > 0.10:
    verdict = "Draw recall improved but accuracy did not beat prior"
else:
    verdict = "Neither metric improved -- investigate before proceeding"

print(f"Accuracy : {overall_acc:.3f}")
print(f"Draw rec : {draw_recall:.3f}")
print(f"Log-loss : {ll:.4f}")
print(f"Verdict  : {verdict}")

stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
entry = f"""
## V4.2 Phase 3 Step 3 — holdout validation ({stamp})
- Model: football_v4.pth (Dixon-Coles strength features)
- Holdout season: {HOLDOUT_SEASON} ({len(df_holdout):,} matches)
- Evaluation: pre-game only (minute=0, score 0-0)
- Overall accuracy: {overall_acc:.3f}  (baseline {baseline_acc:.3f})
- Log-loss: {ll:.4f}
- Draw recall: {draw_recall:.3f}
- Verdict: {verdict}
"""
rp = RESULTS_DIR / "RESULTS.md"
if not rp.exists():
    rp.write_text("# V4.2 Results Log\n")
with rp.open("a") as f:
    f.write(entry)

row = pd.DataFrame([{
    "timestamp": stamp, "phase": "phase3_holdout_v4",
    "test_acc": round(overall_acc, 4),
    "baseline_acc": round(baseline_acc, 4),
    "test_logloss": round(ll, 4),
    "draw_recall": round(draw_recall, 4),
}])
mc = RESULTS_DIR / "metrics.csv"
row.to_csv(mc, mode="a", header=not mc.exists(), index=False)
print(f"\nLogged to results/RESULTS.md and results/metrics.csv")
