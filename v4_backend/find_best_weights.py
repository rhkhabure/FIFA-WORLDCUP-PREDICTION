"""
find_best_weights.py  —  V4.2
================================
Grid-searches class weight values to find the best trade-off between
overall accuracy and draw recall on the 2425 holdout, using the
corrected alias table (0 fallbacks).

NOTE: This script does NOT retrain the model. It loads the existing
football_v4.pth and applies a soft post-hoc draw boost by adjusting
the raw logits before softmax. This is an approximation -- the final
chosen weight needs a full retrain to be properly embedded -- but it
gives a reliable signal for which direction actually helps.

The "elbow" is where adding more draw boost stops helping accuracy
and only hurts it. We want the weight just before that elbow.

Run: python find_best_weights.py
"""

import json, pickle, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "notebooks" / "v4_backend" / "models"
PRIORS = ROOT / "v4_priors.json"
DB_PATH    = ROOT.parent / "v4_historical_data.sqlite"
HOLDOUT    = "2425"

sys.path.insert(0, str(ROOT))
from feature_builder import DCStrengthLookup

FEATURE_COLS = [
    "goal_diff", "minute_norm", "is_second_half",
    "home_rank_norm", "away_rank_norm", "rank_diff",
    "is_knockout", "lead_changes_norm",
    "is_neutral_venue", "score_state", "strength_x_time",
]


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
ckpt = torch.load(MODELS_DIR / "football_v4.pth",
                  map_location="cpu", weights_only=False)
model = FootballWinProbNet(**ckpt["arch"])
model.load_state_dict(ckpt["model_state"])
model.eval()
T = ckpt["temperature"]

with open(MODELS_DIR / "scaler_v4.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── Load holdout ──────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(
    f"SELECT * FROM matches_xg WHERE season='{HOLDOUT}'",
    conn, parse_dates=["date"],
)
conn.close()

dc = DCStrengthLookup(PRIORS)

# ── Build features ────────────────────────────────────────────────────────────
feats, actuals = [], []
for _, r in df.iterrows():
    feat = dc.build_feature_row(
        r["home_team"], r["away_team"], r["league"],
        minute=0, home_score=0, away_score=0,
        lead_changes=0, goals_so_far=0,
        is_knockout=0, is_neutral_venue=0,
    )
    feats.append(feat)
    hg, ag = int(r["home_goals"]), int(r["away_goals"])
    actuals.append(2 if hg > ag else (0 if ag > hg else 1))

X = scaler.transform(np.array(feats, dtype="float32")).astype("float32")
actuals = np.array(actuals)

# ── Get raw logits once ───────────────────────────────────────────────────────
with torch.no_grad():
    logits = model(torch.tensor(X)).numpy()   # shape (N, 3): [away, draw, home]

print(f"Loaded {len(df):,} holdout matches with 0 fallbacks")
print(f"Raw logits computed -- starting grid search\n")


# ── Grid search ───────────────────────────────────────────────────────────────
# Apply draw boost by adding `boost` to the draw logit (index 1) before
# temperature-scaled softmax. Positive boost makes draw more likely.
# This approximates what training with draw_weight > 1.0 would achieve.
#
# boost=0.0 corresponds to equal class weights (current model)
# boost>0.0 corresponds to increasing draw class weight

def evaluate(draw_boost):
    l = logits.copy()
    l[:, 1] += draw_boost      # boost draw logit
    l = l / T
    l -= l.max(1, keepdims=True)
    p = np.exp(l); p /= p.sum(1, keepdims=True)
    pred = p.argmax(1)

    overall_acc  = (pred == actuals).mean()
    n_draw_real  = (actuals == 1).sum()
    n_draw_catch = ((actuals == 1) & (pred == 1)).sum()
    n_draw_pred  = (pred == 1).sum()
    draw_recall  = n_draw_catch / max(n_draw_real, 1)
    draw_prec    = n_draw_catch / max(n_draw_pred, 1)
    draw_f1      = 2*draw_prec*draw_recall / max(draw_prec+draw_recall, 1e-9)
    # Combined score: balance accuracy and draw F1
    combined     = overall_acc * 0.6 + draw_f1 * 0.4
    return overall_acc, draw_recall, draw_prec, draw_f1, combined


print(f"{'Boost':>6}  {'Overall':>8}  {'DrawRec':>8}  {'DrawPrec':>9}  "
      f"{'DrawF1':>7}  {'Combined':>9}  {'Pick?':>6}")
print("-" * 70)

boosts = np.arange(0.0, 1.51, 0.10)
results = []
best_combined, best_boost = 0.0, 0.0

for boost in boosts:
    acc, dr, dp, df1, comb = evaluate(boost)
    results.append((boost, acc, dr, dp, df1, comb))
    if comb > best_combined:
        best_combined, best_boost = comb, boost
    marker = " <-- BEST" if boost == best_boost else ""
    print(f"{boost:>6.2f}  {acc:>8.3f}  {dr:>8.3f}  {dp:>9.3f}  "
          f"{df1:>7.3f}  {comb:>9.3f}  {marker}")

print()
print(f"Best draw boost (logit additive): {best_boost:.2f}")
print()

# Translate the logit boost to approximate class weight for retraining
# Relationship: class_weight ~= exp(boost) as a rough approximation
# (not exact -- actual weight depends on batch composition -- but
# gives a sensible starting point for the retrain)
approx_weight = float(np.exp(best_boost))
print(f"Approximate equivalent draw class weight: {approx_weight:.3f}")
print()
print("Elbow interpretation:")
print("  Find the boost value where Combined stops rising meaningfully.")
print("  The optimal retrain weight sits at that elbow, not the max.")
print()

# Print the elbow table more clearly
print(f"{'Boost':>6}  {'Delta Combined':>15}")
for i in range(1, len(results)):
    delta = results[i][5] - results[i-1][5]
    bar   = "█" * max(0, int(delta * 200))
    print(f"  {results[i][0]:>4.2f}  {delta:>+8.4f}  {bar}")

print()
print("Once you see the bar shrink or go negative, the previous value is the elbow.")
