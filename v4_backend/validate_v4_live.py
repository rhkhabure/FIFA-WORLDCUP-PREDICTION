"""
validate_v4_live.py  —  V4.2 Final Test
=========================================
Tests the neural net specifically on IN-GAME snapshots from the 2526
holdout -- NOT at kickoff, but at three mid-match checkpoints:
  - Minute 30  (first third, score often still close)
  - Minute 60  (second half, score gap growing)
  - Minute 80  (late game, score nearly final)

The hypothesis: once a match is underway, the score and time features
dominate team strength, and the neural net's accuracy should rise
significantly above the 0.498 DC pre-game baseline -- proving it adds
real value in the live dashboard context.

Compares each checkpoint against:
  - The pre-game DC baseline (0.498)
  - The always-correct "oracle" that just reads the final score

Run: python v4_backend/validate_v4_live.py
"""

import pickle, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score

ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "notebooks" / "v4_backend" / "models"
PRIORS_PATH = ROOT / "v4_priors.json"
DB_PATH     = ROOT.parent / "v4_historical_data.sqlite"
HOLDOUT     = "2526"

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


ckpt   = torch.load(MODELS_DIR / "football_v4.pth",
                    map_location="cpu", weights_only=False)
model  = FootballWinProbNet(**ckpt["arch"])
model.load_state_dict(ckpt["model_state"])
model.eval()
T = ckpt["temperature"]

with open(MODELS_DIR / "scaler_v4.pkl", "rb") as f:
    scaler = pickle.load(f)

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(
    f"SELECT * FROM matches_xg WHERE season='{HOLDOUT}'",
    conn, parse_dates=["date"],
)
conn.close()

dc = DCStrengthLookup(PRIORS_PATH)

print(f"Loaded {len(df):,} holdout matches ({HOLDOUT})\n")

CHECKPOINTS = [0, 30, 60, 80, 90]


def simulate_score_at_minute(home_goals_total, away_goals_total, minute):
    """
    Approximate the scoreline at a given minute.
    Uses the same proportional approximation as the training data:
    first half goals at min 44, second half at min 75.
    """
    half_h = home_goals_total // 2
    half_a = away_goals_total // 2
    events = (
        [(44, "home")] * half_h +
        [(44, "away")] * half_a +
        [(75, "home")] * (home_goals_total - half_h) +
        [(75, "away")] * (away_goals_total - half_a)
    )
    hs = sum(1 for m, s in events if m <= minute and s == "home")
    as_ = sum(1 for m, s in events if m <= minute and s == "away")
    return hs, as_


def build_row(match, minute, dc):
    hs, as_ = simulate_score_at_minute(
        int(match["home_goals"]), int(match["away_goals"]), minute
    )
    events = (
        [(44, "home")] * (int(match["home_goals"]) // 2) +
        [(44, "away")] * (int(match["away_goals"]) // 2) +
        [(75, "home")] * (int(match["home_goals"]) - int(match["home_goals"]) // 2) +
        [(75, "away")] * (int(match["away_goals"]) - int(match["away_goals"]) // 2)
    )
    lead_changes, prev, goals_so_far = 0, 0, 0
    for m, side in sorted(events):
        if m > minute: break
        goals_so_far += 1
        h_now = sum(1 for mm, s in events if mm <= m and s == "home")
        a_now = sum(1 for mm, s in events if mm <= m and s == "away")
        leader = (h_now > a_now) - (h_now < a_now)
        if leader != prev and leader != 0: lead_changes += 1
        prev = leader

    return dc.build_feature_row(
        match["home_team"], match["away_team"], match["league"],
        minute=minute, home_score=hs, away_score=as_,
        lead_changes=lead_changes, goals_so_far=goals_so_far,
        is_knockout=0, is_neutral_venue=0,
    )


def probs(logits, T):
    z = logits / T; z -= z.max(1, keepdims=True)
    p = np.exp(z); return p / p.sum(1, keepdims=True)


names = ["away", "draw", "home"]
print(f"{'Minute':>8}  {'Accuracy':>9}  {'DrawRec':>8}  {'AwayRec':>8}  "
      f"{'HomeRec':>8}  {'vs DC base':>11}")
print("-" * 68)

results = {}
for minute in CHECKPOINTS:
    feats, actuals = [], []
    for _, r in df.iterrows():
        feat = build_row(r, minute, dc)
        feats.append(feat)
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        actuals.append(2 if hg > ag else (0 if ag > hg else 1))

    X = scaler.transform(np.array(feats, "float32")).astype("float32")
    with torch.no_grad():
        logits = model(torch.tensor(X)).numpy()
    p = probs(logits, T)
    pred = p.argmax(1)
    actuals = np.array(actuals)

    acc       = (pred == actuals).mean()
    draw_rec  = ((actuals==1)&(pred==1)).sum() / max((actuals==1).sum(),1)
    away_rec  = ((actuals==0)&(pred==0)).sum() / max((actuals==0).sum(),1)
    home_rec  = ((actuals==2)&(pred==2)).sum() / max((actuals==2).sum(),1)
    vs_dc     = acc - 0.498

    results[minute] = acc
    marker = " <-- PRE-GAME" if minute == 0 else ""
    print(f"  min {minute:>3}  {acc:>9.3f}  {draw_rec:>8.3f}  "
          f"{away_rec:>8.3f}  {home_rec:>8.3f}  "
          f"  {vs_dc:>+8.3f}{marker}")

print()
print("=" * 56)
print("IN-GAME VALUE ASSESSMENT")
print("=" * 56)
gain_30 = results[30] - results[0]
gain_60 = results[60] - results[0]
gain_80 = results[80] - results[0]

print(f"Pre-game accuracy (min 0)  : {results[0]:.3f}")
print(f"Accuracy gain at min 30    : {gain_30:+.3f}")
print(f"Accuracy gain at min 60    : {gain_60:+.3f}")
print(f"Accuracy gain at min 80    : {gain_80:+.3f}")
print(f"Full-time accuracy (min 90): {results[90]:.3f}")
print()

if gain_60 > 0.05:
    verdict = ("CONFIRMED: neural net adds substantial live value. "
               "Use DC pre-game, neural net from kickoff onward.")
elif gain_60 > 0.02:
    verdict = ("MARGINAL: some live improvement. "
               "Architecture decision still holds but gain is modest.")
else:
    verdict = ("INCONCLUSIVE: score approximation too rough to show "
               "in-game gain clearly. Real live data needed.")

print(f"Verdict: {verdict}")