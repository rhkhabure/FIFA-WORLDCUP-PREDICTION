"""
close_phase2c.py
================
Run ONCE from your project folder to write the final, closing verdict on the
Phase 2c draw-fix investigation into the permanent record.

Usage:
    python close_phase2c.py

This does NOT delete football_v2c.pth or scaler_v2c.pkl -- we keep failed
experiments on disk and on the record, same as we keep football_v1.pth around
from the very first draw-spam attempt. A logged negative result is still a
useful result: it stops anyone (including us, months from now) from re-trying
the same idea and wondering why.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

entry = f"""
## Phase 2c — CLOSED, rolled back to football_v2 ({stamp})

**Verdict: the two research-grounded draw fixes did not help. Rolling back.**

Two pushes were tested together in football_v2c.pth:
  1. draw_signal feature (Dixon-Coles-inspired closeness signal)
  2. Ordinal output head (2 sigmoids, guaranteed-valid probabilities) instead of softmax

Full, corrected comparison (both datasets, same split, same real games):

| Dataset                  | v2 (softmax, 11 feat) | v2c (ordinal + draw_signal, 12 feat) |
|---------------------------|------------------------|----------------------------------------|
| Club football test set    | acc 0.550, ll 0.8782   | acc 0.551, ll 0.9030                    |
| Real 2026 World Cup games | acc 0.703, ll 0.6869   | acc 0.661, ll 0.6931                    |

The number that settled it: draw recall on the real World Cup games dropped from
**0.332** (v2, caught ~128 of 385 real draws) to **0.164** (v2c, caught only 63).
The fixes made the model WORSE at recognising draws -- the opposite of the goal.

Working hypotheses for why (kept for the record, not re-tested):
  1. The ordinal head's single-scale structure may be too rigid for a network
     this small (1,382 params) -- it can't express "confident away AND confident
     home are both unlikely, but for different reasons" the way free softmax can.
  2. draw_signal is built from club-football patterns (rank_diff closeness) that
     may not transfer to this specific 48-team expanded World Cup, where many
     debut teams (Curacao, Haiti, Cape Verde, Uzbekistan) default to a generic
     mid-table strength rating rather than a real FIFA rank.
  3. Asking a tiny network to learn a new feature AND a new output structure at
     once, on modest data, may have diluted its limited capacity rather than
     sharpening focus on draws specifically.

**Decision: football_v2.pth (plain softmax, 11 features) remains the production
model.** football_v2c.pth and scaler_v2c.pkl are kept on disk for the audit trail
but are NOT used going forward. The dashboard build starts from football_v2.pth.
"""

rp = RESULTS / "RESULTS.md"
if not rp.exists():
    rp.write_text("# World Cup 2026 Win Probability — Results Log\n")
with rp.open("a") as f:
    f.write(entry)

mc = RESULTS / "metrics.csv"
row = pd.DataFrame([{
    "timestamp": stamp, "phase": "phase2c_CLOSED_rollback",
    "test_acc": 0.551, "test_logloss": 0.9030,          # v2c club-football, for the record
    "draw_recall": 0.164, "draw_precision": 0.656,       # v2c real-WC draw numbers -- the damning ones
    "notes": "rolled back to football_v2.pth as production model",
}])
row.to_csv(mc, mode="a", header=not mc.exists(), index=False)

print("Appended closing entry to results/RESULTS.md and results/metrics.csv")
print(entry)
print("PHASE 2c CLOSED. Production model: football_v2.pth")
