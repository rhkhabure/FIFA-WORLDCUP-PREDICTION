"""
build_v4_dataset.py  —  V4.2 Phase 3 Step 1
=============================================
Builds the neural net training dataset from the SQLite match database,
using DCStrengthLookup for team strength features instead of the old
hand-curated FIFA_RANK table.

Each finished match becomes ~20 snapshots (one at kickoff, every 5
minutes, and right after each goal). Every snapshot is labelled with
the match's final result (0=away win, 1=draw, 2=home win).

This is the same snapshot-building logic from the World Cup Phase 1
notebook, just reading from SQLite instead of StatsBomb/football-data.

Output: data/processed/features_v4.parquet
        (same 11-column format as features_v2.parquet -- drop-in for
        the Phase 2b training notebook)

Run: python build_v4_dataset.py
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
DB_PATH     = Path("v4_historical_data.sqlite")
PRIORS_PATH = Path("v4_backend/v4_priors.json")
OUT_DIR     = Path("data/processed")
OUT_PATH    = OUT_DIR / "features_v4.parquet"
RESULTS_DIR = Path("results")

for d in (OUT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

HOLDOUT_SEASON = "2425"   # never touched -- reserved for validation

# ── Feature columns (must match neural net's FEATURE_COLS exactly) ────────────
FEATURE_COLS = [
    "goal_diff", "minute_norm", "is_second_half",
    "home_rank_norm", "away_rank_norm", "rank_diff",
    "is_knockout", "lead_changes_norm",
    "is_neutral_venue", "score_state", "strength_x_time",
]
TARGET_COL = "outcome"   # 0=away win, 1=draw, 2=home win

# ── Load strength lookup ──────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_builder import DCStrengthLookup

dc = DCStrengthLookup(PRIORS_PATH)
print(f"Loaded Dixon-Coles priors for: {dc.leagues}")

# ── Load training matches (exclude holdout) ───────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df_matches = pd.read_sql(
    f"SELECT * FROM matches_xg WHERE season != '{HOLDOUT_SEASON}'",
    conn, parse_dates=["date"],
)
conn.close()

print(f"\nTraining matches: {len(df_matches):,}  (holdout '{HOLDOUT_SEASON}' excluded)")
print("Per-league / per-season:")
print(df_matches.groupby(["league","season"]).size()
                .reset_index(name="matches").to_string(index=False))


# ── Snapshot builder ──────────────────────────────────────────────────────────
def build_snapshots(row, dc):
    """
    Turn one finished match into a list of snapshot dicts.

    For a league match we only have the final score, not individual goal
    minutes. The same approximation we used in Phase 1 of the World Cup
    project applies: half-time goals go at minute 44, second-half goals
    at minute 75. This is a known simplification -- it's fine for the
    volume data (club leagues) since it only affects the training
    snapshot distribution slightly, not the final-score label.

    The key insight from Phase 1: what matters is that at minute 0
    (kickoff) the model sees only team strength, and at minute 90 it
    sees the full score. The intermediate snapshots add training signal
    about how leads change over time.
    """
    league     = row["league"]
    home_team  = row["home_team"]
    away_team  = row["away_team"]
    home_goals = int(row["home_goals"])
    away_goals = int(row["away_goals"])

    # Final outcome label (same for every snapshot in this match)
    if   home_goals > away_goals: outcome = 2   # home win
    elif home_goals < away_goals: outcome = 0   # away win
    else:                         outcome = 1   # draw

    # Approximate goal timeline: half the goals at minute 44, rest at 75
    # (same pattern as Phase 1 Cell 7 for football-data.org league matches)
    half_h = home_goals // 2
    half_a = away_goals // 2
    events = (
        [(44, "home")] * half_h +
        [(44, "away")] * half_a +
        [(75, "home")] * (home_goals - half_h) +
        [(75, "away")] * (away_goals - half_a)
    )
    events.sort(key=lambda e: e[0])

    checkpoints = sorted(set([0] + list(range(5, 91, 5)) +
                              [m for m, _ in events] + [90]))

    rows = []
    lead_changes, prev_leader = 0, 0

    for minute in checkpoints:
        hs  = sum(1 for m, s in events if m <= minute and s == "home")
        as_ = sum(1 for m, s in events if m <= minute and s == "away")
        goals_so_far = hs + as_

        leader = (hs > as_) - (hs < as_)
        if leader != prev_leader and leader != 0:
            lead_changes += 1
        prev_leader = leader

        feat = dc.build_feature_row(
            home_team=home_team, away_team=away_team,
            league=league,
            minute=minute,
            home_score=hs, away_score=as_,
            lead_changes=lead_changes,
            goals_so_far=goals_so_far,
            is_knockout=0,      # all league matches, never knockout
            is_neutral_venue=0, # all league matches at home ground
        )

        rows.append({
            "match_id"  : f"{league}_{row['season']}_{home_team}_{away_team}",
            "league"    : league,
            "season"    : row["season"],
            "minute"    : minute,
            **dict(zip(FEATURE_COLS, feat)),
            TARGET_COL  : outcome,
        })

    return rows


# ── Build all snapshots ───────────────────────────────────────────────────────
print("\nBuilding snapshots...")
all_rows  = []
skipped   = 0
fallbacks = 0

for i, (_, row) in enumerate(df_matches.iterrows()):
    # Count fallback usage for diagnostics
    if dc.was_fallback(row["home_team"], row["league"]) or \
       dc.was_fallback(row["away_team"], row["league"]):
        fallbacks += 1

    try:
        snaps = build_snapshots(row, dc)
        all_rows.extend(snaps)
    except Exception as e:
        skipped += 1
        continue

    if (i + 1) % 500 == 0:
        print(f"  {i+1:>5}/{len(df_matches)}  snapshots so far: {len(all_rows):,}")

print(f"\nDone.  Matches processed: {len(df_matches) - skipped:,}  "
      f"(skipped {skipped})")
print(f"Matches using fallback strength: {fallbacks:,} "
      f"({fallbacks/len(df_matches)*100:.1f}%)")


# ── Assemble and validate ─────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
print(f"\nTotal snapshots: {len(df):,}")
print(f"Unique matches : {df['match_id'].nunique():,}")

print("\nVALIDATION")
print("=" * 50)
passed, failed = 0, 0

def chk(name, ok, detail=""):
    global passed, failed
    status = "PASS" if ok else "FAIL"
    print(f"{status}  {name}" + (f"  ({detail})" if detail else ""))
    passed += ok; failed += (not ok)

chk("no nulls in features",
    df[FEATURE_COLS].isnull().sum().sum() == 0)
chk("outcome in {0,1,2}",
    df[TARGET_COL].isin([0,1,2]).all())
chk("goal_diff within +-5",
    df["goal_diff"].between(-5, 5).all())
chk("minute_norm within 0-1",
    df["minute_norm"].between(0, 1).all())
chk("strength scores within 0-1",
    df["home_rank_norm"].between(0,1).all() and
    df["away_rank_norm"].between(0,1).all())

hr = (df[TARGET_COL]==2).mean()
dr = (df[TARGET_COL]==1).mean()
ar = (df[TARGET_COL]==0).mean()
chk("realistic outcome split",
    0.35 <= hr <= 0.60 and 0.10 <= dr <= 0.40,
    f"home {hr:.1%}  draw {dr:.1%}  away {ar:.1%}")
chk("enough matches",
    df["match_id"].nunique() > 2000,
    f"{df['match_id'].nunique()} matches")

early = df[df["minute_norm"] < 0.2]["strength_x_time"].abs().mean()
late  = df[df["minute_norm"] > 0.8]["strength_x_time"].abs().mean()
chk("strength_x_time decays over match",
    late < early,
    f"early {early:.3f} -> late {late:.3f}")

print(f"\nResult: {passed} passed, {failed} failed")

if failed > 0:
    raise RuntimeError(
        f"{failed} validation checks failed -- do not proceed to training "
        f"until these are fixed."
    )


# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}  shape={df.shape}")

# Log to the permanent record
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
entry = f"""
## V4.2 Phase 3 Step 1 — dataset built ({stamp})
- Matches: {df['match_id'].nunique():,}  (seasons 2122-2324, holdout 2425 excluded)
- Snapshots: {len(df):,}
- Outcome split: home {hr:.1%} / draw {dr:.1%} / away {ar:.1%}
- Strength features: DCStrengthLookup (Dixon-Coles alpha/beta, global norm)
- Fallback usage: {fallbacks} matches ({fallbacks/len(df_matches)*100:.1f}%)
- Saved: {OUT_PATH}
"""
rp = RESULTS_DIR / "RESULTS.md"
if not rp.exists():
    rp.write_text("# V4.2 Results Log\n")
with rp.open("a") as f:
    f.write(entry)

print(entry)
print("Phase 3 Step 1 complete -- run the training notebook next.")
