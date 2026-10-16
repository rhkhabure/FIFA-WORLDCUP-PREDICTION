"""
fix_nba_predictions.py
======================
Repairs NBA rows in predictions.db that were logged while the Elo ratings
failed to load (every team = 1500, every home team = exactly 64%).

What it does:
  1. Backs up predictions.db  ->  predictions.db.bak-before-elo-fix  (once)
  2. Recomputes home_win_prob / away_win_prob for every league='nba' row from
     the real Elo ratings (same formula the live model uses, +100 home adv.)
  3. For finished games, recomputes correct + brier from the stored score
  4. Prints a before/after summary

Football rows are never touched.

Run from v4_web/ (any env with Python):
    python fix_nba_predictions.py
"""
import json, os, shutil, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).parent
DB   = ROOT / "predictions.db"
HOME_ADV = 100.0
NAME = "elo_ratings_v3.json"

# ── find the Elo file (same search as nba_model.py) ───────────────────────────
cands = []
if os.environ.get("NBA_ELO_PATH"):
    cands.append(Path(os.environ["NBA_ELO_PATH"]))
cands.append(ROOT / "nba" / "model" / NAME)
for anc in list(ROOT.parents)[:5]:
    for proj in ("NBA LIVE PREDICTIONS/NBA Live Win Probability", "NBA Live Win Probability"):
        for sub in ("data/raw", "V3/model", "V3/data/raw", "model"):
            cands.append(anc / proj / sub / NAME)
elo_file = next((p for p in cands if p.is_file()), None)
if elo_file is None:
    sys.exit("ERROR: elo_ratings_v3.json not found — copy it from "
             "<NBA project>/data/raw/ into v4_web/nba/model/ first.")
elo = {str(k): float(v) for k, v in json.loads(elo_file.read_text(encoding="utf-8")).items()}
if len(elo) < 30:
    sys.exit(f"ERROR: Elo file has only {len(elo)} teams (expected 30): {elo_file}")
print(f"Elo file : {elo_file}  ({len(elo)} teams)")

if not DB.exists():
    sys.exit(f"ERROR: {DB} not found")

# ── backup once ───────────────────────────────────────────────────────────────
bak = DB.with_name(DB.name + ".bak-before-elo-fix")
if not bak.exists():
    shutil.copy2(DB, bak)
    print(f"Backup   : {bak}")
else:
    print(f"Backup   : already exists ({bak.name}) — not overwritten")

conn = sqlite3.connect(str(DB))
rows = conn.execute(
    "SELECT id, game_id, home_team, away_team, home_team_id, away_team_id, "
    "home_win_prob, home_score, away_score, correct, brier "
    "FROM predictions WHERE league='nba' ORDER BY created_at, id"
).fetchall()

# duplicate check (report only)
seen, dups = set(), 0
for r in rows:
    if r[1] in seen: dups += 1
    seen.add(r[1])
if dups:
    print(f"WARNING  : {dups} duplicate game_id row(s) found — not deleted, please review")


def prob(hid, aid):
    h = elo.get(str(hid), 1500.0) + HOME_ADV
    a = elo.get(str(aid), 1500.0)
    return 1.0 / (1.0 + 10.0 ** ((a - h) / 400.0))


def summarise(items):
    done = [(c, b) for c, b in items if c is not None]
    if not done:
        return "no finished games"
    acc = sum(c for c, _ in done) / len(done) * 100
    br  = sum(b for _, b in done if b is not None) / len(done)
    return f"{sum(c for c, _ in done)}/{len(done)} correct ({acc:.1f}%)  Brier {br:.4f}"


before = [(r[9], r[10]) for r in rows]
after, changed, missing = [], 0, 0
print(f"\n{'game':<12}{'matchup':<12}{'old':>6}{'new':>7}   result")
for (rid, gid, ht, at, hid, aid, old_p, hs, as_, old_c, old_b) in rows:
    if str(hid) not in elo or str(aid) not in elo:
        missing += 1
    p = prob(hid, aid)
    if hs is not None and as_ is not None:
        won = hs > as_
        c = 1 if (p >= 0.5) == won else 0
        b = round((p - (1 if won else 0)) ** 2, 6)
        res = "✓" if c else "✗"
    else:
        c = b = None
        res = "pending"
    conn.execute(
        "UPDATE predictions SET home_win_prob=?, away_win_prob=?, correct=?, brier=? WHERE id=?",
        (round(p, 4), round(1 - p, 4), c, b, rid))
    after.append((c, b))
    changed += 1
    print(f"{str(gid)[-8:]:<12}{(at or '?')+'@'+(ht or '?'):<12}"
          f"{(old_p or 0)*100:>5.0f}%{p*100:>6.0f}%   {res}")
conn.commit()
conn.close()

print(f"\nRows recomputed : {changed}")
if missing:
    print(f"WARNING         : {missing} row(s) had a team ID not in the Elo file (used 1500)")
print(f"Before          : {summarise(before)}")
print(f"After           : {summarise(after)}")
