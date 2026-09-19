"""
backfill_history.py — V4.2
===========================
Retroactively populate predictions DB for GW1-GW3 of 2025/26 season.

Run once from v4_web/ directory:
    python backfill_history.py

What it does:
1. Calls FotMob get_matches_by_date for each past matchday
2. Extracts PL matches with scores and lineups
3. Computes DC prior odds for each match
4. Calls find_finished_match_by_teams for actual result
5. Inserts everything into predictions DB

Requires:
- PARSE_BOT_KEY in .env
- v4_priors.json accessible
- predictions.db already initialised (run server once first)
"""

import asyncio
import json
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.stats import poisson

# ── Config ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DB_PATH    = ROOT / "data" / "predictions.db"
PRIORS_PATH= ROOT.parent / "v4_backend" / "v4_priors.json"
ENV_PATH   = ROOT / ".env"

# Load API key
PARSE_BOT_KEY = ""
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("PARSE_BOT_KEY="):
            PARSE_BOT_KEY = line.split("=", 1)[1].strip()

# Load priors
with open(PRIORS_PATH) as f:
    priors_db = json.load(f)

pl   = priors_db["ENG-Premier League"]["teams"]
meta = priors_db["ENG-Premier League"]["meta"]
GAMMA_CAL = meta["gamma_home_advantage"]
RHO       = meta["rho_draw_correction"]
DP        = 0.10
SEASON    = 2025

# All GW1-GW3 matchdays (2025/26 season started Aug 21, 2026)
MATCHDAYS = [
    "2026-08-21", "2026-08-22", "2026-08-23", "2026-08-24",  # GW1
    "2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31",  # GW2
    "2026-09-13",                                              # GW3 extra (Leeds)
]

TEAM_ALIASES = {
    "AFC Bournemouth":              "Bournemouth",
    "Nottingham Forest":            "Nottingham Forest",
    "Tottenham Hotspur":            "Tottenham Hotspur",
    "Wolverhampton Wanderers":      "Wolverhampton Wanderers",
    "Manchester City":              "Manchester City",
    "Manchester United":            "Manchester United",
    "Newcastle United":             "Newcastle United",
    "Brighton & Hove Albion":       "Brighton & Hove Albion",
    "West Ham United":              "West Ham United",
}

PRIORS_ALIASES = {
    "Bournemouth":          "AFC Bournemouth",
    "Tottenham Hotspur":    "Tottenham Hotspur",
    "Newcastle United":     "Newcastle United",
    "Brighton & Hove Albion": "Brighton & Hove Albion",
    "West Ham United":      "West Ham United",
    "Man City":             "Manchester City",
    "Man Utd":              "Manchester United",
    "Spurs":                "Tottenham Hotspur",
    "Nott'm Forest":        "Nottingham Forest",
    "Wolves":               "Wolverhampton Wanderers",
    "Coventry":             "Coventry City",
    "Hull":                 "Hull City",
    "Leeds":                "Leeds United",
    "Sunderland":           "Sunderland AFC",
    "Ipswich":              "Ipswich Town",
}

FOTMOB_BASE = "https://api.parse.bot/scraper/645b8e03-271d-4c85-97e7-35d5733a2d78"

# ── Helpers ─────────────────────────────────────────────────────────────────

def fotmob_fetch(endpoint: str, params: dict) -> dict:
    qs  = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{FOTMOB_BASE}/{endpoint}?{qs}"
    req = urllib.request.Request(url, headers={"X-API-Key": PARSE_BOT_KEY})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def get_priors_key(name: str) -> str:
    return PRIORS_ALIASES.get(name, name)


def effective_gamma(date_str: str) -> float:
    d = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    s = datetime(2026, 8, 21, tzinfo=timezone.utc)
    days  = max(0, (d - s).days)
    gw    = max(1, days // 7 + 1)
    FLOOR = 0.60; GW_FULL = 6
    decay = FLOOR + (1.0 - FLOOR) * min(gw / GW_FULL, 1.0)
    return round(1.0 + (GAMMA_CAL - 1.0) * decay, 4)


def dc_probs(home: str, away: str, gamma: float):
    all_alpha = [v["alpha"] for v in pl.values()]
    all_beta  = [v["beta"]  for v in pl.values()]
    q25a = float(np.percentile(all_alpha, 25))
    q75b = float(np.percentile(all_beta,  75))
    h = pl.get(get_priors_key(home), {"alpha": q25a, "beta": q75b})
    a = pl.get(get_priors_key(away), {"alpha": q25a, "beta": q75b})
    lam = float(np.clip(h["alpha"] * a["beta"] * gamma, 1e-5, 15))
    mu  = float(np.clip(a["alpha"] * h["beta"],         1e-5, 15))
    N = 9
    j = np.outer(poisson.pmf(np.arange(N), lam), poisson.pmf(np.arange(N), mu))
    j[0,0]*=max(1-lam*mu*RHO,1e-5); j[1,0]*=max(1+mu*RHO,1e-5)
    j[0,1]*=max(1+lam*RHO,1e-5);    j[1,1]*=max(1-RHO,1e-5)
    j /= j.sum()
    ph=float(np.tril(j,-1).sum()); pd=float(np.trace(j)); pa=float(np.triu(j,+1).sum())
    tr=DP/2; ph2=max(ph-tr,0); pa2=max(pa-tr,0); pd2=pd+DP
    t=ph2+pd2+pa2
    return round(ph2/t*100,1), round(pd2/t*100,1), round(pa2/t*100,1), round(lam,3), round(mu,3)


def parse_score(score_str: str):
    """Parse '2 - 1' → (2, 1)"""
    try:
        parts = score_str.strip().split(" - ")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None, None


def get_pl_matches_for_date(date_str: str) -> list[dict]:
    """Get all PL matches from FotMob for a given date."""
    date_code = date_str.replace("-", "")
    try:
        data = fotmob_fetch("get_matches_by_date", {"date": date_code})
        leagues = data.get("data", {}).get("leagues", [])
        for league in leagues:
            if "Premier" in league.get("name", ""):
                return league.get("matches", [])
    except Exception as e:
        print(f"  FotMob error for {date_str}: {e}")
    return []


def get_lineup_for_match(fotmob_id: str) -> dict | None:
    """Fetch confirmed lineup from FotMob."""
    try:
        time.sleep(1)  # rate limit
        data = fotmob_fetch("get_match_lineup", {"match_id": fotmob_id})
        if data.get("status") != "success":
            return None
        d = data.get("data", {})
        return {
            "home_formation": d.get("home", {}).get("formation", ""),
            "away_formation": d.get("away", {}).get("formation", ""),
            "home_players"  : [p.get("name","") for p in d.get("home",{}).get("players",[{}])[:11]],
            "away_players"  : [p.get("name","") for p in d.get("away",{}).get("players",[{}])[:11]],
        }
    except Exception as e:
        print(f"  Lineup error {fotmob_id}: {e}")
        return None


# ── Main backfill ───────────────────────────────────────────────────────────

def backfill():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    now = datetime.now(timezone.utc).isoformat()

    total_inserted = 0
    total_updated  = 0

    for date_str in MATCHDAYS:
        print(f"\n── {date_str} ──────────────────────────────")
        matches = get_pl_matches_for_date(date_str)
        if not matches:
            print("  No PL matches found")
            continue

        gamma = effective_gamma(date_str)
        print(f"  {len(matches)} PL matches  |  gamma={gamma:.3f}")

        for m in matches:
            home_raw = m.get("home", {}).get("name", "")
            away_raw = m.get("away", {}).get("name", "")
            home     = TEAM_ALIASES.get(home_raw, home_raw)
            away     = TEAM_ALIASES.get(away_raw, away_raw)
            fotmob_id= str(m.get("id", ""))
            st       = m.get("status", {}) or {}

            # Parse score
            score_str = st.get("scoreStr", "")
            hg, ag    = parse_score(score_str) if score_str else (None, None)
            finished  = st.get("finished", False)
            started   = st.get("started", False)

            if not started:
                print(f"  Skip (not started): {home} vs {away}")
                continue

            # Compute DC odds
            ph, pd, pa, lam, mu = dc_probs(home, away, gamma)

            # Determine actual result
            actual = None
            if finished and hg is not None and ag is not None:
                actual = "H" if hg > ag else ("D" if hg == ag else "A")

            # Determine pre_correct
            pred = "H" if ph>pd and ph>pa else ("D" if pd>pa else "A")
            pre_correct = (1 if pred==actual else 0) if actual else None

            # Use FotMob ID as match_id for historical matches
            mid = f"fm_{fotmob_id}"

            existing = conn.execute(
                "SELECT match_id, pre_logged_at FROM predictions WHERE match_id=?",
                (mid,)
            ).fetchone()

            if existing:
                # Update result if we now have it
                if actual and not conn.execute(
                    "SELECT actual_result FROM predictions WHERE match_id=?", (mid,)
                ).fetchone()["actual_result"]:
                    conn.execute("""
                        UPDATE predictions SET
                            actual_result=?, actual_hg=?, actual_ag=?,
                            result_logged_at=?, pre_correct=?
                        WHERE match_id=?
                    """, (actual, hg, ag, now, pre_correct, mid))
                    print(f"  Updated result: {home} {hg}-{ag} {away} → {actual}")
                    total_updated += 1
                else:
                    print(f"  Skip (exists): {home} vs {away}")
                continue

            # Get lineup
            lineup = get_lineup_for_match(fotmob_id) if fotmob_id else None
            home_form = lineup["home_formation"] if lineup else ""
            away_form = lineup["away_formation"] if lineup else ""
            home_pl   = json.dumps(lineup["home_players"] if lineup else [])
            away_pl   = json.dumps(lineup["away_players"] if lineup else [])
            adj_logged = now if lineup else None

            conn.execute("""
                INSERT INTO predictions
                    (match_id, home_team, away_team, kickoff_utc, season,
                     pre_dc_home, pre_dc_draw, pre_dc_away, pre_dc_lam, pre_dc_mu,
                     pre_logged_at,
                     adj_home, adj_draw, adj_away, adj_lam, adj_mu,
                     absent_home, absent_away, adj_logged_at,
                     actual_result, actual_hg, actual_ag, result_logged_at,
                     pre_correct)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                mid, home, away, f"{date_str}T12:00:00Z", SEASON,
                ph, pd, pa, lam, mu, now,
                ph, pd, pa, lam, mu,   # adj same as pre for historical (no real-time lineup)
                "[]", "[]", adj_logged,
                actual, hg, ag, now if actual else None,
                pre_correct,
            ))

            mark = f"→ {actual} {'✓' if pre_correct==1 else '✗' if pre_correct==0 else '?'}" if actual else "→ pending"
            print(f"  Inserted: {home} vs {away}  |  DC {ph}/{pd}/{pa}  {mark}")
            total_inserted += 1

        conn.commit()
        time.sleep(2)  # between matchdays

    conn.close()
    print(f"\n{'='*50}")
    print(f"Done. {total_inserted} inserted, {total_updated} updated")


if __name__ == "__main__":
    backfill()
