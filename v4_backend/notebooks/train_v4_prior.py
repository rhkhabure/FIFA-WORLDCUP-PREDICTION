"""
train_v4_prior.py  —  V4.2
===========================
Trains the Dixon-Coles xG prior on seasons 2122-2324 ONLY.
Season 2425 is explicitly excluded -- it is the holdout for Phase 2 validation.
Saves output to v4_priors.json.

Run with: python train_v4_prior.py
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DB_PATH     = Path("v4_historical_data.sqlite")
OUTPUT_PATH = Path("v4_backend/v4_priors.json")
HOLDOUT_SEASON = "2425"   # Never touched during training
DECAY_RATE  = 0.0018      # e^(-xi * days_ago) -- same as before

# ── Load training data ───────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df_all = pd.read_sql(
    f"SELECT * FROM matches_xg WHERE season != '{HOLDOUT_SEASON}'",
    conn,
    parse_dates=["date"],
)
conn.close()

print(f"Training rows (seasons 2122-2324): {len(df_all):,}")
print(f"Holdout season '{HOLDOUT_SEASON}' explicitly excluded.\n")

# ── NLL function (same logic as dixon_coles_xg.py) ──────────────────────────
def continuous_dixon_coles_nll(params, df, n_teams, static_gamma,
                                decay_rate=DECAY_RATE):
    alphas = params[:n_teams]
    betas  = params[n_teams:2*n_teams]
    rho    = params[2*n_teams]

    h_idx      = df["home_idx"].values
    a_idx      = df["away_idx"].values
    obs_xg_h   = df["obs_xg_h"].values
    obs_xg_a   = df["obs_xg_a"].values
    act_g_h    = df["act_g_h"].values
    act_g_a    = df["act_g_a"].values
    days_ago   = df["days_ago"].values

    lambdas = np.clip(alphas[h_idx] * betas[a_idx] * static_gamma, 1e-5, 15.0)
    mus     = np.clip(alphas[a_idx] * betas[h_idx],                1e-5, 15.0)

    ll_home = obs_xg_h * np.log(lambdas) - lambdas
    ll_away = obs_xg_a * np.log(mus)     - mus

    rho_terms = np.ones(len(df))
    m00 = (act_g_h == 0) & (act_g_a == 0)
    m10 = (act_g_h == 1) & (act_g_a == 0)
    m01 = (act_g_h == 0) & (act_g_a == 1)
    m11 = (act_g_h == 1) & (act_g_a == 1)
    rho_terms[m00] = 1.0 - lambdas[m00] * mus[m00] * rho
    rho_terms[m10] = 1.0 + mus[m10] * rho
    rho_terms[m01] = 1.0 + lambdas[m01] * rho
    rho_terms[m11] = 1.0 - rho
    rho_terms = np.clip(rho_terms, 1e-5, 10.0)

    weights = np.exp(-decay_rate * days_ago)
    return -np.sum((ll_home + ll_away + np.log(rho_terms)) * weights)

# ── Train one league ──────────────────────────────────────────────────────────
def train_league(league_name, df_league):
    teams = sorted(df_league["home_team"].unique())
    n     = len(teams)
    t2i   = {t: i for i, t in enumerate(teams)}

    ref_date = df_league["date"].max()
    df_league = df_league.copy()
    df_league["home_idx"]  = df_league["home_team"].map(t2i)
    df_league["away_idx"]  = df_league["away_team"].map(t2i)
    df_league["obs_xg_h"]  = df_league["home_xg"]
    df_league["obs_xg_a"]  = df_league["away_xg"]
    df_league["act_g_h"]   = df_league["home_goals"].astype(int)
    df_league["act_g_a"]   = df_league["away_goals"].astype(int)
    df_league["days_ago"]  = (ref_date - df_league["date"]).dt.days

    static_gamma = df_league["home_xg"].sum() / max(df_league["away_xg"].sum(), 1.0)

    init  = np.concatenate([np.ones(n), np.ones(n), [0.0]])
    bounds = [(0.1, 3.5)] * (2 * n) + [(-0.25, 0.25)]

    def constraint_mean_alpha(params):
        return np.mean(params[:n]) - 1.0

    print(f"  Optimising {league_name} ({n} teams, {len(df_league):,} matches) …")
    res = minimize(
        fun=continuous_dixon_coles_nll,
        x0=init,
        args=(df_league, n, static_gamma),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": constraint_mean_alpha},
        options={"maxiter": 300, "ftol": 1e-9, "disp": False},
    )

    if not res.success:
        raise RuntimeError(
            f"Optimizer did not converge for {league_name}: {res.message}\n"
            f"This usually means too few matches or extreme xG values.  "
            f"Inspect the raw data before proceeding."
        )

    opt = res.x
    teams_dict = {
        team: {"alpha": float(opt[i]), "beta": float(opt[i + n])}
        for i, team in enumerate(teams)
    }
    meta = {
        "gamma_home_advantage": float(static_gamma),
        "rho_draw_correction":  float(opt[2 * n]),
        "trained_on_seasons":   sorted(df_league["season"].unique().tolist()),
        "holdout_season":       HOLDOUT_SEASON,
        "trained_at":           datetime.now(timezone.utc).isoformat(),
        "n_matches":            len(df_league),
    }
    mean_alpha = float(np.mean([v["alpha"] for v in teams_dict.values()]))
    print(f"  ✅ Converged. mean(α)={mean_alpha:.4f}  γ={meta['gamma_home_advantage']:.3f}"
          f"  ρ={meta['rho_draw_correction']:.4f}")
    return {"teams": teams_dict, "meta": meta}

# ── Run all leagues ───────────────────────────────────────────────────────────
all_priors = {}
for league in df_all["league"].unique():
    df_league = df_all[df_all["league"] == league]
    all_priors[league] = train_league(league, df_league)

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(all_priors, f, indent=2)

print(f"\n✅ Saved clean priors to {OUTPUT_PATH}")
print(f"   Training seasons : 2122, 2223, 2324")
print(f"   Holdout season   : {HOLDOUT_SEASON}  (never seen by the optimizer)")
print(f"\nPhase 2 ready: run validate_v4_prior.py to test on the holdout.")