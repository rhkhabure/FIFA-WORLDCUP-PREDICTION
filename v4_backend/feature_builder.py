"""
feature_builder.py  —  V4.2
=============================
Replaces the hand-curated FIFA_RANK lookup in common.py with real
Dixon-Coles strength parameters fitted from actual match xG data.

The three features that change:
  home_rank_norm  <- norm(alpha_home)     attack strength, 0-1
  away_rank_norm  <- norm(1/beta_away)    defensive solidity, 0-1
  rank_diff       <- derived from above   same as before, now real data

Everything else (goal_diff, time features, lead_changes_norm,
is_knockout, score_state, strength_x_time) stays identical --
same formula, same column order, same 11-feature vector the
neural net expects.

Usage in common.py:
  from feature_builder import DCStrengthLookup
  dc = DCStrengthLookup("v4_backend/v4_priors.json")
  h_str = dc.get_strength(home_code, league, side="attack")
  a_str = dc.get_strength(away_code, league, side="defence")

Usage in build_feature_row:
  Replace:  h_str, a_str = get_strength(home_code), get_strength(away_code)
  With:     h_str = dc.get_strength(home_code, league, side="attack")
            a_str = dc.get_strength(away_code, league, side="defence")
"""

import json
import numpy as np
from pathlib import Path


# Team-name aliases: soccerdata/Understat sometimes uses different spellings
# from what the live feed or the old FIFA_RANK table uses.
# Add entries here as they're discovered rather than in multiple places.
TEAM_NAME_ALIASES = {
    # Understat name          -> canonical name in v4_priors.json
    "Manchester United"       : "Manchester Utd",
    "Manchester City"         : "Manchester City",
    "Wolverhampton Wanderers" : "Wolves",
    "Tottenham Hotspur"       : "Tottenham",
    "Nottingham Forest"       : "Nott'ham Forest",
    "Leicester City"          : "Leicester",
    "West Bromwich Albion"    : "West Brom",
    "Newcastle United"        : "Newcastle Utd",
    "West Ham United"         : "West Ham",
    "Paris Saint Germain"     : "Paris Saint-Germain",
    "PSG"                     : "Paris Saint-Germain",
    "Atletico Madrid"         : "Atlético Madrid",
    "Atletico de Madrid"      : "Atlético Madrid",
    "Athletic Club"           : "Athletic Club",
}


class DCStrengthLookup:
    """
    Converts Dixon-Coles alpha/beta parameters into the same 0-to-1
    normalised strength scores the neural net expects.

    Normalization is fitted once from the prior file and cached -- the
    same min/max is used for every lookup so the scale is consistent
    across all teams and both sides of the lookup.

    Parameters
    ----------
    priors_path : str or Path
        Path to v4_priors.json produced by train_v4_prior.py.
    draw_propensity : float
        The draw-propensity value confirmed in Phase 2 validation (0.10).
        Stored here so the dashboard can read it from one place rather
        than hardcoding it in multiple files.
    """

    DRAW_PROPENSITY = 0.10  # confirmed optimal in Phase 2 validation

    def __init__(self, priors_path: str | Path):
        self.priors_path = Path(priors_path)
        if not self.priors_path.exists():
            raise FileNotFoundError(
                f"Priors file not found: {self.priors_path}\n"
                f"Run train_v4_prior.py first."
            )

        with open(self.priors_path) as f:
            self._priors = json.load(f)

        # Fit normalization bounds once from all leagues together.
        # Using global bounds (not per-league) so that a team's strength
        # score is comparable across leagues -- necessary for any
        # cross-league tournament simulation.
        all_alpha, all_beta = [], []
        for league_data in self._priors.values():
            for params in league_data["teams"].values():
                all_alpha.append(params["alpha"])
                all_beta.append(params["beta"])

        self._alpha_min = float(min(all_alpha))
        self._alpha_max = float(max(all_alpha))
        self._beta_min  = float(min(all_beta))
        self._beta_max  = float(max(all_beta))

        # Pre-compute per-league bottom-quartile fallbacks (Fix 2 from Phase 2)
        self._fallbacks = {}
        for league, league_data in self._priors.items():
            alphas = [v["alpha"] for v in league_data["teams"].values()]
            betas  = [v["beta"]  for v in league_data["teams"].values()]
            self._fallbacks[league] = {
                "alpha": float(np.percentile(alphas, 25)),
                "beta":  float(np.percentile(betas,  75)),
            }

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def leagues(self) -> list[str]:
        return list(self._priors.keys())

    def get_raw_params(self, team_name: str, league: str) -> dict:
        """
        Return the raw {'alpha': ..., 'beta': ...} for a team.
        Applies name aliases and falls back to bottom-quartile if the team
        is not in the prior (promoted club, cup opponent, etc.).
        """
        canonical = TEAM_NAME_ALIASES.get(team_name, team_name)
        if league not in self._priors:
            # League not in priors at all -- use global mean
            return {"alpha": 1.0, "beta": 1.33}

        teams = self._priors[league]["teams"]
        if canonical in teams:
            return teams[canonical]

        # Not found -- use bottom-quartile fallback (Fix 2)
        return self._fallbacks.get(league, {"alpha": 1.0, "beta": 1.33})

    def was_fallback(self, team_name: str, league: str) -> bool:
        """True if this team will use the fallback (not in the prior)."""
        canonical = TEAM_NAME_ALIASES.get(team_name, team_name)
        if league not in self._priors:
            return True
        return canonical not in self._priors[league]["teams"]

    def get_strength(self, team_name: str, league: str,
                     side: str = "attack") -> float:
        """
        Return a 0-to-1 normalised strength score for use as a neural
        net feature.

        side="attack"  -> normalises alpha  (high alpha = strong attack)
        side="defence" -> normalises 1/beta (low beta = solid defence,
                          so we invert so that high score = good defence)

        The two sides are intentionally separate because:
        - home_rank_norm should reflect how well the HOME team creates
          chances (alpha matters more when you're at home)
        - away_rank_norm should reflect how well the AWAY team defends
          (beta matters more when you're away, absorbing pressure)

        This is a simplification -- a full implementation would use
        lambda = alpha_home * beta_away * gamma for the expected goal
        rate, which the bivariate Poisson already does. But for the neural
        net's feature space, separating attack and defence signals has
        historically worked well and keeps the feature set interpretable.
        """
        params = self.get_raw_params(team_name, league)

        if side == "attack":
            raw = params["alpha"]
            lo, hi = self._alpha_min, self._alpha_max
            norm = (raw - lo) / max(hi - lo, 1e-9)
        elif side == "defence":
            raw = params["beta"]
            lo, hi = self._beta_min, self._beta_max
            # Invert: low beta (solid defence) -> high score
            norm = 1.0 - (raw - lo) / max(hi - lo, 1e-9)
        else:
            raise ValueError(f"side must be 'attack' or 'defence', got '{side}'")

        return float(np.clip(norm, 0.0, 1.0))

    def get_meta(self, league: str) -> dict:
        """Return gamma (home advantage) and rho (draw correction) for a league."""
        if league not in self._priors:
            return {"gamma_home_advantage": 1.25, "rho_draw_correction": -0.05}
        return self._priors[league]["meta"]

    def build_feature_row(
        self,
        home_team: str,
        away_team: str,
        league: str,
        minute: int,
        home_score: int,
        away_score: int,
        lead_changes: int,
        goals_so_far: int,
        is_knockout: int,
        is_neutral_venue: int = 0,
    ) -> list[float]:
        """
        Build one 11-feature row in the exact same column order as the
        neural net's FEATURE_COLS:

          [goal_diff, minute_norm, is_second_half, home_rank_norm,
           away_rank_norm, rank_diff, is_knockout, lead_changes_norm,
           is_neutral_venue, score_state, strength_x_time]

        Replaces common.py's build_feature_row() -- drop-in compatible,
        just adds the league argument.
        """
        goal_diff       = int(np.clip(home_score - away_score, -5, 5))
        minute_norm     = min(minute / 90.0, 1.0)
        is_second_half  = 1 if minute > 45 else 0

        h_str = self.get_strength(home_team, league, side="attack")
        a_str = self.get_strength(away_team, league, side="defence")
        rank_diff = h_str - a_str

        lead_changes_norm = lead_changes / max(1, goals_so_far)
        score_state       = 0 if goal_diff < 0 else (2 if goal_diff > 0 else 1)
        strength_x_time   = rank_diff * (1.0 - minute_norm)

        return [
            goal_diff, minute_norm, is_second_half,
            h_str, a_str, rank_diff,
            int(is_knockout), lead_changes_norm, int(is_neutral_venue),
            score_state, strength_x_time,
        ]

    # ── Convenience: produce a dataframe of all team strengths ────────────────

    def team_strength_table(self, league: str) -> "pd.DataFrame":
        """
        Return a dataframe of every team in a league with their
        normalised attack and defence scores -- useful for sanity-checking
        and for the dashboard's team-profile page.
        """
        import pandas as pd
        if league not in self._priors:
            raise KeyError(f"League '{league}' not in priors.")
        rows = []
        for team, params in self._priors[league]["teams"].items():
            rows.append({
                "team"          : team,
                "alpha_raw"     : params["alpha"],
                "beta_raw"      : params["beta"],
                "attack_norm"   : self.get_strength(team, league, "attack"),
                "defence_norm"  : self.get_strength(team, league, "defence"),
            })
        df = pd.DataFrame(rows).sort_values("attack_norm", ascending=False)
        df["combined_strength"] = (df["attack_norm"] + df["defence_norm"]) / 2
        return df.reset_index(drop=True)


# ── Self-test (run this file directly to confirm everything works) ─────────────
if __name__ == "__main__":
    import sys

    priors_path = Path("v4_backend/v4_priors.json")
    if not priors_path.exists():
        print(f"Priors not found at {priors_path} -- adjust path and rerun.")
        sys.exit(1)

    dc = DCStrengthLookup(priors_path)
    print(f"Loaded priors for leagues: {dc.leagues}\n")

    # Spot checks -- same ones we verified during normalization design
    checks = [
        ("Liverpool",    "ENG-Premier League", "attack"),
        ("Luton",        "ENG-Premier League", "attack"),
        ("Ipswich",      "ENG-Premier League", "attack"),   # should use fallback
        ("Barcelona",    "ESP-La Liga",        "attack"),
        ("Bayern Munich","GER-Bundesliga",     "attack"),
        ("Liverpool",    "ENG-Premier League", "defence"),
        ("Luton",        "ENG-Premier League", "defence"),
    ]
    print(f"{'Team':<22} {'League':<22} {'Side':<8} {'Score':>7}  {'Fallback?':>10}")
    for team, league, side in checks:
        score = dc.get_strength(team, league, side)
        fallback = dc.was_fallback(team, league)
        print(f"  {team:<20} {league:<22} {side:<8} {score:>7.3f}  {'YES' if fallback else 'no':>10}")

    print()
    print("Full ENG-Premier League strength table:")
    import pandas as pd
    print(dc.team_strength_table("ENG-Premier League").to_string(index=False))

    print()
    print("build_feature_row test (Liverpool vs Luton, minute 60, score 2-0):") 
    row = dc.build_feature_row(
        home_team="Liverpool", away_team="Luton",
        league="ENG-Premier League",
        minute=60, home_score=2, away_score=0,
        lead_changes=1, goals_so_far=2,
        is_knockout=0, is_neutral_venue=0,
    )
    cols = ["goal_diff","minute_norm","is_second_half","home_rank_norm",
            "away_rank_norm","rank_diff","is_knockout","lead_changes_norm",
            "is_neutral_venue","score_state","strength_x_time"]
    for name, val in zip(cols, row):
        print(f"  {name:<22}: {val:.4f}")

    print("\nSelf-test complete.")
