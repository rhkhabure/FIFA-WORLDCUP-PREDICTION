"""
lineup_adjustment.py  —  V4.2
================================
Lineup-adjusted win probability panel.

Takes the Dixon-Coles pre-game lambda/mu (expected goals) and scales
them based on which key players are absent from the confirmed lineup.

Approach (Option C — hardcoded expert weights):
  - For each team, define key players and their impact on attack/defence
  - If a key player is ABSENT from the confirmed lineup, reduce the
    relevant parameter (attack alpha or defensive beta effectiveness)
  - Re-run DC Poisson with adjusted lambda/mu
  - Return adjusted [p_home%, p_draw%, p_away%]

Impact weights are expressed as fractions of lambda/mu:
  attack_weight  : how much this player contributes to team's attack
                   (fraction of expected goals they're responsible for)
  defence_weight : how much this player contributes to defensive solidity
                   (affects opponent's mu via effective beta)

These are calibrated estimates based on:
  - Season xG contribution data (public)
  - Historical performance when key players absent
  - General football knowledge about player roles

The adjustment is intentionally conservative -- DC already captures
team strength well. We're adjusting for the DELTA from expected squad.

DRAW_PROPENSITY stays at 0.10 throughout.
"""

import numpy as np
from scipy.stats import poisson

DRAW_PROPENSITY = 0.10

# ── Key player impact weights ─────────────────────────────────────────────────
# Format: team_name -> { player_name: (attack_weight, defence_weight) }
#
# attack_weight:  fraction of team lambda lost if this player is absent
# defence_weight: fraction of opponent mu gained if this player is absent
#                 (i.e. team's defensive solidity drops)
#
# Weights per team sum to max ~0.45 for attack, ~0.30 for defence
# (leaving room for collective team play beyond individual stars)

KEY_PLAYERS: dict[str, dict[str, tuple[float, float]]] = {

    # ── Premier League 2025/26 ────────────────────────────────────────────────

    "Arsenal": {
        "Saka"      : (0.18, 0.04),   # Elite creator + pressing leader
        "Odegaard"  : (0.15, 0.05),   # Playmaker, controls tempo
        "Havertz"   : (0.12, 0.02),   # Main striker / hold-up
        "Raya"      : (0.00, 0.10),   # Sweeper keeper, crucial to high line
        "Saliba"    : (0.01, 0.09),   # Best defender in PL
    },

    "Aston Villa": {
        "Watkins"   : (0.18, 0.02),   # Top scorer, pressing engine
        "Rogers"    : (0.12, 0.03),   # Creative hub since Coutinho era
        "McGinn"    : (0.07, 0.06),   # Box-to-box anchor
        "Martinez"  : (0.00, 0.10),   # World-class keeper
        "Konsa"     : (0.01, 0.07),   # Best CB, set-piece threat
    },

    "Bournemouth": {
        "Evanilson" : (0.16, 0.02),   # Top scorer
        "Kluivert"  : (0.13, 0.03),   # Creative outlet
        "Kerkez"    : (0.06, 0.05),   # Left back who attacks relentlessly
        "Flekken"   : (0.00, 0.09),   # Shot-stopper, key to low-block
        "Zabarnyi"  : (0.01, 0.07),
    },

    "Brentford": {
        "Mbeumo"    : (0.17, 0.02),   # Star forward
        "Wissa"     : (0.14, 0.02),   # Goals and pace
        "Norgaard"  : (0.04, 0.08),   # Midfield anchor, everything goes through him
        "Flekken"   : (0.00, 0.09),
        "Collins"   : (0.01, 0.07),
    },

    "Brighton": {
        "Joao Pedro": (0.16, 0.02),
        "Mitoma"    : (0.12, 0.03),   # Dribbling and crossing
        "Gross"     : (0.08, 0.06),   # Creative linchpin in midfield
        "Verbruggen": (0.00, 0.09),
        "Dunk"      : (0.01, 0.08),   # Captain, organises defence
    },

    "Chelsea": {
        "Palmer"    : (0.22, 0.02),   # By far their most important player
        "Jackson"   : (0.12, 0.02),
        "Caicedo"   : (0.03, 0.09),   # Defensive midfielder, protects back 4
        "Sanchez"   : (0.00, 0.08),
        "Colwill"   : (0.01, 0.07),
    },

    "Crystal Palace": {
        "Eze"       : (0.18, 0.03),   # Most creative player
        "Mateta"    : (0.14, 0.02),
        "Olise"     : (0.13, 0.03),   # If still there
        "Henderson" : (0.00, 0.09),
        "Guehi"     : (0.01, 0.08),   # England CB, key organiser
    },

    "Everton": {
        "Calvert-Lewin": (0.14, 0.02),
        "McNeil"    : (0.11, 0.03),
        "Pickford"  : (0.00, 0.10),   # Often their best player
        "Tarkowski" : (0.01, 0.08),
        "Branthwaite": (0.01, 0.07),
    },

    "Fulham": {
        "Vinicius"  : (0.15, 0.02),
        "Andreas"   : (0.12, 0.04),   # Runs the midfield
        "Wilson"    : (0.10, 0.03),
        "Leno"      : (0.00, 0.10),   # Consistently excellent
        "Diop"      : (0.01, 0.07),
    },

    "Ipswich": {
        "Hirst"     : (0.14, 0.02),
        "Hutchinson": (0.10, 0.04),
        "Morsy"     : (0.04, 0.08),   # Captain, midfield press leader
        "Flaherty"  : (0.00, 0.08),
        "O'Shea"    : (0.01, 0.07),
    },

    "Leeds": {
        "Gnonto"    : (0.14, 0.02),   # Most dangerous attacker
        "Bamford"   : (0.13, 0.02),
        "Adams"     : (0.05, 0.08),   # Midfield destroyer
        "Meslier"   : (0.00, 0.09),
        "Struijk"   : (0.01, 0.07),
    },

    "Leicester": {
        "Vardy"     : (0.14, 0.01),
        "Tielemans" : (0.12, 0.04),
        "Ndidi"     : (0.03, 0.09),
        "Ward"      : (0.00, 0.08),
        "Faes"      : (0.01, 0.07),
    },

    "Liverpool": {
        "Salah"     : (0.22, 0.03),   # Irreplaceable — leads PL in G+A
        "Nunez"     : (0.13, 0.02),
        "Mac Allister": (0.08, 0.06), # Controls tempo from deep
        "Alisson"   : (0.00, 0.11),   # World class, huge impact
        "Van Dijk"  : (0.01, 0.09),   # Organises entire backline
    },

    "Manchester City": {
        "Haaland"   : (0.22, 0.01),   # 30+ goals per season machine
        "De Bruyne" : (0.16, 0.03),   # When fit, elite creator
        "Rodri"     : (0.03, 0.10),   # Defensive anchor, everything breaks without him
        "Ederson"   : (0.00, 0.08),
        "Dias"      : (0.01, 0.08),
    },

    "Manchester United": {
        "Fernandes"  : (0.15, 0.04),  # Captain, set pieces, creativity
        "Rashford"   : (0.13, 0.02),
        "Hojlund"    : (0.12, 0.02),
        "Onana"      : (0.00, 0.08),
        "Maguire"    : (0.01, 0.06),
    },

    "Newcastle United": {
        "Isak"       : (0.20, 0.02),  # Top 5 striker in PL
        "Gordon"     : (0.13, 0.03),
        "Guimaraes"  : (0.08, 0.07), # Box-to-box, crucial in press
        "Pope"       : (0.00, 0.10),
        "Schar"      : (0.01, 0.07),
    },

    "Nottingham Forest": {
        "Awoniyi"    : (0.14, 0.02),
        "Gibbs-White": (0.14, 0.04),  # Creative hub
        "Elanga"     : (0.11, 0.02),
        "Turner"     : (0.00, 0.09),
        "Murillo"    : (0.01, 0.08),
    },

    "Southampton": {
        "Ward-Prowse": (0.13, 0.04),  # Set pieces specialist
        "Armstrong"  : (0.12, 0.03),
        "Bazunu"     : (0.00, 0.08),
        "Harwood-Bellis": (0.01, 0.07),
        "Walker-Peters": (0.05, 0.04),
    },

    "Sunderland": {
        "Amad"       : (0.16, 0.02),  # On loan from Man Utd, key attacker
        "Stewart"    : (0.13, 0.02),
        "Neil"       : (0.05, 0.07),
        "Patterson"  : (0.00, 0.08),
        "Batth"      : (0.01, 0.07),
    },

    "Tottenham": {
        "Son"        : (0.18, 0.03),  # Still their best player
        "Maddison"   : (0.14, 0.04),
        "Kulusevski" : (0.11, 0.03),
        "Vicario"    : (0.00, 0.09),
        "Romero"     : (0.01, 0.08),  # Aggressive CB, important to press
    },

    "West Ham": {
        "Paqueta"    : (0.14, 0.03),
        "Bowen"      : (0.14, 0.03),
        "Kudus"      : (0.12, 0.02),
        "Fabianski"  : (0.00, 0.08),
        "Zouma"      : (0.01, 0.07),
    },

    "Wolverhampton Wanderers": {
        "Cunha"      : (0.18, 0.02),  # Their best player by a distance
        "Hwang"      : (0.12, 0.02),
        "Lemina"     : (0.05, 0.07),
        "Sa"         : (0.00, 0.09),
        "Semedo"     : (0.05, 0.04),
    },

    # Championship / cup sides (minimal data, conservative weights)
    "Hull City": {
        "Ozan Tufan" : (0.12, 0.02),
        "Liam Delap" : (0.14, 0.02),
    },
    "Coventry": {
        "Gyokeres"   : (0.18, 0.02),  # If still at club
        "Palmer"     : (0.12, 0.03),
    },
}


# ── Adjustment function ────────────────────────────────────────────────────────

def compute_lineup_adjusted_odds(
    home_name: str,
    away_name: str,
    home_lineup: list[str],
    away_lineup: list[str],
    base_lam: float,
    base_mu: float,
    rho: float = 0.0,
) -> tuple[list[float], float, float]:
    """
    Adjust DC Poisson probabilities based on confirmed lineup.

    Parameters
    ----------
    home_name    : team name matching KEY_PLAYERS keys
    away_name    : team name matching KEY_PLAYERS keys
    home_lineup  : list of confirmed home starter names
    away_lineup  : list of confirmed away starter names
    base_lam     : home expected goals from DC (pre-lineup)
    base_mu      : away expected goals from DC (pre-lineup)
    rho          : DC draw correction parameter

    Returns
    -------
    probs        : [p_home%, p_draw%, p_away%]
    adj_lam      : adjusted home xG
    adj_mu       : adjusted away xG
    """
    home_keys = KEY_PLAYERS.get(home_name, {})
    away_keys = KEY_PLAYERS.get(away_name, {})

    # Normalise lineup names for matching -- strip accents, lower case
    def normalise(name: str) -> str:
        import unicodedata
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))
        return name.lower().strip()

    home_lineup_norm = {normalise(p) for p in home_lineup}
    away_lineup_norm = {normalise(p) for p in away_lineup}

    def is_absent(player_name: str, lineup_norm: set[str]) -> bool:
        pn = normalise(player_name)
        # Match on last name or full name
        for lp in lineup_norm:
            if pn in lp or lp in pn or pn.split()[-1] == lp.split()[-1]:
                return False
        return True

    # Accumulate adjustments
    home_attack_loss  = 0.0   # fraction lost from home lambda
    home_defence_loss = 0.0   # fraction added to away mu
    away_attack_loss  = 0.0
    away_defence_loss = 0.0

    for player, (atk, dfc) in home_keys.items():
        if is_absent(player, home_lineup_norm):
            home_attack_loss  += atk
            home_defence_loss += dfc

    for player, (atk, dfc) in away_keys.items():
        if is_absent(player, away_lineup_norm):
            away_attack_loss  += atk
            away_defence_loss += dfc

    # Cap total adjustments so we don't go negative
    home_attack_loss  = min(home_attack_loss,  0.45)
    away_attack_loss  = min(away_attack_loss,  0.45)
    home_defence_loss = min(home_defence_loss, 0.30)
    away_defence_loss = min(away_defence_loss, 0.30)

    # Apply: home attack down, away attack up (home defence weak)
    adj_lam = float(np.clip(
        base_lam * (1.0 - home_attack_loss) * (1.0 + away_defence_loss),
        0.05, 12.0
    ))
    adj_mu = float(np.clip(
        base_mu * (1.0 - away_attack_loss) * (1.0 + home_defence_loss),
        0.05, 12.0
    ))

    # Re-run DC Poisson with adjusted lambdas
    N  = 9
    hp = poisson.pmf(np.arange(N), adj_lam)
    ap = poisson.pmf(np.arange(N), adj_mu)
    j  = np.outer(hp, ap)

    j[0,0] *= max(1.0 - adj_lam * adj_mu * rho, 1e-5)
    j[1,0] *= max(1.0 + adj_mu * rho,            1e-5)
    j[0,1] *= max(1.0 + adj_lam * rho,           1e-5)
    j[1,1] *= max(1.0 - rho,                     1e-5)
    j /= j.sum()

    ph  = float(np.tril(j, -1).sum())
    pd_ = float(np.trace(j))
    pa  = float(np.triu(j, +1).sum())

    tr  = DRAW_PROPENSITY / 2.0
    ph2 = max(ph - tr, 0.0)
    pa2 = max(pa - tr, 0.0)
    pd2 = pd_ + DRAW_PROPENSITY
    tot = ph2 + pd2 + pa2
    if tot == 0:
        return [33.3, 33.3, 33.4], adj_lam, adj_mu

    return (
        [round(ph2/tot*100, 1), round(pd2/tot*100, 1), round(pa2/tot*100, 1)],
        round(adj_lam, 2),
        round(adj_mu, 2),
    )


def get_absent_key_players(
    team_name: str,
    confirmed_lineup: list[str],
) -> list[str]:
    """Return list of key players missing from the confirmed lineup."""
    team_keys = KEY_PLAYERS.get(team_name, {})
    absent = []

    def normalise(name: str) -> str:
        import unicodedata
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))
        return name.lower().strip()

    lineup_norm = {normalise(p) for p in confirmed_lineup}
    for player in team_keys:
        pn = normalise(player)
        found = any(
            pn in lp or lp in pn or pn.split()[-1] == lp.split()[-1]
            for lp in lineup_norm
        )
        if not found:
            absent.append(player)
    return absent
