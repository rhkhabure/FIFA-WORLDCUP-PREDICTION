"""
constants.py  —  V4.2
======================
Single source of truth for shared constants used across the app.

Import from here — never redefine these inline.
"""

from datetime import timezone, timedelta

# ── Timezone ──────────────────────────────────────────────────────────────────
EAT = timezone(timedelta(hours=3))   # East Africa Time (UTC+3, Nairobi)

# ── Model parameters ──────────────────────────────────────────────────────────
DRAW_PROPENSITY = 0.10   # Dixon-Coles draw correction (ρ fitted at 0.10)

# ── League URL key → FotMob name fragments ────────────────────────────────────
# Used to match FotMob league names when scanning date-cache responses.
# More fragments = more robust matching across FotMob's inconsistent naming.
LEAGUE_FILTERS: dict[str, list[str]] = {
    "pl"        : ["premier league", "england", "eng"],
    "laliga"    : ["laliga", "la liga", "primera", "spain", "esp"],
    "bundesliga": ["bundesliga", "germany", "ger"],
    "seriea"    : ["serie a", "calcio", "italy", "ita"],
    "ligue1"    : ["ligue 1", "ligue1", "france", "fra"],
}

# ── League URL key → full metadata ───────────────────────────────────────────
# Single definition used by the match route, tournament page, and history page.
# priors   : key into v4_priors.json
# comp     : football-data.org competition code
# name     : human-readable display name
# ctx_key  : URL param value (matches the dict key, kept explicit for clarity)
LEAGUE_MAP: dict[str, dict] = {
    "pl": {
        "priors" : "ENG-Premier League",
        "comp"   : "PL",
        "name"   : "Premier League",
        "ctx_key": "pl",
    },
    "laliga": {
        "priors" : "ESP-La Liga",
        "comp"   : "PD",
        "name"   : "La Liga",
        "ctx_key": "laliga",
    },
    "bundesliga": {
        "priors" : "GER-Bundesliga",
        "comp"   : "BL1",
        "name"   : "Bundesliga",
        "ctx_key": "bundesliga",
    },
    "seriea": {
        "priors" : "ITA-Serie A",
        "comp"   : "SA",
        "name"   : "Serie A",
        "ctx_key": "seriea",
    },
    "ligue1": {
        "priors" : "FRA-Ligue 1",
        "comp"   : "FL1",
        "name"   : "Ligue 1",
        "ctx_key": "ligue1",
    },
}
