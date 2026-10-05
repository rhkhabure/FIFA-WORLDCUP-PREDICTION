"""
constants.py  —  v4_web shared constants
=========================================
Football constants (unchanged from V4.2) + NBA additions.
Import from any module in v4_web/:
    from constants import EAT, NBA_TEAMS, NBA_CONFERENCES
"""

from datetime import timezone, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# SHARED
# ─────────────────────────────────────────────────────────────────────────────

EAT = timezone(timedelta(hours=3))

# ─────────────────────────────────────────────────────────────────────────────
# FOOTBALL (unchanged from V4.2)
# ─────────────────────────────────────────────────────────────────────────────

DRAW_PROPENSITY = 0.10

LEAGUE_FILTERS = {
    "pl":     {"competition_id": "PL", "name": "Premier League"},
    "laliga": {"competition_id": "PD", "name": "La Liga"},
    "cl":     {"competition_id": "CL", "name": "Champions League"},
    "wc":     {"competition_id": "WC", "name": "World Cup"},
}

LEAGUE_MAP = {
    "PL": "Premier League",
    "PD": "La Liga",
    "CL": "Champions League",
    "WC": "World Cup 2026",
}

# ─────────────────────────────────────────────────────────────────────────────
# NBA — 30 teams, all nba_api integer team IDs
# primary  = main colour (prob bars, badges, chart lines)
# secondary = accent colour
# ─────────────────────────────────────────────────────────────────────────────

NBA_TEAMS: dict = {
    # ── East · Atlantic ───────────────────────────────────────────────────────
    1610612738: {"name":"Celtics",      "city":"Boston",        "tricode":"BOS", "division":"Atlantic",  "conference":"East", "primary":"#007A33", "secondary":"#BA9653"},
    1610612751: {"name":"Nets",         "city":"Brooklyn",      "tricode":"BKN", "division":"Atlantic",  "conference":"East", "primary":"#000000", "secondary":"#FFFFFF"},
    1610612752: {"name":"Knicks",       "city":"New York",      "tricode":"NYK", "division":"Atlantic",  "conference":"East", "primary":"#006BB6", "secondary":"#F58426"},
    1610612755: {"name":"76ers",        "city":"Philadelphia",  "tricode":"PHI", "division":"Atlantic",  "conference":"East", "primary":"#006BB6", "secondary":"#ED174C"},
    1610612761: {"name":"Raptors",      "city":"Toronto",       "tricode":"TOR", "division":"Atlantic",  "conference":"East", "primary":"#CE1141", "secondary":"#000000"},
    # ── East · Central ────────────────────────────────────────────────────────
    1610612741: {"name":"Bulls",        "city":"Chicago",       "tricode":"CHI", "division":"Central",   "conference":"East", "primary":"#CE1141", "secondary":"#000000"},
    1610612739: {"name":"Cavaliers",    "city":"Cleveland",     "tricode":"CLE", "division":"Central",   "conference":"East", "primary":"#860038", "secondary":"#FDBB30"},
    1610612765: {"name":"Pistons",      "city":"Detroit",       "tricode":"DET", "division":"Central",   "conference":"East", "primary":"#C8102E", "secondary":"#006BB6"},
    1610612754: {"name":"Pacers",       "city":"Indiana",       "tricode":"IND", "division":"Central",   "conference":"East", "primary":"#002D62", "secondary":"#FDBB30"},
    1610612749: {"name":"Bucks",        "city":"Milwaukee",     "tricode":"MIL", "division":"Central",   "conference":"East", "primary":"#00471B", "secondary":"#EEE1C6"},
    # ── East · Southeast ──────────────────────────────────────────────────────
    1610612737: {"name":"Hawks",        "city":"Atlanta",       "tricode":"ATL", "division":"Southeast", "conference":"East", "primary":"#C1272D", "secondary":"#1D1160"},
    1610612766: {"name":"Hornets",      "city":"Charlotte",     "tricode":"CHA", "division":"Southeast", "conference":"East", "primary":"#1D1160", "secondary":"#00788C"},
    1610612748: {"name":"Heat",         "city":"Miami",         "tricode":"MIA", "division":"Southeast", "conference":"East", "primary":"#98002E", "secondary":"#F9A01B"},
    1610612753: {"name":"Magic",        "city":"Orlando",       "tricode":"ORL", "division":"Southeast", "conference":"East", "primary":"#0077C0", "secondary":"#C4CED4"},
    1610612764: {"name":"Wizards",      "city":"Washington",    "tricode":"WAS", "division":"Southeast", "conference":"East", "primary":"#002B5C", "secondary":"#E31837"},
    # ── West · Northwest ──────────────────────────────────────────────────────
    1610612743: {"name":"Nuggets",      "city":"Denver",        "tricode":"DEN", "division":"Northwest", "conference":"West", "primary":"#0E2240", "secondary":"#FEC524"},
    1610612750: {"name":"Timberwolves", "city":"Minnesota",     "tricode":"MIN", "division":"Northwest", "conference":"West", "primary":"#0C2340", "secondary":"#236192"},
    1610612760: {"name":"Thunder",      "city":"Oklahoma City", "tricode":"OKC", "division":"Northwest", "conference":"West", "primary":"#007AC1", "secondary":"#EF3B24"},
    1610612757: {"name":"Trail Blazers","city":"Portland",      "tricode":"POR", "division":"Northwest", "conference":"West", "primary":"#E03A3E", "secondary":"#000000"},
    1610612762: {"name":"Jazz",         "city":"Utah",          "tricode":"UTA", "division":"Northwest", "conference":"West", "primary":"#002B5C", "secondary":"#00471B"},
    # ── West · Pacific ────────────────────────────────────────────────────────
    1610612744: {"name":"Warriors",     "city":"Golden State",  "tricode":"GSW", "division":"Pacific",   "conference":"West", "primary":"#1D428A", "secondary":"#FFC72C"},
    1610612746: {"name":"Clippers",     "city":"LA",            "tricode":"LAC", "division":"Pacific",   "conference":"West", "primary":"#C8102E", "secondary":"#1D428A"},
    1610612747: {"name":"Lakers",       "city":"Los Angeles",   "tricode":"LAL", "division":"Pacific",   "conference":"West", "primary":"#552583", "secondary":"#FDB927"},
    1610612756: {"name":"Suns",         "city":"Phoenix",       "tricode":"PHX", "division":"Pacific",   "conference":"West", "primary":"#1D1160", "secondary":"#E56020"},
    1610612758: {"name":"Kings",        "city":"Sacramento",    "tricode":"SAC", "division":"Pacific",   "conference":"West", "primary":"#5A2D81", "secondary":"#63727A"},
    # ── West · Southwest ──────────────────────────────────────────────────────
    1610612742: {"name":"Mavericks",    "city":"Dallas",        "tricode":"DAL", "division":"Southwest", "conference":"West", "primary":"#00538C", "secondary":"#002F5F"},
    1610612745: {"name":"Rockets",      "city":"Houston",       "tricode":"HOU", "division":"Southwest", "conference":"West", "primary":"#CE1141", "secondary":"#000000"},
    1610612763: {"name":"Grizzlies",    "city":"Memphis",       "tricode":"MEM", "division":"Southwest", "conference":"West", "primary":"#5D76A9", "secondary":"#12173F"},
    1610612740: {"name":"Pelicans",     "city":"New Orleans",   "tricode":"NOP", "division":"Southwest", "conference":"West", "primary":"#0C2340", "secondary":"#C8102E"},
    1610612759: {"name":"Spurs",        "city":"San Antonio",   "tricode":"SAS", "division":"Southwest", "conference":"West", "primary":"#C4CED4", "secondary":"#000000"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def nba_team(team_id) -> dict:
    return NBA_TEAMS.get(int(team_id), {
        "name":"Unknown","city":"","tricode":"UNK","division":"","conference":"",
        "primary":"#6b7280","secondary":"#374151",
    })

def nba_primary_colour(team_id) -> str:
    return nba_team(team_id).get("primary", "#6b7280")

def nba_tricode(team_id) -> str:
    return nba_team(team_id).get("tricode", "UNK")

# ── Conference / division structure ───────────────────────────────────────────

NBA_CONFERENCES: dict = {
    "East": {
        "Atlantic":  [1610612738,1610612751,1610612752,1610612755,1610612761],
        "Central":   [1610612741,1610612739,1610612765,1610612754,1610612749],
        "Southeast": [1610612737,1610612766,1610612748,1610612753,1610612764],
    },
    "West": {
        "Northwest": [1610612743,1610612750,1610612760,1610612757,1610612762],
        "Pacific":   [1610612744,1610612746,1610612747,1610612756,1610612758],
        "Southwest": [1610612742,1610612745,1610612763,1610612740,1610612759],
    },
}

NBA_TEAM_IDS: list = [
    tid
    for conf in NBA_CONFERENCES.values()
    for div_ids in conf.values()
    for tid in div_ids
]
