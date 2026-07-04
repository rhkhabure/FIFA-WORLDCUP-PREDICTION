"""
common.py — shared code for every page of the dashboard.

Why this file exists: app.py and the History page both need the same model
class, the same FIFA strength table, and the same scorer parser. Keeping one
copy here means they can never quietly drift out of sync with each other —
which is exactly the kind of bug that cost us three rounds of fixes earlier
(the model class in app.py not matching what was actually saved to disk).
"""

import re
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent  # this file lives at repo root, same as app.py

MODELS_CANDIDATES = [
    ROOT / "models",
    ROOT / "notebooks" / "models",  # confirmed actual location in this repo
]
MODELS = next((p for p in MODELS_CANDIDATES if (p / "football_v2.pth").exists()), MODELS_CANDIDATES[0])

DATA_DIR = ROOT / "data"
STATS_CACHE_PATH = DATA_DIR / "match_stats_cache.json"

WC26_BASE = "https://worldcup26.ir"

FEATURE_COLS = [
    "goal_diff", "minute_norm", "is_second_half", "home_rank_norm",
    "away_rank_norm", "rank_diff", "is_knockout", "lead_changes_norm",
    "is_neutral_venue", "score_state", "strength_x_time",
]

# ═══════════════════════════════════════════════════════════════════════════
# TEAM STRENGTH
# ═══════════════════════════════════════════════════════════════════════════

FIFA_RANK = {
    'BRA':1,'ARG':2,'FRA':3,'ENG':4,'BEL':5,'NED':6,'POR':7,'ESP':8,'ITA':9,'GER':10,
    'CRO':11,'URU':12,'COL':13,'MEX':14,'USA':15,'SUI':16,'DEN':17,'SEN':18,'WAL':19,'IRN':20,
    'SRB':21,'MAR':22,'PER':23,'JPN':24,'SWE':25,'POL':26,'CHI':27,'KOR':28,'TUN':29,'CRC':30,
    'AUS':31,'NGA':32,'EGY':33,'SCO':34,'NOR':35,'TUR':36,'GHA':37,'ECU':38,'CIV':39,'QAT':40,
    'CAN':41,'CMR':42,'KSA':43,'IRQ':44,'RSA':45,'DZA':46,'CPV':47,'JOR':48,'UZB':49,'NZL':50,
    'CZE':51,'AUT':52,'BIH':53,'COD':54,'PAN':55,'SVK':56,
}
DEFAULT_RANK, MAX_RANK = 50, 100
def rank_to_norm(rank): return max(0.0, (MAX_RANK - rank) / (MAX_RANK - 1))
def get_strength(code): return rank_to_norm(FIFA_RANK.get(code, DEFAULT_RANK))

# ═══════════════════════════════════════════════════════════════════════════
# SCORER PARSING (proven against every format we've hit so far)
# ═══════════════════════════════════════════════════════════════════════════

_GOAL_RE = re.compile(r"(\d+)(?:'(?:\+(\d+)')?|\+(\d+)')")
_SEGMENT_RE = re.compile(r'["“”]([^"“”]+)["“”]')


def parse_scorers(raw):
    """Minutes only — used for building model features."""
    if raw is None or str(raw).strip().lower() == "null":
        return []
    minutes = []
    for base, extra_a, extra_b in _GOAL_RE.findall(str(raw)):
        extra = extra_a or extra_b or 0
        minutes.append(min(int(base) + int(extra), 90))
    return sorted(minutes)


def parse_scorers_named(raw):
    """Name + minute pairs — used for display on the History page."""
    if raw is None or str(raw).strip().lower() == "null":
        return []
    results = []
    for seg in _SEGMENT_RE.findall(str(raw)):
        m = _GOAL_RE.search(seg)
        if not m:
            continue
        base, extra_a, extra_b = m.groups()
        minute = min(int(base) + int(extra_a or extra_b or 0), 90)
        name = seg[:m.start()].strip()
        results.append((name, minute))
    return sorted(results, key=lambda x: x[1])


def safe_int(val, default=0):
    """The feed sends literal text 'null' (not real JSON null) for scores
    on games that haven't started. This catches that specific case."""
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("", "null", "none"):
        return default
    try:
        return int(s)
    except ValueError:
        return default


def parse_local_date(s):
    """local_date format seen in the feed: 'MM/DD/YYYY HH:MM'"""
    try:
        return datetime.strptime(s, "%m/%d/%Y %H:%M")
    except Exception:
        return None


def parse_time_elapsed(raw, status_finished):
    """Turn the feed's time_elapsed field into a minute number (0-90)."""
    if status_finished:
        return 90
    s = str(raw or "").strip().upper()
    if s in ("", "NOT STARTED", "PRE", "SCHEDULED"):
        return 0
    if s in ("HT", "FINISHED"):
        return 45 if s == "HT" else 90
    m = re.match(r"^(\d+)(?:\+(\d+))?$", s)
    if m:
        return min(int(m.group(1)) + int(m.group(2) or 0), 90)
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# LIVE DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=85)
def fetch_wc26(endpoint):
    r = requests.get(f"{WC26_BASE}/get/{endpoint}", timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        for key in ("games", "teams", "groups"):
            if key in data:
                return data[key]
    return data


def build_team_lookup(teams):
    return {t["id"]: t for t in teams}


# ═══════════════════════════════════════════════════════════════════════════
# MODEL — must match football_v2.pth's saved arch dict EXACTLY:
# {'n_features': 11, 'n_classes': 3, 'h1': 40, 'h2': 20}
# ═══════════════════════════════════════════════════════════════════════════

class FootballWinProbNet(nn.Module):
    def __init__(self, n_features=11, n_classes=3, h1=40, h2=20, dropout=0.30):
        super().__init__()
        self.fc1 = nn.Linear(n_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.head = nn.Linear(h2, n_classes)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.ReLU()

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return self.head(x)


@st.cache_resource
def load_model():
    model_path = MODELS / "football_v2.pth"
    if not model_path.exists():
        st.error(
            f"Can't find football_v2.pth. Checked: "
            f"{[str(p / 'football_v2.pth') for p in MODELS_CANDIDATES]}."
        )
        st.stop()
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = FootballWinProbNet(**ckpt["arch"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    T = ckpt["temperature"]
    with open(MODELS / "scaler_v2.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler, T


def predict(model, scaler, T, feat_row):
    """feat_row: list of 11 floats in FEATURE_COLS order. Returns (p_away, p_draw, p_home)."""
    x = np.array([feat_row], dtype=np.float32)
    x_scaled = scaler.transform(x).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(x_scaled))
        probs = F.softmax(logits / T, dim=1).numpy()[0]
    return float(probs[0]), float(probs[1]), float(probs[2])


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_row(home_code, away_code, minute, hs, as_, lead_changes, goals_so_far, is_knockout):
    """Build one 11-feature row for a single point in time in a match."""
    goal_diff = int(np.clip(hs - as_, -5, 5))
    minute_norm = min(minute / 90.0, 1.0)
    is_second_half = 1 if minute > 45 else 0
    h_str, a_str = get_strength(home_code), get_strength(away_code)
    rank_diff = h_str - a_str
    lead_changes_norm = lead_changes / max(1, goals_so_far)
    score_state = 0 if goal_diff < 0 else (2 if goal_diff > 0 else 1)
    strength_x_time = rank_diff * (1.0 - minute_norm)
    return [goal_diff, minute_norm, is_second_half, h_str, a_str,
            rank_diff, int(is_knockout), lead_changes_norm, 1, score_state, strength_x_time]


def resolve_advance_prob(home_code, away_code, model, scaler, T):
    """
    Pre-game probability that home_code advances past away_code in a
    KNOCKOUT match. A knockout game can't end in a draw at full-time -- it
    goes to extra time then penalties -- so we split the model's draw
    probability evenly between both teams, since a shootout is close to a
    coin flip regardless of which side was rated better in 90 minutes.
    """
    feat_row = build_feature_row(home_code, away_code, minute=0, hs=0, as_=0,
                                 lead_changes=0, goals_so_far=0, is_knockout=1)
    p_away, p_draw, p_home = predict(model, scaler, T, feat_row)
    return p_home + p_draw / 2  # p_away's share is just 1 - this


def simulate_tournament(fixture_tree, model, scaler, T, n_trials=20000):
    """
    Monte Carlo simulate an entire knockout bracket many times and tally,
    per team, the fraction of trials where they reached each stage.

    fixture_tree: a team code (string, already confirmed) OR a (left, right)
    pair where left/right are themselves fixture_trees -- so a whole bracket
    is one big nested tuple, e.g. (("PAR","FRA"), ("CAN","MAR")) for a
    semifinal-and-final bracket built from two confirmed quarterfinal
    winners... one level down.

    Returns {team_code: {stage_label: probability, ...}, ...}
    """
    import random

    def height(node):
        if isinstance(node, str):
            return 0
        return 1 + max(height(node[0]), height(node[1]))

    total_height = height(fixture_tree)
    GENERIC_LABELS = ["Reaches Round of 16", "Reaches Quarterfinal",
                      "Reaches Semifinal", "Reaches Final", "Champion"]
    labels = GENERIC_LABELS[-total_height:] if total_height > 0 else []

    pair_cache = {}
    def get_prob(a, b):
        key = tuple(sorted([a, b]))
        if key not in pair_cache:
            pair_cache[key] = resolve_advance_prob(key[0], key[1], model, scaler, T)
        p_first = pair_cache[key]
        return p_first if key[0] == a else 1 - p_first

    tallies = {}

    def resolve(node):
        if isinstance(node, str):
            return node, 0
        left, right = node
        left_team, lh = resolve(left)
        right_team, rh = resolve(right)
        h = max(lh, rh) + 1
        p_left = get_prob(left_team, right_team)
        winner = left_team if random.random() < p_left else right_team
        label = labels[h - 1]
        tallies.setdefault(winner, {}).setdefault(label, 0)
        tallies[winner][label] += 1
        return winner, h

    for _ in range(n_trials):
        resolve(fixture_tree)

    return {team: {lbl: cnt / n_trials for lbl, cnt in counts.items()}
            for team, counts in tallies.items()}


def build_fixture_tree_from_matches(matches):
    """
    matches: list of (home_code, away_code) tuples, in bracket order
    (adjacent pairs meet in the next round -- match 0&1 feed one semifinal
    slot, match 2&3 feed the other, and so on). Returns a fixture_tree ready
    for simulate_tournament().
    """
    level = [(m[0], m[1]) for m in matches]
    while len(level) > 1:
        level = [(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0] if level else None


def build_live_features(game, team_lookup):
    """Build the CURRENT single feature row for a live/pre-game match."""
    home_id, away_id = game["home_team_id"], game["away_team_id"]
    home_code = team_lookup.get(home_id, {}).get("fifa_code", "UNK")
    away_code = team_lookup.get(away_id, {}).get("fifa_code", "UNK")

    hs = safe_int(game.get("home_score"))
    as_ = safe_int(game.get("away_score"))
    is_finished = str(game.get("finished", "")).upper() == "TRUE"
    minute = parse_time_elapsed(game.get("time_elapsed"), is_finished)
    is_knockout = 0 if game.get("type") == "group" else 1

    home_goals = parse_scorers(game.get("home_scorers"))
    away_goals = parse_scorers(game.get("away_scorers"))
    events = sorted([(m, "home") for m in home_goals] + [(m, "away") for m in away_goals])

    lead_changes, prev_leader, goals_so_far = 0, 0, 0
    for m, side in events:
        if m > minute:
            break
        goals_so_far += 1
        h_now = sum(1 for mm, s in events if mm <= m and s == "home")
        a_now = sum(1 for mm, s in events if mm <= m and s == "away")
        leader = (h_now > a_now) - (h_now < a_now)
        if leader != prev_leader and leader != 0:
            lead_changes += 1
        prev_leader = leader

    feat_row = build_feature_row(home_code, away_code, minute, hs, as_,
                                 lead_changes, goals_so_far, is_knockout)
    return feat_row, home_code, away_code, hs, as_, minute, is_finished


def build_match_timeline(game, team_lookup, model, scaler, T):
    """
    For a FINISHED match, reconstruct the win-probability curve over time:
    a snapshot every 5 minutes plus right after every goal. Returns a list
    of dicts: {minute, p_away, p_draw, p_home, home_score, away_score}.
    """
    home_id, away_id = game["home_team_id"], game["away_team_id"]
    home_code = team_lookup.get(home_id, {}).get("fifa_code", "UNK")
    away_code = team_lookup.get(away_id, {}).get("fifa_code", "UNK")
    is_knockout = 0 if game.get("type") == "group" else 1

    home_goals = parse_scorers(game.get("home_scorers"))
    away_goals = parse_scorers(game.get("away_scorers"))
    events = sorted([(m, "home") for m in home_goals] + [(m, "away") for m in away_goals])

    checkpoints = sorted(set([0] + list(range(5, 91, 5)) + [m for m, _ in events] + [90]))

    rows = []
    lead_changes, prev_leader = 0, 0
    for minute in checkpoints:
        hs = sum(1 for m, s in events if m <= minute and s == "home")
        as_ = sum(1 for m, s in events if m <= minute and s == "away")
        goals_so_far = hs + as_
        leader = (hs > as_) - (hs < as_)
        if leader != prev_leader and leader != 0:
            lead_changes += 1
        prev_leader = leader

        feat_row = build_feature_row(home_code, away_code, minute, hs, as_,
                                     lead_changes, goals_so_far, is_knockout)
        p_away, p_draw, p_home = predict(model, scaler, T, feat_row)
        rows.append({"minute": minute, "p_away": p_away, "p_draw": p_draw,
                    "p_home": p_home, "home_score": hs, "away_score": as_})
    return rows
