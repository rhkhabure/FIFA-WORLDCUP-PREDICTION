"""
World Cup 2026 Live Win Probability — Dashboard (Part 1: Live view)

Scope for this build: fetch live 2026 matches, run football_v2.pth (the
production model, plain softmax, 11 features), and display the odds via
the flag-push + draw-fog visual. History / bracket pages come in Part 2.

Run with:  streamlit run app.py
Needs:     pip install streamlit torch numpy pandas scikit-learn requests streamlit-autorefresh
"""

import json, pickle, re, time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="World Cup Win Probability", page_icon="⚽", layout="wide")

ROOT      = Path(__file__).parent  # app.py lives at repo root, models/ is right beside it
MODELS    = ROOT / "models"
DATA_RAW  = ROOT / "data" / "raw"
WC26_BASE = "https://worldcup26.ir"

FEATURE_COLS = [
    "goal_diff", "minute_norm", "is_second_half", "home_rank_norm",
    "away_rank_norm", "rank_diff", "is_knockout", "lead_changes_norm",
    "is_neutral_venue", "score_state", "strength_x_time",
]

# Same curated FIFA strength table used throughout the project.
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

_GOAL_RE = re.compile(r"(\d+)(?:'(?:\+(\d+)')?|\+(\d+)')")
def parse_scorers(raw):
    """Same proven parser from Phase 3 — handles both stoppage-time formats."""
    if raw is None or str(raw).strip().lower() == "null":
        return []
    minutes = []
    for base, extra_a, extra_b in _GOAL_RE.findall(str(raw)):
        extra = extra_a or extra_b or 0
        minutes.append(min(int(base) + int(extra), 90))
    return sorted(minutes)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════

class FootballWinProbNet(nn.Module):
    """Matches football_v2.pth exactly: 11 -> 40 -> 20 -> 3, softmax output."""
    def __init__(self, n_features=11, h1=40, h2=20, dropout=0.30):
        super().__init__()
        self.fc1 = nn.Linear(n_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.head = nn.Linear(h2, 3)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.ReLU()

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        return self.head(x)


@st.cache_resource
def load_model():
    ckpt = torch.load(MODELS / "football_v2.pth", map_location="cpu", weights_only=False)
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
# LIVE DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=85)  # slightly under the 90s refresh so we always get fresh data
def fetch_wc26(endpoint):
    r = requests.get(f"{WC26_BASE}/get/{endpoint}", timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        for key in ("games", "teams", "groups"):
            if key in data:
                return data[key]
    return data


def parse_local_date(s):
    """local_date format seen in the feed: 'MM/DD/YYYY HH:MM'"""
    try:
        return datetime.strptime(s, "%m/%d/%Y %H:%M")
    except Exception:
        return None


def parse_time_elapsed(raw, status_finished):
    """
    Turn the feed's time_elapsed field into a minute number (0-90).
    Handles: plain minute strings, stoppage format 'MM+SS', 'HT', 'finished'.
    Falls back to 0 if the format is unrecognised, rather than guessing wildly.
    """
    if status_finished:
        return 90
    s = str(raw or "").strip().upper()
    if s in ("", "NOT STARTED", "PRE", "SCHEDULED"):
        return 0
    if s == "HT":
        return 45
    if s == "FINISHED":
        return 90
    m = re.match(r"^(\d+)(?:\+(\d+))?$", s)
    if m:
        base = int(m.group(1))
        extra = int(m.group(2) or 0)
        return min(base + extra, 90)
    return 0  # unknown format — honest fallback, not a guess


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE CONSTRUCTION FOR ONE LIVE GAME
# ═══════════════════════════════════════════════════════════════════════════

def build_live_features(game, team_lookup):
    home_id, away_id = game["home_team_id"], game["away_team_id"]
    home_code = team_lookup.get(home_id, {}).get("fifa_code", "UNK")
    away_code = team_lookup.get(away_id, {}).get("fifa_code", "UNK")

    hs = int(game.get("home_score") or 0)
    as_ = int(game.get("away_score") or 0)
    is_finished = str(game.get("finished", "")).upper() == "TRUE"
    minute = parse_time_elapsed(game.get("time_elapsed"), is_finished)

    goal_diff = int(np.clip(hs - as_, -5, 5))
    minute_norm = min(minute / 90.0, 1.0)
    is_second_half = 1 if minute > 45 else 0
    h_str, a_str = get_strength(home_code), get_strength(away_code)
    rank_diff = h_str - a_str
    is_knockout = 0 if game.get("type") == "group" else 1

    # Reconstruct lead-changes-so-far from the scorer strings, same causal
    # (goals-so-far, not final total) definition fixed back in Phase 1/3.
    home_goals = parse_scorers(game.get("home_scorers"))
    away_goals = parse_scorers(game.get("away_scorers"))
    events = sorted([(m, "home") for m in home_goals] + [(m, "away") for m in away_goals])
    lead_changes, prev_leader = 0, 0
    goals_so_far = 0
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
    lead_changes_norm = lead_changes / max(1, goals_so_far)

    score_state = 0 if goal_diff < 0 else (2 if goal_diff > 0 else 1)
    strength_x_time = rank_diff * (1.0 - minute_norm)

    feat_row = [goal_diff, minute_norm, is_second_half, h_str, a_str,
                rank_diff, is_knockout, lead_changes_norm, 1, score_state, strength_x_time]

    return feat_row, home_code, away_code, hs, as_, minute, is_finished


# ═══════════════════════════════════════════════════════════════════════════
# FLAG-PUSH + DRAW-FOG VISUAL
# ═══════════════════════════════════════════════════════════════════════════

def flag_push_html(home_flag_url, away_flag_url, home_code, away_code, p_away, p_draw, p_home):
    denom = p_home + p_away
    lean = round((p_home / denom) * 100) if denom > 0 else 50
    fog_px = 20 + (p_draw) * 160
    track_pad = 44
    lean_fill = track_pad + (lean / 100) * (600 - 2 * track_pad)

    return f"""
    <div style="font-family: sans-serif; padding: 8px 0;">
      <div style="position: relative; height: 96px; margin: 0 auto; max-width: 640px;">
        <div style="position:absolute; left:0; right:0; top:42px; height:12px; background:#2a2f3a; border-radius:6px;"></div>
        <div style="position:absolute; top:42px; height:12px; width:{lean_fill:.0f}px; background:#378ADD; border-radius:6px;"></div>
        <div style="position:absolute; top:34px; height:28px; width:{fog_px:.0f}px; left:{lean_fill - fog_px/2:.0f}px;
                    background:#8b949e; opacity:{0.35 + p_draw*0.5:.2f}; border-radius:6px;"></div>
        <img src="{home_flag_url}" style="position:absolute; left:4px; top:0; width:40px; height:28px; object-fit:cover; border-radius:3px;" />
        <img src="{away_flag_url}" style="position:absolute; right:4px; top:0; width:40px; height:28px; object-fit:cover; border-radius:3px;" />
        <div style="position:absolute; left:4px; top:66px; font-size:12px; color:#8b949e;">{home_code}</div>
        <div style="position:absolute; right:4px; top:66px; font-size:12px; color:#8b949e; text-align:right;">{away_code}</div>
      </div>
      <div style="display:flex; justify-content:center; gap:32px; margin-top:12px; font-size:15px;">
        <span style="color:#378ADD; font-weight:600;">{home_code} {p_home:.1%}</span>
        <span style="color:#8b949e;">Draw {p_draw:.1%}</span>
        <span style="color:#E24B4A; font-weight:600;">{away_code} {p_away:.1%}</span>
      </div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.title("⚽ World Cup 2026 — Live Win Probability")

    with st.sidebar:
        st.markdown("**Model**")
        st.caption("football_v2.pth · softmax · 11 features")
        refresh_choice = st.selectbox("Auto-refresh", ["90 s", "60 s", "30 s", "Manual"], index=0)
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()

    if HAS_AUTOREFRESH and refresh_choice != "Manual":
        interval_ms = {"90 s": 90_000, "60 s": 60_000, "30 s": 30_000}[refresh_choice]
        st_autorefresh(interval=interval_ms, key="live_refresh")
    elif not HAS_AUTOREFRESH:
        st.sidebar.warning("Install streamlit-autorefresh for automatic refresh:\npip install streamlit-autorefresh")

    try:
        games = fetch_wc26("games")
        teams = fetch_wc26("teams")
    except Exception as e:
        st.error(f"Couldn't reach worldcup26.ir: {e}")
        st.stop()

    team_lookup = {t["id"]: t for t in teams}

    today = datetime.now().date()
    todays_games = [g for g in games if (parse_local_date(g.get("local_date")) or datetime.min).date() == today]

    if not todays_games:
        st.info("No games scheduled today according to the feed.")
        st.stop()

    def label_for(g):
        h = team_lookup.get(g["home_team_id"], {}).get("name_en", "?")
        a = team_lookup.get(g["away_team_id"], {}).get("name_en", "?")
        status = "FINISHED" if str(g.get("finished","")).upper()=="TRUE" else str(g.get("time_elapsed","upcoming"))
        return f"{h} vs {a}  ({status})"

    labels = [label_for(g) for g in todays_games]
    choice = st.selectbox("Pick today's game to track", labels)
    game = todays_games[labels.index(choice)]

    model, scaler, T = load_model()
    feat_row, home_code, away_code, hs, as_, minute, is_finished = build_live_features(game, team_lookup)
    p_away, p_draw, p_home = predict(model, scaler, T, feat_row)

    home_flag = team_lookup.get(game["home_team_id"], {}).get("flag", "")
    away_flag = team_lookup.get(game["away_team_id"], {}).get("flag", "")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.metric(home_code, hs)
    with col2:
        status = "FINAL" if is_finished else f"Min {minute}"
        st.markdown(f"<div style='text-align:center; padding-top:8px; color:#8b949e'>{status}</div>", unsafe_allow_html=True)
    with col3:
        st.metric(away_code, as_)

    st.components.v1.html(
        flag_push_html(home_flag, away_flag, home_code, away_code, p_away, p_draw, p_home),
        height=180,
    )

    st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


if __name__ == "__main__":
    main()
