"""
History page — replay the win-probability curve for any finished match,
plus goal scorers and (if you've run fetch_match_stats.py) final stats.

This page is built to be deep-linked from the future Bracket page: clicking
a finished match there will set ?match_id=... in the URL, which this page
reads on load to jump straight to that game.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
import common as c

st.set_page_config(page_title="Match History", page_icon="📼", layout="wide")
st.title("📼 Match History")


@st.cache_data(ttl=300)
def load_stats_cache():
    if c.STATS_CACHE_PATH.exists():
        return json.loads(c.STATS_CACHE_PATH.read_text())
    return {}


def label_for(g, team_lookup):
    h = team_lookup.get(g["home_team_id"], {}).get("name_en", "?")
    a = team_lookup.get(g["away_team_id"], {}).get("name_en", "?")
    hs, as_ = c.safe_int(g.get("home_score")), c.safe_int(g.get("away_score"))
    return f"{h} {hs} — {as_} {a}"


def main():
    try:
        games = c.fetch_wc26("games")
        teams = c.fetch_wc26("teams")
    except Exception as e:
        st.error(f"Couldn't reach worldcup26.ir: {e}")
        st.stop()

    team_lookup = c.build_team_lookup(teams)
    finished_games = [g for g in games if str(g.get("finished", "")).upper() == "TRUE"]

    if not finished_games:
        st.info("No finished matches yet.")
        st.stop()

    # Newest matches first — easier to find recent games by default
    def sort_key(g):
        return c.parse_local_date(g.get("local_date")) or datetime.min
    finished_games = sorted(finished_games, key=sort_key, reverse=True)

    # ── Search box — filters the list by team name as you type ────────────
    search = st.text_input("Search by team name", placeholder="e.g. Brazil, Germany, Japan...")
    if search:
        matches = [g for g in finished_games if search.lower() in label_for(g, team_lookup).lower()]
    else:
        matches = finished_games

    if not matches:
        st.warning(f"No finished matches found matching '{search}'.")
        st.stop()

    # ── Deep-link support: ?match_id=<id> jumps straight to that game ──────
    query_match_id = st.query_params.get("match_id")
    game_ids = [g["id"] for g in matches]
    default_idx = game_ids.index(query_match_id) if query_match_id in game_ids else 0

    labels = [label_for(g, team_lookup) for g in matches]
    choice_idx = st.selectbox(
        f"Pick a finished match ({len(matches)} shown)", range(len(labels)),
        index=default_idx, format_func=lambda i: labels[i],
    )
    game = matches[choice_idx]
    st.query_params["match_id"] = game["id"]  # keep the URL in sync for sharing/linking

    home_code = team_lookup.get(game["home_team_id"], {}).get("fifa_code", "UNK")
    away_code = team_lookup.get(game["away_team_id"], {}).get("fifa_code", "UNK")
    home_flag = team_lookup.get(game["home_team_id"], {}).get("flag", "")
    away_flag = team_lookup.get(game["away_team_id"], {}).get("flag", "")
    hs, as_ = c.safe_int(game.get("home_score")), c.safe_int(game.get("away_score"))

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.image(home_flag, width=48)
        st.metric(home_code, hs)
    with col2:
        st.markdown("<div style='text-align:center; padding-top:32px; color:#8b949e'>FINAL</div>", unsafe_allow_html=True)
    with col3:
        st.image(away_flag, width=48)
        st.metric(away_code, as_)

    # ── Win probability over time ────────────────────────────────────────
    st.subheader("Win probability over time")
    model, scaler, T = c.load_model()
    timeline = c.build_match_timeline(game, team_lookup, model, scaler, T)
    df = pd.DataFrame(timeline).set_index("minute")[["p_home", "p_draw", "p_away"]]
    df.columns = [home_code, "Draw", away_code]
    st.line_chart(df)
    st.caption(
        "Football swings hard on a single goal — expect sharp steps here, "
        "not the smoother drift you'd see in a basketball chart."
    )

    # ── Goal scorers ──────────────────────────────────────────────────────
    st.subheader("Goal scorers")
    home_scorers = c.parse_scorers_named(game.get("home_scorers"))
    away_scorers = c.parse_scorers_named(game.get("away_scorers"))
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"**{home_code}**")
        if home_scorers:
            for name, minute in home_scorers:
                st.write(f"{minute}' — {name}")
        else:
            st.caption("No goals")
    with sc2:
        st.markdown(f"**{away_code}**")
        if away_scorers:
            for name, minute in away_scorers:
                st.write(f"{minute}' — {name}")
        else:
            st.caption("No goals")

    # ── Final match stats (from the one-time enrichment script) ──────────
    st.subheader("Match stats")
    stats_cache = load_stats_cache()
    stats = stats_cache.get(str(game["id"]))
    if stats:
        stage_raw = stats.get("stage", "")
        stage_label = "Group stage" if stage_raw == "group" else (
            stage_raw.replace("_", " ").title() if stage_raw else "Unknown stage"
        )
        date_raw = stats.get("date")
        try:
            date_label = datetime.strptime(date_raw, "%m/%d/%Y %H:%M").strftime("%d %b %Y, %H:%M")
        except Exception:
            date_label = date_raw or "Unknown date"

        st.write(f"**Stage:** {stage_label}")
        st.write(f"**Kickoff:** {date_label}")
        with st.expander("Raw data"):
            st.json(stats)
    else:
        st.caption(
            "No cached stats for this match yet. Run `python fetch_match_stats.py` "
            "to fill this in for all finished games."
        )


if __name__ == "__main__":
    main()
