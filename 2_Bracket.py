"""
Bracket page — Monte Carlo tournament odds.

Auto-detects the current stage of the knockout bracket from the live feed
(finds every not-yet-played knockout match that already has both teams
confirmed) and simulates the rest of the tournament thousands of times to
answer: "what's each team's odds of reaching each remaining stage, and of
winning it all?"

One documented assumption: adjacent confirmed fixtures pair up in the
standard tournament way (fixture 1 & 2's winners meet in the next round,
3 & 4's winners meet in the other slot, etc.) rather than us hand-typing
the entire bracket from a screenshot -- a much safer bet than transcribing
30+ match pairings by eye.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
import common as c

st.set_page_config(page_title="Tournament Bracket Odds", page_icon="🏆", layout="wide")
st.title("🏆 Tournament Bracket — Odds by Stage")


def main():
    try:
        games = c.fetch_wc26("games")
        teams = c.fetch_wc26("teams")
    except Exception as e:
        st.error(f"Couldn't reach worldcup26.ir: {e}")
        st.stop()

    team_lookup = c.build_team_lookup(teams)

    def code_of(team_id):
        return team_lookup.get(team_id, {}).get("fifa_code", "UNK")

    def name_of(code):
        for t in teams:
            if t.get("fifa_code") == code:
                return t.get("name_en", code)
        return code

    # ── Find confirmed, not-yet-played knockout fixtures ───────────────────
    knockout_games = [g for g in games if g.get("type") != "group"]
    upcoming = [
        g for g in knockout_games
        if str(g.get("finished", "")).upper() != "TRUE"
        and g.get("home_team_id") in team_lookup
        and g.get("away_team_id") in team_lookup
    ]

    if not upcoming:
        st.info(
            "No confirmed upcoming knockout fixtures found in the feed right now. "
            "This page will fill in automatically once the next round's matchups are set."
        )
        st.stop()

    # Sort for a stable, repeatable pairing order (by kickoff date)
    upcoming = sorted(upcoming, key=lambda g: c.parse_local_date(g.get("local_date")) or "")
    matches = [(code_of(g["home_team_id"]), code_of(g["away_team_id"])) for g in upcoming]

    n_matches = len(matches)
    stage_now = {1: "Final", 2: "Semifinal", 4: "Quarterfinal", 8: "Round of 16"}.get(n_matches, f"{n_matches} matches")
    st.caption(
        f"Found **{n_matches}** confirmed upcoming knockout match(es) — treating this as the "
        f"**{stage_now}** stage. Adjacent matches are assumed to pair up in the next round "
        f"(standard bracket convention)."
    )

    with st.expander("Which matches were found"):
        for (h, a) in matches:
            st.write(f"{name_of(h)} ({h}) vs {name_of(a)} ({a})")

    if n_matches not in (1, 2, 4, 8, 16):
        st.warning(
            f"{n_matches} confirmed fixtures isn't a clean power of two -- the pairing "
            f"below may not exactly match the real bracket. Double-check against the "
            f"official bracket if this looks off."
        )

    fixture_tree = c.build_fixture_tree_from_matches(matches)

    n_trials = st.select_slider("Monte Carlo trials", [5_000, 20_000, 50_000, 100_000], value=20_000)

    with st.spinner(f"Running {n_trials:,} simulated tournaments..."):
        model, scaler, T = c.load_model()
        odds = c.simulate_tournament(fixture_tree, model, scaler, T, n_trials=n_trials)

    # ── Build a clean results table ────────────────────────────────────────
    all_stages = ["Reaches Round of 16", "Reaches Quarterfinal",
                  "Reaches Semifinal", "Reaches Final", "Champion"]
    present_stages = [s for s in all_stages if any(s in v for v in odds.values())]

    rows = []
    for team, stages in odds.items():
        row = {"Team": f"{name_of(team)} ({team})"}
        for s in present_stages:
            row[s] = stages.get(s, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Team")
    df = df.sort_values("Champion", ascending=False)

    st.subheader("Odds by stage")
    st.dataframe(
        df.style.format("{:.1%}"),
        use_container_width=True,
    )
    st.caption(
        f"Based on {n_trials:,} simulated tournaments using football_v2.pth. "
        f"Draw probability in each knockout match is split 50/50 between the two teams "
        f"(no draws survive to a next round -- extra time and penalties decide it)."
    )


if __name__ == "__main__":
    main()
