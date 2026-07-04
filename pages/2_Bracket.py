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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
import common as c

st.set_page_config(page_title="Worldcup Bracket Odds", page_icon="🏆", layout="wide")
st.title("🏆 Worldcup Bracket — Odds by Stage")


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

    def flag_of(code):
        for t in teams:
            if t.get("fifa_code") == code:
                return t.get("flag", "")
        return ""

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

    # ── The visual bracket, native Streamlit widgets so clicks actually work ──
    def build_rounds(matches_subset):
        rounds = [matches_subset]
        while len(rounds[-1]) > 1:
            prev = rounds[-1]
            rounds.append([(prev[i], prev[i + 1]) for i in range(0, len(prev), 2)])
        return rounds

    def render_team_row(code, prob, side):
        """One team's row: flag, clickable name button, win% for this match."""
        col_flag, col_name, col_pct = st.columns([1, 4, 1])
        with col_flag:
            flag = flag_of(code)
            if flag:
                st.image(flag, width=28)
        with col_name:
            clicked = st.button(name_of(code), key=f"team_{code}_{side}", use_container_width=True)
        with col_pct:
            st.markdown(
                f"<div style='padding-top:8px;color:#8b949e;font-size:13px'>{prob:.0%}</div>",
                unsafe_allow_html=True,
            )
        if clicked:
            st.query_params.clear()
            st.query_params["team"] = code
            st.switch_page("pages/1_History.py")

    # H = approximate pixel height of one match box (button+flag rows+padding).
    # This is an estimate -- Streamlit's native widgets don't report their exact
    # rendered height, so this may need a small manual tweak once you see it live.
    H = 132

    def render_round_column(rnd, stage_label, is_first_round, side, round_index):
        st.markdown(f"<div style='text-align:center;font-weight:600;margin-bottom:8px'>{stage_label}</div>",
                   unsafe_allow_html=True)

        # Classic bracket-alignment trick: each round's boxes sit centered
        # between their two feeder boxes from the round before. The gap
        # between entries doubles every round, and the top gets a half-size
        # push down so the very first box centers correctly too.
        pitch = H * (2 ** round_index)
        top_offset = max(0, (pitch - H) / 2)
        gap_between = max(0, pitch - H)

        if top_offset > 0:
            st.markdown(f"<div style='height:{top_offset:.0f}px'></div>", unsafe_allow_html=True)

        if is_first_round:
            for idx, (h, a) in enumerate(rnd):
                p_h = c.resolve_advance_prob(h, a, model, scaler, T)
                with st.container(border=True):
                    render_team_row(h, p_h, f"{side}_{h}")
                    render_team_row(a, 1 - p_h, f"{side}_{a}")
                if idx < len(rnd) - 1 and gap_between > 0:
                    st.markdown(f"<div style='height:{gap_between:.0f}px'></div>", unsafe_allow_html=True)
        else:
            for idx in range(len(rnd)):
                st.markdown(
                    f"<div style='border:1px dashed #30363d;border-radius:8px;padding:16px 12px;"
                    f"min-height:{H-32}px;display:flex;align-items:center;justify-content:center;"
                    f"color:#8b949e;text-align:center;font-size:13px'>TBD</div>",
                    unsafe_allow_html=True,
                )
                if idx < len(rnd) - 1 and gap_between > 0:
                    st.markdown(f"<div style='height:{gap_between:.0f}px'></div>", unsafe_allow_html=True)

    n = len(matches)
    half = n // 2
    left_rounds = build_rounds(matches[:half])
    right_rounds = build_rounds(matches[half:])
    n_side_rounds = len(left_rounds)

    stage_pool = ["Round of 16", "Quarterfinal", "Semifinal", "Final"]
    stage_names = stage_pool[-(n_side_rounds + 1):]

    st.subheader("Bracket")
    model, scaler, T = c.load_model()

    total_cols = n_side_rounds * 2 + 1
    columns = st.columns(total_cols)

    for i in range(n_side_rounds):
        with columns[i]:
            render_round_column(left_rounds[i], stage_names[i], i == 0, side="L", round_index=i)

    with columns[n_side_rounds]:
        # Final column, vertically centered the same way, with a trophy.
        # Using an emoji rather than a real photo of the actual trophy --
        # that's a specific, trademarked physical object, and a generic
        # emoji sidesteps any of that cleanly while still looking festive.
        # The Final only needs to line up with ONE Semifinal box per side
        # (there's just one per side at this point), so it uses the SAME
        # pitch as that last per-side round -- not one level deeper.
        final_pitch = H * (2 ** (n_side_rounds - 1))
        final_offset = max(0, (final_pitch - H) / 2)
        st.markdown(f"<div style='text-align:center;font-weight:600;margin-bottom:8px'>{stage_names[-1]}</div>",
                   unsafe_allow_html=True)
        if final_offset > 0:
            st.markdown(f"<div style='height:{final_offset:.0f}px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='border:1px dashed #30363d;border-radius:8px;padding:16px 12px;"
            f"min-height:{H-32}px;display:flex;flex-direction:column;align-items:center;"
            f"justify-content:center;gap:6px;color:#8b949e;text-align:center;font-size:13px'>"
            f"<span style='font-size:28px'>🏆</span>TBD</div>",
            unsafe_allow_html=True,
        )

    for i in range(n_side_rounds):
        mirrored_i = n_side_rounds - 1 - i  # right side reads inward-to-outward
        with columns[n_side_rounds + 1 + i]:
            render_round_column(right_rounds[mirrored_i], stage_names[mirrored_i], mirrored_i == 0,
                               side="R", round_index=mirrored_i)

    st.caption("Click a team name to see their most recent match on the History page.")

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
    if st.button("📌 Save this snapshot to the permanent record"):
        log_path = Path(__file__).parent.parent / "results" / "bracket_odds_history.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        log_rows = [
            {"timestamp": stamp, "stage_detected": stage_now, "team": team,
             "predicted_stage": stage, "probability": round(prob, 4)}
            for team, stages in odds.items() for stage, prob in stages.items()
        ]
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        st.success(
            f"Saved {len(log_rows)} odds to results/bracket_odds_history.csv at {stamp[:19]} UTC. "
            f"Once the tournament ends, compare this file against what actually happened."
        )
    st.caption(
        "This is a manual save, on purpose -- Streamlit reruns this page on every click and "
        "slider move, so auto-saving every time would flood the file with near-duplicate rows. "
        "Click the button once per day (or whenever you want a checkpoint) instead."
    )

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
