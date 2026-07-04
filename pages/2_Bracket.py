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

Stage 2 additions: hovering the small info icon next to a team shows their
full stage-by-stage odds via a pure-CSS tooltip (no click needed, works
regardless of Streamlit's own click-handling). Clicking the small flag
button next to it shows that team's path-to-final summary below the
bracket -- a real click for a real action, same reliable pattern as the
History link, rather than trying to fit a whole popup into a hover.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
import common as c

st.set_page_config(page_title="Tournament Bracket Odds", page_icon="🏆", layout="wide")
st.title("🏆 Tournament Bracket — Odds by Stage")

# One shared CSS block for every hover tooltip on the page -- pure CSS,
# no JS, no Streamlit interaction needed, so it always works reliably.
TOOLTIP_CSS = """
<style>
.tt-wrap { position: relative; display: inline-block; cursor: help; }
.tt-content {
  visibility: hidden; opacity: 0; transition: opacity 0.15s;
  position: absolute; bottom: 130%; left: 50%; transform: translateX(-50%);
  background: #161b22; border: 1px solid #30363d; border-radius: 6px;
  padding: 8px 10px; white-space: nowrap; z-index: 50; font-size: 12px;
  color: #e6edf3;
}
.tt-wrap:hover .tt-content { visibility: visible; opacity: 1; }
</style>
"""


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

    upcoming = sorted(upcoming, key=lambda g: c.parse_local_date(g.get("local_date")) or "")
    matches = [(code_of(g["home_team_id"]), code_of(g["away_team_id"])) for g in upcoming]

    def build_rounds(matches_subset):
        rounds = [matches_subset]
        while len(rounds[-1]) > 1:
            prev = rounds[-1]
            rounds.append([(prev[i], prev[i + 1]) for i in range(0, len(prev), 2)])
        return rounds

    n = len(matches)
    half = n // 2
    left_rounds = build_rounds(matches[:half])
    right_rounds = build_rounds(matches[half:])
    n_side_rounds = len(left_rounds)

    stage_pool = ["Round of 16", "Quarterfinal", "Semifinal", "Final"]
    stage_names = stage_pool[-(n_side_rounds + 1):]

    n_matches = len(matches)
    stage_now = {1: "Final", 2: "Semifinal", 4: "Quarterfinal", 8: "Round of 16"}.get(n_matches, f"{n_matches} matches")

    # ── Run the Monte Carlo simulation FIRST, so the odds are ready
    #    before we render the bracket (needed for the hover tooltips) ──────
    model, scaler, T = c.load_model()
    fixture_tree = c.build_fixture_tree_from_matches(matches)
    n_trials = st.select_slider("Monte Carlo trials", [5_000, 20_000, 50_000, 100_000], value=20_000)
    with st.spinner(f"Running {n_trials:,} simulated tournaments..."):
        odds = c.simulate_tournament(fixture_tree, model, scaler, T, n_trials=n_trials)

    def tooltip_text(code):
        stages = odds.get(code, {})
        if not stages:
            return "No simulation data yet"
        order = ["Reaches Round of 16", "Reaches Quarterfinal", "Reaches Semifinal", "Reaches Final", "Champion"]
        parts = [f"{s.replace('Reaches ', '')}: {stages[s]:.0%}" for s in order if s in stages]
        return " · ".join(parts)

    # ── The visual bracket ──────────────────────────────────────────────────
    st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)
    st.subheader("Bracket")

    H = 132  # estimated pixel height of one match box -- tune if spacing looks off

    def render_team_row(code, prob, side):
        col_flag, col_name, col_info, col_path = st.columns([1, 6, 0.6, 0.6])
        with col_flag:
            flag = flag_of(code)
            if flag:
                st.image(flag, width=26)
        with col_name:
            clicked = st.button(f"{name_of(code)}  ·  {prob:.0%}",
                                key=f"team_{code}_{side}", use_container_width=True)
        with col_info:
            st.markdown(
                f"<div class='tt-wrap' style='padding-top:8px'>ⓘ"
                f"<div class='tt-content'>{tooltip_text(code)}</div></div>",
                unsafe_allow_html=True,
            )
        with col_path:
            if st.button("🏁", key=f"path_{code}_{side}", help="See path to Final"):
                st.session_state["path_team"] = code
        if clicked:
            st.query_params.clear()
            st.query_params["team"] = code
            st.switch_page("pages/1_History.py")

    def render_round_column(rnd, stage_label, is_first_round, side, round_index):
        st.markdown(f"<div style='text-align:center;font-weight:600;margin-bottom:8px'>{stage_label}</div>",
                   unsafe_allow_html=True)
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

    def col_weight(round_idx):
        # The first round has real content (flag+button+icons) and needs
        # real width. Every later round is just a small dashed placeholder
        # box until those matches are actually played, so it needs far less.
        return 2.4 if round_idx == 0 else 1.0

    left_weights = [col_weight(i) for i in range(n_side_rounds)]
    right_weights = [col_weight(n_side_rounds - 1 - i) for i in range(n_side_rounds)]
    final_weight = 1.3
    columns = st.columns(left_weights + [final_weight] + right_weights)

    for i in range(n_side_rounds):
        with columns[i]:
            render_round_column(left_rounds[i], stage_names[i], i == 0, side="L", round_index=i)

    with columns[n_side_rounds]:
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
        mirrored_i = n_side_rounds - 1 - i
        with columns[n_side_rounds + 1 + i]:
            render_round_column(right_rounds[mirrored_i], stage_names[mirrored_i], mirrored_i == 0,
                               side="R", round_index=mirrored_i)

    st.caption(
        "Click a team name to see their most recent match on History. "
        "Hover the ⓘ for their full stage odds. Click 🏁 to see their path to the Final."
    )

    # ── Stage 3: animated path to the Final ─────────────────────────────────
    if st.session_state.get("path_team"):
        pt = st.session_state["path_team"]
        stages = odds.get(pt, {})

        # Find this team's actual first-round match and per-match odds
        opponent, first_match_prob = None, None
        for (h, a) in matches:
            if h == pt:
                opponent = a
                first_match_prob = c.resolve_advance_prob(h, a, model, scaler, T)
                break
            if a == pt:
                opponent = h
                first_match_prob = 1 - c.resolve_advance_prob(h, a, model, scaler, T)
                break

        if stages and opponent:
            order = ["Reaches Round of 16", "Reaches Quarterfinal", "Reaches Semifinal", "Reaches Final"]
            markers = [("vs " + name_of(opponent), first_match_prob)]
            markers += [(s.replace("Reaches ", ""), stages[s]) for s in order if s in stages]
            champion_pct = stages.get("Champion", 0.0)

            marker_html = ""
            for i, (label, pct) in enumerate(markers):
                delay = i * 0.5
                marker_html += f"""
                <div style="display:flex;flex-direction:column;align-items:center;min-width:90px">
                  <div style="width:22px;height:22px;border-radius:50%;background:#30363d;
                              animation:lightUp 0.5s ease forwards;animation-delay:{delay}s"></div>
                  <div style="font-size:12px;color:#e6edf3;margin-top:6px;text-align:center">{label}</div>
                  <div style="font-size:12px;color:#8b949e">{pct:.0%}</div>
                </div>"""
                if i < len(markers) - 1:
                    marker_html += f"""
                    <div style="flex:1;height:4px;background:#30363d;position:relative;
                                overflow:hidden;margin-top:11px;min-width:40px">
                      <div style="position:absolute;left:0;top:0;height:100%;width:0%;
                                  background:#1DB954;animation:drawLine 0.5s linear forwards;
                                  animation-delay:{delay + 0.25}s"></div>
                    </div>"""

            final_delay = len(markers) * 0.5
            path_html = f"""
            <div style="font-family:sans-serif;padding:20px 16px;background:#0d1117;border-radius:10px">
              <div style="font-weight:600;color:#e6edf3;margin-bottom:16px">
                {name_of(pt)}'s path to the Final
              </div>
              <div style="display:flex;align-items:flex-start">
                {marker_html}
              </div>
              <div style="text-align:center;margin-top:20px;opacity:0;
                          animation:fadeIn 0.6s ease forwards;animation-delay:{final_delay}s">
                <span style="font-size:32px">🏆</span><br>
                <span style="font-size:20px;font-weight:700;color:#F0C040">
                  Champion odds: {champion_pct:.1%}
                </span>
              </div>
            </div>
            <style>
              @keyframes lightUp {{
                from {{ background:#30363d; box-shadow:none; }}
                to   {{ background:#1DB954; box-shadow:0 0 10px #1DB954; }}
              }}
              @keyframes drawLine {{ from {{ width:0%; }} to {{ width:100%; }} }}
              @keyframes fadeIn   {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
            </style>
            """
            st.components.v1.html(path_html, height=180)
            st.caption(f"Based on {n_trials:,} simulated tournaments.")
        if st.button("Clear"):
            del st.session_state["path_team"]
            st.rerun()

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

    # ── Build the results table ──────────────────────────────────────────
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
