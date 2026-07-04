"""
Bracket page — built around the feed's REAL bracket wiring.

Every knockout match beyond Round of 16 carries real labels like
"Winner Match 89" telling us exactly which earlier match decides that
slot. We follow those labels directly instead of assuming adjacent
matches pair up -- proven necessary: the real data showed match 90 fed
by matches 73 AND 75, not the adjacent 73/74 our old code assumed.

Each side of every match resolves INDEPENDENTLY: the moment a team wins
their earlier match, they show up in their next box immediately, even
if the team who'll eventually face them hasn't been decided yet -- same
as the real FIFA bracket, where a team can sit alone waiting.
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
import common as c

st.set_page_config(page_title="Tournament Bracket Odds", page_icon="🏆", layout="wide")
st.title("🏆 Tournament Bracket — Odds by Stage")

PATH_CSS = """
<style>
@keyframes lightUp {
  from { background:#0d1117; border-color:#30363d; box-shadow:none; }
  to   { background:#0d2818; border-color:#1DB954; box-shadow:0 0 10px rgba(29,185,84,0.5); }
}
</style>
"""

_WINNER_RE = re.compile(r"Winner Match (\d+)")


def main():
    try:
        games = c.fetch_wc26("games")
        teams = c.fetch_wc26("teams")
    except Exception as e:
        st.error(f"Couldn't reach worldcup26.ir: {e}")
        st.stop()

    team_lookup = c.build_team_lookup(teams)

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

    def contains_team(node, team_code):
        """Does this node's real, eventual lineage include this team?"""
        if node is None:
            return False
        if node["kind"] == "match":
            game = node["game"]
            for side in ("home", "away"):
                tid = game.get(f"{side}_team_id")
                if tid in team_lookup and team_lookup[tid].get("fifa_code") == team_code:
                    return True
            return False
        return contains_team(node["left"], team_code) or contains_team(node["right"], team_code)

    # ── Find every knockout-type match and index it by ID ───────────────────
    knockout_games = [g for g in games if (g.get("type") or "").lower() in c.ROUND_ORDER]
    if not knockout_games:
        st.info("No knockout-stage matches found in the feed yet.")
        st.stop()
    game_by_id = {g["id"]: g for g in knockout_games}

    final_games = [g for g in knockout_games if (g.get("type") or "").lower() == "final"]
    if not final_games:
        st.warning(
            "Couldn't find a 'final'-type match in the feed yet -- the bracket page "
            "needs that to build the full tree. Check back once it appears."
        )
        st.stop()
    final_game = final_games[0]

    # ── Build the REAL bracket tree by following the actual labels ──────────
    tree = c.build_real_bracket_tree(final_game, game_by_id)
    total_height = c.tree_height(tree)

    # ── Monte Carlo simulation, cached so clicking something unrelated
    #    (like the path button) doesn't re-run thousands of fresh trials
    #    every single time -- only a real change to the bracket or trial
    #    count should trigger a genuine recomputation ──────────────────────
    @st.cache_data(show_spinner=False)
    def cached_simulate(_model, _scaler, _T, fixture_tree, n_trials):
        return c.simulate_tournament(fixture_tree, _model, _scaler, _T, n_trials=n_trials)

    fixture_tree = c.tree_to_fixture(tree, game_by_id, team_lookup)
    n_trials = st.select_slider("Monte Carlo trials", [5_000, 20_000, 50_000, 100_000], value=20_000)
    model, scaler, T = c.load_model()
    with st.spinner(f"Running {n_trials:,} simulated tournaments..."):
        odds = cached_simulate(model, scaler, T, fixture_tree, n_trials)

    path_team = st.session_state.get("path_team")

    def render_team_row(code, value_text, key_suffix):
        """One team's row -- flag, clickable name, and a value on the right
        (a live match's win %, a finished match's score, or nothing at all
        for a bare confirmed-but-opponent-unknown slot). Every call site uses
        this SAME layout so every box in the bracket ends up the same
        height -- that's what keeps the whole column's spacing lined up."""
        col_flag, col_name, col_path = st.columns([1, 7, 0.6])
        with col_flag:
            flag = flag_of(code)
            if flag:
                st.image(flag, width=26)
        with col_name:
            label = f"{name_of(code)}  ·  {value_text}" if value_text else name_of(code)
            clicked = st.button(label, key=f"team_{code}_{key_suffix}", use_container_width=True)
        with col_path:
            if st.button("🏁", key=f"path_{code}_{key_suffix}", help="See path to Final"):
                st.session_state["path_team"] = code
        if clicked:
            st.query_params.clear()
            st.query_params["team"] = code
            st.switch_page("pages/1_History.py")

    def render_side(game, side, key_suffix):
        """One side of a match box -- confirmed team, a speculative path
        preview, or a plain waiting label, decided independently of the
        other side."""
        code, confirmed = c.resolve_slot(game, side, game_by_id, team_lookup)
        if confirmed:
            render_team_row(code, "", key_suffix)
            return

        label = game.get(f"{side}_team_label", "") or "TBD"
        m = _WINNER_RE.search(label)
        if path_team and m and m.group(1) in game_by_id:
            src_tree = c.build_real_bracket_tree(game_by_id[m.group(1)], game_by_id)
            if contains_team(src_tree, path_team):
                # Delay comes straight from this match's REAL stage -- r16
                # always lights first, then qf, then sf, then final, in that
                # exact fixed order every time. No calculation to get wrong.
                game_type = (game.get("type") or "r16").lower()
                stage_index = c.ROUND_ORDER.index(game_type) if game_type in c.ROUND_ORDER else 0
                delay = stage_index * 0.5
                stage_label = c.TYPE_TO_STAGE_LABEL.get(game_type, "")
                stage_key = f"Reaches {stage_label}" if stage_label else None
                pct = odds.get(path_team, {}).get(stage_key, 0.0) if stage_key else 0.0
                flag = flag_of(path_team)
                st.markdown(
                    f"<div style='border:1px solid #30363d;border-radius:8px;padding:8px 10px;"
                    f"min-height:44px;display:flex;align-items:center;gap:8px;"
                    f"animation:lightUp 0.6s ease forwards;animation-delay:{delay}s'>"
                    f"<img src='{flag}' style='width:22px;border-radius:3px'/>"
                    f"<div style='font-size:13px;color:#e6edf3'>{name_of(path_team)}</div>"
                    f"<div style='margin-left:auto;font-size:12px;color:#8b949e'>{pct:.0%}</div></div>",
                    unsafe_allow_html=True,
                )
                return

        # A real (disabled) button instead of a custom HTML box -- since
        # it's the SAME widget type Streamlit uses for every clickable row,
        # its height is guaranteed to match them exactly. No more guessing
        # a pixel number and hoping it's close enough.
        col_flag, col_name, col_path = st.columns([1, 7, 0.6])
        with col_name:
            st.button(label, key=f"waiting_{game.get('id')}_{side}_{key_suffix}",
                     use_container_width=True, disabled=True)

    def render_node(node, key_suffix):
        game = node["game"]
        is_finished = str(game.get("finished", "")).upper() == "TRUE"
        home_code, home_conf = c.resolve_slot(game, "home", game_by_id, team_lookup)
        away_code, away_conf = c.resolve_slot(game, "away", game_by_id, team_lookup)

        with st.container(border=True):
            if is_finished and home_conf and away_conf:
                hs, as_ = c.safe_int(game.get("home_score")), c.safe_int(game.get("away_score"))
                render_team_row(home_code, str(hs), f"{key_suffix}_h")
                render_team_row(away_code, str(as_), f"{key_suffix}_a")
            elif home_conf and away_conf:
                p_h = c.resolve_advance_prob(home_code, away_code, model, scaler, T)
                render_team_row(home_code, f"{p_h:.0%}", f"{key_suffix}_h")
                render_team_row(away_code, f"{1 - p_h:.0%}", f"{key_suffix}_a")
            else:
                render_side(game, "home", f"{key_suffix}_h")
                render_side(game, "away", f"{key_suffix}_a")

    def nodes_at_height(node, target_h):
        if node is None:
            return []
        h = c.tree_height(node)
        if h == target_h:
            return [node]
        if node["kind"] == "match":
            return []
        return nodes_at_height(node["left"], target_h) + nodes_at_height(node["right"], target_h)

    def render_side_columns(root, columns, col_start, mirrored):
        H = 132
        side_height = c.tree_height(root)

        heights = list(range(0, side_height + 1))
        col_order = heights if not mirrored else list(reversed(heights))
        for i, h in enumerate(col_order):
            col_idx = col_start + i
            nodes = nodes_at_height(root, h)
            stage_label = c.TYPE_TO_STAGE_LABEL.get((nodes[0]["game"].get("type") or "").lower(), f"Round {h}") if nodes else ""
            with columns[col_idx]:
                st.markdown(f"<div style='text-align:center;font-weight:600;margin-bottom:8px'>{stage_label}</div>",
                           unsafe_allow_html=True)
                pitch = H * (2 ** h)
                top_offset = max(0, (pitch - H) / 2)
                gap_between = max(0, pitch - H)
                if top_offset > 0:
                    st.markdown(f"<div style='height:{top_offset:.0f}px'></div>", unsafe_allow_html=True)
                for idx, node in enumerate(nodes):
                    render_node(node, f"{'L' if not mirrored else 'R'}_{h}_{idx}")
                    if idx < len(nodes) - 1 and gap_between > 0:
                        st.markdown(f"<div style='height:{gap_between:.0f}px'></div>", unsafe_allow_html=True)

    st.markdown(PATH_CSS, unsafe_allow_html=True)
    st.subheader("Bracket")

    left_side_height = c.tree_height(tree["left"]) if tree["left"] else 0
    right_side_height = c.tree_height(tree["right"]) if tree["right"] else 0
    n_left_cols = left_side_height + 1
    n_right_cols = right_side_height + 1

    def col_weight(h):
        return 2.4 if h == 0 else 1.0

    left_weights = [col_weight(h) for h in range(n_left_cols)]
    right_weights = [col_weight(h) for h in reversed(range(n_right_cols))]
    columns = st.columns(left_weights + [1.3] + right_weights)

    if tree["left"]:
        render_side_columns(tree["left"], columns, 0, mirrored=False)

    with columns[n_left_cols]:
        st.markdown(f"<div style='text-align:center;font-weight:600;margin-bottom:8px'>Final</div>",
                   unsafe_allow_html=True)
        H = 132
        pitch = H * (2 ** max(left_side_height, right_side_height))
        offset = max(0, (pitch - H) / 2)
        if offset > 0:
            st.markdown(f"<div style='height:{offset:.0f}px'></div>", unsafe_allow_html=True)
        render_node(tree, "final")
        st.markdown("<div style='text-align:center;font-size:24px'>🏆</div>", unsafe_allow_html=True)

    if tree["right"]:
        render_side_columns(tree["right"], columns, n_left_cols + 1, mirrored=True)

    st.caption("Click a team name to see their most recent match on History. Click 🏁 to trace their path to the Final.")

    if path_team is not None and st.button("Clear path"):
        del st.session_state["path_team"]
        st.rerun()

    with st.expander("Which matches were found"):
        for g in knockout_games:
            hc, _ = c.resolve_slot(g, "home", game_by_id, team_lookup)
            ac, _ = c.resolve_slot(g, "away", game_by_id, team_lookup)
            hl = name_of(hc) if hc else g.get("home_team_label", "TBD")
            al = name_of(ac) if ac else g.get("away_team_label", "TBD")
            stage = c.TYPE_TO_STAGE_LABEL.get((g.get("type") or "").lower(), g.get("type", "?"))
            status = "FINISHED" if str(g.get("finished","")).upper()=="TRUE" else "upcoming"
            st.write(f"[{stage}] {hl} vs {al}  ({status})")

    # ── Results table ─────────────────────────────────────────────────────
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
            {"timestamp": stamp, "team": team, "predicted_stage": stage, "probability": round(prob, 4)}
            for team, stages in odds.items() for stage, prob in stages.items()
        ]
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        st.success(f"Saved {len(log_rows)} odds to results/bracket_odds_history.csv at {stamp[:19]} UTC.")
    st.caption(
        "This is a manual save, on purpose -- Streamlit reruns this page on every click and "
        "slider move, so auto-saving every time would flood the file with near-duplicate rows."
    )

    st.dataframe(df.style.format("{:.1%}"), use_container_width=True)
    st.caption(
        f"Based on {n_trials:,} simulated tournaments using football_v2.pth, built from the feed's "
        f"real 'Winner Match N' wiring -- not an assumed pairing. Draw probability in each knockout "
        f"match is split 50/50 between the two teams."
    )


if __name__ == "__main__":
    main()
