"""
World Cup 2026 Live Win Probability — Live page (main entrypoint).

Run with:  streamlit run app.py
Needs:     pip install -r requirements.txt
"""

from datetime import datetime, timezone

import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

import common as c

st.set_page_config(page_title="World Cup Win Probability", page_icon="⚽", layout="wide")


def flag_push_html(home_flag_url, away_flag_url, home_code, away_code, p_away, p_draw, p_home):
    denom = p_home + p_away
    lean = round((p_home / denom) * 100) if denom > 0 else 50
    fog_px = 20 + p_draw * 160
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
        st.sidebar.warning("Install streamlit-autorefresh for automatic refresh.")

    try:
        games = c.fetch_wc26("games")
        teams = c.fetch_wc26("teams")
    except Exception as e:
        st.error(f"Couldn't reach worldcup26.ir: {e}")
        st.stop()

    team_lookup = c.build_team_lookup(teams)

    today = datetime.now().date()
    todays_games = [g for g in games if (c.parse_local_date(g.get("local_date")) or datetime.min).date() == today]

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

    model, scaler, T = c.load_model()
    feat_row, home_code, away_code, hs, as_, minute, is_finished = c.build_live_features(game, team_lookup)
    p_away, p_draw, p_home = c.predict(model, scaler, T, feat_row)

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

    if is_finished:
        st.caption("This match has finished — see the **History** page in the sidebar for the full probability timeline.")

    st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


if __name__ == "__main__":
    main()
