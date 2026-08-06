import streamlit as st
import v3_ui_utils as ui
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
# import the Sofascore fetcher logic (mocked here for the UI demo)
# from generate_v3_sofascore import fetch_sofascore

st.set_page_config(page_title="V3 League Hub", page_icon="⚽", layout="wide")

st.sidebar.title("⚽ Universal Football Model")
st.sidebar.markdown("---")

league = st.sidebar.selectbox("Select League", ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"])

# Dynamic Team Selection based on League
league_teams = {
    "Premier League": ["Arsenal", "Manchester City", "Manchester United", "Chelsea", "Liverpool"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid"],
    "Serie A": ["Juventus", "AC Milan", "Inter Milan"],
    "Bundesliga": ["Bayern Munich", "Dortmund", "Bayer Leverkusen"],
    "Ligue 1": ["Paris SG", "Marseille", "Lyon"]
}

available_teams = league_teams.get(league, ["Default"])
team = st.sidebar.selectbox("Select Team Theme", ["Default"] + available_teams, index=1)


ui.apply_theme(team)

st.title(f"🌍 League Hub: {league}")
st.markdown(f"Welcome to the **V3 Universal Model** dashboard for the {league}. Data is dynamically powered by Sofascore.")

# Featured Match of the Day
st.markdown("---")
st.subheader("Featured Match")

col_match_text, col_pitch = st.columns([1, 1.5])
with col_match_text:
    st.markdown("### Match of the Week")
    if league == "Premier League":
        home, away = "Manchester City", "Arsenal"
    elif league == "La Liga":
        home, away = "Real Madrid", "Barcelona"
    else:
        home, away = available_teams[0], available_teams[-1] if len(available_teams)>1 else "Default"
        
    st.markdown(f"**{home}** vs **{away}**")
    st.info("Status: LIVE - 65'")
    st.metric("V3 Projected Winner", home, "+12% Prob Shift")
    st.markdown("👈 *Navigate to the 'Live Match' page in the sidebar for full play-by-play odds.*")

with col_pitch:
    # Render the dynamic pitch
    h_color = ui.TEAM_THEMES.get(home, ui.TEAM_THEMES['Default'])['primary']
    a_color = ui.TEAM_THEMES.get(away, ui.TEAM_THEMES['Default'])['primary']
    
    svg = ui.generate_pitch_svg(
        home_formation="4-3-3", 
        away_formation="4-2-3-1",
        home_color=h_color,
        away_color=a_color
    )
    st.markdown(svg, unsafe_allow_html=True)


st.markdown("---")
st.subheader("Live Standings")
# Mock standings table that matches the theme
standings = pd.DataFrame({
    "Pos": range(1, len(available_teams)+1),
    "Team": available_teams,
    "Played": [38]*len(available_teams),
    "Points": [89, 87, 75, 66, 60][:len(available_teams)]
})
st.markdown(f"""
<style>
[data-testid="stDataFrame"] {{
    background-color: {ui.TEAM_THEMES.get(team, ui.TEAM_THEMES['Default'])['bg']} !important;
}}
</style>
""", unsafe_allow_html=True)
st.dataframe(standings, use_container_width=True, hide_index=True)
