import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import v3_ui_utils as ui

st.set_page_config(page_title="Match View", layout="wide")

team_theme = st.sidebar.selectbox("Match Theme", ["Manchester City", "Arsenal", "Real Madrid", "Barcelona", "Default"])
ui.apply_theme(team_theme)

st.title("🔴 Live Match Center")

# Match Banner
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown(f"<h2 style='text-align: center; color:{ui.TEAM_THEMES.get(team_theme, ui.TEAM_THEMES['Default'])['primary']}'>Man City</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Home | 4-3-3</p>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>2 - 1</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>88' | Etihad Stadium | Ref: Michael Oliver</p>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<h2 style='text-align: center; color:{ui.TEAM_THEMES.get('Arsenal', ui.TEAM_THEMES['Default'])['primary']}'>Arsenal</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Away | 4-2-3-1</p>", unsafe_allow_html=True)

st.markdown("---")

# Win Probability Bar
st.subheader("Live Win Probability")
p1, p2, p3 = st.columns(3)
p1.metric("Man City (Home)", "72%", "+15%")
p2.metric("Draw", "18%", "-10%")
p3.metric("Arsenal (Away)", "10%", "-5%")

st.markdown("---")

# Pitch View
st.subheader("Live Formations & Lineups")

home_players = ["Ederson", "Walker", "Dias", "Akanji", "Gvardiol", "Rodri", "De Bruyne", "Silva", "Foden", "Haaland", "Grealish"]
away_players = ["Raya", "White", "Saliba", "Gabriel", "Zinchenko", "Rice", "Odegaard", "Saka", "Martinelli", "Jesus", "Trossard"]

svg = ui.generate_pitch_svg(
    home_formation="4-3-3", 
    away_formation="4-2-3-1",
    home_color=ui.TEAM_THEMES["Manchester City"]["primary"],
    away_color=ui.TEAM_THEMES["Arsenal"]["primary"],
    home_players=home_players,
    away_players=away_players
)

col_pitch, col_events = st.columns([2, 1])
with col_pitch:
    st.markdown(svg, unsafe_allow_html=True)
with col_events:
    st.markdown("### Match Events")
    st.markdown("⚽ **15'** - Goal (Arsenal) - Saka")
    st.markdown("⚽ **40'** - Goal (Man City) - De Bruyne")
    st.markdown("🟥 **45'** - Red Card (Arsenal) - Gabriel")
    st.markdown("⚽ **88'** - Goal (Man City) - Haaland")
    
    st.markdown("---")
    st.markdown("### Managers")
    st.markdown("**Home:** Pep Guardiola")
    st.markdown("**Away:** Mikel Arteta")
