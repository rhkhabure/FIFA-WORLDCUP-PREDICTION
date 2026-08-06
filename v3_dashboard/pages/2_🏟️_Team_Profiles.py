import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import v3_ui_utils as ui
import pandas as pd

st.set_page_config(page_title="Team Profiles", layout="wide")

team_theme = st.sidebar.selectbox("Team Theme", ["Manchester United", "Arsenal", "Manchester City", "Real Madrid", "Barcelona", "Chelsea", "Liverpool", "Bayern Munich", "Paris SG", "Juventus", "Default"])
ui.apply_theme(team_theme)

st.title(f"🏟️ Team Profile: {team_theme}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Manager & Standard XI")
    st.markdown("**Manager:** Mikel Arteta" if team_theme == "Arsenal" else "**Manager:** Pep Guardiola" if team_theme == "Manchester City" else "**Manager:** Carlo Ancelotti" if team_theme == "Real Madrid" else "**Manager:** Head Coach")
    
    # Just a visual demo using the SVG pitch
    svg = ui.generate_pitch_svg(
        home_formation="4-3-3", 
        away_formation="0-0",
        home_color=ui.TEAM_THEMES.get(team_theme, ui.TEAM_THEMES['Default'])['primary']
    )
    st.markdown(svg, unsafe_allow_html=True)

with col2:
    st.subheader("Match History")
    
    # Mock Match History Table
    df = pd.DataFrame({
        "Date": ["Aug 01", "Aug 08", "Aug 15", "Aug 22", "Aug 29"],
        "Opponent": ["Chelsea", "Tottenham", "Liverpool", "Aston Villa", "Newcastle"],
        "Result": ["W 2-0", "D 1-1", "L 1-2", "W 3-0", "W 1-0"],
        "xG": ["2.1 - 0.8", "1.5 - 1.5", "0.9 - 2.4", "3.0 - 0.5", "1.2 - 0.4"],
        "Possession": ["55%", "48%", "42%", "65%", "58%"]
    })
    
    # Custom CSS for the dataframe so it isn't pure white/boring
    st.markdown(f"""
    <style>
    [data-testid="stDataFrame"] {{
        background-color: {ui.TEAM_THEMES.get(team_theme, ui.TEAM_THEMES['Default'])['bg']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.subheader("Upcoming Fixture")
    st.info("Next Match: vs Everton (Away) - Win Prob: 68%")
