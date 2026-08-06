import streamlit as st
import v3_ui_utils as ui
import os
import sys

st.set_page_config(page_title="V3 Dashboard", page_icon="⚽", layout="wide")

st.sidebar.title("⚽ Universal Football Model")
st.sidebar.markdown("---")

league = st.sidebar.selectbox("Select League", ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"])
team = st.sidebar.selectbox("Select Team Theme", ["Default", "Manchester City", "Arsenal", "Manchester United", "Real Madrid", "Barcelona"])

ui.apply_theme(team)

st.title(f"🌍 Live Dashboard: {league}")
st.markdown("Welcome to the V3 Universal Model dashboard. We have moved beyond the \"AI slop\" default Streamlit theme. The colors, accents, and visuals now dynamically adapt to your selected club.")

st.info("👈 Use the sidebar to navigate to the **Live Match Center** or **Team Profiles** pages!")

st.markdown("### Model Engine Status")
c1, c2, c3 = st.columns(3)
c1.metric("Pipeline", "V3 Universal", "Active")
c2.metric("API Provider", "Sofascore", "Connected")
c3.metric("Supported Leagues", "Big 5", "+4")
