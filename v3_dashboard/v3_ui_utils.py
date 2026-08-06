import streamlit as st

# Expanded Theme Dictionary
TEAM_THEMES = {
    "Manchester City": {"primary": "#6CABDD", "secondary": "#1C2C5B", "bg": "#0B1320"},
    "Arsenal": {"primary": "#EF0107", "secondary": "#9C824A", "bg": "#1A0505"},
    "Manchester United": {"primary": "#DA291C", "secondary": "#FBE122", "bg": "#1C0A0A"},
    "Real Madrid": {"primary": "#FFFFFF", "secondary": "#00529F", "bg": "#0F1626"},
    "Barcelona": {"primary": "#A50044", "secondary": "#004D98", "bg": "#0A1128"},
    "Chelsea": {"primary": "#034694", "secondary": "#DBA111", "bg": "#04142B"},
    "Liverpool": {"primary": "#C8102E", "secondary": "#00B2A9", "bg": "#180205"},
    "Bayern Munich": {"primary": "#DC052D", "secondary": "#0066B2", "bg": "#1C0005"},
    "Paris SG": {"primary": "#004170", "secondary": "#DA291C", "bg": "#020D1A"},
    "Juventus": {"primary": "#000000", "secondary": "#FFFFFF", "bg": "#111111"},
    "Default": {"primary": "#00FF87", "secondary": "#008F4C", "bg": "#0E1117"}
}

def apply_theme(team_name="Default"):
    theme = TEAM_THEMES.get(team_name, TEAM_THEMES["Default"])
    css = f"""
    <style>
    /* Override base background */
    .stApp {{
        background-color: {theme['bg']} !important;
        color: #FAFAFA !important;
    }}
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #050810 !important;
        border-right: 2px solid {theme['primary']} !important;
    }}
    /* Typography */
    h1, h2, h3, h4 {{
        color: {theme['primary']} !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
    }}
    /* Metric styling */
    .stMetric label {{
        color: #CCCCCC !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {theme['primary']} !important;
        font-weight: 900 !important;
    }}
    /* SVG Pitch styling */
    .player-label {{
        fill: white !important;
        font-family: 'Trebuchet MS', sans-serif !important;
        font-size: 3.5px !important;
        font-weight: bold !important;
        text-anchor: middle !important;
    }}
    .pitch-bg {{
        fill: #2e7d32 !important;
    }}
    .pitch-lines {{
        fill: none !important;
        stroke: rgba(255,255,255,0.6) !important;
        stroke-width: 0.5 !important;
    }}
    hr {{
        border-color: {theme['secondary']} !important;
        opacity: 0.5;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def generate_pitch_svg(home_formation="4-3-3", away_formation="4-2-3-1", home_color="#6CABDD", away_color="#EF0107", home_players=None, away_players=None):
    if home_players is None:
        home_players = [f"H{i}" for i in range(11)]
    if away_players is None:
        away_players = [f"A{i}" for i in range(11)]
        
    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return []
        return [1] + [int(x) for x in fmt_str.split("-")]
        
    h_lines = parse_formation(home_formation)
    a_lines = parse_formation(away_formation)
    
    W, H = 100, 140
    
    # We must wrap the SVG in a div and remove comments that streamlit might try to markdown-parse
    svg = f"""<div style="display: flex; justify-content: center; width: 100%; margin: 20px 0;">
        <svg width="100%" max-width="400px" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
        <rect width="{W}" height="{H}" class="pitch-bg" rx="2" ry="2"/>
        <rect x="5" y="5" width="90" height="130" class="pitch-lines" />
        <line x1="5" y1="70" x2="95" y2="70" class="pitch-lines" />
        <circle cx="50" cy="70" r="12" class="pitch-lines" />
        <circle cx="50" cy="70" r="0.8" fill="white" />
        <rect x="25" y="5" width="50" height="18" class="pitch-lines" />
        <rect x="25" y="117" width="50" height="18" class="pitch-lines" />
        <rect x="38" y="5" width="24" height="6" class="pitch-lines" />
        <rect x="38" y="129" width="24" height="6" class="pitch-lines" />
        <line x1="44" y1="5" x2="56" y2="5" stroke="white" stroke-width="1.5" />
        <line x1="44" y1="135" x2="56" y2="135" stroke="white" stroke-width="1.5" />
    """
    
    # Draw Home Players (Top Half, attacking downwards)
    if h_lines:
        y_steps = len(h_lines)
        p_idx = 0
        for row_idx, num_players in enumerate(h_lines):
            y = 12 + (50 / max(1, y_steps - 1)) * row_idx
            for col_idx in range(num_players):
                x = 10 + (80 / (num_players + 1)) * (col_idx + 1)
                name = home_players[p_idx] if p_idx < len(home_players) else ""
                svg += f"""
                    <circle cx="{x}" cy="{y}" r="2.8" fill="{home_color}" stroke="white" stroke-width="0.5"/>
                    <text x="{x}" y="{y+4.8}" class="player-label">{name}</text>
                """
                p_idx += 1

    # Draw Away Players (Bottom Half, attacking upwards)
    if a_lines:
        y_steps = len(a_lines)
        p_idx = 0
        for row_idx, num_players in enumerate(a_lines):
            y = 128 - (50 / max(1, y_steps - 1)) * row_idx
            for col_idx in range(num_players):
                x = 10 + (80 / (num_players + 1)) * (col_idx + 1)
                name = away_players[p_idx] if p_idx < len(away_players) else ""
                border = "#00529F" if away_color.upper() == "#FFFFFF" else "white"
                svg += f"""
                    <circle cx="{x}" cy="{y}" r="2.8" fill="{away_color}" stroke="{border}" stroke-width="0.5"/>
                    <text x="{x}" y="{y+4.8}" class="player-label">{name}</text>
                """
                p_idx += 1
            
    svg += "</svg></div>"
    return svg
