import json

# Comprehensive 20-team mappings for the Big 5 Leagues
LEAGUE_TEAMS = {
    "Premier League": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town", 
        "Leicester City", "Liverpool", "Manchester City", "Manchester United", 
        "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham Hotspur", 
        "West Ham United", "Wolverhampton"
    ],
    "La Liga": [
        "Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Club", "Real Sociedad",
        "Girona", "Real Betis", "Villarreal", "Valencia", "Sevilla",
        "Osasuna", "Celta Vigo", "Alaves", "Mallorca", "Getafe",
        "Las Palmas", "Rayo Vallecano", "Espanyol", "Valladolid", "Leganes"
    ],
    "Serie A": [
        "Inter Milan", "AC Milan", "Juventus", "Atalanta", "Bologna",
        "Roma", "Lazio", "Fiorentina", "Napoli", "Torino",
        "Genoa", "Monza", "Lecce", "Empoli", "Udinese",
        "Hellas Verona", "Cagliari", "Como", "Parma", "Venezia"
    ],
    "Bundesliga": [
        "Bayer Leverkusen", "Bayern Munich", "VfB Stuttgart", "RB Leipzig", "Borussia Dortmund",
        "Eintracht Frankfurt", "Hoffenheim", "Freiburg", "Heidenheim", "Augsburg",
        "Werder Bremen", "Wolfsburg", "Borussia Monchengladbach", "Bochum", "Union Berlin",
        "Mainz 05", "St. Pauli", "Holstein Kiel"
    ],
    "Ligue 1": [
        "Paris SG", "Monaco", "Marseille", "Lille", "Lens",
        "Nice", "Lyon", "Rennes", "Reims", "Toulouse",
        "Montpellier", "Strasbourg", "Nantes", "Le Havre", "Brest",
        "Auxerre", "Angers", "Saint-Etienne"
    ]
}

# Distinct colors for major teams (falling back to a hash generator for others so they all get unique colors)
TEAM_THEMES = {
    "Arsenal": {"primary": "#EF0107", "secondary": "#023474", "bg": "#1A0505"},
    "Manchester United": {"primary": "#DA291C", "secondary": "#FBE122", "bg": "#1C0A0A"},
    "Liverpool": {"primary": "#C8102E", "secondary": "#00B2A9", "bg": "#180205"},
    "Manchester City": {"primary": "#6CABDD", "secondary": "#1C2C5B", "bg": "#0B1320"},
    "Chelsea": {"primary": "#034694", "secondary": "#DBA111", "bg": "#04142B"},
    "Tottenham Hotspur": {"primary": "#132257", "secondary": "#FFFFFF", "bg": "#02040A"},
    "Aston Villa": {"primary": "#670E36", "secondary": "#95BFE5", "bg": "#110209"},
    "Newcastle United": {"primary": "#FFFFFF", "secondary": "#242424", "bg": "#111111"},
    
    "Real Madrid": {"primary": "#FFFFFF", "secondary": "#00529F", "bg": "#0F1626"},
    "Barcelona": {"primary": "#A50044", "secondary": "#004D98", "bg": "#0A1128"},
    "Atletico Madrid": {"primary": "#CB3524", "secondary": "#272E61", "bg": "#140503"},
    
    "Juventus": {"primary": "#FFFFFF", "secondary": "#000000", "bg": "#111111"},
    "AC Milan": {"primary": "#FB090B", "secondary": "#000000", "bg": "#1A0101"},
    "Inter Milan": {"primary": "#010E80", "secondary": "#FFFFFF", "bg": "#010214"},
    
    "Bayern Munich": {"primary": "#DC052D", "secondary": "#0066B2", "bg": "#1C0005"},
    "Borussia Dortmund": {"primary": "#FDE100", "secondary": "#000000", "bg": "#1A1700"},
    "Bayer Leverkusen": {"primary": "#E32221", "secondary": "#000000", "bg": "#170303"},
    
    "Paris SG": {"primary": "#004170", "secondary": "#DA291C", "bg": "#020D1A"},
    "Marseille": {"primary": "#2FAEE0", "secondary": "#FFFFFF", "bg": "#05161C"},
    
    "Default": {"primary": "#00FF87", "secondary": "#008F4C", "bg": "#0E1117"}
}

# Real Managers Mapping
TEAM_MANAGERS = {
    "Arsenal": "Mikel Arteta",
    "Manchester City": "Pep Guardiola",
    "Liverpool": "Arne Slot",
    "Manchester United": "Erik ten Hag",
    "Chelsea": "Enzo Maresca",
    "Tottenham Hotspur": "Ange Postecoglou",
    "Aston Villa": "Unai Emery",
    "Newcastle United": "Eddie Howe",
    "Real Madrid": "Carlo Ancelotti",
    "Barcelona": "Hansi Flick",
    "Atletico Madrid": "Diego Simeone",
    "Juventus": "Thiago Motta",
    "AC Milan": "Paulo Fonseca",
    "Inter Milan": "Simone Inzaghi",
    "Bayern Munich": "Vincent Kompany",
    "Borussia Dortmund": "Nuri Sahin",
    "Bayer Leverkusen": "Xabi Alonso",
    "Paris SG": "Luis Enrique",
}

def get_theme_for_team(team_name):
    if team_name in TEAM_THEMES:
        return TEAM_THEMES[team_name]
    
    # Generate a consistent pseudo-random theme for teams not explicitly mapped
    h = hash(team_name)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = (h & 0x0000FF)
    
    # Make sure primary is bright enough for dark mode
    r = max(100, r)
    g = max(100, g)
    b = max(100, b)
    
    primary = f"#{r:02x}{g:02x}{b:02x}"
    bg = f"#{r//15:02x}{g//15:02x}{b//15:02x}"
    
    return {"primary": primary, "secondary": "#FFFFFF", "bg": bg}


def generate_pitch_svg_vertical(formation="4-3-3", team_color="#EF0107", players=None, team_name="Default"):
    """Generates a vertical half-pitch for the Team Profile page."""
    if players is None:
        players = [f"P{i}" for i in range(11)]
        
    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return []
        return [1] + [int(x) for x in fmt_str.split("-")]
        
    lines = parse_formation(formation)
    W, H = 160, 200  # Significantly widened and lengthened to prevent any overlapping
    
    svg = f"""<svg width="100%" style="max-width: 600px; display: block; margin: 0 auto;" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
        <rect width="{W}" height="{H}" fill="#2e7d32" rx="2" ry="2"/>
        <rect x="5" y="5" width="150" height="190" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        <!-- Halfway line at bottom -->
        <line x1="5" y1="195" x2="155" y2="195" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        <circle cx="80" cy="195" r="20" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        <circle cx="80" cy="195" r="1.5" fill="white" />
        
        <rect x="35" y="5" width="90" height="30" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        <rect x="60" y="5" width="40" height="10" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        <line x1="72" y1="5" x2="88" y2="5" stroke="white" stroke-width="2" />
    """
    
    if lines:
        y_steps = len(lines)
        p_idx = 0
        for row_idx, num_players in enumerate(lines):
            y = 20 + (160 / max(1, y_steps - 1)) * row_idx
            for col_idx in range(num_players):
                x = 10 + (140 / (num_players + 1)) * (col_idx + 1)
                name = players[p_idx] if p_idx < len(players) else ""
                svg += f"""
                    <a href="/player?name={name}&team={team_name}" style="cursor: pointer;">
                        <circle cx="{x}" cy="{y}" r="5" fill="{team_color}" stroke="white" stroke-width="1"/>
                        <text x="{x}" y="{y+9}" fill="white" font-family="'Trebuchet MS', sans-serif" font-size="6px" font-weight="bold" text-anchor="middle">{name}</text>
                    </a>
                """
                p_idx += 1
                
    svg += "</svg>"
    return svg


def generate_pitch_svg_horizontal(home_formation="4-3-3", away_formation="4-2-3-1", home_color="#6CABDD", away_color="#EF0107", home_players=None, away_players=None, home_team="Home", away_team="Away"):
    """Generates a native horizontal full-pitch for the Live Match and Hub pages."""
    if home_players is None:
        home_players = [f"H{i}" for i in range(11)]
    if away_players is None:
        away_players = [f"A{i}" for i in range(11)]
        
    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return []
        return [1] + [int(x) for x in fmt_str.split("-")]
        
    h_lines = parse_formation(home_formation)
    a_lines = parse_formation(away_formation)
    
    W, H = 160, 100  # Made even wider so it expands beautifully when sidebar closes
    
    svg = f"""<svg width="100%" style="max-width: 100%; display: block; margin: 0 auto; transition: all 0.3s ease;" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
        <rect width="{W}" height="{H}" fill="#2e7d32" rx="2" ry="2"/>
        <rect x="5" y="5" width="150" height="90" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        
        <!-- Halfway line & circle -->
        <line x1="80" y1="5" x2="80" y2="95" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        <circle cx="80" cy="50" r="15" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        <circle cx="80" cy="50" r="0.8" fill="white" />
        
        <!-- Penalty Areas -->
        <rect x="5" y="22" width="22" height="56" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        <rect x="133" y="22" width="22" height="56" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        
        <!-- Goal Areas -->
        <rect x="5" y="37" width="8" height="26" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        <rect x="147" y="37" width="8" height="26" fill="none" stroke="rgba(255,255,255,0.6)" stroke-width="0.5"/>
        
        <!-- Goals -->
        <line x1="5" y1="44" x2="5" y2="56" stroke="white" stroke-width="2" />
        <line x1="155" y1="44" x2="155" y2="56" stroke="white" stroke-width="2" />
    """
    
    # Draw Home Players
    if h_lines:
        x_steps = len(h_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(h_lines):
            x = 15 + (60 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 10 + (80 / (num_players + 1)) * (row_idx + 1)
                name = home_players[p_idx] if p_idx < len(home_players) else ""
                svg += f"""
                    <a href="/player?name={name}&team={home_team}" style="cursor: pointer;">
                        <circle cx="{x}" cy="{y}" r="3" fill="{home_color}" stroke="white" stroke-width="0.5"/>
                        <text x="{x}" y="{y+5}" fill="white" font-family="'Trebuchet MS', sans-serif" font-size="3.5px" font-weight="bold" text-anchor="middle">{name}</text>
                    </a>
                """
                p_idx += 1

    # Draw Away Players
    if a_lines:
        x_steps = len(a_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(a_lines):
            x = 145 - (60 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 10 + (80 / (num_players + 1)) * (row_idx + 1)
                name = away_players[p_idx] if p_idx < len(away_players) else ""
                border = "#00529F" if away_color.upper() == "#FFFFFF" else "white"
                svg += f"""
                    <a href="/player?name={name}&team={away_team}" style="cursor: pointer;">
                        <circle cx="{x}" cy="{y}" r="3" fill="{away_color}" stroke="{border}" stroke-width="0.5"/>
                        <text x="{x}" y="{y+5}" fill="white" font-family="'Trebuchet MS', sans-serif" font-size="3.5px" font-weight="bold" text-anchor="middle">{name}</text>
                    </a>
                """
                p_idx += 1
            
    svg += "</svg>"
    return svg
