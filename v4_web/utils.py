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

TEAM_MANAGERS = {
    "Arsenal": "Mikel Arteta",
    "Manchester City": "Pep Guardiola",
    "Liverpool": "Arne Slot",
    "Manchester United": "Erik ten Hag",
    "Chelsea": "Enzo Maresca",
    "Tottenham Hotspur": "Ange Postecoglou",
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

# Distinct colors for major teams
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
    
    "Default": {"primary": "#14b8a6", "secondary": "#0f172a", "bg": "#0b0f19"}
}

def get_theme_for_team(team_name):
    if team_name in TEAM_THEMES:
        return TEAM_THEMES[team_name]
    
    # Generate a consistent pseudo-random theme for teams not explicitly mapped
    h = hash(team_name)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b = (h & 0x0000FF)
    
    r = max(100, r)
    g = max(100, g)
    b = max(100, b)
    
    primary = f"#{r:02x}{g:02x}{b:02x}"
    bg = f"#{r//15:02x}{g//15:02x}{b//15:02x}"
    
    return {"primary": primary, "secondary": "#FFFFFF", "bg": bg}



# Sample default starting lineups so pitches render immediately without external API sync
DEFAULT_SQUADS = {
    "Arsenal": ["Raya", "White", "Saliba", "Gabriel", "Timber", "Partey", "Rice", "Odegaard", "Saka", "Havertz", "Martinelli"],
    "Manchester City": ["Ederson", "Walker", "Dias", "Akanji", "Gvardiol", "Rodri", "Kovacic", "De Bruyne", "Silva", "Haaland", "Foden"],
    "Liverpool": ["Alisson", "Alexander-Arnold", "Konate", "Van Dijk", "Robertson", "Gravenberch", "Mac Allister", "Szoboszlai", "Salah", "Jota", "Diaz"],
    "Aston Villa": ["Martinez", "Cash", "Konsa", "Torres", "Digne", "Onana", "Tielemans", "McGinn", "Bailey", "Watkins", "Rogers"],
    "Chelsea": ["Sanchez", "Gusto", "Fofana", "Colwill", "Cucurella", "Caicedo", "Lavia", "Palmer", "Madueke", "Jackson", "Neto"],
    "Real Madrid": ["Courtois", "Carvajal", "Militao", "Rudiger", "Mendy", "Valverde", "Tchouameni", "Bellingham", "Rodrygo", "Mbappe", "Vinicius Jr"],
    "Barcelona": ["Ter Stegen", "Kounde", "Cubarsi", "Martinez", "Balde", "Casado", "Pedri", "Yamal", "Olmo", "Raphinha", "Lewandowski"]
}
def get_squad_for_team(team_name):
    if team_name in DEFAULT_SQUADS:
        return DEFAULT_SQUADS[team_name]
    # Generic fallback: P1 through P11
    return [f"{team_name[:3]}_{i+1}" for i in range(11)]

def generate_pitch_svg_vertical(formation="4-3-3", team_color="#14b8a6", players=None, team_name="Default"):
    """Generates a vertical half-pitch for the Team Profile page. Quant Theme."""
    if players is None or len(players) == 0:
        players = get_squad_for_team(team_name)
        
    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return [1, 4, 3, 3]
        return [1] + [int(x) for x in fmt_str.split("-")]
        
    lines = parse_formation(formation)
    W, H = 160, 200
    
    svg = f"""<svg width="100%" style="max-width: 450px; display: block; margin: 0 auto;" viewBox="0 0 {W} {H}" xmlns="http://w3.org">
        <defs>
            <circle id="jersey-vert" cx="0" cy="0" r="5" stroke="#1e293b" stroke-width="1.2"/>
            <circle id="jersey-ghost" cx="0" cy="0" r="5" stroke="#64748b" stroke-dasharray="2,2" stroke-width="1" fill="none"/>
        </defs>
        <rect width="{W}" height="{H}" fill="#0b0f19" rx="6" ry="6" stroke="#1e293b" stroke-width="1.5"/>
        <rect x="6" y="6" width="148" height="188" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        
        <!-- Penalty Box & Goal -->
        <rect x="35" y="6" width="90" height="32" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        <rect x="58" y="6" width="44" height="12" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        <circle cx="80" cy="24" r="1.5" fill="#334155"/>
        
        <!-- Halfway line & Center Circle -->
        <line x1="6" y1="194" x2="154" y2="194" stroke="#1e293b" stroke-width="0.75"/>
        <path d="M 60 194 A 20 20 0 0 1 100 194" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        <circle cx="80" cy="194" r="1.5" fill="#334155"/>
    """
    
    if lines:
        y_steps = len(lines)
        p_idx = 0
        for row_idx, num_players in enumerate(lines):
            # Goalkeeper at the top, attackers towards halfway line
            y = 22 + (150 / max(1, y_steps - 1)) * row_idx
            for col_idx in range(num_players):
                x = 12 + (136 / (num_players + 1)) * (col_idx + 1)
                name = players[p_idx] if p_idx < len(players) else f"P{p_idx+1}"
                
                # Draw Ideal-XI ghost node slightly offset
                svg += f'<use href="#jersey-ghost" x="{x+3}" y="{y-3}" />'
                
                # Active Starter node
                svg += f"""
                    <a href="/player?name={name}&team={team_name}" style="cursor: pointer;">
                        <use href="#jersey-vert" x="{x}" y="{y}" fill="{team_color}" />
                        <text x="{x}" y="{y+9}" fill="#94a3b8" font-family="'JetBrains Mono', monospace" font-size="4.5px" text-anchor="middle">{name}</text>
                    </a>
                """
                p_idx += 1
                
    svg += "</svg>"
    return svg

def generate_pitch_svg_horizontal(home_formation="4-3-3", away_formation="4-2-3-1", home_color="#14b8a6", away_color="#f43f5e", home_players=None, away_players=None, home_team="Home", away_team="Away"):
    """Generates a native horizontal full-pitch. Quant Theme."""
    # Ensure players default to team rosters rather than leaving pitch blank
    if home_players is None or len(home_players) == 0:
        home_players = get_squad_for_team(home_team)
    if away_players is None or len(away_players) == 0:
        away_players = get_squad_for_team(away_team)

    W, H = 160, 100
    
    svg = f"""<svg width="100%" style="max-width: 100%; display: block; margin: 0 auto;" viewBox="0 0 {W} {H}" xmlns="http://w3.org">
        <defs>
            <filter id="glow-h" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="1.2" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="glow-a" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="1.2" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        </defs>
        <rect width="{W}" height="{H}" fill="#0b0f19" rx="6" ry="6" stroke="#1e293b" stroke-width="1.5"/>
        <rect x="5" y="5" width="150" height="90" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        
        <!-- Pitch Markings -->
        <line x1="80" y1="5" x2="80" y2="95" stroke="#1e293b" stroke-width="0.75"/>
        <circle cx="80" cy="50" r="14" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        <circle cx="80" cy="50" r="1" fill="#334155"/>
        
        <!-- Home Box (Left) -->
        <rect x="5" y="22" width="22" height="56" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        <rect x="5" y="36" width="8" height="28" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        <circle cx="16" cy="50" r="1" fill="#334155"/>
        
        <!-- Away Box (Right) -->
        <rect x="133" y="22" width="22" height="56" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        <rect x="147" y="36" width="8" height="28" fill="none" stroke="#1e293b" stroke-width="0.75"/>
        <circle cx="144" cy="50" r="1" fill="#334155"/>
    """
    
    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return [1, 4, 3, 3]
        return [1] + [int(x) for x in fmt_str.split("-")]
        
    h_lines = parse_formation(home_formation)
    a_lines = parse_formation(away_formation)
    
    # Home Team (Left -> Right)
    if h_lines and home_players:
        x_steps = len(h_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(h_lines):
            x = 12 + (58 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 8 + (84 / (num_players + 1)) * (row_idx + 1)
                name = home_players[p_idx] if p_idx < len(home_players) else f"H{p_idx+1}"
                glow = 'filter="url(#glow-h)"' if (p_idx % 3 == 0) else ''
                svg += f"""
                    <a href="/player?name={name}&team={home_team}" style="cursor: pointer;">
                        <circle cx="{x}" cy="{y}" r="3.2" fill="{home_color}" stroke="#0b0f19" stroke-width="0.8" {glow}/>
                        <text x="{x}" y="{y+6}" fill="#94a3b8" font-family="'JetBrains Mono', monospace" font-size="2.6px" text-anchor="middle">{name}</text>
                    </a>
                """
                p_idx += 1

    # Away Team (Right -> Left)
    if a_lines and away_players:
        x_steps = len(a_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(a_lines):
            x = 148 - (58 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 8 + (84 / (num_players + 1)) * (row_idx + 1)
                name = away_players[p_idx] if p_idx < len(away_players) else f"A{p_idx+1}"
                glow = 'filter="url(#glow-a)"' if (p_idx % 4 == 0) else ''
                svg += f"""
                    <a href="/player?name={name}&team={away_team}" style="cursor: pointer;">
                        <circle cx="{x}" cy="{y}" r="3.2" fill="{away_color}" stroke="#0b0f19" stroke-width="0.8" {glow}/>
                        <text x="{x}" y="{y+6}" fill="#94a3b8" font-family="'JetBrains Mono', monospace" font-size="2.6px" text-anchor="middle">{name}</text>
                    </a>
                """
                p_idx += 1
            
    svg += "</svg>"
    return svg
