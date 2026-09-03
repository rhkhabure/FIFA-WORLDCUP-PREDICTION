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

def generate_pitch_svg_vertical(formation="4-3-3", team_color="#14b8a6", players=None, team_name="Default"):
    """Generates a vertical half-pitch for the Team Profile page. Quant Theme."""
    if players is None: players = [f"P{i}" for i in range(11)]
        
    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return []
        return [1] + [int(x) for x in fmt_str.split("-")]
        
    lines = parse_formation(formation)
    W, H = 160, 200
    
    svg = f"""<svg width="100%" style="max-width: 600px; display: block; margin: 0 auto;" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <circle id="jersey-vert" cx="0" cy="0" r="5" stroke="#1e293b" stroke-width="1.5"/>
            <circle id="jersey-ghost" cx="0" cy="0" r="5" stroke="#64748b" stroke-dasharray="2,2" stroke-width="1" fill="none"/>
        </defs>
        <rect width="{W}" height="{H}" fill="#111827" rx="4" ry="4" stroke="#1e293b" stroke-width="2"/>
        <rect x="5" y="5" width="150" height="190" fill="none" stroke="#334155" stroke-width="0.5"/>
        <line x1="5" y1="195" x2="155" y2="195" stroke="#334155" stroke-width="0.5"/>
        <circle cx="80" cy="195" r="20" fill="none" stroke="#334155" stroke-width="0.5"/>
        <circle cx="80" cy="195" r="1.5" fill="#334155" />
        <rect x="35" y="5" width="90" height="30" fill="none" stroke="#334155" stroke-width="0.5"/>
        <rect x="60" y="5" width="40" height="10" fill="none" stroke="#334155" stroke-width="0.5"/>
        <line x1="72" y1="5" x2="88" y2="5" stroke="#64748b" stroke-width="2" />
    """
    
    if not players:
        svg += f"""<text x="80" y="100" fill="#64748b" font-family="'JetBrains Mono', monospace" font-size="8px" text-anchor="middle">AWAITING ROSTER SYNC</text></svg>"""
        return svg
        
    if lines:
        y_steps = len(lines)
        p_idx = 0
        for row_idx, num_players in enumerate(lines):
            y = 20 + (160 / max(1, y_steps - 1)) * row_idx
            for col_idx in range(num_players):
                x = 10 + (140 / (num_players + 1)) * (col_idx + 1)
                name = players[p_idx] if p_idx < len(players) else ""
                # Draw Ghost (Ideal XI) slightly offset
                svg += f'<use href="#jersey-ghost" x="{x+4}" y="{y-4}" />'
                # Draw Starter
                svg += f"""
                    <a href="/player?name={name}&team={team_name}" style="cursor: pointer;">
                        <use href="#jersey-vert" x="{x}" y="{y}" fill="{team_color}" />
                        <text x="{x}" y="{y+9}" fill="#f8fafc" font-family="'JetBrains Mono', monospace" font-size="5px" text-anchor="middle">{name}</text>
                    </a>
                """
                p_idx += 1
                
    svg += "</svg>"
    return svg


def generate_pitch_svg_horizontal(home_formation="4-3-3", away_formation="4-2-3-1", home_color="#14b8a6", away_color="#f43f5e", home_players=None, away_players=None, home_team="Home", away_team="Away"):
    """Generates a native horizontal full-pitch. Quant Theme."""
    W, H = 160, 100
    
    svg = f"""<svg width="100%" style="max-width: 100%; display: block; margin: 0 auto; transition: all 0.3s ease;" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <circle id="jersey-home" cx="0" cy="0" r="3" fill="#14b8a6" stroke="#0b0f19" stroke-width="0.8"/>
            <circle id="jersey-away" cx="0" cy="0" r="3" fill="#f43f5e" stroke="#0b0f19" stroke-width="0.8"/>
            <filter id="glow-h" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="1.5" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="glow-a" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="1.5" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        </defs>
        <rect width="{W}" height="{H}" fill="#111827" rx="4" ry="4" stroke="#1e293b" stroke-width="1.5"/>
        <rect x="5" y="5" width="150" height="90" fill="none" stroke="#334155" stroke-width="0.5"/>
        
        <line x1="80" y1="5" x2="80" y2="95" stroke="#334155" stroke-width="0.5"/>
        <circle cx="80" cy="50" r="15" fill="none" stroke="#334155" stroke-width="0.5"/>
        <circle cx="80" cy="50" r="1" fill="#334155" />
        
        <rect x="5" y="22" width="22" height="56" fill="none" stroke="#334155" stroke-width="0.5"/>
        <rect x="133" y="22" width="22" height="56" fill="none" stroke="#334155" stroke-width="0.5"/>
        <rect x="5" y="37" width="8" height="26" fill="none" stroke="#334155" stroke-width="0.5"/>
        <rect x="147" y="37" width="8" height="26" fill="none" stroke="#334155" stroke-width="0.5"/>
        <line x1="5" y1="44" x2="5" y2="56" stroke="#64748b" stroke-width="1.5" />
        <line x1="155" y1="44" x2="155" y2="56" stroke="#64748b" stroke-width="1.5" />
    """
    
    if not home_players and not away_players:
        svg += f"""<text x="80" y="50" fill="#64748b" font-family="'JetBrains Mono', monospace" font-size="6px" text-anchor="middle">AWAITING LIVE DATA SYNC</text></svg>"""
        return svg
        
    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return []
        return [1] + [int(x) for x in fmt_str.split("-")]
        
    h_lines = parse_formation(home_formation)
    a_lines = parse_formation(away_formation)
    
    # Home
    if h_lines and home_players:
        x_steps = len(h_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(h_lines):
            x = 15 + (60 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 10 + (80 / (num_players + 1)) * (row_idx + 1)
                name = home_players[p_idx] if p_idx < len(home_players) else ""
                glow = 'filter="url(#glow-h)"' if (p_idx % 3 == 0) else '' # Mock heat rating
                svg += f"""
                    <a href="/player?name={name}&team={home_team}" style="cursor: pointer;">
                        <circle cx="{x}" cy="{y}" r="3.5" fill="#14b8a6" stroke="#0b0f19" stroke-width="0.8" {glow}/>
                        <text x="{x}" y="{y+6}" fill="#94a3b8" font-family="'JetBrains Mono', monospace" font-size="2.8px" text-anchor="middle">{name}</text>
                        <text x="{x+4.5}" y="{y-2}" fill="#14b8a6" font-family="'JetBrains Mono', monospace" font-size="2.2px" text-anchor="start">{round(0.05 + p_idx*0.02, 2)} xT</text>
                    </a>
                """
                p_idx += 1

    # Away
    if a_lines and away_players:
        x_steps = len(a_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(a_lines):
            x = 145 - (60 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 10 + (80 / (num_players + 1)) * (row_idx + 1)
                name = away_players[p_idx] if p_idx < len(away_players) else ""
                glow = 'filter="url(#glow-a)"' if (p_idx % 4 == 0) else ''
                svg += f"""
                    <a href="/player?name={name}&team={away_team}" style="cursor: pointer;">
                        <circle cx="{x}" cy="{y}" r="3.5" fill="#f43f5e" stroke="#0b0f19" stroke-width="0.8" {glow}/>
                        <text x="{x}" y="{y+6}" fill="#94a3b8" font-family="'JetBrains Mono', monospace" font-size="2.8px" text-anchor="middle">{name}</text>
                    </a>
                """
                p_idx += 1
            
    svg += "</svg>"
    return svg
