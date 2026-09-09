"""
utils.py  —  V4.2
==================
Team themes, default squads, formation-aware SVG pitch generation.

TEAM_THEMES: real kit colours for all 20 PL teams + Big 5 clubs.
             get_theme_for_team() falls back to a DETERMINISTIC hash
             (using hashlib, not Python's randomised built-in hash())
             so colours are always the same across restarts.

DEFAULT_SQUADS: typical starting XI for all 20 PL teams, in position
                order matching the formation (GK first, then DEF, MID, FWD).
                Labelled "typical XI" in the UI -- not confirmed lineups.
                Update manually when significant squad changes happen.

DEFAULT_FORMATIONS: each team's most commonly used shape this season.
"""

import hashlib
import json

# ── League team lists ─────────────────────────────────────────────────────────
LEAGUE_TEAMS = {
    "Premier League": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
        "Ipswich", "Leeds", "Leicester", "Liverpool", "Manchester City",
        "Manchester United", "Newcastle United", "Nottingham Forest",
        "Southampton", "Sunderland", "Tottenham", "West Ham", "Wolverhampton Wanderers",
    ],
    "La Liga": [
        "Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Club",
        "Real Sociedad", "Villarreal", "Real Betis", "Valencia", "Sevilla",
        "Osasuna", "Celta Vigo", "Girona", "Mallorca", "Getafe",
        "Las Palmas", "Rayo Vallecano", "Espanyol", "Alaves", "Valladolid", "Leganes",
    ],
    "Serie A": [
        "Inter Milan", "AC Milan", "Juventus", "Atalanta", "Bologna",
        "Roma", "Lazio", "Fiorentina", "Napoli", "Torino",
        "Genoa", "Monza", "Lecce", "Empoli", "Udinese",
        "Hellas Verona", "Cagliari", "Como", "Parma", "Venezia",
    ],
    "Bundesliga": [
        "Bayer Leverkusen", "Bayern Munich", "VfB Stuttgart", "RB Leipzig",
        "Borussia Dortmund", "Eintracht Frankfurt", "Hoffenheim", "Freiburg",
        "Heidenheim", "Augsburg", "Werder Bremen", "Wolfsburg",
        "Borussia Monchengladbach", "Bochum", "Union Berlin",
        "Mainz 05", "St. Pauli", "Holstein Kiel",
    ],
    "Ligue 1": [
        "Paris Saint Germain", "Monaco", "Marseille", "Lille", "Lens",
        "Nice", "Lyon", "Rennes", "Reims", "Toulouse",
        "Montpellier", "Strasbourg", "Nantes", "Le Havre", "Brest",
        "Auxerre", "Angers", "Saint-Etienne",
    ],
}

TEAM_MANAGERS = {
    "Arsenal"             : "Mikel Arteta",
    "Manchester City"     : "Pep Guardiola",
    "Liverpool"           : "Arne Slot",
    "Manchester United"   : "Ruben Amorim",
    "Chelsea"             : "Enzo Maresca",
    "Tottenham"           : "Ange Postecoglou",
    "Aston Villa"         : "Unai Emery",
    "Newcastle United"    : "Eddie Howe",
    "West Ham"            : "Julen Lopetegui",
    "Brighton"            : "Fabian Hurzeler",
    "Brentford"           : "Thomas Frank",
    "Fulham"              : "Marco Silva",
    "Wolverhampton Wanderers": "Gary O'Neil",
    "Crystal Palace"      : "Oliver Glasner",
    "Nottingham Forest"   : "Nuno Espirito Santo",
    "Everton"             : "Sean Dyche",
    "Bournemouth"         : "Andoni Iraola",
    "Ipswich"             : "Kieran McKenna",
    "Leicester"           : "Steve Cooper",
    "Southampton"         : "Russell Martin",
    "Real Madrid"         : "Carlo Ancelotti",
    "Barcelona"           : "Hansi Flick",
    "Atletico Madrid"     : "Diego Simeone",
    "Juventus"            : "Thiago Motta",
    "AC Milan"            : "Paulo Fonseca",
    "Inter Milan"         : "Simone Inzaghi",
    "Bayern Munich"       : "Vincent Kompany",
    "Borussia Dortmund"   : "Nuri Sahin",
    "Bayer Leverkusen"    : "Xabi Alonso",
    "Paris Saint Germain" : "Luis Enrique",
    "Marseille"           : "Roberto De Zerbi",
}

# ── Team colours (real kit primaries) ─────────────────────────────────────────
# All 20 PL teams + major Big 5 clubs.
# primary = dominant kit colour shown on the pitch dots.
# secondary = accent / away colour.
# bg = subtle page tint when this team is active.
TEAM_THEMES = {
    # ── Premier League ─────────────────────────────────────────────────────────
    "Arsenal"                  : {"primary": "#EF0107", "secondary": "#023474", "bg": "#1A0505"},
    "Aston Villa"              : {"primary": "#670E36", "secondary": "#95BFE5", "bg": "#110209"},
    "Bournemouth"              : {"primary": "#DA291C", "secondary": "#000000", "bg": "#1A0505"},
    "Brentford"                : {"primary": "#D40000", "secondary": "#FBB800", "bg": "#1A0505"},
    "Brighton"                 : {"primary": "#0057B8", "secondary": "#FFFFFF", "bg": "#021529"},
    "Burnley"                  : {"primary": "#6C1D45", "secondary": "#99D6EA", "bg": "#120208"},
    "Chelsea"                  : {"primary": "#034694", "secondary": "#DBA111", "bg": "#04142B"},
    "Crystal Palace"           : {"primary": "#1B458F", "secondary": "#C4122E", "bg": "#030D1E"},
    "Everton"                  : {"primary": "#003399", "secondary": "#FFFFFF", "bg": "#00092B"},
    "Fulham"                   : {"primary": "#FFFFFF", "secondary": "#000000", "bg": "#111111"},
    "Ipswich"                  : {"primary": "#0044A9", "secondary": "#FFFFFF", "bg": "#001430"},
    "Leeds"                    : {"primary": "#FFFFFF", "secondary": "#FFCD00", "bg": "#111111"},
    "Leicester"                : {"primary": "#003090", "secondary": "#FDBE11", "bg": "#000C28"},
    "Liverpool"                : {"primary": "#C8102E", "secondary": "#00B2A9", "bg": "#180205"},
    "Manchester City"          : {"primary": "#6CABDD", "secondary": "#1C2C5B", "bg": "#0B1320"},
    "Manchester United"        : {"primary": "#DA291C", "secondary": "#FBE122", "bg": "#1C0A0A"},
    "Newcastle United"         : {"primary": "#241F20", "secondary": "#FFFFFF", "bg": "#111111"},
    "Nottingham Forest"        : {"primary": "#DD0000", "secondary": "#FFFFFF", "bg": "#1A0000"},
    "Southampton"              : {"primary": "#D71920", "secondary": "#130C0E", "bg": "#1A0000"},
    "Sunderland"               : {"primary": "#EB172B", "secondary": "#000000", "bg": "#1A0005"},
    "Tottenham"                : {"primary": "#132257", "secondary": "#FFFFFF", "bg": "#02040A"},
    "West Ham"                 : {"primary": "#7A263A", "secondary": "#1BB1E7", "bg": "#110005"},
    "Wolverhampton Wanderers"  : {"primary": "#FDB913", "secondary": "#231F20", "bg": "#1A1200"},

    # ── La Liga ────────────────────────────────────────────────────────────────
    "Real Madrid"              : {"primary": "#FFFFFF", "secondary": "#00529F", "bg": "#0F1626"},
    "Barcelona"                : {"primary": "#A50044", "secondary": "#004D98", "bg": "#0A1128"},
    "Atletico Madrid"          : {"primary": "#CB3524", "secondary": "#272E61", "bg": "#140503"},

    # ── Serie A ────────────────────────────────────────────────────────────────
    "Juventus"                 : {"primary": "#FFFFFF", "secondary": "#000000", "bg": "#111111"},
    "AC Milan"                 : {"primary": "#FB090B", "secondary": "#000000", "bg": "#1A0101"},
    "Inter Milan"              : {"primary": "#010E80", "secondary": "#FFFFFF", "bg": "#010214"},
    "Napoli"                   : {"primary": "#12A0C3", "secondary": "#FFFFFF", "bg": "#031620"},

    # ── Bundesliga ─────────────────────────────────────────────────────────────
    "Bayern Munich"            : {"primary": "#DC052D", "secondary": "#0066B2", "bg": "#1C0005"},
    "Borussia Dortmund"        : {"primary": "#FDE100", "secondary": "#000000", "bg": "#1A1700"},
    "Bayer Leverkusen"         : {"primary": "#E32221", "secondary": "#000000", "bg": "#170303"},
    "RB Leipzig"               : {"primary": "#DD0741", "secondary": "#FFFFFF", "bg": "#170010"},

    # ── Ligue 1 ────────────────────────────────────────────────────────────────
    "Paris Saint Germain"      : {"primary": "#004170", "secondary": "#DA291C", "bg": "#020D1A"},
    "Marseille"                : {"primary": "#2FAEE0", "secondary": "#FFFFFF", "bg": "#05161C"},

    # ── Default fallback ───────────────────────────────────────────────────────
    "Default"                  : {"primary": "#14b8a6", "secondary": "#0f172a", "bg": "#0b0f19"},
}


def get_theme_for_team(team_name: str) -> dict:
    if team_name in TEAM_THEMES:
        return TEAM_THEMES[team_name]
    # Deterministic colour from team name -- uses hashlib (not Python's
    # randomised built-in hash()) so the same colour every restart.
    h = int(hashlib.md5(team_name.encode()).hexdigest(), 16)
    r = max(80, (h >> 16) & 0xFF)
    g = max(80, (h >> 8)  & 0xFF)
    b = max(80,  h        & 0xFF)
    return {
        "primary"  : f"#{r:02x}{g:02x}{b:02x}",
        "secondary": "#FFFFFF",
        "bg"       : f"#{r//8:02x}{g//8:02x}{b//8:02x}",
    }


# ── Default formations ────────────────────────────────────────────────────────
# Each team's most commonly used shape this season.
# Format: "DEF-MID-FWD" (GK always assumed as +1).
DEFAULT_FORMATIONS = {
    "Arsenal"                  : "4-3-3",
    "Aston Villa"              : "4-2-3-1",
    "Bournemouth"              : "4-2-3-1",
    "Brentford"                : "4-3-3",
    "Brighton"                 : "4-2-3-1",
    "Burnley"                  : "4-4-2",
    "Chelsea"                  : "4-2-3-1",
    "Crystal Palace"           : "4-3-3",
    "Everton"                  : "4-4-2",
    "Fulham"                   : "4-2-3-1",
    "Ipswich"                  : "4-2-3-1",
    "Leeds"                    : "4-3-3",
    "Leicester"                : "4-2-3-1",
    "Liverpool"                : "4-3-3",
    "Manchester City"          : "4-2-3-1",
    "Manchester United"        : "3-4-2-1",
    "Newcastle United"         : "4-3-3",
    "Nottingham Forest"        : "4-2-3-1",
    "Southampton"              : "4-4-2",
    "Sunderland"               : "4-2-3-1",
    "Tottenham"                : "4-3-3",
    "West Ham"                 : "4-2-3-1",
    "Wolverhampton Wanderers"  : "4-3-3",
    # Big 5 clubs
    "Real Madrid"              : "4-3-3",
    "Barcelona"                : "4-3-3",
    "Atletico Madrid"          : "4-4-2",
    "Bayern Munich"            : "4-2-3-1",
    "Borussia Dortmund"        : "4-2-3-1",
    "Bayer Leverkusen"         : "3-4-2-1",
    "Paris Saint Germain"      : "4-3-3",
}


# ── Default squads (typical XI, position order: GK, DEF..., MID..., FWD...) ──
# Ordered to match the formation in DEFAULT_FORMATIONS.
# Update manually when major squad changes happen.
DEFAULT_SQUADS = {
    "Arsenal"        : ["Raya", "White", "Saliba", "Gabriel", "Timber",
                        "Partey", "Rice", "Odegaard",
                        "Saka", "Havertz", "Martinelli"],
    "Aston Villa"    : ["Martinez", "Cash", "Konsa", "Torres", "Digne",
                        "Onana", "Tielemans",
                        "McGinn", "Bailey", "Watkins", "Rogers"],
    "Bournemouth"    : ["Flekken", "Smith", "Zabarnyi", "Senesi", "Kerkez",
                        "Cook", "Lerma",
                        "Semenyo", "Kluivert", "Ouattara", "Evanilson"],
    "Brentford"      : ["Flekken", "Roerslev", "Collins", "Pinnock", "Henry",
                        "Norgaard", "Janelt", "Jensen",
                        "Schade", "Wissa", "Mbeumo"],
    "Brighton"       : ["Verbruggen", "Veltman", "Dunk", "van Hecke", "Estupinan",
                        "Gross", "Gilmour",
                        "Minteh", "Joao Pedro", "Adingra", "Welbeck"],
    "Burnley"        : ["Trafford", "Roberts", "Beyer", "Taylor", "Maatsen",
                        "Brownhill", "Cork", "Cullen", "Berge",
                        "Rodriguez", "Weghorst"],
    "Chelsea"        : ["Sanchez", "Gusto", "Fofana", "Colwill", "Cucurella",
                        "Caicedo", "Fernandez",
                        "Madueke", "Palmer", "Nkunku", "Jackson"],
    "Crystal Palace" : ["Henderson", "Clyne", "Guehi", "Andersen", "Mitchell",
                        "Eze", "Hughes", "Wharton",
                        "Schlupp", "Mateta", "Olise"],
    "Everton"        : ["Pickford", "Coleman", "Tarkowski", "Branthwaite", "Mykolenko",
                        "Gueye", "Onana", "Doucoure", "McNeil",
                        "Calvert-Lewin", "Beto"],
    "Fulham"         : ["Leno", "Tete", "Diop", "Ream", "Robinson",
                        "Lukic", "Reed",
                        "Andreas", "Wilson", "Vinicius", "Iwobi"],
    "Ipswich"        : ["Flaherty", "Walton", "Woolfenden", "O'Shea", "Davis",
                        "Morsy", "Luongo",
                        "Hutchinson", "Chaplin", "Hirst", "Broadhead"],
    "Leeds"          : ["Meslier", "Ayling", "Cooper", "Struijk", "Firpo",
                        "Adams", "Roca", "Gruev",
                        "Gnonto", "Bamford", "Rodriguez"],
    "Leicester"      : ["Ward", "Justin", "Faes", "Evans", "Thomas",
                        "Ndidi", "Winks",
                        "Daka", "Tielemans", "Vardy", "Iheanacho"],
    "Liverpool"      : ["Alisson", "Alexander-Arnold", "Konate", "Van Dijk", "Robertson",
                        "Gravenberch", "Mac Allister", "Szoboszlai",
                        "Salah", "Nunez", "Diaz"],
    "Manchester City": ["Ederson", "Walker", "Dias", "Akanji", "Gvardiol",
                        "Rodri", "De Bruyne",
                        "Bernardo", "Doku", "Haaland", "Foden"],
    "Manchester United": ["Onana", "Dalot", "Maguire", "Martinez", "Shaw",
                          "Casemiro", "Mainoo",
                          "Fernandes", "Rashford",
                          "Garnacho", "Hojlund"],
    "Newcastle United": ["Pope", "Trippier", "Schar", "Botman", "Hall",
                         "Guimaraes", "Longstaff", "Tonali",
                         "Murphy", "Isak", "Gordon"],
    "Nottingham Forest": ["Turner", "Aina", "Murillo", "Milenkovic", "Williams",
                          "Yates", "Mangala",
                          "Elanga", "Gibbs-White", "Hudson-Odoi", "Awoniyi"],
    "Southampton"    : ["Bazunu", "Walker-Peters", "Harwood-Bellis", "Bednarek", "Manning",
                        "Ward-Prowse", "Romeo", "Ugochukwu", "Smallbone",
                        "Armstrong", "Archer"],
    "Sunderland"     : ["Patterson", "Hume", "Batth", "O'Nien", "Cirkin",
                        "Neil", "Evans",
                        "Amad", "Clarke", "Stewart", "Roberts"],
    "Tottenham"      : ["Vicario", "Pedro Porro", "Romero", "Van de Ven", "Udogie",
                        "Bissouma", "Sarr", "Maddison",
                        "Son", "Johnson", "Kulusevski"],
    "West Ham"       : ["Fabianski", "Coufal", "Zouma", "Aguerd", "Emerson",
                        "Soucek", "Rice",
                        "Paqueta", "Bowen", "Antonio", "Kudus"],
    "Wolverhampton Wanderers": ["Sa", "Semedo", "Dawson", "Toti", "Ait-Nouri",
                                "Lemina", "Joao Gomes", "Neves",
                                "Hwang", "Cunha", "Sarabia"],
    # Big 5
    "Real Madrid"    : ["Courtois", "Carvajal", "Militao", "Rudiger", "Mendy",
                        "Valverde", "Tchouameni", "Bellingham",
                        "Rodrygo", "Vinicius Jr", "Mbappe"],
    "Barcelona"      : ["Ter Stegen", "Kounde", "Cubarsi", "Inigo Martinez", "Balde",
                        "Casado", "Pedri", "De Jong",
                        "Yamal", "Lewandowski", "Raphinha"],
    "Atletico Madrid": ["Oblak", "Molina", "Gimenez", "Witsel", "Reinildo",
                        "Koke", "De Paul", "Saul", "Llorente",
                        "Griezmann", "Morata"],
    "Bayern Munich"  : ["Neuer", "Kimmich", "Upamecano", "De Ligt", "Davies",
                        "Goretzka", "Laimer",
                        "Muller", "Sane", "Kane", "Gnabry"],
    "Borussia Dortmund": ["Kobel", "Ryerson", "Hummels", "Schlotterbeck", "Maatsen",
                          "Can", "Kramer",
                          "Brandt", "Reus", "Adeyemi", "Fullkrug"],
    "Paris Saint Germain": ["Donnarumma", "Hakimi", "Marquinhos", "Skriniar", "Hernandez",
                            "Ugarte", "Fabian", "Vitinha",
                            "Dembele", "Mbappe", "Barcola"],
}


def get_squad_for_team(team_name: str) -> list[str]:
    if team_name in DEFAULT_SQUADS:
        return DEFAULT_SQUADS[team_name]
    return [f"P{i+1}" for i in range(11)]


def get_formation_for_team(team_name: str) -> str:
    return DEFAULT_FORMATIONS.get(team_name, "4-3-3")


# ── SVG pitch generators ──────────────────────────────────────────────────────

def generate_pitch_svg_vertical(formation="4-3-3", team_color="#14b8a6",
                                players=None, team_name="Default"):
    """Vertical half-pitch for the Team Profile page."""
    if players is None or len(players) == 0:
        players = get_squad_for_team(team_name)

    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return [1, 4, 3, 3]
        return [1] + [int(x) for x in fmt_str.split("-")]

    lines = parse_formation(formation)
    W, H = 160, 200

    svg = (f'<svg width="100%" style="max-width:450px;display:block;margin:0 auto;"'
           f' viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
           '<defs>'
           '<circle id="jersey-vert" cx="0" cy="0" r="5" stroke="#1e293b" stroke-width="1.2"/>'
           '<circle id="jersey-ghost" cx="0" cy="0" r="5" stroke="#64748b"'
           ' stroke-dasharray="2,2" stroke-width="1" fill="none"/>'
           '</defs>'
           f'<rect width="{W}" height="{H}" fill="#0b0f19" rx="6" ry="6"'
           ' stroke="#1e293b" stroke-width="1.5"/>'
           f'<rect x="6" y="6" width="148" height="188" fill="none"'
           ' stroke="#1e293b" stroke-width="0.75"/>'
           '<rect x="35" y="6" width="90" height="32" fill="none"'
           ' stroke="#1e293b" stroke-width="0.75"/>'
           '<rect x="58" y="6" width="44" height="12" fill="none"'
           ' stroke="#1e293b" stroke-width="0.75"/>'
           '<circle cx="80" cy="24" r="1.5" fill="#334155"/>'
           '<line x1="6" y1="194" x2="154" y2="194" stroke="#1e293b" stroke-width="0.75"/>'
           '<path d="M 60 194 A 20 20 0 0 1 100 194" fill="none"'
           ' stroke="#1e293b" stroke-width="0.75"/>'
           '<circle cx="80" cy="194" r="1.5" fill="#334155"/>')

    if lines:
        y_steps = len(lines)
        p_idx = 0
        for row_idx, num_players in enumerate(lines):
            y = 22 + (150 / max(1, y_steps - 1)) * row_idx
            for col_idx in range(num_players):
                x = 12 + (136 / (num_players + 1)) * (col_idx + 1)
                name = players[p_idx] if p_idx < len(players) else f"P{p_idx+1}"
                svg += f'<use href="#jersey-ghost" x="{x+3}" y="{y-3}"/>'
                svg += (f'<a href="/player?name={name}&team={team_name}"'
                        f' style="cursor:pointer;">'
                        f'<use href="#jersey-vert" x="{x}" y="{y}" fill="{team_color}"/>'
                        f'<text x="{x}" y="{y+9}" fill="#94a3b8"'
                        f' font-family="\'JetBrains Mono\',monospace"'
                        f' font-size="4.5px" text-anchor="middle">{name}</text>'
                        f'</a>')
                p_idx += 1

    svg += "</svg>"
    return svg


def generate_pitch_svg_horizontal(home_formation="4-3-3", away_formation="4-3-3",
                                  home_color="#14b8a6", away_color="#f43f5e",
                                  home_players=None, away_players=None,
                                  home_team="Home", away_team="Away"):
    """Full horizontal pitch for the Live Match page."""
    if home_players is None or len(home_players) == 0:
        home_players = get_squad_for_team(home_team)
    if away_players is None or len(away_players) == 0:
        away_players = get_squad_for_team(away_team)

    W, H = 160, 100

    svg = (f'<svg width="100%" style="max-width:100%;display:block;margin:0 auto;"'
           f' viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
           '<defs>'
           '<filter id="glow-h" x="-50%" y="-50%" width="200%" height="200%">'
           '<feGaussianBlur stdDeviation="1.2" result="blur"/>'
           '<feComposite in="SourceGraphic" in2="blur" operator="over"/>'
           '</filter>'
           '<filter id="glow-a" x="-50%" y="-50%" width="200%" height="200%">'
           '<feGaussianBlur stdDeviation="1.2" result="blur"/>'
           '<feComposite in="SourceGraphic" in2="blur" operator="over"/>'
           '</filter>'
           '</defs>'
           f'<rect width="{W}" height="{H}" fill="#0b0f19" rx="6" ry="6"'
           ' stroke="#1e293b" stroke-width="1.5"/>'
           '<rect x="5" y="5" width="150" height="90" fill="none"'
           ' stroke="#1e293b" stroke-width="0.75"/>'
           '<line x1="80" y1="5" x2="80" y2="95" stroke="#1e293b" stroke-width="0.75"/>'
           '<circle cx="80" cy="50" r="14" fill="none" stroke="#1e293b" stroke-width="0.75"/>'
           '<circle cx="80" cy="50" r="1" fill="#334155"/>'
           '<rect x="5" y="22" width="22" height="56" fill="none"'
           ' stroke="#1e293b" stroke-width="0.75"/>'
           '<rect x="5" y="36" width="8" height="28" fill="none"'
           ' stroke="#1e293b" stroke-width="0.75"/>'
           '<circle cx="16" cy="50" r="1" fill="#334155"/>'
           '<rect x="133" y="22" width="22" height="56" fill="none"'
           ' stroke="#1e293b" stroke-width="0.75"/>'
           '<rect x="147" y="36" width="8" height="28" fill="none"'
           ' stroke="#1e293b" stroke-width="0.75"/>'
           '<circle cx="144" cy="50" r="1" fill="#334155"/>')

    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return [1, 4, 3, 3]
        return [1] + [int(x) for x in fmt_str.split("-")]

    h_lines = parse_formation(home_formation)
    a_lines = parse_formation(away_formation)

    if h_lines and home_players:
        x_steps = len(h_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(h_lines):
            x = 12 + (58 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 8 + (84 / (num_players + 1)) * (row_idx + 1)
                name = home_players[p_idx] if p_idx < len(home_players) else f"H{p_idx+1}"
                glow = 'filter="url(#glow-h)"' if (p_idx % 3 == 0) else ""
                svg += (f'<a href="/player?name={name}&team={home_team}"'
                        f' style="cursor:pointer;">'
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2"'
                        f' fill="{home_color}" stroke="#0b0f19" stroke-width="0.8" {glow}/>'
                        f'<text x="{x:.1f}" y="{y+6:.1f}" fill="#94a3b8"'
                        f' font-family="\'JetBrains Mono\',monospace"'
                        f' font-size="2.6px" text-anchor="middle">{name}</text>'
                        f'</a>')
                p_idx += 1

    if a_lines and away_players:
        x_steps = len(a_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(a_lines):
            x = 148 - (58 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 8 + (84 / (num_players + 1)) * (row_idx + 1)
                name = away_players[p_idx] if p_idx < len(away_players) else f"A{p_idx+1}"
                glow = 'filter="url(#glow-a)"' if (p_idx % 4 == 0) else ""
                svg += (f'<a href="/player?name={name}&team={away_team}"'
                        f' style="cursor:pointer;">'
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2"'
                        f' fill="{away_color}" stroke="#0b0f19" stroke-width="0.8" {glow}/>'
                        f'<text x="{x:.1f}" y="{y+6:.1f}" fill="#94a3b8"'
                        f' font-family="\'JetBrains Mono\',monospace"'
                        f' font-size="2.6px" text-anchor="middle">{name}</text>'
                        f'</a>')
                p_idx += 1

    svg += "</svg>"
    return svg
