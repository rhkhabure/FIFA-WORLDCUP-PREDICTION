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


# ── Team crest URLs (football-data.org static CDN, no auth needed) ────────────
_CREST_IDS: dict[str, int] = {
    # Verified from football-data.org /competitions/PL/teams?season=2025
    "Arsenal"                 : 57,
    "Aston Villa"             : 58,
    "Chelsea"                 : 61,
    "Everton"                 : 62,
    "Fulham"                  : 63,
    "Liverpool"               : 64,
    "Manchester City"         : 65,
    "Manchester United"       : 66,
    "Newcastle United"        : 67,
    "Sunderland"              : 71,
    "Tottenham"               : 73,
    "Wolverhampton Wanderers" : 76,
    "Burnley"                 : 328,
    "Leeds"                   : 341,
    "Nottingham Forest"       : 351,
    "Crystal Palace"          : 354,
    "Brighton"                : 397,
    "Brentford"               : 402,
    "West Ham"                : 563,
    # Bournemouth uses a name slug not an ID on the CDN
    # handled separately in get_crest_url below
    "Bournemouth"             : 1044,
    # Teams missing from 2025/26 API list -- use best known IDs
    "Ipswich"                 : 610,
    "Leicester"               : 338,
    "Southampton"             : 340,
    # Hull City and Coventry are Championship sides, no PL crest
    # Big 5 extras
    "Real Madrid"             : 86,
    "Barcelona"               : 81,
    "Atletico Madrid"         : 78,
    "Bayern Munich"           : 5,
    "Borussia Dortmund"       : 4,
    "Paris Saint Germain"     : 524,
}
_CREST_BASE = "https://crests.football-data.org"

# Special cases where the CDN uses a name slug instead of numeric ID
_CREST_SLUGS: dict[str, str] = {
    "Bournemouth": "bournemouth",
}


def get_crest_url(team_name: str, fallback_url: str = "") -> str:
    """
    Returns the football-data.org CDN crest URL for a team.
    Checks slug overrides first, then numeric IDs, then fallback_url.
    """
    if team_name in _CREST_SLUGS:
        return f"{_CREST_BASE}/{_CREST_SLUGS[team_name]}.png"
    tid = _CREST_IDS.get(team_name)
    if tid:
        return f"{_CREST_BASE}/{tid}.png"
    return fallback_url


def get_crest_proxy_url(team_name: str) -> str:
    """
    Returns the local proxy URL for a team crest.
    Use this in SVG <image> tags to avoid CORS issues.
    Returns "" if the team has no known crest.
    """
    if team_name in _CREST_SLUGS or team_name in _CREST_IDS:
        # URL-encode spaces
        safe_name = team_name.replace(" ", "%20")
        return f"/crest/{safe_name}"
    return ""
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
    # Try direct match first, then common aliases
    if team_name in DEFAULT_SQUADS:
        return DEFAULT_SQUADS[team_name]
    # Common API name -> short name mappings
    _SQUAD_ALIASES = {
        "West Ham United"        : "West Ham",
        "Leeds United"           : "Leeds",
        "Newcastle United"       : "Newcastle United",
        "Nottingham Forest"      : "Nottingham Forest",
        "Wolverhampton Wanderers": "Wolverhampton Wanderers",
        "Tottenham Hotspur"      : "Tottenham",
        "Leicester City"         : "Leicester",
        "Manchester United"      : "Manchester United",
        "Manchester City"        : "Manchester City",
        "Nott'm Forest"          : "Nottingham Forest",
        "Ipswich Town"           : "Ipswich",
    }
    canonical = _SQUAD_ALIASES.get(team_name, team_name)
    return DEFAULT_SQUADS.get(canonical, [f"P{i+1}" for i in range(11)])


def get_formation_for_team(team_name: str) -> str:
    if team_name in DEFAULT_FORMATIONS:
        return DEFAULT_FORMATIONS[team_name]
    _FORMATION_ALIASES = {
        "West Ham United"        : "West Ham",
        "Leeds United"           : "Leeds",
        "Tottenham Hotspur"      : "Tottenham",
        "Leicester City"         : "Leicester",
        "Nott'm Forest"          : "Nottingham Forest",
        "Ipswich Town"           : "Ipswich",
    }
    canonical = _FORMATION_ALIASES.get(team_name, team_name)
    return DEFAULT_FORMATIONS.get(canonical, "4-3-3")


# ── SVG pitch generators ──────────────────────────────────────────────────────

def generate_pitch_svg_vertical(formation="4-3-3", team_color="#14b8a6",
                                players=None, team_name="Default"):
    """
    Attacking half-pitch for the Team Profile page.
    Shows GK → halfway line only (top half of pitch).
    Green turf, white markings, faded crest watermark.
    """
    if players is None or len(players) == 0:
        players = get_squad_for_team(team_name)

    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return [1, 4, 3, 3]
        return [1] + [int(x) for x in fmt_str.split("-")]

    lines = parse_formation(formation)
    W, H = 200, 320   # taller: fills centre column above last 5 widget

    crest_url = get_crest_proxy_url(team_name)

    svg = (
        f'<svg width="100%" style="max-width:480px;display:block;margin:0 auto;"'
        f' viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
        '<defs>'
        f'<pattern id="grass-v" x="0" y="0" width="20" height="{H}"'
        ' patternUnits="userSpaceOnUse">'
        f'<rect width="10" height="{H}" fill="#1a6b2e"/>'
        f'<rect x="10" width="10" height="{H}" fill="#1d7533"/>'
        '</pattern>'
        '<clipPath id="pitch-clip-v">'
        f'<rect x="8" y="8" width="{W-16}" height="{H-8}" rx="4"/>'
        '</clipPath>'
        '</defs>'

        # Background + grass
        f'<rect width="{W}" height="{H}" fill="#1a6b2e" rx="6 6 0 0"/>'
        f'<rect x="8" y="8" width="{W-16}" height="{H}"'
        f' fill="url(#grass-v)" clip-path="url(#pitch-clip-v)"/>'
    )

    # Faded crest — larger, centred in the half
    if crest_url:
        svg += (
            f'<image href="{crest_url}"'
            f' x="{W//2 - 36}" y="{H//2 - 30}" width="72" height="72"'
            f' opacity="0.18" clip-path="url(#pitch-clip-v)"/>'
        )

    # Pitch markings (white)
    svg += (
        # Side and top boundary (no bottom line — open at halfway)
        f'<line x1="8" y1="8" x2="8" y2="{H}" stroke="white" stroke-width="1.2"/>'
        f'<line x1="{W-8}" y1="8" x2="{W-8}" y2="{H}" stroke="white" stroke-width="1.2"/>'
        f'<line x1="8" y1="8" x2="{W-8}" y2="8" stroke="white" stroke-width="1.2"/>'
        # Penalty area
        f'<rect x="{W//2 - 40}" y="8" width="80" height="44"'
        f' fill="none" stroke="white" stroke-width="1"/>'
        # Goal area
        f'<rect x="{W//2 - 18}" y="8" width="36" height="16"'
        f' fill="none" stroke="white" stroke-width="1"/>'
        # Penalty spot
        f'<circle cx="{W//2}" cy="34" r="2" fill="white"/>'
        # Goal
        f'<rect x="{W//2 - 18}" y="2" width="36" height="6"'
        f' fill="none" stroke="white" stroke-width="1"/>'
        # Halfway line (bottom of visible area)
        f'<line x1="8" y1="{H}" x2="{W-8}" y2="{H}"'
        f' stroke="white" stroke-width="1" stroke-dasharray="4,3"/>'
        # Centre circle arc (only the top half visible)
        f'<path d="M {W//2 - 28} {H} A 28 28 0 0 1 {W//2 + 28} {H}"'
        f' fill="none" stroke="white" stroke-width="1"/>'
        f'<circle cx="{W//2}" cy="{H}" r="2" fill="white"/>'
    )

    # Players — spread across the half-pitch
    if lines:
        y_steps = len(lines)
        p_idx = 0
        for row_idx, num_players in enumerate(lines):
            # GK near top (y=20), last row near halfway (y=H-14)
            y = 20 + ((H - 34) / max(1, y_steps - 1)) * row_idx
            for col_idx in range(num_players):
                x = 14 + ((W - 28) / (num_players + 1)) * (col_idx + 1)
                name = players[p_idx] if p_idx < len(players) else f"P{p_idx+1}"
                svg += (
                    f'<a href="/player?name={name}&team={team_name}"'
                    f' style="cursor:pointer;">'
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7"'
                    f' fill="{team_color}" stroke="white" stroke-width="1.5"/>'
                    f'<text x="{x:.1f}" y="{y + 13:.1f}" fill="white"'
                    f' font-family="\'JetBrains Mono\',monospace"'
                    f' font-size="6px" text-anchor="middle"'
                    f' style="text-shadow:0 0 3px rgba(0,0,0,0.9)">{name}</text>'
                    f'</a>'
                )
                p_idx += 1

    svg += "</svg>"
    return svg


def generate_pitch_svg_horizontal(home_formation="4-3-3", away_formation="4-3-3",
                                  home_color="#14b8a6", away_color="#f43f5e",
                                  home_players=None, away_players=None,
                                  home_team="Home", away_team="Away",
                                  home_crest_url: str = "",
                                  away_crest_url: str = "",
                                  h_score: int | None = None,
                                  a_score: int | None = None,
                                  status: str = ""):
    """
    Full horizontal pitch for the Live Match page.

    Changes from original:
    - Grass green pitch (#1a6b2e) instead of dark background
    - White pitch markings instead of slate
    - Translucent home-team crest watermarked at centre circle
      (only when home_crest_url is supplied -- graceful no-op otherwise)
    - Player dots now have a subtle drop-shadow glow
    - Player name text is white-on-dark for legibility on green
    """
    if home_players is None or len(home_players) == 0:
        home_players = get_squad_for_team(home_team)
    if away_players is None or len(away_players) == 0:
        away_players = get_squad_for_team(away_team)

    W, H = 320, 200   # doubled from 160x100 for better player label legibility

    svg = (
        f'<svg width="100%" style="max-width:100%;display:block;margin:0 auto;"'
        f' viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"'
        f' xmlns:xlink="http://www.w3.org/1999/xlink">'

        # ── Defs ──────────────────────────────────────────────────────────────
        '<defs>'
        # Subtle grass stripe pattern (alternating shades)
        '<pattern id="grass" x="0" y="0" width="20" height="200"'
        ' patternUnits="userSpaceOnUse">'
        '<rect width="10" height="200" fill="#1a6b2e"/>'
        '<rect x="10" width="10" height="200" fill="#1d7533"/>'
        '</pattern>'
        # Player glow filters
        '<filter id="glow-h" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="2" result="blur"/>'
        '<feComposite in="SourceGraphic" in2="blur" operator="over"/>'
        '</filter>'
        '<filter id="glow-a" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="2" result="blur"/>'
        '<feComposite in="SourceGraphic" in2="blur" operator="over"/>'
        '</filter>'
        # Clip path for rounded pitch boundary
        '<clipPath id="pitch-clip">'
        '<rect x="8" y="8" width="304" height="184" rx="4"/>'
        '</clipPath>'
        '</defs>'

        # ── Pitch surface ──────────────────────────────────────────────────────
        '<rect width="320" height="200" fill="#1a6b2e" rx="6" ry="6"/>'
        '<rect x="8" y="8" width="304" height="184" fill="url(#grass)" clip-path="url(#pitch-clip)"/>'

        # ── Home crest watermark (translucent, centred on pitch) ────────────
    )

    if home_crest_url:
        # Semi-transparent crest at the centre circle -- opacity 0.07 gives
        # a barely-there watermark visible enough to read, not so loud it
        # competes with the players
        svg += (
            f'<image href="{home_crest_url}" '
            f'x="{W//2 - 24}" y="{H//2 - 24}" width="48" height="48" '
            f'opacity="0.22" clip-path="url(#pitch-clip)"/>'
        )

    # ── Pitch markings (white) ────────────────────────────────────────────────
    svg += (
        # Outer boundary
        '<rect x="8" y="8" width="304" height="184" fill="none"'
        ' stroke="white" stroke-width="1.2" rx="4"/>'
        # Halfway line
        '<line x1="160" y1="8" x2="160" y2="192" stroke="white" stroke-width="1"/>'
        # Centre circle
        '<circle cx="160" cy="100" r="28" fill="none" stroke="white" stroke-width="1"/>'
        '<circle cx="160" cy="100" r="2" fill="white"/>'
        # Home penalty area (left)
        '<rect x="8" y="44" width="44" height="112" fill="none" stroke="white" stroke-width="1"/>'
        # Home goal area (left)
        '<rect x="8" y="72" width="16" height="56" fill="none" stroke="white" stroke-width="1"/>'
        # Home penalty spot
        '<circle cx="36" cy="100" r="2" fill="white"/>'
        # Home goal
        '<rect x="2" y="82" width="6" height="36" fill="none" stroke="white" stroke-width="1"/>'
        # Away penalty area (right)
        '<rect x="268" y="44" width="44" height="112" fill="none" stroke="white" stroke-width="1"/>'
        # Away goal area (right)
        '<rect x="296" y="72" width="16" height="56" fill="none" stroke="white" stroke-width="1"/>'
        # Away penalty spot
        '<circle cx="284" cy="100" r="2" fill="white"/>'
        # Away goal
        '<rect x="312" y="82" width="6" height="36" fill="none" stroke="white" stroke-width="1"/>'
    )

    # ── Players ───────────────────────────────────────────────────────────────
    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return [1, 4, 3, 3]
        return [1] + [int(x) for x in fmt_str.split("-")]

    h_lines = parse_formation(home_formation)
    a_lines = parse_formation(away_formation)

    # Home team (left → right), GK near left goal
    if h_lines and home_players:
        x_steps = len(h_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(h_lines):
            x = 24 + (116 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 14 + (172 / (num_players + 1)) * (row_idx + 1)
                name = home_players[p_idx] if p_idx < len(home_players) else f"H{p_idx+1}"
                glow = 'filter="url(#glow-h)"' if (p_idx % 3 == 0) else ""
                svg += (
                    f'<a href="/player?name={name}&team={home_team}" style="cursor:pointer;">'
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6"'
                    f' fill="{home_color}" stroke="white" stroke-width="1.2" {glow}/>'
                    f'<text x="{x:.1f}" y="{y+11:.1f}" fill="white"'
                    f' font-family="\'JetBrains Mono\',monospace"'
                    f' font-size="5px" text-anchor="middle"'
                    f' style="text-shadow:0 0 3px rgba(0,0,0,0.8)">{name}</text>'
                    f'</a>'
                )
                p_idx += 1

    # Away team (right → left), GK near right goal
    if a_lines and away_players:
        x_steps = len(a_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(a_lines):
            x = 296 - (116 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 14 + (172 / (num_players + 1)) * (row_idx + 1)
                name = away_players[p_idx] if p_idx < len(away_players) else f"A{p_idx+1}"
                glow = 'filter="url(#glow-a)"' if (p_idx % 4 == 0) else ""
                svg += (
                    f'<a href="/player?name={name}&team={away_team}" style="cursor:pointer;">'
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6"'
                    f' fill="{away_color}" stroke="white" stroke-width="1.2" {glow}/>'
                    f'<text x="{x:.1f}" y="{y+11:.1f}" fill="white"'
                    f' font-family="\'JetBrains Mono\',monospace"'
                    f' font-size="5px" text-anchor="middle"'
                    f' style="text-shadow:0 0 3px rgba(0,0,0,0.8)">{name}</text>'
                    f'</a>'
                )
                p_idx += 1

    # ── Score overlay (pill at top centre, inside pitch boundary) ─────────────
    if h_score is not None and a_score is not None:
        svg += (
            '<rect x="118" y="8" width="84" height="22"'
            ' fill="rgba(0,0,0,0.60)" rx="11"/>'
        )
        if home_crest_url:
            svg += (
                f'<image href="{home_crest_url}"'
                f' x="121" y="10" width="14" height="14" opacity="0.95"/>'
            )
        if away_crest_url:
            svg += (
                f'<image href="{away_crest_url}"'
                f' x="185" y="10" width="14" height="14" opacity="0.95"/>'
            )
        svg += (
            f'<text id="live-score" x="160" y="23" fill="white"'
            f' font-family="\'JetBrains Mono\',monospace"'
            f' font-size="11" font-weight="bold" text-anchor="middle"'
            f' style="text-shadow:0 1px 4px rgba(0,0,0,0.9)">'
            f'{h_score}  –  {a_score}</text>'
        )
        if status:
            badge = {"Finished": "FT", "Half Time": "HT"}.get(status, status)
            svg += (
                f'<text id="live-status-badge" x="160" y="31" fill="#94a3b8"'
                f' font-family="\'JetBrains Mono\',monospace"'
                f' font-size="4.5" text-anchor="middle">{badge}</text>'
            )

    svg += "</svg>"
    return svg
