import json
import random
from pathlib import Path

# Core teams to generate mock data for
leagues = {
    "Premier League": ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town", 
        "Leicester City", "Liverpool", "Manchester City", "Manchester United", 
        "Newcastle United", "Nottingham Forest", "Southampton", "Tottenham Hotspur", 
        "West Ham United", "Wolverhampton"],
    "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Athletic Club", "Real Sociedad"],
    "Serie A": ["Inter Milan", "AC Milan", "Juventus", "Atalanta", "Bologna"],
    "Bundesliga": ["Bayer Leverkusen", "Bayern Munich", "VfB Stuttgart", "RB Leipzig", "Borussia Dortmund"],
    "Ligue 1": ["Paris SG", "Monaco", "Marseille", "Lille", "Lens"]
}

db = {
    "teams": {},
    "players": {},
    "match_history": {}
}

positions = ["GK", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD", "FWD", "FWD"]

def generate_player(name, team, pos):
    return {
        "id": f"p_{hash(name) % 100000}",
        "name": name,
        "team": team,
        "position": pos,
        "jersey": random.randint(1, 30),
        "age": random.randint(19, 34),
        "nationality": random.choice(["ENG", "FRA", "ESP", "GER", "BRA", "ARG", "ITA"]),
        "stats": {
            "attacking": random.randint(40, 95) if pos != "GK" else random.randint(10, 30),
            "technical": random.randint(50, 95),
            "tactical": random.randint(50, 95),
            "defending": random.randint(50, 95) if pos in ["GK", "DEF", "MID"] else random.randint(20, 50),
            "creativity": random.randint(50, 95)
        },
        "summary": {
            "rating": round(random.uniform(6.5, 8.5), 2),
            "matches": random.randint(15, 38),
            "goals": random.randint(0, 25) if pos != "GK" else 0,
            "assists": random.randint(0, 15) if pos != "GK" else 0
        },
        "image": "https://api-sports.io/football/players/" + str(random.randint(100, 999)) + ".png" # generic fallback image
    }

for league, teams in leagues.items():
    for team in teams:
        # Generate 11 players for the team
        team_players = [f"{team[:3]} {pos}{i}" for i, pos in enumerate(positions)]
        
        # specific hardcodes for top teams
        if team == "Wolverhampton":
            team_players = ["Sa", "Semedo", "Dawson", "Kilman", "Ait-Nouri", "Lemina", "Gomes", "Bellegarde", "Neto", "Cunha", "Hwang"]
        elif team == "Arsenal":
            team_players = ["Raya", "White", "Saliba", "Gabriel", "Zinchenko", "Rice", "Odegaard", "Saka", "Martinelli", "Jesus", "Trossard"]
        elif team == "Liverpool":
            team_players = ["Alisson", "Alexander-Arnold", "Konate", "Van Dijk", "Robertson", "Mac Allister", "Szoboszlai", "Diaz", "Salah", "Nunez", "Gakpo"]
        elif team == "Manchester City":
            team_players = ["Ederson", "Walker", "Dias", "Akanji", "Gvardiol", "Rodri", "De Bruyne", "Silva", "Foden", "Haaland", "Grealish"]
        elif team == "Real Madrid":
            team_players = ["Courtois", "Carvajal", "Militao", "Rudiger", "Mendy", "Valverde", "Tchouameni", "Bellingham", "Rodrygo", "Mbappe", "Vinicius"]
        elif team == "Barcelona":
            team_players = ["Ter Stegen", "Kounde", "Araujo", "Cubarsi", "Balde", "De Jong", "Pedri", "Gundogan", "Yamal", "Lewandowski", "Raphinha"]
        elif team == "Inter Milan":
            team_players = ["Sommer", "Pavard", "Acerbi", "Bastoni", "Dumfries", "Barella", "Calhanoglu", "Mkhitaryan", "Dimarco", "Thuram", "Martinez"]
        elif team == "Juventus":
            team_players = ["Szczesny", "Gatti", "Bremer", "Danilo", "Cambiaso", "Locatelli", "Rabiot", "McKennie", "Chiesa", "Vlahovic", "Yildiz"]
        elif team == "Bayern Munich":
            team_players = ["Neuer", "Kimmich", "Upamecano", "Kim", "Davies", "Laimer", "Goretzka", "Sane", "Musiala", "Coman", "Kane"]
        elif team == "Bayer Leverkusen":
            team_players = ["Hradecky", "Kossounou", "Tah", "Tapsoba", "Frimpong", "Palacios", "Xhaka", "Grimaldo", "Hofmann", "Wirtz", "Schick"]
        elif team == "Paris SG":
            team_players = ["Donnarumma", "Hakimi", "Marquinhos", "Skriniar", "Mendes", "Zaire-Emery", "Ugarte", "Vitinha", "Dembele", "Ramos", "Barcola"]
        elif team == "Marseille":
            team_players = ["Lopez", "Clauss", "Mbemba", "Balerdi", "Merlin", "Rongier", "Veretout", "Harit", "Sarr", "Aubameyang", "Ndiaye"]
            
        roster = []
        for i, pname in enumerate(team_players):
            player_obj = generate_player(pname, team, positions[i])
            db["players"][pname] = player_obj
            roster.append(pname)
            
        db["teams"][team] = {
            "roster": roster,
            "manager": "Head Coach",
            "manager_image": "https://api-sports.io/football/coachs/1.png"
        }
        
        # Generate match history
        opponents = [t for t in teams if t != team]
        random.shuffle(opponents)
        history = []
        for i in range(min(5, len(opponents))):
            scored = random.randint(0, 3)
            conceded = random.randint(0, 3)
            res = "W" if scored > conceded else "L" if conceded > scored else "D"
            history.append({
                "date": f"Sep {random.randint(1, 28):02d}",
                "opponent": opponents[i],
                "result": f"{res} {scored}-{conceded}",
                "xg": f"{scored*0.8 + random.uniform(0, 0.5):.1f} - {conceded*0.9 + random.uniform(0, 0.5):.1f}",
                "possession": f"{random.randint(40, 60)}%"
            })
        db["match_history"][team] = history

DB_PATH = Path(__file__).parent / "v3_web" / "data.json"
with open(DB_PATH, "w") as f:
    json.dump(db, f, indent=2)

print(f"Created comprehensive mock DB at {DB_PATH}!")