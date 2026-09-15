"""
laliga_teamdata.py  —  V4.2
============================
Historical La Liga match data from football-data.co.uk free CSVs.
No API key required. CSVs go back to 2000/01.

URL pattern:
  https://www.football-data.co.uk/mmz4281/{YYRR}/SP1.csv
  SP1 = La Liga Primera División
  YYRR = e.g. "2324" for 2023/24, "2425" for 2024/25

CSV columns used:
  Date     : match date (DD/MM/YY or DD/MM/YYYY)
  HomeTeam : home team name
  AwayTeam : away team name
  FTHG     : full-time home goals
  FTAG     : full-time away goals
  FTR      : full-time result (H / D / A)

Mirrors football_co_uk.py exactly — same return formats,
same caching strategy, just SP1 instead of E0.
"""

import csv
import io
import ssl
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

EAT       = timezone(timedelta(hours=3))
BASE_URL  = "https://www.football-data.co.uk/mmz4281"
CACHE_DIR = Path(__file__).resolve().parent / "data" / "fdco"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

# ── Name normalisation ─────────────────────────────────────────────
# football-data.co.uk uses their own Spanish name conventions.
# Map these to the names in our v4_priors.json (ESP-La Liga teams).

_LL_ALIASES: dict[str, str] = {
    # Real Madrid variants
    "Real Madrid"               : "Real Madrid",
    # Barcelona
    "Barcelona"                 : "Barcelona",
    "FC Barcelona"              : "Barcelona",
    # Atletico
    "Ath Madrid"                : "Atletico Madrid",
    "Atletico Madrid"           : "Atletico Madrid",
    "Atlético Madrid"           : "Atletico Madrid",
    # Athletic Bilbao
    "Ath Bilbao"                : "Athletic Club",
    "Athletic Bilbao"           : "Athletic Club",
    "Athletic Club"             : "Athletic Club",
    # Sevilla
    "Sevilla"                   : "Sevilla",
    # Real Betis
    "Betis"                     : "Real Betis",
    "Real Betis"                : "Real Betis",
    # Valencia
    "Valencia"                  : "Valencia",
    # Villarreal
    "Villarreal"                : "Villarreal",
    # Real Sociedad
    "Sociedad"                  : "Real Sociedad",
    "Real Sociedad"             : "Real Sociedad",
    # Celta Vigo
    "Celta"                     : "Celta Vigo",
    "Celta Vigo"                : "Celta Vigo",
    "RC Celta"                  : "Celta Vigo",
    # Getafe
    "Getafe"                    : "Getafe",
    # Alaves
    "Alaves"                    : "Alaves",
    "Alavés"                    : "Alaves",
    "Dep. Alavés"               : "Alaves",
    # Girona
    "Girona"                    : "Girona",
    # Mallorca
    "Mallorca"                  : "Mallorca",
    "RCD Mallorca"              : "Mallorca",
    # Osasuna
    "Osasuna"                   : "Osasuna",
    "CA Osasuna"                : "Osasuna",
    # Espanyol
    "Espanol"                   : "Espanyol",
    "Espanyol"                  : "Espanyol",
    "RCD Espanyol"              : "Espanyol",
    # Rayo Vallecano
    "Rayo Vallecano"            : "Rayo Vallecano",
    "Vallecano"                 : "Rayo Vallecano",
    # Levante
    "Levante"                   : "Levante",
    # Elche
    "Elche"                     : "Elche",
    # Las Palmas
    "Las Palmas"                : "Las Palmas",
    "UD Las Palmas"             : "Las Palmas",
    # Oviedo
    "Oviedo"                    : "Oviedo",
    "Real Oviedo"               : "Oviedo",
    # Leganes
    "Leganes"                   : "Leganes",
    "CD Leganés"                : "Leganes",
    # Valladolid
    "Valladolid"                : "Real Valladolid",
    "Real Valladolid"           : "Real Valladolid",
    # Granada
    "Granada"                   : "Granada",
    "Granada CF"                : "Granada",
    # Cadiz
    "Cadiz"                     : "Cadiz",
    "Cádiz"                     : "Cadiz",
    # Almeria
    "Almeria"                   : "Almeria",
    "UD Almería"                : "Almeria",
    # Huesca
    "Huesca"                    : "Huesca",
    # Eibar
    "Eibar"                     : "Eibar",
    "SD Eibar"                  : "Eibar",
    # Deportivo La Coruna
    "La Coruna"                 : "Deportivo La Coruna",
    "Deportivo"                 : "Deportivo La Coruna",
    # Malaga
    "Malaga"                    : "Malaga",
    "Málaga"                    : "Malaga",
    # Espanyol old
    "Espanol"                   : "Espanyol",
    # Sporting Gijon
    "Sp. Gijon"                 : "Sporting Gijon",
    "Sporting Gijon"            : "Sporting Gijon",
    # Real Zaragoza
    "Zaragoza"                  : "Real Zaragoza",
    # Recreativo
    "Recreativo"                : "Recreativo",
    # Numancia
    "Numancia"                  : "Numancia",
    # Tenerife
    "Tenerife"                  : "Tenerife",
}


def _normalise(name: str) -> str:
    name = name.strip()
    return _LL_ALIASES.get(name, name)


def _season_code(season_start: int) -> str:
    """2023 → '2324'"""
    y1 = str(season_start)[-2:]
    y2 = str(season_start + 1)[-2:]
    return f"{y1}{y2}"


def _cache_path(season_start: int) -> Path:
    return CACHE_DIR / f"LL_{season_start}.csv"


def _download_season(season_start: int) -> str | None:
    """Download SP1.csv for the given season and cache locally."""
    code = _season_code(season_start)
    url  = f"{BASE_URL}/{code}/SP1.csv"
    req  = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; v4dashboard/1.0)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        _cache_path(season_start).write_text(raw, encoding="utf-8")
        print(f"[laliga_fdco] downloaded {season_start}/{season_start+1} "
              f"({len(raw):,} bytes)")
        return raw
    except Exception as e:
        print(f"[laliga_fdco] download failed {season_start}: {e}")
        return None


def _load_season_csv(season_start: int) -> str | None:
    """Load from cache; download if missing or stale."""
    cache = _cache_path(season_start)
    current_year = datetime.now().year
    is_current   = season_start >= current_year - 1
    if cache.exists():
        age = time.time() - cache.stat().st_mtime
        # Past seasons: cache forever
        # Current season: refresh every 6 hours
        if not is_current or age < 6 * 3600:
            return cache.read_text(encoding="utf-8")
    return _download_season(season_start)


def _parse_date(date_str: str) -> str:
    """DD/MM/YY or DD/MM/YYYY → 'Sat 12 Sep'"""
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%a %d %b")
        except ValueError:
            continue
    return date_str.strip()


def _parse_date_iso(date_str: str) -> str:
    """DD/MM/YY → 'YYYY-MM-DD' for sorting."""
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str.strip()


# ── Public API ──────────────────────────────────────────────────────

def get_team_results_historical(
    team_name: str,
    season_start: int,
) -> list[dict]:
    """
    Returns match results for a La Liga team in a given season.

    team_name   : normalised name matching our priors (e.g. "Real Madrid",
                  "Atletico Madrid", "Athletic Club")
    season_start: season start year, e.g. 2023 for 2023/24

    Returns list of dicts (newest first):
      match_id, date, utc_date, matchday,
      opponent, home_away, h_score, a_score,
      result (W/D/L from team perspective), status, is_home
    """
    raw = _load_season_csv(season_start)
    if not raw:
        return []

    reader  = csv.DictReader(io.StringIO(raw))
    results = []
    matchday = 0

    for row in reader:
        home = _normalise(row.get("HomeTeam", ""))
        away = _normalise(row.get("AwayTeam", ""))
        if not home or not away:
            continue

        is_home = (home == team_name)
        is_away = (away == team_name)
        if not is_home and not is_away:
            continue

        matchday += 1
        date_raw = row.get("Date", "").strip()
        fthg     = row.get("FTHG", "").strip()
        ftag     = row.get("FTAG", "").strip()
        ftr      = row.get("FTR",  "").strip()

        try:
            h_score = int(fthg) if fthg else None
            a_score = int(ftag) if ftag else None
        except ValueError:
            h_score = a_score = None

        result = ""
        if ftr == "H":
            result = "W" if is_home else "L"
        elif ftr == "D":
            result = "D"
        elif ftr == "A":
            result = "W" if is_away else "L"

        results.append({
            "match_id" : None,
            "date"     : _parse_date(date_raw),
            "utc_date" : _parse_date_iso(date_raw),
            "matchday" : matchday,
            "opponent" : away if is_home else home,
            "home_away": "H" if is_home else "A",
            "h_score"  : h_score,
            "a_score"  : a_score,
            "result"   : result,
            "status"   : "FINISHED" if result else "SCHEDULED",
            "is_home"  : is_home,
        })

    results.sort(key=lambda x: x["utc_date"], reverse=True)
    return results


def get_ll_standings_historical(season_start: int) -> list[dict]:
    """
    Compute La Liga standings table from SP1.csv for a given season.
    Returns list of dicts (same format as teamdata.get_standings):
      position, team_name, team_id (None), crest (''),
      played, won, draw, lost, gf, ga, gd, points
    """
    raw = _load_season_csv(season_start)
    if not raw:
        return []

    reader = csv.DictReader(io.StringIO(raw))
    table: dict[str, dict] = {}

    for row in reader:
        home = _normalise(row.get("HomeTeam", ""))
        away = _normalise(row.get("AwayTeam", ""))
        ftr  = row.get("FTR",  "").strip()
        fthg = row.get("FTHG", "").strip()
        ftag = row.get("FTAG", "").strip()
        if not home or not away or not ftr:
            continue
        try:
            hg, ag = int(fthg), int(ftag)
        except ValueError:
            continue

        for team, scored, conceded in [
            (home, hg, ag),
            (away, ag, hg),
        ]:
            if team not in table:
                table[team] = dict(
                    team_name=team, team_id=None, crest="",
                    played=0, won=0, draw=0, lost=0,
                    gf=0, ga=0, gd=0, points=0,
                )
            t = table[team]
            t["played"] += 1
            t["gf"]     += scored
            t["ga"]     += conceded
            t["gd"]      = t["gf"] - t["ga"]
            if scored > conceded:
                t["won"]    += 1
                t["points"] += 3
            elif scored == conceded:
                t["draw"]   += 1
                t["points"] += 1
            else:
                t["lost"]   += 1

    rows = sorted(
        table.values(),
        key=lambda x: (-x["points"], -x["gd"], -x["gf"])
    )
    for i, row in enumerate(rows, 1):
        row["position"] = i
    return rows


def get_form_string(results: list[dict], n: int = 5) -> str:
    """Return 'WDWLW' form string from last N finished results."""
    finished = [r for r in results
                if r.get("result") in ("W", "D", "L")
                and r.get("status") == "FINISHED"]
    return "".join(r["result"] for r in finished[:n])


def get_last_n_results(results: list[dict], n: int = 5) -> list[dict]:
    """Return last N finished results."""
    return [r for r in results
            if r.get("result") in ("W", "D", "L")
            and r.get("status") == "FINISHED"][:n]


# ── Hardcoded La Liga team info ────────────────────────────────────
# Since we don't have a football-data.org PD endpoint on the free tier,
# we store the key profile facts statically.
# Updated for 2024/25 season.

LL_TEAM_INFO: dict[str, dict] = {
    "Real Madrid": {
        "venue": "Estadio Santiago Bernabéu", "city": "Madrid",
        "founded": 1902, "colours": "White / Gold",
        "coach": "Carlo Ancelotti", "capacity": 81044,
        "website": "https://www.realmadrid.com",
    },
    "Barcelona": {
        "venue": "Estadi Olímpic Lluís Companys", "city": "Barcelona",
        "founded": 1899, "colours": "Blue / Dark Red",
        "coach": "Hansi Flick", "capacity": 54000,
        "website": "https://www.fcbarcelona.com",
    },
    "Atletico Madrid": {
        "venue": "Cívitas Metropolitano", "city": "Madrid",
        "founded": 1903, "colours": "Red / White",
        "coach": "Diego Simeone", "capacity": 68456,
        "website": "https://www.atleticodemadrid.com",
    },
    "Sevilla": {
        "venue": "Estadio Ramón Sánchez-Pizjuán", "city": "Seville",
        "founded": 1890, "colours": "White / Red",
        "coach": "Francisco Javier García Pimienta", "capacity": 43883,
        "website": "https://www.sevillafc.es",
    },
    "Real Betis": {
        "venue": "Estadio Benito Villamarín", "city": "Seville",
        "founded": 1907, "colours": "Green / White",
        "coach": "Manuel Pellegrini", "capacity": 60720,
        "website": "https://www.realbetisbalompie.es",
    },
    "Athletic Club": {
        "venue": "San Mamés", "city": "Bilbao",
        "founded": 1898, "colours": "Red / White",
        "coach": "Ernesto Valverde", "capacity": 53289,
        "website": "https://www.athletic-club.eus",
    },
    "Real Sociedad": {
        "venue": "Reale Arena", "city": "San Sebastián",
        "founded": 1909, "colours": "Blue / White",
        "coach": "Imanol Alguacil", "capacity": 39500,
        "website": "https://www.realsociedad.eus",
    },
    "Valencia": {
        "venue": "Estadio Mestalla", "city": "Valencia",
        "founded": 1919, "colours": "White / Black",
        "coach": "Rubén Baraja", "capacity": 49430,
        "website": "https://www.valenciacf.com",
    },
    "Villarreal": {
        "venue": "Estadio de la Cerámica", "city": "Villarreal",
        "founded": 1923, "colours": "Yellow",
        "coach": "Marcelino", "capacity": 23500,
        "website": "https://www.villarrealcf.es",
    },
    "Celta Vigo": {
        "venue": "Abanca-Balaídos", "city": "Vigo",
        "founded": 1923, "colours": "Sky Blue / White",
        "coach": "Claudio Giráldez", "capacity": 29000,
        "website": "https://www.celtavigo.net",
    },
    "Getafe": {
        "venue": "Coliseum Alfonso Pérez", "city": "Getafe",
        "founded": 1983, "colours": "Blue",
        "coach": "José Bordalás", "capacity": 17393,
        "website": "https://www.getafecf.com",
    },
    "Rayo Vallecano": {
        "venue": "Campo de Fútbol de Vallecas", "city": "Madrid",
        "founded": 1924, "colours": "White / Red",
        "coach": "Íñigo Pérez", "capacity": 14708,
        "website": "https://www.rayovallecano.es",
    },
    "Alaves": {
        "venue": "Estadio de Mendizorroza", "city": "Vitoria-Gasteiz",
        "founded": 1921, "colours": "Blue / White",
        "coach": "Luis García Plaza", "capacity": 19840,
        "website": "https://www.deportivoalaves.com",
    },
    "Osasuna": {
        "venue": "El Sadar", "city": "Pamplona",
        "founded": 1920, "colours": "Red / Black",
        "coach": "Vicente Moreno", "capacity": 23576,
        "website": "https://www.osasuna.es",
    },
    "Girona": {
        "venue": "Estadi Municipal de Montilivi", "city": "Girona",
        "founded": 1930, "colours": "Red / White",
        "coach": "Míchel Sánchez", "capacity": 13450,
        "website": "https://www.gironafc.cat",
    },
    "Mallorca": {
        "venue": "Visit Mallorca Estadi", "city": "Palma",
        "founded": 1916, "colours": "Red / Black / Yellow",
        "coach": "Jagoba Arrasate", "capacity": 23142,
        "website": "https://www.rcdmallorca.es",
    },
    "Espanyol": {
        "venue": "Stage Front Stadium", "city": "Barcelona",
        "founded": 1900, "colours": "Blue / White",
        "coach": "Manolo González", "capacity": 40000,
        "website": "https://www.rcdespanyol.com",
    },
    "Levante": {
        "venue": "Estadio Ciudad de Valencia", "city": "Valencia",
        "founded": 1909, "colours": "Blue / Red",
        "coach": "Felipe Miñambres", "capacity": 25354,
        "website": "https://www.levanteud.com",
    },
    "Oviedo": {
        "venue": "Estadio Carlos Tartiere", "city": "Oviedo",
        "founded": 1926, "colours": "Blue",
        "coach": "Luis Carrión", "capacity": 30500,
        "website": "https://www.realoviedo.es",
    },
    "Elche": {
        "venue": "Estadio Martínez Valero", "city": "Elche",
        "founded": 1923, "colours": "Green / White",
        "coach": "Chus Herrero", "capacity": 33732,
        "website": "https://www.elchecf.es",
    },
}

# Seasons available from football-data.co.uk SP1.csv
LL_AVAILABLE_SEASONS = list(range(2000, 2025))   # 2000/01 → 2024/25
LL_SEASONS_DISPLAY = {
    s: f"{s}/{str(s+1)[-2:]}" for s in LL_AVAILABLE_SEASONS
}

# Map our internal name → fdco name variants (for reverse lookup)
LL_PRIORS_NAMES = list(LL_TEAM_INFO.keys())
