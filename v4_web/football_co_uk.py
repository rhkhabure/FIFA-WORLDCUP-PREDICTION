"""
football_co_uk.py  —  V4.2
===========================
Historical match data from football-data.co.uk
Free, no API key, CSVs going back to 1888.

URL pattern:
  https://www.football-data.co.uk/mmz4281/{YYRR}/E0.csv
  E0 = Premier League
  YYRR = e.g. "1415" for 2014/15, "2324" for 2023/24

CSV columns we use:
  Date       : match date (DD/MM/YY or DD/MM/YYYY)
  HomeTeam   : home team name (their own naming convention)
  AwayTeam   : away team name
  FTHG       : full-time home goals
  FTAG       : full-time away goals
  FTR        : full-time result (H=home win, D=draw, A=away win)

Caches each season CSV locally in a `data/fdco/` folder so we
don't re-download on every page load.
"""

import csv
import io
import os
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

EAT = timezone(timedelta(hours=3))

BASE_URL  = "https://www.football-data.co.uk/mmz4281"
CACHE_DIR = Path(__file__).parent / "data" / "fdco"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# football-data.co.uk team names → our internal names
# Their names differ slightly from football-data.org
_TEAM_ALIASES: dict[str, str] = {
    "Man United"       : "Manchester United",
    "Man City"         : "Manchester City",
    "Tottenham"        : "Tottenham",
    "Spurs"            : "Tottenham",
    "Newcastle"        : "Newcastle United",
    "Wolves"           : "Wolverhampton Wanderers",
    "Nott'm Forest"    : "Nottingham Forest",
    "Nottm Forest"     : "Nottingham Forest",
    "West Ham"         : "West Ham",
    "West Brom"        : "West Brom",
    "Sheffield United" : "Sheffield United",
    "Sheffield Weds"   : "Sheffield Wednesday",
    "Leicester"        : "Leicester",
    "Leeds"            : "Leeds",
    "Burnley"          : "Burnley",
    "Fulham"           : "Fulham",
    "Brentford"        : "Brentford",
    "Brighton"         : "Brighton",
    "Bournemouth"      : "Bournemouth",
    "Ipswich"          : "Ipswich",
    "Coventry"         : "Coventry",
    "Sunderland"       : "Sunderland",
    "Hull"             : "Hull City",
    "QPR"              : "QPR",
    "Stoke"            : "Stoke City",
    "Swansea"          : "Swansea City",
    "Wigan"            : "Wigan Athletic",
    "Reading"          : "Reading",
    "Norwich"          : "Norwich City",
    "Cardiff"          : "Cardiff City",
    "Southampton"      : "Southampton",
    "Crystal Palace"   : "Crystal Palace",
    "Watford"          : "Watford",
    "Huddersfield"     : "Huddersfield Town",
    "Middlesbrough"    : "Middlesbrough",
    "Derby"            : "Derby County",
    "Blackburn"        : "Blackburn Rovers",
    "Bolton"           : "Bolton Wanderers",
}


def _season_code(season_start: int) -> str:
    """Convert season start year to FDCO code. 2014 → '1415'."""
    y1 = str(season_start)[-2:]
    y2 = str(season_start + 1)[-2:]
    return f"{y1}{y2}"


def _cache_path(season_start: int) -> Path:
    return CACHE_DIR / f"PL_{season_start}.csv"


def _download_season(season_start: int) -> str | None:
    """Download season CSV and cache it. Returns raw CSV text or None."""
    code = _season_code(season_start)
    url  = f"{BASE_URL}/{code}/E0.csv"
    req  = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (compatible; v4dashboard/1.0)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Write to cache
        cache = _cache_path(season_start)
        cache.write_text(raw, encoding="utf-8")
        print(f"fdco: downloaded {season_start}/{season_start+1} ({len(raw)} bytes)")
        return raw
    except Exception as e:
        print(f"fdco: failed to download {season_start}: {e}")
        return None


def _load_season_csv(season_start: int) -> str | None:
    """Load from cache or download. Returns raw CSV text."""
    cache = _cache_path(season_start)
    # Use cache if it exists and is less than 6 hours old (for current season)
    # For past seasons, cache forever
    current_year = datetime.now().year
    is_current   = season_start >= current_year - 1
    if cache.exists():
        age = time.time() - cache.stat().st_mtime
        if not is_current or age < 6 * 3600:
            return cache.read_text(encoding="utf-8")
    return _download_season(season_start)


def _parse_date(date_str: str) -> str:
    """Parse DD/MM/YY or DD/MM/YYYY → 'Sat 12 Sep' in EAT."""
    date_str = date_str.strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%a %d %b")
        except ValueError:
            continue
    return date_str


def _parse_date_iso(date_str: str) -> str:
    """Parse to ISO string for sorting."""
    date_str = date_str.strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def _normalise_team(name: str) -> str:
    """Map fdco team name to our internal name."""
    return _TEAM_ALIASES.get(name.strip(), name.strip())


def get_team_results_historical(
    team_name: str,
    season_start: int,
) -> list[dict]:
    """
    Returns match results for a team in a given season from fdco CSVs.
    team_name: our internal name (e.g. "Arsenal", "Manchester United")
    season_start: e.g. 2014 for 2014/15

    Each result dict matches the format from teamdata.get_team_season_results:
      date, utc_date, matchday, opponent, home_away,
      h_score, a_score, result, status, is_home, match_id
    """
    raw = _load_season_csv(season_start)
    if not raw:
        return []

    reader = csv.DictReader(io.StringIO(raw))
    results = []
    matchday = 0

    for row in reader:
        home = _normalise_team(row.get("HomeTeam", ""))
        away = _normalise_team(row.get("AwayTeam", ""))
        if not home or not away:
            continue

        is_home = (home == team_name)
        is_away = (away == team_name)
        if not is_home and not is_away:
            continue

        matchday += 1
        date_raw = row.get("Date", "").strip()
        fthg = row.get("FTHG", "").strip()
        ftag = row.get("FTAG", "").strip()
        ftr  = row.get("FTR",  "").strip()

        try:
            h_score = int(fthg) if fthg else None
            a_score = int(ftag) if ftag else None
        except ValueError:
            h_score = a_score = None

        # FTR: H=home win, D=draw, A=away win → W/D/L from team perspective
        result = ""
        if ftr == "H":
            result = "W" if is_home else "L"
        elif ftr == "D":
            result = "D"
        elif ftr == "A":
            result = "W" if is_away else "L"

        opponent = away if is_home else home

        results.append({
            "match_id"  : None,          # no ID in CSV
            "date"      : _parse_date(date_raw),
            "utc_date"  : _parse_date_iso(date_raw),
            "matchday"  : matchday,
            "opponent"  : opponent,
            "home_away" : "H" if is_home else "A",
            "h_score"   : h_score,
            "a_score"   : a_score,
            "result"    : result,
            "status"    : "FINISHED" if result else "SCHEDULED",
            "is_home"   : is_home,
        })

    # Oldest first → newest first
    results.sort(key=lambda x: x["utc_date"], reverse=True)
    return results


def get_pl_standings_historical(season_start: int) -> list[dict]:
    """
    Compute PL standings table from fdco CSV for a given season.
    Returns list of dicts matching teamdata.get_standings format
    (minus team_id and crest which aren't in the CSV).
    """
    raw = _load_season_csv(season_start)
    if not raw:
        return []

    reader = csv.DictReader(io.StringIO(raw))
    table: dict[str, dict] = {}

    for row in reader:
        home = _normalise_team(row.get("HomeTeam", ""))
        away = _normalise_team(row.get("AwayTeam", ""))
        ftr  = row.get("FTR", "").strip()
        fthg = row.get("FTHG", "").strip()
        ftag = row.get("FTAG", "").strip()
        if not home or not away or not ftr:
            continue
        try:
            hg, ag = int(fthg), int(ftag)
        except ValueError:
            continue

        for team, scored, conceded, is_home_team in [
            (home, hg, ag, True),
            (away, ag, hg, False),
        ]:
            if team not in table:
                table[team] = dict(
                    team_name=team, team_id=None, crest="",
                    played=0, won=0, draw=0, lost=0,
                    gf=0, ga=0, gd=0, points=0, form=""
                )
            t = table[team]
            t["played"] += 1
            t["gf"] += scored
            t["ga"] += conceded
            t["gd"] = t["gf"] - t["ga"]
            if scored > conceded:
                t["won"]    += 1
                t["points"] += 3
            elif scored == conceded:
                t["draw"]   += 1
                t["points"] += 1
            else:
                t["lost"]   += 1

    # Sort: points desc, gd desc, gf desc
    rows = sorted(
        table.values(),
        key=lambda x: (-x["points"], -x["gd"], -x["gf"])
    )
    for i, row in enumerate(rows, 1):
        row["position"] = i

    return rows
