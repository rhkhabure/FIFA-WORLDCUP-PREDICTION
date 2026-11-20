# Historical football player statistics

The player page can read historical player-season aggregates from
`football_player_stats.sqlite`. The database is generated locally and excluded
from Git; the web application reads it only and never scrapes a provider while
serving a request.

From the repository root, install the project requirements and run:

```powershell
python v4_backend\import_player_stats.py
```

By default, the importer requests Understat player-season statistics through
SoccerData for the five supported leagues, from seasons 2014/15 through
2025/26. The initial import can take time. To import a smaller range or one
league:

```powershell
python v4_backend\import_player_stats.py --from-season 2020 --through-season 2025 --league pl
```

Supported league keys are `pl`, `laliga`, `bundesliga`, `seriea`, and `ligue1`.
The default database location is
`v4_web/football/sql/football_player_stats.sqlite`; set
`FOOTBALL_PLAYER_STATS_DB_PATH` to use another local file. Re-running an import
updates the matching league, season, and player rows.

Displayed fields are Understat season aggregates (appearances, minutes, goals,
assists, xG, xA, shots, key passes, and cards). This is historical season data,
not a match log or live feed. Coverage and metrics vary by league and season;
the application does not fabricate values for missing records. The page labels
the data source.

SoccerData is a community-maintained scraper, not a license for the source
data. Before redistributing the database or using it on a public deployment,
review Understat's current terms and confirm that the intended collection and
display are permitted. Respect site access limits; do not use the importer to
bypass blocks or access controls.
