# ⚽🏀 Melios Dira Odds — V4.2 Football & NBA Prediction Terminal

> A multi-sport prediction dashboard combining Dixon-Coles football modelling with an NBA win-probability model. It serves football leagues and NBA live scores, predictions, team pages, player profiles, and historical statistics through a FastAPI dashboard.

---

## What it does

The football system predicts match outcomes using two different models:

- **Pre-game:** Dixon–Coles scoreline probabilities from fitted xG-based team priors. A separate bounded attack/concession factor can adapt to settled results, but only when per-league walk-forward checks improve both Brier score and log-loss.
- **Live (in-game):** An 11→40→20→3 PyTorch classifier using score, minute, and normalized team-strength features. It does **not** consume xG; its role is live state prediction, not pre-match forecasting.

See [`FOOTBALL_MODEL.md`](FOOTBALL_MODEL.md) for the equations, exact inputs,
training and validation procedure, current adaptive-rating status and known
limitations. The application-level reference is
[`DOCUMENTATION.md`](DOCUMENTATION.md).

The NBA dashboard uses Elo for pregame probabilities and a 16-feature neural
network for live probabilities. See [`NBA_MODEL.md`](NBA_MODEL.md) for the
features, probability calculations, training artifacts and temporal test
folds, result-driven Elo adjustment, current evaluation, and provenance
limitations. Historical NBA statistics are queried from local SQLite files;
current schedules, rosters, player information, and live game state come from
the NBA data feeds.

The application is a FastAPI server with Jinja2 templates, a dark-mode UI, football pitches, and an SVG basketball court.

---

## Quick start

```bash
# 1. Install all repository dependencies (Python 3.11 recommended)
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env and fill in:
#   FOOTBALLDATA_ORG_KEY=<your key from football-data.org>
#   BBS_API_KEY=<your Big Balls Sports key>
#   PARSE_BOT_KEY=<your parse.bot key — optional, used for FotMob lineups>

# 3. Run the dashboard
cd v4_web
python -m uvicorn main:app --reload

# 4. Open in browser
# http://localhost:8000
```

## Continuous integration

Every push and pull request runs the CI workflow in
`.github/workflows/ci.yml`. The workflow uses Python 3.11, installs the
declared dependencies, compiles the Python sources, validates all Jinja
templates, and runs the application test suite.

Run the same core checks locally from the repository root:

```bash
python -m compileall -q v4_backend v4_web
python -c "from pathlib import Path; from jinja2 import Environment, FileSystemLoader; env = Environment(loader=FileSystemLoader('v4_web/templates')); [env.get_template(path.name) for path in Path('v4_web/templates').glob('*.html')]; print('All templates compiled successfully')"
PYTHONPATH=v4_web python -m unittest discover -s v4_web/tests -p "test_*.py"
```

## Security baseline

The application validates request hosts, adds standard security response
headers, disables caching for API and administrative responses, and protects
administrative/debug endpoints with the `ADMIN_API_TOKEN` bearer token.
Configure `ALLOWED_HOSTS` with the deployed hostname and never use a real
administrative token in source control.

Health probes are available at `/health/live` for process liveness and
`/health/ready` for local model/database readiness. Operational request
counters are available at `/admin/metrics` with administrative authorization.

External read-only provider requests use bounded retries for transient
transport and server/rate-limit failures. Existing in-memory or SQLite-backed
caches serve stale data where supported, so temporary provider outages degrade
data freshness rather than taking down page rendering.

## Database backups

The application uses local SQLite files for prediction history and fan votes.
Create an integrity-checked backup outside the repository with:

```bash
python v4_web/scripts/backup_databases.py --output-dir /path/to/private/backups
```

Schedule this command on the host and copy the resulting files to durable
off-host storage. The application does not expose database backups over HTTP.

## Performance checks

Responses larger than 1 KiB are gzip-compressed when the client supports it.
The protected `/admin/metrics` endpoint reports aggregate request duration and
slow-request counts for operational monitoring.

With the server running, execute a deterministic local smoke load test:

```bash
python v4_web/scripts/load_test.py --url http://127.0.0.1:8000/health/live
```

Use the output p95 latency and failure count as a baseline before changing
deployment resources or provider timeouts. This utility is not a substitute
for a production load test with representative traffic.

## Privacy and fan voting

Fan voting is optional. The browser does not create its random voting
identifier until the visitor enables the feature and reviews the linked
[`/privacy`](http://localhost:8000/privacy) notice. The server stores only a
one-way hash, the selected outcome, the event identifier, and a timestamp.
Vote rows are deleted after `FAN_VOTE_RETENTION_DAYS` (90 by default).

Before deployment, complete the privacy notice with the operator's legal
identity and contact channel, confirm GDPR and Kenya Data Protection Act
responsibilities, configure hosting-log retention, and document the process
for data-subject requests. This repository provides an engineering baseline,
not legal advice.

## SEO and sharing metadata

HTML pages include descriptions, canonical URLs, Open Graph/Twitter metadata,
and `WebSite` structured data. `robots.txt` and `sitemap.xml` expose only
public navigational pages; API, admin, and debug paths are excluded. Set
`PUBLIC_BASE_URL` to the final HTTPS hostname before deployment so generated
canonical and sitemap URLs are correct.

The social preview uses `v4_web/static/social-preview.png`, a 1200×630 PNG
compatible with common sharing clients. The Tabler icon font is pinned to
version 2.47.0. Tailwind utilities are compiled ahead of time into
`v4_web/static/tailwind.css`; the browser no longer runs the Tailwind CDN
compiler.

The shared layout includes a solid keyboard focus outline, improved light-mode
text contrast, decorative images with empty alternative text, tabular numeric
figures, and polling that pauses while a browser tab is hidden. To rebuild the
Tailwind asset locally, download the v3.4.17 standalone executable as
`v4_web/tailwindcss.exe` and run:

Navigation is presented as a responsive top bar across the application. On
narrow screens, the navigation row scrolls horizontally so all league and team
links remain available without reducing the main content to a narrow sidebar.
The header uses a compact sports-broadcast layout with the sport switcher,
page context, live state, and clock sharing one identity row; privacy and model
details are kept in the page footer.
The identity row uses the NBA league mark on NBA pages and a UEFA football mark
on Football pages, alongside a high-contrast page title.
On NBA pages, the Conferences control reveals Eastern and Western links on
hover or keyboard focus; selecting one jumps to that conference's team list.
On Football pages, the Leagues control reveals all five supported leagues on
hover or keyboard focus; selecting one opens that league's dashboard.
The dropdown uses the matching league crest rather than a colour-only marker.
Quick Teams and All Teams use the same hover/focus dropdown pattern; All Teams
is scrollable to keep the navigation usable on smaller screens.

```bash
cd v4_web
./tailwindcss.exe -c tailwind.config.js -i tailwind.input.css -o static/tailwind.css --minify
```

## About, contact, and disclaimer

The site includes `/about`, `/contact`, and `/disclaimer` pages. It provides
information only, takes no bets, is intended for adults 18+, and links users
to help resources. The NBA hub shows a projected playoff countdown and season
status; football league cards show countdowns to each league's final matchday.

## Deployment and rollback

Before deploying, run the full checks above, create an off-host database
backup, verify all provider/admin secrets, and confirm `ALLOWED_HOSTS` and
`PUBLIC_BASE_URL`. Render uses `/health/ready` as its deployment health check.
After release, verify the public endpoints with:

```bash
python v4_web/scripts/release_check.py --base-url https://your-public-host
```

For rollback, first preserve a backup of the current databases and record the
failing release commit. Redeploy the last known-good commit through the hosting
provider, then rerun the release check. Do not roll back application code
across a database migration unless the old code supports the newer schema;
restore a compatible database backup only after preserving the current copy.
Rotate any credentials exposed during the incident and document the event.

---

## Project structure

```
FIFA-WORLDCUP-PREDICTION/
├── v4_backend/
│   ├── v4_priors.json               # Fitted DC priors by league and team
│   ├── historical_scraper.py        # Scrapes 4 seasons of xG from Understat → SQLite
│   ├── build_v4_dataset.py          # Builds live-model snapshots from match history
│   ├── import_player_stats.py       # Imports historical Understat player-season stats
│   ├── dixon_coles_xg.py            # DC MLE optimiser (SLSQP, draw correction)
│   ├── feature_builder.py           # DC strength lookup + TEAM_NAME_ALIASES
│   ├── likelihood_adjustment.py     # Lineup delta → adjusted λ/μ
│   └── notebooks/
│       ├── train_v4_priors.ipynb    # Fits static DC priors
│       ├── train_v4_neural.ipynb    # Trains the live classifier
│       └── v4_backend/models/
│           ├── football_v4.pth      # Live neural-network checkpoint
│           └── scaler_v4.pkl        # Training-fitted feature scaler
│
├── v4_web/                          # The running application
│   ├── nba/                         # NBA model artifacts and historical datasets
│   │   ├── model/                   # Trained model, scaler and calibration artifacts
│   │   ├── sql/                     # Historical SQLite files and setup notes
│   │   └── csv/                     # Audited source CSV exports
│   ├── main.py                      # FastAPI app and route registration
│   ├── constants.py                 # EAT, DRAW_PROPENSITY, LEAGUE_FILTERS, LEAGUE_MAP
│   ├── routes/
│   │   ├── match.py                 # Football match dashboard and API routes
│   │   └── nba.py                   # NBA hub, match, history, team and player routes
│   ├── bbs.py                       # Big Balls Sports API client (La Liga)
│   ├── footballdata.py              # football-data.org client (PL scores/fixtures)
│   ├── fotmob.py                    # FotMob via parse.bot (lineups, live xG)
│   ├── fpl.py                       # FPL API client (PL fixtures, team map)
│   ├── thesportsdb.py               # TheSportsDB client (player photos, career)
│   ├── predictions.py               # SQLite predictions DB (log + retrieve)
│   ├── model_updates.py             # Walk-forward evaluation and bounded team factors
│   ├── prediction_job.py            # Background job — logs predictions + results
│   ├── nba_api_client.py            # NBA schedules, live scores, rosters and player data
│   ├── nba_model.py                 # NBA win-probability model and Elo ratings
│   ├── nba_predictions.py            # NBA prediction and result history
│   ├── historical_nba.py             # Read-only queries for local NBA history
│   ├── simulate.py                  # Monte Carlo engines (CL knockout + PL season)
│   ├── lineup_adjustment.py         # Applies roster delta to DC λ/μ
│   ├── scoreline_matrix.py          # SVG scoreline probability heatmap
│   ├── timeline.py                  # SVG match event timeline
│   ├── utils.py                     # Pitch SVG, kit definitions, crest proxy
│   ├── teamdata.py                  # PL team profiles
│   ├── laliga_teamdata.py           # La Liga team metadata
│   ├── templates/                   # Jinja2 HTML templates
│   │   ├── base.html                # Sidebar, nav, dark theme
│   │   ├── index.html               # Hub page
│   │   ├── match.html               # Main match prediction page
│   │   ├── nba_hub.html             # NBA schedule and predictions
│   │   ├── nba_match.html           # NBA match dashboard and court
│   │   ├── nba_team.html            # NBA roster and team history
│   │   ├── nba_player.html          # NBA player profile and statistics
│   │   ├── history.html             # Prediction history + accuracy stats
│   │   ├── tournament.html          # CL simulator + PL season simulator
│   │   ├── player.html              # Player profile page
│   │   ├── team.html                # Team profile page
│   │   ├── teams.html               # PL team map
│   │   └── teams_laliga.html        # La Liga team map
│   └── scripts/                     # Utility scripts (not part of app)
│       ├── weekend_odds.py          # Weekend fixture/odds utility
│       └── backfill_history.py      # One-time predictions DB backfill
│
├── v4_historical_data.sqlite        # 4 seasons xG data — training set
├── .env                             # API keys (not committed)
├── README.md                        # This file
├── USER_MANUAL.md                   # Step-by-step setup and usage guide
├── DOCUMENTATION.md                 # Application and integration reference
├── FOOTBALL_MODEL.md                # Football model equations, training and status
└── NBA_MODEL.md                     # NBA model equations, features, training evidence and status
```

---

## Pages

| URL | Description |
|-----|-------------|
| `/` | Hub — today's fixtures with DC odds across both leagues |
| `/match` | PL match prediction — DC pre-game, pitch, scoreline matrix, fixture strip |
| `/match?league=laliga` | La Liga match prediction — same layout, BBS data source |
| `/teams` | PL team map — DC strength heatmap |
| `/teams/laliga` | La Liga team map |
| `/team/{id}` | Team profile — form, DC rating, head-to-head |
| `/player?name=X&team=Y` | Player profile — TheSportsDB photo + career history |
| `/history` | Prediction history — accuracy by confidence band, per league |
| `/tournament` | CL knockout simulator + PL season Monte Carlo (10,000 runs) |
| `/league/laliga` | La Liga recent results, current fixtures and predictions |
| `/league/seriea` | Serie A fixtures and Dixon-Coles pre-match predictions |
| `/teams/seriea` | Serie A club map and directory |
| `/nba` | NBA hub — EAT-day live games, today's schedule, tomorrow's fixtures and predictions |
| `/nba/match?game_id=X` | NBA match dashboard — live win probability, game state and court lineup |
| `/nba/history` | NBA prediction history, results, accuracy and Brier score |
| `/nba/teams` | All 30 NBA teams grouped by conference |
| `/nba/team/{team_id}` | Current roster, Elo, prediction history and historical team records |
| `/nba/player/{player_id}` | Player profile, current stats and historical game/shot analysis |

---

## Football league expansion and reliability updates

The football dashboard has also been extended beyond the original Premier League focus:

| Change | Why it was made | Impact |
|--------|----------------|--------|
| La Liga fixture views now include recent results with scores and restrict upcoming fixtures to the relevant current-day/weekend window. Match detail uses the selected fixture's current data where available. | Old fixtures were lingering in widgets, and games could be missing or inconsistently marked live when switching views. | Fixture lists stay focused on recent, current and near-term matches, with fewer stale results and more consistent score/status display. |
| Prediction history uses normalized team/result matching and derives the predicted outcome from the stored probabilities. Verified missing results were added to the history database. | Team-name variants and mismatched result handling made some correct predictions appear wrong or left completed games out of history. | Displayed outcomes are consistent with the saved prediction probabilities and confirmed scores; corrected historical rows remain in the database. |
| The main hub now gathers fixtures across the supported football competitions, groups them by league and gives live matches priority. Fixture links retain their league context. | The home screen was limited to Premier League fixtures, and multi-league navigation could open the wrong competition context. | One hub presents the available league fixtures and sends each selection to the matching dashboard. |
| Added a Serie A dashboard and prediction-history integration, league-aware team navigation, and a club map/directory. | Serie A was the first vertical slice in a staged expansion to more leagues; its teams, IDs and data feeds differ from Premier League conventions. | Serie A fixtures and model estimates have their own overview/history context, with club pages reachable from the sidebar and map. Missing priors or unavailable club data are identified instead of being presented as verified facts. |

---

## NBA: what changed, why, and impact

NBA support was built as a separate basketball experience rather than adapting football pages and assumptions. This keeps basketball scoring, schedules, teams, predictions, and court rendering independent while reusing the site's shared navigation, styling, and prediction database.

| Change | Why it was made | Impact |
|--------|----------------|--------|
| Added NBA model artifacts, team/conference metadata, and NBA-specific routes and pages. | NBA games need basketball-specific inputs and outputs; football match logic does not apply. | The app now has a dedicated NBA hub, match dashboard, prediction history, team directory, and linked team/player profiles. |
| Added EAT-aware schedule retrieval for both the NBA schedule date and the preceding date, then filtered games by their actual EAT tip-off date. | NBA schedule dates are published in U.S. time zones, so an evening U.S. game may start on the next calendar day in East Africa. | Games are assigned to the correct local day instead of disappearing from "today" or appearing under the wrong date. |
| Added a separate tomorrow schedule that remains visible after today's games finish, plus a tomorrow fallback when today is empty. | Previously, a non-empty list of completed games prevented tomorrow's upcoming games from being requested or displayed. | Users can see completed/current-day games and the next day's fixtures together, and can open tomorrow's match pages directly. |
| Added periodic score/status refresh and cached schedule retrieval. | Live states and scores change during a game, while repeatedly fetching the same schedule can add latency and provider load. | The hub and match views can refresh current game information without requiring a manual page reload. |
| Added the basketball match court, live probability display/history, and lineup markers. The lineup endpoint uses NBA live box-score data and falls back to confirmed ESPN starters when that feed has no usable lineup. | The original court was decorative without player data, and NBA lineup fields may be absent or delayed in the live feed. | Player markers can be displayed when a provider reports players; the page identifies the lineup source. Unannounced future lineups remain blank rather than being guessed. |
| Added current team rosters and linked player profiles, then expanded profiles with headline and season statistics. | The team and player pages needed usable current information, not only model ratings. | Team rosters lead to profiles with NBA-sourced bio/stat data, including the provider's PIE (Player Impact Estimate) headline value. |
| Audited the uploaded historical CSV/SQLite datasets and added read-only historical queries. | Re-fetching old seasons from live endpoints adds rate-limit risk and does not make historical career analysis reliable. | Historical player/team pages query local indexed SQLite data; no importer, API mirror, or write-time database synchronization is required. |
| Split data by freshness: local regular-season history through 2024–25; NBA API data for 2025–26 and 2026–27. | The local dataset is historical and its latest season is incomplete for some event-level data. | Current/near-current season information remains provider-backed while older stats are available without live historical API calls. |
| Added historical regular-season totals, recent game logs, team history, single-game highs, and shot profiles. | Box scores and play-by-play answer different questions: box scores provide game stats while event data allows shot-type calculations. | Profiles show historical regular-season production, calculated career highs, most-attempted shot category, and make accuracy where the dataset supports it. |
| Added historical regular-season records to NBA team profiles. | Season-by-season team context should remain available independently of the live roster API. | Team pages can show local wins, losses, games, points for, and points against through 2024–25. |
| Added NBA prediction logging and settlement to the shared prediction database. | NBA predictions need the same durable audit/history trail as football predictions. | A game is logged once, then its final score, correctness, and Brier score are filled in when it finishes. |
| Improved NBA team pages with the team's conference standings and clickable team links, larger recent-form markers, and consistent green win/red loss colors. Updated the NBA match probability history to render as SVG and improved text contrast across NBA and football dashboards. | Team context and game history were difficult to scan, the probability graph could appear blank, and team-colored labels were not always legible. | Standings are easier to navigate, W/L form colors mean the same thing for every team, the chart renders without relying on Chart.js, and key labels remain readable. |
| Expanded NBA player profiles with side-by-side bio/experience information, objective seasons-played tiers, season-level team history, and NBA Stats awards. Clarified favorite vs. best shot (non-dunk, minimum 25 attempts), and moved detailed season tables lower on the page. | Important context was buried, career honors and team movement were missing, and shot preference was conflated with accuracy. | Player tenure, NBA championships and regular-season MVPs are presented separately from current production; badges and one-franchise indicators are shown only when supporting history is available. |
| Kept large NBA history datasets out of Git and documented local dataset setup. | Historical box scores, odds and play-by-play data exceed GitHub's file-size limits and should not be bundled into application history. | The repository stays cloneable; NBA historical pages and shot-profile features use separately provisioned local files. See [`v4_web/nba/sql/README.md`](v4_web/nba/sql/README.md). |

### NBA data boundaries and known limitations

- The historical stats database contains regular-season player and team box scores, game results, schedule data, and odds. Historical NBA queries use read-only SQLite connections and bounded in-process caches.
- The play-by-play file contains shot events used for broad shot-category summaries. Historical shot labels are grouped into broad categories; the calculated "favorite shot" means the most attempted category, not the most accurate one.
- Historical career highs are the largest single-game regular-season values available in this dataset. They are not independently verified NBA all-time records.
- Play-by-play coverage is incomplete for 2025–26 and does not cover every game in the companion game index. The app therefore excludes 2025–26 from local shot profiles and uses API-backed data for that season.
- The uploaded 2025–26 schedule contains uncompleted Finals placeholders; it should not be used to determine that season's champion. Completed Finals results should be derived from the game-results dataset when complete.
- NBA starters may not be available before a game. The ESPN fallback supplies confirmed starters only after a provider publishes them; it does not create projected lineups.
- Player honors and full season-by-season NBA career records use NBA Stats `PlayerAwards` and `PlayerCareerStats` through `nba_api`. Championship and regular-season MVP counts are deduplicated by season; Finals MVP is not included in the MVP count. The experience tier counts distinct regular seasons with games played. The team timeline is season-level, and a one-franchise badge is withheld unless career-season data is available, covers at least three played seasons, and lists one team.
- Player shot profiles call the most-attempted shot type the favorite. “Best shot” is the highest-accuracy non-dunk type with at least 25 attempts; if none qualifies, the page reports that there is not enough data.
- Local historical database paths can be overridden with `NBA_HISTORY_DB_PATH` and `NBA_PBP_DB_PATH`. The defaults are `v4_web/nba/sql/nba_stats.sqlite` and `v4_web/nba/sql/nba_stats_pbp.sqlite`.

---

## Data sources

| Source | Used for | Key |
|--------|----------|-----|
| football-data.org | PL finished scores, standings | `FOOTBALLDATA_ORG_KEY` in `.env` |
| FPL API | PL upcoming fixtures, team names | None (free) |
| Big Balls Sports | La Liga live scores, fixtures | `BBS_API_KEY` in `.env` |
| FotMob via parse.bot | Lineups, live xG | `PARSE_BOT_KEY` in `.env` |
| TheSportsDB | Player photos, career history | None (key `3`, free) |
| Understat via SoccerData | Historical player-season aggregates for the football profile | Local import; see [`v4_web/football/sql/README.md`](v4_web/football/sql/README.md) |
| NBA stats/live feeds (`nba_api`) | Current NBA scoreboard, live game state, rosters and player profiles | No application key; provider availability/rate limits apply |
| ESPN public NBA endpoints | Confirmed starter fallback when NBA live lineup data is unavailable | No application key; only returns lineups after publication |
| Local NBA SQLite datasets | Historical regular-season box scores, game history and play-by-play shot analysis | Local files; see [`v4_web/nba/sql/README.md`](v4_web/nba/sql/README.md) |

---

## Football model performance and status

Recorded validation shows why the system uses Dixon–Coles before kickoff and
the neural network for live score/time updates. The current per-league
result-driven rating layer is selectively promoted only after a minimum-sample
and proper-scoring-rule gate. Detailed validation numbers and the dated local
status are maintained in [`FOOTBALL_MODEL.md`](FOOTBALL_MODEL.md); those
short-window figures are not a guarantee of future performance.

---

## Environment variables (`.env`)

Copy [`.env.example`](.env.example) to `.env` for local development. The
application loads this file only as a convenience; deployment environments
should provide secrets through their secret manager or environment settings.
Existing environment variables always take precedence.

```
FOOTBALLDATA_ORG_KEY=your_key_here
BBS_API_KEY=your_bbs_key_here
PARSE_BOT_KEY=your_parsebot_key_here   # optional
SOFASCORE_RAPIDAPI_KEY=your_rapidapi_key_here   # legacy v3 sync only
NBA_HISTORY_DB_PATH=v4_web/nba/sql/nba_stats.sqlite       # optional override
NBA_PBP_DB_PATH=v4_web/nba/sql/nba_stats_pbp.sqlite      # optional override
```

Never commit `.env` or real credentials. Any credential that was previously
committed must be revoked and replaced at its provider before production use.
