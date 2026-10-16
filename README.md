# ⚽🏀 Melios Dira Odds — V4.2 Football & NBA Prediction Terminal

> A multi-sport prediction dashboard combining Dixon-Coles football modelling with an NBA win-probability model. It serves football leagues and NBA live scores, predictions, team pages, player profiles, and historical statistics through a FastAPI dashboard.

---

## What it does

The football system predicts match outcomes using two complementary models:

- **Pre-game:** Dixon-Coles bivariate Poisson, calibrated on 4 seasons of xG data across 125 teams and 5 leagues. Outputs home / draw / away probabilities with a scoreline probability matrix.
- **Live (in-game):** A PyTorch neural network (11→40→20→3) that takes over once the match has started, adjusting odds dynamically using the current score, minute, and xG state.

The NBA dashboard has a separate NBA win-probability model and basketball-specific pages. Historical NBA statistics are queried from local SQLite files; current schedules, rosters, player information, and live game state come from the NBA data feeds.

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

---

## Project structure

```
FIFA-WORLDCUP-PREDICTION/
├── v4_backend/
│   ├── historical_scraper.py        # Scrapes 4 seasons of xG from Understat → SQLite
│   ├── dixon_coles_xg.py            # DC MLE optimiser (SLSQP, draw correction)
│   ├── feature_builder.py           # DC strength lookup + TEAM_NAME_ALIASES
│   ├── likelihood_adjustment.py     # Lineup delta → adjusted λ/μ
│   └── notebooks/
│       └── train_v4_priors.ipynb    # Runs the optimiser → outputs v4_priors.json
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
│   ├── feature_builder.py           # (symlink to v4_backend equivalent)
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
│   └── scripts/                     # Build/train/validate scripts (not part of app)
│       ├── build_phase1-4.py        # DC training pipeline phases
│       ├── validate_*.py            # Model validation scripts
│       └── backfill_history.py      # One-time predictions DB backfill
│
├── v4_priors.json                   # Trained DC priors — 125 teams, 5 leagues
├── football_v4.pth                  # Neural net weights
├── v4_historical_data.sqlite        # 4 seasons xG data — training set
├── .env                             # API keys (not committed)
├── README.md                        # This file
├── USER_MANUAL.md                   # Step-by-step setup and usage guide
└── DOCUMENTATION.md                 # Full technical reference
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
| Kept the multi-gigabyte play-by-play SQLite file out of Git and documented how to download and validate it. | The full file is too large to version with the application and should be validated before use. | The app repository stays lighter; shot-profile functionality requires the separately downloaded file. See [`v4_web/nba/sql/README.md`](v4_web/nba/sql/README.md). |

### NBA data boundaries and known limitations

- The historical stats database contains regular-season player and team box scores, game results, schedule data, and odds. Historical NBA queries use read-only SQLite connections and bounded in-process caches.
- The play-by-play file contains shot events used for broad shot-category summaries. Historical shot labels are grouped into broad categories; the calculated "favorite shot" means the most attempted category, not the most accurate one.
- Historical career highs are the largest single-game regular-season values available in this dataset. They are not independently verified NBA all-time records.
- Play-by-play coverage is incomplete for 2025–26 and does not cover every game in the companion game index. The app therefore excludes 2025–26 from local shot profiles and uses API-backed data for that season.
- The uploaded 2025–26 schedule contains uncompleted Finals placeholders; it should not be used to determine that season's champion. Completed Finals results should be derived from the game-results dataset when complete.
- NBA starters may not be available before a game. The ESPN fallback supplies confirmed starters only after a provider publishes them; it does not create projected lineups.
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
| NBA stats/live feeds (`nba_api`) | Current NBA scoreboard, live game state, rosters and player profiles | No application key; provider availability/rate limits apply |
| ESPN public NBA endpoints | Confirmed starter fallback when NBA live lineup data is unavailable | No application key; only returns lineups after publication |
| Local NBA SQLite datasets | Historical regular-season box scores, game history and play-by-play shot analysis | Local files; see [`v4_web/nba/sql/README.md`](v4_web/nba/sql/README.md) |

---

## Model performance (2025/26 season, through GW5)

| League | Correct | Total | Accuracy | Baseline |
|--------|---------|-------|----------|---------|
| La Liga | 2 | 3 | 66.7% | 43% |
| Premier League | 1 | 5 | 20.0% | 43% |

*Small sample — meaningful evaluation expected after GW10.*

---

## Environment variables (`.env`)

```
FOOTBALLDATA_ORG_KEY=your_key_here
BBS_API_KEY=your_bbs_key_here
PARSE_BOT_KEY=your_parsebot_key_here   # optional
NBA_HISTORY_DB_PATH=v4_web/nba/sql/nba_stats.sqlite       # optional override
NBA_PBP_DB_PATH=v4_web/nba/sql/nba_stats_pbp.sqlite      # optional override
```
