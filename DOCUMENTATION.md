# 📐 DOCUMENTATION.md — V4.2 Technical Reference

Full technical documentation for the Melios Dira Odds V4.2 football and NBA
prediction system. Covers the mathematical models, data pipelines, API
integrations, architecture decisions, and known limitations.

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Mathematical models](#2-mathematical-models)
3. [Data pipeline](#3-data-pipeline)
4. [API integrations](#4-api-integrations)
5. [Application architecture](#5-application-architecture)
6. [Pages and routes](#6-pages-and-routes)
7. [Database schema](#7-database-schema)
8. [Key constants and configuration](#8-key-constants-and-configuration)
9. [Model performance](#9-model-performance)
10. [Known issues and limitations](#10-known-issues-and-limitations)
11. [Architecture decisions log](#11-architecture-decisions-log)

---

## 1. System overview

The system has two layers:

**Mathematical engines** — football uses Dixon–Coles (DC) for pre-match
probabilities and an 11→40→20→3 PyTorch classifier for live match-state
probabilities. The NBA dashboard uses Elo before tip-off and a 16-feature
PyTorch classifier during games. Separate, result-driven rating layers can
adjust football DC factors or NBA Elo ratings when their walk-forward
evaluations pass scoring-rule gates; neither retrains neural-network weights.

**Dashboard** — a FastAPI server serving Jinja2 HTML templates. All predictions, pitch rendering, and match data are computed server-side and injected into templates. No client-side JavaScript frameworks; SVG graphics are generated in Python and served as inline strings.

The models have distinct jobs: DC is used before kickoff; the neural network is
used after the match starts. The neural net performed worse than DC in the
recorded pre-game holdout, so its strong internal live-snapshot score is not
used to justify pre-match predictions. See [`FOOTBALL_MODEL.md`](FOOTBALL_MODEL.md)
for the full equations, feature definitions, training procedure and current
adaptive-rating status.

The NBA model has its own lifecycle and feature contract; see
[`NBA_MODEL.md`](NBA_MODEL.md) for its probability equations, all 16 inputs,
training artifacts and temporal test folds, result-driven Elo adjustment, and
documented provenance gaps.

---

## 2. Mathematical models

The football prediction system is documented in detail in
[`FOOTBALL_MODEL.md`](FOOTBALL_MODEL.md). In summary:

- **Pre-match:** fitted Dixon–Coles team attack/defence priors generate expected
  goal rates and a 1X2 scoreline distribution; a bounded history-driven
  attack/concession factor is applied only for leagues where walk-forward
  validation accepts it.
- **Live:** the 11-feature neural classifier consumes score, minute and
  normalized DC-strength features. It does not consume xG. Stored validation
  indicates that it belongs in the live role, not the pre-match role.
- **Optional lineup panel:** manually assigned key-player weights adjust
  expected goals when confirmed lineups are available; this remains a
  heuristic, separate from both trained models.

The NBA prediction system is documented in detail in
[`NBA_MODEL.md`](NBA_MODEL.md). In summary, pregame probabilities come from
Elo ratings with a 100-point home-court advantage; live probabilities come
from a calibrated 16-feature neural classifier. A separate chronological Elo
replay can replace the baseline ratings after at least 30 settled games and
only when Brier score improves without worsening log-loss. The Elo replay
does not retrain the live classifier.

### 2.1 Monte Carlo simulation

**Purpose:** Simulate tournament outcomes and season finishes probabilistically.

**Champions League simulator (`simulate_cl_tournament`):**
- Input: 16 teams selected by the user from any of the 5 leagues
- Each simulation: random draw → R16 → QF → SF → Final
- Each match: sample outcome from DC win/draw/loss probabilities (γ=1.0, neutral venue)
- Draws go to a 50/50 penalty shootout
- Output: win probability per team, semi-final/quarter-final appearance rates, most probable bracket path
- 10,000 simulations, ~0.3 seconds

**PL season simulator (`simulate_pl_season`):**
- Input: current standings from fd.org + remaining fixtures from FPL
- Each simulation: sample a scoreline from Poisson(λ), Poisson(μ) for every remaining fixture, accumulate points
- Output: median final table with P10/P90 range, title/top 4/relegation probabilities per team
- 10,000 simulations, ~1.5 seconds

---

## 3. Data pipeline

### 3.1 Historical training data

**Source:** Understat via the `soccerdata` Python library.

**Script:** `v4_backend/historical_scraper.py`

**Output:** `v4_historical_data.sqlite` with table `matches_xg`:
```
league, season, date, home_team, away_team,
home_goals, away_goals, home_xg, away_xg
```

**Coverage:** 4 seasons (2021/22–2024/25) across all Big 5 leagues; the local
training ledger and feature set contain approximately 7,156 matches.

### 3.2 Prior training

**Script:** `v4_backend/notebooks/train_v4_priors.ipynb`, using
`v4_backend/dixon_coles_xg.py`.

**Method:**
1. Load `matches_xg` from SQLite
2. Set home advantage to total home xG / total away xG and apply match weight
   `exp(-0.0018 × days_ago)`.
3. Run SLSQP per league to fit α, β and ρ.
4. Enforce mean(α) = 1 and bounds on α, β and ρ.
5. Save to `v4_priors.json`

The objective combines continuous xG Poisson log-likelihood terms with a
Dixon–Coles low-score correction based on actual integer scores. Full
equations, constraints, fallback behavior and the runtime probability
calculation are in [`FOOTBALL_MODEL.md`](FOOTBALL_MODEL.md).

**Output format:**
```json
{
  "ENG-Premier League": {
    "meta": { "gamma_home_advantage": 1.236, "rho_draw_correction": -0.003 },
    "teams": {
      "Arsenal": { "alpha": 1.327, "beta": 0.873 },
      ...
    }
  }
}
```

The figures shown are the rounded current Premier League entries. The fitted
\(\rho\) is not the separate `DRAW_PROPENSITY` 1X2 adjustment.

### 3.3 Neural net training

**Dataset builder:** `v4_backend/build_v4_dataset.py`

**Training notebook:** `v4_backend/notebooks/train_v4_neural.ipynb`

The builder creates 5-minute snapshots from completed league scores. Since the
historical ledger has no goal times, it approximates events at minutes 44 and
75. The recorded parquet contains 140,620 snapshots from 7,156 matches across
five leagues and seasons 2021/22–2024/25. The notebook splits by `match_id`
(60/20/20), scales features using training rows only, trains the 11→40→20→3
network with weighted cross-entropy, selects a validation temperature, and
saves `football_v4.pth` plus `scaler_v4.pkl`. It does not train from live xG.
See [`FOOTBALL_MODEL.md`](FOOTBALL_MODEL.md) for all parameters and recorded
validation results.

### 3.4 Live data flow (runtime)

```
FPL API (no key)
  → get_upcoming_fixtures()
  → Fixture strip on match page
  → Prediction job: log upcoming PL matches

Big Balls Sports (BBS_API_KEY)
  → La Liga fixtures, live scores, match detail
  → Fixture strip on La Liga match page
  → Prediction job: log upcoming La Liga matches + results

football-data.org (FOOTBALLDATA_ORG_KEY)
  → Supported competition fixtures and finished scores (PL, Serie A, Bundesliga)
  → Prediction job: log pre-match estimates and settle results

FotMob via parse.bot (PARSE_BOT_KEY)
  → Confirmed lineups (home/away starting XI + formation)
  → Live scores during match (xG, scoreline, status; xG is not a neural-net input)
  → Note: credit-based, gets exhausted during development

TheSportsDB (key "3", free)
  → Player cutout photos
  → Player career history (former teams + badges)
```

---

## 4. API integrations

### 4.1 football-data.org

**Base URL:** `https://api.football-data.org/v4`

**Auth:** `X-Auth-Token: {FOOTBALLDATA_ORG_KEY}` header

**Endpoints used:**
| Endpoint | Purpose |
|----------|---------|
| `GET /competitions/PL/matches?status=FINISHED&season={current}` | PL finished scores for the current season |
| `GET /competitions/PL/standings?season={current}` | Current PL table (for season simulator) |
| `GET /matches/{match_id}` | Single match detail (not reliable on free tier) |

**Key facts:**
- Free tier: 10 requests/minute, PL only
- Uses the season start year: 2026/27 → `season=2026`; the client selects it dynamically
- Returns team names with `FC` suffix (e.g. `"Arsenal FC"`) — stripped in `norm()` before matching
- Score updates are delayed by several hours on the free tier

**Client:** `v4_web/footballdata.py`

---

### 4.2 FPL API (Fantasy Premier League)

**Base URL:** `https://fantasy.premierleague.com/api`

**Auth:** None

**Endpoints used:**
| Endpoint | Purpose |
|----------|---------|
| `GET /bootstrap-static/` | Team names, player list |
| `GET /fixtures/` | All PL fixtures with kickoff times |

**Key facts:**
- Completely free, no key required
- Occasionally returns 403 (IP rate limit) — the client (`fpl.py`) serves stale cache when this happens
- Kickoff times are in UTC; displayed in EAT (UTC+3) throughout the UI

**Client:** `v4_web/fpl.py`

---

### 4.3 Big Balls Sports (BBS)

**Base URL:** `https://api.bigballssports.com/v1`

**Auth:** `x-api-key: {BBS_API_KEY}` header

**Endpoints used:**
| Endpoint | Purpose |
|----------|---------|
| `GET /matches?sport=football&league=laliga` | La Liga upcoming + live fixtures |
| `GET /matches/{uuid}` | Single La Liga match detail (live score, events) |
| `GET /scores?sport=football&league=laliga` | Today's La Liga scores |

**Key facts:**
- Used exclusively for La Liga (PL endpoints exist but not yet integrated)
- Returns BBS-specific UUIDs (prefixed as `bbs_` in our system)
- The live poll endpoint `/live/bbs_{uuid}` hits BBS every 30 seconds during a live match
- Stale cache served if BBS is down (TTL = 5 minutes)

**Client:** `v4_web/bbs.py`

---

### 4.4 FotMob via parse.bot

**Base URL:** `https://parse.bot/api`

**Auth:** `Authorization: Bearer {PARSE_BOT_KEY}` header

**Endpoints used:**
| Endpoint | Purpose | Credits |
|----------|---------|---------|
| `get_lineup` | Starting XI + formation | 1/call |
| `get_matches_by_date` | Date cache (all leagues) | 3/call |
| `get_match_xg` | Live xG data | 0 (often 402) |

**Key facts:**
- Credit-based — burns quickly during development (get_matches_by_date = 3 credits × many hot reloads)
- Without credits: lineup pitch shows dots; xG chart falls back to DC expected goals
- The date cache is stored in `_lineup_cache` (in-memory only — restarts clear it and burn credits)

**Status as of September 2026:** Credits exhausted. FPL/BBS cover most functionality. Lineups not available until credits are topped up or a replacement API is found.

**Client:** `v4_web/fotmob.py`

---

### 4.5 TheSportsDB

**Base URL:** `https://www.thesportsdb.com/api/v1/json/3`

**Auth:** Key `3` embedded in URL (free tier)

**Endpoints used:**
| Endpoint | Purpose |
|----------|---------|
| `GET /searchplayers.php?p={name}` | Player search → photo, bio, ID |
| `GET /lookupformerteams.php?id={player_id}` | Career history + team badges |

**Key facts:**
- Free tier has no rate limit for non-commercial use
- Returns player cutout images (transparent background) at `strCutout`
- Career history includes team badge URLs, join/departure years, appearances, goals
- Coverage is community-maintained — major European club players are well covered

**Client:** `v4_web/thesportsdb.py`

---

## 5. Application architecture

### 5.1 File structure

```
v4_web/
├── main.py              # FastAPI app, lifespan startup and routes
├── constants.py         # Shared constants: EAT, DRAW_PROPENSITY, LEAGUE_FILTERS, LEAGUE_MAP
├── routes/
│   └── match.py         # /match, /live/{id}, /api/simulate/* (APIRouter)
├── [data clients]       # bbs.py, footballdata.py, fotmob.py, fpl.py, thesportsdb.py
├── model_updates.py     # Walk-forward football factors and NBA Elo evaluation
├── [model modules]      # lineup_adjustment.py, simulate.py, feature_builder.py
├── [render modules]     # scoreline_matrix.py, timeline.py, utils.py
├── [persistence]        # predictions.py, prediction_job.py
└── templates/           # Jinja2 HTML
```

### 5.2 Startup sequence (`lifespan` in `main.py`)

1. Import-time bootstrap loads `football_v4.pth` into `FootballWinProbNet`
2. Load scaler and temperature `nn_T` from pickle
3. Load `v4_priors.json` into `priors_db` dict
4. Build `dc_lookup` (DCStrengthLookup from feature_builder)
5. Initialise Jinja2 `templates` object
6. Build `LEAGUE_CONTEXTS` dict (theme colours, league name, gamma per league)
7. Initialize the prediction database; evaluate/restore result-driven football
   factors from settled results. This does not train the neural network.
8. Load NBA model artifacts and call `nba_setup(...)`; logged NBA results are
   replayed to evaluate Elo updates, not to fine-tune neural-network weights.
   See [`NBA_MODEL.md`](NBA_MODEL.md).
9. Call `match_setup(...)` — passes model globals into `routes/match.py`
10. Warm FPL cache (bootstrap + fixtures)
11. Schedule the prediction job (startup and every 30 minutes)
12. Yield (server is live)

### 5.3 Match route globals

The `routes/match.py` module needs access to `priors_db`, `nn_model`, etc. which are initialised during startup in `main.py`. These are passed via `match_setup()` which sets module-level variables in `routes/match.py`:

```python
# main.py lifespan
match_setup(priors_db, nn_model, nn_scaler, nn_T, dc_lookup,
            templates, TEAM_NAME_ALIASES, LEAGUE_KEY, LEAGUE_CONTEXTS)

# routes/match.py
priors_db = {}   # populated by setup()
nn_model  = None # populated by setup()
# ...

def setup(p_db, model, ...):
    global priors_db, nn_model, ...
    priors_db = p_db
    # ...
```

`dc_pregame()` and `nn_live()` are helper functions in `routes/match.py` that access these module-level refs directly (they don't receive a `Request` object and therefore can't use `request.app.state`).

### 5.4 Background prediction job

`prediction_job.py` runs as an `asyncio` background task, triggered every 30 minutes by an `asyncio.create_task` in `main.py`'s lifespan.

**Each cycle:**
1. Fetch upcoming PL fixtures (FPL), La Liga (BBS), and Serie A/Bundesliga
   (football-data.org) → log new DC predictions to SQLite.
2. Where available, fetch confirmed lineups and save a separate heuristic
   lineup-adjusted snapshot. Serie A lineup adjustments are disabled.
3. Check unresolved past predictions (kickoff > 2 hours ago) → fill in final
   scores from football-data.org or BBS, depending on competition.
4. Save match snapshots (score, status, colours) for the match/history pages.

**Result resolution flow:**
```
Prediction DB (unresolved, kickoff > 2hr ago)
  PL / Serie A / Bundesliga → find_finished_match_by_teams() → fd.org competition cache
  La Liga                   → bbs.get_match_detail(uuid) → BBS API
  → log_result(match_id, result, home_goals, away_goals)
```

---

## 6. Pages and routes

### 6.1 Route map

| Method | Path | Handler | File |
|--------|------|---------|------|
| GET | `/` | `home_page` | main.py |
| GET | `/match` | `match` | routes/match.py |
| GET | `/live/{match_id}` | `live_poll` | routes/match.py |
| GET | `/crest/{team_name}` | `crest_proxy` | main.py |
| GET | `/teams` | `teams_page` | main.py |
| GET | `/teams/laliga` | `teams_laliga` | main.py |
| GET | `/team/{team_id}` | `team_profile` | main.py |
| GET | `/player` | `player_page` | main.py |
| GET | `/history` | `history_page` | main.py |
| GET | `/tournament` | `tournament_page` | main.py |
| GET | `/league/{league_key}` | redirect | main.py |
| POST | `/api/simulate/cl` | `api_simulate_cl` | routes/match.py |
| POST | `/api/simulate/pl` | `api_simulate_pl` | routes/match.py |
| GET | `/admin/run-predictions` | `admin_run` | main.py |
| GET | `/admin/cache-status` | `admin_cache` | main.py |

### 6.2 League routing

```
/league/pl        → RedirectResponse("/match")
/league/laliga    → RedirectResponse("/match?league=laliga")
/match?league=pl  → PL match page (FPL + fd.org)
/match?league=laliga → La Liga match page (BBS)
/match?match_id=bbs_{UUID}&league=laliga → specific La Liga match
/match?match_id={fd_id}&league=pl        → specific PL match
```

### 6.3 Match page data flow

```
GET /match?league=laliga&match_id=bbs_abc123

1. Parse league → active_league = "ESP-La Liga", active_ctx_key = "laliga"
2. Build fixture strip:
   - BBS today's matches (with live scores)
   - BBS upcoming fixtures
   - Dedup by bbs_id
   - Fallback: FotMob date cache if BBS empty
3. Resolve featured match:
   - match_id starts with "bbs_" → BBS get_match_detail()
   - match_id is numeric → fd.org finished cache
   - No match_id → most recent BBS match (today's if available)
4. Compute DC odds: dc_pregame(home, away, league)
5. Fetch lineup (FotMob if credits available)
6. If lineup available: compute adj_dc odds via compute_lineup_adjusted_odds()
7. Generate pitch SVG with jersey shapes + player names
8. Generate scoreline matrix SVG
9. Render match.html with all context
```

---

## 7. Database schema

**File:** `v4_web/data/predictions.db` (SQLite)

**Table: `predictions`**

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | TEXT PK | Provider fixture ID; La Liga history IDs use an `ll_` prefix |
| `league` | TEXT | `"pl"` or `"laliga"` |
| `season` | INT | Season start year used for evaluation grouping |
| `home_team` | TEXT | Home team name |
| `away_team` | TEXT | Away team name |
| `kickoff_utc` | TEXT | ISO 8601 UTC kickoff time |
| `pre_dc_home` | REAL | DC home win probability (%) |
| `pre_dc_draw` | REAL | DC draw probability (%) |
| `pre_dc_away` | REAL | DC away win probability (%) |
| `pre_dc_lam` | REAL | Pre-match home expected goals |
| `pre_dc_mu` | REAL | Pre-match away expected goals |
| `pre_rating_lam_factor` | REAL | Result-driven multiplier applied to home expected goals |
| `pre_rating_mu_factor` | REAL | Result-driven multiplier applied to away expected goals |
| `rating_version` | TEXT | Whether the logged estimate used static or accepted rating factors |
| `adj_home` / `adj_draw` / `adj_away` | REAL | Optional confirmed-lineup probabilities (%) |
| `adj_lam` / `adj_mu` | REAL | Optional lineup-adjusted expected goals |
| `absent_home` | TEXT | JSON list of absent home players |
| `absent_away` | TEXT | JSON list of absent away players |
| `actual_result` | TEXT | `"H"`, `"D"`, or `"A"` (filled post-match) |
| `actual_hg` | INT | Final home goals |
| `actual_ag` | INT | Final away goals |
| `pre_logged_at` | TEXT | Timestamp when prediction was logged |
| `result_logged_at` | TEXT | Timestamp when result was filled in |

The same database contains `football_rating_state` (accepted factor snapshots
keyed by league, season and priors hash) and `model_update_evaluations`
(sample size, baseline/candidate Brier and log-loss, candidate accuracy,
acceptance flag and parameter values). See [`FOOTBALL_MODEL.md`](FOOTBALL_MODEL.md)
for the formulas and the dated local evaluation.

---

## 8. Key constants and configuration

**`constants.py`**

```python
EAT = timezone(timedelta(hours=3))   # East Africa Time (Nairobi)
DRAW_PROPENSITY = 0.10               # additional 1X2 draw-propensity adjustment

LEAGUE_FILTERS = {                    # FotMob date cache name matching
    "pl":         ["premier league", "england", "eng"],
    "laliga":     ["laliga", "la liga", "primera", "spain", "esp"],
    "bundesliga": ["bundesliga", "germany", "ger"],
    "seriea":     ["serie a", "calcio", "italy", "ita"],
    "ligue1":     ["ligue 1", "ligue1", "france", "fra"],
}

LEAGUE_MAP = {                        # URL param → metadata
    "pl":    {"priors": "ENG-Premier League", "comp": "PL", "name": "Premier League"},
    "laliga":{"priors": "ESP-La Liga",        "comp": "PD", "name": "La Liga"},
    ...
}
```

**`main.py` globals (set at startup)**

```python
priors_db         # dict — full v4_priors.json contents
nn_model          # FootballWinProbNet — loaded from football_v4.pth
nn_scaler         # sklearn StandardScaler — loaded from pickle
nn_T              # float — temperature scaling factor
dc_lookup         # DCStrengthLookup — normalized neural features and strength data
templates         # Jinja2 Templates instance
TEAM_NAME_ALIASES # dict — maps variant names to canonical names
LEAGUE_CONTEXTS   # dict — per-league theme (colours, name, gamma)
```

---

## 9. Model performance and current status

The stored 2526 holdout report found 49.8% accuracy and 19.3% draw recall for
pre-match Dixon–Coles, compared with 49.3% accuracy and 7.0% draw recall for
the neural network evaluated at minute 0. The neural network's 71.3% accuracy
is from a grouped internal test of in-match snapshots, not a pre-match forecast
or a set of independent matches. Its production role is live match-state
prediction. These are historical validation figures; see
[`FOOTBALL_MODEL.md`](FOOTBALL_MODEL.md) for training/test definitions, proper
scoring metrics, limitations and the latest local adaptive-rating snapshot.

The local result-driven update snapshot dated 2026-11-03 shows Serie A
accepted, Premier League and La Liga rejected because both proper scoring
metrics worsened, and Bundesliga below the 30-settled-game minimum. These
short-window outcomes can change when more results are settled and are not a
general guarantee of model performance.

The NBA model's saved validation metrics and three expanding-window
season-held-out folds are documented in [`NBA_MODEL.md`](NBA_MODEL.md).
Its result-driven Elo evaluation was accepted locally on 2026-11-06 using 62
settled regular-season games (Brier 0.25558 → 0.25274; log-loss 0.71609 →
0.70896). This is an early-season sample, not a long-term performance
guarantee. The training code and raw training manifest are not present in this
checkout, so the exact final-model train/validation membership and raw data
source cannot be independently verified here.

---

## 10. Known issues and limitations

### 10.1 FotMob parse.bot credits exhausted

**Impact:** Lineup pitch shows placeholder jersey shapes instead of confirmed player names. Live xG is unavailable. Adjusted odds not shown.

**Workaround:** Top up parse.bot credits, or integrate an alternative lineup source (API-Football at 100 req/day free, or BBS PL endpoints if available).

**What still works:** All DC pre-game odds, scoreline matrix, match timeline (for finished matches), fixture strip, history page.

### 10.2 fd.org free tier score delays

**Impact:** PL match final scores appear in the history page with a delay of several hours after the match ends. The free tier is not real-time.

**Workaround:** Results are checked every 30 minutes automatically. All results are eventually populated.

### 10.3 get_matches_by_date burning parse.bot credits

**Root cause:** FotMob's date cache (`get_matches_by_date`) costs 3 credits per call and is fetched on every hot reload during development. With frequent restarts (due to debugging), this depleted credits rapidly.

**Mitigation implemented:** The date cache uses an in-memory TTL. However the cache is lost on server restart.

**Permanent fix needed:** Persist the date cache to disk (e.g. a JSON file) so restarts don't trigger a fresh API call.

### 10.4 fd.org team name mismatch

**Root cause:** fd.org returns `"Arsenal FC"`, `"Chelsea FC"` etc. with an `FC` suffix. Our predictions DB stores `"Arsenal"`, `"Chelsea"`. The `norm()` function in `find_finished_match_by_teams` was not stripping the `FC` suffix.

**Fix applied:** Added regex strip `\s+(fc|afc|...)$` to `norm()` in `footballdata.py`.

### 10.5 FPL bootstrap 403

**Impact:** Fixture strip occasionally shows stale or empty data.

**Behaviour:** FPL rate-limits by IP. The client serves stale cache when this happens. Recovers automatically within a few hours.

---

## 11. Architecture decisions log

| Decision | Chosen approach | Alternatives considered | Reason |
|----------|----------------|------------------------|--------|
| Pre-game model | Dixon-Coles bivariate Poisson | Logistic regression, ELO | Principled, interpretable, well-validated in literature |
| Live model | PyTorch neural net (score+time only) | DC with live xG update, Markov chain | Season-agnostic features eliminated structural overfitting |
| Web framework | FastAPI + Jinja2 | Streamlit, Dash, React | Full control over HTML/SVG; server-side rendering; no JS build step |
| Jersey rendering | SVG path in Python | Emoji, CSS shapes | Kit colours, stripes, initials all in one SVG; no images needed |
| La Liga data | BBS API | FotMob, football-data.org | BBS has La Liga live scores; fd.org free tier is PL-only |
| Match route extraction | FastAPI APIRouter | Stay in main.py, separate process | main.py was 2,218 lines; router halved it to 930 with no URL changes |
| Globals in routes/match.py | Module-level refs via setup() | request.app.state, state.py | dc_pregame/nn_live don't receive Request; module refs are simplest |
| Constants deduplication | constants.py | Leave inline | LEAGUE_FILTERS was defined 4× across 2 files; single source eliminates drift |
| Prediction storage | SQLite | JSON files, PostgreSQL | Lightweight, zero-config, queryable, sufficient for one-user use |
| Player profiles | TheSportsDB (free, key 3) | FotMob, Transfermarkt scraping | Free with no rate limit; has photos, career history, covers all 5 leagues |
