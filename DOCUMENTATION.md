# 📐 DOCUMENTATION.md — V4.2 Technical Reference

Full technical documentation for the Melios Dira Odds V4.2 football prediction system. Covers the mathematical models, data pipeline, API integrations, architecture decisions, and known limitations.

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

**Mathematical engine** — a two-component prediction model:
- Dixon-Coles (DC) bivariate Poisson for pre-game probability estimation
- PyTorch neural network for live in-game probability updating

**Dashboard** — a FastAPI server serving Jinja2 HTML templates. All predictions, pitch rendering, and match data are computed server-side and injected into templates. No client-side JavaScript frameworks; SVG graphics are generated in Python and served as inline strings.

The two components complement each other: DC is statistically rigorous and interpretable, the neural net is fast and responsive to live match state. Neither is used outside its intended context.

---

## 2. Mathematical models

### 2.1 Dixon-Coles bivariate Poisson (pre-game)

**Purpose:** Compute home / draw / away probabilities before kickoff.

**Inputs:** Team attack (α) and defence (β) ratings from `v4_priors.json`, plus the league home advantage parameter (γ).

**How it works:**

Each team's expected goals are modelled as:

```
λ (home expected goals) = α_home × β_away × γ
μ (away expected goals) = α_away × β_home
```

The probability of each scoreline (h goals, a goals) follows a bivariate Poisson distribution:

```
P(h, a) = Poisson(h | λ) × Poisson(a | μ)
```

With a low-score correction (Dixon-Coles ρ parameter) applied to the 0-0, 1-0, 0-1, 1-1 cells to correct for the observed over-frequency of low-scoring draws:

```
P(0,0) *= max(1 - λ×μ×ρ, 1e-5)
P(1,0) *= max(1 + μ×ρ,   1e-5)
P(0,1) *= max(1 + λ×ρ,   1e-5)
P(1,1) *= max(1 - ρ,     1e-5)
```

ρ is set globally to `DRAW_PROPENSITY = 0.10`, tuned via elbow test on holdout data.

Home, draw, and away probabilities are then the sums of the lower triangle, diagonal, and upper triangle of the 9×9 scoreline matrix respectively.

**Prior estimation:** α and β values are estimated by maximum likelihood estimation (MLE) using SciPy's SLSQP optimiser, trained on 4 seasons of xG data (from Understat via `soccerdata`). A time-decay factor weights more recent matches more heavily. A sum-to-1 constraint on attack ratings is enforced to ensure identifiability.

**Coverage:** 5 leagues × ~25 teams each = 125 teams total.

| League | Teams | γ (home advantage) |
|--------|-------|-------------------|
| ENG-Premier League | 26 | 1.204 |
| ESP-La Liga | 25 | 1.341 |
| GER-Bundesliga | 24 | 1.261 |
| ITA-Serie A | 26 | 1.177 |
| FRA-Ligue 1 | 24 | 1.223 |

---

### 2.2 Lineup adjustment

**Purpose:** Adjust DC odds when key players are confirmed absent from the starting XI.

**How it works:**

For each confirmed absent player, their contribution to team strength is estimated from the DC lookup table (`feature_builder.py` / `DCStrengthLookup`). The adjustment is applied as a fractional reduction to λ or μ before the scoreline matrix is computed:

```
adj_λ = λ × (1 - Σ Δ_attack_i × λ_a)
adj_μ = μ × (1 - Σ Δ_defence_i × λ_d)
```

Where `λ_a` and `λ_d` are hyperparameters controlling the dampening (set conservatively to avoid overcorrecting on single-player absences). Implementation in `lineup_adjustment.py` and `compute_lineup_adjusted_odds()`.

---

### 2.3 Neural network (live in-game)

**Purpose:** Update win/draw/lose probabilities dynamically during a live match.

**Architecture:** 11→40→20→3 feed-forward network with ReLU activations and softmax output.

**Input features (11):**
1. Home goals scored
2. Away goals scored
3. Goal difference (home - away)
4. Match minute (normalised 0–1)
5. Remaining time fraction
6. Home xG
7. Away xG
8. xG difference
9. Home xG per minute
10. Away xG per minute
11. Game state indicator (winning / drawing / losing from home perspective)

**Output:** 3 probabilities (home win, draw, away win) that sum to 1.

**Training:** ~140,000 match snapshots from the 2024/25 season. Trained with cross-entropy loss, Adam optimiser. Internal accuracy ~71.3%.

**Key design decision:** The neural net is deliberately restricted to **score and time features only** — no team identity, no DC ratings, no pre-match information. This makes it season-agnostic and avoids the structural overfitting that plagued earlier versions (where non-score features encoded season-specific patterns, causing a ~22-point internal/holdout gap).

**Transition:** At kickoff, DC probabilities are displayed. Once the match starts (minute > 0), the neural net takes over and updates every 30 seconds via the `/live/{match_id}` polling endpoint. The DC odds remain visible as the pre-game baseline for comparison.

---

### 2.4 Monte Carlo simulation

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

**Coverage:** 4 seasons (2021/22–2024/25) across all Big 5 leagues. ~7,156 matches total.

### 3.2 Prior training

**Script:** `v4_backend/notebooks/train_v4_priors.ipynb`

**Method:**
1. Load `matches_xg` from SQLite
2. Apply exponential time-decay: `weight = exp(-decay_rate × days_ago)`
3. Run SLSQP MLE optimiser per league to find α, β, γ, ρ
4. Enforce Σ α = N (sum-to-1 normalisation)
5. Save to `v4_priors.json`

**Output format:**
```json
{
  "ENG-Premier League": {
    "meta": { "gamma_home_advantage": 1.204, "rho_draw_correction": 0.10 },
    "teams": {
      "Arsenal": { "alpha": 1.326, "beta": 0.886 },
      ...
    }
  }
}
```

### 3.3 Neural net training

**Script:** `v4_web/scripts/build_train_v4.py`

**Method:**
1. Load finished PL matches from fd.org for 2024/25
2. Compute match snapshots at each minute using cumulative goals and xG
3. Train 11→40→20→3 network on 80/20 train/val split
4. Save best checkpoint to `football_v4.pth`

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
  → Finished PL match scores (season=2026)
  → Prediction job: fill in PL results

FotMob via parse.bot (PARSE_BOT_KEY)
  → Confirmed lineups (home/away starting XI + formation)
  → Live scores during match (xG, scoreline, status)
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
| `GET /competitions/PL/matches?status=FINISHED&season=2026` | PL finished scores for 2025/26 |
| `GET /competitions/PL/standings?season=2026` | Current PL table (for season simulator) |
| `GET /matches/{match_id}` | Single match detail (not reliable on free tier) |

**Key facts:**
- Free tier: 10 requests/minute, PL only
- Uses end-year for season parameter: 2025/26 → `season=2026`
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
├── main.py              # FastAPI app, lifespan startup, all routes except /match
├── constants.py         # Shared constants: EAT, DRAW_PROPENSITY, LEAGUE_FILTERS, LEAGUE_MAP
├── routes/
│   ├── __init__.py      # Empty — makes routes/ a Python package
│   └── match.py         # /match, /live/{id}, /api/simulate/* (APIRouter)
├── [data clients]       # bbs.py, footballdata.py, fotmob.py, fpl.py, thesportsdb.py
├── [model modules]      # lineup_adjustment.py, simulate.py, feature_builder.py
├── [render modules]     # scoreline_matrix.py, timeline.py, utils.py
├── [persistence]        # predictions.py, prediction_job.py
└── templates/           # Jinja2 HTML
```

### 5.2 Startup sequence (`lifespan` in `main.py`)

1. Load `football_v4.pth` into a `torch.nn.Module` (`FootballNet`)
2. Load scaler and temperature `nn_T` from pickle
3. Load `v4_priors.json` into `priors_db` dict
4. Build `dc_lookup` (DCStrengthLookup from feature_builder)
5. Initialise Jinja2 `templates` object
6. Build `LEAGUE_CONTEXTS` dict (theme colours, league name, gamma per league)
7. Call `match_setup(...)` — passes all globals into `routes/match.py` as module-level refs
8. Warm FPL cache (bootstrap + fixtures)
9. Schedule background prediction job (every 30 minutes)
10. Yield (server is live)

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
1. Fetch upcoming PL fixtures (FPL) → log new predictions to SQLite
2. Fetch upcoming La Liga fixtures (BBS) → log new predictions to SQLite
3. Check unresolved past predictions (kickoff > 2 hours ago) → fill in actual scores from fd.org (PL) or BBS (La Liga)
4. Save match snapshots (pre-game odds, lineup, colours) for the history page

**Result resolution flow:**
```
Prediction DB (unresolved, kickoff > 2hr ago)
  PL matches  → find_finished_match_by_teams() → fd.org cache (season=2026)
  La Liga     → bbs.get_match_detail(uuid) → BBS API
  → log_result(match_id, result, h_score, a_score)
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
| `match_id` | TEXT PK | fd.org fixture ID (PL) or `bbs_{uuid}` (La Liga) |
| `league` | TEXT | `"pl"` or `"laliga"` |
| `home_team` | TEXT | Home team name |
| `away_team` | TEXT | Away team name |
| `kickoff_utc` | TEXT | ISO 8601 UTC kickoff time |
| `pre_dc_home` | REAL | DC home win probability (%) |
| `pre_dc_draw` | REAL | DC draw probability (%) |
| `pre_dc_away` | REAL | DC away win probability (%) |
| `pre_adj_home` | REAL | Lineup-adjusted home probability (%) |
| `pre_adj_draw` | REAL | Lineup-adjusted draw probability (%) |
| `pre_adj_away` | REAL | Lineup-adjusted away probability (%) |
| `absent_home` | TEXT | JSON list of absent home players |
| `absent_away` | TEXT | JSON list of absent away players |
| `actual_result` | TEXT | `"H"`, `"D"`, or `"A"` (filled post-match) |
| `actual_h_score` | INT | Final home goals |
| `actual_a_score` | INT | Final away goals |
| `pre_logged_at` | TEXT | Timestamp when prediction was logged |
| `result_logged_at` | TEXT | Timestamp when result was filled in |

---

## 8. Key constants and configuration

**`constants.py`**

```python
EAT = timezone(timedelta(hours=3))   # East Africa Time (Nairobi)
DRAW_PROPENSITY = 0.10               # DC ρ parameter (tuned via elbow test)

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
nn_model          # FootballNet — loaded from football_v4.pth
nn_scaler         # sklearn StandardScaler — loaded from pickle
nn_T              # float — temperature scaling factor
dc_lookup         # DCStrengthLookup — for lineup adjustment
templates         # Jinja2 Templates instance
TEAM_NAME_ALIASES # dict — maps variant names to canonical names
LEAGUE_CONTEXTS   # dict — per-league theme (colours, name, gamma)
```

---

## 9. Model performance

### 9.1 Pre-season validation (2024/25 holdout)

Dixon-Coles pre-game:
- Overall accuracy: ~49.8% (baseline for correct-outcome prediction ~43%)
- Draw recall: 19.3% (historically difficult — most models predict 0%)
- Accuracy improves at higher confidence bands (>60% confidence → ~58% accuracy)

Neural net (live in-game, 2024/25 training set):
- Internal accuracy: ~71.3% (on held-out snapshots from the training season)
- Note: this is in-game accuracy — given a match snapshot (score + time), predict final outcome

### 9.2 Live 2025/26 season (through GW5)

| League | DC Correct | Total | Accuracy | Baseline |
|--------|-----------|-------|----------|---------|
| La Liga | 2 | 3 | 66.7% | 43% |
| Premier League | 1 | 5 | 20.0% | 43% |

*8 games is too small a sample for conclusions. Evaluating at GW10 (October 2026).*

**Known issue:** PL draw recall is 0/3 — the model predicted H or A for all three actual draws in GW1–5. This matches the pre-season validation pattern and suggests `DRAW_PROPENSITY` may need upward adjustment for PL specifically.

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

