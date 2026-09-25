# ⚽ Melios Dira Odds — V4.2 Football Prediction Terminal

> A production-grade football match prediction suite combining Dixon-Coles statistical modelling with a live neural network, served through a FastAPI dashboard. Covers the Premier League and La Liga in real-time, with priors built for all Big 5 leagues.

---

## What it does

The system predicts football match outcomes using two complementary models:

- **Pre-game:** Dixon-Coles bivariate Poisson, calibrated on 4 seasons of xG data across 125 teams and 5 leagues. Outputs home / draw / away probabilities with a scoreline probability matrix.
- **Live (in-game):** A PyTorch neural network (11→40→20→3) that takes over once the match has started, adjusting odds dynamically using the current score, minute, and xG state.

The dashboard is a local FastAPI server with Jinja2 templates, a dark-mode UI, and an SVG pitch that renders confirmed lineups with team-coloured jerseys.

---

## Quick start

```bash
# 1. Install dependencies
pip install fastapi uvicorn jinja2 pandas numpy scipy torch python-dotenv requests

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
│   ├── main.py                      # FastAPI app — all routes except /match
│   ├── constants.py                 # EAT, DRAW_PROPENSITY, LEAGUE_FILTERS, LEAGUE_MAP
│   ├── routes/
│   │   └── match.py                 # /match, /live/{id}, /api/simulate/* routes
│   ├── bbs.py                       # Big Balls Sports API client (La Liga)
│   ├── footballdata.py              # football-data.org client (PL scores/fixtures)
│   ├── fotmob.py                    # FotMob via parse.bot (lineups, live xG)
│   ├── fpl.py                       # FPL API client (PL fixtures, team map)
│   ├── thesportsdb.py               # TheSportsDB client (player photos, career)
│   ├── predictions.py               # SQLite predictions DB (log + retrieve)
│   ├── prediction_job.py            # Background job — logs predictions + results
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

---

## Data sources

| Source | Used for | Key |
|--------|----------|-----|
| football-data.org | PL finished scores, standings | `FOOTBALLDATA_ORG_KEY` in `.env` |
| FPL API | PL upcoming fixtures, team names | None (free) |
| Big Balls Sports | La Liga live scores, fixtures | `BBS_API_KEY` in `.env` |
| FotMob via parse.bot | Lineups, live xG | `PARSE_BOT_KEY` in `.env` |
| TheSportsDB | Player photos, career history | None (key `3`, free) |

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
```
