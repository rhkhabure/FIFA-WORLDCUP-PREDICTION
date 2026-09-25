# 📘 User Manual: Melios Dira Odds — V4.2

This manual covers everything you need to set up, run, and use the V4.2 football prediction dashboard from scratch.

---

## Prerequisites

- Python 3.11 or later
- An NVIDIA GPU with CUDA (recommended for neural net inference — CPU works but is slower)
- API keys for football-data.org and Big Balls Sports (see Step 2)

---

## Step 1: Install dependencies

```bash
pip install fastapi uvicorn jinja2 pandas numpy scipy torch torchvision \
            python-dotenv requests soccerdata
```

If you have a CUDA-capable GPU, install the CUDA version of PyTorch instead:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 2: Configure API keys

Copy `.env.example` to `.env` in the project root and fill in your keys:

```
FOOTBALLDATA_ORG_KEY=your_key_from_football-data.org
BBS_API_KEY=your_key_from_bigballssports.com
PARSE_BOT_KEY=your_key_from_parse.bot    # optional — used for lineups
```

**Getting the keys:**
- **football-data.org** — free tier, sign up at https://www.football-data.org/client/register. Gives you PL finished scores and fixtures.
- **Big Balls Sports** — used for La Liga live data. Contact bigballssports.com for a key.
- **parse.bot** — used for FotMob lineups. This is a paid service with a credit system; the dashboard works without it but the pitch will show dots instead of confirmed player names.

---

## Step 3: Build the historical database (first time only)

The Dixon-Coles model needs 4 seasons of xG data to train on. This step scrapes it from Understat via the `soccerdata` library.

```bash
# From the project root
python v4_backend/historical_scraper.py
```

This creates `v4_historical_data.sqlite` in the project root. It takes 5–15 minutes depending on your connection. **You only need to do this once**, or at the start of a new season when you want to retrain.

---

## Step 4: Train the DC priors (first time only, or new season)

With the SQLite database ready, run the Jupyter notebook to compute the attack (α) and defence (β) ratings for all 125 teams across the Big 5 leagues:

```bash
cd v4_backend/notebooks
jupyter notebook train_v4_priors.ipynb
```

Click **Run All**. The notebook runs the SLSQP MLE optimiser with exponential time-decay weighting and saves the results to `v4_priors.json` in the project root. Takes around 2–5 minutes.

> **Note:** `v4_priors.json` and `football_v4.pth` are already included in the repository. You only need to retrain if you want to update the model with new season data.

---

## Step 5: Run the dashboard

```bash
cd v4_web
python -m uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser. The `--reload` flag means the server automatically restarts when you edit a file — useful during development.

**What happens at startup:**
1. `football_v4.pth` loads into memory (neural net, ~1 second)
2. FPL bootstrap is fetched and cached (PL team names + fixtures)
3. The prediction logging job is scheduled to run every 30 minutes
4. The first prediction job cycle runs immediately

---

## Navigating the dashboard

### Sidebar

The collapsible sidebar (☰) gives access to all pages. Click the league name to go directly to that league's match page.

| Icon | Page | What it shows |
|------|------|---------------|
| 🔲 | Hub | Today's fixtures across both leagues with DC odds |
| ⚽ | Match | Full match prediction for selected game |
| 🗺 | Team map | DC strength visualisation for all teams |
| 📈 | Prediction history | Model accuracy tracker |
| 🏆 | Tournament | CL simulator + PL season Monte Carlo |

---

### Match page

The match page is the core of the dashboard. It shows:

**Fixture strip** — upcoming matches for the active league. Click any fixture to load it.

**Pre-game DC odds** — home/draw/away percentages from the Dixon-Coles bivariate Poisson model. Updated from `v4_priors.json`.

**Adjusted odds** — same calculation but with the announced lineup factored in. If key players are absent, the odds shift accordingly. Only shows when lineup data is available.

**SVG pitch** — the starting XI rendered as jersey shapes on a football pitch. Each jersey shows the team's kit colours and a 3-letter club abbreviation. Click any player to go to their profile page.

**Scoreline matrix** — heatmap of the most likely scorelines. Darker cells = higher probability.

**Match timeline** — appears after kickoff, showing goals and key events as a horizontal timeline.

**Live odds** — the neural network takes over once the match is live, updating odds every 30 seconds using the current score and minute.

---

### History page

Shows all logged predictions with their outcomes. Three tabs: All / Premier League / La Liga.

The accuracy stats shown are:
- **DC Accuracy** — how often the DC model's top-probability outcome was correct
- **Adj Accuracy** — same but only for matches where lineup data was available
- **Draw Recall** — how many actual draws the model predicted as draws (historically the hardest to get right)
- **Accuracy by Confidence Band** — broken down into High (>60%), Mid (40–60%), Low (<40%) confidence predictions

The model logs predictions automatically every 30 minutes. Results are filled in once fd.org confirms the final score (usually within a few hours of the match ending on the free tier).

---

### Tournament page

Two modes, toggled via the tabs:

**Champions League simulator** — pick 16 teams from any of the 5 leagues. Click "Run 10,000 simulations" to run the Monte Carlo. The results show win probability per team and the most likely bracket path.

**PL season simulator** — fetches the current league standings from fd.org and simulates the remaining fixtures 10,000 times. Shows predicted final table with title/top 4/relegation probabilities and P10–P90 confidence ranges.

---

### Player page

Reached by clicking a player name on the pitch SVG. Shows:
- Player cutout photo and bio from TheSportsDB (nationality, age, position)
- Team DC ratings (α attack, β defence bars)
- Next match DC odds with a link to the match page
- Career timeline with club badges, years, appearances and goals

---

## Troubleshooting

**App won't start — `ModuleNotFoundError: No module named 'routes'`**
Make sure `v4_web/routes/__init__.py` exists. If not, create it as an empty file.

**Match page crashes — `NameError: name 'X' is not defined`**
A module-level import is missing. Check the error name against `routes/match.py` imports.

**`[fpl] fetch failed for bootstrap — serving stale cache`**
FPL is temporarily blocking the IP. The dashboard serves stale cache and recovers automatically within a few hours. All functionality continues working.

**`FotMob HTTP 402 on get_matches_by_date: Payment Required`**
parse.bot credits exhausted. Lineups will show placeholder dots instead of confirmed player names. Scores and odds are unaffected — they come from fd.org and BBS respectively.

**Prediction history not updating**
Results come from fd.org's free tier which has a delay of several hours. Results are checked every 30 minutes automatically. If a result is still missing after 24 hours, check whether fd.org has the match under `season=2026` for PL.

**La Liga fixtures not showing**
BBS API may be rate-limiting or temporarily down. The dashboard falls back to FotMob date cache if available, then fd.org. Restart the server to clear the in-memory cache and trigger a fresh fetch.

---

## Running without a GPU

The neural net runs on CPU if no CUDA device is detected. It is slower (inference goes from ~5ms to ~200ms per call) but fully functional. The dashboard detects this automatically — no configuration change needed.

---

## Retraining the model for a new season

At the start of each season (August):

1. Run `v4_backend/historical_scraper.py` to fetch the latest xG data
2. Open `train_v4_priors.ipynb` and run all cells — new `v4_priors.json` generated
3. Run `v4_web/scripts/build_train_v4.py` to retrain the neural net on updated data
4. Replace `v4_priors.json` and `football_v4.pth` in the project root
5. Update `_current_season()` in `footballdata.py` if the fd.org season parameter has changed
6. Restart the dashboard

