# ⚽ V4 Quantitative Match Terminal (Bayesian Tri-State Engine)

> **V4 Update:** This repository has evolved from a World Cup predictor (V2) and a basic domestic league tracker (V3) into a professional-grade **Bayesian Tri-State Quantitative Engine (V4)**. It abandons "black box" neural networks in favor of a structurally transparent, Continuous xG Dixon-Coles Maximum Likelihood Estimator combined with live Markov Chain/Bivariate Poisson in-game projections.

*(For a deep dive into the math and API pipeline, see `DOCUMENTATION.md`)*

---

## 🚀 Key V4 Features

### 🧠 The Mathematical Engine
*   **Module 1: The Prior (Continuous xG Dixon-Coles).** A custom Maximum Likelihood Estimator (MLE) built in SciPy. It bypasses discrete, noisy goal metrics and trains directly on Continuous Expected Goals (xG). It features an exponential time-decay factor (weighting recent games) and a strict sum-to-1 normalization constraint across attack parameters to guarantee global optimums.
*   **Module 2: The Likelihood (Lineup Adjustment).** Calculates Roster Deltas ($\Delta$) by dynamically comparing the announced Matchday XI against the historical Ideal XI. It uses hyperparameters ($\lambda_a, \lambda_d$) to safely dampen and apply these capability drops to the base Dixon-Coles parameters *before* kickoff. *(Pending API integration for live XI extraction).*
*   **Module 3: The Posterior (In-Game Bivariate Poisson).** Overcomes the notorious "Ghost xG" bias. Instead of naively inflating odds for teams racking up xG while trailing against a low block, the live engine uses Game-State adjustments ($\omega$) and Remainder Time fractions ($\Delta t$) to project true fair odds dynamically through minute 90.

### 🔌 Live API Ingestion (Footballdata.io)
*   **Zero-Placeholder Policy:** The system strictly rejects hardcoded data or mock UI fallbacks. If the API cannot supply live data, the backend halts execution and the UI safely explicitly renders `NO DATA`.
*   **Nested JSON Introspection:** A custom data parser (`v4_web/footballdata.py`) unpacks nested Footballdata.io JSON payloads to extract true `match_id` metadata.
*   **Authoritative Score Parsing:** Immune to "VAR phantom goals." The ingestion pipeline extracts definitive `home_score` and `away_score` attributes directly from match metadata rather than tallying raw event arrays, while correctly extracting the `live_xg` dict payloads for real-time model updates.

### 💻 The Web Dashboard (FastAPI + Tailwind)
*   **Strict Blank Canvas:** Migrated from Streamlit to a blazing-fast FastAPI server utilizing Jinja2 Templates. The UI was intentionally wiped clean to enforce the "Zero-Placeholder Policy", rebuilding only strictly functional, mathematically backed widgets.
*   **Dynamic Fallback Routing:** Automatic `/match` endpoint redirection. If no `match_id` is supplied, the backend securely hits the `/leagues/15/matches` endpoint to automatically load and evaluate the most recently completed Premier League match to guarantee the math engine operates on genuine data.
*   **Three-Segment Probability Widget:** The live anchor of the terminal, visually projecting the Prior $\rightarrow$ Likelihood $\rightarrow$ Posterior evolution directly from the Bayesian matrices. 

---

## 🏗️ Project Structure

```text
FIFA-WORLDCUP-PREDICTION/
├── v4_backend/                         # The V4 Bayesian Mathematical Engine
│   ├── historical_scraper.py           # Understat xG scraper via soccerdata -> SQLite
│   ├── dixon_coles_xg.py               # Module 1: Continuous xG NLL Optimizer
│   ├── likelihood_adjustment.py        # Module 2: Bayesian Lineup Adjustments
│   ├── bivariate_poisson.py            # Module 3: 1X2 Probabilities & Infinite Tail capture
│   ├── in_play_posterior.py            # Module 3: Live Game-State Offset & Remainder Matrix
│   └── v4_priors.json                  # The trained NLL static baseline alphas/betas
├── v4_web/                             # The FastAPI Web Application (Dashboard)
│   ├── main.py                         # Application routing and math engine triggers
│   ├── footballdata.py                 # Live REST ingestion parser and API handler
│   ├── utils.py                        # Legacy SVG generators (Pitch/Radar)
│   └── templates/                      # Jinja2 HTML/Tailwind templates
│       ├── base.html                   # Strict dark-mode master layout canvas
│       ├── index.html                  # League Hub (WIP)
│       └── match.html                  # Live Dashboard (3-Stage Probability Widget)
├── notebooks/
│   └── phase4_v4_footballdata_ingestion.ipynb # Core Footballdata.io testing & parsing env
└── DOCUMENTATION.md                    # Deep dive into the V4 Mathematics & API Spec
