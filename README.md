# ⚽ V4 Universal Football Model (Bayesian Tri-State Engine)

> **V4 Update:** This repository has evolved from a World Cup predictor (V2) and a basic domestic league tracker (V3) into a professional-grade **Bayesian Tri-State Quantitative Engine (V4)**. It abandons "black box" neural networks in favor of a structurally transparent, Continuous xG Dixon-Coles Maximum Likelihood Estimator combined with live Markov Chain/Bivariate Poisson in-game projections.

*(For a deep dive into the math, see `DOCUMENTATION.md` and `USER_MANUAL.md`)*

---

## 🚀 Key V4 Features

### 🧠 The Mathematical Engine
*   **Module 1: The Prior (Continuous xG Dixon-Coles).** A custom Maximum Likelihood Estimator (MLE) built in SciPy. It bypasses discrete, noisy goal metrics and trains directly on Continuous Expected Goals (xG). It features an exponential time-decay factor (weighting recent games) and a strict sum-to-1 normalization constraint across attack parameters to guarantee global optimums.
*   **Module 2: The Likelihood (Lineup Adjustment).** Calculates Roster Deltas ($\Delta$) by dynamically comparing the announced Matchday XI against the historical Ideal XI. It uses hyperparameters ($\lambda_a, \lambda_d$) to safely dampen and apply these capability drops to the base Dixon-Coles parameters *before* kickoff.
*   **Module 3: The Posterior (In-Game Bivariate Poisson).** Overcomes the notorious "Ghost xG" bias. Instead of naively inflating odds for teams racking up xG while trailing against a low block, the live engine uses Game-State adjustments ($\omega$) and Remainder Time fractions ($\Delta t$) to project true fair odds dynamically through minute 90.

### 💻 The Web Dashboard (FastAPI + Tailwind)
*   **Zero-Slop UI:** We abandoned Streamlit's constraints for a blazing-fast, custom FastAPI web application with Jinja2 templates and TailwindCSS.
*   **Universal Theming:** A sleek, unified dark-mode tracking all "Big 5" European Leagues.
*   **Interactive SVG Pitches:** Real-time generation of horizontal and vertical football pitches. Plots exact formations (e.g., 4-3-3), renders jerseys, and maps clickable player links without relying on external image libraries.
*   **Player Radar Profiles:** Click on any player on the pitch to see their age, position, and a dynamic **Chart.js** pentagon radar chart tracking their stats.
*   **Monte Carlo Tournament Simulator:** An animated bracket simulator simulating 10,000 paths to visualize cup tournament endpoints.

### 📊 The Data Pipeline (`soccerdata`)
*   Fully integrated with the open-source `soccerdata` package.
*   Bypasses rate-limited APIs to pull pristine, match-by-match continuous xG historical records straight from Understat into a local SQLite ledger (`v4_historical_data.sqlite`).

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
│   └── notebooks/                      # Execution Notebooks for Model Training & Eval
│       ├── train_v4_priors.ipynb       # Executes the SLSQP Optimizer across Big 5 Leagues
│       └── v4_opening_weekend_eval.ipynb # Out-of-sample prediction evaluation
├── v3_web/                             # The FastAPI Web Application (Dashboard)
│   ├── main.py                         # Application routing and context engine
│   ├── utils.py                        # Dynamic SVG generation and team metadata
│   ├── data.json                       # The local JSON DB for player stats / match histories
│   └── templates/                      # Jinja2 HTML/Tailwind templates
│       ├── base.html                   # Master layout and CSS overrides
│       ├── index.html                  # League Hub
│       ├── match.html                  # Live Match Center
│       ├── team.html                   # Team Profiles
│       ├── player.html                 # Player Profiles & Radar Charts
│       └── bracket.html                # Monte Carlo Tournament Simulator
├── USER_MANUAL.md                      # Step-by-step guide to running the model
└── DOCUMENTATION.md                    # Deep dive into the V4 Mathematics
```

---

## ⚙️ Quick Start

See **`USER_MANUAL.md`** for the complete execution instructions, from scraping the historical xG database to launching the live FastAPI web dashboard!

---
*Developed as an evolution of the World Cup 2026 Win Probability model.*