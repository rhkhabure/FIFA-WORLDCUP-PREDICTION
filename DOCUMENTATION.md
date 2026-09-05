Understood. I will not execute any commands, and I will not touch your repository or file system in any capacity. 

Here is the exhaustive, extreme-detail technical documentation covering every mathematical model, script, pipeline, and API test implemented during the V3 and V4 phases. 

You can copy the markdown block below in its entirety and paste it directly into your `DOCUMENTATION.md` file.

***

```markdown
# V3 & V4 System Architecture & Mathematical Engine Documentation

This document provides a highly granular, component-by-component breakdown of the codebase, mathematical models, API ingestion pipelines, and UI reconstructions developed during the V3 and V4 iterations of the Quantitative Match Terminal.

---

## 1. Phase 3 (V3): API Prototyping & Framework Transition

The V3 phase focused on transitioning the localized Streamlit prototype into an asynchronous, production-ready backend capable of handling real-time data ingestion and web routing.

### 1.1 Web Application Transition
* **File:** `v3_web/main.py`
* **Architecture:** Migrated from Streamlit to FastAPI. 
* **Templating:** Implemented `Jinja2Templates` to allow for server-side rendering of dynamic data into pure HTML/TailwindCSS frontends (`base.html`, `match.html`, `team.html`).
* **Routing Setup:** Created modular endpoints (`/`, `/match`, `/team`, `/player`) designed to accept query parameters (e.g., `?match_id=...` or `?team_name=...`) to dynamically construct context dictionaries (`ctx`).

### 1.2 Initial API Prototyping (Deprecated)
During V3, two primary API providers were evaluated via Jupyter Notebooks to replace the static CSV files.
1. **API-Football (RapidAPI):** 
   * **Notebook:** `notebooks/phase4_v3_api_football_ingestion.ipynb`
   * **Purpose:** Targeted for structural match data (Fixtures, Lineups, basic events).
   * **Result:** Successfully mapped JSON hierarchies, but lacked continuous Expected Goals (xG) data necessary for advanced Bayesian modeling.
2. **Sofascore API (RapidAPI):** 
   * **Notebook:** `notebooks/phase4_v3_sofascore_ingestion.ipynb`
   * **Purpose:** Targeted for deep statistical metrics (continuous xG, shot maps, player momentum ratings).
   * **Result:** Deprecated. During live sandbox testing (Match ID: 8897222), the API continuously threw SSL/CAPTCHA blocking errors (Cloudflare 403s), rendering it unstable for a live server backend.

---

## 2. Phase 4 (V4): The Bayesian Tri-State Engine

V4 entirely stripped out discrete goal-based historical data and replaced it with Continuous Expected Goals (xG). The prediction model was split into three distinct, sequentially evaluated mathematical states: Prior, Likelihood, and Posterior.

### 2.1 The Data Scraper & SQLite DB
* **File:** `v4_backend/historical_scraper.py`
* **Mechanism:** Bypassed API blocks by utilizing the `soccerdata` (Understat) library to scrape continuous xG directly.
* **Extraction Logic:** 
  * Extracted data for Target Leagues (`ENG-Premier League`, `ESP-La Liga`, `ITA-Serie A`, `GER-Bundesliga`, `FRA-Ligue 1`) across seasons `2122` through `2425`.
  * Parsed the Pandas MultiIndex dataframes, explicitly mapping columns to avoid duplicate `home_team` vs `home_team_id` conflicts.
  * Extracted `home_xg` and `away_xg`.
  * Processed the `score` string (e.g., "2-1") by running a regex `str.replace(r'\(.*?\)', '')` to strip out penalty shootout data, and splitting it into `home_goals` and `away_goals`.
* **Storage:** Dropped into `v4_historical_data.sqlite` containing over 7,156 matches.

### 2.2 Stage 1: The Prior (Dixon-Coles Continuous xG Optimization)
* **File:** `v4_backend/dixon_coles_xg.py`
* **Objective:** Establish the static, baseline expected strength of teams before lineup variations.
* **Math Implementation (Negative Log-Likelihood):** 
  Instead of utilizing discrete goals, the NLL optimizer evaluates the continuous `home_xg` and `away_xg`. 
  * $\lambda = \alpha_H \cdot \beta_A \cdot \gamma$
  * $\mu = \alpha_A \cdot \beta_H$
  * Decay factor ($\xi$) = `0.0018` (weights recent matches more heavily).
* **Optimizer Constraints:** Uses `scipy.optimize.minimize` (SLSQP).
  * Forces the mean of all $\alpha$ (Attack) values to equal exactly `1.0`. Without this constraint, $\alpha$ and $\beta$ values drift to infinity during optimization.
  * Static $\gamma$ (Home Advantage). Fixed to avoid double-counting team strength.
* **Output:** JSON dictionary (`v4_priors.json`) containing exactly mapped `alpha` and `beta` floats for every team.

### 2.3 Stage 2: The Likelihood (Lineup Adjustment)
* **File:** `v4_backend/likelihood_adjustment.py`
* **Objective:** Adjust the prior $\alpha$ and $\beta$ based on the strength of the announced Starting XI.
* **Math Implementation:** 
  * Compares active Matchday Roster (`s_prime`) against the team's historical Optimal Roster (`ideal_roster`).
  * Calculates attacking delta ($\Delta_{att}$) and defensive delta ($\Delta_{def}$). 
  * Uses `np.clip(delta, 0.80, 1.20)` to strictly cap lineup deviations at ±20%, preventing catastrophic model drift if a star player is injured.
  * Adjusted parameters: $\alpha_{adj} = \alpha_{prior} \cdot \Delta_{att}$, and $\beta_{adj} = \beta_{prior} \cdot \Delta_{def}$.

### 2.4 Stage 3: The Posterior (Live In-Play Engine)
* **File:** `v4_backend/in_play_posterior.py`
* **Objective:** Continuously recalculate match probabilities as time passes and live events (goals, xG) occur.
* **Math Implementation (`generate_live_in_play_odds`):**
  1. **Time Decay:** $T_{rem} = (90 - \text{current\_minute}) / 90.0$.
  2. **Game-State Scalars ($\omega$):** Modifies expected scoring rates based on the current scoreline. 
     * If Home leads: $\omega_H = 0.75$ (sit back), $\omega_A = 1.35$ (push forward).
     * If Away leads: $\omega_H = 1.40$, $\omega_A = 0.70$.
  3. **Live Performance Modifiers ($Perf$):** Calculates $Perf_H = \text{clip}(xG_{live\_h} / xG_{expected\_h}, 0.5, 2.0)$.
  4. **Remaining Expectancy:** Calculates the new remaining lambdas: 
     $\lambda_{rem} = (\alpha_{adj} \cdot \beta_{adj} \cdot \gamma) \cdot T_{rem} \cdot \omega_H \cdot Perf_H$.

### 2.5 Bivariate Poisson Engine & The Infinite Tail Fix
* **File:** `v4_backend/bivariate_poisson.py`
* **Objective:** Transform the final $\lambda$ and $\mu$ values into a probability matrix for the 1X2 market.
* **Infinite Tail Implementation:** 
  A standard Poisson PMF capped at a 9x9 matrix will sum to ~`0.999`, leaking probability mass. To create an airtight matrix summing exactly to `1.0`:
  * `prob_home[-1] = poisson.sf(max_goals - 2, lambda_h)`
  * This forces the final index (index 8) to capture all probabilities of scoring "8 or more goals to infinity".
* **Rho ($\rho$) Adjustment:** Adjusts for the inter-dependency of low-scoring games (0-0, 1-0, 0-1, 1-1) to correct the Poisson distribution's assumption of strict independence.
* **Market Mapping:** 
  * Home Win = `np.sum(np.tril(score_matrix, -1))`
  * Draw = `np.sum(np.diag(score_matrix))`
  * Away Win = `np.sum(np.triu(score_matrix, 1))`

---

## 3. Footballdata.io Ingestion Pipeline

To replace Sofascore, V4 implements Footballdata.io as the sole live data provider.

### 3.1 API Prototyping & Debugging
* **File:** `notebooks/phase4_v4_footballdata_ingestion.ipynb`
* **Base URL Fix:** Corrected from `api.footballdata.io` to `https://footballdata.io/api/v1` to resolve `getaddrinfo` socket errors.
* **KeyError 0 Fix:** Discovered that Footballdata.io returns a heavily nested JSON dictionary for `/fixtures/today` rather than a standard array. Implemented strict `isinstance(data, dict)` introspection to unpack nested match IDs.
* **Payload Mappings Discovered:**
  * Base Match Metadata: `data -> match`
  * Live xG: `data -> stats -> xg -> home/away` (Note: xG is stored in a dictionary, not a list, requiring specific `isinstance` checks).
  * Events Timeline: `data -> events`

### 3.2 Live Backend Integration
* **File:** `v4_web/footballdata.py`
* **Auth Implementation:** Bypassed `python-dotenv` and `requests` module dependency errors on the server by writing a manual `.env` file parser and utilizing standard library `urllib.request`. Added `timeout=5` and TLS/SSL error handling.
* **The "VAR Disallowed Goal" Bug Fix:** 
  * Initially, the parser tallied goals by iterating through the `events` array and counting `event_type == "goal"`.
  * *Bug:* This erroneously counted disallowed/VAR-cancelled goals (resulting in a 3-2 scoreline for a 2-2 match).
  * *Solution:* Rewrote the parser to ignore goal events entirely, instead extracting the authoritative `home_score` and `away_score` integers strictly from the `match` metadata header, relegating the `events` array to tallying red cards and text timeline output.
* **Fallback Logic (`get_last_completed_pl_match`):** Implemented an automatic query to `/leagues/15/matches` (filtering by `"status": "complete"`) to automatically fetch the most recent Premier League match ID if the user navigates to the UI without supplying a query parameter.

---

## 4. UI Reconstruction & Web Architecture

The frontend was completely rebuilt to enforce strict compliance with real-time mathematical outputs, deliberately wiping out all fake data, mocked placeholders, and non-functional UI elements.

### 4.1 Base Layout
* **File:** `v4_web/templates/base.html`
* **Status:** Erased all vestigial "FBD_TERMINAL" sidebars, filter dropdowns, and diagnostic tickers. 
* **Design:** Reduced to a pure TailwindCSS layout utilizing `#0b0f19` (slate graphite background) and `JetBrains Mono` for data elements.

### 4.2 Endpoint Routing
* **File:** `v4_web/main.py`
* **Hub Routing (`/`):** Currently wiped and returning a blank WIP status to prevent mock data rendering.
* **Match Routing (`/match`):**
  * Automatically intercepts empty requests and triggers `get_last_completed_pl_match()`, utilizing `fastapi.responses.RedirectResponse` to physically append the valid `match_id` to the URL.
  * Connects directly to `v4_priors.json` to extract exact $\alpha$ and $\beta$ constraints.
  * Feeds live score, minute, and xG directly into the `generate_live_in_play_odds()` engine.

### 4.3 The Match Dashboard
* **File:** `v4_web/templates/match.html`
* **Status:** Cleaned of all scatter plots, matrices, and pitches to prevent hardcoding. Only contains the **Three-Segment Probability Widget**.
* **Widget Logic:** 
  * **Stage 1 (Prior):** Uses Jinja2 conditional `{% if prior %}`. Binds Tailwind progress bar widths directly to the mathematical array outputs of the NLL engine. Falls back to a red `NO PRIOR DATA` badge if the teams are missing from the DB.
  * **Stage 2 (Likelihood):** Hardcoded to output an amber `NO LINEUP DATA - AWAITING XI` badge, as API starting XI ingestion is not yet complete.
  * **Stage 3 (Posterior):** Uses Jinja2 conditional `{% if posterior %}`. Binds directly to the live outputs of the Bivariate Poisson matrix adjusting for time decay. Outputs a red `NO LIVE API DATA` badge if the API drops or the match hasn't started.
```
