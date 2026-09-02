# 🌍 V3 Universal Football Model

> **V3 Update:** This repository has been successfully upgraded from the *V2 World Cup 2026 Live Win Probability* model into the **V3 Universal Football Model**. It now supports continuous domestic league play across Europe's "Big 5" leagues, featuring a completely overhauled PyTorch architecture, robust Sofascore API integration, and a blazing-fast FastAPI + Tailwind web dashboard.

*(For the original World Cup 2026 Streamlit implementation, see `README_worldcup.md`)*

---

## 🚀 Key V3 Features

### 🧠 The Machine Learning Engine
*   **League Embeddings:** The neural network takes `league_id` through an `nn.Embedding` layer, allowing it to natively learn the stylistic differences between leagues (e.g., a 1-0 lead in Serie A is statistically safer than a 1-0 lead in the Bundesliga).
*   **Focal Loss:** Replaced static class weights with a dynamic Focal Loss module to specifically penalize the model for missing difficult outcomes like Draws and massive upsets.
*   **Multi-Class Isotonic Calibration:** Neural networks are notoriously overconfident. V3 runs raw softmax probabilities through three independent Isotonic Regressors to guarantee statistically honest output percentages.
*   **Dynamic Starting XI Value:** Instead of static FIFA ranks, the model evaluates live team strength based on the exact players stepping onto the pitch.

### 💻 The Web App (FastAPI + Tailwind)
*   **Zero-Slop UI:** Completely migrated away from Streamlit into a custom, deeply controllable FastAPI application using Jinja2 templates and Tailwind CSS.
*   **Dynamic Theming:** The entire app instantly recolors its accents, borders, and charts to match the primary/secondary hex colors of the currently selected team.
*   **Interactive SVG Pitch:** Real-time generation of horizontal and vertical football pitches. It plots formations (e.g., 4-3-3), renders jerseys, and displays real player names without using external image libraries.
*   **Player Profiles:** Click on any player on the SVG pitch to view their profile, complete with a **Chart.js** radar chart mapping their Attacking, Technical, Tactical, Defending, and Creativity ratings.
*   **Monte Carlo Tournament Simulator:** Simulates 10,000 knockout paths and features a step-by-step animated UI to visualize cup tournament predictions.

### 📊 The Data Pipeline (Sofascore API)
*   Integrates natively with the **Sofascore API** (via RapidAPI) to extract pristine live play-by-play events, starting XIs, precise player ages (via timestamp calculation), and granular match histories.
*   Includes a `sync_sofascore.py` script that safely pulls data into a local `data.json` cache, preventing API rate-limiting while allowing rapid UI development.

---

## 🏗️ Project Structure

```text
FIFA-WORLDCUP-PREDICTION/
├── v3_web/                             # The FastAPI Web Application
│   ├── main.py                         # Application routing and context engine
│   ├── utils.py                        # Dynamic SVG generation and color theming
│   ├── sync_sofascore.py               # Syncs live API data to local JSON cache
│   ├── data.json                       # The local SQLite-alternative data cache
│   └── templates/                      # Jinja2 HTML/Tailwind templates
│       ├── base.html                   # Master layout and dynamic CSS variables
│       ├── index.html                  # The League Hub 
│       ├── match.html                  # Live Match center & Win Probability
│       ├── team.html                   # Team Profiles & Match History
│       ├── player.html                 # Player Radar Charts
│       └── bracket.html                # Monte Carlo Tournament Simulator
├── notebooks/                          # V3 Machine Learning Pipelines
│   ├── phase1_v3_data_pipeline.ipynb      # Snapshot extraction and Leakage validation
│   ├── phase2_v3_model_architecture.ipynb # Neural Net & Focal Loss definitions
│   ├── phase3_v3_training_calibration.ipynb # PyTorch training loop & Isotonic scaling
│   ├── phase4_v3_sofascore_ingestion.ipynb  # JSON flattening logic for Sofascore
│   ├── phase5_real_data_sync.ipynb        # Jupyter-based interactive API syncer
│   └── pl_opening_weekend_eval.ipynb      # Dynamic real-world evaluation script
├── common.py                           # Legacy V2 Shared inference utilities
├── generate_mock_db.py                 # Fills data.json with offline mock data
└── .env                                # Environment variables (API Keys)
```

---

## ⚙️ Quick Start (Running Locally)

### 1. Install Dependencies
Ensure you are using Python 3.11+. Create a virtual environment and install the requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn jinja2 requests pandas numpy torch scikit-learn python-dotenv
```

### 2. Configure the API Key
Get a free API key from [Sofascore via RapidAPI](https://rapidapi.com/apidojo/api/sofascore).
Create a `.env` file in the root directory:
```env
SOFASCORE_API_KEY=your_rapidapi_key_here
```

### 3. Sync Real Data (Optional but Recommended)
You can sync a specific team (e.g., Arsenal) to populate `data.json` with real live players, match histories, and stats.
```bash
cd v3_web
python sync_sofascore.py "Arsenal"
```
*(If you don't sync, the app will gracefully fall back to default mock data and empty pitch layouts).*

### 4. Run the Web Dashboard
Launch the FastAPI server:
```bash
cd v3_web
python -m uvicorn main:app --reload
```
Open your browser and navigate to **`http://localhost:8000`**.

---

## 📈 Supported Leagues (Soft Launch)
V3 focuses on the highest fidelity data environments:
1. **English Premier League** (England)
2. **La Liga** (Spain)
3. **Serie A** (Italy)
4. **Bundesliga** (Germany)
5. **Ligue 1** (France)

---
*Created as an evolution of the World Cup 2026 Win Probability model.*