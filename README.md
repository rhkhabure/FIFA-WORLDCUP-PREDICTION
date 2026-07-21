# ⚽ World Cup 2026 Live Win Probability

A real-time win probability system for the 2026 FIFA World Cup — built using transfer
learning from an NBA win probability neural network. Tracks live match state every 30
seconds and outputs three probabilities: home win, draw, and away win.

> **Live dashboard →** *([Streamlit Cloud link — add after deployment](https://fifa-worldcup-prediction-kc3azpj8tc2ynw4ylnjqv8.streamlit.app/))*  
> **Related project →** [NBA Live Win Probability](https://github.com/rhkhabure/NBA-Live-Win-Probability)

---

## How it works

The model was originally trained on 962,000 NBA play-by-play snapshots to predict
binary (win/lose) outcomes. For football, we:

1. Replace the output layer: `1 node sigmoid → 3 node softmax` (home win / draw / away win)
2. Add a `is_knockout` flag that suppresses the draw output in elimination rounds
3. Retrain using transfer learning on ~550,000 international football snapshots
4. Feed live World Cup 2026 match data through the same inference pipeline

The in-game probability updates every 30 seconds using the free
[worldcup26.ir](https://worldcup26.ir) API — no authentication required.

---

## Model architecture

```
Input (14 features)
  ↓
128 neurons  [BatchNorm → ReLU → Dropout 0.30]   ← transferred from NBA model
  ↓
 64 neurons  [BatchNorm → ReLU → Dropout 0.30]   ← transferred from NBA model
  ↓
 32 neurons  [BatchNorm → ReLU → Dropout 0.30]   ← transferred from NBA model
  ↓
  3 outputs  [Softmax]                            ← retrained for football
    P(home win) · P(draw) · P(away win)
```

### Features

| Feature | Description |
|---|---|
| `goal_diff` | Home score − Away score, clipped to ±5 |
| `time_remaining_sec` | Seconds until 90 min (0–5400) |
| `half` | 1 (first half) or 2 (second half) |
| `match_time_pct` | Minutes elapsed / 90 |
| `is_extra_time` | 1 if in extra time, else 0 |
| `is_knockout` | 1 for Round of 16 onwards (draw suppressed) |
| `lead_changes_norm` | Lead changes / total goal events |
| `home_rank_norm` | Home team FIFA rank normalised [0, 1] |
| `away_rank_norm` | Away team FIFA rank normalised [0, 1] |
| `rank_diff` | home_rank_norm − away_rank_norm |
| `home_group_pts` | Home team's group-stage points before this match |
| `away_group_pts` | Away team's group-stage points before this match |
| `is_neutral_venue` | 1 for World Cup (all neutral ground) |
| `score_state` | 0=behind, 1=level, 2=ahead |

### Training data sources

| Source | Purpose | Matches |
|---|---|---|
| [football-data.org](https://www.football-data.org) | World Cup 1966–2022 | ~400 |
| [football-data.org](https://www.football-data.org) | Top-5 leagues 2015–2024 | ~19,000 |
| [Transfermarkt](https://www.transfermarkt.com) | Squad market values (one-time scrape) | lookup table |

---

## Project structure

```
world_cup_win_prob/
├── model/
│   ├── win_prob_net.pth        ← trained PyTorch weights
│   ├── scaler.pkl              ← StandardScaler fitted on training data
│   ├── temperature.json        ← calibrated temperature T
│   └── squad_values.json       ← Transfermarkt squad values by team
├── processed/
│   └── features_raw.parquet    ← 14-feature training dataset
├── raw/                        ← cached API responses (auto-generated)
├── plots/                      ← post-match analysis charts
├── game_history/               ← auto-saved completed match timelines
│   └── <match_id>.json
├── notebooks/
│   ├── phase1_data_pipeline.ipynb      ← pull data, build features
│   ├── phase2_transfer_learning.ipynb  ← retrain from NBA weights
│   └── phase3_calibration.ipynb        ← temperature scaling, validation
├── app.py                      ← Streamlit live dashboard
├── requirements.txt
├── runtime.txt                 ← python-3.12
└── .streamlit/config.toml
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/worldcup-win-probability.git
cd worldcup-win-probability
pip install -r requirements.txt
```

### 2. Get a free API key

Register at [football-data.org](https://www.football-data.org/client/register) — free,
no credit card. Copy your token into a `.env` file:

```
FOOTBALL_DATA_API_KEY=your_token_here
```

### 3. Run the data pipeline

```bash
jupyter notebook notebooks/phase1_data_pipeline.ipynb
```

This pulls ~19,000 matches and builds the training dataset. Cached after first run —
subsequent runs are instant.

### 4. Train the model

```bash
jupyter notebook notebooks/phase2_transfer_learning.ipynb
```

Requires CUDA (RTX 4060 or better). Training takes ~5 minutes with transfer learning
from the NBA model weights.

### 5. Run the dashboard

```bash
streamlit run app.py
```

---

## Live data API

The dashboard uses [worldcup26.ir](https://worldcup26.ir) for live World Cup 2026 data.
No authentication required.

| Endpoint | Data |
|---|---|
| `GET /get/games` | All matches with scores and status |
| `GET /get/groups` | Group standings |
| `GET /get/teams` | All 48 teams |
| `GET /get/stadiums` | All 16 stadiums |

Live score updates activate from **June 11, 2026** (first World Cup match).

---

## Key differences from the NBA model

| Aspect | NBA model | Football model |
|---|---|---|
| Outcomes | Binary (win/loss) | 3-class (home/draw/away) |
| Output | Sigmoid (1 node) | Softmax (3 nodes) |
| Loss function | BCEWithLogitsLoss | CrossEntropyLoss |
| Playoff/knockout | `is_playoffs` flag | `is_knockout` flag (suppresses draw) |
| Team quality signal | NET rating | FIFA ranking (normalised) |
| Scoring frequency | ~200 events/game | ~3 events/game |
| Snapshot method | Every play | Every goal + halftime |
| Training samples | 962,000 | ~550,000 |

---

## Dashboard features

- **Live 3-way gauge** — home win / draw / away win probabilities update every 30s
- **Win probability chart** — full match timeline with goal markers
- **Group standings panel** — auto-updates from worldcup26.ir
- **Qualification simulator** — Monte Carlo for group advancement probability
- **Knockout bracket** — probability of reaching each round
- **Game selector** — choose which live match to track when multiple games are simultaneous
- **Game history** — replay any completed match's win probability curve

---

## Phases

| Phase | Status | Description |
|---|---|---|
| Phase 1 | 🔄 In progress | Data pipeline — historical match collection |
| Phase 2 | ⏳ Planned | Transfer learning — retrain NBA model for football |
| Phase 3 | ⏳ Planned | Calibration — 3-class temperature scaling |
| Phase 4 | ⏳ Planned | Live dashboard — worldcup26.ir integration |
| Phase 5 | ⏳ Planned | Squad enrichment — Transfermarkt market values |

---

## Requirements

```
torch
numpy
pandas
scikit-learn
scipy
streamlit
plotly
requests
python-dotenv
pyarrow
```

---

## Acknowledgements

- [football-data.org](https://www.football-data.org) — free historical match data
- [worldcup26.ir](https://worldcup26.ir) — free live World Cup 2026 API  
- [rezarahiminia/worldcup2026](https://github.com/rezarahiminia/worldcup2026) — open-source WC2026 API
- [Transfermarkt](https://www.transfermarkt.com) — squad market values
