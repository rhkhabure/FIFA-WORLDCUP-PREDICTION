# NBA win-probability model: mathematical and operational reference

This document describes the NBA prediction system that is currently wired into
the dashboard. It separates the **pregame Elo forecast**, the **live neural
network**, and the **result-driven Elo update**. They serve different purposes:
the neural network is not used before tip-off, and the Elo update does not
retrain neural-network weights.

The implementation is in `v4_web/nba_model.py`; live state is assembled by
`v4_web/nba_api_client.py`; NBA routes and Elo evaluation are wired in
`v4_web/routes/nba.py`; Elo replay and promotion are implemented in
`v4_web/model_updates.py`.

## 1. Model at a glance

| Match phase | Production estimate | Inputs | Main artifacts |
|---|---|---|---|
| Before tip-off | Logistic Elo probability | Home/away team ratings and a fixed 100-point home-court advantage | `v4_web/nba/model/elo_ratings_v3.json`; optional accepted result-updated ratings in `predictions.db` |
| During a game | 16-feature binary neural classifier | Score and clock, Elo, game state, play-by-play-derived state, and two availability fields | `win_prob_net_v3.pth`, `scaler_v3.pkl`, `calibration_t.pkl`, `model_config.json` |
| After games settle | Chronological Elo replay, promoted only if scoring rules improve | Logged pregame predictions and completed regular-season results | `nba_elo_state` and `model_update_evaluations` tables in `v4_web/predictions.db` |

Pregame probabilities are displayed and logged when the game first appears in
the schedule. Once the game starts, the match page constructs a live snapshot
and uses the neural classifier if that snapshot and the loaded model are
available. If the live model is unavailable, the route falls back to the Elo
pregame probability.

## 2. Pregame Elo probability

The model stores one rating \(R_i\) for each NBA team. The initial ratings are
loaded from `elo_ratings_v3.json`; an unknown team ID falls back to 1500. For a
home team \(H\), an away team \(A\), and fixed home advantage \(H_A=100\):

```text
R_home_adjusted = R_H + 100

P(home win) = 1 / (1 + 10 ^ ((R_A - R_home_adjusted) / 400))
P(away win) = 1 - P(home win)
```

The divisor 400 is the conventional Elo probability scale. The displayed
pregame estimate is rounded to four decimal places. The neural network is not
queried for this forecast: it needs an in-game score and clock, which do not
exist before tip-off.

The Elo JSON is a required runtime artifact. The loader searches
`NBA_ELO_PATH`, `v4_web/nba/model/`, and several historical NBA-project
locations. If it finds a file outside `nba/model`, it attempts a one-time copy
to that model directory. The loader logs an error if the ratings cannot be
found; team lookups then default to 1500, so a valid model load alone does not
prove that meaningful team ratings were loaded.

## 3. Live neural-network inputs

The checkpoint expects these features in exactly this order. A feature-order
or scaling change without retraining would change the meaning of the learned
weights.

| # | Feature | Meaning and runtime construction |
|---:|---|---|
| 1 | `score_diff` | Home score minus away score; clipped to `[-60, 60]` |
| 2 | `time_remaining_sec` | Total seconds remaining until the end of regulation; 2880 at the opening of a game and 0 during every overtime period |
| 3 | `quarter` | Period number: 1–4 for regulation, 5 or greater for overtime |
| 4 | `quarter_time_elapsed_pct` | Fraction of the current period elapsed, calculated from the period clock; regulation periods are 720 seconds and overtime periods 300 seconds |
| 5 | `home_elo` | Home team's active pregame Elo rating |
| 6 | `away_elo` | Away team's active pregame Elo rating |
| 7 | `elo_diff` | `home_elo - away_elo` before adding the 100-point home advantage |
| 8 | `is_playoffs` | 1 for a playoff game, else 0; the current route explicitly supplies 0 |
| 9 | `is_overtime` | 1 when the period is 5 or greater, otherwise 0 |
| 10 | `lead_changes_norm` | Direct changes from one non-tied leader to the other divided by scoring plays so far; a tie resets the remembered leader, and the feature is 0 if no play-by-play is available |
| 11 | `possession` | Proxy based on the team that scored most recently: 1 for home, 0 for away; 0.5 if unavailable. This is not a trusted live possession feed |
| 12 | `home_in_bonus` | 1 after the away team has recorded at least five fouls in the current quarter; otherwise 0 |
| 13 | `away_in_bonus` | 1 after the home team has recorded at least five fouls in the current quarter; otherwise 0 |
| 14 | `home_avail_delta` | Optional DARKO-like home-player availability adjustment; defaults to 0 |
| 15 | `away_avail_delta` | Optional DARKO-like away-player availability adjustment; defaults to 0 |
| 16 | `elo_prior_weight` | State-dependent Elo feature defined below |

### 3.1 Elo-prior feature

The feature builder computes:

```text
sigmoid(x) = 1 / (1 + exp(-x))

elo_prior_weight =
    sigmoid(time_remaining_sec / 2880)
    * sigmoid(1 / (abs(score_diff) + 1))
```

It is a learned input feature; the runtime does not add it as a separate
weighted probability blend after neural inference. It generally reduces as
the clock runs down or the score margin grows, allowing the network to learn
how predictive pregame team strength should be in different game states.

### 3.2 How runtime state is assembled

`build_live_snapshot()` requests the current scoreboard and play-by-play:

- With play-by-play, a shared event walker derives scoring plays, lead changes,
  the last-scoring-team possession proxy, and quarter foul counts. Fouls reset
  at each new quarter. It counts every foul event, including offensive fouls,
  to stay consistent with the training feature implementation; changing this
  rule on only the runtime or training side would create train/serve skew.
- If play-by-play is missing, the code uses scoreboard state and substitutes
  zero lead changes, neutral possession `0.5`, and zero bonus flags.
- Quarter-time elapsed is derived from the clock. In overtime, the period
  length is five minutes.
- The match route currently sets `is_playoffs=0`. Both availability deltas are
  currently left at their default of zero in this route and in the rebuilt
  chart series. They are supported by the model interface but are not presently
  populated from a live player-availability source.

Before neural inference, the 16 values are transformed by the saved
scikit-learn `StandardScaler`. That scaler was fit on 962,871 feature rows
according to its serialized `n_samples_seen_` value. No values are re-fit or
updated during dashboard inference.

## 4. Neural architecture and probability calculation

`WinProbNet` is a fully connected binary classifier:

```text
16 inputs
  -> Linear(16, 128; no bias) -> BatchNorm(128) -> ReLU -> Dropout(0.25)
  -> Linear(128, 64; no bias) -> BatchNorm(64) -> ReLU -> Dropout(0.25)
  -> Linear(64, 32; no bias) -> BatchNorm(32) -> ReLU -> Dropout(0.25)
  -> Linear(32, 1) -> logit
```

Batch normalization supplies the hidden layers' learned shift and scale; the
linear layers before it therefore do not use bias. The checkpoint has 12,769
learned parameters, including batch-normalization parameters. Dropout is active
only during training; inference calls `eval()` and disables it.

For a live feature row \(x\), the runtime first standardizes it with the saved
scaler, evaluates the network to get logit \(z=f(x)\), then applies the saved
temperature \(T\):

```text
x_scaled = StandardScaler.transform(x)
z        = WinProbNet.logits(x_scaled)
p_home   = sigmoid(z / T)
p_home   = clip(p_home, 0.001, 0.999)
p_away   = 1 - p_home
```

The current artifact has `T = 1.003437856...`, effectively close to 1. The
temperature calibration is a post-training probability adjustment: it scales
the logit before the sigmoid, but does not change model weights or feature
values. Probabilities are rounded to four decimal places in the API response.

The returned `confidence` label is a simple display heuristic based only on
distance from 50%:

```text
high    if abs(p_home - 0.5) > 0.25
medium  if abs(p_home - 0.5) > 0.10
low     otherwise
```

This is not a separately trained confidence score or a guarantee of
calibration. The response identifies whether the estimate came from
`neural_net` or the `elo_fallback`.

## 5. Training data, date coverage, and split evidence

### 5.1 What is recorded in this checkout

The deployed artifacts include:

- `model_config.json`: architecture, feature order, target name, selected
  temperature, validation metrics, epoch summary, and three walk-forward
  folds.
- `scaler_v3.pkl`: feature count and scaler statistics. It records
  `n_features_in_ = 16` and `n_samples_seen_ = 962,871`.
- `training_history.json`: per-epoch training loss, validation loss,
  validation AUC, and validation Brier score for 38 epochs.
- `walk_forward_results.csv`: the per-season chronological evaluation results.

The feature names show that training examples represent in-game snapshots
labelled with the final binary outcome `home_team_won`. The saved scaler's
962,871 rows are snapshot/feature rows, not 962,871 independent games.
The live play-by-play walker also accepts the action-type spellings used by
NBA Stats V3, which its source comments identify as the training-data format.
This is evidence about the pipeline's expected event shape, but it does not
identify the exact downloaded dataset/version used to create the deployed
checkpoint.

The serialized history contains 38 training epochs. The configuration
identifies epoch 23 as the best epoch; its validation fields report the
selected snapshot's overall metrics. The history records training and
validation loss, validation AUC, and validation Brier at each epoch. Exact
optimizer, batch size, loss definition and checkpoint-selection rule are not
included in the tracked training source, so they cannot be reproduced from
this repository alone.

### 5.2 Chronological test folds

The strongest reproducible evidence for the temporal date range and test split
is the artifact's expanding-window walk-forward report:

| Fold | Training seasons | Held-out season | Test snapshots | Brier | ROC AUC | Brier skill | Log loss | Fold temperature |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 2018–19 through 2021–22 | 2022–23 | 171,070 | 0.16630 | 0.83240 | 0.31687 | 0.49132 | 1.01520 |
| 2 | 2018–19 through 2022–23 | 2023–24 | 167,957 | 0.14908 | 0.86611 | 0.39825 | 0.44898 | 1.01225 |
| 3 | 2018–19 through 2023–24 | 2024–25 | 166,954 | 0.15540 | 0.85530 | 0.37321 | 0.46260 | 0.95343 |

Thus the recorded evaluation spans NBA seasons **2018–19 to 2024–25**. Each
test fold is a later full season, and each fold's training window expands to
include prior seasons. This is a more meaningful future-season check than
randomly splitting adjacent snapshots, because it tests later seasons against
earlier ones.

The artifact's separate `val_metrics` entry reports ROC AUC 0.85648, PR AUC
0.88082, Brier 0.15507, Brier skill 0.37455, log loss 0.46185, and accuracy
0.76216 (76.2%). These are marked **validation** metrics in the artifact; they
are not a substitute for the three season-held-out results above, and they
should not be presented as a fresh, independent final test score.

ROC AUC and PR AUC summarize ranking/discrimination over thresholds. Accuracy
uses the classifier's 0.5 decision threshold. Brier score and log loss assess
the probability values themselves, with lower values better. The model
configuration records Brier skill score but does not include its reference
forecast definition, so comparisons should use the raw Brier score unless
that reference is separately recovered.

### 5.3 Provenance limits

The model-training notebook/script, raw training manifest, and exact final
train/validation membership are not present in this repository. Therefore the
checked-in evidence does **not** establish:

- the exact raw provider or file version used to create the training
  snapshots;
- the exact date boundaries within each season or whether every game in each
  season was included;
- the final model's precise train/validation split, random seed, or whether
  the reported validation metrics are from the same population as the
  walk-forward folds; or
- the training optimizer, loss definition, batch size, or exact model
  selection rule.

Do not treat the separately provisioned local historical NBA profile and
play-by-play databases as a verified copy of the neural model's training set:
their current documented purpose is historical site features and shot
analysis. A reproducible retraining run should add the training code and a
dataset manifest (source/version, season coverage, row/match counts,
split-by-season protocol and seeds) next to these model artifacts.

## 6. What was adjusted after evaluation

There are two different kinds of adjustment to distinguish.

### 6.1 Neural model post-processing

The saved temperature is applied after the network produces its logit:

```text
p = sigmoid(logit / T)
```

The selected value (`T = 1.003437856...`) is stored in
`calibration_t.pkl` and is also recorded in `model_config.json`. It slightly
rescales probabilities; it does not update the neural network. The stored
walk-forward folds also report fold-specific temperatures, but the live
application loads the single deployed temperature artifact.

The current repository does not contain a before/after model comparison or
training notebook documenting every earlier architecture/training experiment.
It is therefore not possible to claim from this checkout that a particular
loss, optimizer, or training intervention was adopted because of a specific
observed error pattern.

### 6.2 Result-driven Elo adaptation

The application has a separate, online-in-the-operational-sense but
not-gradient-learning Elo replay. It uses settled results to create candidate
team ratings and only activates them when they beat the saved pregame forecast
under proper scoring rules. This is the concrete post-results learning
mechanism currently implemented; it does **not** fine-tune `win_prob_net_v3.pth`.

For each completed eligible game, in chronological order:

```text
p_expected = 1 / (1 + 10 ^ ((R_away - (R_home + 100)) / 400))
delta      = 20 * (home_won - p_expected)
R_home     = R_home + delta
R_away     = R_away - delta
```

The model update code replays game outcomes from its baseline rating file. The
candidate probability for each game is calculated **before** applying that
game's result, so the game cannot improve its own candidate prediction. The
baseline is the probability stored when the pregame forecast was logged (or
baseline Elo probability if no stored forecast exists).

The eligible result set uses NBA regular-season game IDs beginning with `002`
and excludes rows labelled preseason; postseason/play-in and All-Star games
are not intended to contribute. Candidate ratings are accepted only if there
are at least 30 eligible games and:

```text
candidate Brier    < baseline Brier - 0.00001
candidate log-loss <= baseline log-loss + 0.00001
```

For binary results \(y_m \in \{0,1\}\):

```text
Brier    = mean((p_m - y_m)^2)
Log-loss = -mean(y_m * log(p_m) + (1-y_m) * log(1-p_m))
```

Both lower scores are better. The gate requires a material Brier improvement
and protects log-loss from a meaningful regression; accuracy is recorded but
does not determine promotion. The implementation clips each probability into
`[0.000001, 0.999999]` before scoring to keep logarithms finite. If a candidate
fails, an already accepted rating set remains active; otherwise the static
baseline ratings stay active.

Refresh runs during NBA route setup at application startup and again after the
hub logs newly settled NBA results. It does not run as an optimizer on every
live-poll request. The accepted ratings feed both future pregame Elo estimates
and the Elo features used by the live network.

### 6.3 Current local result-driven evaluation

The local `v4_web/predictions.db` recorded an accepted evaluation on
**2026-11-06**:

| Eligible settled games | Baseline Brier → candidate | Baseline log-loss → candidate | Candidate accuracy | Decision |
|---:|---:|---:|---:|---|
| 62 | 0.25558 → 0.25274 | 0.71609 → 0.70896 | 61.3% | Accepted |

The accepted replay used `K=20`, a 100-point home advantage, and the 30-game
minimum. This is a small, early-season operating sample from the local
prediction history, not a long-term or out-of-season guarantee. The result may
change as games are settled or corrected.

## 7. Known limitations and interpretation

- Pregame Elo and live neural probability are different models and are
  evaluated on different information. Do not compare a live forecast with a
  pregame forecast without matching the game state and evaluation sample.
- The live network learns from the supplied score, period, and clock. It does
  not forecast future scoring possessions and does not receive sportsbook
  odds, player identities, box-score statistics, or the pregame Elo
  probability as a separate feature.
- Availability fields exist but currently arrive as zero from the dashboard's
  live route; they should not be described as active injury/player adjustments.
- The current route marks all matches as non-playoff. That feature does not
  currently distinguish postseason context in production.
- Possession is inferred from the last scoring team, which can be wrong after
  turnovers, rebounds, timeouts, or other intervening events. The missing-PBP
  path substitutes a neutral value and zero bonus/lead-change features.
- The confidence category is a distance-from-50% label, not a calibrated
  uncertainty estimate.
- The stored validation and walk-forward scores describe historical snapshots.
  Consecutive snapshots within the same game are correlated, so snapshot count
  must not be interpreted as the same number of independent games.
- The Elo update only has results in the site's prediction database. A game
  that was never logged before tip-off may be absent from the baseline/update
  comparison.
- The local historical profile database and current provider feeds have
  separate coverage and freshness boundaries; those records do not
  automatically refresh the trained model.
