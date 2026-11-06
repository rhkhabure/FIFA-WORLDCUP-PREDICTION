# Football prediction model: mathematical and operational reference

This is the detailed reference for the football prediction pipeline in Melios
Dira Odds. It distinguishes the model trained offline from the result-driven
rating adjustment evaluated from the site's own prediction history. They are
separate mechanisms: the history-based adjustment does **not** retrain the
neural network.

## 1. Current model at a glance

| Match phase | Production calculation | Main inputs | Artifact / implementation |
|---|---|---|---|
| Before kickoff | Dixon–Coles (DC) scoreline model, with the accepted season's result-driven team factors when available | Team attack/defence priors, league home advantage and low-score correction; settled match results may contribute bounded rating factors | `v4_backend/v4_priors.json`, `v4_web/routes/match.py::dc_pregame`, `v4_web/model_updates.py` |
| After the match has started | 11→40→20→3 PyTorch classifier | Current score and minute, plus normalized DC attack/defence-strength features | `v4_backend/notebooks/v4_backend/models/football_v4.pth` and `scaler_v4.pkl`; `v4_web/routes/match.py::nn_live` |
| Optional confirmed-lineup panel | Hand-weighted expected-goals adjustment, followed by scoreline probabilities | DC expected goals and the presence/absence of a small, manually curated set of key players | `v4_web/lineup_adjustment.py` |

The application loads the DC priors and neural-network artifacts at startup.
The history-based factors are restored/evaluated from `v4_web/data/predictions.db`
at startup as well. The prediction job logs fixtures and settles scores in that
database; it runs periodically, but football rating evaluation is currently
triggered at app startup, not after every prediction-job cycle.

The live network's stored logits are converted to probabilities with
temperature scaling. The DC model remains the pre-match baseline; the live
network's output is the match-state estimate once a live score and minute are
available.

## 2. Dixon–Coles pre-match model

### 2.1 Team ratings and expected goals

For each supported league, the prior file contains an attack rating
\(\alpha_i\), a defensive-concession rating \(\beta_i\), a home-advantage
multiplier \(\gamma\), and a low-score dependence parameter \(\rho\). For home
team \(H\) and away team \(A\):

```text
lambda = alpha_H * beta_A * gamma       # expected home goals
mu     = alpha_A * beta_H               # expected away goals
```

Higher \(\alpha\) means more attacking output. A lower \(\beta\) means stronger
defence: it reduces the opponent's expected goals. The beta parameter is
therefore a *concession multiplier*, not a “higher is better” defensive score.

At the start of a season, the application temporarily shrinks home advantage
toward 1.0, then restores the league-calibrated value by approximately gameweek
6:

```text
gw       = max(1, floor(days_since_season_start / 7) + 1)
decay    = 0.60 + 0.40 * min(gw / 6, 1)
gammaEff = 1 + (gammaCalibrated - 1) * decay
```

This is a simple calendar approximation, not a fixture-count calculation.
The season start defaults to August 21 in `get_effective_gamma()`.

### 2.2 Fitting the static priors

The offline DC fitting routine is `v4_backend/dixon_coles_xg.py`. It estimates
team attack and defence ratings by minimizing a time-weighted negative
log-likelihood with SciPy's SLSQP constrained optimizer.

For each historical match \(m\), let \(x_{Hm}\) and \(x_{Am}\) be the observed
home and away xG, and let \(g_{Hm}, g_{Am}\) be the actual integer goals. The
continuous xG contribution to log-likelihood is:

```text
ell_xG(m) = x_Hm * log(lambda_m) - lambda_m
          + x_Am * log(mu_m)     - mu_m
```

The terms \(-\log\Gamma(x+1)\), which do not depend on the fitted parameters,
are omitted. The observed integer score also supplies a Dixon–Coles correction
\(\tau_m\) for the four low-score outcomes:

```text
tau(0,0) = 1 - lambda * mu * rho
tau(1,0) = 1 + mu * rho
tau(0,1) = 1 + lambda * rho
tau(1,1) = 1 - rho
tau(other score) = 1
```

The weighted objective minimized by the implementation is:

```text
NLL = -sum_m [ exp(-xi * daysAgo_m)
               * (ell_xG(m) + log(max(tau_m, 1e-5))) ]
```

The default decay rate is `xi = 0.0018` per day. Newer matches consequently
have more influence than older matches. The home multiplier used during fitting
is fixed per league to the ratio of total home xG to total away xG; it is not
jointly optimized with every team rating:

```text
gamma = sum(home xG) / max(sum(away xG), 1)
```

To keep the parameters identifiable and bounded, the optimizer applies:

- Mean attack rating equal to 1: `mean(alpha) = 1`.
- Each attack and defence rating constrained to `[0.1, 3.5]`.
- `rho` constrained to `[-0.25, 0.25]`.
- Expected goal rates clipped to `[1e-5, 15]` when evaluating the objective.

The checked-in prior file in this checkout contains the following league
parameters (rounded to three decimals):

| League | Teams in prior | Fitted \(\gamma\) | Fitted \(\rho\) |
|---|---:|---:|---:|
| Premier League | 26 | 1.236 | -0.003 |
| La Liga | 25 | 1.352 | 0.023 |
| Ligue 1 | 24 | 1.223 | -0.061 |
| Bundesliga | 24 | 1.278 | -0.117 |
| Serie A | 26 | 1.181 | -0.061 |

These are fitted prior-file values, not the current early-season effective
\(\gamma\). The fitted \(\rho\) values are distinct from the separate
`DRAW_PROPENSITY = 0.10` outcome adjustment described below.

The resulting JSON is a set of fitted priors, not a neural-network checkpoint.
Rebuilding it is an offline training operation; it is not done on web requests.

### 2.3 From goal rates to 1X2 probabilities

The runtime builds a 9×9 score matrix for scores 0–8:

```text
P0(h,a) = Poisson(h; lambda) * Poisson(a; mu)
```

It applies the four \(\rho\) multipliers to score cells 0–0, 1–0, 0–1, and
1–1. Each correction multiplier is floored at `1e-5` to keep the adjusted cell
non-negative, then the matrix is normalized to sum to 1. The current runtime
uses PMF values for goals 0 through 8 and normalizes that finite matrix; it
does not explicitly collapse the 9+ goal tail into the last row or column.

The raw outcomes are the matrix sums:

```text
P_H = sum(P(h,a) where h > a)
P_D = sum(P(h,a) where h = a)
P_A = sum(P(h,a) where h < a)
```

The application then applies the shared draw-propensity adjustment
`DRAW_PROPENSITY = 0.10`. This is a second adjustment, distinct from \(\rho\):

```text
P_H' = max(P_H - 0.05, 0)
P_D' = P_D + 0.10
P_A' = max(P_A - 0.05, 0)
P_i_final = P_i' / (P_H' + P_D' + P_A')
```

Displayed percentages are these final values rounded to one decimal place.
Fair decimal odds, where displayed, are the reciprocal of the unrounded
probability, with a small lower bound to avoid division by zero.

### 2.4 Missing teams, league names and unsupported data

Provider team names are mapped through `TEAM_NAME_ALIASES` before lookup. For a
team missing from an otherwise available league prior, the pre-match route uses
the league's 25th percentile attack rating and 75th percentile beta rating.
That is a conservative fallback, not a newly learned rating. If the league's
prior itself is unavailable, the route returns no DC estimate rather than
inventing one.

The current principal prior set covers the five major European leagues. Which
leagues have live fixtures or result-history coverage is separately determined
by their data providers and local prediction records.

## 3. Result-driven, recent-season team ratings

### 3.1 What changed and what this does

The site now uses its own settled prediction history to test whether team-level
attack/concession adjustments improve DC probabilities. This adds a bounded
rating layer to the pre-match \(\lambda\) and \(\mu\) rates when (and only when)
the candidate passes a per-league out-of-sample-style walk-forward comparison.

This is **not** online neural-network training and is not an automatic rewrite
of `v4_priors.json`. It estimates multiplicative team factors from the latest
season of fully settled predictions. If a candidate does not pass validation,
the existing static priors remain in use; a previously accepted factor set is
retained if a later evaluation fails.

The implementation is in `v4_web/model_updates.py`:

- `refresh_football_ratings()` selects settled rows with stored DC rates and
  probabilities, then limits evaluation to the most recent season represented
  for each league.
- Rows are ordered chronologically by kickoff and evaluated one at a time.
- For a match, candidate probabilities use factors estimated only from earlier
  rows. Only after scoring that prediction are the match's actual goals added
  to the residual history. This prevents a match's own result from informing
  its candidate prediction.
- Team/league factor state and evaluation metrics are persisted in
  `football_rating_state` and `model_update_evaluations` tables in the
  predictions SQLite database.
- Hashes of the priors and settled results make unchanged evaluations
  idempotent. Correcting a result changes the hash and triggers a new
  evaluation on the next app startup.

### 3.2 Residuals, shrinkage and factor bounds

For a match with baseline goal rates \(\lambda_m,\mu_m\) and observed goals
\(g_{Hm},g_{Am}\), define goal residuals:

```text
home attack residual += g_Hm - lambda_m
away defence residual += g_Hm - lambda_m
away attack residual += g_Am - mu_m
home defence residual += g_Am - mu_m
```

The corresponding expected-goal totals are incremented by
\(\lambda_m\) or \(\mu_m\). The factors are estimated using the aggregate
residual divided by aggregate expected goals plus a 12-goal prior:

```text
attackLog_i  = attackResidual_i  / (attackExpected_i  + 12)
defenceLog_i = defenceResidual_i / (defenceExpected_i + 12)
rawFactor_i  = exp(clamp(logFactor_i, -log(1.25), +log(1.25)))
```

Adding 12 to the denominator shrinks estimates from small samples toward 1.
For each league and factor type, the raw values are divided by their geometric
mean so the league-wide geometric mean is approximately 1. The final
multipliers are bounded to `[1/1.25, 1.25]`, or `[0.8, 1.25]`.

The `defence` value is operationally a *goals-conceded multiplier*: a value
above 1 increases the opposing team's goal rate; a value below 1 reduces it.
This sign convention follows directly from how the factor is applied:

```text
lambdaAdjusted = lambdaBase * homeAttackFactor * awayDefenceFactor
muAdjusted     = muBase     * awayAttackFactor * homeDefenceFactor
```

Factors default to 1.0 if the team has no accepted factor. Name aliases are
applied before factor lookup.

### 3.3 Acceptance gate and evaluation metrics

The system requires at least 30 settled predictions for a league. It compares
the candidate walk-forward predictions to the probabilities saved when those
fixtures were first logged. A candidate is promoted only when both conditions
hold:

```text
candidate Brier < baseline Brier - 0.00001
candidate log-loss <= baseline log-loss + 0.00001
```

After a candidate is accepted, the factor set used for upcoming fixtures is
estimated from all settled matches in that league's latest season. The
walk-forward predictions above are used for the promotion decision; the final
factor estimates then use the full settled season history available at startup.

The multiclass Brier score used here is the mean of the per-match sum of
squared errors over home/draw/away:

```text
Brier = mean_m sum_k (p_mk - 1[actual_m = k])^2
```

The log-loss is the mean negative log probability assigned to the actual
outcome:

```text
LogLoss = -mean_m log(p_m,actual)
```

Lower values are better for both. Accuracy is also recorded but is **not** an
acceptance condition. Candidate probabilities are clipped away from zero and
renormalized before the metrics are calculated.

This gate intentionally favors conservative promotion: a candidate that
improves one proper scoring rule while materially worsening the other is
rejected. It is not a guarantee of future performance; these metrics are
computed on the available season history, and small or unusual samples remain
uncertain.

### 3.4 Current observed status

The local prediction database's latest recorded evaluations are dated
**2026-11-03** for season 2026. This is an operational snapshot, not a permanent
performance claim; it changes as results are settled and the app is restarted.

| League | Settled games used | Baseline Brier → candidate | Baseline log-loss → candidate | Candidate accuracy | Decision |
|---|---:|---:|---:|---:|---|
| Premier League | 60 | 0.6125 → 0.6272 | 1.0353 → 1.0595 | 46.7% | Rejected; both scores worsened |
| La Liga | 52 | 0.5449 → 0.5518 | 0.9231 → 0.9404 | 57.7% | Rejected; both scores worsened |
| Bundesliga | 9 | 0.6660 → 0.6661 | 1.0736 → 1.0737 | 33.3% | Rejected; below 30-game minimum |
| Serie A | 47 | 0.6001 → 0.5835 | 1.0146 → 0.9948 | 53.2% | Accepted; both scores improved |

In that snapshot, the adaptive rating layer is active for Serie A only. The
other leagues continue using their unmodified static DC priors. Candidate
accuracy should not be read as a standalone quality guarantee; the Brier and
log-loss gates determine promotion.

## 4. Live neural network

### 4.1 Feature vector and output

The network is a fully connected PyTorch classifier with 11 inputs, hidden
layers of 40 and 20 units, and 3 output logits. Each hidden layer uses ReLU and
dropout with probability 0.30 during training. At inference, `model.eval()`
disables dropout. The trained checkpoint contains 1,363 parameters.

Ignoring dropout for notation, the learned mapping is:

```text
h1 = ReLU(W1 * x + b1)          # 11 -> 40
h2 = ReLU(W2 * h1 + b2)         # 40 -> 20
z  = W3 * h2 + b3               # 20 -> 3 logits
```

The parameter count is `(11*40 + 40) + (40*20 + 20) + (20*3 + 3) = 1,363`.

The input order is part of the checkpoint contract and must match
`DCStrengthLookup.build_feature_row()` exactly:

| # | Feature | Definition |
|---:|---|---|
| 1 | `goal_diff` | Home score minus away score, clipped to `[-5, 5]` |
| 2 | `minute_norm` | `min(minute / 90, 1)` |
| 3 | `is_second_half` | 1 when minute > 45, otherwise 0 |
| 4 | `home_rank_norm` | Home team's normalized attack rating |
| 5 | `away_rank_norm` | Away team's normalized defensive strength |
| 6 | `rank_diff` | Home attack strength minus away defensive strength |
| 7 | `is_knockout` | Knockout flag; current match route passes 0 |
| 8 | `lead_changes_norm` | `lead_changes / max(1, goals_so_far)`; live inference approximates `lead_changes` as 1 if both teams have scored, otherwise 0 |
| 9 | `is_neutral_venue` | Neutral-venue flag; current match route passes 0 |
| 10 | `score_state` | 0 if home is behind, 1 if level, 2 if home is ahead |
| 11 | `strength_x_time` | `rank_diff * (1 - minute_norm)` |

The two normalized strength values use a single set of min/max bounds fit across
all leagues from the prior file. Attack strength maps \(\alpha\) linearly to
`[0,1]`; defensive strength maps \(1-\text{normalized}(\beta)\), so a lower
concession multiplier produces a higher defensive-strength feature. The same
league-specific 25th-percentile-alpha / 75th-percentile-beta fallback is used
for teams absent from priors.

The network does **not** consume xG, player identity, confirmed lineups,
bookmaker odds, or a separately supplied DC probability. The feature builder
does use team strength derived from the static DC priors. During a live match,
the current score and match minute are supplied by the scoreboard provider;
the model therefore does not estimate its own score or validate the feed. Its
training data has final scores but not goal timestamps; the dataset builder
approximates goal times by placing portions of goals at minutes 44 and 75.
Intermediate training snapshots are therefore synthetic reconstructions, not
observed live event histories.

Training/inference class indices and live route ordering are:

```text
network class 0 = away win
network class 1 = draw
network class 2 = home win
route response = [home %, draw %, away %]
```

For logits \(z_k\) and stored temperature \(T\), inference computes:

```text
p_k = exp(z_k / T) / sum_j exp(z_j / T)
```

The current artifact records `T = 1.008`. The route returns the probabilities
rounded to one decimal place. The neural network is not calibrated or
fine-tuned from the recent-results rating table.

### 4.2 Training data and procedure

The neural model is trained offline in
`v4_backend/notebooks/train_v4_neural.ipynb`. It consumes
`data/processed/features_v4.parquet`, generated from historical football
matches. The local parquet inspected for this documentation contains 140,620
snapshots from 7,156 matches across five leagues and four seasons, 2021/22
through 2024/25. The notebook's introductory markdown still calls this three
seasons; that description is stale. The dataset builder excludes season
`2526` from its training set.

Important safeguards and training choices:

1. Split groups by `match_id`, not by individual minute snapshot, so snapshots
   from one fixture cannot appear in both train and test splits. The stored
   run used 60% train, 20% validation, 20% test: 4,293, 1,431 and 1,432
   distinct matches respectively.
2. Fit `StandardScaler` on training features only; serialize it beside the
   checkpoint and apply the same scaler at inference.
3. Use weighted cross-entropy:

   ```text
   loss = -sum_i(weight[y_i] * log(p_i[y_i])) / sum_i(weight[y_i])
   ```

   with class weights `[1.0, 1.20, 1.0]` for `[away, draw, home]`. The modest
   draw upweight was selected after an elbow search on the 2024/25 holdout; it
   is not a claim that draws are 20% more likely. That season therefore
   informed tuning and is not an untouched final test for this choice.
4. Train in two stages: freeze the first layer for 12 epochs while training
   the remaining layers with Adam at `3e-3`; then unfreeze and fine-tune with
   Adam at `3e-4`, up to 40 epochs, stopping after six epochs without
   validation-loss improvement and restoring the best validation checkpoint.
5. Search temperatures from 0.5 to 3.0 on validation logits to minimize
   multiclass negative log-likelihood; save the selected temperature in the
   checkpoint.

The key feature change from the earlier classifier was replacing hand-curated
FIFA rank inputs with normalized attack/defence strengths derived from the DC
\(\alpha,\beta\) priors. The checkpoint retains the same 11→40→20→3 shape, but
the meaning of those strength inputs changed. Training warm-started the first
40-unit layer from the earlier `football_v2.pth` checkpoint (a compatible
40×11 weight slice), froze that layer for stage 1, and then fine-tuned it in
stage 2. The actual saved run uses fixed class weights `[1.0, 1.20, 1.0]`;
the notebook's introductory text referring to “sqrt-inverse” class weights is
outdated.

The notebook stores the trained weights, exact feature names, architecture,
temperature, warm-start metadata and strength-source metadata in the
checkpoint. The scaler is stored separately. Changing feature order or
normalization without rebuilding the model artifacts would silently change
what its weights mean.

### 4.3 Validation and the intended use of each model

The recorded grouped internal test run contained 28,167 snapshots from 1,432
unseen matches. It reported:

| Metric | Recorded internal test |
|---|---:|
| Three-way accuracy | 71.3% |
| Majority-class baseline | 43.6% |
| Log-loss | 0.6002 |
| Draw recall | 59.4% |
| Draw precision | 57.5% |

These are **grouped internal match-snapshot** results, not pre-kickoff accuracy
or a chronological future-season test, and not evidence that 28,167
independent matches were tested. Snapshots within a match are correlated, and
a live snapshot includes the score/time information that makes the final
result easier to infer.

For clarity, accuracy is the fraction of snapshots whose top-probability class
matches the final outcome. Draw recall is the fraction of actual draws that
were predicted as draws; draw precision is the fraction of predicted draws
that were actually draws. A high overall accuracy can coexist with poor draw
recall if the model mostly predicts home wins.

The separate 2025/26 pre-game holdout comparison recorded:

| Model used at minute 0 | Accuracy | Draw recall |
|---|---:|---:|
| Dixon–Coles with draw propensity 0.10 | 49.8% | 19.3% |
| Neural network | 49.3% | 7.0% |

Therefore the production decision is to use DC before kickoff and the neural
network for live match-state updates. A strong internal live score does not
justify using the neural network as the pre-match predictor. The holdout
numbers are historical evaluation results, not a rolling guarantee; the
site's current history page and model-update tables are the operational
sources for later outcomes.

### 4.4 Runtime data flow and fallback behavior

1. `main.py` loads `v4_priors.json`, the `.pth` checkpoint and the scaler;
   missing artifacts produce a startup warning and disable the corresponding
   calculation rather than loading guessed values.
2. The match route computes pre-match DC probabilities for the selected league.
   The adaptive factors multiply the expected-goal rates when accepted factors
   exist; otherwise their values are 1.
3. Once a provider reports the match as started and supplies a score/minute,
   the route builds the 11 features, standardizes them, runs CPU inference,
   temperature-scales the logits and returns the live probabilities.
4. During a match, the browser polls `/live/{match_id}` for scoreboard changes;
   the model runs again with the updated state.

The model itself has no direct clock or network dependency. Missing live status,
score, or minute data can delay or prevent live inference. At full time, the
network is conditioned on the final score and minute, but its output may still
be non-degenerate; it is not an independent forecast of an already-known
result.

## 5. Optional lineup adjustment: separate and heuristic

`v4_web/lineup_adjustment.py` applies manually assigned attack and defence
weights when confirmed starters are available. It is not the same as the
history-based rating layer and is not part of neural-network training.

For absent key players, attack losses are capped at 0.45 per team and
defensive-concession losses at 0.30:

```text
lambdaAdjusted = lambdaBase
                 * (1 - homeAttackLoss) * (1 + awayDefenceLoss)
muAdjusted     = muBase
                 * (1 - awayAttackLoss) * (1 + homeDefenceLoss)
```

Rates are clipped to `[0.05, 12]`; the scoreline matrix, Dixon–Coles
low-score correction and shared draw-propensity adjustment are then recomputed.
The match page currently constructs its lineup panel's base rates directly
from static priors, so that panel does not inherit accepted result-driven
factors. The background prediction job instead starts its stored
lineup-adjusted snapshot from the already logged pre-match rates, which may
include accepted factors. The weights and player list are hand-maintained,
matching uses normalized name substrings/last names, and the adjustment
depends on a confirmed lineup feed. Treat this panel as a heuristic
sensitivity estimate, not a validated player-impact model. It must not be
described as having been learned from every player's current season.

## 6. Artifacts, configuration and reproducibility

| Purpose | Default path | Override |
|---|---|---|
| Static DC priors | `v4_backend/v4_priors.json` | `FOOTBALL_PRIORS_PATH` |
| Neural checkpoint | `v4_backend/notebooks/v4_backend/models/football_v4.pth` | `FOOTBALL_MODEL_PATH` |
| Feature scaler | `v4_backend/notebooks/v4_backend/models/scaler_v4.pkl` | `FOOTBALL_SCALER_PATH` |
| Prediction history / adaptive state | `v4_web/data/predictions.db` | Defined by `v4_web/predictions.py` |
| Historical xG training ledger | `v4_historical_data.sqlite` | Used by offline training scripts |

To update static priors, refresh the historical xG ledger and run the
`v4_backend/notebooks/train_v4_priors.ipynb` notebook. To retrain the live
network, rebuild `features_v4.parquet` and run
`v4_backend/notebooks/train_v4_neural.ipynb`. Validate the resulting artifacts
on a chronologically separate season before replacing production files.

The web runtime does not scrape Understat, fit DC priors, train the neural
network, or update neural weights. It performs inference and a bounded,
result-history-based DC rating evaluation at startup.

## 7. Current boundaries and follow-up validation

- The adaptive rating layer is per league and currently only promotes if the
  candidate improves both Brier score and log-loss. Fewer than 30 settled
  predictions means no new promotion.
- Current-season result coverage, provider score delays, team aliases, and
  fixture-to-result matching affect the evidence available to the updater.
- A result correction is picked up on the next application startup because
  settled result hashes are recomputed then.
- The live classifier is a categorical outcome model. It does not generate
  exact future scorelines or use live xG.
- The optional lineup weights are manual and have not been independently
  validated by the walk-forward rating evaluation.
- Continue logging predictions before kickoff, preserve their original
  probabilities, settle results from verified scores, and report proper
  scoring metrics as well as accuracy. Do not select or tune a model solely
  because it had a good short streak.
- A production report should keep pre-match DC, post-lineup odds and live
  network evaluations separate; they are different models at different
  information times.
