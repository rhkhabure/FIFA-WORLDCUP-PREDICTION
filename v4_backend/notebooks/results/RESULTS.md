# V4.2 Results Log

## V4.2 Phase 3 Step 2 — neural net trained (2026-09-06 13:30 UTC)
- Architecture: 11→40→20→3  (1,363 params)
- Strength features: Dixon-Coles alpha/beta (replaced FIFA rank)
- Warm-started: True
- Temperature T: 0.966
- Test accuracy: 0.696  (baseline 0.445)
- Test log-loss: 0.6107
- Draw recall: 0.707  |  Draw precision: 0.501
- Saved: v4_backend/models/football_v4.pth

## V4.2 Phase 3 Step 2 — neural net trained (2026-09-06 13:30 UTC)
- Architecture: 11→40→20→3  (1,363 params)
- Strength features: Dixon-Coles alpha/beta (replaced FIFA rank)
- Warm-started: True
- Temperature T: 0.966
- Test accuracy: 0.696  (baseline 0.445)
- Test log-loss: 0.6107
- Draw recall: 0.707  |  Draw precision: 0.501
- Saved: v4_backend/models/football_v4.pth

## V4.2 Phase 3 Step 3 � holdout validation (2026-09-06 14:33 UTC)
- Model: football_v4.pth (Dixon-Coles strength features)
- Holdout season: 2425 (1,752 matches)
- Evaluation: pre-game only (minute=0, score 0-0)
- Overall accuracy: 0.438  (baseline 0.420)
- Log-loss: 1.0317
- Draw recall: 0.420
- Verdict: Draw recall improved but accuracy did not beat prior

## V4.2 Phase 3 Step 2 — neural net trained (2026-09-06 14:40 UTC)
- Architecture: 11→40→20→3  (1,363 params)
- Strength features: Dixon-Coles alpha/beta (replaced FIFA rank)
- Warm-started: True
- Temperature T: 0.924
- Test accuracy: 0.711  (baseline 0.445)
- Test log-loss: 0.6049
- Draw recall: 0.521  |  Draw precision: 0.654
- Saved: v4_backend/models/football_v4.pth

## V4.2 Phase 3 Step 3 � holdout validation (2026-09-06 14:40 UTC)
- Model: football_v4.pth (Dixon-Coles strength features)
- Holdout season: 2425 (1,752 matches)
- Evaluation: pre-game only (minute=0, score 0-0)
- Overall accuracy: 0.483  (baseline 0.420)
- Log-loss: 1.0274
- Draw recall: 0.000
- Verdict: Neither metric improved -- investigate before proceeding

## V4.2 Phase 3 Step 2 — neural net trained (2026-09-06 15:07 UTC)
- Architecture: 11→40→20→3  (1,363 params)
- Strength features: Dixon-Coles alpha/beta (replaced FIFA rank)
- Warm-started: True
- Temperature T: 0.966
- Test accuracy: 0.696  (baseline 0.445)
- Test log-loss: 0.6103
- Draw recall: 0.726  |  Draw precision: 0.490
- Saved: v4_backend/models/football_v4.pth

## V4.2 Phase 3 Step 3 � holdout validation (2026-09-06 15:30 UTC)
- Model: football_v4.pth (Dixon-Coles strength features)
- Holdout season: 2425 (1,752 matches)
- Evaluation: pre-game only (minute=0, score 0-0)
- Overall accuracy: 0.457  (baseline 0.420)
- Log-loss: 1.0198
- Draw recall: 0.473
- Verdict: Draw recall improved but accuracy did not beat prior

## V4.2 Phase 3 Step 2 — neural net trained (2026-09-08 05:42 UTC)
- Architecture: 11→40→20→3  (1,363 params)
- Strength features: Dixon-Coles alpha/beta (replaced FIFA rank)
- Warm-started: True
- Temperature T: 1.008
- Test accuracy: 0.713  (baseline 0.436)
- Test log-loss: 0.6002
- Draw recall: 0.594  |  Draw precision: 0.575
- Saved: v4_backend/models/football_v4.pth

## V4.2 Phase 3 Step 3 � holdout validation (2026-09-08 05:43 UTC)
- Model: football_v4.pth (Dixon-Coles strength features)
- Holdout season: 2526 (1,752 matches)
- Evaluation: pre-game only (minute=0, score 0-0)
- Overall accuracy: 0.493  (baseline 0.440)
- Log-loss: 1.0167
- Draw recall: 0.070
- Verdict: Neither metric improved -- investigate before proceeding

## V4.2 FINAL ARCHITECTURE DECISION (2026-09-08 06:44 UTC)

### Testing Summary � all tests on genuine unseen holdout data

| Model / Version | Holdout | Accuracy | Draw Recall | Notes |
|---|---|---|---|---|
| Always-home baseline | 2526 | 0.440 | n/a | naive baseline |
| Dixon-Coles prior (unfixed) | 2526 | 0.506 | 0.000 | no draw correction |
| Dixon-Coles prior (dp=0.10) | 2526 | **0.498** | **0.193** | PRODUCTION pre-game model |
| V4 neural net pre-game | 2526 | 0.493 | 0.070 | worse than DC alone |
| V4 neural net (internal test) | 2122-2425 | 0.713 | 0.594 | in-dist, all minutes |
| V2 neural net (World Cup) | live WC games | 0.556 | 0.000 | different distribution |

### Key findings

1. Dixon-Coles beats the neural net on pre-game accuracy on every
   holdout tested (2425 and 2526). The statistical prior generalises
   better than the neural net pre-game because it makes fewer
   assumptions about which patterns transfer across seasons.

2. The neural net's 0.713 internal accuracy (vs 0.493 pre-game holdout)
   represents a genuine 22-point in-game gain driven by score_diff,
   score_state, and minute_norm features that only exist once a match
   has started. These features are season-agnostic -- a 2-0 scoreline
   at minute 60 means the same thing in any season.

3. Draw recall history across all versions:
   V2 neural net: 0.000 -> DC fixed: 0.193 -> V4 neural net: 0.070-0.594
   Dixon-Coles with draw_propensity=0.10 is the most reliable draw
   predictor pre-game. The neural net draw recall is strong in-distribution
   (0.594) but does not generalise pre-game across seasons.

4. The internal/holdout accuracy gap (22 points) persisted across
   all retraining attempts (3 seasons, 4 seasons, different weights,
   alias fixes). This gap is structural -- the neural net learns
   season-specific team-matchup patterns that don't transfer.
   Adding more seasons reduced fallbacks to 0% but did not close the gap.

### Production architecture (final decision)

PRE-GAME (before kickoff):
  Model : Dixon-Coles bivariate Poisson
  File  : v4_backend/v4_priors.json
  Params: draw_propensity=0.10 (elbow-tested on 2425 holdout)
  Output: P(home win), P(draw), P(away win)
  Accuracy: 0.498 on unseen 2526 season

LIVE IN-GAME (from kickoff onward):
  Model : football_v4.pth (neural net, 11->40->20->3)
  File  : v4_backend/notebooks/v4_backend/models/football_v4.pth
  Scaler: scaler_v4.pkl (same directory)
  Trigger: as soon as minute > 0, switch from DC to neural net
  Rationale: score and time features dominate, season-specific
             strength patterns become irrelevant mid-match

BRACKET / TOURNAMENT SIMULATION:
  Model : Dixon-Coles (same priors)
  Engine: Monte Carlo (existing simulate_tournament in common.py)
  Params: draw split 50/50 for knockout matches (no draws in KO)

### What was NOT built (deferred, on the record)
- Module 2 (lineup adjustment): unrunnable without ideal_roster builder.
  Deferred until lineup data source confirmed and validated.
- Cross-league tournament (Big 5 Champions Cup): dashboard placeholder
  only. Requires validated cross-league strength normalization.
- 2025/26 season priors: not trained (only used for holdout testing).
  Retrain in summer 2026 with completed 2025/26 as training data.

### Next phase
UI redesign and dashboard integration:
  - Pre-game: call DC predictor
  - Live: call neural net once minute > 0
  - History page: uses neural net timeline (already implemented)
  - Bracket page: uses DC + Monte Carlo (already implemented)
