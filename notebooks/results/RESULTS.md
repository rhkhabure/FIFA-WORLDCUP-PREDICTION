# World Cup 2026 Win Probability — Results Log

The permanent record. Every run appends here.

## Phase 1 — data pipeline (2026-07-03 20:12 UTC)
- Total snapshots: 74,598
- Matches: 3,630  (WC: 128, league: 3502)
- Features: 11  |  Classes: 3
- Outcome split: home 42.8% / draw 25.2% / away 32.0%
- Saved to: data/processed/features_v2.parquet

## Phase 2 — build & train (2026-07-03 20:24 UTC)
- Architecture: 11 -> 24 -> 12 -> 3  (627 params)
- Warm-started from NBA weights: True
- Test accuracy: 0.469  (baseline always-home: 0.434)
- Test log-loss: 0.9037
- Temperature T: 0.881
- Draw recall: 0.825  (the hard class)
- Saved: models/football_v1.pth

## Phase 2b — revised build (2026-07-03 20:43 UTC)
- Fixes: sqrt-inverse class weights + bigger net (40->20, 1363 params)
- Test accuracy: 0.550  (v1 was 0.469, baseline 0.434)
- Test log-loss: 0.8782  (v1 was 0.9037)
- Draw recall: 0.319  (v1 0.825) | Draw precision: 0.401  (v1 0.295)
- Times model guessed draw: 2785 (v1 ~9790, real draws ~3500)
- Verdict: BETTER than v1
- Saved: models/football_v2.pth

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 09:07 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2.pth
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,451
- Accuracy: 0.723  (baseline always-home: 0.467)
- Log-loss: 0.6171
- For comparison, Phase 2 club-football test accuracy was: 0.550
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 09:18 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2.pth
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,451
- Accuracy: 0.723  (baseline always-home: 0.467)
- Log-loss: 0.6171
- For comparison, Phase 2 club-football test accuracy was: 0.550
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 09:26 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2.pth
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,439
- Accuracy: 0.712  (baseline always-home: 0.468)
- Log-loss: 0.6382
- For comparison, Phase 2 club-football test accuracy was: 0.550
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 09:29 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2.pth
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,439
- Accuracy: 0.712  (baseline always-home: 0.468)
- Log-loss: 0.6382
- For comparison, Phase 2 club-football test accuracy was: 0.550
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)

## Phase 1 — data pipeline (2026-07-04 09:35 UTC)
- Total snapshots: 74,598
- Matches: 3,630  (WC: 128, league: 3502)
- Features: 11  |  Classes: 3
- Outcome split: home 42.8% / draw 25.2% / away 32.0%
- Saved to: data/processed/features_v2.parquet

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 09:36 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2.pth
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,439
- Accuracy: 0.711  (baseline always-home: 0.468)
- Log-loss: 1.0538
- For comparison, Phase 2 club-football test accuracy was: 0.550
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)

## Phase 1 — data pipeline (2026-07-04 09:37 UTC)
- Total snapshots: 74,598
- Matches: 3,630  (WC: 128, league: 3502)
- Features: 11  |  Classes: 3
- Outcome split: home 42.8% / draw 25.2% / away 32.0%
- Saved to: data/processed/features_v2.parquet

## Phase 2 — build & train (2026-07-04 09:37 UTC)
- Architecture: 11 -> 24 -> 12 -> 3  (627 params)
- Warm-started from NBA weights: True
- Test accuracy: 0.470  (baseline always-home: 0.434)
- Test log-loss: 0.9251
- Temperature T: 0.881
- Draw recall: 0.790  (the hard class)
- Saved: models/football_v1.pth

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 09:38 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2.pth
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,439
- Accuracy: 0.711  (baseline always-home: 0.468)
- Log-loss: 1.0538
- For comparison, Phase 2 club-football test accuracy was: 0.550
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 09:38 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2.pth
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,439
- Accuracy: 0.711  (baseline always-home: 0.468)
- Log-loss: 1.0538
- For comparison, Phase 2 club-football test accuracy was: 0.550
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)

## Phase 2b — revised build (2026-07-04 09:40 UTC)
- Fixes: sqrt-inverse class weights + bigger net (40->20, 1363 params)
- Test accuracy: 0.551  (v1 was 0.469, baseline 0.434)
- Test log-loss: 0.9072  (v1 was 0.9037)
- Draw recall: 0.322  (v1 0.825) | Draw precision: 0.406  (v1 0.295)
- Times model guessed draw: 2774 (v1 ~9790, real draws ~3500)
- Verdict: BETTER than v1
- Saved: models/football_v2.pth

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 09:41 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2.pth
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,439
- Accuracy: 0.703  (baseline always-home: 0.468)
- Log-loss: 0.6869
- For comparison, Phase 2 club-football test accuracy was: 0.550
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)

## Phase 1 — data pipeline (2026-07-04 09:47 UTC)
- Total snapshots: 74,598
- Matches: 3,630  (WC: 128, league: 3502)
- Features: 11  |  Classes: 3
- Outcome split: home 42.8% / draw 25.2% / away 32.0%
- Saved to: data/processed/features_v2.parquet

## Phase 2b — revised build (2026-07-04 09:48 UTC)
- Fixes: sqrt-inverse class weights + bigger net (40->20, 1363 params)
- Test accuracy: 0.551  (v1 was 0.469, baseline 0.434)
- Test log-loss: 0.9072  (v1 was 0.9037)
- Draw recall: 0.322  (v1 0.825) | Draw precision: 0.406  (v1 0.295)
- Times model guessed draw: 2774 (v1 ~9790, real draws ~3500)
- Verdict: BETTER than v1
- Saved: models/football_v2.pth

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 09:48 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2.pth
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,439
- Accuracy: 0.703  (baseline always-home: 0.468)
- Log-loss: 0.6869
- For comparison, Phase 2 club-football test accuracy was: 0.550
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)

## Phase 2c — draw-weakness fixes (2026-07-04 10:07 UTC)
- Push 1: draw_signal feature (Dixon-Coles-inspired closeness signal), 12 features total
- Push 2: ordinal output head (2 sigmoids -> guaranteed-valid 3 probabilities) instead of softmax
- Test accuracy: 0.551  (v2 was 0.703, baseline 0.434)
- Test log-loss: 0.9030  (v2 was 0.6869)
- Draw recall: 0.317  (v2 0.319) | Draw precision: 0.405  (v2 0.401)
- Ordinal guarantee held on real data: P(draw) never negative, always sums to 1
- Saved: models/football_v2c.pth

## Phase 3 — REAL 2026 World Cup group-stage validation (2026-07-04 10:23 UTC)
**This is entry #1 of the real-games track record.**
- Model graded: football_v2c.pth (ordinal head + draw_signal feature)
- Matches graded: 68  (skipped 4 due to goal-count mismatch)
- Snapshots graded: 1,439
- Accuracy: 0.661  (baseline always-home: 0.468)
- Log-loss: 0.6931
- For comparison: v2 (softmax) scored 0.712 accuracy / 0.6382 log-loss on these SAME real games
- Known caveats: host-nation home advantage not modelled; some 2026 debut teams
  default to rank 50 (no real FIFA rank in our table)
