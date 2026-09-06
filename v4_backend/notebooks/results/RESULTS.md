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
