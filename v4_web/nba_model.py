"""
nba_model.py  —  NBA Win Probability V3 model loader + inference
================================================================
Loads the trained WinProbNet from nba/model/, exposes two public
functions:

    predict_pregame(home_elo, away_elo, is_playoffs) -> float
        Returns P(home team wins) before tip-off.

    predict_live(snapshot) -> float
        Returns P(home team wins) from a live game-state dict.

Drop this file in v4_web/. The model files live in v4_web/nba/model/.

Usage in routes/nba.py:
    from nba_model import nba_model
    prob = nba_model.predict_live(snapshot)
"""

import json
import pickle
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit  # sigmoid

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
MODEL_DIR    = _HERE / "nba" / "model"
WEIGHTS_PATH = MODEL_DIR / "win_prob_net_v3.pth"
CONFIG_PATH  = MODEL_DIR / "model_config.json"
SCALER_PATH  = MODEL_DIR / "scaler_v3.pkl"
TEMP_PATH    = MODEL_DIR / "calibration_t.pkl"
ELO_PATH     = MODEL_DIR / "elo_ratings_v3.json"

# ── Feature order (must match Phase 1 FEATURE_COLS exactly) ──────────────────
FEATURE_COLS = [
    "score_diff",
    "time_remaining_sec",
    "quarter",
    "quarter_time_elapsed_pct",
    "home_elo",
    "away_elo",
    "elo_diff",
    "is_playoffs",
    "is_overtime",
    "lead_changes_norm",
    "possession",
    "home_in_bonus",
    "away_in_bonus",
    "home_avail_delta",
    "away_avail_delta",
    "elo_prior_weight",
]
N_FEATURES = len(FEATURE_COLS)

# ── League-wide constants ─────────────────────────────────────────────────────
ELO_START      = 1500
HOME_ADVANTAGE = 100   # Elo points — same as training


# ─────────────────────────────────────────────────────────────────────────────
# Network definition — must match phase2_training.py WinProbNet exactly
# ─────────────────────────────────────────────────────────────────────────────

class WinProbNet(nn.Module):
    def __init__(self, n_features, hidden_dims, dropout=0.25, use_batchnorm=True):
        super().__init__()
        layers = []
        in_dim = n_features
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))

    def logits(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper — singleton loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────

class NBAModel:
    """
    Singleton wrapper around WinProbNet. Loaded once in FastAPI lifespan,
    then reused for every prediction request.
    """

    def __init__(self):
        self._model:   WinProbNet | None = None
        self._scaler   = None
        self._T:       float = 1.0
        self._elo:     dict  = {}
        self._config:  dict  = {}
        self.loaded:   bool  = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """Load all model artefacts. Call once from FastAPI lifespan."""
        try:
            # Config
            with open(CONFIG_PATH, encoding="utf-8") as f:
                self._config = json.load(f)

            mc = self._config.get("model_config", {})
            n_feat  = mc.get("n_features", N_FEATURES)
            h_dims  = mc.get("hidden_dims", [128, 64, 32])
            dropout = mc.get("dropout", 0.25)
            bn      = mc.get("use_batchnorm", True)

            # Network
            self._model = WinProbNet(n_feat, h_dims, dropout, bn)

            # Weights — handle both state_dict-only and wrapped saves
            raw = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            if isinstance(raw, dict) and "state_dict" in raw:
                state_dict = raw["state_dict"]
            elif isinstance(raw, dict) and any(
                k.startswith("net.") for k in raw.keys()
            ):
                state_dict = raw
            else:
                state_dict = raw
            self._model.load_state_dict(state_dict)
            self._model.eval()

            # Scaler
            with open(SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)

            # Temperature
            with open(TEMP_PATH, "rb") as f:
                t_data = pickle.load(f)
            self._T = float(t_data.get("T", 1.0)) if isinstance(t_data, dict) else float(t_data)

            # Elo ratings
            if ELO_PATH.exists():
                with open(ELO_PATH, encoding="utf-8") as f:
                    self._elo = {str(k): float(v) for k, v in json.load(f).items()}

            self.loaded = True
            metrics = self._config.get("val_metrics", {})
            log.info(
                "NBA model loaded — AUC %.4f  Brier %.4f  T=%.4f  Elo teams=%d",
                metrics.get("roc_auc", 0),
                metrics.get("brier", 0),
                self._T,
                len(self._elo),
            )

        except Exception as exc:
            log.error("NBA model failed to load: %s", exc, exc_info=True)
            self.loaded = False

    # ── Elo helpers ───────────────────────────────────────────────────────────

    def get_elo(self, team_id: str | int) -> float:
        """Return current Elo for a team ID. Falls back to league average."""
        return self._elo.get(str(team_id), float(ELO_START))

    def pregame_win_prob(self, home_elo: float, away_elo: float) -> float:
        """
        Simple Elo-only pregame win probability (no neural net).
        Used for pre-tip-off display and as the prior anchor.
        P(home wins) = 1 / (1 + 10^((away_elo - (home_elo + home_adv)) / 400))
        """
        adj_home = home_elo + HOME_ADVANTAGE
        return 1.0 / (1.0 + 10.0 ** ((away_elo - adj_home) / 400.0))

    # ── Feature builder ───────────────────────────────────────────────────────

    def _build_feature_vector(self, snapshot: dict) -> np.ndarray:
        """
        Convert a live game-state dict into the 16-feature vector
        the neural net expects. All features have defaults so partial
        snapshots work gracefully.

        Expected snapshot keys (all optional, have sensible defaults):
            score_diff          int   home_score - away_score
            time_remaining_sec  float seconds left in regulation (0 in OT)
            quarter             int   1-4 (5+ = OT)
            quarter_time_elapsed_pct  float 0.0-1.0
            home_elo            float pre-game Elo, home team
            away_elo            float pre-game Elo, away team
            is_playoffs         int   0 or 1
            is_overtime         int   0 or 1
            lead_changes_norm   float lead_changes / plays_so_far
            possession          float 1.0 = home has ball, 0.0 = away
            home_in_bonus       int   0 or 1
            away_in_bonus       int   0 or 1
            home_avail_delta    float DARKO availability adjustment (default 0)
            away_avail_delta    float same for away
        """
        s = snapshot

        score_diff         = float(s.get("score_diff", 0))
        time_remaining_sec = float(s.get("time_remaining_sec", 2880))
        quarter            = int(s.get("quarter", 1))
        is_overtime        = int(s.get("is_overtime", 0))

        # Quarter time elapsed
        qte = s.get("quarter_time_elapsed_pct")
        if qte is None:
            clock = s.get("clock_sec", 720 if quarter <= 4 else 300)
            period_len = 300.0 if quarter >= 5 else 720.0
            qte = 1.0 - min(float(clock) / period_len, 1.0)
        qte = float(qte)

        home_elo = float(s.get("home_elo", ELO_START))
        away_elo = float(s.get("away_elo", ELO_START))
        elo_diff = home_elo - away_elo

        is_playoffs       = int(s.get("is_playoffs", 0))
        lead_changes_norm = float(s.get("lead_changes_norm", 0.0))
        possession        = float(s.get("possession", 0.5))
        home_in_bonus     = int(s.get("home_in_bonus", 0))
        away_in_bonus     = int(s.get("away_in_bonus", 0))
        home_avail_delta  = float(s.get("home_avail_delta", 0.0))
        away_avail_delta  = float(s.get("away_avail_delta", 0.0))

        # State-dependent Elo prior weight — decays as margin widens and time runs out
        # Matches training formula exactly
        elo_prior_weight = (
            expit(time_remaining_sec / 2880.0) *
            expit(1.0 / (abs(score_diff) + 1.0))
        )

        return np.array([
            np.clip(score_diff, -60, 60),
            time_remaining_sec,
            quarter,
            qte,
            home_elo,
            away_elo,
            elo_diff,
            is_playoffs,
            is_overtime,
            lead_changes_norm,
            possession,
            home_in_bonus,
            away_in_bonus,
            home_avail_delta,
            away_avail_delta,
            elo_prior_weight,
        ], dtype=np.float32)

    # ── Public inference API ──────────────────────────────────────────────────

    def predict_pregame(
        self,
        home_team_id: str | int,
        away_team_id: str | int,
        is_playoffs: int = 0,
        home_avail_delta: float = 0.0,
        away_avail_delta: float = 0.0,
    ) -> dict:
        """
        Pregame win probability from Elo only (neural net not used —
        at tipoff there is no game-state to feed the net).

        Returns:
            {
                "home_win_prob": float,   # P(home wins) 0-1
                "away_win_prob": float,
                "home_elo": float,
                "away_elo": float,
                "elo_diff": float,
            }
        """
        h_elo = self.get_elo(home_team_id)
        a_elo = self.get_elo(away_team_id)
        p     = self.pregame_win_prob(h_elo, a_elo)

        return {
            "home_win_prob": round(p, 4),
            "away_win_prob": round(1.0 - p, 4),
            "home_elo":      round(h_elo, 1),
            "away_elo":      round(a_elo, 1),
            "elo_diff":      round(h_elo - a_elo, 1),
        }

    def predict_live(self, snapshot: dict) -> dict:
        """
        Live in-game win probability from the neural net.

        snapshot must contain at minimum: score_diff, time_remaining_sec,
        quarter. Everything else has a safe default.

        Returns:
            {
                "home_win_prob": float,   # P(home wins) 0-1, temperature-scaled
                "away_win_prob": float,
                "elo_prior_weight": float,  # how much Elo still matters
                "confidence": str,          # "high" | "medium" | "low"
            }
        """
        if not self.loaded or self._model is None:
            # Fallback to Elo-only if model not loaded
            h_elo = float(snapshot.get("home_elo", ELO_START))
            a_elo = float(snapshot.get("away_elo", ELO_START))
            p     = self.pregame_win_prob(h_elo, a_elo)
            return {
                "home_win_prob":    round(p, 4),
                "away_win_prob":    round(1.0 - p, 4),
                "elo_prior_weight": 1.0,
                "confidence":       "low",
                "source":           "elo_fallback",
            }

        vec  = self._build_feature_vector(snapshot)
        X    = self._scaler.transform(vec.reshape(1, -1)).astype(np.float32)
        X_t  = torch.from_numpy(X)

        self._model.eval()
        with torch.no_grad():
            logit = self._model.logits(X_t).item()

        # Temperature scaling
        p = float(expit(logit / self._T))
        p = max(0.001, min(0.999, p))

        # Confidence: how far from 50/50?
        dist = abs(p - 0.5)
        confidence = "high" if dist > 0.25 else "medium" if dist > 0.10 else "low"

        return {
            "home_win_prob":    round(p, 4),
            "away_win_prob":    round(1.0 - p, 4),
            "elo_prior_weight": round(float(vec[FEATURE_COLS.index("elo_prior_weight")]), 3),
            "confidence":       confidence,
            "source":           "neural_net",
        }

    def predict_live_batch(self, snapshots: list[dict]) -> list[dict]:
        """
        Batch inference for multiple snapshots (e.g. chart history rebuild).
        More efficient than calling predict_live() in a loop.
        """
        if not self.loaded or self._model is None:
            return [self.predict_live(s) for s in snapshots]

        vecs = np.stack([self._build_feature_vector(s) for s in snapshots])
        X    = self._scaler.transform(vecs).astype(np.float32)
        X_t  = torch.from_numpy(X)

        self._model.eval()
        with torch.no_grad():
            logits = self._model.logits(X_t).squeeze(-1).numpy()

        results = []
        for i, logit in enumerate(logits):
            p    = float(expit(logit / self._T))
            p    = max(0.001, min(0.999, p))
            dist = abs(p - 0.5)
            results.append({
                "home_win_prob":    round(p, 4),
                "away_win_prob":    round(1.0 - p, 4),
                "elo_prior_weight": round(float(vecs[i][FEATURE_COLS.index("elo_prior_weight")]), 3),
                "confidence":       "high" if dist > 0.25 else "medium" if dist > 0.10 else "low",
                "source":           "neural_net",
            })
        return results

    # ── Model metadata ────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return model metadata for the /nba/debug or dashboard header."""
        if not self.loaded:
            return {"loaded": False}
        metrics = self._config.get("val_metrics", {})
        return {
            "loaded":      True,
            "roc_auc":     metrics.get("roc_auc"),
            "brier":       metrics.get("brier"),
            "accuracy":    metrics.get("accuracy"),
            "temperature": self._T,
            "n_teams_elo": len(self._elo),
            "features":    N_FEATURES,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this in routes/nba.py and main.py:
#   from nba_model import nba_model
nba_model = NBAModel()
