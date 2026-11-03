"""Conservative, result-driven rating updates and walk-forward evaluation."""

import hashlib
import json
import logging
import math
import sqlite3
import threading
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from constants import DRAW_PROPENSITY

log = logging.getLogger(__name__)

NBA_ELO_K = 20.0
NBA_HOME_ADVANTAGE = 100.0
NBA_MIN_EVALUATION_GAMES = 30
FOOTBALL_MIN_EVALUATION_GAMES = 30
FOOTBALL_PRIOR_GOALS = 12.0
FOOTBALL_MAX_FACTOR = 1.25

_lock = threading.RLock()
_active_football_factors: dict[str, dict[str, dict[str, float]]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ensure_tables(db: sqlite3.Connection) -> None:
    db.execute("""
        CREATE TABLE IF NOT EXISTS model_update_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            league TEXT NOT NULL,
            evaluated_at TEXT NOT NULL,
            sample_size INTEGER NOT NULL,
            baseline_brier REAL,
            candidate_brier REAL,
            baseline_log_loss REAL,
            candidate_log_loss REAL,
            candidate_accuracy REAL,
            accepted INTEGER NOT NULL,
            parameters TEXT NOT NULL
        )
    """)
    db.commit()


def _binary_metrics(probabilities: list[float], outcomes: list[int]) -> dict:
    if not probabilities:
        return {"count": 0, "brier": None, "log_loss": None, "accuracy": None}
    clipped = [min(max(float(p), 1e-6), 1.0 - 1e-6) for p in probabilities]
    return {
        "count": len(clipped),
        "brier": sum((p - y) ** 2 for p, y in zip(clipped, outcomes)) / len(clipped),
        "log_loss": -sum(
            y * math.log(p) + (1 - y) * math.log(1 - p)
            for p, y in zip(clipped, outcomes)
        ) / len(clipped),
        "accuracy": sum((p >= 0.5) == bool(y) for p, y in zip(clipped, outcomes))
        / len(clipped),
    }


def _record_evaluation(
    db: sqlite3.Connection,
    model: str,
    league: str,
    baseline: dict,
    candidate: dict,
    accepted: bool,
    parameters: dict,
) -> None:
    _ensure_tables(db)
    db.execute("""
        INSERT INTO model_update_evaluations
            (model, league, evaluated_at, sample_size, baseline_brier,
             candidate_brier, baseline_log_loss, candidate_log_loss,
             candidate_accuracy, accepted, parameters)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model, league, _utc_now(), candidate["count"],
        baseline["brier"], candidate["brier"],
        baseline["log_loss"], candidate["log_loss"],
        candidate["accuracy"], int(accepted),
        json.dumps(parameters, sort_keys=True),
    ))
    db.commit()


def _nba_expected(home_rating: float, away_rating: float) -> float:
    adjusted_home = home_rating + NBA_HOME_ADVANTAGE
    return 1.0 / (1.0 + 10.0 ** ((away_rating - adjusted_home) / 400.0))


def refresh_nba_elo(
    db: sqlite3.Connection,
    baseline_ratings: dict[str, float],
    minimum_games: int = NBA_MIN_EVALUATION_GAMES,
) -> dict:
    """Replay finalized regular-season NBA results and promote Elo if validated."""
    with _lock:
        for column, definition in (
            ("game_time_utc", "TEXT"),
            ("model_version", "TEXT NOT NULL DEFAULT 'nba-elo-static-v0'"),
        ):
            try:
                db.execute(
                    f"ALTER TABLE predictions ADD COLUMN {column} {definition}"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        _ensure_tables(db)
        db.execute("""
            CREATE TABLE IF NOT EXISTS nba_elo_state (
                baseline_hash TEXT PRIMARY KEY,
                results_hash TEXT NOT NULL,
                active_ratings TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        base = {str(team_id): float(rating) for team_id, rating in baseline_ratings.items()}
        if not base:
            log.warning("Skipping NBA Elo update because no baseline ratings are loaded")
            return {
                "ratings": {},
                "accepted": False,
                "sample_size": 0,
                "changed": False,
            }
        baseline_hash = _stable_hash(base)
        rows = db.execute("""
            SELECT game_id, home_team_id, away_team_id, home_win_prob,
                   home_score, away_score, created_at, conference,
                   game_time_utc, model_version
            FROM predictions
            WHERE league='nba' AND home_team_id IS NOT NULL
              AND away_team_id IS NOT NULL
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY COALESCE(game_time_utc, created_at), created_at, game_id
        """).fetchall()
        games = []
        for row in rows:
            game_id = str(row[0] or "")
            # Only regular-season games are included; exclude preseason,
            # All-Star, play-in, and playoff results.
            if not game_id.startswith("002") or row[7] == "Preseason":
                continue
            games.append(row)

        results_hash = _stable_hash([
            (
                str(row[0]), str(row[1]), str(row[2]), int(row[4]),
                int(row[5]), row[8],
            )
            for row in games
        ])
        state = db.execute(
            "SELECT results_hash, active_ratings, accepted "
            "FROM nba_elo_state WHERE baseline_hash=?",
            (baseline_hash,),
        ).fetchone()
        if state and state[0] == results_hash:
            return {
                "ratings": json.loads(state[1]),
                "accepted": bool(state[2]),
                "sample_size": len(games),
                "changed": False,
            }

        candidate_ratings = dict(base)
        baseline_probs: list[float] = []
        candidate_probs: list[float] = []
        outcomes: list[int] = []
        for row in games:
            _, home_id, away_id, stored_prob, home_score, away_score = row[:6]
            home_id, away_id = str(home_id), str(away_id)
            home_rating = candidate_ratings.get(home_id, 1500.0)
            away_rating = candidate_ratings.get(away_id, 1500.0)
            candidate_probs.append(_nba_expected(home_rating, away_rating))
            if stored_prob is not None:
                baseline_probs.append(float(stored_prob))
            else:
                baseline_probs.append(_nba_expected(
                    base.get(home_id, 1500.0), base.get(away_id, 1500.0)
                ))
            home_won = int(home_score > away_score)
            outcomes.append(home_won)

            expected_home = _nba_expected(home_rating, away_rating)
            change = NBA_ELO_K * (home_won - expected_home)
            candidate_ratings[home_id] = home_rating + change
            candidate_ratings[away_id] = away_rating - change

        baseline_metrics = _binary_metrics(baseline_probs, outcomes)
        candidate_metrics = _binary_metrics(candidate_probs, outcomes)
        accepted = (
            len(games) >= minimum_games
            and candidate_metrics["brier"] is not None
            and candidate_metrics["log_loss"] is not None
            and candidate_metrics["brier"] < baseline_metrics["brier"] - 1e-5
            and candidate_metrics["log_loss"] <= baseline_metrics["log_loss"] + 1e-5
        )
        if accepted:
            active_ratings = candidate_ratings
        elif state:
            active_ratings = json.loads(state[1])
        else:
            active_ratings = base
        is_active = accepted or bool(state and state[2])

        db.execute("""
            INSERT INTO nba_elo_state
                (baseline_hash, results_hash, active_ratings, accepted, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(baseline_hash) DO UPDATE SET
                results_hash=excluded.results_hash,
                active_ratings=excluded.active_ratings,
                accepted=excluded.accepted,
                updated_at=excluded.updated_at
        """, (
            baseline_hash, results_hash,
            json.dumps(active_ratings, sort_keys=True),
            int(is_active), _utc_now(),
        ))
        _record_evaluation(
            db, "nba_elo", "nba", baseline_metrics, candidate_metrics, accepted,
            {
                "k_factor": NBA_ELO_K,
                "home_advantage": NBA_HOME_ADVANTAGE,
                "minimum_games": minimum_games,
            },
        )
        return {
            "ratings": active_ratings,
            "accepted": is_active,
            "sample_size": len(games),
            "changed": True,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        }


def get_active_football_factors() -> dict[str, dict[str, dict[str, float]]]:
    with _lock:
        return json.loads(json.dumps(_active_football_factors))


def get_football_match_factors(
    league: str, home_team: str, away_team: str
) -> tuple[float, float]:
    """Return multiplicative lambda/mu adjustments for one football fixture."""
    league_factors = get_active_football_factors().get(league, {})
    home = league_factors.get(home_team, {})
    away = league_factors.get(away_team, {})
    lam_factor = float(home.get("attack", 1.0)) * float(away.get("defense", 1.0))
    mu_factor = float(away.get("attack", 1.0)) * float(home.get("defense", 1.0))
    return lam_factor, mu_factor


def _dc_probabilities(lam: float, mu: float, rho: float) -> tuple[float, float, float]:
    lam = float(np.clip(lam, 1e-5, 15.0))
    mu = float(np.clip(mu, 1e-5, 15.0))
    goals = np.arange(9)
    joint = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    joint[0, 0] *= max(1.0 - lam * mu * rho, 1e-5)
    joint[1, 0] *= max(1.0 + mu * rho, 1e-5)
    joint[0, 1] *= max(1.0 + lam * rho, 1e-5)
    joint[1, 1] *= max(1.0 - rho, 1e-5)
    joint /= joint.sum()
    ph = float(np.tril(joint, -1).sum())
    pd = float(np.trace(joint))
    pa = float(np.triu(joint, 1).sum())
    half_draw = DRAW_PROPENSITY / 2.0
    ph = max(ph - half_draw, 0.0)
    pa = max(pa - half_draw, 0.0)
    pd += DRAW_PROPENSITY
    total = ph + pd + pa
    return ph / total, pd / total, pa / total


def _factor_map(residuals: dict, expected: dict) -> dict[str, dict[str, float]]:
    teams = set(residuals["attack"]) | set(residuals["defense"])
    factors = {}
    max_log = math.log(FOOTBALL_MAX_FACTOR)
    for team in teams:
        attack_log = residuals["attack"][team] / (
            expected["attack"][team] + FOOTBALL_PRIOR_GOALS
        )
        defense_log = residuals["defense"][team] / (
            expected["defense"][team] + FOOTBALL_PRIOR_GOALS
        )
        factors[team] = {
            "attack": math.exp(max(-max_log, min(max_log, attack_log))),
            "defense": math.exp(max(-max_log, min(max_log, defense_log))),
        }

    for field in ("attack", "defense"):
        values = [entry[field] for entry in factors.values()]
        if values:
            geometric_mean = math.exp(sum(math.log(v) for v in values) / len(values))
            for entry in factors.values():
                entry[field] = min(
                    FOOTBALL_MAX_FACTOR,
                    max(1.0 / FOOTBALL_MAX_FACTOR, entry[field] / geometric_mean),
                )
    return factors


def _football_games(path: Path) -> dict[str, list[sqlite3.Row]]:
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT league, season, match_id, home_team, away_team, kickoff_utc,
                   pre_dc_home, pre_dc_draw, pre_dc_away, pre_dc_lam, pre_dc_mu,
                   actual_result, actual_hg, actual_ag,
                   COALESCE(pre_rating_lam_factor, 1.0),
                   COALESCE(pre_rating_mu_factor, 1.0)
            FROM predictions
            WHERE actual_result IS NOT NULL
              AND actual_hg IS NOT NULL AND actual_ag IS NOT NULL
              AND pre_dc_home IS NOT NULL AND pre_dc_draw IS NOT NULL
              AND pre_dc_away IS NOT NULL
              AND pre_dc_lam IS NOT NULL AND pre_dc_mu IS NOT NULL
            ORDER BY league, season, kickoff_utc, match_id
        """).fetchall()
    latest_season: dict[str, int] = {}
    for row in rows:
        league = row["league"] or "pl"
        season = int(row["season"] or 0)
        latest_season[league] = max(latest_season.get(league, season), season)
    result: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        league = row["league"] or "pl"
        if int(row["season"] or 0) == latest_season[league]:
            result[league].append(row)
    return result


def refresh_football_ratings(
    db_path: str | Path,
    priors_db: dict,
    minimum_games: int = FOOTBALL_MIN_EVALUATION_GAMES,
) -> dict:
    """Estimate shrunken attack/defense multipliers and validate walk-forward."""
    global _active_football_factors
    path = Path(db_path)
    if not path.is_file():
        return {}
    with _lock:
        from v4_backend.feature_builder import TEAM_NAME_ALIASES

        games_by_league = _football_games(path)
        accepted_factors: dict[str, dict[str, dict[str, float]]] = {}
        evaluations = []
        for league, games in games_by_league.items():
            league_cfg = {
                "ENG-Premier League": "pl",
                "ESP-La Liga": "laliga",
                "GER-Bundesliga": "bundesliga",
                "ITA-Serie A": "seriea",
            }
            priors_key = next(
                (key for key, value in league_cfg.items() if value == league),
                league,
            )
            league_priors = priors_db.get(priors_key, {})
            meta = league_priors.get("meta", {})
            rho = float(meta.get("rho_draw_correction", 0.0))
            season = int(games[0]["season"] or 0)
            priors_hash = _stable_hash(league_priors)
            results_hash = _stable_hash([
                (
                    row["match_id"], row["home_team"], row["away_team"],
                    row["kickoff_utc"], row["actual_hg"], row["actual_ag"],
                    row["pre_dc_lam"], row["pre_dc_mu"],
                    row[14], row[15],
                )
                for row in games
            ])
            with closing(sqlite3.connect(path)) as db:
                _ensure_tables(db)
                db.execute("""
                    CREATE TABLE IF NOT EXISTS football_rating_state (
                        league TEXT NOT NULL,
                        season INTEGER NOT NULL,
                        priors_hash TEXT NOT NULL,
                        results_hash TEXT NOT NULL,
                        factors TEXT NOT NULL,
                        accepted INTEGER NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (league, season, priors_hash)
                    )
                """)
                previous = db.execute("""
                    SELECT results_hash, factors, accepted
                    FROM football_rating_state
                    WHERE league=? AND season=? AND priors_hash=?
                """, (league, season, priors_hash)).fetchone()
                if previous and previous[0] == results_hash:
                    accepted_factors[league] = json.loads(previous[1])
                    continue

            residuals = {"attack": defaultdict(float), "defense": defaultdict(float)}
            expected = {"attack": defaultdict(float), "defense": defaultdict(float)}
            baseline_probs: list[tuple[float, float, float]] = []
            candidate_probs: list[tuple[float, float, float]] = []
            outcomes: list[str] = []

            for row in games:
                home = TEAM_NAME_ALIASES.get(row["home_team"], row["home_team"])
                away = TEAM_NAME_ALIASES.get(row["away_team"], row["away_team"])
                factors = _factor_map(residuals, expected)
                home_factors = factors.get(home, {"attack": 1.0, "defense": 1.0})
                away_factors = factors.get(away, {"attack": 1.0, "defense": 1.0})
                applied_lam = max(float(row[14] or 1.0), 1e-6)
                applied_mu = max(float(row[15] or 1.0), 1e-6)
                candidate_lam = (
                    float(row["pre_dc_lam"])
                    * home_factors["attack"] * away_factors["defense"] / applied_lam
                )
                candidate_mu = (
                    float(row["pre_dc_mu"])
                    * away_factors["attack"] * home_factors["defense"] / applied_mu
                )
                candidate_probs.append(_dc_probabilities(candidate_lam, candidate_mu, rho))
                baseline_probs.append((
                    float(row["pre_dc_home"] or 0.0) / 100.0,
                    float(row["pre_dc_draw"] or 0.0) / 100.0,
                    float(row["pre_dc_away"] or 0.0) / 100.0,
                ))
                outcomes.append(row["actual_result"])

                lam = float(row["pre_dc_lam"]) / applied_lam
                mu = float(row["pre_dc_mu"]) / applied_mu
                hg, ag = int(row["actual_hg"]), int(row["actual_ag"])
                residuals["attack"][home] += hg - lam
                expected["attack"][home] += lam
                residuals["defense"][away] += hg - lam
                expected["defense"][away] += lam
                residuals["attack"][away] += ag - mu
                expected["attack"][away] += mu
                residuals["defense"][home] += ag - mu
                expected["defense"][home] += mu

            baseline_metrics = _multiclass_metrics(baseline_probs, outcomes)
            candidate_metrics = _multiclass_metrics(candidate_probs, outcomes)
            accepted = (
                len(games) >= minimum_games
                and candidate_metrics["brier"] is not None
                and candidate_metrics["log_loss"] is not None
                and candidate_metrics["brier"] < baseline_metrics["brier"] - 1e-5
                and candidate_metrics["log_loss"] <= baseline_metrics["log_loss"] + 1e-5
            )
            final_factors = (
                _factor_map(residuals, expected)
                if accepted else json.loads(previous[1]) if previous else {}
            )
            accepted_factors[league] = final_factors

            with closing(sqlite3.connect(path)) as db:
                _ensure_tables(db)
                db.execute("""
                    INSERT INTO football_rating_state
                        (league, season, priors_hash, results_hash,
                         factors, accepted, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(league, season, priors_hash) DO UPDATE SET
                        results_hash=excluded.results_hash,
                        factors=excluded.factors,
                        accepted=excluded.accepted,
                        updated_at=excluded.updated_at
                """, (
                    league, season, priors_hash, results_hash,
                    json.dumps(final_factors, sort_keys=True),
                    int(accepted or bool(previous and previous[2])),
                    _utc_now(),
                ))
                _record_evaluation(
                    db, "football_team_ratings", league,
                    baseline_metrics, candidate_metrics, accepted,
                    {
                        "season": season,
                        "minimum_games": minimum_games,
                        "prior_expected_goals": FOOTBALL_PRIOR_GOALS,
                        "max_factor": FOOTBALL_MAX_FACTOR,
                    },
                )
            evaluations.append({
                "league": league,
                "sample_size": len(games),
                "accepted": accepted,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
            })

        _active_football_factors.update(accepted_factors)
        for result in evaluations:
            log.info(
                "Football rating evaluation %s: %d games, candidate accepted=%s",
                result["league"], result["sample_size"], result["accepted"],
            )
        return {"factors": accepted_factors, "evaluations": evaluations}


def _multiclass_metrics(
    probabilities: list[tuple[float, float, float]], outcomes: list[str]
) -> dict:
    if not probabilities:
        return {"count": 0, "brier": None, "log_loss": None, "accuracy": None}
    labels = {"H": 0, "D": 1, "A": 2}
    brier, losses, correct = [], [], 0
    for probs, outcome in zip(probabilities, outcomes):
        if outcome not in labels:
            continue
        p = [min(max(float(value), 1e-6), 1.0) for value in probs]
        total = sum(p)
        p = [value / total for value in p]
        actual = labels[outcome]
        brier.append(sum((value - (i == actual)) ** 2 for i, value in enumerate(p)))
        losses.append(-math.log(p[actual]))
        correct += int(max(range(3), key=p.__getitem__) == actual)
    count = len(brier)
    return {
        "count": count,
        "brier": sum(brier) / count if count else None,
        "log_loss": sum(losses) / count if count else None,
        "accuracy": correct / count if count else None,
    }
