import asyncio
import sqlite3
import sys
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prediction_job import _dc_probs
from prediction_job import run_prediction_job_sync
from model_updates import (
    _dc_probabilities,
    get_football_match_factors,
    refresh_football_ratings,
    refresh_nba_elo,
)


class NBARatingUpdateTests(unittest.TestCase):
    def test_promotes_regular_season_replay_and_is_idempotent(self):
        db = sqlite3.connect(":memory:")
        db.execute("""
            CREATE TABLE predictions (
                game_id TEXT, league TEXT, home_team_id INTEGER,
                away_team_id INTEGER, home_win_prob REAL,
                home_score INTEGER, away_score INTEGER,
                created_at TEXT, conference TEXT
            )
        """)
        for index in range(40):
            db.execute("""
                INSERT INTO predictions
                    (game_id, league, home_team_id, away_team_id,
                     home_win_prob, home_score, away_score, created_at, conference)
                VALUES (?, 'nba', 1, 2, 0.5, 110, 100, ?, 'East')
            """, (f"002260{index:05d}", f"2026-11-{index % 28 + 1:02d}"))
        for prefix in ("001", "003", "004", "005"):
            db.execute("""
                INSERT INTO predictions
                    (game_id, league, home_team_id, away_team_id,
                     home_win_prob, home_score, away_score, created_at, conference)
                VALUES (?, 'nba', 1, 2, 0.5, 80, 120, '2026-11-30', 'East')
            """, (f"{prefix}2600001",))

        baseline = {"1": 1500.0, "2": 1500.0}
        first = refresh_nba_elo(db, baseline)
        again = refresh_nba_elo(db, baseline)

        self.assertEqual(first["sample_size"], 40)
        self.assertTrue(first["accepted"])
        self.assertFalse(again["changed"])
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM model_update_evaluations").fetchone()[0],
            1,
        )
        db.execute(
            "UPDATE predictions SET home_score=90, away_score=110 "
            "WHERE game_id='00226000000'"
        )
        corrected = refresh_nba_elo(db, baseline)
        self.assertTrue(corrected["changed"])
        self.assertEqual(
            db.execute("SELECT COUNT(*) FROM model_update_evaluations").fetchone()[0],
            2,
        )
        db.close()


class FootballRatingUpdateTests(unittest.TestCase):
    def test_prediction_job_looks_up_factors_by_context_key(self):
        priors = {
            "ENG-Premier League": {
                "meta": {"gamma_home_advantage": 1.25},
                "teams": {
                    "Home": {"alpha": 1.2, "beta": 1.0},
                    "Away": {"alpha": 0.9, "beta": 1.1},
                },
            },
        }
        with patch(
            "model_updates.get_football_match_factors",
            return_value=(1.1, 0.9),
        ) as factors:
            result = _dc_probs("Home", "Away", priors, "ENG-Premier League")

        factors.assert_called_once_with("pl", "Home", "Away")
        self.assertEqual(len(result), 7)
        self.assertGreater(result[3], 0)

    def test_promotes_shrunken_factors_and_restores_them_idempotently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "football.db"
            with closing(sqlite3.connect(db_path)) as db:
                db.execute("""
                    CREATE TABLE predictions (
                        match_id TEXT, home_team TEXT, away_team TEXT,
                        kickoff_utc TEXT, season INTEGER, league TEXT,
                        pre_dc_home REAL, pre_dc_draw REAL, pre_dc_away REAL,
                        pre_dc_lam REAL, pre_dc_mu REAL, actual_result TEXT,
                        actual_hg INTEGER, actual_ag INTEGER,
                        pre_rating_lam_factor REAL, pre_rating_mu_factor REAL
                    )
                """)
                probs = _dc_probabilities(1.0, 1.0, 0.0)
                for index in range(35):
                    db.execute("""
                        INSERT INTO predictions VALUES
                            (?, 'Home', 'Away', ?, 2026, 'pl',
                             ?, ?, ?, 1.0, 1.0, 'H', 3, 0, 1.0, 1.0)
                    """, (
                        f"match-{index}",
                        f"2026-08-{index % 28 + 1:02d}",
                        probs[0] * 100,
                        probs[1] * 100,
                        probs[2] * 100,
                    ))
                db.commit()

            priors = {
                "ENG-Premier League": {
                    "meta": {"rho_draw_correction": 0.0},
                },
            }
            first = refresh_football_ratings(db_path, priors)
            again = refresh_football_ratings(db_path, priors)

            evaluation = first["evaluations"][0]
            self.assertEqual(evaluation["sample_size"], 35)
            self.assertTrue(evaluation["accepted"])
            self.assertLess(
                evaluation["candidate"]["brier"],
                evaluation["baseline"]["brier"],
            )
            self.assertLess(
                evaluation["candidate"]["log_loss"],
                evaluation["baseline"]["log_loss"],
            )
            self.assertEqual(again["evaluations"], [])
            self.assertEqual(again["factors"]["pl"], first["factors"]["pl"])
            lam_factor, mu_factor = get_football_match_factors(
                "pl", "Home", "Away"
            )
            self.assertGreater(lam_factor, 1.0)
            self.assertLess(mu_factor, 1.0)


class PredictionJobSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_cycle_runs_outside_the_server_event_loop(self):
        async def blocking_cycle(*args, **kwargs):
            time.sleep(0.15)

        with patch("prediction_job.run_prediction_job", blocking_cycle):
            task = asyncio.create_task(
                asyncio.to_thread(run_prediction_job_sync, {}, "pl")
            )
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
            await task


if __name__ == "__main__":
    unittest.main()
