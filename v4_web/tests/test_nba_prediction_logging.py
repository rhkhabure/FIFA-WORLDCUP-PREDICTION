import sqlite3
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nba_predictions
from routes import nba as nba_routes


class NBAPredictionLoggingTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY,
                league TEXT,
                game_id TEXT,
                home_team TEXT,
                away_team TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                conference TEXT,
                game_time_utc TEXT,
                model_version TEXT,
                home_win_prob REAL,
                away_win_prob REAL,
                home_score INTEGER,
                away_score INTEGER,
                correct INTEGER,
                brier REAL,
                created_at TEXT
            )
        """)
        nba_predictions.set_db(self.db)
        self.now = datetime.fromisoformat("2030-01-01T00:00:00+00:00")
        self.game = {
            "game_id": "0022600197",
            "home_tricode": "LAL",
            "away_tricode": "DET",
            "home_team_id": 1610612747,
            "away_team_id": 1610612765,
            "game_time_utc": "2030-01-01T01:00:00Z",
            "status": "pre",
        }

    def tearDown(self):
        nba_predictions.set_db(None)
        self.db.close()

    def test_only_future_pre_tipoff_games_are_logged(self):
        self.assertFalse(nba_predictions.log_pregame(
            {**self.game, "status": "live"}, 0.7, now=self.now
        ))
        self.assertFalse(nba_predictions.log_pregame(
            {**self.game, "game_time_utc": "2026-11-08T23:59:00Z"},
            0.7,
            now=self.now,
        ))
        self.assertIsNone(nba_predictions.get_prediction(self.game["game_id"]))

    def test_first_pregame_snapshot_is_immutable_and_scored_from_saved_pick(self):
        self.assertTrue(nba_predictions.log_pregame(
            self.game,
            0.67,
            "nba-elo-result-updated-v1",
            now=self.now,
        ))
        later = datetime.fromisoformat("2030-01-01T00:30:00+00:00")
        self.assertFalse(nba_predictions.log_pregame(
            self.game, 0.2, "different-model-version", now=later
        ))

        snapshot = nba_predictions.get_prediction(self.game["game_id"])
        self.assertEqual(snapshot["home_win_prob"], 0.67)
        self.assertEqual(
            snapshot["pregame_logged_at"],
            "2030-01-01T00:00:00+00:00",
        )
        self.assertTrue(nba_predictions.is_verified_pregame(snapshot))

        with self.assertRaises(sqlite3.IntegrityError):
            self.db.execute(
                "UPDATE predictions SET home_win_prob=0.2 WHERE game_id=?",
                (self.game["game_id"],),
            )

        self.assertTrue(nba_predictions.log_result(
            self.game["game_id"], 110, 100
        ))
        snapshot = nba_predictions.get_prediction(self.game["game_id"])
        self.assertEqual(snapshot["correct"], 1)
        self.assertEqual(snapshot["home_win_prob"], 0.67)
        self.assertFalse(nba_predictions.log_result(
            self.game["game_id"], 90, 100
        ))

    def test_future_legacy_row_gets_its_first_verified_snapshot(self):
        self.db.execute("""
            INSERT INTO predictions
                (league, game_id, home_team, away_team, home_team_id,
                 away_team_id, game_time_utc, model_version, home_win_prob,
                 away_win_prob, created_at)
            VALUES ('nba', ?, 'LAL', 'DET', ?, ?, ?, 'legacy', 0.5, 0.5, '2030-01-01')
        """, (
            self.game["game_id"],
            self.game["home_team_id"],
            self.game["away_team_id"],
            self.game["game_time_utc"],
        ))
        self.db.commit()

        self.assertTrue(nba_predictions.log_pregame(
            self.game, 0.67, now=self.now
        ))
        snapshot = nba_predictions.get_prediction(self.game["game_id"])
        self.assertTrue(nba_predictions.is_verified_pregame(snapshot))
        self.assertEqual(snapshot["home_win_prob"], 0.67)

    def test_legacy_rows_can_store_scores_but_never_count_as_verified(self):
        self.db.execute("""
            INSERT INTO predictions
                (league, game_id, home_team, away_team, game_time_utc,
                 home_win_prob, away_win_prob, correct)
            VALUES ('nba', 'legacy', 'LAL', 'DET', ?, 0.7, 0.3, NULL)
        """, (self.game["game_time_utc"],))
        self.db.commit()

        self.assertTrue(nba_predictions.log_result("legacy", 110, 100))
        legacy = nba_predictions.get_prediction("legacy")
        self.assertEqual(legacy["home_score"], 110)
        self.assertIsNone(legacy["correct"])
        self.assertFalse(nba_predictions.is_verified_pregame(legacy))
        displayed = nba_routes._apply_saved_pregame({
            "game_id": "legacy",
            "home_win_prob": 0.2,
            "away_win_prob": 0.8,
        })
        self.assertFalse(displayed["pregame_verified"])
        self.assertIsNone(displayed["pregame_correct"])

    def test_dashboard_uses_saved_snapshot_not_current_probability(self):
        nba_predictions.log_pregame(self.game, 0.67, now=self.now)
        nba_predictions.log_result(self.game["game_id"], 110, 100)

        game = nba_routes._apply_saved_pregame({
            **self.game,
            "home_win_prob": 0.2,
            "away_win_prob": 0.8,
        })

        self.assertTrue(game["pregame_verified"])
        self.assertEqual(game["home_win_prob"], 0.67)
        self.assertEqual(game["pregame_correct"], 1)


if __name__ == "__main__":
    unittest.main()
