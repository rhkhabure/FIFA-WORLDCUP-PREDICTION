import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nba_predictions


class NBAPredictionStatsTests(unittest.TestCase):
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
                pregame_logged_at TEXT,
                home_score INTEGER,
                away_score INTEGER,
                correct INTEGER,
                brier REAL
            )
        """)
        self.db.executemany(
            "INSERT INTO predictions "
            "(league, conference, correct, brier, pregame_logged_at, "
            "game_time_utc, home_win_prob, away_win_prob) "
            "VALUES ('nba', ?, ?, ?, ?, ?, ?, ?)",
            [
                ("East", 1, 0.1, "2026-11-09T00:00:00+00:00", "2026-11-10T00:00:00Z", 0.6, 0.4),
                ("East", 0, 0.4, "2026-11-09T00:00:00+00:00", "2026-11-10T00:00:00Z", 0.6, 0.4),
                ("West", 1, 0.2, "2026-11-09T00:00:00+00:00", "2026-11-10T00:00:00Z", 0.6, 0.4),
                ("Preseason", 0, 0.5, "2026-11-09T00:00:00+00:00", "2026-11-10T00:00:00Z", 0.6, 0.4),
                ("East", None, None, "2026-11-09T00:00:00+00:00", "2026-11-10T00:00:00Z", 0.6, 0.4),
                ("East", 1, 0.1, None, "2026-11-10T00:00:00Z", 0.6, 0.4),
            ],
        )
        self.db.commit()
        nba_predictions.set_db(self.db)

    def tearDown(self):
        nba_predictions.set_db(None)
        self.db.close()

    def test_stats_reflect_selected_conference_and_exclude_unplayed_from_accuracy(self):
        east = nba_predictions.get_stats("East")

        self.assertEqual(east["total"], 4)
        self.assertEqual(east["verified"], 3)
        self.assertEqual(east["unverified"], 1)
        self.assertEqual(east["played"], 2)
        self.assertEqual(east["correct"], 1)
        self.assertEqual(east["accuracy"], 50.0)
        self.assertEqual(east["avg_brier"], 0.25)

    def test_all_stats_include_both_conferences(self):
        stats = nba_predictions.get_stats()

        self.assertEqual(stats["total"], 6)
        self.assertEqual(stats["verified"], 5)
        self.assertEqual(stats["unverified"], 1)
        self.assertEqual(stats["played"], 4)
        self.assertEqual(stats["correct"], 2)
        self.assertEqual(stats["accuracy"], 50.0)

    def test_normalize_conference_reclassifies_legacy_preseason_rows(self):
        self.db.execute(
            """INSERT INTO predictions
               (league, conference, correct, brier, home_team_id, away_team_id)
               VALUES ('nba', 'Preseason', 1, 0.1, 1610612737, 1610612748)"""
        )
        self.db.execute(
            """INSERT INTO predictions
               (league, conference, correct, brier, home_team_id, away_team_id)
               VALUES ('nba', 'Preseason', 0, 0.4, 1610612743, 1610612738)"""
        )

        updated = nba_predictions.normalize_conferences()

        self.assertEqual(updated, 2)
        self.assertEqual(
            self.db.execute(
                "SELECT conference FROM predictions "
                "WHERE conference IN ('East', 'West') ORDER BY id DESC LIMIT 2"
            ).fetchall(),
            [("West",), ("East",)],
        )


if __name__ == "__main__":
    unittest.main()
