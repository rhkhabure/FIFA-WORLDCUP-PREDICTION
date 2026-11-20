import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routes import nba


class NBATeamSeasonRecordTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("""
            CREATE TABLE predictions (
                league TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_score INTEGER,
                away_score INTEGER,
                game_time_utc TEXT,
                created_at TEXT
            )
        """)
        self.db.executemany(
            "INSERT INTO predictions VALUES ('nba', ?, ?, ?, ?, ?, ?)",
            [
                (10, 20, 100, 90, "2026-10-10T00:00:00Z", "2026-10-10"),
                (20, 10, 90, 100, "2026-10-12T00:00:00Z", "2026-10-12"),
                (20, 10, 100, 90, "2026-10-14T00:00:00Z", "2026-10-14"),
                (20, 10, 100, 90, "", "2026-10-15"),
                (10, 20, 100, 90, "2026-08-31T00:00:00Z", "2026-08-31"),
                (10, 20, 100, 90, "2027-09-01T00:00:00Z", "2027-09-01"),
                (10, 20, None, None, "2026-10-16T00:00:00Z", "2026-10-16"),
                (30, 20, 100, 90, "2026-10-18T00:00:00Z", "2026-10-18"),
            ],
        )
        self.previous_db = nba._db
        nba._db = self.db

    def tearDown(self):
        nba._db = self.previous_db
        self.db.close()

    def test_record_uses_team_scores_and_season_date_range(self):
        self.assertEqual(
            nba._get_nba_team_season_record(10, "2026-27"),
            {"wins": 2, "losses": 2, "played": 4},
        )


if __name__ == "__main__":
    unittest.main()
