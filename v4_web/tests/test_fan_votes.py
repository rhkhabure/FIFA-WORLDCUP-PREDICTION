import sqlite3
import sys
import tempfile
import unittest
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fan_votes import cast_vote, get_vote_summary, init_fan_votes


class FanVoteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "predictions.db"
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute("""
                CREATE TABLE predictions (
                    match_id TEXT PRIMARY KEY,
                    kickoff_utc TEXT,
                    actual_result TEXT
                )
            """)
            future = datetime.now(timezone.utc) + timedelta(hours=2)
            db.execute(
                "INSERT INTO predictions VALUES (?, ?, NULL)",
                ("future", future.isoformat()),
            )
            db.execute(
                "INSERT INTO predictions VALUES (?, ?, ?)",
                ("finished", future.isoformat(), "H"),
            )
            db.commit()
        init_fan_votes(self.db_path)
        self.now = datetime.now(timezone.utc)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_accepts_one_vote_and_summarizes_counts(self):
        voter = str(uuid.uuid4())
        result = cast_vote(
            self.db_path, "future", "H", voter, now=self.now
        )

        self.assertTrue(result["accepted"])
        self.assertEqual(result["counts"], {"H": 1, "D": 0, "A": 0})
        self.assertEqual(result["total"], 1)
        self.assertTrue(get_vote_summary(self.db_path, "future", voter)["has_voted"])

    def test_rejects_duplicate_vote_for_same_browser_and_match(self):
        voter = str(uuid.uuid4())
        cast_vote(self.db_path, "future", "H", voter, now=self.now)

        duplicate = cast_vote(
            self.db_path, "future", "A", voter, now=self.now
        )

        self.assertFalse(duplicate["accepted"])
        self.assertEqual(duplicate["reason"], "already_voted")
        self.assertEqual(duplicate["counts"], {"H": 1, "D": 0, "A": 0})

    def test_rejects_unknown_invalid_and_closed_votes(self):
        voter = str(uuid.uuid4())
        self.assertEqual(
            cast_vote(self.db_path, "missing", "H", voter, now=self.now)["reason"],
            "unknown_match",
        )
        self.assertEqual(
            cast_vote(self.db_path, "future", "X", voter, now=self.now)["reason"],
            "invalid_choice",
        )
        self.assertEqual(
            cast_vote(self.db_path, "future", "H", "not-a-uuid", now=self.now)["reason"],
            "invalid_voter",
        )
        self.assertEqual(
            cast_vote(self.db_path, "finished", "H", voter, now=self.now)["reason"],
            "poll_closed",
        )

    def test_closes_poll_at_kickoff(self):
        voter = str(uuid.uuid4())
        past_kickoff = self.now - timedelta(seconds=1)
        with closing(sqlite3.connect(self.db_path)) as db:
            db.execute(
                "UPDATE predictions SET kickoff_utc=? WHERE match_id='future'",
                (past_kickoff.isoformat(),),
            )
            db.commit()

        result = cast_vote(self.db_path, "future", "D", voter, now=self.now)

        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "poll_closed")


if __name__ == "__main__":
    unittest.main()
