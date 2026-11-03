import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routes import nba


class FakeModel:
    def predict_pregame(self, home_team_id, away_team_id):
        return {
            "home_win_prob": 0.65,
            "home_elo": 1600,
            "away_elo": 1500,
        }


class NBAMatchCurveTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_curve_includes_result_when_play_by_play_is_unavailable(self):
        game = {
            "home_team_id": 1,
            "away_team_id": 2,
            "home_score": 117,
            "away_score": 112,
            "status": "final",
        }
        with (
            patch.object(nba, "_model", FakeModel()),
            patch.object(nba, "build_snapshot_series", return_value=[]),
            patch.object(nba, "run_in_threadpool", new=AsyncMock(return_value=game)),
        ):
            response = await nba.nba_curve("game-1", "2026-11-03")

        payload = json.loads(response.body)
        self.assertEqual(len(payload["points"]), 2)
        self.assertEqual(payload["points"][-1]["hp"], 1.0)
        self.assertIn("play-by-play unavailable", payload["points"][-1]["desc"])


if __name__ == "__main__":
    unittest.main()
