import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nba_api_client


class NBAPBPFallbackTests(unittest.TestCase):
    def test_espn_plays_are_reoriented_to_the_nba_game_home_team(self):
        home_id = 1610612760
        away_id = 1610612739
        game = {
            "game_id": "game-1",
            "home_team_id": home_id,
            "away_team_id": away_id,
        }
        plays = [
            {
                "type": {"text": "Driving Layup Shot"},
                "team": {"id": "5"},
                "period": {"number": 1},
                "clock": {"displayValue": "11:43"},
                "homeScore": 2,
                "awayScore": 0,
                "text": "Cleveland scores",
            },
            {
                "type": {"text": "Jump Shot"},
                "team": {"id": "25"},
                "period": {"number": 1},
                "clock": {"displayValue": "11:23"},
                "homeScore": 2,
                "awayScore": 2,
                "text": "Oklahoma City scores",
            },
        ]
        with (
            patch.object(nba_api_client, "_playbyplay", side_effect=ValueError("feed down")),
            patch.object(
                nba_api_client,
                "_get_espn_game_plays",
                return_value=(plays, {"5": away_id, "25": home_id}, True),
            ),
        ):
            actions = nba_api_client.get_live_pbp("game-1", game)

        self.assertEqual(actions[0]["teamId"], away_id)
        self.assertEqual((actions[0]["scoreHome"], actions[0]["scoreAway"]), (0, 2))
        series = nba_api_client.build_snapshot_series(
            "game-1", home_id, game=game
        )
        self.assertEqual(
            (series[-1]["home_score"], series[-1]["away_score"]),
            (2, 2),
        )


if __name__ == "__main__":
    unittest.main()
