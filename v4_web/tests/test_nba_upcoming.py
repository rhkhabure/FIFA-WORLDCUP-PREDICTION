import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nba_api_client import get_live_game, get_next_team_game, get_scheduled_game


class NBAUpcomingGameTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone(timedelta(hours=3)))
        self.game = {
            "game_id": "nba-123",
            "home_team_id": 17,
            "away_team_id": 1,
            "status": "pre",
            "game_time_utc": "2026-11-06T00:00:00Z",
        }

    def scoreboard(self, date_str):
        return [self.game] if date_str == "11/05/2026" else []

    @patch("nba_api_client.get_scoreboard_by_date")
    def test_finds_next_game_across_eat_midnight(self, get_scoreboard):
        get_scoreboard.side_effect = self.scoreboard

        result = get_next_team_game(17, days_ahead=7, now=self.now)

        self.assertEqual(result["game_id"], "nba-123")

    @patch("nba_api_client.get_scoreboard_by_date")
    def test_finds_game_by_id_using_its_date(self, get_scoreboard):
        get_scoreboard.side_effect = self.scoreboard

        result = get_scheduled_game("nba-123", game_date="2026-11-06")

        self.assertEqual(result, self.game)

    @patch("nba_api_client.get_scheduled_game")
    @patch("nba_api_client.get_tomorrow_scoreboard", return_value=[])
    @patch("nba_api_client.get_today_scoreboard", return_value=[])
    def test_live_game_lookup_falls_back_to_requested_date(
        self, _today, _tomorrow, get_scheduled
    ):
        get_scheduled.return_value = self.game

        result = get_live_game("nba-123", "2026-11-06")

        self.assertEqual(result, self.game)
        get_scheduled.assert_called_once_with("nba-123", game_date="2026-11-06")


if __name__ == "__main__":
    unittest.main()
