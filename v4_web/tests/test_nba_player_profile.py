import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import historical_nba
import nba_api_client
from teamdata import _flag_img
from routes.nba import (
    _has_one_franchise_career,
    _player_experience,
    _player_honors,
    _player_team_timeline,
)


class NBAPlayerProfileTests(unittest.TestCase):
    def test_profile_keeps_full_career_rows_and_awards(self):
        class Dataset:
            def __init__(self, rows):
                self.rows = rows

            def get_data_frame(self):
                return self

            def to_dict(self, orient):
                self_orient = orient
                assert self_orient == "records"
                return self.rows

        class PlayerInfo:
            def __init__(self, **_kwargs):
                self.common_player_info = Dataset([{"DISPLAY_FIRST_LAST": "Test"}])
                self.player_headline_stats = Dataset([])

        class CareerStats:
            def __init__(self, **_kwargs):
                self.season_totals_regular_season = Dataset(
                    [
                        {"SEASON_ID": f"{year}-{str(year + 1)[-2:]}", "GP": 1}
                        for year in range(2000, 2020)
                    ]
                )

        class PlayerAwards:
            def __init__(self, **_kwargs):
                self.player_awards = Dataset(
                    [{"DESCRIPTION": "NBA Champion", "SEASON": "2019-20"}]
                )

        player_id = 9999991
        nba_api_client._cache.pop(f"player_profile_{player_id}", None)
        with (
            patch(
                "nba_api.stats.endpoints.commonplayerinfo.CommonPlayerInfo",
                PlayerInfo,
            ),
            patch(
                "nba_api.stats.endpoints.playercareerstats.PlayerCareerStats",
                CareerStats,
            ),
            patch(
                "nba_api.stats.endpoints.playerawards.PlayerAwards",
                PlayerAwards,
            ),
        ):
            profile = nba_api_client.get_player_profile(player_id)

        self.assertEqual(len(profile["career_seasons"]), 20)
        self.assertEqual(len(profile["seasons"]), 5)
        self.assertTrue(profile["career_history_available"])
        self.assertTrue(profile["awards_available"])
        self.assertEqual(profile["awards"][0]["DESCRIPTION"], "NBA Champion")

    def test_experience_tiers_count_only_seasons_with_games(self):
        labels = [
            (1, "Rookie"),
            (2, "Early Career"),
            (4, "Established"),
            (8, "Veteran"),
            (12, "Long-Tenured"),
            (16, "Career Veteran"),
        ]
        for count, label in labels:
            with self.subTest(count=count):
                seasons = [
                    {"SEASON_ID": f"{2000 + i}-{str(2001 + i)[-2:]}", "GP": 1}
                    for i in range(count)
                ]
                seasons.append({"SEASON_ID": "2026-27", "GP": 0})
                result = _player_experience(seasons, True)
                self.assertEqual(result["label"], label)
                self.assertEqual(result["seasons_played"], count)

        self.assertFalse(_player_experience([], False)["available"])
        self.assertIsNone(_player_experience([], False)["label"])

    def test_honors_count_championships_and_regular_season_mvps_only(self):
        awards = [
            {"DESCRIPTION": "NBA Champion", "SEASON": "2012-13",
             "TEAM": "Miami Heat"},
            {"DESCRIPTION": "NBA Champion", "SEASON": "2019-20",
             "TEAM": "Los Angeles Lakers"},
            {"DESCRIPTION": "NBA Champion", "SEASON": "2019-20"},
            {"DESCRIPTION": "NBA Most Valuable Player", "SEASON": "2020-21"},
            {"DESCRIPTION": "NBA Most Valuable Player", "SEASON": "2021-22"},
            {"DESCRIPTION": "NBA Finals Most Valuable Player", "SEASON": "2021-22"},
        ]
        result = _player_honors(awards, True)
        self.assertEqual(result["championships"], 2)
        self.assertEqual(result["mvps"], 2)
        self.assertEqual(
            result["championship_seasons"],
            [
                {
                    "season": "2012-13",
                    "team_id": 1610612748,
                    "team_name": "Miami Heat",
                    "tricode": "MIA",
                },
                {
                    "season": "2019-20",
                    "team_id": 1610612747,
                    "team_name": "Los Angeles Lakers",
                    "tricode": "LAL",
                },
            ],
        )
        self.assertFalse(_player_honors([], False)["available"])

    def test_nba_country_abbreviation_resolves_to_existing_flag(self):
        self.assertIn("flagcdn.com/16x12/us.png", _flag_img("USA"))

    def test_team_timeline_groups_multiple_teams_in_a_season(self):
        api_rows = [
            {"SEASON_ID": "2023-24", "TEAM_ID": 1610612744,
             "TEAM_ABBREVIATION": "GSW", "GP": 40},
            {"SEASON_ID": "2024-25", "TEAM_ID": 1610612739,
             "TEAM_ABBREVIATION": "CLE", "GP": 30},
            {"SEASON_ID": "2024-25", "TEAM_ID": 1610612744,
             "TEAM_ABBREVIATION": "GSW", "GP": 20},
            {"SEASON_ID": "2024-25", "TEAM_ID": 0,
             "TEAM_ABBREVIATION": "TOT", "GP": 50},
        ]
        timeline = _player_team_timeline(api_rows, [])
        self.assertEqual(len(timeline["seasons"]), 2)
        self.assertEqual(len(timeline["seasons"][1]["teams"]), 2)
        experience = {"available": True, "seasons_played": 3}
        self.assertFalse(_has_one_franchise_career(timeline, experience))

        loyal_timeline = _player_team_timeline(
            [
                {"SEASON_ID": f"{year}-{str(year + 1)[-2:]}",
                 "TEAM_ID": 1610612744, "TEAM_ABBREVIATION": "GSW", "GP": 1}
                for year in range(2022, 2025)
            ],
            [],
        )
        self.assertTrue(_has_one_franchise_career(loyal_timeline, experience))
        self.assertFalse(
            _has_one_franchise_career(
                loyal_timeline, {"available": False, "seasons_played": 20}
            )
        )

    def test_best_shot_excludes_dunks_and_requires_minimum_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "shots.sqlite"
            connection = sqlite3.connect(database)
            connection.execute(
                """CREATE TABLE play_by_play (
                       person_id INTEGER, game_id TEXT, game_date TEXT,
                       action_type TEXT, shot_value INTEGER, subtype TEXT
                   )"""
            )
            for player_id, subtype, made, missed in (
                (1, "Dunk Shot", 80, 0),
                (1, "Jump Shot", 20, 10),
                (1, "Layup Shot", 23, 7),
                (2, "Dunk Shot", 80, 0),
                (2, "Layup Shot", 20, 4),
            ):
                connection.executemany(
                    """INSERT INTO play_by_play VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        (player_id, "0022400001", "2024-01-01",
                         "Made Shot", 2, subtype)
                    ] * made
                    + [
                        (player_id, "0022400001", "2024-01-01",
                         "Missed Shot", 2, subtype)
                    ] * missed,
                )
            connection.commit()
            connection.close()

            with patch.object(
                historical_nba, "get_play_by_play_db_path", return_value=database
            ):
                best_profile = historical_nba.get_player_shot_profile(1)
                low_sample_profile = historical_nba.get_player_shot_profile(2)

        self.assertEqual(best_profile["favorite"]["shot_type"], "Dunk")
        self.assertEqual(best_profile["best"]["shot_type"], "Layup")
        self.assertEqual(best_profile["best"]["attempts"], 30)
        self.assertIsNone(low_sample_profile["best"])
        self.assertEqual(low_sample_profile["best_min_attempts"], 25)


if __name__ == "__main__":
    unittest.main()
