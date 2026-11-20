import asyncio
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import main
from football_player_stats import get_player_season_stats
from v4_backend.import_player_stats import normalize_player_stats


class FootballPlayerStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "player-stats.sqlite"
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                """
                CREATE TABLE player_season_stats (
                    league TEXT, season INTEGER, season_label TEXT, team TEXT,
                    player TEXT, player_key TEXT, player_id INTEGER, position TEXT,
                    appearances INTEGER, minutes INTEGER, goals INTEGER,
                    assists INTEGER, xg REAL, xa REAL, shots INTEGER,
                    key_passes INTEGER, yellow_cards INTEGER, red_cards INTEGER,
                    np_goals INTEGER, np_xg REAL, xg_chain REAL, xg_buildup REAL,
                    PRIMARY KEY (league, season, player_id)
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO player_season_stats VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    ("pl", 2022, "2022-23", "Manchester City", "Test Player", "test player",
                     1, "Forward", 30, 1800, 10, 4, 8.0, 2.0, 40, 15, 2, 0,
                     9, 7.0, 12.0, 3.0),
                    ("pl", 2023, "2023-24", "Arsenal", "Test Player", "test player",
                     1, "Forward", 20, 900, 6, 3, 5.0, 1.0, 22, 8, 1, 0,
                     5, 4.0, 8.0, 2.0),
                    ("pl", 2023, "2023-24", "Arsenal", "Test Player Jr", "test player jr",
                     2, "Midfielder", 10, 500, 2, 1, 1.5, 0.5, 9, 5, 0, 0,
                     2, 1.2, 3.0, 1.0),
                    ("laliga", 2023, "2023-24", "Arsenal", "Test Player", "test player",
                     1, "Forward", 99, 5000, 99, 99, 99.0, 99.0, 99, 99, 0, 0,
                     99, 99.0, 99.0, 99.0),
                ],
            )
        connection.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_selects_player_season_and_computes_per_90_without_losing_old_clubs(self):
        result = get_player_season_stats(
            "Test Player",
            "pl",
            current_team="Arsenal",
            selected_season="pl:2022",
            database_path=self.database,
        )

        self.assertTrue(result["available"])
        self.assertEqual(len(result["seasons"]), 3)
        self.assertEqual(result["selected"]["team"], "Manchester City")
        self.assertEqual(result["selected"]["xg_per90"], 0.4)
        self.assertEqual(result["selected_season"], "pl:2022")

    def test_exact_name_search_covers_supported_leagues(self):
        result = get_player_season_stats(
            "Test Player",
            "pl",
            database_path=self.database,
        )
        self.assertEqual(len(result["seasons"]), 3)
        self.assertIn("La Liga", [row["league_name"] for row in result["seasons"]])

        missing = get_player_season_stats(
            "Player",
            "seriea",
            database_path=self.database,
        )
        self.assertFalse(missing["available"])
        self.assertTrue(missing["archive_available"])

    def test_normalizes_understat_multiindex_rows(self):
        index = pd.MultiIndex.from_tuples(
            [("ENG-Premier League", "2023", "Arsenal", "Test Player")],
            names=["league", "season", "team", "player"],
        )
        frame = pd.DataFrame(
            [{
                "player_id": 45, "position": "F", "matches": 20,
                "minutes": 900, "goals": 6, "assists": 3, "xg": 5.0,
                "xa": 1.0, "shots": 22, "key_passes": 8, "yellow_cards": 1,
                "red_cards": 0, "np_goals": 5, "np_xg": 4.0,
                "xg_chain": 8.0, "xg_buildup": 2.0,
            }],
            index=index,
        )

        rows = normalize_player_stats(frame, "pl", 2023)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], "2023-24")
        self.assertEqual(rows[0][4], "Test Player")
        self.assertEqual(rows[0][12], 5.0)

    def test_player_page_passes_selected_stats_season_to_local_lookup(self):
        request = Request({
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/player",
            "raw_path": b"/player",
            "query_string": b"name=Test+Player&team=Arsenal&league=pl&stats_season=pl%3A2022",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
        })
        stats = {
            "available": True, "seasons": [], "selected": {},
            "selected_season": "pl:2022",
        }
        with (
            patch.object(main, "search_player", return_value=None),
            patch.object(main, "get_career_history", return_value=[]),
            patch.object(main, "get_all_predictions", return_value=[]),
            patch.object(main, "get_player_season_stats", return_value=stats) as lookup,
            patch.object(
                main.templates,
                "TemplateResponse",
                side_effect=lambda **kwargs: kwargs["context"],
            ),
        ):
            context = asyncio.run(main.player_page(request))

        self.assertEqual(context["player_stats"]["selected_season"], "pl:2022")
        self.assertEqual(lookup.call_args.kwargs["selected_season"], "pl:2022")


if __name__ == "__main__":
    unittest.main()
