import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nba_api_client
from routes.nba import _get_team_standing, _ordinal_label


class FakeDataFrame:
    def to_dict(self, orient):
        assert orient == "records"
        return [
            {
                "TeamID": 1610612739,
                "Conference": "East",
                "WINS": 7,
                "LOSSES": 3,
                "WinPCT": 0.7,
                "PlayoffRank": 2,
            },
            {
                "TeamID": 1610612760,
                "Conference": "West",
                "WINS": 8,
                "LOSSES": 2,
                "WinPCT": 0.8,
                "PlayoffRank": 1,
            },
        ]


class FakeDataset:
    def get_data_frame(self):
        return FakeDataFrame()


class NBAStandingsTests(unittest.TestCase):
    def test_standings_are_grouped_and_mapped_to_team_records(self):
        with (
            patch.object(nba_api_client, "current_nba_season", return_value="2026-27"),
            patch(
                "nba_api.stats.endpoints.leaguestandingsv3.LeagueStandingsV3"
            ) as endpoint,
        ):
            endpoint.return_value.standings = FakeDataset()
            standings = nba_api_client.get_conference_standings()

        self.assertEqual(len(standings["East"]), 1)
        self.assertEqual(standings["East"][0]["tricode"], "CLE")
        self.assertEqual(standings["East"][0]["rank"], 2)
        self.assertEqual(standings["West"][0]["tricode"], "OKC")
        self.assertEqual(standings["West"][0]["wins"], 8)

    def test_team_rank_uses_its_conference_table_and_ordinal_suffix(self):
        standings = {
            "East": [{"team_id": 1610612739, "rank": 3}],
            "West": [{"team_id": 1610612760, "rank": 1}],
        }

        row = _get_team_standing(standings, 1610612739, "East")

        self.assertEqual(row["rank"], 3)
        self.assertIsNone(_get_team_standing(standings, 1610612739, "West"))
        self.assertEqual(
            [_ordinal_label(rank) for rank in (1, 2, 3, 4, 11, 12, 13, 21)],
            ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st"],
        )


if __name__ == "__main__":
    unittest.main()
