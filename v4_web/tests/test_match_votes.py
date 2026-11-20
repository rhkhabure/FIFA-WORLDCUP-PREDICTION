import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import routes.match as match_route


class MatchVoteLookupTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.prediction = {
            "match_id": "ll_fixture-1",
            "league": "laliga",
            "home_team": "Atletico Madrid",
            "away_team": "Barcelona",
            "kickoff_utc": (self.now + timedelta(days=2)).isoformat(),
            "actual_result": None,
        }

    def test_matches_bbs_fixture_id_to_prediction_id(self):
        result = match_route._find_open_vote_match_id(
            [self.prediction],
            "bbs_fixture-1",
            "laliga",
            "Some other provider name",
            "Some other away name",
            self.now,
        )

        self.assertEqual(result, "ll_fixture-1")

    def test_matches_provider_team_aliases(self):
        aliases = {"Atletico de Madrid": "Atletico Madrid"}
        with patch.object(match_route, "TEAM_NAME_ALIASES", aliases):
            result = match_route._find_open_vote_match_id(
                [self.prediction],
                "football-data-id",
                "laliga",
                "Atletico de Madrid",
                "Barcelona",
                self.now,
            )

        self.assertEqual(result, "ll_fixture-1")

    def test_does_not_offer_closed_or_other_league_polls(self):
        finished = {**self.prediction, "actual_result": "H"}
        other_league = {**self.prediction, "league": "seriea"}

        self.assertIsNone(match_route._find_open_vote_match_id(
            [finished], "bbs_fixture-1", "laliga", "Home", "Away", self.now
        ))
        self.assertIsNone(match_route._find_open_vote_match_id(
            [other_league], "bbs_fixture-1", "laliga", "Home", "Away", self.now
        ))


if __name__ == "__main__":
    unittest.main()
