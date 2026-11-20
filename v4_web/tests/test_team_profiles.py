import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


class TeamProfileTests(unittest.TestCase):
    def test_finds_bundesliga_team_after_provider_name_cleanup(self):
        team = main._find_league_team_info(
            [{"name": "Bayern München", "id": 5}],
            "Bayern Munich",
        )

        self.assertEqual(team["id"], 5)

    def test_next_fixture_uses_home_away_data(self):
        fixture = main._next_fixture_from_team_fixtures(
            [{
                "match_id": 42,
                "home": "Barcelona",
                "away": "Real Madrid",
                "kickoff_eat": "Sat 21 Nov · 18:30",
            }],
            "Barcelona",
        )

        self.assertEqual(fixture["opponent"], "Real Madrid")
        self.assertTrue(fixture["is_home"])

    def test_laliga_uses_shared_profile_when_feed_metadata_is_missing(self):
        with (
            patch.object(main, "get_competition_teams_fd", return_value=[]),
            patch.object(main, "get_team_details_fd", return_value={}),
            patch.object(main, "get_team_matches_fd", return_value=[]),
            patch.object(main, "get_competition_standings_fd", return_value=[]),
            patch.object(
                main.templates,
                "TemplateResponse",
                side_effect=lambda **kwargs: kwargs["context"],
            ),
        ):
            context = asyncio.run(
                main.league_team_page(None, "laliga", "Barcelona", season=2025)
            )

        self.assertEqual(context["team_page_league"], "laliga")
        self.assertEqual(context["profile"]["name"], "Barcelona")
        self.assertTrue(context["pitch_svg"])

    def test_inter_uses_serie_a_fallback_profile_when_feed_is_unavailable(self):
        with (
            patch.object(main, "get_competition_teams_fd", return_value=[]),
            patch.object(main, "get_team_details_fd", return_value={}),
            patch.object(main, "get_team_matches_fd", return_value=[]),
            patch.object(main, "get_competition_standings_fd", return_value=[]),
            patch.object(
                main.templates,
                "TemplateResponse",
                side_effect=lambda **kwargs: kwargs["context"],
            ),
        ):
            context = asyncio.run(
                main.league_team_page(None, "seriea", "Inter", season=2026)
            )

        self.assertEqual(context["team_id"], 108)
        self.assertEqual(context["profile"]["venue"], "San Siro")
        self.assertEqual(context["pitch_player_count"], 11)
        self.assertTrue(context["squad_is_fallback"])
        self.assertTrue(context["pitch_svg"])

    def test_known_team_renders_when_profile_provider_is_unavailable(self):
        with (
            patch.object(main, "get_team_profile", return_value={}),
            patch.object(main, "get_formation_for_team", return_value={}),
            patch.object(main, "get_squad_for_team", return_value=[]),
            patch.object(main, "generate_pitch_svg_vertical", return_value="<svg/>"),
            patch.object(main, "get_crest_proxy_url", return_value=""),
            patch.object(main, "get_upcoming_fixtures", return_value=[]),
            patch.object(main, "get_team_season_results", return_value=[]),
            patch.object(main, "get_standings", return_value=[]),
            patch.object(main, "get_last_n_results", return_value=[]),
            patch.object(main, "get_next_fixture", return_value=None),
            patch.object(main, "priors_db", None),
            patch.object(
                main.templates,
                "TemplateResponse",
                side_effect=lambda **kwargs: kwargs["context"],
            ),
        ):
            context = asyncio.run(
                main.team_profile(None, 57, season=2025)
            )

        self.assertFalse(context["profile_data_available"])
        self.assertEqual(context["profile"]["name"], "Arsenal")


if __name__ == "__main__":
    unittest.main()
