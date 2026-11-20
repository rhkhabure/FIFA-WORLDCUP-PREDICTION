import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictions import (
    find_pregame_prediction,
    get_accuracy_stats,
    prediction_correct_for_score,
)


ALIASES = {
    "Nott'm Forest": "Nottingham Forest",
    "Man City": "Manchester City",
    "RC Celta de Vigo": "Celta Vigo",
}


def saved_prediction(**overrides):
    prediction = {
        "league": "pl",
        "match_id": "saved-provider-id",
        "home_team": "Nott'm Forest",
        "away_team": "Man City",
        "kickoff_utc": "2026-11-08T00:00:00Z",
        "pre_dc_home": 20,
        "pre_dc_draw": 25,
        "pre_dc_away": 55,
    }
    return {**prediction, **overrides}


def test_fixture_lookup_uses_aliases_and_tolerates_midnight_kickoff():
    prediction = saved_prediction()

    match = find_pregame_prediction(
        [prediction],
        league="pl",
        match_id="fixture-provider-id",
        home_team="Nottingham Forest",
        away_team="Manchester City",
        kickoff_utc="2026-11-07T15:00:00Z",
        team_aliases=ALIASES,
    )

    assert match is prediction


def test_fixture_lookup_prefers_nearest_repeat_fixture():
    old_match = saved_prediction(
        match_id="old",
        kickoff_utc="2026-03-01T15:00:00Z",
    )
    nearest_match = saved_prediction(match_id="nearest")

    match = find_pregame_prediction(
        [old_match, nearest_match],
        league="pl",
        match_id="fixture-provider-id",
        home_team="Nottingham Forest",
        away_team="Manchester City",
        kickoff_utc="2026-11-07T15:00:00Z",
        team_aliases=ALIASES,
    )

    assert match is nearest_match


def test_fixture_lookup_matches_laliga_alias_and_accented_team_names():
    prediction = saved_prediction(
        league="laliga",
        home_team="Celta Vigo",
        away_team="Real Betis",
        kickoff_utc="2026-11-08T00:00:00Z",
    )

    match = find_pregame_prediction(
        [prediction],
        league="laliga",
        match_id="fixture-provider-id",
        home_team="RC Celta de Vigo",
        away_team="Real Betis Balompié",
        kickoff_utc="2026-11-07T18:00:00Z",
        team_aliases={
            **ALIASES,
            "Real Betis Balompié": "Real Betis",
        },
    )

    assert match is prediction


def test_fixture_lookup_does_not_match_another_league_or_distant_kickoff():
    wrong_league = saved_prediction(league="laliga")
    distant = saved_prediction(kickoff_utc="2026-11-11T00:00:00Z")

    match = find_pregame_prediction(
        [wrong_league, distant],
        league="pl",
        match_id="fixture-provider-id",
        home_team="Nottingham Forest",
        away_team="Manchester City",
        kickoff_utc="2026-11-07T15:00:00Z",
        team_aliases=ALIASES,
    )

    assert match is None


def test_accuracy_stats_can_be_scoped_to_one_league(monkeypatch):
    predictions = [
        {
            "league": "pl",
            "actual_result": "H",
            "pre_correct": 1,
            "adj_correct": None,
        },
        {
            "league": "pl",
            "actual_result": None,
            "pre_correct": None,
            "adj_correct": None,
        },
        {
            "league": "laliga",
            "actual_result": "A",
            "pre_correct": 0,
            "adj_correct": None,
        },
    ]
    monkeypatch.setattr(
        "predictions.get_all_predictions",
        lambda season=None: predictions,
    )

    stats = get_accuracy_stats(season=2026, league="pl")

    assert stats["total_logged"] == 2
    assert stats["total_finished"] == 1
    assert stats["pre_total"] == 1
    assert stats["pre_accuracy"] == 100


def test_pregame_home_pick_is_correct_for_home_win():
    assert prediction_correct_for_score(60, 25, 15, 2, 1) is True


def test_pregame_draw_pick_is_correct_for_draw():
    assert prediction_correct_for_score(25, 50, 25, 1, 1) is True


def test_pregame_away_pick_is_marked_wrong_for_home_win():
    assert prediction_correct_for_score(20, 25, 55, 2, 1) is False


def test_missing_prediction_or_score_has_no_result():
    assert prediction_correct_for_score(None, None, None, 1, 0) is None
    assert prediction_correct_for_score(60, 25, 15, None, None) is None
