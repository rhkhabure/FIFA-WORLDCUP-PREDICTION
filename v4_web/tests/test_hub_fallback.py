import main


def test_hub_uses_stored_prediction_fixture_when_provider_is_unavailable():
    predictions = [
        {
            "match_id": "fl1-1",
            "league": "ligue1",
            "home_team": "Lyon",
            "away_team": "Marseille",
            "kickoff_utc": "2026-11-28T19:00:00Z",
            "actual_result": None,
            "actual_hg": None,
            "actual_ag": None,
        },
        {
            "match_id": "pl-1",
            "league": "pl",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "kickoff_utc": "2026-11-28T15:00:00Z",
            "actual_result": None,
        },
    ]

    fixtures = main._stored_fixture_fallback(
        predictions, "ligue1", "2026-11-27", "2026-12-02"
    )

    assert len(fixtures) == 1
    assert fixtures[0]["match_id"] == "fl1-1"
    assert fixtures[0]["status"] == "Scheduled"
    assert fixtures[0]["home_score"] is None
