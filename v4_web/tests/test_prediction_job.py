from unittest.mock import Mock, patch

import prediction_job


def test_fd_prediction_logging_includes_ligue1_with_configured_prior():
    fixture_fetcher = Mock(
        side_effect=lambda competition, max_fixtures: (
            [{
                "match_id": "fl1-123",
                "home": "Lyon",
                "away": "Marseille",
                "kickoff_utc": "2026-11-28T19:00:00Z",
            }]
            if competition == "FL1"
            else []
        )
    )
    logger = Mock()
    dc_probs = Mock(return_value=(45.0, 25.0, 30.0, 1.2, 0.9, 1.0, 1.0))

    with patch.object(prediction_job, "_dc_probs", dc_probs):
        prediction_job._log_fd_predictions({}, 2026, fixture_fetcher, logger)

    assert fixture_fetcher.call_args_list[-1].args == ("FL1",)
    assert dc_probs.call_args.args[-1] == "FRA-Ligue 1"
    assert logger.call_args.kwargs["league"] == "ligue1"
    assert logger.call_args.kwargs["match_id"] == "fl1-123"
