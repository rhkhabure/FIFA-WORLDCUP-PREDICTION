import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictions import prediction_correct_for_score


def test_pregame_home_pick_is_correct_for_home_win():
    assert prediction_correct_for_score(60, 25, 15, 2, 1) is True


def test_pregame_draw_pick_is_correct_for_draw():
    assert prediction_correct_for_score(25, 50, 25, 1, 1) is True


def test_pregame_away_pick_is_marked_wrong_for_home_win():
    assert prediction_correct_for_score(20, 25, 55, 2, 1) is False


def test_missing_prediction_or_score_has_no_result():
    assert prediction_correct_for_score(None, None, None, 1, 0) is None
    assert prediction_correct_for_score(60, 25, 15, None, None) is None
