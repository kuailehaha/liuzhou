from __future__ import annotations

import pytest

from tools.eval_portable_league import analyze_league_results


def _match(left: str, right: str, wins: int, losses: int, draws: int) -> dict:
    return {
        "challenger": left,
        "opponent": right,
        "result": {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": wins + losses + draws,
        },
    }


def test_league_analysis_reports_worst_case_ratings_and_cycles() -> None:
    report = analyze_league_results(
        ["a", "b", "c"],
        [
            _match("a", "b", 60, 40, 0),
            _match("b", "c", 60, 40, 0),
            _match("a", "c", 40, 60, 0),
        ],
    )
    assert report["score_matrix"]["a"]["b"] == pytest.approx(0.6)
    assert report["score_matrix"]["b"]["a"] == pytest.approx(0.4)
    assert report["worst_case_score"]["a"] == pytest.approx(0.4)
    assert report["directed_three_cycle_count"] == 1
    assert set(report["elo_centered"]) == {"a", "b", "c"}
