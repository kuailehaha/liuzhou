from __future__ import annotations

import pytest

from tools.benchmark_portable_eval_budget import summarize_reports


def _report(seed: int, wins: int, losses: int, draws: int, elapsed: float) -> dict:
    games = wins + losses + draws
    return {
        "elapsed_sec": elapsed,
        "device_resolution": {"fallback_count": 0, "fallback_reasons": []},
        "results": [
            {
                "seed": seed,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "total_games": games,
                "color_breakdown": {
                    "challenger_black": {"games": games // 2},
                    "challenger_white": {"games": games // 2},
                },
            }
        ],
    }


def test_summarize_reports_tracks_score_variance_and_elapsed_time() -> None:
    summary = summarize_reports(
        [_report(1, 200, 150, 150, 250.0), _report(2, 215, 155, 130, 260.0)]
    )

    assert summary["games_per_repeat"] == 500
    assert summary["repeats"] == 2
    assert summary["score_mean"] == pytest.approx(0.555)
    assert summary["score_between_seed_sd"] == pytest.approx(0.0070710678)
    assert summary["elapsed_total_sec"] == pytest.approx(510.0)


def test_summarize_reports_rejects_fallback_and_color_imbalance() -> None:
    fallback = _report(1, 200, 150, 150, 250.0)
    fallback["device_resolution"]["fallback_count"] = 1
    with pytest.raises(ValueError, match="fallback"):
        summarize_reports([fallback])

    imbalance = _report(1, 200, 150, 150, 250.0)
    imbalance["results"][0]["color_breakdown"]["challenger_black"]["games"] = 251
    with pytest.raises(ValueError, match="evenly split"):
        summarize_reports([imbalance])
