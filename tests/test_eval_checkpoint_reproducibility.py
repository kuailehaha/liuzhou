from __future__ import annotations

from scripts.eval_checkpoint import (
    _aggregate_v1_worker_results,
    _stats_to_payload,
    build_parser,
)


def test_eval_parser_accepts_explicit_seed() -> None:
    args = build_parser().parse_args(
        ["--challenger_checkpoint", "candidate.pt", "--seed", "20260722"]
    )
    assert args.seed == 20260722


def test_eval_parser_accepts_multiple_portable_workers() -> None:
    args = build_parser().parse_args(
        ["--challenger_checkpoint", "candidate.pt", "--backend", "portable", "--eval_workers", "2"]
    )
    assert args.backend == "portable"
    assert args.eval_workers == 2


def test_worker_aggregation_preserves_color_breakdown_and_seed() -> None:
    stats = _aggregate_v1_worker_results(
        [
            (3, 1, 2, 2, 1, 0, 1, 0, 2),
            (4, 0, 2, 1, 0, 2, 3, 0, 0),
        ],
        total_games=12,
        seed=91,
    )

    assert (stats.wins, stats.losses, stats.draws, stats.total_games) == (7, 1, 4, 12)
    assert stats.seed == 91
    assert stats.color_breakdown == {
        "challenger_black": {"wins": 3, "losses": 1, "draws": 2, "games": 6},
        "challenger_white": {"wins": 4, "losses": 0, "draws": 2, "games": 6},
    }

    payload = _stats_to_payload("vs_random", stats)
    assert payload["seed"] == 91
    assert payload["color_breakdown"] == stats.color_breakdown


def test_worker_aggregation_rejects_inconsistent_totals() -> None:
    try:
        _aggregate_v1_worker_results(
            [(1, 0, 0, 1, 0, 0, 0, 0, 0)],
            total_games=2,
            seed=7,
        )
    except ValueError as exc:
        assert "expected 2 games" in str(exc)
    else:
        raise AssertionError("inconsistent worker totals must fail")


def test_worker_aggregation_rejects_unbalanced_challenger_colors() -> None:
    try:
        _aggregate_v1_worker_results(
            [(2, 0, 2, 2, 0, 1, 0, 0, 1)],
            total_games=4,
            seed=7,
        )
    except ValueError as exc:
        assert "250/250-style color balance" in str(exc)
    else:
        raise AssertionError("an even evaluation must split challenger colors equally")
