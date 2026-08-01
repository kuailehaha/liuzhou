"""Backend-neutral self-play statistics shared by V1 runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


def merge_mcts_metrics(
    rows: list[Dict[str, Any]], elapsed_sec: float
) -> Dict[str, Any]:
    rows = [dict(row) for row in rows if isinstance(row, dict) and row]
    if not rows:
        return {}
    sum_keys = {
        "leaf_evaluations",
        "inference_batches",
        "arena_overflow",
        "illegal_actions",
        "non_finite",
        "fallback",
        "graph_capture",
        "graph_replay",
    }
    merged: Dict[str, Any] = {}
    for key in sum_keys:
        merged[key] = int(sum(int(row.get(key, 0) or 0) for row in rows))
    merged["arena_peak_bytes"] = int(
        max(int(row.get("arena_peak_bytes", 0) or 0) for row in rows)
    )
    for key in ("precision", "search_backend"):
        values = sorted({str(row[key]) for row in rows if row.get(key) is not None})
        if values:
            merged[key] = values[0] if len(values) == 1 else ",".join(values)
    elapsed = max(1e-9, float(elapsed_sec))
    merged["leaf_evaluations_per_sec"] = float(
        merged["leaf_evaluations"] / elapsed
    )
    return merged


@dataclass
class SelfPlayV1Stats:
    num_games: int
    num_positions: int
    black_wins: int
    white_wins: int
    draws: int
    avg_game_length: float
    elapsed_sec: float
    positions_per_sec: float
    games_per_sec: float
    step_timing_ms: Dict[str, float]
    step_timing_ratio: Dict[str, float]
    step_timing_calls: Dict[str, int]
    mcts_counters: Dict[str, int]
    piece_delta_buckets: Dict[str, int]
    mcts_metrics: Dict[str, Any] = field(default_factory=dict)
    policy_target_audit: Dict[str, Any] = field(default_factory=dict)
    device: str = ""
    fallback_count: int = 0
    fallback_reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "num_games": float(self.num_games),
            "num_positions": float(self.num_positions),
            "black_wins": float(self.black_wins),
            "white_wins": float(self.white_wins),
            "draws": float(self.draws),
            "avg_game_length": float(self.avg_game_length),
            "elapsed_sec": float(self.elapsed_sec),
            "positions_per_sec": float(self.positions_per_sec),
            "games_per_sec": float(self.games_per_sec),
        }
        payload["step_timing_ms"] = {k: float(v) for k, v in self.step_timing_ms.items()}
        payload["step_timing_ratio"] = {k: float(v) for k, v in self.step_timing_ratio.items()}
        payload["step_timing_calls"] = {k: int(v) for k, v in self.step_timing_calls.items()}
        payload["mcts_counters"] = {k: int(v) for k, v in self.mcts_counters.items()}
        payload["mcts_metrics"] = dict(self.mcts_metrics or {})
        payload["piece_delta_buckets"] = {
            k: int(v) for k, v in self.piece_delta_buckets.items()
        }
        payload["policy_target_audit"] = dict(self.policy_target_audit or {})
        payload["device"] = str(self.device)
        payload["fallback_count"] = int(self.fallback_count)
        payload["fallback_reasons"] = list(self.fallback_reasons)
        return payload
