"""Backend-neutral self-play statistics shared by V1 runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


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
        payload["piece_delta_buckets"] = {
            k: int(v) for k, v in self.piece_delta_buckets.items()
        }
        payload["device"] = str(self.device)
        payload["fallback_count"] = int(self.fallback_count)
        payload["fallback_reasons"] = list(self.fallback_reasons)
        return payload
