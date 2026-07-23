"""Single-process portable V1 self-play using CPU rules and batched PyTorch."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from src.game_state import GameState, Player
from src.policy_batch import TOTAL_DIM

from .portable_mcts import PortableMCTS, PortableMCTSConfig, PortableTree
from .trajectory_buffer import TensorSelfPlayBatch


@dataclass
class PortableSelfPlayStats:
    num_games: int
    num_positions: int
    black_wins: int
    white_wins: int
    draws: int
    avg_game_length: float
    elapsed_sec: float
    positions_per_sec: float
    games_per_sec: float
    device: str
    fallback_count: int
    fallback_reasons: Tuple[str, ...]
    inference_batches: int
    step_timing_ms: Dict[str, float]
    step_timing_ratio: Dict[str, float]
    step_timing_calls: Dict[str, int]
    mcts_counters: Dict[str, int]
    piece_delta_buckets: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_games": float(self.num_games),
            "num_positions": float(self.num_positions),
            "black_wins": float(self.black_wins),
            "white_wins": float(self.white_wins),
            "draws": float(self.draws),
            "avg_game_length": float(self.avg_game_length),
            "elapsed_sec": float(self.elapsed_sec),
            "positions_per_sec": float(self.positions_per_sec),
            "games_per_sec": float(self.games_per_sec),
            "device": self.device,
            "fallback_count": int(self.fallback_count),
            "fallback_reasons": list(self.fallback_reasons),
            "inference_batches": int(self.inference_batches),
            "step_timing_ms": dict(self.step_timing_ms),
            "step_timing_ratio": dict(self.step_timing_ratio),
            "step_timing_calls": dict(self.step_timing_calls),
            "mcts_counters": dict(self.mcts_counters),
            "piece_delta_buckets": dict(self.piece_delta_buckets),
        }


def _soft_value_from_black(state: GameState, soft_value_k: float) -> float:
    black = state.count_player_pieces(Player.BLACK)
    white = state.count_player_pieces(Player.WHITE)
    scaled = max(-(math.pi * 0.5 - 1e-3), min(math.pi * 0.5 - 1e-3, ((black - white) / 18.0) * float(soft_value_k)))
    return max(-1.0, min(1.0, math.tan(scaled)))


def _result_from_black(state: GameState) -> float:
    winner = state.get_winner()
    if winner == Player.BLACK:
        return 1.0
    if winner == Player.WHITE:
        return -1.0
    return 0.0


def self_play_v1_portable(
    model,
    num_games: int,
    mcts_simulations: int,
    temperature_init: float,
    temperature_final: float,
    temperature_threshold: int,
    exploration_weight: float,
    device: str = "auto",
    add_dirichlet_noise: bool = True,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    soft_value_k: float = 2.0,
    opening_random_moves: int = 0,
    max_game_plies: int = 512,
    sample_moves: bool = True,
    concurrent_games: int = 8,
    verbose: bool = False,
) -> Tuple[TensorSelfPlayBatch, PortableSelfPlayStats]:
    """Produce the existing tensor payload contract without ``v0_core``."""

    if int(num_games) <= 0:
        raise ValueError("num_games must be positive.")
    if int(concurrent_games) <= 0:
        raise ValueError("concurrent_games must be positive.")
    config = PortableMCTSConfig(
        num_simulations=max(1, int(mcts_simulations)),
        exploration_weight=float(exploration_weight),
        temperature=float(temperature_init),
        add_dirichlet_noise=bool(add_dirichlet_noise),
        dirichlet_alpha=float(dirichlet_alpha),
        dirichlet_epsilon=float(dirichlet_epsilon),
        sample_moves=bool(sample_moves),
    )
    search = PortableMCTS(model=model, config=config, device=device)
    max_plies = max(1, int(max_game_plies))
    wave_size = max(1, min(int(concurrent_games), int(num_games)))

    state_rows: List[torch.Tensor] = []
    legal_rows: List[torch.Tensor] = []
    policy_rows: List[torch.Tensor] = []
    value_rows: List[float] = []
    soft_rows: List[float] = []
    black_wins = 0
    white_wins = 0
    draws = 0
    lengths: List[int] = []
    piece_delta_buckets = {str(delta): 0 for delta in range(-18, 19)}
    started = time.perf_counter()

    for wave_base in range(0, int(num_games), wave_size):
        wave_games = min(wave_size, int(num_games) - wave_base)
        contexts = [
            {
                "tree": PortableTree(GameState()),
                "steps": [],
                "plies": 0,
                "done": False,
            }
            for _ in range(wave_games)
        ]
        while any(not bool(ctx["done"]) for ctx in contexts):
            active = [ctx for ctx in contexts if not bool(ctx["done"])]
            trees = [ctx["tree"] for ctx in active]
            temperatures = [
                float(temperature_init)
                if int(ctx["plies"]) < int(temperature_threshold)
                else float(temperature_final)
                for ctx in active
            ]
            force_uniform = [
                int(ctx["plies"]) < int(opening_random_moves) for ctx in active
            ]
            outputs = search.search_batch(
                trees,
                temperatures=temperatures,
                add_dirichlet_noise=add_dirichlet_noise,
                force_uniform_random=force_uniform,
            )
            trajectory_started = time.perf_counter()
            for ctx, output in zip(active, outputs):
                tree: PortableTree = ctx["tree"]
                state = tree.root.state
                if output.terminal or output.chosen_action_index is None or output.chosen_move is None:
                    ctx["done"] = True
                else:
                    ctx["steps"].append(
                        (
                            output.model_input.clone(),
                            output.legal_mask.clone(),
                            output.policy_dense.clone(),
                            int(state.current_player.value),
                        )
                    )
                    if not tree.advance_root(int(output.chosen_action_index)):
                        raise RuntimeError(
                            f"Portable subtree reuse failed for action {output.chosen_action_index}."
                        )
                    ctx["plies"] = int(ctx["plies"]) + 1
                    if tree.root.state.is_game_over() or int(ctx["plies"]) >= max_plies:
                        ctx["done"] = True

                if bool(ctx["done"]):
                    final_state = tree.root.state
                    result_black = _result_from_black(final_state)
                    soft_black = _soft_value_from_black(final_state, soft_value_k)
                    if result_black > 0:
                        black_wins += 1
                    elif result_black < 0:
                        white_wins += 1
                    else:
                        draws += 1
                    delta = final_state.count_player_pieces(Player.BLACK) - final_state.count_player_pieces(Player.WHITE)
                    piece_delta_buckets[str(max(-18, min(18, int(delta))))] += 1
                    lengths.append(int(ctx["plies"]))
                    for model_input, legal_mask, policy, sign in ctx["steps"]:
                        state_rows.append(model_input)
                        legal_rows.append(legal_mask)
                        policy_rows.append(policy)
                        value_rows.append(float(result_black * sign))
                        soft_rows.append(float(soft_black * sign))
            search.record_timing(
                "trajectory_commit", time.perf_counter() - trajectory_started
            )
            if verbose:
                completed = sum(bool(ctx["done"]) for ctx in contexts)
                print(
                    f"[v1.portable] wave={wave_base // wave_size + 1} "
                    f"completed={completed}/{wave_games}"
                )

    finalize_started = time.perf_counter()
    if state_rows:
        samples = TensorSelfPlayBatch(
            state_tensors=torch.stack(state_rows).to(torch.float32),
            legal_masks=torch.stack(legal_rows).to(torch.bool),
            policy_targets=torch.stack(policy_rows).to(torch.float32),
            value_targets=torch.tensor(value_rows, dtype=torch.float32),
            soft_value_targets=torch.tensor(soft_rows, dtype=torch.float32),
        )
    else:
        samples = TensorSelfPlayBatch(
            state_tensors=torch.empty((0, 11, 6, 6), dtype=torch.float32),
            legal_masks=torch.empty((0, TOTAL_DIM), dtype=torch.bool),
            policy_targets=torch.empty((0, TOTAL_DIM), dtype=torch.float32),
            value_targets=torch.empty((0,), dtype=torch.float32),
            soft_value_targets=torch.empty((0,), dtype=torch.float32),
        )
    for name, tensor in (
        ("state_tensors", samples.state_tensors),
        ("policy_targets", samples.policy_targets),
        ("value_targets", samples.value_targets),
        ("soft_value_targets", samples.soft_value_targets),
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Portable self-play produced NaN/Inf in {name}.")
    search.record_timing("payload_finalize", time.perf_counter() - finalize_started)

    elapsed = max(1e-9, time.perf_counter() - started)
    step_timing_ms, step_timing_ratio, step_timing_calls = search.timing_snapshot(
        elapsed
    )
    stats = PortableSelfPlayStats(
        num_games=int(num_games),
        num_positions=samples.num_samples,
        black_wins=black_wins,
        white_wins=white_wins,
        draws=draws,
        avg_game_length=(float(sum(lengths) / len(lengths)) if lengths else 0.0),
        elapsed_sec=elapsed,
        positions_per_sec=float(samples.num_samples / elapsed),
        games_per_sec=float(num_games / elapsed),
        device=str(search.device),
        fallback_count=int(search.device_resolution.fallback_count),
        fallback_reasons=search.device_resolution.fallback_reasons,
        inference_batches=int(search.inference_batches),
        step_timing_ms=step_timing_ms,
        step_timing_ratio=step_timing_ratio,
        step_timing_calls=step_timing_calls,
        mcts_counters={
            "portable_inference_batches": int(search.inference_batches),
            "portable_virtual_loss_count": 0,
            "portable_fallback_count": int(search.device_resolution.fallback_count),
        },
        piece_delta_buckets=piece_delta_buckets,
    )
    return samples, stats
