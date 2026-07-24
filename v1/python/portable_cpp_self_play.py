"""V1 tensor self-play using threaded C++ search and PyTorch/MPS inference."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import torch

from src.game_state import GameState
from src.policy_batch import TOTAL_DIM

from .portable_cpp_mcts import PortableCppMCTS
from .portable_mcts import PortableMCTSConfig
from .policy_target_audit import PolicyTargetAudit
from .portable_self_play import PortableSelfPlayStats
from .trajectory_buffer import TensorSelfPlayBatch


def _soft_value_from_counts(black: int, white: int, soft_value_k: float) -> float:
    limit = math.pi * 0.5 - 1e-3
    scaled = max(
        -limit,
        min(limit, ((int(black) - int(white)) / 18.0) * float(soft_value_k)),
    )
    return max(-1.0, min(1.0, math.tan(scaled)))


def self_play_v1_portable_cpp(
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
    cpu_threads: int = 1,
    verbose: bool = False,
    policy_target_temperature: float | None = None,
    policy_target_prior_pseudocount: float = 0.0,
) -> Tuple[TensorSelfPlayBatch, PortableSelfPlayStats]:
    """Produce the existing V1 tensor payload with an opt-in C++ search core."""

    if int(num_games) <= 0:
        raise ValueError("num_games must be positive.")
    if int(concurrent_games) <= 0:
        raise ValueError("concurrent_games must be positive.")
    if int(cpu_threads) <= 0:
        raise ValueError("cpu_threads must be positive.")
    config = PortableMCTSConfig(
        num_simulations=max(1, int(mcts_simulations)),
        exploration_weight=float(exploration_weight),
        temperature=float(temperature_init),
        policy_target_temperature=(
            None
            if policy_target_temperature is None
            else float(policy_target_temperature)
        ),
        policy_target_prior_pseudocount=float(
            policy_target_prior_pseudocount
        ),
        add_dirichlet_noise=bool(add_dirichlet_noise),
        dirichlet_alpha=float(dirichlet_alpha),
        dirichlet_epsilon=float(dirichlet_epsilon),
        sample_moves=bool(sample_moves),
    )
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
    policy_target_audit = PolicyTargetAudit()
    timing_seconds: Dict[str, float] = {}
    timing_calls: Dict[str, int] = {}
    inference_batches = 0
    illegal_actions = 0
    non_finite = 0
    resolved_device = ""
    fallback_count = 0
    fallback_reasons: Tuple[str, ...] = ()
    started = time.perf_counter()

    for wave_base in range(0, int(num_games), wave_size):
        wave_games = min(wave_size, int(num_games) - wave_base)
        contexts = [
            {
                "steps": [],
                "plies": 0,
                "done": False,
                "finalized": False,
            }
            for _ in range(wave_games)
        ]
        search = PortableCppMCTS(
            model=model,
            config=config,
            device=device,
            num_threads=int(cpu_threads),
            initial_states=[GameState() for _ in range(wave_games)],
        )
        resolved_device = str(search.device)
        fallback_count += int(search.device_resolution.fallback_count)
        fallback_reasons = (
            *fallback_reasons,
            *search.device_resolution.fallback_reasons,
        )

        while any(not bool(ctx["done"]) for ctx in contexts):
            temperatures = [
                (
                    float(temperature_init)
                    if int(ctx["plies"]) < int(temperature_threshold)
                    else float(temperature_final)
                )
                for ctx in contexts
            ]
            force_uniform = [
                (
                    not bool(ctx["done"])
                    and int(ctx["plies"]) < int(opening_random_moves)
                )
                for ctx in contexts
            ]
            outputs = search.search_batch(
                temperatures=temperatures,
                add_dirichlet_noise=add_dirichlet_noise,
                force_uniform_random=force_uniform,
            )
            actions = [-1] * wave_games
            trajectory_started = time.perf_counter()
            for row, (ctx, output) in enumerate(zip(contexts, outputs)):
                if bool(ctx["done"]):
                    continue
                if output.terminal or output.chosen_action_index is None:
                    ctx["done"] = True
                    continue
                action_index = int(output.chosen_action_index)
                if not bool(output.legal_mask[action_index].item()):
                    raise RuntimeError(
                        f"Portable C++ selected illegal action {action_index}."
                    )
                policy_target_audit.observe(
                    output,
                    ply=int(ctx["plies"]),
                )
                ctx["steps"].append(
                    (
                        output.model_input.clone(),
                        output.legal_mask.clone(),
                        output.policy_dense.clone(),
                        int(search.last_current_players[row]),
                    )
                )
                actions[row] = action_index
                ctx["plies"] = int(ctx["plies"]) + 1
            search.advance_roots(actions)
            status = search.root_status()
            deactivate: List[int] = []
            for row, ctx in enumerate(contexts):
                if bool(ctx["done"]) and bool(ctx["finalized"]):
                    continue
                if (
                    bool(status["game_over"][row])
                    or int(ctx["plies"]) >= max_plies
                ):
                    ctx["done"] = True
                if not bool(ctx["done"]):
                    continue

                winner = int(status["winner"][row])
                result_black = 1.0 if winner == 1 else (-1.0 if winner == -1 else 0.0)
                black_pieces = int(status["black_pieces"][row])
                white_pieces = int(status["white_pieces"][row])
                soft_black = _soft_value_from_counts(
                    black_pieces, white_pieces, soft_value_k
                )
                if result_black > 0:
                    black_wins += 1
                elif result_black < 0:
                    white_wins += 1
                else:
                    draws += 1
                delta = max(-18, min(18, black_pieces - white_pieces))
                piece_delta_buckets[str(delta)] += 1
                lengths.append(int(ctx["plies"]))
                for model_input, legal_mask, policy, sign in ctx["steps"]:
                    state_rows.append(model_input)
                    legal_rows.append(legal_mask)
                    policy_rows.append(policy)
                    value_rows.append(float(result_black * sign))
                    soft_rows.append(float(soft_black * sign))
                ctx["finalized"] = True
                deactivate.append(row)
            if deactivate:
                search.deactivate(deactivate)
            search.record_timing(
                "trajectory_commit", time.perf_counter() - trajectory_started
            )
            if verbose:
                completed = sum(bool(ctx["done"]) for ctx in contexts)
                print(
                    f"[v1.portable.cpp] wave={wave_base // wave_size + 1} "
                    f"completed={completed}/{wave_games} "
                    f"threads={int(cpu_threads)}"
                )

        inference_batches += int(search.inference_batches)
        illegal_actions += int(search.illegal_action_count)
        non_finite += int(search.non_finite_count)
        for key, value in search.timing_seconds.items():
            timing_seconds[key] = timing_seconds.get(key, 0.0) + float(value)
        for key, value in search.timing_calls.items():
            timing_calls[key] = timing_calls.get(key, 0) + int(value)

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
            non_finite += 1
            raise ValueError(f"Portable C++ self-play produced NaN/Inf in {name}.")
    timing_seconds["payload_finalize"] = timing_seconds.get(
        "payload_finalize", 0.0
    ) + (time.perf_counter() - finalize_started)
    timing_calls["payload_finalize"] = timing_calls.get("payload_finalize", 0) + 1

    if illegal_actions != 0:
        raise RuntimeError(
            f"Portable C++ self-play observed {illegal_actions} illegal actions."
        )
    if non_finite != 0:
        raise RuntimeError(
            f"Portable C++ self-play observed {non_finite} non-finite values."
        )
    elapsed = max(1e-9, time.perf_counter() - started)
    step_timing_ms = {
        key: float(value * 1000.0) for key, value in timing_seconds.items()
    }
    step_timing_ratio = {
        key: min(1.0, max(0.0, float(value / elapsed)))
        for key, value in timing_seconds.items()
    }
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
        device=resolved_device,
        fallback_count=int(fallback_count),
        fallback_reasons=tuple(fallback_reasons),
        inference_batches=int(inference_batches),
        step_timing_ms=step_timing_ms,
        step_timing_ratio=step_timing_ratio,
        step_timing_calls=timing_calls,
        mcts_counters={
            "portable_inference_batches": int(inference_batches),
            "portable_virtual_loss_count": 0,
            "portable_fallback_count": int(fallback_count),
            "portable_cpp_threads": int(cpu_threads),
            "portable_cpp_illegal_actions": int(illegal_actions),
            "portable_cpp_non_finite": int(non_finite),
        },
        piece_delta_buckets=piece_delta_buckets,
        policy_target_audit=policy_target_audit.to_dict(),
    )
    return samples, stats
