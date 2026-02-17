#!/usr/bin/env python
"""A/B semantic regression for v1 child value-only evaluation.

Compares:
1) Root search output consistency: chosen action + root value diff.
2) Full self-play consistency: W/L/D and tensor trajectory parity.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Tuple

import torch

import v0_core
from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.mcts_gpu import (
    GpuStateBatch,
    V1RootMCTS,
    V1RootMCTSConfig,
    batch_apply_moves_compat,
    encode_actions_fast,
)
from v1.python.self_play_gpu_runner import SelfPlayV1Stats, self_play_v1_gpu


def _ensure_v0_binary_compat() -> None:
    if hasattr(v0_core.MCTSConfig, "max_actions_per_batch"):
        return
    v0_core.MCTSConfig.max_actions_per_batch = property(  # type: ignore[attr-defined]
        lambda self: 0,
        lambda self, value: None,
    )


def _set_seed(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _index_copy_state(dst: GpuStateBatch, idx: torch.Tensor, src: GpuStateBatch) -> None:
    dst.board.index_copy_(0, idx, src.board)
    dst.marks_black.index_copy_(0, idx, src.marks_black)
    dst.marks_white.index_copy_(0, idx, src.marks_white)
    dst.phase.index_copy_(0, idx, src.phase)
    dst.current_player.index_copy_(0, idx, src.current_player)
    dst.pending_marks_required.index_copy_(0, idx, src.pending_marks_required)
    dst.pending_marks_remaining.index_copy_(0, idx, src.pending_marks_remaining)
    dst.pending_captures_required.index_copy_(0, idx, src.pending_captures_required)
    dst.pending_captures_remaining.index_copy_(0, idx, src.pending_captures_remaining)
    dst.forced_removals_done.index_copy_(0, idx, src.forced_removals_done)
    dst.move_count.index_copy_(0, idx, src.move_count)
    dst.moves_since_capture.index_copy_(0, idx, src.moves_since_capture)


def _sample_valid_states(
    *,
    device: torch.device,
    num_states: int,
    plies: int,
    seed: int,
) -> GpuStateBatch:
    """Generate valid states by random legal playout on GPU tensors."""

    if num_states <= 0:
        raise ValueError("num_states must be positive.")
    states = GpuStateBatch.initial(device=device, batch_size=int(num_states))
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(int(seed))
    for _ in range(max(0, int(plies))):
        legal_mask, metadata = encode_actions_fast(states)
        legal_mask_bool = legal_mask.to(torch.bool)
        picked_actions = []
        picked_parents = []
        for i in range(int(num_states)):
            legal_idx = torch.nonzero(legal_mask_bool[i], as_tuple=False).view(-1)
            if int(legal_idx.numel()) == 0:
                continue
            sampled_local = int(
                torch.randint(
                    low=0,
                    high=int(legal_idx.numel()),
                    size=(1,),
                    generator=cpu_gen,
                ).item()
            )
            action_index = int(legal_idx[sampled_local].item())
            picked_actions.append(metadata[i, action_index].to(torch.int32))
            picked_parents.append(i)
        if not picked_actions:
            break
        actions = torch.stack(picked_actions, dim=0).to(device=device, dtype=torch.int32)
        parents = torch.tensor(picked_parents, dtype=torch.int64, device=device)
        next_partial = batch_apply_moves_compat(states, actions, parents)
        _index_copy_state(states, parents, next_partial)
    return states


@dataclass
class RootABMetrics:
    num_states: int
    valid_rows_full: int
    valid_rows_value_only: int
    action_match_ratio: float
    root_value_mean_abs_diff: float
    root_value_max_abs_diff: float


def _run_root_ab(
    *,
    model: ChessNet,
    device: torch.device,
    states: GpuStateBatch,
    mcts_simulations: int,
    exploration_weight: float,
) -> RootABMetrics:
    cfg_common = dict(
        num_simulations=max(1, int(mcts_simulations)),
        exploration_weight=float(exploration_weight),
        temperature=1.0,
        add_dirichlet_noise=False,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        sample_moves=False,
        autocast_dtype="float16",
    )
    mcts_full = V1RootMCTS(
        model=model,
        config=V1RootMCTSConfig(**cfg_common, child_eval_mode="full"),
        device=device,
    )
    mcts_value_only = V1RootMCTS(
        model=model,
        config=V1RootMCTSConfig(**cfg_common, child_eval_mode="value_only"),
        device=device,
    )
    out_full = mcts_full.search_batch(states, temperatures=1.0, add_dirichlet_noise=False)
    out_value_only = mcts_value_only.search_batch(states, temperatures=1.0, add_dirichlet_noise=False)

    valid_full = out_full.chosen_valid_mask.to(torch.bool)
    valid_value = out_value_only.chosen_valid_mask.to(torch.bool)
    both_invalid = (~valid_full) & (~valid_value)
    both_valid = valid_full & valid_value
    same_action = out_full.chosen_action_indices.eq(out_value_only.chosen_action_indices)
    action_match = (both_invalid | (both_valid & same_action)).to(torch.float32)
    value_abs = (out_full.root_value - out_value_only.root_value).abs().to(torch.float32)

    return RootABMetrics(
        num_states=int(states.batch_size),
        valid_rows_full=int(valid_full.sum().item()),
        valid_rows_value_only=int(valid_value.sum().item()),
        action_match_ratio=float(action_match.mean().item()),
        root_value_mean_abs_diff=float(value_abs.mean().item()),
        root_value_max_abs_diff=float(value_abs.max().item()),
    )


@dataclass
class SelfPlayABMetrics:
    stats_full: Dict[str, float]
    stats_value_only: Dict[str, float]
    wld_match: bool
    num_samples_match: bool
    value_targets_mean_abs_diff: float
    value_targets_max_abs_diff: float
    policy_targets_mean_abs_diff: float
    policy_targets_max_abs_diff: float


def _run_self_play_once(
    *,
    model: ChessNet,
    device: str,
    num_games: int,
    mcts_simulations: int,
    concurrent_games: int,
    child_eval_mode: str,
    seed: int,
) -> Tuple[SelfPlayV1Stats, torch.Tensor, torch.Tensor]:
    _set_seed(seed)
    batch, stats = self_play_v1_gpu(
        model=model,
        num_games=int(num_games),
        mcts_simulations=int(mcts_simulations),
        temperature_init=1.0,
        temperature_final=0.2,
        temperature_threshold=8,
        exploration_weight=1.0,
        device=str(device),
        add_dirichlet_noise=False,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        soft_value_k=2.0,
        max_game_plies=512,
        sample_moves=False,
        concurrent_games=int(concurrent_games),
        child_eval_mode=str(child_eval_mode),
        inference_engine=None,
        verbose=False,
    )
    return stats, batch.value_targets.detach().clone(), batch.policy_targets.detach().clone()


def _run_self_play_ab(
    *,
    model: ChessNet,
    device: str,
    num_games: int,
    mcts_simulations: int,
    concurrent_games: int,
    seed: int,
) -> SelfPlayABMetrics:
    stats_full, values_full, policy_full = _run_self_play_once(
        model=model,
        device=device,
        num_games=num_games,
        mcts_simulations=mcts_simulations,
        concurrent_games=concurrent_games,
        child_eval_mode="full",
        seed=seed,
    )
    stats_value, values_value, policy_value = _run_self_play_once(
        model=model,
        device=device,
        num_games=num_games,
        mcts_simulations=mcts_simulations,
        concurrent_games=concurrent_games,
        child_eval_mode="value_only",
        seed=seed,
    )
    stats_full_dict = stats_full.to_dict()
    stats_value_dict = stats_value.to_dict()

    wld_match = (
        int(stats_full.black_wins) == int(stats_value.black_wins)
        and int(stats_full.white_wins) == int(stats_value.white_wins)
        and int(stats_full.draws) == int(stats_value.draws)
        and int(stats_full.num_games) == int(stats_value.num_games)
    )
    num_samples_match = int(values_full.numel()) == int(values_value.numel())

    if num_samples_match:
        value_abs = (values_full - values_value).abs().to(torch.float32)
        policy_abs = (policy_full - policy_value).abs().to(torch.float32)
        value_mean = float(value_abs.mean().item()) if int(value_abs.numel()) > 0 else 0.0
        value_max = float(value_abs.max().item()) if int(value_abs.numel()) > 0 else 0.0
        policy_mean = float(policy_abs.mean().item()) if int(policy_abs.numel()) > 0 else 0.0
        policy_max = float(policy_abs.max().item()) if int(policy_abs.numel()) > 0 else 0.0
    else:
        value_mean = float("inf")
        value_max = float("inf")
        policy_mean = float("inf")
        policy_max = float("inf")

    return SelfPlayABMetrics(
        stats_full=stats_full_dict,
        stats_value_only=stats_value_dict,
        wld_match=bool(wld_match),
        num_samples_match=bool(num_samples_match),
        value_targets_mean_abs_diff=value_mean,
        value_targets_max_abs_diff=value_max,
        policy_targets_mean_abs_diff=policy_mean,
        policy_targets_max_abs_diff=policy_max,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A/B semantic regression for v1 child value-only eval.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-states", type=int, default=32)
    parser.add_argument("--state-plies", type=int, default=8)
    parser.add_argument("--mcts-simulations", type=int, default=128)
    parser.add_argument("--exploration-weight", type=float, default=1.0)
    parser.add_argument("--self-play-games", type=int, default=8)
    parser.add_argument("--self-play-concurrent-games", type=int, default=8)
    parser.add_argument("--max-root-value-mean-abs-diff", type=float, default=1e-6)
    parser.add_argument("--max-root-value-max-abs-diff", type=float, default=1e-5)
    parser.add_argument("--min-action-match-ratio", type=float, default=1.0)
    parser.add_argument("--max-self-play-value-mean-abs-diff", type=float, default=1e-6)
    parser.add_argument("--max-self-play-value-max-abs-diff", type=float, default=1e-5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join("results", f"v1_child_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    _ensure_v0_binary_compat()

    device = torch.device(args.device)
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()

    _set_seed(int(args.seed))
    sampled_states = _sample_valid_states(
        device=device,
        num_states=int(args.num_states),
        plies=int(args.state_plies),
        seed=int(args.seed) + 17,
    )
    root_metrics = _run_root_ab(
        model=model,
        device=device,
        states=sampled_states,
        mcts_simulations=int(args.mcts_simulations),
        exploration_weight=float(args.exploration_weight),
    )
    self_play_metrics = _run_self_play_ab(
        model=model,
        device=args.device,
        num_games=int(args.self_play_games),
        mcts_simulations=int(args.mcts_simulations),
        concurrent_games=int(args.self_play_concurrent_games),
        seed=int(args.seed) + 33,
    )

    criteria = {
        "root_action_match_ratio_ge_threshold": bool(
            root_metrics.action_match_ratio >= float(args.min_action_match_ratio)
        ),
        "root_value_mean_abs_diff_le_threshold": bool(
            root_metrics.root_value_mean_abs_diff <= float(args.max_root_value_mean_abs_diff)
        ),
        "root_value_max_abs_diff_le_threshold": bool(
            root_metrics.root_value_max_abs_diff <= float(args.max_root_value_max_abs_diff)
        ),
        "self_play_wld_match": bool(self_play_metrics.wld_match),
        "self_play_num_samples_match": bool(self_play_metrics.num_samples_match),
        "self_play_value_mean_abs_diff_le_threshold": bool(
            self_play_metrics.value_targets_mean_abs_diff
            <= float(args.max_self_play_value_mean_abs_diff)
        ),
        "self_play_value_max_abs_diff_le_threshold": bool(
            self_play_metrics.value_targets_max_abs_diff
            <= float(args.max_self_play_value_max_abs_diff)
        ),
    }

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(args.device),
        "seed": int(args.seed),
        "config": {
            "num_states": int(args.num_states),
            "state_plies": int(args.state_plies),
            "mcts_simulations": int(args.mcts_simulations),
            "exploration_weight": float(args.exploration_weight),
            "self_play_games": int(args.self_play_games),
            "self_play_concurrent_games": int(args.self_play_concurrent_games),
            "thresholds": {
                "min_action_match_ratio": float(args.min_action_match_ratio),
                "max_root_value_mean_abs_diff": float(args.max_root_value_mean_abs_diff),
                "max_root_value_max_abs_diff": float(args.max_root_value_max_abs_diff),
                "max_self_play_value_mean_abs_diff": float(args.max_self_play_value_mean_abs_diff),
                "max_self_play_value_max_abs_diff": float(args.max_self_play_value_max_abs_diff),
            },
        },
        "root_ab": asdict(root_metrics),
        "self_play_ab": asdict(self_play_metrics),
        "criteria": criteria,
    }

    out_path = str(args.output_json)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(
        "[root_ab] "
        f"action_match={root_metrics.action_match_ratio:.4f} "
        f"value_mean_abs_diff={root_metrics.root_value_mean_abs_diff:.6g} "
        f"value_max_abs_diff={root_metrics.root_value_max_abs_diff:.6g}"
    )
    print(
        "[self_play_ab] "
        f"wld_match={self_play_metrics.wld_match} "
        f"num_samples_match={self_play_metrics.num_samples_match} "
        f"value_mean_abs_diff={self_play_metrics.value_targets_mean_abs_diff:.6g} "
        f"value_max_abs_diff={self_play_metrics.value_targets_max_abs_diff:.6g}"
    )
    print("[criteria]")
    for key, passed in criteria.items():
        print(f"{key}: {'PASS' if passed else 'FAIL'}")
    print(f"report_saved={out_path}")

    if args.strict and not all(criteria.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
