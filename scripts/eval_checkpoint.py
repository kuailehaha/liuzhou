#!/usr/bin/env python
"""Evaluate a checkpoint against RandomAgent and/or a previous checkpoint."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
_EXTRA_PATHS = [
    os.path.join(ROOT_DIR, "build", "v0", "src"),
    os.path.join(ROOT_DIR, "v0", "build", "src"),
]
for _p in _EXTRA_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from src.evaluate import (  # noqa: E402
    EvaluationStats,
    evaluate_against_agent_parallel,
    evaluate_against_agent_parallel_v0,
)
from src.game_state import GameState, Player  # noqa: E402
from src.move_generator import apply_move, generate_all_legal_moves  # noqa: E402
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS  # noqa: E402
from src.random_agent import RandomAgent  # noqa: E402
BOARD_SIZE = int(GameState.BOARD_SIZE)

V1WorkerResult = Tuple[int, int, int, int, int, int, int, int, int]


def _parse_devices(device: str, eval_devices: Optional[str]) -> List[str]:
    raw = str(eval_devices).strip() if eval_devices is not None else ""
    items = [x.strip() for x in raw.split(",") if x.strip()] if raw else [str(device).strip()]
    return items or [str(device).strip()]


def _normalize_eval_games(num_games: int) -> int:
    n = max(0, int(num_games))
    if n == 0:
        return 0
    if n % 2 != 0:
        n = max(2, (n // 2) * 2)
    return n


def _split_game_indices(num_games: int, num_workers: int) -> List[List[int]]:
    if num_workers <= 0:
        return []
    buckets: List[List[int]] = [[] for _ in range(num_workers)]
    for idx in range(num_games):
        buckets[idx % num_workers].append(idx)
    return [bucket for bucket in buckets if bucket]


def _seed_worker(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _aggregate_v1_worker_results(
    results: Sequence[V1WorkerResult],
    *,
    total_games: int,
    seed: int,
) -> EvaluationStats:
    totals = [sum(int(row[idx]) for row in results) for idx in range(9)]
    wins, losses, draws = totals[:3]
    black_wins, black_losses, black_draws = totals[3:6]
    white_wins, white_losses, white_draws = totals[6:9]
    observed_games = wins + losses + draws
    if observed_games != int(total_games):
        raise ValueError(
            f"Portable evaluation produced {observed_games} outcomes; "
            f"expected {int(total_games)} games."
        )
    if (
        black_wins + white_wins != wins
        or black_losses + white_losses != losses
        or black_draws + white_draws != draws
    ):
        raise ValueError("Portable evaluation color totals do not match aggregate outcomes.")
    black_games = black_wins + black_losses + black_draws
    white_games = white_wins + white_losses + white_draws
    if int(total_games) % 2 != 0 or black_games != white_games:
        raise ValueError(
            "Portable evaluation requires 250/250-style color balance for an even game count; "
            f"got challenger_black={black_games} challenger_white={white_games}."
        )

    stats = EvaluationStats(
        wins=wins,
        losses=losses,
        draws=draws,
        total_games=int(total_games),
    )
    stats.seed = int(seed)
    stats.color_breakdown = {
        "challenger_black": {
            "wins": black_wins,
            "losses": black_losses,
            "draws": black_draws,
            "games": black_games,
        },
        "challenger_white": {
            "wins": white_wins,
            "losses": white_losses,
            "draws": white_draws,
            "games": white_games,
        },
    }
    return stats


def _load_model_from_checkpoint(checkpoint_path: str, device: str) -> ChessNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    board_size = checkpoint.get("board_size", GameState.BOARD_SIZE)
    num_inputs = checkpoint.get("num_input_channels", NUM_INPUT_CHANNELS)
    model = ChessNet(board_size=int(board_size), num_input_channels=int(num_inputs))
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def _stats_to_payload(name: str, stats: EvaluationStats) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": str(name),
        "wins": int(stats.wins),
        "losses": int(stats.losses),
        "draws": int(stats.draws),
        "total_games": int(stats.total_games),
        "win_rate": float(stats.win_rate),
        "loss_rate": float(stats.loss_rate),
        "draw_rate": float(stats.draw_rate),
    }
    if hasattr(stats, "seed"):
        payload["seed"] = int(stats.seed)
    if hasattr(stats, "color_breakdown"):
        payload["color_breakdown"] = dict(stats.color_breakdown)
    return payload


def _print_stats_line(title: str, payload: Dict[str, Any]) -> None:
    print(
        f"[eval] {title}: "
        f"W-L-D={int(payload['wins'])}-{int(payload['losses'])}-{int(payload['draws'])} "
        f"({int(payload['total_games'])} games), "
        f"win={float(payload['win_rate']) * 100.0:.2f}% "
        f"loss={float(payload['loss_rate']) * 100.0:.2f}% "
        f"draw={float(payload['draw_rate']) * 100.0:.2f}%"
    )
    total = int(payload["total_games"])
    draw_rate = float(payload["draw_rate"])
    if total >= 100 and draw_rate >= 0.95:
        print(
            "[eval] note: very high draw rate. "
            "Try EVAL_SAMPLE_MOVES=1 and EVAL_V1_OPENING_RANDOM_MOVES=4~8 for a more decisive probe."
        )


def _build_gpu_state_batch(states: Sequence[GameState], device: torch.device):
    from v1.python.mcts_gpu import GpuStateBatch

    batch_size = len(states)
    board = torch.zeros((batch_size, BOARD_SIZE, BOARD_SIZE), dtype=torch.int8, device=device)
    marks_black = torch.zeros((batch_size, BOARD_SIZE, BOARD_SIZE), dtype=torch.bool, device=device)
    marks_white = torch.zeros((batch_size, BOARD_SIZE, BOARD_SIZE), dtype=torch.bool, device=device)
    phase = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    current_player = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    pending_marks_required = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    pending_marks_remaining = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    pending_captures_required = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    pending_captures_remaining = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    forced_removals_done = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    move_count = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    moves_since_capture = torch.zeros((batch_size,), dtype=torch.int64, device=device)

    for idx, state in enumerate(states):
        board[idx] = torch.tensor(state.board, dtype=torch.int8, device=device)
        if state.marked_black:
            rb, cb = zip(*state.marked_black)
            marks_black[idx, list(rb), list(cb)] = True
        if state.marked_white:
            rw, cw = zip(*state.marked_white)
            marks_white[idx, list(rw), list(cw)] = True
        phase[idx] = int(state.phase.value)
        current_player[idx] = int(state.current_player.value)
        pending_marks_required[idx] = int(state.pending_marks_required)
        pending_marks_remaining[idx] = int(state.pending_marks_remaining)
        pending_captures_required[idx] = int(state.pending_captures_required)
        pending_captures_remaining[idx] = int(state.pending_captures_remaining)
        forced_removals_done[idx] = int(state.forced_removals_done)
        move_count[idx] = int(state.move_count)
        moves_since_capture[idx] = int(state.moves_since_capture)

    return GpuStateBatch(
        board=board,
        marks_black=marks_black,
        marks_white=marks_white,
        phase=phase,
        current_player=current_player,
        pending_marks_required=pending_marks_required,
        pending_marks_remaining=pending_marks_remaining,
        pending_captures_required=pending_captures_required,
        pending_captures_remaining=pending_captures_remaining,
        forced_removals_done=forced_removals_done,
        move_count=move_count,
        moves_since_capture=moves_since_capture,
    )


def _decode_moves(states: Sequence[GameState], action_indices: torch.Tensor) -> List[Optional[dict]]:
    from v0.python.move_encoder import DEFAULT_ACTION_SPEC, decode_action_indices
    from v0.python.state_batch import from_game_states

    batch_cpu = from_game_states(states, device=torch.device("cpu"))
    return decode_action_indices(
        action_indices.to(device=torch.device("cpu"), dtype=torch.int64).view(-1),
        batch_cpu,
        DEFAULT_ACTION_SPEC,
    )


def _is_challenger_to_move(state: GameState, challenger_is_black: bool) -> bool:
    if state.current_player == Player.BLACK:
        return bool(challenger_is_black)
    return not bool(challenger_is_black)


class _V1EvalAgent:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str,
        mcts_simulations: int,
        temperature: float,
        sample_moves: bool,
    ) -> None:
        from v1.python.mcts_gpu import V1RootMCTS, V1RootMCTSConfig

        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.temperature = float(temperature)
        self.model = _load_model_from_checkpoint(checkpoint_path, str(self.device))
        cfg = V1RootMCTSConfig(
            num_simulations=max(1, int(mcts_simulations)),
            exploration_weight=1.0,
            temperature=float(temperature),
            add_dirichlet_noise=False,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            sample_moves=bool(sample_moves),
            child_eval_mode="value_only",
        )
        self.mcts = V1RootMCTS(
            model=self.model,
            config=cfg,
            device=self.device,
            inference_engine=None,
            collect_timing=False,
        )

    def select_moves(self, states: Sequence[GameState], concurrent_games: int) -> List[Optional[dict]]:
        if not states:
            return []
        chunk = max(1, int(concurrent_games))
        out: List[Optional[dict]] = [None for _ in range(len(states))]
        for start in range(0, len(states), chunk):
            end = min(start + chunk, len(states))
            sub_states = list(states[start:end])
            batch = _build_gpu_state_batch(sub_states, self.device)
            search = self.mcts.search_batch(
                batch,
                temperatures=float(self.temperature),
                add_dirichlet_noise=False,
            )
            chosen_idx = search.chosen_action_indices.detach().to("cpu", dtype=torch.int64)
            chosen_valid = search.chosen_valid_mask.detach().to("cpu", dtype=torch.bool)
            decoded = _decode_moves(sub_states, chosen_idx)
            for local_i, move in enumerate(decoded):
                dst_i = start + local_i
                if bool(chosen_valid[local_i].item()) and move is not None:
                    out[dst_i] = move
                    continue
                legal = generate_all_legal_moves(sub_states[local_i])
                out[dst_i] = legal[0] if legal else None
        return out


class _PortableEvalAgent:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str,
        mcts_simulations: int,
        temperature: float,
        sample_moves: bool,
    ) -> None:
        from v1.python.portable_device import resolve_portable_device
        from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig

        resolution = resolve_portable_device(device)
        self.device = resolution.device
        self.temperature = float(temperature)
        self.model = _load_model_from_checkpoint(checkpoint_path, str(self.device))
        self.mcts = PortableMCTS(
            model=self.model,
            config=PortableMCTSConfig(
                num_simulations=max(1, int(mcts_simulations)),
                exploration_weight=1.0,
                temperature=float(temperature),
                add_dirichlet_noise=False,
                sample_moves=bool(sample_moves),
            ),
            device=self.device,
        )

    def select_moves(self, states: Sequence[GameState], concurrent_games: int) -> List[Optional[dict]]:
        from v1.python.portable_mcts import PortableTree

        if not states:
            return []
        out: List[Optional[dict]] = []
        chunk = max(1, int(concurrent_games))
        for start in range(0, len(states), chunk):
            sub_states = list(states[start : start + chunk])
            trees = [PortableTree(state) for state in sub_states]
            search = self.mcts.search_batch(
                trees,
                temperatures=float(self.temperature),
                add_dirichlet_noise=False,
            )
            for state, row in zip(sub_states, search):
                if row.chosen_move is not None:
                    out.append(row.chosen_move)
                else:
                    legal = generate_all_legal_moves(state)
                    out.append(legal[0] if legal else None)
        return out


def _eval_worker_v1(
    worker_id: int,
    game_indices: List[int],
    num_games: int,
    device: str,
    mcts_simulations: int,
    temperature: float,
    challenger_checkpoint: str,
    opponent_checkpoint: Optional[str],
    seed: int,
    concurrent_games: int,
    opening_random_moves: int,
    sample_moves: bool,
    search_backend: str,
) -> V1WorkerResult:
    _seed_worker(int(seed) + int(worker_id))
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    agent_type = _PortableEvalAgent if str(search_backend) == "portable" else _V1EvalAgent
    challenger_agent = agent_type(
        challenger_checkpoint,
        device=str(device),
        mcts_simulations=int(mcts_simulations),
        temperature=float(temperature),
        sample_moves=bool(sample_moves),
    )
    if opponent_checkpoint:
        opponent_agent: Any = agent_type(
            opponent_checkpoint,
            device=str(device),
            mcts_simulations=int(mcts_simulations),
            temperature=float(temperature),
            sample_moves=bool(sample_moves),
        )
    else:
        opponent_agent = RandomAgent()

    half = num_games / 2
    contexts: List[Dict[str, Any]] = []
    for idx in game_indices:
        contexts.append(
            {
                "state": GameState(),
                "challenger_is_black": bool(idx < half),
                "done": False,
            }
        )

    totals = [0, 0, 0]
    black_totals = [0, 0, 0]
    white_totals = [0, 0, 0]

    def record_outcome(outcome: int, challenger_is_black: bool) -> None:
        totals[outcome] += 1
        color_totals = black_totals if challenger_is_black else white_totals
        color_totals[outcome] += 1

    unfinished = len(contexts)

    while unfinished > 0:
        pending_moves: Dict[int, Optional[dict]] = {}
        challenger_slots: List[int] = []
        challenger_states: List[GameState] = []
        opponent_slots: List[int] = []
        opponent_states: List[GameState] = []

        for slot, ctx in enumerate(contexts):
            if bool(ctx["done"]):
                continue

            state: GameState = ctx["state"]
            challenger_is_black = bool(ctx["challenger_is_black"])

            winner = state.get_winner()
            if winner is not None:
                challenger_win = (
                    (winner == Player.BLACK and challenger_is_black)
                    or (winner == Player.WHITE and not challenger_is_black)
                )
                if challenger_win:
                    record_outcome(0, challenger_is_black)
                else:
                    record_outcome(1, challenger_is_black)
                ctx["done"] = True
                unfinished -= 1
                continue

            if state.has_reached_move_limit():
                record_outcome(2, challenger_is_black)
                ctx["done"] = True
                unfinished -= 1
                continue

            legal = generate_all_legal_moves(state)
            if not legal:
                challenger_to_move = _is_challenger_to_move(state, challenger_is_black)
                if challenger_to_move:
                    record_outcome(1, challenger_is_black)
                else:
                    record_outcome(0, challenger_is_black)
                ctx["done"] = True
                unfinished -= 1
                continue

            if int(state.move_count) < int(opening_random_moves):
                pending_moves[slot] = random.choice(legal)
                continue

            challenger_to_move = _is_challenger_to_move(state, challenger_is_black)
            if challenger_to_move:
                challenger_slots.append(slot)
                challenger_states.append(state)
            else:
                if isinstance(opponent_agent, RandomAgent):
                    pending_moves[slot] = random.choice(legal)
                else:
                    opponent_slots.append(slot)
                    opponent_states.append(state)

        if challenger_states:
            chosen = challenger_agent.select_moves(
                challenger_states,
                concurrent_games=max(1, int(concurrent_games)),
            )
            for slot, mv in zip(challenger_slots, chosen):
                pending_moves[slot] = mv

        if opponent_states:
            chosen = opponent_agent.select_moves(
                opponent_states,
                concurrent_games=max(1, int(concurrent_games)),
            )
            for slot, mv in zip(opponent_slots, chosen):
                pending_moves[slot] = mv

        if not pending_moves and unfinished > 0:
            for ctx in contexts:
                if not bool(ctx["done"]):
                    record_outcome(2, bool(ctx["challenger_is_black"]))
                    ctx["done"] = True
                    unfinished -= 1
            continue

        for slot, move in pending_moves.items():
            ctx = contexts[slot]
            if bool(ctx["done"]):
                continue
            state: GameState = ctx["state"]
            challenger_is_black = bool(ctx["challenger_is_black"])
            challenger_to_move = _is_challenger_to_move(state, challenger_is_black)

            if move is None:
                if challenger_to_move:
                    record_outcome(1, challenger_is_black)
                else:
                    record_outcome(0, challenger_is_black)
                ctx["done"] = True
                unfinished -= 1
                continue

            try:
                next_state = apply_move(state, move, quiet=True)
            except Exception:
                if challenger_to_move:
                    record_outcome(1, challenger_is_black)
                else:
                    record_outcome(0, challenger_is_black)
                ctx["done"] = True
                unfinished -= 1
                continue

            ctx["state"] = next_state
            winner = next_state.get_winner()
            if winner is not None:
                challenger_win = (
                    (winner == Player.BLACK and challenger_is_black)
                    or (winner == Player.WHITE and not challenger_is_black)
                )
                if challenger_win:
                    record_outcome(0, challenger_is_black)
                else:
                    record_outcome(1, challenger_is_black)
                ctx["done"] = True
                unfinished -= 1
            elif next_state.has_reached_move_limit():
                record_outcome(2, challenger_is_black)
                ctx["done"] = True
                unfinished -= 1

    return (
        totals[0],
        totals[1],
        totals[2],
        black_totals[0],
        black_totals[1],
        black_totals[2],
        white_totals[0],
        white_totals[1],
        white_totals[2],
    )


def evaluate_against_agent_parallel_v1(
    *,
    challenger_checkpoint: str,
    opponent_checkpoint: Optional[str],
    num_games: int,
    device: str,
    mcts_simulations: int,
    temperature: float,
    num_workers: int,
    devices: Optional[Sequence[str]],
    concurrent_games: int,
    opening_random_moves: int,
    sample_moves: bool,
    search_backend: str = "cuda_root",
    seed: Optional[int] = None,
) -> EvaluationStats:
    num_games_n = _normalize_eval_games(int(num_games))
    if num_games_n == 0:
        return EvaluationStats(wins=0, losses=0, draws=0, total_games=0)

    devices_list = _parse_devices(device=str(device), eval_devices=",".join(devices) if isinstance(devices, list) else (devices if isinstance(devices, str) else None))
    workers = max(1, min(int(num_workers), int(num_games_n)))
    if str(search_backend) == "portable" and workers != 1:
        raise RuntimeError(
            "The portable eval backend is single-process in its first version; set --eval_workers 1."
        )
    chunks = _split_game_indices(num_games_n, workers)
    if not chunks:
        return EvaluationStats(wins=0, losses=0, draws=0, total_games=num_games_n)

    seed_base = (
        int(seed)
        if seed is not None
        else int(time.time() * 1e6) & 0x7FFFFFFF
    )
    if len(chunks) == 1:
        result = _eval_worker_v1(
            0,
            chunks[0],
            num_games_n,
            devices_list[0],
            int(mcts_simulations),
            float(temperature),
            challenger_checkpoint,
            opponent_checkpoint,
            seed_base,
            int(concurrent_games),
            int(opening_random_moves),
            bool(sample_moves),
            str(search_backend),
        )
        return _aggregate_v1_worker_results(
            [result], total_games=num_games_n, seed=seed_base
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(chunks)) as pool:
        results = pool.starmap(
            _eval_worker_v1,
            [
                (
                    worker_id,
                    chunks[worker_id],
                    num_games_n,
                    devices_list[worker_id % len(devices_list)],
                    int(mcts_simulations),
                    float(temperature),
                    challenger_checkpoint,
                    opponent_checkpoint,
                    seed_base,
                    int(concurrent_games),
                    int(opening_random_moves),
                    bool(sample_moves),
                    str(search_backend),
                )
                for worker_id in range(len(chunks))
            ],
        )

    return _aggregate_v1_worker_results(
        results, total_games=num_games_n, seed=seed_base
    )


def _run_eval_one(
    *,
    backend: str,
    challenger_checkpoint: str,
    opponent_checkpoint: Optional[str],
    num_games: int,
    device: str,
    devices: List[str],
    workers: int,
    mcts_simulations: int,
    temperature: float,
    sample_moves: bool,
    batch_leaves: int,
    inference_backend: str,
    inference_batch_size: int,
    inference_warmup_iters: int,
    v1_concurrent_games: int,
    v1_opening_random_moves: int,
    seed: Optional[int],
) -> EvaluationStats:
    if backend in {"v1", "portable"}:
        return evaluate_against_agent_parallel_v1(
            challenger_checkpoint=challenger_checkpoint,
            opponent_checkpoint=opponent_checkpoint,
            num_games=int(num_games),
            device=str(device),
            mcts_simulations=int(mcts_simulations),
            temperature=float(temperature),
            num_workers=int(workers),
            devices=devices,
            concurrent_games=int(v1_concurrent_games),
            opening_random_moves=int(v1_opening_random_moves),
            sample_moves=bool(sample_moves),
            search_backend=("portable" if backend == "portable" else "cuda_root"),
            seed=seed,
        )

    if seed is not None:
        _seed_worker(int(seed))

    if backend == "v0":
        try:
            return evaluate_against_agent_parallel_v0(
                challenger_checkpoint=challenger_checkpoint,
                opponent_checkpoint=opponent_checkpoint,
                num_games=int(num_games),
                device=str(device),
                mcts_simulations=int(mcts_simulations),
                temperature=float(temperature),
                add_dirichlet_noise=False,
                num_workers=int(workers),
                devices=devices,
                verbose=False,
                game_verbose=False,
                mcts_verbose=False,
                batch_leaves=int(batch_leaves),
                inference_backend=str(inference_backend),
                torchscript_path=None,
                torchscript_dtype=None,
                inference_batch_size=int(inference_batch_size),
                inference_warmup_iters=int(inference_warmup_iters),
                sample_moves=bool(sample_moves),
            )
        except Exception as exc:
            print(
                "[eval] warning: backend=v0 failed, fallback to legacy evaluate path. "
                f"reason={exc}"
            )

    return evaluate_against_agent_parallel(
        challenger_checkpoint=challenger_checkpoint,
        opponent_checkpoint=opponent_checkpoint,
        num_games=int(num_games),
        device=str(device),
        mcts_simulations=int(mcts_simulations),
        temperature=float(temperature),
        add_dirichlet_noise=False,
        num_workers=int(workers),
        devices=devices,
        verbose=False,
        game_verbose=False,
        mcts_verbose=False,
        sample_moves=bool(sample_moves),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint against random/previous.")
    parser.add_argument("--challenger_checkpoint", type=str, required=True)
    parser.add_argument("--previous_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_devices", type=str, default=None)
    parser.add_argument("--eval_workers", type=int, default=1)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["v0", "legacy", "v1", "portable"],
        default="v0",
    )
    parser.add_argument("--mcts_simulations", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--sample_moves", action="store_true")
    parser.add_argument("--eval_games_vs_random", type=int, default=0)
    parser.add_argument("--eval_games_vs_previous", type=int, default=0)
    parser.add_argument("--batch_leaves", type=int, default=256)
    parser.add_argument("--inference_backend", type=str, default="graph")
    parser.add_argument("--inference_batch_size", type=int, default=512)
    parser.add_argument("--inference_warmup_iters", type=int, default=5)
    parser.add_argument("--v1_concurrent_games", type=int, default=64)
    parser.add_argument("--v1_opening_random_moves", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Explicit evaluation seed. Fully applied and persisted by v1/portable; "
            "legacy/v0 reproducibility can still depend on their worker implementation."
        ),
    )
    parser.add_argument("--match_name", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    challenger = str(args.challenger_checkpoint)
    if not os.path.exists(challenger):
        raise FileNotFoundError(f"challenger checkpoint not found: {challenger}")

    prev_path = str(args.previous_checkpoint).strip() if args.previous_checkpoint is not None else ""
    previous_checkpoint = prev_path if prev_path and os.path.exists(prev_path) else None
    if prev_path and previous_checkpoint is None:
        print(f"[eval] warning: previous checkpoint not found, skip vs_previous: {prev_path}")

    backend = str(args.backend).strip().lower()
    requested_devices = _parse_devices(device=str(args.device), eval_devices=args.eval_devices)
    device_resolution_payload: Dict[str, Any] = {
        "requested": list(requested_devices),
        "resolved": list(requested_devices),
        "fallback_count": 0,
        "fallback_reasons": [],
    }
    if backend == "portable":
        from v1.python.portable_device import resolve_portable_device

        if len(requested_devices) != 1:
            raise RuntimeError(
                "The portable eval backend is single-device in its first version; "
                f"got {requested_devices}."
            )
        resolution = resolve_portable_device(requested_devices[0])
        eval_devices = [str(resolution.device)]
        device_resolution_payload = {
            "requested": list(requested_devices),
            "resolved": list(eval_devices),
            "fallback_count": int(resolution.fallback_count),
            "fallback_reasons": list(resolution.fallback_reasons),
        }
    else:
        eval_devices = requested_devices
    effective_device = eval_devices[0]
    workers = max(1, int(args.eval_workers))
    started = time.perf_counter()

    result_rows: List[Dict[str, Any]] = []
    games_vs_random = max(0, int(args.eval_games_vs_random))
    if games_vs_random > 0:
        random_name = str(args.match_name or "vs_random").strip() or "vs_random"
        stats = _run_eval_one(
            backend=backend,
            challenger_checkpoint=challenger,
            opponent_checkpoint=None,
            num_games=games_vs_random,
            device=effective_device,
            devices=eval_devices,
            workers=workers,
            mcts_simulations=int(args.mcts_simulations),
            temperature=float(args.temperature),
            sample_moves=bool(args.sample_moves),
            batch_leaves=int(args.batch_leaves),
            inference_backend=str(args.inference_backend),
            inference_batch_size=int(args.inference_batch_size),
            inference_warmup_iters=int(args.inference_warmup_iters),
            v1_concurrent_games=int(args.v1_concurrent_games),
            v1_opening_random_moves=int(args.v1_opening_random_moves),
            seed=args.seed,
        )
        payload = _stats_to_payload(random_name, stats)
        result_rows.append(payload)
        _print_stats_line(random_name, payload)

    games_vs_previous = max(0, int(args.eval_games_vs_previous))
    if games_vs_previous > 0 and previous_checkpoint:
        match_name = str(args.match_name or "vs_previous").strip() or "vs_previous"
        stats = _run_eval_one(
            backend=backend,
            challenger_checkpoint=challenger,
            opponent_checkpoint=previous_checkpoint,
            num_games=games_vs_previous,
            device=effective_device,
            devices=eval_devices,
            workers=workers,
            mcts_simulations=int(args.mcts_simulations),
            temperature=float(args.temperature),
            sample_moves=bool(args.sample_moves),
            batch_leaves=int(args.batch_leaves),
            inference_backend=str(args.inference_backend),
            inference_batch_size=int(args.inference_batch_size),
            inference_warmup_iters=int(args.inference_warmup_iters),
            v1_concurrent_games=int(args.v1_concurrent_games),
            v1_opening_random_moves=int(args.v1_opening_random_moves),
            seed=args.seed,
        )
        payload = _stats_to_payload(match_name, stats)
        result_rows.append(payload)
        _print_stats_line(match_name, payload)
    elif games_vs_previous > 0 and not previous_checkpoint:
        skipped_name = str(args.match_name or "vs_previous").strip() or "vs_previous"
        print(f"[eval] skip {skipped_name}: previous checkpoint unavailable.")

    elapsed = max(1e-9, time.perf_counter() - started)
    report = {
        "challenger_checkpoint": challenger,
        "previous_checkpoint": previous_checkpoint,
        "backend": backend,
        "device": effective_device,
        "eval_devices": eval_devices,
        "device_resolution": device_resolution_payload,
        "eval_workers": workers,
        "mcts_simulations": int(args.mcts_simulations),
        "temperature": float(args.temperature),
        "sample_moves": bool(args.sample_moves),
        "v1_concurrent_games": int(args.v1_concurrent_games),
        "v1_opening_random_moves": int(args.v1_opening_random_moves),
        "requested_seed": int(args.seed) if args.seed is not None else None,
        "elapsed_sec": float(elapsed),
        "results": result_rows,
    }

    output_json = str(args.output_json).strip() if args.output_json is not None else ""
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[eval] report saved: {output_json}")

    if not result_rows:
        print("[eval] no evaluation configured (all eval_games are zero).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
