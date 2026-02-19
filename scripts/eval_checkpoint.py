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
from v0.python.move_encoder import DEFAULT_ACTION_SPEC, decode_action_indices  # noqa: E402
from v0.python.state_batch import from_game_states  # noqa: E402
from v1.python.mcts_gpu import GpuStateBatch, V1RootMCTS, V1RootMCTSConfig  # noqa: E402


BOARD_SIZE = int(GameState.BOARD_SIZE)


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


def _load_model_from_checkpoint(checkpoint_path: str, device: str) -> ChessNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    board_size = checkpoint.get("board_size", GameState.BOARD_SIZE)
    num_inputs = checkpoint.get("num_input_channels", NUM_INPUT_CHANNELS)
    model = ChessNet(board_size=int(board_size), num_input_channels=int(num_inputs))
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model


def _stats_to_payload(name: str, stats: EvaluationStats) -> Dict[str, Any]:
    return {
        "name": str(name),
        "wins": int(stats.wins),
        "losses": int(stats.losses),
        "draws": int(stats.draws),
        "total_games": int(stats.total_games),
        "win_rate": float(stats.win_rate),
        "loss_rate": float(stats.loss_rate),
        "draw_rate": float(stats.draw_rate),
    }


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


def _build_gpu_state_batch(states: Sequence[GameState], device: torch.device) -> GpuStateBatch:
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
) -> Tuple[int, int, int]:
    _seed_worker(int(seed) + int(worker_id))
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    challenger_agent = _V1EvalAgent(
        challenger_checkpoint,
        device=str(device),
        mcts_simulations=int(mcts_simulations),
        temperature=float(temperature),
        sample_moves=bool(sample_moves),
    )
    if opponent_checkpoint:
        opponent_agent: Any = _V1EvalAgent(
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

    wins = 0
    losses = 0
    draws = 0
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
                    wins += 1
                else:
                    losses += 1
                ctx["done"] = True
                unfinished -= 1
                continue

            if state.has_reached_move_limit():
                draws += 1
                ctx["done"] = True
                unfinished -= 1
                continue

            legal = generate_all_legal_moves(state)
            if not legal:
                challenger_to_move = _is_challenger_to_move(state, challenger_is_black)
                if challenger_to_move:
                    losses += 1
                else:
                    wins += 1
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
                    draws += 1
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
                    losses += 1
                else:
                    wins += 1
                ctx["done"] = True
                unfinished -= 1
                continue

            try:
                next_state = apply_move(state, move, quiet=True)
            except Exception:
                if challenger_to_move:
                    losses += 1
                else:
                    wins += 1
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
                    wins += 1
                else:
                    losses += 1
                ctx["done"] = True
                unfinished -= 1
            elif next_state.has_reached_move_limit():
                draws += 1
                ctx["done"] = True
                unfinished -= 1

    return wins, losses, draws


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
) -> EvaluationStats:
    num_games_n = _normalize_eval_games(int(num_games))
    if num_games_n == 0:
        return EvaluationStats(wins=0, losses=0, draws=0, total_games=0)

    devices_list = _parse_devices(device=str(device), eval_devices=",".join(devices) if isinstance(devices, list) else (devices if isinstance(devices, str) else None))
    workers = max(1, min(int(num_workers), int(num_games_n)))
    chunks = _split_game_indices(num_games_n, workers)
    if not chunks:
        return EvaluationStats(wins=0, losses=0, draws=0, total_games=num_games_n)

    seed_base = int(time.time() * 1e6) & 0x7FFFFFFF
    if len(chunks) == 1:
        wins, losses, draws = _eval_worker_v1(
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
        )
        return EvaluationStats(wins=wins, losses=losses, draws=draws, total_games=num_games_n)

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
                )
                for worker_id in range(len(chunks))
            ],
        )

    wins = sum(int(r[0]) for r in results)
    losses = sum(int(r[1]) for r in results)
    draws = sum(int(r[2]) for r in results)
    return EvaluationStats(wins=wins, losses=losses, draws=draws, total_games=num_games_n)


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
) -> EvaluationStats:
    if backend == "v1":
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
        )

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
    parser.add_argument("--backend", type=str, choices=["v0", "legacy", "v1"], default="v0")
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

    eval_devices = _parse_devices(device=str(args.device), eval_devices=args.eval_devices)
    workers = max(1, int(args.eval_workers))
    backend = str(args.backend).strip().lower()
    started = time.perf_counter()

    result_rows: List[Dict[str, Any]] = []
    games_vs_random = max(0, int(args.eval_games_vs_random))
    if games_vs_random > 0:
        stats = _run_eval_one(
            backend=backend,
            challenger_checkpoint=challenger,
            opponent_checkpoint=None,
            num_games=games_vs_random,
            device=str(args.device),
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
        )
        payload = _stats_to_payload("vs_random", stats)
        result_rows.append(payload)
        _print_stats_line("vs_random", payload)

    games_vs_previous = max(0, int(args.eval_games_vs_previous))
    if games_vs_previous > 0 and previous_checkpoint:
        stats = _run_eval_one(
            backend=backend,
            challenger_checkpoint=challenger,
            opponent_checkpoint=previous_checkpoint,
            num_games=games_vs_previous,
            device=str(args.device),
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
        )
        payload = _stats_to_payload("vs_previous", stats)
        result_rows.append(payload)
        _print_stats_line("vs_previous", payload)
    elif games_vs_previous > 0 and not previous_checkpoint:
        print("[eval] skip vs_previous: previous checkpoint unavailable.")

    elapsed = max(1e-9, time.perf_counter() - started)
    report = {
        "challenger_checkpoint": challenger,
        "previous_checkpoint": previous_checkpoint,
        "backend": backend,
        "device": str(args.device),
        "eval_devices": eval_devices,
        "eval_workers": workers,
        "mcts_simulations": int(args.mcts_simulations),
        "temperature": float(args.temperature),
        "sample_moves": bool(args.sample_moves),
        "v1_concurrent_games": int(args.v1_concurrent_games),
        "v1_opening_random_moves": int(args.v1_opening_random_moves),
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
