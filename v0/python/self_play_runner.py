"""Shared v0 self-play helpers used by both training and data-generation."""

from __future__ import annotations

import io
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from src.game_state import GameState, Player
from src.move_generator import apply_move
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS

from .mcts import MCTS as FastMCTS

__all__ = [
    "self_play_single_game_v0",
    "self_play_v0",
]


def _serialize_model_state(model: ChessNet) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _soft_value_from_state(state: GameState, soft_value_k: float) -> float:
    board_area = GameState.BOARD_SIZE * GameState.BOARD_SIZE
    black_count = state.count_player_pieces(Player.BLACK)
    white_count = state.count_player_pieces(Player.WHITE)
    material_delta = (black_count - white_count) / float(board_area)
    return float(np.tanh(soft_value_k * material_delta))


def self_play_single_game_v0(
    model: ChessNet,
    mcts_simulations: int,
    temperature_init: float,
    temperature_final: float,
    temperature_threshold: int,
    exploration_weight: float,
    device: str,
    add_dirichlet_noise: bool,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    batch_leaves: int,
    seed: int,
    verbose: bool = False,
    mcts_verbose: bool = False,
    soft_value_k: float = 2.0,
) -> Tuple[List[GameState], List[np.ndarray], float, float]:
    rng = np.random.default_rng(seed)
    mcts = FastMCTS(
        model=model,
        num_simulations=mcts_simulations,
        exploration_weight=exploration_weight,
        temperature=temperature_init,
        device=device,
        add_dirichlet_noise=add_dirichlet_noise,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        batch_K=batch_leaves,
        seed=seed,
    )

    state = GameState()
    game_states: List[GameState] = []
    game_policies: List[np.ndarray] = []
    move_count = 0

    while True:
        current_temp = temperature_init if move_count < temperature_threshold else temperature_final
        mcts.set_temperature(current_temp)
        moves, policy = mcts.search(state)

        if verbose:
            player = "BLACK" if state.current_player == Player.BLACK else "WHITE"
            print(
                f"[Self-Play] Move {move_count + 1} | player={player} | "
                f"temperature={current_temp:.2f} | legal_moves={len(moves)}"
            )

        policy = np.asarray(policy, dtype=np.float64)
        if policy.size and policy.sum() <= 0:
            policy.fill(1.0 / len(policy))

        game_states.append(state.copy())
        game_policies.append(policy.copy())

        if not moves:
            winner = state.get_winner()
            result = 0.0 if winner is None else (1.0 if winner == Player.BLACK else -1.0)
            soft_value = _soft_value_from_state(state, soft_value_k)
            return game_states, game_policies, result, soft_value

        move_idx = rng.choice(len(moves), p=policy if policy.size else None)
        move = moves[int(move_idx)]

        state = apply_move(state, move, quiet=True)
        mcts.advance_root(move)
        move_count += 1

        winner = state.get_winner()
        if winner is not None:
            result = 1.0 if winner == Player.BLACK else -1.0
            soft_value = _soft_value_from_state(state, soft_value_k)
            return game_states, game_policies, result, soft_value

        if state.has_reached_move_limit():
            soft_value = _soft_value_from_state(state, soft_value_k)
            return game_states, game_policies, 0.0, soft_value


def _v0_self_play_worker(
    worker_id: int,
    cfg: Dict[str, Any],
    model_state_bytes: bytes,
    return_queue: "mp.Queue",
) -> None:
    try:
        try:
            cpu_total = os.cpu_count() or 1
            per_worker = max(1, cpu_total // max(1, int(cfg.get("num_workers", 1))))
            torch.set_num_threads(per_worker)
        except Exception:
            pass

        worker_seed = int(cfg.get("worker_seed", 0))
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        device = torch.device(cfg["device"])
        if device.type == "cuda":
            torch.cuda.set_device(device.index or 0)

        model = ChessNet(
            board_size=GameState.BOARD_SIZE,
            num_input_channels=cfg.get("num_input_channels", NUM_INPUT_CHANNELS),
        )
        buffer = io.BytesIO(model_state_bytes)
        state_dict = torch.load(buffer, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        games: List[Tuple[List[GameState], List[np.ndarray], float, float]] = []
        base_seed = worker_seed

        for game_idx in range(cfg["games_per_worker"]):
            game_seed = base_seed + 1000007 * (game_idx + 1)
            games.append(
                self_play_single_game_v0(
                    model=model,
                    mcts_simulations=cfg["mcts_simulations"],
                    temperature_init=cfg["temperature_init"],
                    temperature_final=cfg["temperature_final"],
                    temperature_threshold=cfg["temperature_threshold"],
                    exploration_weight=cfg["exploration_weight"],
                    device=cfg["device"],
                    add_dirichlet_noise=cfg["add_dirichlet_noise"],
                    dirichlet_alpha=cfg["dirichlet_alpha"],
                    dirichlet_epsilon=cfg["dirichlet_epsilon"],
                    batch_leaves=cfg["batch_leaves"],
                    seed=game_seed,
                    verbose=bool(cfg.get("verbose", False)) and worker_id == 0,
                    mcts_verbose=bool(cfg.get("mcts_verbose", False)),
                    soft_value_k=cfg["soft_value_k"],
                )
            )

        return_queue.put(("ok", worker_id, games))
    except Exception as exc:
        return_queue.put(("err", worker_id, repr(exc)))


def self_play_v0(
    model: ChessNet,
    num_games: int,
    mcts_simulations: int,
    temperature_init: float,
    temperature_final: float,
    temperature_threshold: int,
    exploration_weight: float,
    device: str,
    add_dirichlet_noise: bool,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    batch_leaves: int,
    num_workers: int,
    games_per_worker: Optional[int],
    base_seed: Optional[int],
    soft_value_k: float,
    mcts_verbose: bool,
    verbose: bool,
) -> List[Tuple[List[GameState], List[np.ndarray], float, float]]:
    model.eval()

    if num_workers <= 1:
        rng = random.Random(base_seed or int(time.time() * 1e6))
        games: List[Tuple[List[GameState], List[np.ndarray], float, float]] = []
        for _ in range(num_games):
            seed = rng.randint(0, 2**31 - 1)
            games.append(
                self_play_single_game_v0(
                    model=model,
                    mcts_simulations=mcts_simulations,
                    temperature_init=temperature_init,
                    temperature_final=temperature_final,
                    temperature_threshold=temperature_threshold,
                    exploration_weight=exploration_weight,
                    device=device,
                    add_dirichlet_noise=add_dirichlet_noise,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_epsilon=dirichlet_epsilon,
                    batch_leaves=batch_leaves,
                    seed=seed,
                    verbose=verbose,
                    mcts_verbose=mcts_verbose,
                    soft_value_k=soft_value_k,
                )
            )
        return games

    if games_per_worker is None or games_per_worker <= 0:
        raise ValueError("games_per_worker must be provided when num_workers > 1.")

    ctx = mp.get_context("spawn")
    return_queue: "mp.Queue" = ctx.Queue()
    model_bytes = _serialize_model_state(model)
    workers: List[mp.Process] = []
    games: List[Tuple[List[GameState], List[np.ndarray], float, float]] = []

    common_cfg = {
        "mcts_simulations": mcts_simulations,
        "temperature_init": temperature_init,
        "temperature_final": temperature_final,
        "temperature_threshold": temperature_threshold,
        "exploration_weight": exploration_weight,
        "device": device,
        "add_dirichlet_noise": add_dirichlet_noise,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_epsilon": dirichlet_epsilon,
        "batch_leaves": batch_leaves,
        "soft_value_k": soft_value_k,
        "verbose": verbose,
        "mcts_verbose": mcts_verbose,
        "num_workers": num_workers,
        "num_input_channels": NUM_INPUT_CHANNELS,
        "games_per_worker": games_per_worker,
    }

    base_seed = base_seed or random.randint(1, 10**9)

    try:
        for worker_id in range(num_workers):
            worker_cfg = dict(common_cfg)
            worker_cfg["worker_seed"] = base_seed + 1000003 * worker_id
            process = ctx.Process(
                target=_v0_self_play_worker,
                args=(worker_id, worker_cfg, model_bytes, return_queue),
                daemon=False,
            )
            process.start()
            workers.append(process)

        finished = 0
        while finished < num_workers:
            status, worker_id, payload = return_queue.get()
            if status == "ok":
                games.extend(payload)
                finished += 1
            else:
                raise RuntimeError(f"Self-play worker {worker_id} failed: {payload}")
    finally:
        for process in workers:
            if process.is_alive():
                process.join(timeout=0.1)
                if process.is_alive():
                    process.terminate()
        return_queue.close()

    if len(games) != num_workers * games_per_worker:
        print(
            f"Warning: expected {num_workers * games_per_worker} games but received {len(games)}."
        )

    return games[:num_games]
