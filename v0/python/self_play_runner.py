"""Shared v0 self-play helpers used by both training and data-generation."""

from __future__ import annotations

import copy
import gc
import io
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


def _unwrap_model(model: ChessNet) -> ChessNet:
    return model.module if hasattr(model, "module") else model


def _serialize_model_state(model: ChessNet) -> bytes:
    buffer = io.BytesIO()
    torch.save(_unwrap_model(model).state_dict(), buffer)
    return buffer.getvalue()


def _normalize_device_list(devices: Optional[Sequence[str]], fallback: str) -> List[str]:
    if devices is None:
        return [fallback]
    if isinstance(devices, str):
        text = devices.strip()
        if not text:
            return [fallback]
        key = text.lower()
        if key in ("auto", "all", "visible", "available"):
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                if count > 0:
                    return [f"cuda:{i}" for i in range(count)]
            return [fallback]
        tokens = [item.strip() for item in text.split(",") if item.strip()]
    else:
        tokens = []
        for item in devices:
            if item is None:
                continue
            token = str(item).strip()
            if token:
                tokens.append(token)

    normalized: List[str] = []
    for token in tokens:
        lower = token.lower()
        if lower == "cuda":
            normalized.append("cuda:0")
        elif lower in ("cpu", "mps"):
            normalized.append(lower)
        elif lower.startswith(("cuda:", "cpu", "mps")):
            normalized.append(token)
        elif token.isdigit():
            normalized.append(f"cuda:{token}")
        else:
            normalized.append(token)

    seen = set()
    unique: List[str] = []
    for dev in normalized:
        if dev not in seen:
            unique.append(dev)
            seen.add(dev)
    return unique or [fallback]


def _dtype_from_string(value: str) -> torch.dtype:
    key = value.strip().lower()
    if key in ("float32", "fp32", "f32"):
        return torch.float32
    if key in ("float16", "fp16", "f16"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def _cleanup_cuda_memory(device: torch.device) -> None:
    """Aggressively release unused CUDA memory back to the system.

    This runs ``gc.collect()`` to ensure Python destructs C++ extension
    objects (e.g. InferenceEngine / CUDAGraph) before calling
    ``torch.cuda.empty_cache()`` so the allocator actually returns the
    blocks.  Called between games and between pipeline phases to prevent
    gradual VRAM accumulation that leads to OOM.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _maybe_clear_cuda_cache(device: torch.device) -> None:
    """Legacy helper – now always delegates to ``_cleanup_cuda_memory``."""
    _cleanup_cuda_memory(device)


def _default_torchscript_dtype(device: torch.device) -> str:
    if device.type == "cuda":
        return "float16"
    return "float32"


def _export_torchscript(
    model: ChessNet,
    device: torch.device,
    dtype_str: str,
    batch_size: int,
) -> str:
    dtype = _dtype_from_string(dtype_str)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 export is not supported on CPU.")
    model_copy = copy.deepcopy(_unwrap_model(model))
    model_copy.to(device=device, dtype=dtype)
    model_copy.eval()
    inputs = torch.zeros(
        batch_size,
        NUM_INPUT_CHANNELS,
        GameState.BOARD_SIZE,
        GameState.BOARD_SIZE,
        device=device,
        dtype=dtype,
    )
    with torch.inference_mode():
        scripted = torch.jit.trace(model_copy, inputs, strict=False)
        tmp = tempfile.NamedTemporaryFile(prefix="v0_self_play_", suffix=".pt", delete=False)
        tmp_path = Path(tmp.name)
    tmp.close()
    scripted.save(str(tmp_path))
    return str(tmp_path)


_EVAL_STATS_BUCKET_LABELS = [
    "1-32",
    "33-64",
    "65-96",
    "97-128",
    "129-160",
    "161-192",
    "193-224",
    "225-256",
    "257-288",
    "289-320",
    "321-352",
    "353-384",
    "385-416",
    "417-448",
    "449-480",
    "481-512",
    "513+",
]


def _eval_stats_enabled() -> bool:
    value = os.environ.get("V0_EVAL_STATS")
    if not value:
        return False
    return value != "0"


def _empty_eval_stats() -> Dict[str, object]:
    return {
        "eval_calls": 0,
        "eval_leaves": 0,
        "full512_calls": 0,
        "hist": [0 for _ in _EVAL_STATS_BUCKET_LABELS],
    }


def _merge_eval_stats(target: Dict[str, object], incoming: Dict[str, object]) -> None:
    if not incoming:
        return
    target["eval_calls"] = int(target.get("eval_calls", 0)) + int(incoming.get("eval_calls", 0))
    target["eval_leaves"] = int(target.get("eval_leaves", 0)) + int(incoming.get("eval_leaves", 0))
    target["full512_calls"] = int(target.get("full512_calls", 0)) + int(incoming.get("full512_calls", 0))
    target_hist = list(target.get("hist", []))
    incoming_hist = list(incoming.get("hist", []))
    limit = min(len(target_hist), len(incoming_hist))
    for i in range(limit):
        target_hist[i] += int(incoming_hist[i])
    target["hist"] = target_hist


def _format_eval_stats(
    stats: Dict[str, object],
    inference_batch_size: int,
    mcts_leaf_batch_cap: int,
) -> Dict[str, object]:
    eval_calls = int(stats.get("eval_calls", 0))
    eval_leaves = int(stats.get("eval_leaves", 0))
    full512_calls = int(stats.get("full512_calls", 0))
    avg_batch = (eval_leaves / eval_calls) if eval_calls else 0.0
    full512_ratio = (full512_calls / eval_calls) if eval_calls else 0.0
    hist_counts = list(stats.get("hist", []))
    hist = {
        label: int(hist_counts[i]) if i < len(hist_counts) else 0
        for i, label in enumerate(_EVAL_STATS_BUCKET_LABELS)
    }
    payload = {
        "eval_calls": eval_calls,
        "eval_leaves": eval_leaves,
        "avg_batch": avg_batch,
        "hist": hist,
        "full512_ratio": full512_ratio,
        "mcts_leaf_batch_cap": int(mcts_leaf_batch_cap),
        "graph_batch_size": int(inference_batch_size),
    }
    if inference_batch_size == 512:
        payload["pad_leaves"] = eval_calls * 512 - eval_leaves
    return payload


def _print_eval_stats(payload: Dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


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
    virtual_loss: float,
    seed: int,
    opening_random_moves: int = 0,
    resign_threshold: float = -0.8,
    resign_min_moves: int = 10,
    resign_consecutive: int = 3,
    verbose: bool = False,
    mcts_verbose: bool = False,
    soft_value_k: float = 2.0,
    inference_backend: str = "graph",
    torchscript_path: Optional[str] = None,
    torchscript_dtype: Optional[str] = None,
    inference_batch_size: int = 512,
    inference_warmup_iters: int = 5,
    eval_stats_sink: Optional[Callable[[Dict[str, object]], None]] = None,
    mcts_instance: Optional[FastMCTS] = None,
) -> Tuple[List[GameState], List[np.ndarray], List[List[dict]], float, float]:
    """Play a single self-play game.

    When *mcts_instance* is supplied the function reuses that object (and
    its already-allocated InferenceEngine / CUDA Graph) instead of
    creating a new one.  This avoids the main source of VRAM accumulation
    that previously caused OOM.
    """
    rng = np.random.default_rng(seed)
    if mcts_instance is not None:
        mcts = mcts_instance
        # Reset the tree but keep the InferenceEngine / CUDA Graph alive.
        mcts.reset()
        mcts.set_temperature(temperature_init)
    else:
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
            virtual_loss=virtual_loss,
            seed=seed,
            verbose=mcts_verbose,
            inference_backend=inference_backend,
            torchscript_path=torchscript_path,
            torchscript_dtype=torchscript_dtype,
            inference_batch_size=inference_batch_size,
            inference_warmup_iters=inference_warmup_iters,
        )
    if eval_stats_sink is not None:
        mcts.reset_eval_stats()

    def _finalize(result: float, soft_value: float):
        if eval_stats_sink is not None:
            eval_stats_sink(mcts.get_eval_stats())
        return game_states, game_policies, game_legal_moves, result, soft_value

    state = GameState()
    game_states: List[GameState] = []
    game_policies: List[np.ndarray] = []
    game_legal_moves: List[List[dict]] = []  # Store legal moves for training
    move_count = 0
    opening_random_moves = max(0, int(opening_random_moves))
    resign_threshold = float(resign_threshold)
    resign_min_moves = max(0, int(resign_min_moves))
    resign_consecutive = max(1, int(resign_consecutive))
    enable_resign = resign_threshold < 0.0
    resign_count = 0

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
            print(state)

        policy = np.asarray(policy, dtype=np.float64)
        if policy.size and policy.sum() <= 0:
            policy.fill(1.0 / len(policy))

        game_states.append(state.copy())
        game_policies.append(policy.copy())
        game_legal_moves.append(list(moves))  # Save legal moves for training

        if not moves:
            winner = state.get_winner()
            result = 0.0 if winner is None else (1.0 if winner == Player.BLACK else -1.0)
            soft_value = _soft_value_from_state(state, soft_value_k)
            return _finalize(result, soft_value)

        root_value = float(mcts.get_root_value())
        if enable_resign and move_count >= resign_min_moves:
            if root_value <= resign_threshold:
                resign_count += 1
                if resign_count >= resign_consecutive:
                    result = -1.0 if state.current_player == Player.BLACK else 1.0
                    soft_value = _soft_value_from_state(state, soft_value_k)
                    return _finalize(result, soft_value)
            else:
                resign_count = 0
        else:
            resign_count = 0

        if opening_random_moves > 0 and move_count < opening_random_moves:
            move_idx = rng.integers(len(moves))
        else:
            move_idx = rng.choice(len(moves), p=policy if policy.size else None)
        move = moves[int(move_idx)]

        state = apply_move(state, move, quiet=True)
        mcts.advance_root(move)
        move_count += 1

        if verbose:
            chosen_prob = float(policy[int(move_idx)]) if policy.size > 0 else 0.0
            print(f"[Self-Play] Selected move: {move} | policy_prob={chosen_prob:.4f}")
            print(state)

        winner = state.get_winner()
        if winner is not None:
            result = 1.0 if winner == Player.BLACK else -1.0
            soft_value = _soft_value_from_state(state, soft_value_k)
            return _finalize(result, soft_value)

        if state.has_reached_move_limit():
            soft_value = _soft_value_from_state(state, soft_value_k)
            return _finalize(0.0, soft_value)


def _v0_self_play_worker(
    worker_id: int,
    cfg: Dict[str, Any],
    model_state_bytes: bytes,
    return_queue: "mp.Queue",
) -> None:
    mcts_obj: Optional[FastMCTS] = None
    model: Optional[ChessNet] = None
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

        eval_stats_enabled = _eval_stats_enabled()
        worker_stats = _empty_eval_stats() if eval_stats_enabled else None

        def _sink(stats: Dict[str, object]) -> None:
            if worker_stats is None:
                return
            _merge_eval_stats(worker_stats, stats)

        # --- KEY FIX: create MCTS once and reuse across all games ---
        # This avoids re-creating InferenceEngine / CUDA Graph per game,
        # which is the primary source of VRAM accumulation.
        mcts_obj = FastMCTS(
            model=model,
            num_simulations=cfg["mcts_simulations"],
            exploration_weight=cfg["exploration_weight"],
            temperature=cfg["temperature_init"],
            device=cfg["device"],
            add_dirichlet_noise=cfg["add_dirichlet_noise"],
            dirichlet_alpha=cfg["dirichlet_alpha"],
            dirichlet_epsilon=cfg["dirichlet_epsilon"],
            batch_K=cfg["batch_leaves"],
            virtual_loss=cfg["virtual_loss"],
            seed=worker_seed,
            verbose=bool(cfg.get("mcts_verbose", False)),
            inference_backend=cfg.get("inference_backend", "graph"),
            torchscript_path=cfg.get("torchscript_path"),
            torchscript_dtype=cfg.get("torchscript_dtype"),
            inference_batch_size=cfg.get("inference_batch_size", 512),
            inference_warmup_iters=cfg.get("inference_warmup_iters", 5),
        )

        games: List[Tuple[List[GameState], List[np.ndarray], List[List[dict]], float, float]] = []
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
                    virtual_loss=cfg["virtual_loss"],
                    opening_random_moves=cfg["opening_random_moves"],
                    resign_threshold=cfg["resign_threshold"],
                    resign_min_moves=cfg["resign_min_moves"],
                    resign_consecutive=cfg["resign_consecutive"],
                    seed=game_seed,
                    verbose=bool(cfg.get("verbose", False)) and worker_id == 0,
                    mcts_verbose=bool(cfg.get("mcts_verbose", False)),
                    soft_value_k=cfg["soft_value_k"],
                    inference_backend=cfg.get("inference_backend", "graph"),
                    torchscript_path=cfg.get("torchscript_path"),
                    torchscript_dtype=cfg.get("torchscript_dtype"),
                    inference_batch_size=cfg.get("inference_batch_size", 512),
                    inference_warmup_iters=cfg.get("inference_warmup_iters", 5),
                    eval_stats_sink=_sink if eval_stats_enabled else None,
                    mcts_instance=mcts_obj,
                )
            )

        stats_raw = None
        if eval_stats_enabled and worker_stats is not None:
            stats_raw = worker_stats
            payload = _format_eval_stats(
                worker_stats,
                int(cfg.get("inference_batch_size", 512)),
                int(cfg.get("batch_leaves", 0)),
            )
            payload["scope"] = "worker"
            payload["worker_id"] = worker_id
            payload["games"] = cfg["games_per_worker"]
            _print_eval_stats(payload)

        return_queue.put(("ok", worker_id, games, stats_raw))
    except Exception as exc:
        return_queue.put(("err", worker_id, repr(exc), None))
    finally:
        # --- KEY FIX: explicit cleanup to release VRAM ---
        # Python GC does not reliably destroy C++ extension objects (CUDA
        # graphs, TorchScript modules) promptly.  Explicit deletion followed
        # by gc.collect + empty_cache ensures the GPU memory is returned.
        if mcts_obj is not None:
            mcts_obj.reset()
            del mcts_obj
        if model is not None:
            del model
        _cleanup_cuda_memory(device)


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
    virtual_loss: float,
    num_workers: int,
    games_per_worker: Optional[int],
    base_seed: Optional[int],
    soft_value_k: float,
    mcts_verbose: bool,
    verbose: bool,
    opening_random_moves: int = 0,
    resign_threshold: float = -0.8,
    resign_min_moves: int = 10,
    resign_consecutive: int = 3,
    inference_backend: str = "graph",
    torchscript_path: Optional[str] = None,
    torchscript_dtype: Optional[str] = None,
    inference_batch_size: int = 512,
    inference_warmup_iters: int = 5,
    devices: Optional[Sequence[str]] = None,
) -> List[Tuple[List[GameState], List[np.ndarray], List[List[dict]], float, float]]:
    model.eval()

    devices_list = _normalize_device_list(devices, str(device))
    single_device = devices_list[0]
    try:
        single_device_obj = torch.device(single_device)
    except (TypeError, ValueError):
        single_device_obj = torch.device("cpu")

    backend = str(inference_backend).lower()
    if backend != "py" and not torchscript_path:
        device_types = set()
        for dev in devices_list:
            try:
                device_obj = torch.device(dev)
                device_types.add(device_obj.type)
            except (TypeError, ValueError):
                device_types.add("unknown")
        shared_torchscript = len(device_types) == 1 and "unknown" not in device_types

        if shared_torchscript:
            device_obj = torch.device(single_device)
            dtype_str = torchscript_dtype or _default_torchscript_dtype(device_obj)
            if dtype_str.strip().lower() in ("auto", "none"):
                dtype_str = _default_torchscript_dtype(device_obj)
            torchscript_path = _export_torchscript(
                model=model,
                device=device_obj,
                dtype_str=dtype_str,
                batch_size=max(1, int(inference_batch_size)),
            )
            torchscript_dtype = dtype_str
        else:
            torchscript_path = None
            if torchscript_dtype is not None and torchscript_dtype.strip().lower() in ("auto", "none"):
                torchscript_dtype = None

    eval_stats_enabled = _eval_stats_enabled()

    if num_workers <= 1:
        rng = random.Random(base_seed or int(time.time() * 1e6))
        games: List[Tuple[List[GameState], List[np.ndarray], List[List[dict]], float, float]] = []
        run_stats = _empty_eval_stats() if eval_stats_enabled else None

        def _sink(stats: Dict[str, object]) -> None:
            if run_stats is None:
                return
            _merge_eval_stats(run_stats, stats)

        # Create MCTS once and reuse across all games (single-process mode).
        single_mcts = FastMCTS(
            model=model,
            num_simulations=mcts_simulations,
            exploration_weight=exploration_weight,
            temperature=temperature_init,
            device=single_device,
            add_dirichlet_noise=add_dirichlet_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            batch_K=batch_leaves,
            virtual_loss=virtual_loss,
            seed=rng.randint(0, 2**31 - 1),
            verbose=mcts_verbose,
            inference_backend=inference_backend,
            torchscript_path=torchscript_path,
            torchscript_dtype=torchscript_dtype,
            inference_batch_size=inference_batch_size,
            inference_warmup_iters=inference_warmup_iters,
        )

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
                    device=single_device,
                    add_dirichlet_noise=add_dirichlet_noise,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_epsilon=dirichlet_epsilon,
                    batch_leaves=batch_leaves,
                    virtual_loss=virtual_loss,
                    opening_random_moves=opening_random_moves,
                    resign_threshold=resign_threshold,
                    resign_min_moves=resign_min_moves,
                    resign_consecutive=resign_consecutive,
                    seed=seed,
                    verbose=verbose,
                    mcts_verbose=mcts_verbose,
                    soft_value_k=soft_value_k,
                    inference_backend=inference_backend,
                    torchscript_path=torchscript_path,
                    torchscript_dtype=torchscript_dtype,
                    inference_batch_size=inference_batch_size,
                    inference_warmup_iters=inference_warmup_iters,
                    eval_stats_sink=_sink if eval_stats_enabled else None,
                    mcts_instance=single_mcts,
                )
            )
        # Cleanup: release MCTS / InferenceEngine VRAM
        single_mcts.reset()
        del single_mcts
        _cleanup_cuda_memory(single_device_obj)

        if eval_stats_enabled and run_stats is not None:
            payload = _format_eval_stats(run_stats, inference_batch_size, batch_leaves)
            payload["scope"] = "run"
            payload["games"] = len(games)
            _print_eval_stats(payload)
        return games

    if games_per_worker is None or games_per_worker <= 0:
        raise ValueError("games_per_worker must be provided when num_workers > 1.")

    # --- KEY FIX: limit concurrent processes per GPU to avoid OOM ---
    # Each process loads its own model + creates an InferenceEngine (CUDA
    # Graph) which pins substantial VRAM.  Spawning all workers at once
    # (e.g. 200 on 4 GPUs = 50 / GPU) easily exhausts GPU memory.
    #
    # We cap the number of simultaneously active processes.  The env var
    # V0_MAX_CONCURRENT_WORKERS allows manual tuning; the default is
    # 2 × number-of-devices which is conservative but safe.
    _max_env = os.environ.get("V0_MAX_CONCURRENT_WORKERS", "")
    if _max_env:
        try:
            max_concurrent = max(1, int(_max_env))
        except (TypeError, ValueError):
            max_concurrent = len(devices_list) * 2
    else:
        max_concurrent = len(devices_list) * 2
    max_concurrent = min(max_concurrent, num_workers)

    ctx = mp.get_context("spawn")
    return_queue: "mp.Queue" = ctx.Queue()
    model_bytes = _serialize_model_state(model)
    active_workers: List[mp.Process] = []
    games: List[Tuple[List[GameState], List[np.ndarray], List[List[dict]], float, float]] = []
    summary_stats = _empty_eval_stats() if eval_stats_enabled else None

    common_cfg = {
        "mcts_simulations": mcts_simulations,
        "temperature_init": temperature_init,
        "temperature_final": temperature_final,
        "temperature_threshold": temperature_threshold,
        "exploration_weight": exploration_weight,
        "add_dirichlet_noise": add_dirichlet_noise,
        "dirichlet_alpha": dirichlet_alpha,
        "dirichlet_epsilon": dirichlet_epsilon,
        "batch_leaves": batch_leaves,
        "virtual_loss": virtual_loss,
        "opening_random_moves": opening_random_moves,
        "resign_threshold": resign_threshold,
        "resign_min_moves": resign_min_moves,
        "resign_consecutive": resign_consecutive,
        "soft_value_k": soft_value_k,
        "verbose": verbose,
        "mcts_verbose": mcts_verbose,
        "num_workers": num_workers,
        "num_input_channels": NUM_INPUT_CHANNELS,
        "games_per_worker": games_per_worker,
        "inference_backend": inference_backend,
        "torchscript_path": torchscript_path,
        "torchscript_dtype": torchscript_dtype,
        "inference_batch_size": inference_batch_size,
        "inference_warmup_iters": inference_warmup_iters,
    }

    base_seed = base_seed or random.randint(1, 10**9)

    if max_concurrent < num_workers:
        print(
            f"[self_play_v0] Limiting concurrent workers to {max_concurrent} "
            f"(total {num_workers}, {len(devices_list)} GPU(s)). "
            f"Set V0_MAX_CONCURRENT_WORKERS to override."
        )

    try:
        launched = 0
        finished = 0
        while finished < num_workers:
            # Launch workers up to the concurrency cap.
            while launched < num_workers and (launched - finished) < max_concurrent:
                worker_id = launched
                worker_cfg = dict(common_cfg)
                worker_cfg["device"] = devices_list[worker_id % len(devices_list)]
                worker_cfg["worker_seed"] = base_seed + 1000003 * worker_id
                process = ctx.Process(
                    target=_v0_self_play_worker,
                    args=(worker_id, worker_cfg, model_bytes, return_queue),
                    daemon=False,
                )
                process.start()
                active_workers.append(process)
                launched += 1

            # Collect one result – this blocks until a worker finishes,
            # making room for the next one to be launched.
            status, wid, payload, stats_raw = return_queue.get()
            if status == "ok":
                games.extend(payload)
                if eval_stats_enabled and summary_stats is not None and stats_raw:
                    _merge_eval_stats(summary_stats, stats_raw)
                finished += 1
            else:
                raise RuntimeError(f"Self-play worker {wid} failed: {payload}")

            # Reap completed processes so we don't leak zombie PIDs.
            still_alive: List[mp.Process] = []
            for proc in active_workers:
                if proc.is_alive():
                    still_alive.append(proc)
                else:
                    proc.join(timeout=0.5)
            active_workers = still_alive
    finally:
        for process in active_workers:
            if process.is_alive():
                process.join(timeout=1.0)
                if process.is_alive():
                    process.terminate()
        return_queue.close()

    if len(games) != num_workers * games_per_worker:
        print(
            f"Warning: expected {num_workers * games_per_worker} games but received {len(games)}."
        )

    if eval_stats_enabled and summary_stats is not None:
        payload = _format_eval_stats(summary_stats, inference_batch_size, batch_leaves)
        payload["scope"] = "summary"
        payload["games"] = len(games)
        payload["num_workers"] = num_workers
        _print_eval_stats(payload)

    return games[:num_games]
