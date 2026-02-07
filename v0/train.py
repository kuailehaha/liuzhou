"""Training pipeline that wires the v0 CUDA/C++ self-play core into the
AlphaZero-style loop.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.game_state import GameState, Player
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from src.evaluate import (
    MCTSAgent,
    V0MCTSAgent,
    evaluate_against_agent,
    evaluate_against_agent_parallel,
    evaluate_against_agent_parallel_v0,
)
from src.random_agent import RandomAgent

from v0.python.self_play_runner import self_play_v0
from v0.python.state_io import (
    flatten_training_games,
    load_examples_from_files,
    sample_to_record,
    write_records_to_jsonl,
)

import src.train as baseline_train

SampleTuple = Tuple[GameState, np.ndarray, float, float]


def _normalize_eval_backend(value: Optional[str]) -> str:
    key = str(value or "legacy").strip().lower()
    if key in ("legacy", "python", "py"):
        return "legacy"
    if key in ("v0", "cpp", "c++", "cuda"):
        return "v0"
    raise ValueError(f"Unsupported eval backend: {value}")


def _coerce_device_list(value: Optional[object], fallback: str) -> List[str]:
    if value is None:
        return [fallback]
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value if str(item).strip()]
    else:
        text = str(value).strip()
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

    devices: List[str] = []
    for token in tokens:
        lower = token.lower()
        if lower == "cuda":
            devices.append("cuda:0")
        elif lower in ("cpu", "mps"):
            devices.append(lower)
        elif lower.startswith(("cuda:", "mps", "cpu")):
            devices.append(token)
        elif token.isdigit():
            devices.append(f"cuda:{token}")
        else:
            devices.append(token)

    seen = set()
    normalized: List[str] = []
    for dev in devices:
        if dev not in seen:
            normalized.append(dev)
            seen.add(dev)
    return normalized or [fallback]


def _cuda_device_ids(devices: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for dev in devices:
        try:
            device_obj = torch.device(dev)
        except (TypeError, ValueError):
            continue
        if device_obj.type != "cuda":
            continue
        idx = 0 if device_obj.index is None else int(device_obj.index)
        ids.append(idx)
    seen = set()
    unique_ids: List[int] = []
    for idx in ids:
        if idx not in seen:
            unique_ids.append(idx)
            seen.add(idx)
    return unique_ids


def train_pipeline_v0(
    iterations: int = 10,
    num_mcts_simulations: int = 800,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    temperature_init: float = 1.0,
    temperature_final: float = 0.1,
    temperature_threshold: int = 10,
    exploration_weight: float = 1.0,
    self_play_workers: int = 1,
    self_play_games_per_worker: Optional[int] = None,
    self_play_base_seed: Optional[int] = None,
    self_play_virtual_loss_weight: float = 1.0,
    self_play_opening_random_moves: int = 4,
    self_play_resign_threshold: float = -0.8,
    self_play_resign_min_moves: int = 10,
    self_play_resign_consecutive: int = 3,
    decisive_only: bool = False,
    value_draw_weight: float = 0.1,
    policy_draw_weight: float = 0.3,
    eval_games_vs_random: int = 20,
    eval_games_vs_best: int = 20,
    eval_games_vs_previous: int = 0,
    win_rate_threshold: float = 0.55,
    promotion_vs_opponent_threshold: float = 0.5,
    mcts_sims_eval: int = 100,
    eval_workers: int = 0,
    eval_backend: str = "legacy",
    checkpoint_dir: str = "./checkpoints_v0",
    device: str = "cpu",
    self_play_devices: Optional[Sequence[str]] = None,
    train_devices: Optional[Sequence[str]] = None,
    eval_devices: Optional[Sequence[str]] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    self_play_batch_leaves: int = 256,
    self_play_dirichlet_alpha: float = 0.3,
    self_play_dirichlet_epsilon: float = 0.25,
    self_play_inference_backend: str = "graph",
    self_play_torchscript_path: Optional[str] = None,
    self_play_torchscript_dtype: Optional[str] = None,
    self_play_inference_batch_size: int = 512,
    self_play_inference_warmup_iters: int = 5,
    data_files: Optional[Sequence[str]] = None,
    data_samples_per_iteration: Optional[int] = None,
    data_shuffle: bool = False,
    save_self_play_dir: Optional[str] = None,
    load_checkpoint: Optional[str] = None,
) -> None:
    """Complete v0 training pipeline (self-play -> training -> evaluation)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if save_self_play_dir:
        os.makedirs(save_self_play_dir, exist_ok=True)

    metrics: List[Dict[str, Any]] = []
    metrics_path = os.path.join(checkpoint_dir, "training_metrics.json")
    stage_history: Dict[str, List[float]] = {"self_play": [], "train": [], "eval": []}

    runtime_config = runtime_config or {}
    verbosity_cfg = runtime_config.get("verbosity", {})
    self_play_verbose = bool(verbosity_cfg.get("self_play", False))
    self_play_mcts_verbose = bool(verbosity_cfg.get("self_play_mcts", False))
    eval_verbose = bool(verbosity_cfg.get("eval", False))
    eval_game_verbose = bool(verbosity_cfg.get("eval_game", False))
    eval_mcts_verbose = bool(verbosity_cfg.get("eval_mcts", False))

    self_play_cfg = runtime_config.get("self_play", {})
    self_play_add_dirichlet = self_play_cfg.get("add_dirichlet_noise", True)
    parallel_cfg = runtime_config.get("self_play_parallel", {})

    data_files = list(data_files or [])
    if data_samples_per_iteration is not None and data_samples_per_iteration <= 0:
        raise ValueError("data_samples_per_iteration must be greater than zero when provided.")
    offline_mode = len(data_files) > 0

    sp_workers_cfg = parallel_cfg.get("workers", self_play_workers)
    try:
        sp_workers = int(sp_workers_cfg)
    except (TypeError, ValueError):
        sp_workers = self_play_workers
    sp_workers = max(1, sp_workers)

    sp_gpw_cfg = parallel_cfg.get("games_per_worker", self_play_games_per_worker)
    try:
        sp_games_per_worker = int(sp_gpw_cfg) if sp_gpw_cfg is not None else None
    except (TypeError, ValueError):
        sp_games_per_worker = None

    if offline_mode:
        total_self_play_games = 0
    else:
        if sp_games_per_worker is None or sp_games_per_worker <= 0:
            raise ValueError("self_play_games_per_worker must be provided and greater than zero.")
        total_self_play_games = sp_games_per_worker * sp_workers

    sp_seed_cfg = parallel_cfg.get("base_seed", self_play_base_seed)
    try:
        sp_base_seed = int(sp_seed_cfg) if sp_seed_cfg is not None else None
    except (TypeError, ValueError):
        sp_base_seed = None
    if sp_base_seed == 0:
        sp_base_seed = None

    sp_virtual_loss_cfg = parallel_cfg.get("virtual_loss_weight", self_play_virtual_loss_weight)
    try:
        sp_virtual_loss = float(sp_virtual_loss_cfg)
    except (TypeError, ValueError):
        sp_virtual_loss = float(self_play_virtual_loss_weight)
    if sp_virtual_loss < 0:
        sp_virtual_loss = 0.0

    sp_opening_cfg = self_play_cfg.get("opening_random_moves", self_play_opening_random_moves)
    try:
        sp_opening_random_moves = int(sp_opening_cfg)
    except (TypeError, ValueError):
        sp_opening_random_moves = int(self_play_opening_random_moves)
    sp_opening_random_moves = max(0, sp_opening_random_moves)

    sp_resign_threshold_cfg = self_play_cfg.get("resign_threshold", self_play_resign_threshold)
    sp_resign_min_cfg = self_play_cfg.get("resign_min_moves", self_play_resign_min_moves)
    sp_resign_consecutive_cfg = self_play_cfg.get("resign_consecutive", self_play_resign_consecutive)
    try:
        sp_resign_threshold = float(sp_resign_threshold_cfg)
    except (TypeError, ValueError):
        sp_resign_threshold = float(self_play_resign_threshold)
    try:
        sp_resign_min_moves = int(sp_resign_min_cfg)
    except (TypeError, ValueError):
        sp_resign_min_moves = int(self_play_resign_min_moves)
    try:
        sp_resign_consecutive = int(sp_resign_consecutive_cfg)
    except (TypeError, ValueError):
        sp_resign_consecutive = int(self_play_resign_consecutive)
    sp_resign_min_moves = max(0, sp_resign_min_moves)
    sp_resign_consecutive = max(1, sp_resign_consecutive)

    try:
        value_draw_weight = float(value_draw_weight)
    except (TypeError, ValueError):
        value_draw_weight = 1.0
    if value_draw_weight < 0:
        value_draw_weight = 0.0

    try:
        policy_draw_weight = float(policy_draw_weight)
    except (TypeError, ValueError):
        policy_draw_weight = 1.0
    if policy_draw_weight < 0:
        policy_draw_weight = 0.0

    evaluation_cfg = runtime_config.get("evaluation", {})
    eval_temperature = evaluation_cfg.get("temperature", 0.05)
    eval_add_dirichlet = evaluation_cfg.get("add_dirichlet_noise", False)
    if "mcts_simulations" in evaluation_cfg:
        mcts_sims_eval = evaluation_cfg["mcts_simulations"]
    eval_games_vs_random = evaluation_cfg.get("games_vs_random", eval_games_vs_random)
    eval_games_vs_best = evaluation_cfg.get("games_vs_best", eval_games_vs_best)
    eval_games_vs_previous = evaluation_cfg.get("games_vs_previous", eval_games_vs_previous)
    promotion_vs_opponent_threshold = float(
        evaluation_cfg.get("promotion_vs_opponent_threshold", promotion_vs_opponent_threshold)
    )
    eval_workers_cfg = evaluation_cfg.get("workers", evaluation_cfg.get("num_workers", eval_workers))
    try:
        eval_workers = int(eval_workers_cfg) if eval_workers_cfg is not None else 0
    except (TypeError, ValueError):
        eval_workers = 0
    if eval_workers <= 0:
        eval_workers = sp_workers
    eval_workers = max(1, eval_workers)

    device = str(device)
    fallback_device = "cuda:0" if device == "cuda" else device
    train_cfg = runtime_config.get("train", {})
    self_play_devices_list = _coerce_device_list(self_play_cfg.get("devices", self_play_devices), fallback_device)
    train_devices_list = _coerce_device_list(train_cfg.get("devices", train_devices), fallback_device)
    eval_devices_list = _coerce_device_list(evaluation_cfg.get("devices", eval_devices), fallback_device)

    self_play_device = self_play_devices_list[0]
    train_device = train_devices_list[0]
    eval_device = eval_devices_list[0]
    train_device_ids = _cuda_device_ids(train_devices_list)
    use_data_parallel = torch.cuda.is_available() and len(train_device_ids) > 1

    if use_data_parallel:
        print(f"DataParallel enabled on devices: {train_devices_list}")

    soft_value_cfg = runtime_config.get("soft_value_labels", {})
    try:
        soft_value_k = float(soft_value_cfg.get("k", 2.0))
    except (TypeError, ValueError):
        soft_value_k = 2.0
    try:
        soft_alpha_start = float(soft_value_cfg.get("alpha_start", 0.3))
    except (TypeError, ValueError):
        soft_alpha_start = 0.3
    try:
        soft_alpha_end = float(soft_value_cfg.get("alpha_end", 0.0))
    except (TypeError, ValueError):
        soft_alpha_end = 0.0
    try:
        soft_alpha_anneal_iters = int(soft_value_cfg.get("anneal_iterations", 20))
    except (TypeError, ValueError):
        soft_alpha_anneal_iters = 20
    soft_alpha_start = max(0.0, min(1.0, soft_alpha_start))
    soft_alpha_end = max(0.0, min(1.0, soft_alpha_end))
    soft_alpha_anneal_iters = max(0, soft_alpha_anneal_iters)

    sp_batch_cfg = self_play_cfg.get("batch_leaves", self_play_batch_leaves)
    try:
        sp_batch_leaves = int(sp_batch_cfg)
    except (TypeError, ValueError):
        sp_batch_leaves = self_play_batch_leaves
    sp_batch_leaves = max(1, sp_batch_leaves)

    sp_backend_cfg = self_play_cfg.get("inference_backend", self_play_inference_backend)
    try:
        sp_inference_backend = str(sp_backend_cfg) if sp_backend_cfg is not None else self_play_inference_backend
    except (TypeError, ValueError):
        sp_inference_backend = self_play_inference_backend

    sp_torchscript_path = self_play_cfg.get("torchscript_path", self_play_torchscript_path)
    if sp_torchscript_path is not None:
        sp_torchscript_path = str(sp_torchscript_path)

    sp_torchscript_dtype = self_play_cfg.get("torchscript_dtype", self_play_torchscript_dtype)
    if sp_torchscript_dtype is not None:
        sp_torchscript_dtype = str(sp_torchscript_dtype)

    sp_batch_size_cfg = self_play_cfg.get("inference_batch_size", self_play_inference_batch_size)
    try:
        sp_inference_batch_size = int(sp_batch_size_cfg)
    except (TypeError, ValueError):
        sp_inference_batch_size = self_play_inference_batch_size
    sp_inference_batch_size = max(1, sp_inference_batch_size)

    sp_warmup_cfg = self_play_cfg.get("inference_warmup_iters", self_play_inference_warmup_iters)
    try:
        sp_inference_warmup_iters = int(sp_warmup_cfg)
    except (TypeError, ValueError):
        sp_inference_warmup_iters = self_play_inference_warmup_iters
    sp_inference_warmup_iters = max(0, sp_inference_warmup_iters)

    eval_backend = _normalize_eval_backend(evaluation_cfg.get("backend", eval_backend))

    eval_batch_cfg = evaluation_cfg.get("batch_leaves", sp_batch_leaves)
    try:
        eval_batch_leaves = int(eval_batch_cfg)
    except (TypeError, ValueError):
        eval_batch_leaves = sp_batch_leaves
    eval_batch_leaves = max(1, eval_batch_leaves)

    eval_inference_backend_cfg = evaluation_cfg.get("inference_backend", sp_inference_backend)
    try:
        eval_inference_backend = (
            str(eval_inference_backend_cfg)
            if eval_inference_backend_cfg is not None
            else sp_inference_backend
        )
    except (TypeError, ValueError):
        eval_inference_backend = sp_inference_backend

    eval_torchscript_path = evaluation_cfg.get("torchscript_path", None)
    if eval_torchscript_path is not None:
        eval_torchscript_path = str(eval_torchscript_path)

    eval_torchscript_dtype = evaluation_cfg.get("torchscript_dtype", sp_torchscript_dtype)
    if eval_torchscript_dtype is not None:
        eval_torchscript_dtype = str(eval_torchscript_dtype)

    eval_batch_size_cfg = evaluation_cfg.get("inference_batch_size", sp_inference_batch_size)
    try:
        eval_inference_batch_size = int(eval_batch_size_cfg)
    except (TypeError, ValueError):
        eval_inference_batch_size = sp_inference_batch_size
    eval_inference_batch_size = max(1, eval_inference_batch_size)

    eval_warmup_cfg = evaluation_cfg.get("inference_warmup_iters", sp_inference_warmup_iters)
    try:
        eval_inference_warmup_iters = int(eval_warmup_cfg)
    except (TypeError, ValueError):
        eval_inference_warmup_iters = sp_inference_warmup_iters
    eval_inference_warmup_iters = max(0, eval_inference_warmup_iters)

    dir_alpha_cfg = self_play_cfg.get("dirichlet_alpha", self_play_dirichlet_alpha)
    dir_eps_cfg = self_play_cfg.get("dirichlet_epsilon", self_play_dirichlet_epsilon)
    try:
        sp_dirichlet_alpha = float(dir_alpha_cfg)
    except (TypeError, ValueError):
        sp_dirichlet_alpha = self_play_dirichlet_alpha
    try:
        sp_dirichlet_epsilon = float(dir_eps_cfg)
    except (TypeError, ValueError):
        sp_dirichlet_epsilon = self_play_dirichlet_epsilon

    data_files = [os.path.abspath(path) for path in data_files]
    offline_examples_all: List[SampleTuple] = []
    offline_cursor = 0
    if offline_mode:
        print(f"Offline training mode: loading samples from {len(data_files)} file(s).")
        offline_examples_all = load_examples_from_files(data_files)
        if not offline_examples_all:
            raise ValueError("No samples were loaded from the provided data_files.")
        print(f"Loaded {len(offline_examples_all)} samples from disk.")

    def _next_offline_batch() -> List[SampleTuple]:
        nonlocal offline_cursor
        if data_shuffle:
            shuffled = offline_examples_all[:]
            random.shuffle(shuffled)
            if data_samples_per_iteration is None:
                return shuffled
            return list(shuffled[: data_samples_per_iteration])
        if not offline_examples_all:
            return []
        if data_samples_per_iteration is None or data_samples_per_iteration >= len(offline_examples_all):
            return list(offline_examples_all)
        total = len(offline_examples_all)
        start = offline_cursor
        end = start + data_samples_per_iteration
        if end <= total:
            batch = offline_examples_all[start:end]
        else:
            batch = offline_examples_all[start:] + offline_examples_all[: end - total]
        offline_cursor = end % total
        return list(batch)

    def _cleanup_gpu():
        """Release unreferenced CUDA memory between pipeline phases.

        This is essential to prevent VRAM accumulation across self-play,
        training and evaluation phases.  C++ extension objects (CUDA
        graphs, TorchScript modules, MCTS trees) may not be promptly
        freed by Python's reference-counting GC; an explicit cycle
        collection followed by ``empty_cache`` reclaims the memory.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    board_size = GameState.BOARD_SIZE
    current_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
    
    # Load checkpoint if provided
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"Loading model checkpoint from: {load_checkpoint}")
        checkpoint = torch.load(load_checkpoint, map_location=train_device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        current_model.load_state_dict(state_dict)
        loaded_iteration = checkpoint.get("iteration", 0)
        print(f"Loaded model from iteration {loaded_iteration}")
    else:
        if load_checkpoint:
            print(f"Warning: Checkpoint not found at {load_checkpoint}, starting with random model.")
        else:
            print("No checkpoint specified, starting with random model.")
    
    current_model.to(train_device)

    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    for iteration in range(iterations):
        print(f"\n{'=' * 20} Iteration {iteration + 1}/{iterations} {'=' * 20}")

        iter_start_time = time.perf_counter()
        iteration_metrics: Dict[str, Any] = {
                "iteration": iteration + 1,
                "self_play_games_requested": (total_self_play_games if not offline_mode else None),
                "mcts_simulations": num_mcts_simulations,
                "epochs_requested": epochs,
                "batch_size": batch_size,
                "timestamp_start": time.time(),
                "self_play_workers": sp_workers,
                "self_play_games_per_worker": sp_games_per_worker,
                "self_play_virtual_loss": sp_virtual_loss,
                "self_play_base_seed": sp_base_seed,
                "self_play_batch_leaves": sp_batch_leaves,
                "self_play_dirichlet_alpha": sp_dirichlet_alpha,
                "self_play_dirichlet_epsilon": sp_dirichlet_epsilon,
                "self_play_opening_random_moves": sp_opening_random_moves,
                "self_play_resign_threshold": sp_resign_threshold,
                "self_play_resign_min_moves": sp_resign_min_moves,
                "self_play_resign_consecutive": sp_resign_consecutive,
                "decisive_only": bool(decisive_only),
                "value_draw_weight": value_draw_weight,
                "policy_draw_weight": policy_draw_weight,
                "eval_backend": eval_backend,
                "self_play_devices": list(self_play_devices_list),
                "train_devices": list(train_devices_list),
                "eval_devices": list(eval_devices_list),
                "train_data_parallel": use_data_parallel,
            }

        if soft_alpha_anneal_iters <= 0:
            current_alpha = soft_alpha_start
        else:
            progress = min(iteration / max(1, soft_alpha_anneal_iters), 1.0)
            cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_alpha = soft_alpha_end + (soft_alpha_start - soft_alpha_end) * cosine_weight
        current_alpha = max(0.0, min(1.0, current_alpha))
        iteration_metrics["soft_label_alpha"] = current_alpha
        iteration_metrics["soft_value_k"] = soft_value_k

        print(
            f"Soft label alpha (value mix): {current_alpha:.3f}, "
            f"soft_value_k={soft_value_k:.2f}, "
            f"self-play batch_leaves={sp_batch_leaves}"
        )

        examples: List[SampleTuple] = []
        total_positions = 0
        train_metrics_data: Dict[str, Any] = {"epoch_stats": []}

        if offline_mode:
            data_label = "Offline data phase"
            baseline_train._stage_banner("self_play", data_label, iteration, iterations, stage_history)
            load_start_time = time.perf_counter()
            examples = _next_offline_batch()
            if decisive_only:
                before = len(examples)
                examples = [ex for ex in examples if abs(ex[3]) > 0.0]
                dropped = before - len(examples)
                iteration_metrics["decisive_samples_dropped"] = dropped
            self_play_time = baseline_train._stage_finish(
                "self_play", data_label, load_start_time, stage_history
            )
            iteration_metrics["self_play_time_sec"] = self_play_time
            iteration_metrics["self_play_games_played"] = None
            iteration_metrics["data_source"] = "offline"
            total_positions = len(examples)
            num_games_generated = None
        else:
            self_play_label = "Self-play phase"
            baseline_train._stage_banner("self_play", self_play_label, iteration, iterations, stage_history)
            print(f"{self_play_label}: generating {total_self_play_games} games using current model...")
            current_model.eval()
            sp_start_time = time.perf_counter()
            training_data = self_play_v0(
                model=current_model,
                num_games=total_self_play_games,
                mcts_simulations=num_mcts_simulations,
                temperature_init=temperature_init,
                temperature_final=temperature_final,
                temperature_threshold=temperature_threshold,
                exploration_weight=exploration_weight,
                device=self_play_device,
                add_dirichlet_noise=self_play_add_dirichlet,
                dirichlet_alpha=sp_dirichlet_alpha,
                dirichlet_epsilon=sp_dirichlet_epsilon,
                batch_leaves=sp_batch_leaves,
                virtual_loss=sp_virtual_loss,
                opening_random_moves=sp_opening_random_moves,
                resign_threshold=sp_resign_threshold,
                resign_min_moves=sp_resign_min_moves,
                resign_consecutive=sp_resign_consecutive,
                num_workers=sp_workers,
                games_per_worker=sp_games_per_worker,
                base_seed=sp_base_seed,
                soft_value_k=soft_value_k,
                mcts_verbose=self_play_mcts_verbose,
                verbose=self_play_verbose,
                inference_backend=sp_inference_backend,
                torchscript_path=sp_torchscript_path,
                torchscript_dtype=sp_torchscript_dtype,
                inference_batch_size=sp_inference_batch_size,
                inference_warmup_iters=sp_inference_warmup_iters,
                devices=self_play_devices_list,
            )
            training_data = training_data or []
            sp_wins = sum(1 for g in training_data if g[3] > 0)
            sp_losses = sum(1 for g in training_data if g[3] < 0)
            sp_draws = sum(1 for g in training_data if g[3] == 0)
            iteration_metrics["self_play_wins"] = sp_wins
            iteration_metrics["self_play_losses"] = sp_losses
            iteration_metrics["self_play_draws"] = sp_draws
            if training_data:
                print(
                    f"Self-play outcomes: {sp_wins} wins, {sp_losses} losses, {sp_draws} draws "
                    f"(draw rate {sp_draws / len(training_data):.1%})"
                )
            if decisive_only:
                before_games = len(training_data)
                # Result is at index 3 in new 5-tuple: (states, policies, legal_moves, result, soft)
                training_data = [game for game in training_data if game[3] != 0.0]
                iteration_metrics["decisive_games_dropped"] = before_games - len(training_data)
            num_games_generated = len(training_data)
            flattened_samples = list(flatten_training_games(training_data))
            examples = flattened_samples
            total_positions = len(examples)
            self_play_time = baseline_train._stage_finish(
                "self_play", self_play_label, sp_start_time, stage_history
            )
            iteration_metrics["self_play_time_sec"] = self_play_time
            iteration_metrics["self_play_games_played"] = num_games_generated
            iteration_metrics["data_source"] = "self_play"

            if save_self_play_dir and examples:
                dump_path = os.path.join(
                    save_self_play_dir, f"self_play_iter_{iteration + 1:03d}.jsonl"
                )
                record_iter = (sample_to_record(*sample) for sample in examples)
                saved = write_records_to_jsonl(record_iter, dump_path)
                print(f"Saved {saved} samples to {dump_path}")

        # Release self-play VRAM (inference engines, CUDA graphs, etc.)
        # before the training phase allocates its own GPU memory.
        _cleanup_gpu()

        train_label = "Training phase"
        baseline_train._stage_banner("train", train_label, iteration, iterations, stage_history)
        train_start_time = time.perf_counter()

        if not examples:
            print("No training data available for this iteration. Skipping training.")
            train_time = baseline_train._stage_finish(
                "train", train_label, train_start_time, stage_history
            )
        else:
            print(f"{train_label}: {len(examples)} examples, {epochs} epochs...")
            current_model.to(train_device)
            current_model.train()
            train_model = current_model
            if use_data_parallel:
                train_model = torch.nn.DataParallel(
                    current_model,
                    device_ids=train_device_ids,
                    output_device=train_device_ids[0],
                )
            train_num_workers = train_cfg.get("num_workers")
            train_prefetch_factor = train_cfg.get("prefetch_factor")
            train_model, train_metrics_data = baseline_train.train_network(
                model=train_model,
                examples=examples,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                soft_label_alpha=current_alpha,
                value_draw_weight=value_draw_weight,
                policy_draw_weight=policy_draw_weight,
                device=train_device,
                board_size=board_size,
                num_workers=train_num_workers,
                prefetch_factor=train_prefetch_factor,
            )
            if isinstance(train_model, torch.nn.DataParallel):
                current_model = train_model.module
            else:
                current_model = train_model
            train_time = baseline_train._stage_finish(
                "train", train_label, train_start_time, stage_history
            )

        iteration_metrics["self_play_positions"] = total_positions
        iteration_metrics["train_time_sec"] = train_time
        iteration_metrics["train_examples"] = len(examples)
        epoch_stats = list(train_metrics_data.get("epoch_stats", []))
        iteration_metrics["train_epoch_stats"] = epoch_stats
        if epoch_stats:
            last_epoch_stats = epoch_stats[-1]
            iteration_metrics["train_last_avg_loss"] = last_epoch_stats.get("avg_loss")
            iteration_metrics["train_last_avg_policy_loss"] = last_epoch_stats.get("avg_policy_loss")
            iteration_metrics["train_last_avg_value_loss"] = last_epoch_stats.get("avg_value_loss")
            iteration_metrics["train_last_soft_alpha"] = last_epoch_stats.get("soft_alpha")
            iteration_metrics["train_last_avg_soft_abs"] = last_epoch_stats.get("avg_soft_abs")
            iteration_metrics["train_last_avg_mix_abs"] = last_epoch_stats.get("avg_mix_abs")
        else:
            iteration_metrics["train_last_avg_loss"] = None
            iteration_metrics["train_last_avg_policy_loss"] = None
            iteration_metrics["train_last_avg_value_loss"] = None
            iteration_metrics["train_last_soft_alpha"] = None
            iteration_metrics["train_last_avg_soft_abs"] = None
            iteration_metrics["train_last_avg_mix_abs"] = None

        iter_model_path = os.path.join(checkpoint_dir, f"model_iter_{iteration + 1}.pt")
        torch.save(
            {
                "iteration": iteration + 1,
                "model_state_dict": current_model.state_dict(),
                "board_size": board_size,
                "num_input_channels": NUM_INPUT_CHANNELS,
            },
            iter_model_path,
        )
        print(f"Model for iteration {iteration + 1} saved to {iter_model_path}")
        iteration_metrics["checkpoint_path"] = iter_model_path

        # Release training VRAM (optimizer states, DataParallel buffers)
        # before the evaluation phase allocates its own GPU memory.
        _cleanup_gpu()

        print("\nEvaluation phase...")
        eval_label = "Evaluation phase"
        baseline_train._stage_banner("eval", eval_label, iteration, iterations, stage_history)
        current_model.to(eval_device)
        current_model.eval()
        eval_start_time = time.perf_counter()

        v0_eval_kwargs = {
            "batch_leaves": eval_batch_leaves,
            "inference_backend": eval_inference_backend,
            "torchscript_path": eval_torchscript_path,
            "torchscript_dtype": eval_torchscript_dtype,
            "inference_batch_size": eval_inference_batch_size,
            "inference_warmup_iters": eval_inference_warmup_iters,
        }

        def _make_eval_agent(model: ChessNet, sample_moves: bool = False):
            if eval_backend == "v0":
                return V0MCTSAgent(
                    model,
                    mcts_simulations=mcts_sims_eval,
                    temperature=eval_temperature,
                    device=eval_device,
                    add_dirichlet_noise=eval_add_dirichlet,
                    verbose=eval_verbose,
                    mcts_verbose=eval_mcts_verbose,
                    sample_moves=sample_moves,
                    **v0_eval_kwargs,
                )
            return MCTSAgent(
                model,
                mcts_simulations=mcts_sims_eval,
                temperature=eval_temperature,
                device=eval_device,
                add_dirichlet_noise=eval_add_dirichlet,
                verbose=eval_verbose,
                mcts_verbose=eval_mcts_verbose,
                sample_moves=sample_moves,
            )

        print(f"Evaluating challenger against RandomAgent ({eval_games_vs_random} games)...")
        if eval_workers > 1:
            if eval_backend == "v0":
                stats_vs_rnd = evaluate_against_agent_parallel_v0(
                    challenger_checkpoint=iter_model_path,
                    opponent_checkpoint=None,
                    num_games=eval_games_vs_random,
                    device=eval_device,
                    mcts_simulations=mcts_sims_eval,
                    temperature=eval_temperature,
                    add_dirichlet_noise=eval_add_dirichlet,
                    num_workers=eval_workers,
                    devices=eval_devices_list,
                    verbose=eval_verbose,
                    game_verbose=eval_game_verbose,
                    mcts_verbose=eval_mcts_verbose,
                    **v0_eval_kwargs,
                )
            else:
                stats_vs_rnd = evaluate_against_agent_parallel(
                    challenger_checkpoint=iter_model_path,
                    opponent_checkpoint=None,
                    num_games=eval_games_vs_random,
                    device=eval_device,
                    mcts_simulations=mcts_sims_eval,
                    temperature=eval_temperature,
                    add_dirichlet_noise=eval_add_dirichlet,
                    num_workers=eval_workers,
                    devices=eval_devices_list,
                    verbose=eval_verbose,
                    game_verbose=eval_game_verbose,
                    mcts_verbose=eval_mcts_verbose,
                )
        else:
            challenger_agent = _make_eval_agent(current_model)
            random_opponent = RandomAgent()
            stats_vs_rnd = evaluate_against_agent(
                challenger_agent,
                random_opponent,
                eval_games_vs_random,
                eval_device,
                verbose=eval_verbose,
                game_verbose=eval_game_verbose,
            )
            # Release eval agent MCTS to free VRAM before next eval round.
            del challenger_agent, random_opponent
            _cleanup_gpu()
        print(f"Challenger win rate vs RandomAgent: {stats_vs_rnd.win_rate:.2%}")
        print(
            "Challenger record vs RandomAgent: "
            f"{stats_vs_rnd.wins}-{stats_vs_rnd.losses}-{stats_vs_rnd.draws} "
            f"(win {stats_vs_rnd.win_rate:.2%} / loss {stats_vs_rnd.loss_rate:.2%} / draw {stats_vs_rnd.draw_rate:.2%})"
        )

        win_rate_vs_previous = None
        loss_rate_vs_previous = None
        if eval_games_vs_previous > 0 and iteration >= 1:
            previous_model_path = os.path.join(checkpoint_dir, f"model_iter_{iteration}.pt")
            if os.path.exists(previous_model_path):
                print(f"Evaluating challenger vs previous iteration ({eval_games_vs_previous} games)...")
                if eval_workers > 1:
                    if eval_backend == "v0":
                        stats_vs_prev = evaluate_against_agent_parallel_v0(
                            challenger_checkpoint=iter_model_path,
                            opponent_checkpoint=previous_model_path,
                            num_games=eval_games_vs_previous,
                            device=eval_device,
                            mcts_simulations=mcts_sims_eval,
                            temperature=eval_temperature,
                            add_dirichlet_noise=eval_add_dirichlet,
                            num_workers=eval_workers,
                            devices=eval_devices_list,
                            verbose=eval_verbose,
                            game_verbose=eval_game_verbose,
                            mcts_verbose=eval_mcts_verbose,
                            sample_moves=True,
                            **v0_eval_kwargs,
                        )
                    else:
                        stats_vs_prev = evaluate_against_agent_parallel(
                            challenger_checkpoint=iter_model_path,
                            opponent_checkpoint=previous_model_path,
                            num_games=eval_games_vs_previous,
                            device=eval_device,
                            mcts_simulations=mcts_sims_eval,
                            temperature=eval_temperature,
                            add_dirichlet_noise=eval_add_dirichlet,
                            num_workers=eval_workers,
                            devices=eval_devices_list,
                            verbose=eval_verbose,
                            game_verbose=eval_game_verbose,
                            mcts_verbose=eval_mcts_verbose,
                            sample_moves=True,
                        )
                else:
                    prev_checkpoint = torch.load(previous_model_path, map_location=eval_device)
                    prev_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
                    prev_model.load_state_dict(prev_checkpoint["model_state_dict"])
                    prev_model.to(eval_device)
                    prev_model.eval()
                    challenger_agent_prev = _make_eval_agent(current_model, sample_moves=True)
                    prev_agent = _make_eval_agent(prev_model, sample_moves=True)
                    stats_vs_prev = evaluate_against_agent(
                        challenger_agent_prev,
                        prev_agent,
                        eval_games_vs_previous,
                        eval_device,
                        verbose=eval_verbose,
                        game_verbose=eval_game_verbose,
                    )
                    del challenger_agent_prev, prev_agent, prev_model, prev_checkpoint
                    _cleanup_gpu()
                win_rate_vs_previous = stats_vs_prev.win_rate
                loss_rate_vs_previous = stats_vs_prev.loss_rate
                print(
                    f"Challenger win rate vs previous iter: {win_rate_vs_previous:.2%} "
                    f"({stats_vs_prev.wins}-{stats_vs_prev.losses}-{stats_vs_prev.draws})"
                )
            else:
                loss_rate_vs_previous = None
        else:
            loss_rate_vs_previous = None

        win_rate_vs_best_model = None
        stats_vs_best_model = None
        best_model_updated = False

        if stats_vs_rnd.win_rate >= win_rate_threshold:
            print(f"Challenger passed RandomAgent threshold (>= {win_rate_threshold:.0%}). Comparing to best model...")
            beats_previous = (
                win_rate_vs_previous is None
                or win_rate_vs_previous > (loss_rate_vs_previous or 0)
            )
            if not os.path.exists(best_model_path):
                if beats_previous:
                    print("No existing best_model.pt. Current model becomes the best.")
                    shutil.copy(iter_model_path, best_model_path)
                    print(f"Best model updated: {best_model_path}")
                    best_model_updated = True
                else:
                    print(
                        "Challenger did not beat previous iteration (win rate <= loss rate). Not promoting."
                    )
            else:
                print("Loading best_model.pt for comparison...")

                print(f"Evaluating challenger against BestModel ({eval_games_vs_best} games)...")
                if eval_workers > 1:
                    if eval_backend == "v0":
                        stats_vs_best_model = evaluate_against_agent_parallel_v0(
                            challenger_checkpoint=iter_model_path,
                            opponent_checkpoint=best_model_path,
                            num_games=eval_games_vs_best,
                            device=eval_device,
                            mcts_simulations=mcts_sims_eval,
                            temperature=eval_temperature,
                            add_dirichlet_noise=eval_add_dirichlet,
                            num_workers=eval_workers,
                            devices=eval_devices_list,
                            verbose=eval_verbose,
                            game_verbose=eval_game_verbose,
                            mcts_verbose=eval_mcts_verbose,
                            sample_moves=True,
                            **v0_eval_kwargs,
                        )
                    else:
                        stats_vs_best_model = evaluate_against_agent_parallel(
                            challenger_checkpoint=iter_model_path,
                            opponent_checkpoint=best_model_path,
                            num_games=eval_games_vs_best,
                            device=eval_device,
                            mcts_simulations=mcts_sims_eval,
                            temperature=eval_temperature,
                            add_dirichlet_noise=eval_add_dirichlet,
                            num_workers=eval_workers,
                            devices=eval_devices_list,
                            verbose=eval_verbose,
                            game_verbose=eval_game_verbose,
                            mcts_verbose=eval_mcts_verbose,
                            sample_moves=True,
                        )
                else:
                    best_model_checkpoint = torch.load(best_model_path, map_location=eval_device)
                    best_model_eval = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
                    best_model_eval.load_state_dict(best_model_checkpoint["model_state_dict"])
                    best_model_eval.to(eval_device)
                    best_model_eval.eval()

                    challenger_agent_best = _make_eval_agent(current_model, sample_moves=True)
                    best_agent_opponent = _make_eval_agent(best_model_eval, sample_moves=True)

                    stats_vs_best_model = evaluate_against_agent(
                        challenger_agent_best,
                        best_agent_opponent,
                        eval_games_vs_best,
                        eval_device,
                        verbose=eval_verbose,
                        game_verbose=eval_game_verbose,
                    )
                    # Release eval models / MCTS agents to free VRAM
                    del challenger_agent_best, best_agent_opponent
                    del best_model_eval, best_model_checkpoint
                    _cleanup_gpu()
                win_rate_vs_best_model = stats_vs_best_model.win_rate
                print(f"Challenger win rate vs BestModel: {win_rate_vs_best_model:.2%}")
                print(
                    "Challenger record vs BestModel: "
                    f"{stats_vs_best_model.wins}-{stats_vs_best_model.losses}-{stats_vs_best_model.draws} "
                    f"(win {stats_vs_best_model.win_rate:.2%} / loss {stats_vs_best_model.loss_rate:.2%} / draw {stats_vs_best_model.draw_rate:.2%})"
                )

                beats_best = (
                    stats_vs_best_model.win_rate > stats_vs_best_model.loss_rate
                )
                if beats_previous and beats_best:
                    print(
                        "Challenger beat previous and BestModel (win/all > loss/all). Promoting new best model."
                    )
                    shutil.copy(iter_model_path, best_model_path)
                    print(f"Best model updated: {best_model_path}")
                    best_model_updated = True
                else:
                    if not beats_previous:
                        print(
                            "Challenger did not beat previous (win rate <= loss rate)."
                        )
                    if not beats_best:
                        print(
                            "Challenger did not beat BestModel (win rate <= loss rate)."
                        )
        else:
            print("Challenger did not pass RandomAgent threshold.")

        eval_time = baseline_train._stage_finish("eval", eval_label, eval_start_time, stage_history)
        iteration_metrics["eval_time_sec"] = eval_time
        iteration_metrics["win_rate_vs_random"] = stats_vs_rnd.win_rate
        iteration_metrics["loss_rate_vs_random"] = stats_vs_rnd.loss_rate
        iteration_metrics["draw_rate_vs_random"] = stats_vs_rnd.draw_rate
        iteration_metrics["win_rate_vs_previous"] = win_rate_vs_previous
        iteration_metrics["win_rate_vs_best"] = win_rate_vs_best_model
        if stats_vs_best_model is not None:
            iteration_metrics["loss_rate_vs_best"] = stats_vs_best_model.loss_rate
            iteration_metrics["draw_rate_vs_best"] = stats_vs_best_model.draw_rate
        iteration_metrics["best_model_updated"] = best_model_updated
        iteration_metrics["win_rate_threshold"] = win_rate_threshold
        iteration_metrics["promotion_vs_opponent_threshold"] = promotion_vs_opponent_threshold

        # Release evaluation VRAM (eval agents, best_model copies, MCTS
        # instances) before the next iteration starts fresh.
        _cleanup_gpu()

        total_iter_time = time.perf_counter() - iter_start_time
        iteration_metrics["iteration_time_sec"] = total_iter_time
        iteration_metrics["timestamp_end"] = time.time()
        metrics.append(iteration_metrics)

        print(
            f"[Iteration {iteration + 1}] Timing summary -> "
            f"total={total_iter_time:.2f}s | self_play={self_play_time:.2f}s | "
            f"train={train_time:.2f}s | eval={eval_time:.2f}s"
        )
        games_text = (
            num_games_generated if num_games_generated is not None else "offline batch"
        )
        print(
            f"[Iteration {iteration + 1}] Generated {games_text} with "
            f"{total_positions} positions; training examples={len(examples)}."
        )

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
    print(f"Iteration metrics written to {metrics_path}")


def _parse_runtime_config(arg: Optional[str]) -> Optional[Dict[str, Any]]:
    if not arg:
        return None
    try:
        if os.path.isfile(arg):
            with open(arg, "r", encoding="utf-8") as cfg_file:
                return json.load(cfg_file)
        return json.loads(arg)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to load runtime configuration from {arg}: {exc}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with the v0 CUDA/C++ self-play pipeline.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations.")
    parser.add_argument("--mcts_simulations", type=int, default=800, help="MCTS simulations per move.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per iteration.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints_v0", help="Directory to save checkpoints."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--self_play_devices",
        type=str,
        default=None,
        help="Comma-separated device list for self-play (e.g., cuda:0,cuda:1) or 'auto'.",
    )
    parser.add_argument(
        "--train_devices",
        type=str,
        default=None,
        help="Comma-separated device list for training/DataParallel or 'auto'.",
    )
    parser.add_argument(
        "--eval_devices",
        type=str,
        default=None,
        help="Comma-separated device list for evaluation or 'auto'.",
    )
    parser.add_argument(
        "--self_play_workers", type=int, default=1, help="Number of parallel self-play worker processes."
    )
    parser.add_argument(
        "--self_play_games_per_worker",
        type=int,
        default=8,
        help="Number of self-play games each worker should play per iteration.",
    )
    parser.add_argument(
        "--self_play_base_seed",
        type=int,
        default=0,
        help="Base RNG seed for parallel self-play (0 = auto).",
    )
    parser.add_argument(
        "--self_play_virtual_loss",
        type=float,
        default=1.0,
        help="Virtual loss weight used during v0 self-play.",
    )
    parser.add_argument(
        "--self_play_opening_random_moves",
        type=int,
        default=4,
        help="Number of opening moves to sample uniformly at random during self-play.",
    )
    parser.add_argument(
        "--self_play_resign_threshold",
        type=float,
        default=-0.8,
        help="Resign when root value <= threshold (set >=0 to disable).",
    )
    parser.add_argument(
        "--self_play_resign_min_moves",
        type=int,
        default=10,
        help="Disable resign for the first N moves.",
    )
    parser.add_argument(
        "--self_play_resign_consecutive",
        type=int,
        default=3,
        help="Require this many consecutive low-value steps to resign.",
    )
    parser.add_argument(
        "--decisive_only",
        action="store_true",
        help="Train only on decisive (win/loss) samples; drop draws regardless of soft_value.",
    )
    parser.add_argument(
        "--value_draw_weight",
        type=float,
        default=0.1,
        help="Weight for draw samples in value loss (1.0 = no downweight).",
    )
    parser.add_argument(
        "--policy_draw_weight",
        type=float,
        default=0.3,
        help="Weight for draw samples in policy loss (1.0 = no downweight).",
    )
    parser.add_argument(
        "--self_play_batch_leaves",
        type=int,
        default=256,
        help="Number of leaf evaluations batched together inside v0 MCTS.",
    )
    parser.add_argument(
        "--self_play_inference_backend",
        "--self_play_inference-backend",
        type=str,
        default="graph",
        choices=["graph", "ts", "py"],
        help="Inference backend for v0 MCTS: graph|ts|py.",
    )
    parser.add_argument(
        "--self_play_torchscript_path",
        "--self_play_torchscript-path",
        type=str,
        default=None,
        help="Optional TorchScript path for v0 inference backends.",
    )
    parser.add_argument(
        "--self_play_torchscript_dtype",
        "--self_play_torchscript-dtype",
        type=str,
        default=None,
        help="Optional TorchScript dtype override (float16/float32/bfloat16).",
    )
    parser.add_argument(
        "--self_play_inference_batch_size",
        "--self_play_inference-batch-size",
        type=int,
        default=512,
        help="Fixed batch size for graph inference backend.",
    )
    parser.add_argument(
        "--self_play_inference_warmup_iters",
        "--self_play_inference-warmup-iters",
        type=int,
        default=5,
        help="Warmup iterations inside the graph inference engine.",
    )
    parser.add_argument(
        "--self_play_dirichlet_alpha",
        type=float,
        default=0.3,
        help="Dirichlet alpha used when injecting root noise.",
    )
    parser.add_argument(
        "--self_play_dirichlet_epsilon",
        type=float,
        default=0.25,
        help="Dirichlet epsilon scaling the mix between prior and noise.",
    )
    parser.add_argument("--eval_games_vs_random", type=int, default=4, help="Games vs RandomAgent.")
    parser.add_argument("--eval_games_vs_best", type=int, default=4, help="Games vs BestModel.")
    parser.add_argument(
        "--eval_games_vs_previous",
        type=int,
        default=0,
        help="Games vs previous iteration (0=disabled). Use to monitor steady improvement.",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=0,
        help="Number of parallel workers for evaluation (0 = use self_play_workers).",
    )
    parser.add_argument(
        "--eval_backend",
        type=str,
        default="legacy",
        help="Evaluation MCTS backend (legacy or v0).",
    )
    parser.add_argument(
        "--win_rate_threshold",
        type=float,
        default=0.55,
        help="Min win rate vs Random to consider for best (e.g. 0.55 = 55%%).",
    )
    parser.add_argument(
        "--promotion_vs_opponent_threshold",
        type=float,
        default=0.5,
        help="Unused; promotion uses win/all > loss/all (win rate > loss rate) for vs previous/best.",
    )
    parser.add_argument(
        "--mcts_sims_eval", type=int, default=100, help="MCTS simulations for evaluation agents."
    )
    parser.add_argument(
        "--runtime_config",
        type=str,
        default=None,
        help="JSON path or string with runtime overrides (verbosity, evaluation, etc.).",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        nargs="*",
        default=None,
        help="JSONL files containing offline samples. When provided, self-play generation is skipped.",
    )
    parser.add_argument(
        "--data_samples_per_iteration",
        type=int,
        default=None,
        help="Limit the number of offline samples consumed per iteration.",
    )
    parser.add_argument(
        "--data_shuffle",
        action="store_true",
        help="Shuffle offline samples before each iteration.",
    )
    parser.add_argument(
        "--save_self_play_dir",
        type=str,
        default=None,
        help="If set, dumps generated self-play samples (JSONL) into this directory each iteration.",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to load and resume training from.",
    )

    args = parser.parse_args()

    if (not args.data_files) and args.self_play_games_per_worker <= 0:
        raise ValueError("self_play_games_per_worker must be greater than zero.")

    print(f"Training on device: {args.device}")
    print(f"Training configuration: {args}")

    runtime_config = _parse_runtime_config(args.runtime_config)

    train_pipeline_v0(
        iterations=args.iterations,
        num_mcts_simulations=args.mcts_simulations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        self_play_workers=args.self_play_workers,
        self_play_games_per_worker=args.self_play_games_per_worker,
        self_play_base_seed=(None if args.self_play_base_seed == 0 else args.self_play_base_seed),
        self_play_virtual_loss_weight=args.self_play_virtual_loss,
        self_play_opening_random_moves=args.self_play_opening_random_moves,
        self_play_resign_threshold=args.self_play_resign_threshold,
        self_play_resign_min_moves=args.self_play_resign_min_moves,
        self_play_resign_consecutive=args.self_play_resign_consecutive,
        decisive_only=args.decisive_only,
        value_draw_weight=args.value_draw_weight,
        policy_draw_weight=args.policy_draw_weight,
        eval_games_vs_random=args.eval_games_vs_random,
        eval_games_vs_best=args.eval_games_vs_best,
        eval_games_vs_previous=args.eval_games_vs_previous,
        eval_workers=args.eval_workers,
        eval_backend=args.eval_backend,
        win_rate_threshold=args.win_rate_threshold,
        promotion_vs_opponent_threshold=args.promotion_vs_opponent_threshold,
        mcts_sims_eval=args.mcts_sims_eval,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        self_play_devices=args.self_play_devices,
        train_devices=args.train_devices,
        eval_devices=args.eval_devices,
        runtime_config=runtime_config,
        self_play_batch_leaves=args.self_play_batch_leaves,
        self_play_dirichlet_alpha=args.self_play_dirichlet_alpha,
        self_play_dirichlet_epsilon=args.self_play_dirichlet_epsilon,
        self_play_inference_backend=args.self_play_inference_backend,
        self_play_torchscript_path=args.self_play_torchscript_path,
        self_play_torchscript_dtype=args.self_play_torchscript_dtype,
        self_play_inference_batch_size=args.self_play_inference_batch_size,
        self_play_inference_warmup_iters=args.self_play_inference_warmup_iters,
        data_files=args.data_files,
        data_samples_per_iteration=args.data_samples_per_iteration,
        data_shuffle=args.data_shuffle,
        save_self_play_dir=args.save_self_play_dir,
        load_checkpoint=args.load_checkpoint,
    )
