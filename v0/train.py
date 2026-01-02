"""Training pipeline that wires the v0 CUDA/C++ self-play core into the
AlphaZero-style loop.
"""

from __future__ import annotations

import argparse
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
from src.evaluate import MCTSAgent, evaluate_against_agent, evaluate_against_agent_parallel
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
    eval_games_vs_random: int = 20,
    eval_games_vs_best: int = 20,
    win_rate_threshold: float = 0.55,
    mcts_sims_eval: int = 100,
    eval_workers: int = 0,
    checkpoint_dir: str = "./checkpoints_v0",
    device: str = "cpu",
    runtime_config: Optional[Dict[str, Any]] = None,
    self_play_batch_leaves: int = 256,
    self_play_dirichlet_alpha: float = 0.3,
    self_play_dirichlet_epsilon: float = 0.25,
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

    evaluation_cfg = runtime_config.get("evaluation", {})
    eval_temperature = evaluation_cfg.get("temperature", 0.05)
    eval_add_dirichlet = evaluation_cfg.get("add_dirichlet_noise", False)
    if "mcts_simulations" in evaluation_cfg:
        mcts_sims_eval = evaluation_cfg["mcts_simulations"]
    eval_games_vs_random = evaluation_cfg.get("games_vs_random", eval_games_vs_random)
    eval_games_vs_best = evaluation_cfg.get("games_vs_best", eval_games_vs_best)
    eval_workers_cfg = evaluation_cfg.get("workers", evaluation_cfg.get("num_workers", eval_workers))
    try:
        eval_workers = int(eval_workers_cfg) if eval_workers_cfg is not None else 0
    except (TypeError, ValueError):
        eval_workers = 0
    if eval_workers <= 0:
        eval_workers = sp_workers
    eval_workers = max(1, eval_workers)

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

    board_size = GameState.BOARD_SIZE
    current_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
    
    # Load checkpoint if provided
    if load_checkpoint and os.path.exists(load_checkpoint):
        print(f"Loading model checkpoint from: {load_checkpoint}")
        checkpoint = torch.load(load_checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        current_model.load_state_dict(state_dict)
        loaded_iteration = checkpoint.get("iteration", 0)
        print(f"Loaded model from iteration {loaded_iteration}")
    else:
        if load_checkpoint:
            print(f"Warning: Checkpoint not found at {load_checkpoint}, starting with random model.")
        else:
            print("No checkpoint specified, starting with random model.")
    
    current_model.to(device)

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
                device=device,
                add_dirichlet_noise=self_play_add_dirichlet,
                dirichlet_alpha=sp_dirichlet_alpha,
                dirichlet_epsilon=sp_dirichlet_epsilon,
                batch_leaves=sp_batch_leaves,
                virtual_loss=sp_virtual_loss,
                num_workers=sp_workers,
                games_per_worker=sp_games_per_worker,
                base_seed=sp_base_seed,
                soft_value_k=soft_value_k,
                mcts_verbose=self_play_mcts_verbose,
                verbose=self_play_verbose,
            )
            training_data = training_data or []
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
            current_model.train()
            current_model, train_metrics_data = baseline_train.train_network(
                model=current_model,
                examples=examples,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                soft_label_alpha=current_alpha,
                device=device,
                board_size=board_size,
            )
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

        print("\nEvaluation phase...")
        eval_label = "Evaluation phase"
        baseline_train._stage_banner("eval", eval_label, iteration, iterations, stage_history)
        current_model.eval()
        eval_start_time = time.perf_counter()

        print(f"Evaluating challenger against RandomAgent ({eval_games_vs_random} games)...")
        if eval_workers > 1:
            stats_vs_rnd = evaluate_against_agent_parallel(
                challenger_checkpoint=iter_model_path,
                opponent_checkpoint=None,
                num_games=eval_games_vs_random,
                device=device,
                mcts_simulations=mcts_sims_eval,
                temperature=eval_temperature,
                add_dirichlet_noise=eval_add_dirichlet,
                num_workers=eval_workers,
                verbose=eval_verbose,
                game_verbose=eval_game_verbose,
                mcts_verbose=eval_mcts_verbose,
            )
        else:
            challenger_agent = MCTSAgent(
                current_model,
                mcts_simulations=mcts_sims_eval,
                temperature=eval_temperature,
                device=device,
                add_dirichlet_noise=eval_add_dirichlet,
                verbose=eval_verbose,
                mcts_verbose=eval_mcts_verbose,
            )
            random_opponent = RandomAgent()
            stats_vs_rnd = evaluate_against_agent(
                challenger_agent,
                random_opponent,
                eval_games_vs_random,
                device,
                verbose=eval_verbose,
                game_verbose=eval_game_verbose,
            )
        print(f"Challenger win rate vs RandomAgent: {stats_vs_rnd.win_rate:.2%}")
        print(
            "Challenger record vs RandomAgent: "
            f"{stats_vs_rnd.wins}-{stats_vs_rnd.losses}-{stats_vs_rnd.draws} "
            f"(win {stats_vs_rnd.win_rate:.2%} / loss {stats_vs_rnd.loss_rate:.2%} / draw {stats_vs_rnd.draw_rate:.2%})"
        )

        win_rate_vs_best_model = None
        stats_vs_best_model = None
        best_model_updated = False

        if stats_vs_rnd.win_rate > win_rate_threshold:
            print(f"Challenger passed RandomAgent threshold ({win_rate_threshold:.0%}). Comparing to best model...")
            if not os.path.exists(best_model_path):
                print("No existing best_model.pt. Current model becomes the best.")
                shutil.copy(iter_model_path, best_model_path)
                print(f"Best model updated: {best_model_path}")
                best_model_updated = True
            else:
                print("Loading best_model.pt for comparison...")

                print(f"Evaluating challenger against BestModel ({eval_games_vs_best} games)...")
                if eval_workers > 1:
                    stats_vs_best_model = evaluate_against_agent_parallel(
                        challenger_checkpoint=iter_model_path,
                        opponent_checkpoint=best_model_path,
                        num_games=eval_games_vs_best,
                        device=device,
                        mcts_simulations=mcts_sims_eval,
                        temperature=eval_temperature,
                        add_dirichlet_noise=eval_add_dirichlet,
                        num_workers=eval_workers,
                        verbose=eval_verbose,
                        game_verbose=eval_game_verbose,
                        mcts_verbose=eval_mcts_verbose,
                    )
                else:
                    best_model_checkpoint = torch.load(best_model_path, map_location=device)
                    best_model_eval = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
                    best_model_eval.load_state_dict(best_model_checkpoint["model_state_dict"])
                    best_model_eval.to(device)
                    best_model_eval.eval()

                    best_agent_opponent = MCTSAgent(
                        best_model_eval,
                        mcts_simulations=mcts_sims_eval,
                        temperature=eval_temperature,
                        device=device,
                        add_dirichlet_noise=eval_add_dirichlet,
                        verbose=eval_verbose,
                        mcts_verbose=eval_mcts_verbose,
                    )

                    stats_vs_best_model = evaluate_against_agent(
                        challenger_agent,
                        best_agent_opponent,
                        eval_games_vs_best,
                        device,
                        verbose=eval_verbose,
                        game_verbose=eval_game_verbose,
                    )
                win_rate_vs_best_model = stats_vs_best_model.win_rate
                print(f"Challenger win rate vs BestModel: {win_rate_vs_best_model:.2%}")
                print(
                    "Challenger record vs BestModel: "
                    f"{stats_vs_best_model.wins}-{stats_vs_best_model.losses}-{stats_vs_best_model.draws} "
                    f"(win {stats_vs_best_model.win_rate:.2%} / loss {stats_vs_best_model.loss_rate:.2%} / draw {stats_vs_best_model.draw_rate:.2%})"
                )

                if stats_vs_best_model.win_rate > win_rate_threshold:
                    print("Challenger beat the BestModel. Promoting new best model.")
                    shutil.copy(iter_model_path, best_model_path)
                    print(f"Best model updated: {best_model_path}")
                    best_model_updated = True
                else:
                    print("Challenger did not surpass BestModel.")
        else:
            print("Challenger did not pass RandomAgent threshold.")

        eval_time = baseline_train._stage_finish("eval", eval_label, eval_start_time, stage_history)
        iteration_metrics["eval_time_sec"] = eval_time
        iteration_metrics["win_rate_vs_random"] = stats_vs_rnd.win_rate
        iteration_metrics["loss_rate_vs_random"] = stats_vs_rnd.loss_rate
        iteration_metrics["draw_rate_vs_random"] = stats_vs_rnd.draw_rate
        iteration_metrics["win_rate_vs_best"] = win_rate_vs_best_model
        if stats_vs_best_model is not None:
            iteration_metrics["loss_rate_vs_best"] = stats_vs_best_model.loss_rate
            iteration_metrics["draw_rate_vs_best"] = stats_vs_best_model.draw_rate
        iteration_metrics["best_model_updated"] = best_model_updated
        iteration_metrics["win_rate_threshold"] = win_rate_threshold

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
        "--self_play_batch_leaves",
        type=int,
        default=256,
        help="Number of leaf evaluations batched together inside v0 MCTS.",
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
        "--eval_workers",
        type=int,
        default=0,
        help="Number of parallel workers for evaluation (0 = use self_play_workers).",
    )
    parser.add_argument("--win_rate_threshold", type=float, default=0.55, help="Win-rate threshold.")
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
        eval_games_vs_random=args.eval_games_vs_random,
        eval_games_vs_best=args.eval_games_vs_best,
        eval_workers=args.eval_workers,
        win_rate_threshold=args.win_rate_threshold,
        mcts_sims_eval=args.mcts_sims_eval,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        runtime_config=runtime_config,
        self_play_batch_leaves=args.self_play_batch_leaves,
        self_play_dirichlet_alpha=args.self_play_dirichlet_alpha,
        self_play_dirichlet_epsilon=args.self_play_dirichlet_epsilon,
        data_files=args.data_files,
        data_samples_per_iteration=args.data_samples_per_iteration,
        data_shuffle=args.data_shuffle,
        save_self_play_dir=args.save_self_play_dir,
        load_checkpoint=args.load_checkpoint,
    )
