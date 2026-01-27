import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional
import random
import argparse
import shutil
import json
from datetime import datetime, timedelta

from src.game_state import GameState, Player, Phase
from src.neural_network import ChessNet, state_to_tensor, NUM_INPUT_CHANNELS, get_move_probabilities
from src.mcts import self_play
from src.evaluate import MCTSAgent, RandomAgent, evaluate_against_agent, evaluate_against_agent_parallel
from src.random_agent import RandomAgent
from src.move_generator import MoveType, generate_all_legal_moves


def _format_eta(seconds: float) -> str:
    """
    Convert seconds to a HH:MM:SS string; fallback to 'unknown' when no data.
    """
    if seconds is None or seconds <= 0:
        return "unknown"
    seconds = int(max(0, seconds))
    return str(timedelta(seconds=seconds))


def _stage_banner(
    stage_key: str,
    label: str,
    iteration: int,
    total_iterations: int,
    history: Dict[str, List[float]]
) -> None:
    """
    Print a timestamped banner indicating the stage has started and its ETA.
    """
    now_text = datetime.now().strftime("%H:%M:%S")
    past = history.get(stage_key) or []
    eta_text = "collecting data..."
    if past:
        avg_duration = sum(past) / len(past)
        eta_text = _format_eta(avg_duration)
    print(f"[{now_text}] Iteration {iteration+1}/{total_iterations} - {label} started (ETA ~ {eta_text})")


def _stage_finish(
    stage_key: str,
    label: str,
    start_time: float,
    history: Dict[str, List[float]]
) -> float:
    """
    Record the duration of a stage, update its history, and print the result.
    """
    duration = time.perf_counter() - start_time
    history.setdefault(stage_key, []).append(duration)
    now_text = datetime.now().strftime("%H:%M:%S")
    print(f"[{now_text}] {label} finished in {_format_eta(duration)}")
    return duration


def _summarize_legal_moves_cache(
    examples: List[Tuple[GameState, np.ndarray, Optional[List[dict]], float, float]]
) -> Dict[str, int]:
    stats = {
        "cached": 0,
        "empty": 0,
        "missing": 0,
        "empty_with_policy": 0,
        "total": len(examples),
    }
    for ex in examples:
        if len(ex) < 3:
            stats["missing"] += 1
            continue
        policy = ex[1] if len(ex) > 1 else []
        legal_moves = ex[2]
        if legal_moves is None:
            stats["missing"] += 1
            continue
        if len(legal_moves) == 0:
            stats["empty"] += 1
            if len(policy) > 0:
                stats["empty_with_policy"] += 1
        else:
            stats["cached"] += 1
    return stats

class ChessDataset(Dataset):
    """
    用于训练神经网络的数据集。
    """
    def __init__(self, examples: List[Tuple[GameState, np.ndarray, Optional[List[dict]], float, float]]):
        """
        初始化数据集。
        
        Args:
            examples: 一个列表，每个元素是一个元组 (GameState, mcts_policy_array, legal_moves, value, soft_value)。
                      mcts_policy_array 是对应 GameState 所有合法走法的MCTS搜索概率。
                      legal_moves 是预计算的合法走法列表。
        """
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[MoveType], torch.Tensor, torch.Tensor, torch.Tensor]:
        state_obj, mcts_policy, legal_moves, value, soft_value = self.examples[idx]

        # Convert state to tensor.
        state_tensor = state_to_tensor(state_obj, state_obj.current_player).squeeze(0)

        # Convert policy to tensor.
        mcts_policy_tensor = torch.FloatTensor(mcts_policy)

        # Convert values to tensors.
        value_tensor = torch.FloatTensor([value])
        soft_tensor = torch.FloatTensor([soft_value])

        # Use cached legal moves unless missing, or empty while policy is non-empty.
        policy_len = int(mcts_policy_tensor.numel())
        if legal_moves is None or (len(legal_moves) == 0 and policy_len > 0):
            # Legacy data or missing cache: regenerate to match policy ordering.
            legal_moves = generate_all_legal_moves(state_obj)

        # Ensure policy length matches legal moves.
        if len(legal_moves) != len(mcts_policy_tensor):
            if not legal_moves and len(mcts_policy_tensor) == 0:
                pass
            else:
                print(f"Critical Warning: Mismatch len(legal_moves)={len(legal_moves)} vs len(mcts_policy)={len(mcts_policy_tensor)}")

        return state_tensor, legal_moves, mcts_policy_tensor, value_tensor, soft_tensor

def mcts_collate_fn(batch: List[Tuple[torch.Tensor, List[MoveType], torch.Tensor, torch.Tensor, torch.Tensor]]) \
    -> Tuple[torch.Tensor, List[List[MoveType]], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    自定义的collate_fn，用于处理包含legal_moves列表的批次。
    """
    state_tensors = torch.stack([item[0] for item in batch])
    list_of_legal_moves = [item[1] for item in batch]  # list of lists of dicts
    target_policies_list = [item[2] for item in batch] # list of Tensors
    value_tensors = torch.stack([item[3] for item in batch])
    soft_tensors = torch.stack([item[4] for item in batch])

    return state_tensors, list_of_legal_moves, target_policies_list, value_tensors, soft_tensors

def train_network(
    model: ChessNet,
    examples: List[Tuple[GameState, np.ndarray, Optional[List[dict]], float, float]],
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    soft_label_alpha: float = 0.0,
    value_draw_weight: float = 1.0,
    policy_draw_weight: float = 1.0,
    device: str = 'cpu',
    board_size: int = GameState.BOARD_SIZE
) -> Tuple[ChessNet, Dict[str, Any]]:
    """
    使用生成的训练数据训练神经网络。
    
    Args:
        model: 要训练的神经网络模型。
        examples: 训练数据，每个元素是一个元组 (state, policy, value)。
        batch_size: 批处理大小。
        epochs: 训练轮数。
        lr: 学习率。
        weight_decay: 权重衰减。
        value_draw_weight: draw 样本在 value loss 中的权重（1.0=不降权）。
        policy_draw_weight: draw 样本在 policy loss 中的权重（1.0=不降权）。
        device: 训练设备。
        board_size: 棋盘大小。
        
    Returns:
        Tuple containing the trained model and a metrics dictionary with per-epoch loss statistics.
    """
    model.to(device)
    model.train()

    if os.environ.get("LIUZHOU_DEBUG_LEGAL_MOVES"):
        stats = _summarize_legal_moves_cache(examples)
        print(
            "[legal_moves] cached={cached} empty={empty} missing={missing} "
            "empty_with_policy={empty_with_policy} total={total}".format(**stats)
        )

    
    # 创建数据集和数据加载器
    dataset = ChessDataset(examples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=mcts_collate_fn,
        num_workers=4,       # Enable parallel data loading
        pin_memory=True,     # Accelerate CPU->GPU transfer
        prefetch_factor=2,   # Prefetch 2 batches per worker
        persistent_workers=True,  # Keep workers alive between epochs
    )
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 创建损失函数
    value_draw_weight = max(0.0, float(value_draw_weight))
    policy_draw_weight = max(0.0, float(policy_draw_weight))
    policy_loss_fn = nn.KLDivLoss(reduction='sum')
    
    # 训练循环
    epoch_stats: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        total_loss_epoch = 0.0
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        num_samples_processed = 0
        total_valid_policy_samples = 0
        policy_weight_sum_epoch = 0.0
        soft_abs_epoch = 0.0
        mix_abs_epoch = 0.0
        soft_batches = 0
        
        for batch_idx, (states_tensor_batch, batch_legal_moves, batch_target_policies, target_values_batch, soft_values_batch) in enumerate(dataloader):
            states_tensor_batch = states_tensor_batch.to(device)
            target_values_batch = target_values_batch.to(device)
            soft_values_batch = soft_values_batch.to(device)
            # batch_target_policies is a list of tensors, they will be moved to device inside the loop

            optimizer.zero_grad()
            
            # 前向传播 (整个批次)
            log_p1_batch, log_p2_batch, log_pmc_batch, value_pred_batch = model(states_tensor_batch)
            
            current_batch_policy_loss = torch.tensor(0.0, device=device)
            y_mix = (1.0 - soft_label_alpha) * target_values_batch + soft_label_alpha * soft_values_batch

            draw_weight = target_values_batch.new_full((), value_draw_weight)
            one_weight = target_values_batch.new_ones(())
            draw_mask = target_values_batch.abs() < 1e-8
            weights = torch.where(draw_mask, draw_weight, one_weight)
            weighted_sq = (value_pred_batch - y_mix) ** 2 * weights
            weight_sum = weights.sum()
            if weight_sum.item() > 0:
                current_batch_value_loss = weighted_sq.sum() / weight_sum
            else:
                current_batch_value_loss = torch.tensor(0.0, device=device)
            soft_abs_epoch += soft_values_batch.abs().mean().item()
            mix_abs_epoch += y_mix.abs().mean().item()
            soft_batches += 1
            
            valid_policy_samples_in_batch = 0
            policy_weight_sum_in_batch = 0.0

            # 处理批次中的每个样本以计算策略损失
            for i in range(states_tensor_batch.size(0)):
                log_p1_sample = log_p1_batch[i]
                log_p2_sample = log_p2_batch[i]
                log_pmc_sample = log_pmc_batch[i]
                
                legal_moves_sample = batch_legal_moves[i]
                target_policy_sample = batch_target_policies[i].to(device) # MCTS target policy (probabilities)

                if not legal_moves_sample or target_policy_sample.nelement() == 0:
                    # 如果没有合法走法或MCTS策略为空，则跳过此样本的策略损失计算
                    continue

                # 使用 get_move_probabilities 获取网络对这些合法走法的原始对数概率(组合形式)
                # get_move_probabilities returns (probs_after_softmax, raw_combined_log_probs_before_softmax)
                _, net_raw_log_probs_for_moves = get_move_probabilities(
                    log_policy_pos1=log_p1_sample,
                    log_policy_pos2=log_p2_sample,
                    log_policy_mark_capture=log_pmc_sample,
                    legal_moves=legal_moves_sample,
                    board_size=board_size, # board_size is needed
                    device=device
                )

                if net_raw_log_probs_for_moves.nelement() == 0:
                    # 这不应该发生，如果legal_moves_sample非空
                    # print(f"Warning: net_raw_log_probs_for_moves is empty for a sample with legal moves.")
                    continue
                
                # KLDivLoss 需要对数概率作为输入, 目标是概率分布
                # net_raw_log_probs_for_moves 是组合后的原始分数，需要进行 log_softmax
                network_log_softmax_for_moves = torch.nn.functional.log_softmax(net_raw_log_probs_for_moves, dim=0)

                # 确保形状匹配
                if network_log_softmax_for_moves.shape != target_policy_sample.shape:
                    # print(f"Warning: Shape mismatch for policy loss. Net: {network_log_softmax_for_moves.shape}, Target: {target_policy_sample.shape}. Skipping sample.")
                    continue

                value_abs = float(target_values_batch[i].abs().item())
                policy_weight = policy_draw_weight if value_abs < 1e-8 else 1.0
                sample_policy_loss = policy_loss_fn(network_log_softmax_for_moves, target_policy_sample)
                current_batch_policy_loss += sample_policy_loss * policy_weight
                valid_policy_samples_in_batch += 1
                policy_weight_sum_in_batch += policy_weight
            
            if policy_weight_sum_in_batch > 0:
                # 平均每个有效样本的策略损失 (因为KLDivLoss(reduction='sum')是对单个策略求和)
                averaged_batch_policy_loss = current_batch_policy_loss / policy_weight_sum_in_batch
            else:
                # 如果批次中没有有效的策略样本，则策略损失为0
                averaged_batch_policy_loss = torch.tensor(0.0, device=device, requires_grad=True) 

            # 总损失
            loss = averaged_batch_policy_loss + current_batch_value_loss # current_batch_value_loss is already averaged by MSELoss (default reduction='mean')
            
            if valid_policy_samples_in_batch > 0 or current_batch_value_loss.requires_grad : # Ensure there's something to backprop
                # 反向传播
                loss.backward()
                optimizer.step()
            
            # 累积损失 (用于打印)
            total_loss_epoch += loss.item() * states_tensor_batch.size(0) # Weighted by batch size if some were skipped
            policy_loss_epoch += averaged_batch_policy_loss.item() * policy_weight_sum_in_batch
            value_loss_epoch += current_batch_value_loss.item() * states_tensor_batch.size(0) # MSE is already mean
            num_samples_processed += states_tensor_batch.size(0)
            total_valid_policy_samples += valid_policy_samples_in_batch
            policy_weight_sum_epoch += policy_weight_sum_in_batch
        
        denom = max(1, soft_batches)
        avg_soft_abs = soft_abs_epoch / denom
        avg_mix_abs = mix_abs_epoch / denom
        
        # 打印每个 epoch 的损失
        if num_samples_processed > 0 :
            avg_loss = total_loss_epoch / num_samples_processed
            avg_policy_loss = policy_loss_epoch / policy_weight_sum_epoch if policy_weight_sum_epoch > 0 else 0.0
            avg_value_loss = value_loss_epoch / num_samples_processed
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}")
            epoch_stats.append({
                "epoch": epoch + 1,
                "avg_loss": float(avg_loss),
                "avg_policy_loss": float(avg_policy_loss),
                "avg_value_loss": float(avg_value_loss),
                "samples": int(num_samples_processed),
                "valid_policy_samples": int(total_valid_policy_samples),
                "policy_weight_sum": float(policy_weight_sum_epoch),
                "soft_alpha": float(soft_label_alpha),
                "avg_soft_abs": float(avg_soft_abs),
                "avg_mix_abs": float(avg_mix_abs),
            })
        else:
            print(f"Epoch {epoch+1}/{epochs}, No samples processed.")
            epoch_stats.append({
                "epoch": epoch + 1,
                "avg_loss": None,
                "avg_policy_loss": None,
                "avg_value_loss": None,
                "samples": 0,
                "valid_policy_samples": 0,
                "soft_alpha": float(soft_label_alpha),
                "avg_soft_abs": None,
                "avg_mix_abs": None,
            })
            
    return model, {"epoch_stats": epoch_stats}

def train_pipeline(
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
    self_play_virtual_loss_weight: float = 0.0,
    eval_games_vs_random: int = 20,
    eval_games_vs_best: int = 20,
    win_rate_threshold: float = 0.55,
    mcts_sims_eval: int = 100,
    eval_workers: int = 0,
    checkpoint_dir: str = "./checkpoints",
    device: str = "cpu",
    runtime_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Complete training pipeline.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    metrics: List[Dict[str, Any]] = []
    metrics_path = os.path.join(checkpoint_dir, "training_metrics.json")
    stage_history: Dict[str, List[float]] = {
        "self_play": [],
        "train": [],
        "eval": [],
    }

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

    if sp_games_per_worker is None:
        raise ValueError("self_play_games_per_worker must be provided and greater than zero.")
    if sp_games_per_worker <= 0:
        raise ValueError("self_play_games_per_worker must be greater than zero.")
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

    board_size = GameState.BOARD_SIZE
    current_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
    current_model.to(device)

    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

    for iteration in range(iterations):
        print(f"\n{'='*20} Iteration {iteration+1}/{iterations} {'='*20}")

        iter_start_time = time.perf_counter()
        iteration_metrics: Dict[str, Any] = {
            "iteration": iteration + 1,
            "self_play_games_requested": total_self_play_games,
            "mcts_simulations": num_mcts_simulations,
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "timestamp_start": time.time(),
            "self_play_workers": sp_workers,
            "self_play_games_per_worker": sp_games_per_worker,
            "self_play_virtual_loss": sp_virtual_loss,
            "self_play_base_seed": sp_base_seed,
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

        print(f"Soft label alpha (value mix): {current_alpha:.3f}, soft_value_k={soft_value_k:.2f}")

        self_play_label = "Self-play phase"
        _stage_banner("self_play", self_play_label, iteration, iterations, stage_history)
        print(f"{self_play_label}: generating {total_self_play_games} games using current model...")
        current_model.eval()
        sp_start_time = time.perf_counter()
        training_data = self_play(
            model=current_model,
            num_games=total_self_play_games,
            mcts_simulations=num_mcts_simulations,
            temperature_init=temperature_init,
            temperature_final=temperature_final,
            temperature_threshold=temperature_threshold,
            exploration_weight=exploration_weight,
            device=device,
            add_dirichlet_noise=self_play_add_dirichlet,
            mcts_verbose=self_play_mcts_verbose,
            verbose=self_play_verbose,
            num_workers=sp_workers,
            games_per_worker=sp_games_per_worker,
            base_seed=sp_base_seed,
            virtual_loss_weight=sp_virtual_loss,
            soft_value_k=soft_value_k,
        )
        training_data = training_data or []
        num_games_generated = len(training_data)
        self_play_time = _stage_finish("self_play", self_play_label, sp_start_time, stage_history)
        iteration_metrics["self_play_time_sec"] = self_play_time
        iteration_metrics["self_play_games_played"] = num_games_generated

        examples: List[Tuple[GameState, np.ndarray, float, float]] = []
        total_positions = 0
        train_metrics_data: Dict[str, Any] = {"epoch_stats": []}

        train_label = "Training phase"
        _stage_banner("train", train_label, iteration, iterations, stage_history)
        train_start_time = time.perf_counter()

        if not training_data:
            print("No training data generated from self-play. Skipping training for this iteration.")
            train_time = _stage_finish("train", train_label, train_start_time, stage_history)
        else:
            for game_states, game_policies, result, soft_value in training_data:
                for i, state in enumerate(game_states):
                    sign = 1.0 if state.current_player == Player.BLACK else -1.0
                    value = sign * result
                    soft_signed = sign * soft_value
                    examples.append((state, game_policies[i], value, soft_signed))
                    total_positions += 1

            print(f"{train_label}: {len(examples)} examples, {epochs} epochs...")
            current_model.train()
            current_model, train_metrics_data = train_network(
                model=current_model,
                examples=examples,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                soft_label_alpha=current_alpha,
                device=device,
                board_size=board_size
            )
            train_time = _stage_finish("train", train_label, train_start_time, stage_history)

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

        iter_model_path = os.path.join(checkpoint_dir, f"model_iter_{iteration+1}.pt")
        torch.save({
            "iteration": iteration + 1,
            "model_state_dict": current_model.state_dict(),
            "board_size": board_size,
            "num_input_channels": NUM_INPUT_CHANNELS
        }, iter_model_path)
        print(f"Model for iteration {iteration+1} saved to {iter_model_path}")
        iteration_metrics["checkpoint_path"] = iter_model_path

        print("\nEvaluation phase...")
        eval_label = "Evaluation phase"
        _stage_banner("eval", eval_label, iteration, iterations, stage_history)
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
                        sample_moves=True,
                    )
                else:
                    best_model_checkpoint = torch.load(best_model_path, map_location=device)
                    best_model_eval = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
                    best_model_eval.load_state_dict(best_model_checkpoint["model_state_dict"])
                    best_model_eval.to(device)
                    best_model_eval.eval()

                    challenger_agent_best = MCTSAgent(
                        current_model,
                        mcts_simulations=mcts_sims_eval,
                        temperature=eval_temperature,
                        device=device,
                        add_dirichlet_noise=eval_add_dirichlet,
                        verbose=eval_verbose,
                        mcts_verbose=eval_mcts_verbose,
                        sample_moves=True,
                    )
                    best_agent_opponent = MCTSAgent(
                        best_model_eval,
                        mcts_simulations=mcts_sims_eval,
                        temperature=eval_temperature,
                        device=device,
                        add_dirichlet_noise=eval_add_dirichlet,
                        verbose=eval_verbose,
                        mcts_verbose=eval_mcts_verbose,
                        sample_moves=True,
                    )

                    stats_vs_best_model = evaluate_against_agent(
                        challenger_agent_best,
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

                if win_rate_vs_best_model > win_rate_threshold:
                    print("Challenger passed BestModel threshold. Updating best_model.pt.")
                    shutil.copy(iter_model_path, best_model_path)
                    print(f"Best model updated: {best_model_path}")
                    best_model_updated = True
                else:
                    print("Challenger did not surpass BestModel.")
        else:
            print("Challenger did not pass RandomAgent threshold.")

        eval_time = _stage_finish("eval", eval_label, eval_start_time, stage_history)
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

        print(f"[Iteration {iteration+1}] Timing summary -> total={total_iter_time:.2f}s | self_play={self_play_time:.2f}s | train={train_time:.2f}s | eval={eval_time:.2f}s")
        print(f"[Iteration {iteration+1}] Generated {num_games_generated} games with {total_positions} positions; training examples={len(examples)}.")

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
    print(f"Iteration metrics written to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a chess AI using AlphaZero-style self-play.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations.")
    parser.add_argument("--mcts_simulations", type=int, default=20, help="Number of MCTS simulations per move for self-play.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs per iteration.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_eval_test", help="Directory to save model checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--self_play_workers", type=int, default=1, help="Number of parallel workers for self-play (>=1).")
    parser.add_argument("--self_play_games_per_worker", type=int, default=2, help="Number of self-play games each worker should play per iteration.")
    parser.add_argument("--self_play_base_seed", type=int, default=0, help="Base RNG seed for parallel self-play (0 = auto).")
    parser.add_argument("--self_play_virtual_loss", type=float, default=0.0, help="Virtual loss weight passed to MCTS during self-play.")
    parser.add_argument("--eval_games_vs_random", type=int, default=4, help="Games vs RandomAgent.")
    parser.add_argument("--eval_games_vs_best", type=int, default=4, help="Games vs BestModel.")
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=0,
        help="Number of parallel workers for evaluation (0 = use self_play_workers).",
    )
    parser.add_argument("--win_rate_threshold", type=float, default=0.55, help="Win rate to beat opponent.")
    parser.add_argument("--mcts_sims_eval", type=int, default=20, help="MCTS sims for evaluation agent.")
    parser.add_argument(
        "--runtime_config",
        type=str,
        default=None,
        help="JSON string or file path specifying runtime options (verbosity, evaluation, etc.).",
    )

    args = parser.parse_args()

    if args.self_play_games_per_worker <= 0:
        raise ValueError("self_play_games_per_worker must be greater than zero.")
    
    print(f"Training on device: {args.device}")
    print(f"Training configuration: {args}")
    
    # 运行训练流程
    runtime_config: Optional[Dict[str, Any]] = None
    if args.runtime_config:
        config_source = args.runtime_config
        try:
            if os.path.isfile(config_source):
                with open(config_source, "r", encoding="utf-8") as cfg_file:
                    runtime_config = json.load(cfg_file)
            else:
                runtime_config = json.loads(config_source)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Failed to load runtime configuration from {config_source}: {exc}") from exc

    train_pipeline(
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
    )
