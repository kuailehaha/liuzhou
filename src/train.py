import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import random
import argparse
import shutil

from src.game_state import GameState, Player, Phase
from src.neural_network import ChessNet, state_to_tensor, NUM_INPUT_CHANNELS, get_move_probabilities
from src.mcts import self_play
from src.evaluate import MCTSAgent, RandomAgent, evaluate_against_agent
from src.random_agent import RandomAgent
from src.move_generator import MoveType, generate_all_legal_moves

class ChessDataset(Dataset):
    """
    用于训练神经网络的数据集。
    """
    def __init__(self, examples: List[Tuple[GameState, np.ndarray, float]]):
        """
        初始化数据集。
        
        Args:
            examples: 一个列表，每个元素是一个元组 (GameState, mcts_policy_array, value)。
                      mcts_policy_array 是对应 GameState 所有合法走法的MCTS搜索概率。
        """
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[MoveType], torch.Tensor, torch.Tensor]:
        state_obj, mcts_policy, value = self.examples[idx]
        
        # 将状态转换为张量
        state_tensor = state_to_tensor(state_obj, state_obj.current_player).squeeze(0)
        
        # MCTS策略已经是numpy数组，转换为张量
        mcts_policy_tensor = torch.FloatTensor(mcts_policy)
        
        # 价值转换为张量
        value_tensor = torch.FloatTensor([value])
        
        # 获取当前状态的合法走法，MCTS策略是针对这些走法的
        # 必须保证 generate_all_legal_moves 的顺序与 MCTS 生成策略时的顺序一致
        legal_moves = generate_all_legal_moves(state_obj)

        # 安全检查：确保MCTS策略的长度与合法走法的数量一致
        if len(legal_moves) != len(mcts_policy_tensor):
            # 这种情况通常不应该发生，如果发生了，说明 self_play 或 MCTS 中存在问题
            # 例如，对于没有合法走法的状态，MCTS可能返回空策略
            # 这里可以添加更复杂的错误处理或日志记录
            # print(f"Warning: Mismatch in legal_moves ({len(legal_moves)}) and mcts_policy ({len(mcts_policy_tensor)}) for a state.")
            # 为了简单起见，如果发生这种情况，可以返回空的legal_moves和policy，训练循环中会跳过它
            if not legal_moves and len(mcts_policy_tensor) == 0: # 无合法走法，空策略，正常
                pass
            else: # 长度不匹配，但至少一方非空，这是个问题
                # For now, let's allow it to proceed, train_network will skip if shapes mismatch later.
                # A robust solution might involve filtering these samples or raising an error.
                print(f"Critical Warning: Mismatch len(legal_moves)={len(legal_moves)} vs len(mcts_policy)={len(mcts_policy_tensor)}")


        return state_tensor, legal_moves, mcts_policy_tensor, value_tensor

def mcts_collate_fn(batch: List[Tuple[torch.Tensor, List[MoveType], torch.Tensor, torch.Tensor]]) \
    -> Tuple[torch.Tensor, List[List[MoveType]], List[torch.Tensor], torch.Tensor]:
    """
    自定义的collate_fn，用于处理包含legal_moves列表的批次。
    """
    state_tensors = torch.stack([item[0] for item in batch])
    list_of_legal_moves = [item[1] for item in batch]  # list of lists of dicts
    target_policies_list = [item[2] for item in batch] # list of Tensors
    value_tensors = torch.stack([item[3] for item in batch])
    
    return state_tensors, list_of_legal_moves, target_policies_list, value_tensors

def train_network(
    model: ChessNet,
    examples: List[Tuple[GameState, np.ndarray, float]],
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = 'cpu',
    board_size: int = GameState.BOARD_SIZE
) -> ChessNet:
    """
    使用生成的训练数据训练神经网络。
    
    Args:
        model: 要训练的神经网络模型。
        examples: 训练数据，每个元素是一个元组 (state, policy, value)。
        batch_size: 批处理大小。
        epochs: 训练轮数。
        lr: 学习率。
        weight_decay: 权重衰减。
        device: 训练设备。
        board_size: 棋盘大小。
        
    Returns:
        训练后的模型。
    """
    model.to(device)
    model.train()
    
    # 创建数据集和数据加载器
    dataset = ChessDataset(examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=mcts_collate_fn)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 创建损失函数
    value_loss_fn = nn.MSELoss()
    policy_loss_fn = nn.KLDivLoss(reduction='sum')
    
    # 训练循环
    for epoch in range(epochs):
        total_loss_epoch = 0
        policy_loss_epoch = 0
        value_loss_epoch = 0
        num_samples_processed = 0
        
        for batch_idx, (states_tensor_batch, batch_legal_moves, batch_target_policies, target_values_batch) in enumerate(dataloader):
            states_tensor_batch = states_tensor_batch.to(device)
            target_values_batch = target_values_batch.to(device)
            # batch_target_policies is a list of tensors, they will be moved to device inside the loop

            optimizer.zero_grad()
            
            # 前向传播 (整个批次)
            log_p1_batch, log_p2_batch, log_pmc_batch, value_pred_batch = model(states_tensor_batch)
            
            current_batch_policy_loss = torch.tensor(0.0, device=device)
            current_batch_value_loss = value_loss_fn(value_pred_batch, target_values_batch)
            
            valid_policy_samples_in_batch = 0

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
                
                sample_policy_loss = policy_loss_fn(network_log_softmax_for_moves, target_policy_sample)
                current_batch_policy_loss += sample_policy_loss
                valid_policy_samples_in_batch += 1
            
            if valid_policy_samples_in_batch > 0:
                # 平均每个有效样本的策略损失 (因为KLDivLoss(reduction='sum')是对单个策略求和)
                averaged_batch_policy_loss = current_batch_policy_loss / valid_policy_samples_in_batch
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
            policy_loss_epoch += averaged_batch_policy_loss.item() * valid_policy_samples_in_batch
            value_loss_epoch += current_batch_value_loss.item() * states_tensor_batch.size(0) # MSE is already mean
            num_samples_processed += states_tensor_batch.size(0)
        
        # 打印每个 epoch 的损失
        if num_samples_processed > 0 :
            avg_loss = total_loss_epoch / num_samples_processed
            # For policy loss, average over samples where it was actually computed.
            # The denominator for policy_loss_epoch accumulation was valid_policy_samples_in_batch.
            # To get true average policy loss per sample in dataset:
            # Need total sum of policy losses / total number of valid policy examples in epoch
            # The current policy_loss_epoch is sum( (sum_kl_div_sample / num_valid_in_batch_i) * num_valid_in_batch_i ) = sum(sum_kl_div_sample)
            # So, we need to count total_valid_policy_samples_in_epoch
            total_valid_policy_samples_in_epoch_accumulator = 0 # This needs to be summed across batches
            # Re-think epoch loss accumulation for proper average, for now, this is an approximation:
            avg_policy_loss = policy_loss_epoch / num_samples_processed # Approximation if not all samples had policy loss
            avg_value_loss = value_loss_epoch / num_samples_processed
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, No samples processed.")
            
    return model

def train_pipeline(
    iterations: int = 10,
    num_self_play_games: int = 100,
    num_mcts_simulations: int = 800,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    temperature_init: float = 1.0,
    temperature_final: float = 0.1,
    temperature_threshold: int = 10,
    exploration_weight: float = 1.0,
    eval_games_vs_random: int = 20,
    eval_games_vs_best: int = 20,
    win_rate_threshold: float = 0.55,
    mcts_sims_eval: int = 100,
    checkpoint_dir: str = './checkpoints',
    device: str = 'cpu'
) -> None:
    """
    完整的训练流程。
    
    Args:
        iterations: 训练迭代次数。
        num_self_play_games: 每次迭代中自我对弈的游戏数量。
        num_mcts_simulations: 每步 MCTS 的模拟次数。
        batch_size: 训练批处理大小。
        epochs: 每次迭代的训练轮数。
        lr: 学习率。
        weight_decay: 权重衰减。
        temperature_init: 初始温度。
        temperature_final: 最终温度。
        temperature_threshold: 温度阈值。
        exploration_weight: MCTS 探索权重。
        eval_games_vs_random: 评估次数 vs RandomAgent。
        eval_games_vs_best: 评估次数 vs BestModel。
        win_rate_threshold: 获胜率阈值。
        mcts_sims_eval: MCTS 模拟次数用于评估。
        checkpoint_dir: 模型检查点保存目录。
        device: 训练设备。
    """
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 创建或加载模型
    board_size = GameState.BOARD_SIZE
    current_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
    current_model.to(device)

    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    # 训练迭代
    for iteration in range(iterations):
        print(f"\n{'='*20} Iteration {iteration+1}/{iterations} {'='*20}")
        
        # 1. 自我对弈生成训练数据
        print(f"Self-play phase: generating {num_self_play_games} games using current model...")
        current_model.eval() 
        training_data = self_play(
            model=current_model,
            num_games=num_self_play_games,
            mcts_simulations=num_mcts_simulations,
            temperature_init=temperature_init,
            temperature_final=temperature_final,
            temperature_threshold=temperature_threshold,
            exploration_weight=exploration_weight,
            device=device
        )
        
        # 2. 准备训练数据
        examples = []
        if not training_data:
            print("No training data generated from self-play. Skipping training for this iteration.")
        else:
            for game_states, game_policies, result in training_data:
                for i, state in enumerate(game_states):
                    if state.current_player == Player.WHITE:
                        value = -result
                    else:
                        value = result
                    examples.append((state, game_policies[i], value))
            
            # 3. 训练神经网络
            print(f"Training phase: {len(examples)} examples, {epochs} epochs...")
            current_model.train()
            current_model = train_network(
                model=current_model,
                examples=examples,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
                board_size=board_size
            )

        # 4. 保存当前迭代的模型
        iter_model_path = os.path.join(checkpoint_dir, f"model_iter_{iteration+1}.pt")
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': current_model.state_dict(),
            'board_size': board_size,
            'num_input_channels': NUM_INPUT_CHANNELS
        }, iter_model_path)
        print(f"Model for iteration {iteration+1} saved to {iter_model_path}")

        # 5. 评估当前模型
        print("\nEvaluation phase...")
        current_model.eval()
        challenger_agent = MCTSAgent(current_model, mcts_simulations=mcts_sims_eval, device=device)
        random_opponent = RandomAgent()

        print(f"Evaluating challenger against RandomAgent ({eval_games_vs_random} games)...")
        win_rate_vs_rnd = evaluate_against_agent(
            challenger_agent, random_opponent, eval_games_vs_random, device
        )
        print(f"Challenger win rate vs RandomAgent: {win_rate_vs_rnd:.2%}")

        if win_rate_vs_rnd > win_rate_threshold:
            print(f"Challenger passed RandomAgent threshold ({win_rate_threshold:.0%}). Comparing to best model...")
            if not os.path.exists(best_model_path):
                print(f"No existing best_model.pt. Current model becomes the best.")
                shutil.copy(iter_model_path, best_model_path)
                print(f"Best model updated: {best_model_path}")
            else:
                print(f"Loading best_model.pt for comparison...")
                best_model_checkpoint = torch.load(best_model_path, map_location=device)
                best_model_eval = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
                best_model_eval.load_state_dict(best_model_checkpoint['model_state_dict'])
                best_model_eval.to(device)
                best_model_eval.eval()
                
                best_agent_opponent = MCTSAgent(best_model_eval, mcts_simulations=mcts_sims_eval, device=device)
                
                print(f"Evaluating challenger against BestModel ({eval_games_vs_best} games)...")
                win_rate_vs_best_model = evaluate_against_agent(
                    challenger_agent, best_agent_opponent, eval_games_vs_best, device
                )
                print(f"Challenger win rate vs BestModel: {win_rate_vs_best_model:.2%}")

                if win_rate_vs_best_model > win_rate_threshold:
                    print(f"Challenger passed BestModel threshold. Updating best_model.pt.")
                    shutil.copy(iter_model_path, best_model_path)
                    print(f"Best model updated: {best_model_path}")
                else:
                    print(f"Challenger did not surpass BestModel.")
        else:
            print(f"Challenger did not pass RandomAgent threshold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a chess AI using AlphaZero-style self-play.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations.")
    parser.add_argument("--self_play_games", type=int, default=2, help="Number of self-play games per iteration.")
    parser.add_argument("--mcts_simulations", type=int, default=20, help="Number of MCTS simulations per move for self-play.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs per iteration.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_eval_test", help="Directory to save model checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--eval_games_vs_random", type=int, default=4, help="Games vs RandomAgent.")
    parser.add_argument("--eval_games_vs_best", type=int, default=4, help="Games vs BestModel.")
    parser.add_argument("--win_rate_threshold", type=float, default=0.55, help="Win rate to beat opponent.")
    parser.add_argument("--mcts_sims_eval", type=int, default=20, help="MCTS sims for evaluation agent.")

    args = parser.parse_args()
    
    print(f"Training on device: {args.device}")
    print(f"Training configuration: {args}")
    
    # 运行训练流程
    train_pipeline(
        iterations=args.iterations,
        num_self_play_games=args.self_play_games,
        num_mcts_simulations=args.mcts_simulations,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_games_vs_random=args.eval_games_vs_random,
        eval_games_vs_best=args.eval_games_vs_best,
        win_rate_threshold=args.win_rate_threshold,
        mcts_sims_eval=args.mcts_sims_eval,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    ) 