import math
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from src.game_state import GameState, Player, Phase
from src.move_generator import generate_all_legal_moves, apply_move, MoveType
from src.neural_network import ChessNet, state_to_tensor, get_move_probabilities

# 保证所有地方都能用到move_to_key


def move_to_key(move):
    if isinstance(move, dict):
        return tuple(sorted((k, move_to_key(v)) for k, v in move.items()))
    elif isinstance(move, list):
        return tuple(move_to_key(x) for x in move)
    else:
        return move


class MCTSNode:
    """
    表示蒙特卡洛树搜索中的一个节点。
    """

    def __init__(
        self,
        state: GameState,
        parent=None,
        move: Optional[MoveType] = None,
        prior: float = 0.0,
        player_to_act: Optional[Player] = None,
    ):
        self.state = state
        self.parent = parent
        self.move = move  # 从父节点到这个节点的动作
        self.prior = prior  # 从策略网络得到的先验概率

        # 如果没有指定 player_to_act，则使用 state.current_player
        self.player_to_act = (
            player_to_act if player_to_act is not None else state.current_player
        )

        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.value_sum = 0.0  # 累积价值
        self.expanded = False  # 是否已经扩展
        self.terminal = False  # 是否是终局节点
        self.terminal_value = None  # 终局节点的价值

    def value(self) -> float:
        """
        返回节点的平均价值。
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, legal_moves: List[MoveType], move_probs: List[float]) -> None:
        """
        使用给定的合法动作和概率扩展节点。
        """
        self.expanded = True

        for move, prob in zip(legal_moves, move_probs):
            # 创建新状态
            new_state = apply_move(self.state.copy(), move)
            # 创建子节点
            child = MCTSNode(
                state=new_state,
                parent=self,
                move=move,
                prior=prob,
                player_to_act=new_state.current_player,
            )
            self.children.append(child)

    def is_fully_expanded(self) -> bool:
        """
        检查节点是否已完全扩展。
        """
        return self.expanded

    def is_terminal(self) -> bool:
        """
        检查节点是否是终局节点。
        """
        return self.terminal

    def set_terminal(self, value: float) -> None:
        """
        将节点标记为终局节点并设置其价值。
        """
        self.terminal = True
        self.terminal_value = value

    def backpropagate(self, value: float) -> None:
        """
        将价值反向传播到根节点。
        """
        # 更新当前节点
        self.visit_count += 1
        self.value_sum += value

        # 递归更新父节点
        if self.parent:
            # 从父节点的角度来看，价值需要取反
            self.parent.backpropagate(-value)

    def get_best_child(self, exploration_weight: float = 1.0) -> "MCTSNode":
        """
        使用 PUCT 公式选择最佳子节点。
        PUCT = Q(s,a) + U(s,a)
        其中 Q(s,a) 是节点的平均价值，U(s,a) 是探索项。
        """
        if not self.children:
            raise ValueError("Node has no children")

        # 计算所有子节点的 PUCT 值
        puct_values = []
        for child in self.children:
            # 计算探索项
            # U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            # 其中 c_puct 是探索权重，P(s,a) 是先验概率，N(s) 是父节点的访问次数，N(s,a) 是子节点的访问次数
            exploration = (
                exploration_weight
                * child.prior
                * math.sqrt(self.visit_count)
                / (1 + child.visit_count)
            )
            # PUCT = Q(s,a) + U(s,a)
            puct = child.value() + exploration
            puct_values.append(puct)

        # 返回 PUCT 值最大的子节点
        return self.children[np.argmax(puct_values)]

    def get_visit_count_policy(self) -> Tuple[List[MoveType], np.ndarray]:
        """
        返回基于访问次数的策略。
        """
        if not self.children:
            raise ValueError("Node has no children")

        moves = [child.move for child in self.children]
        visit_counts = np.array([child.visit_count for child in self.children])

        # 将访问次数归一化为概率分布
        policy = visit_counts / np.sum(visit_counts)

        return moves, policy


class MCTS:
    """
    蒙特卡洛树搜索算法。
    """

    def __init__(
        self,
        model: ChessNet,
        num_simulations: int = 800,
        exploration_weight: float = 1.0,
        temperature: float = 1.0,
        device: str = "cpu",
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.temperature = temperature
        self.device = device
        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.model.to(device)
        self.model.eval()  # 设置为评估模式

    def search(self, state: GameState) -> Tuple[List[MoveType], np.ndarray]:
        """
        执行蒙特卡洛树搜索，返回基于访问次数的策略。
        """
        # 创建根节点
        root = MCTSNode(state)

        # 初始化统计字典：{move: {"black": 0, "white": 0, "draw": 0}}
        move_result_stats = {}
        # 先扩展根节点，获得所有第一步着法
        if not root.is_fully_expanded():
            self._expand_and_evaluate(root)
        for child in root.children:
            move_key = move_to_key(child.move)
            move_result_stats[move_key] = {"black": 0, "white": 0, "draw": 0}

        # 执行指定次数的模拟
        for _ in range(self.num_simulations):
            path = [root]
            node = root
            # 选择叶节点并记录路径
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.get_best_child(self.exploration_weight)
                path.append(node)
            leaf = node
            # 如果叶节点不是终局节点，则扩展并评估
            if not leaf.is_terminal():
                value_estimate = self._expand_and_evaluate(leaf)
                if leaf.is_terminal():
                    terminal_value = leaf.terminal_value
                else:
                    terminal_value = value_estimate
            else:
                terminal_value = leaf.terminal_value
            # 反向传播
            if not leaf.is_terminal():
                leaf.backpropagate(value_estimate)
            else:
                leaf.backpropagate(leaf.terminal_value)
            # 统计：只统计第一步着法
            if len(path) > 1:
                first_child = path[1]
                move_key = move_to_key(first_child.move)
                # 终局值：1.0=黑胜，-1.0=白胜，0.0=平局
                if terminal_value == 1.0:
                    move_result_stats[move_key]["black"] += 1
                elif terminal_value == -1.0:
                    move_result_stats[move_key]["white"] += 1
                else:
                    move_result_stats[move_key]["draw"] += 1
        # 根据访问次数生成策略
        moves, policy = root.get_visit_count_policy()
        # 应用温度
        if self.temperature != 1.0:
            policy = np.power(policy, 1.0 / self.temperature)
            policy /= np.sum(policy)  # 重新归一化
        # 打印统计结果
        print("\n===== MCTS模拟统计结果（每个着法） =====")
        best_black = (None, 0)
        best_white = (None, 0)
        for move in moves:
            move_key = move_to_key(move)
            stats = move_result_stats.get(move_key, {"black": 0, "white": 0, "draw": 0})
            total = stats["black"] + stats["white"] + stats["draw"]
            black_rate = stats["black"] / total if total > 0 else 0
            white_rate = stats["white"] / total if total > 0 else 0
            print(
                f"着法: {move}, 黑胜: {stats['black']}, 白胜: {stats['white']}, 平局: {stats['draw']}, 黑胜率: {black_rate:.3f}, 白胜率: {white_rate:.3f}"
            )
            if black_rate > best_black[1]:
                best_black = (move, black_rate)
            if white_rate > best_white[1]:
                best_white = (move, white_rate)
        print(f"\n黑胜率最高着法: {best_black[0]}, 胜率: {best_black[1]:.3f}")
        print(f"白胜率最高着法: {best_white[0]}, 胜率: {best_white[1]:.3f}")
        print("====================================\n")
        return moves, policy

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        从根节点开始，根据 PUCT 公式选择子节点，直到达到叶节点。
        """
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.get_best_child(self.exploration_weight)
        return node

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        扩展节点并使用神经网络评估其价值。
        """
        # 获取当前状态的所有合法动作
        # 检查是否游戏已结束
        if node.state.is_game_over():
            winner = node.state.get_winner()
            value = 1.0 if winner == node.player_to_act else -1.0
            node.set_terminal(value)
            return value

        legal_moves = generate_all_legal_moves(node.state)

        # 如果没有合法动作，则从规则上视为当前玩家失败
        if not legal_moves:
            node.set_terminal(-1.0)
            return -1.0

        # 使用神经网络评估当前状态
        with torch.no_grad():
            # 将状态转换为张量
            input_tensor = state_to_tensor(node.state, node.player_to_act).to(
                self.device
            )
            # 获取神经网络的输出
            log_p1, log_p2, log_pmc, value = self.model(input_tensor)
            # 将对数概率转换为概率
            log_p1 = log_p1.squeeze(0)
            log_p2 = log_p2.squeeze(0)
            log_pmc = log_pmc.squeeze(0)

            # 获取每个合法动作的概率
            # move_probs 是 softmax 后的概率， raw_log_probs 是 softmax 前的原始分数
            move_probs, _ = (
                get_move_probabilities(  # _ placeholder for raw_log_probs as we don't need it here now
                    log_p1,
                    log_p2,
                    log_pmc,
                    legal_moves,
                    node.state.BOARD_SIZE,
                    self.device,
                )
            )

        # 如果是根节点且启用了Dirichlet噪声，混合探索噪声
        if self.add_dirichlet_noise and node.parent is None:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(move_probs))
            move_probs = (1 - self.dirichlet_epsilon) * np.array(
                move_probs
            ) + self.dirichlet_epsilon * noise
            move_probs = (move_probs / move_probs.sum()).tolist()

        # 扩展节点
        node.expand(legal_moves, move_probs)

        # 返回神经网络评估值，但保持 visit_count 和 value_sum 为 0，
        # 让 backpropagate 负责第一次更新
        return value.item()


def self_play(
    model: ChessNet,
    num_games: int = 1,
    mcts_simulations: int = 800,
    temperature_init: float = 1.0,
    temperature_final: float = 0.1,
    temperature_threshold: int = 10,
    exploration_weight: float = 1.0,
    device: str = "cpu",
) -> List[Tuple[List[GameState], List[np.ndarray], float]]:
    """
    执行自我对弈，生成训练数据。

    返回值是一个列表，每个元素是一个元组 (states, policies, value)，
    其中 states 是游戏中的所有状态，policies 是 MCTS 生成的策略，
    value 是游戏结果 (1 表示黑方胜，-1 表示白方胜)。
    """
    training_data = []

    for game_idx in range(num_games):
        print(f"Starting self-play game {game_idx + 1}/{num_games}")

        # 创建 MCTS
        mcts = MCTS(
            model=model,
            num_simulations=mcts_simulations,
            exploration_weight=exploration_weight,
            device=device,
            add_dirichlet_noise=True,
        )

        # 初始化游戏状态
        state = GameState()

        # 记录游戏中的所有状态和策略
        game_states = []
        game_policies = []

        # 记录每一步的动作
        move_count = 0

        # 游戏循环
        while True:
            # 确定当前温度
            if move_count < temperature_threshold:
                temperature = temperature_init
            else:
                temperature = temperature_final

            mcts.temperature = temperature

            # 执行 MCTS 搜索
            moves, policy = mcts.search(state)

            # ---- 调试打印：当前局面所有合法走法及MCTS最终策略 ----
            print(
                f"\n--- Self-Play Move {move_count+1} for Player {state.current_player} (Game {game_idx+1}) ---"
            )

            print(
                f"MCTS found {len(moves)} legal moves with the following policy distribution (temperature: {mcts.temperature:.2f}):"
            )
            if not moves:
                print("  No legal moves found by MCTS from this state.")
            else:
                # 为了更好的可读性，可以排序或只显示概率较高的部分
                sorted_moves_indices = np.argsort(policy)[
                    ::-1
                ]  # Sort by policy descending
                for i in range(len(sorted_moves_indices)):
                    idx = sorted_moves_indices[i]
                    print(
                        f"  {i+1}. Move: {moves[idx]}, Policy Prob: {policy[idx]:.4f}"
                    )

            # ---- 结束调试打印 ----

            # 记录当前状态和策略
            game_states.append(state.copy())
            game_policies.append(policy)

            # 根据策略选择动作
            move_idx = np.random.choice(len(moves), p=policy)
            move = moves[move_idx]

            # 应用动作
            state = apply_move(state, move)

            move_count += 1

            print(state)
            winner = state.get_winner()
            if winner is not None:
                result = 1.0 if winner == Player.BLACK else -1.0 if winner == Player.WHITE else 0.0
                training_data.append((game_states, game_policies, result))
                break

            # 如果游戏进行了太多步，也结束游戏
            if move_count > 500:
                print(
                    f"Game {game_idx + 1} reached move limit ({move_count} moves), ending as a draw."
                )
                print("Final state before ending due to move limit:")
                print(state)
                training_data.append((game_states, game_policies, 0.0))
                break

    return training_data


if __name__ == "__main__":
    # 测试 MCTS
    from src.neural_network import ChessNet

    # 创建模型
    board_size = GameState.BOARD_SIZE
    model = ChessNet(board_size=board_size)

    # 创建 MCTS
    mcts = MCTS(model=model, num_simulations=10)  # 为了快速测试，只使用10次模拟

    # 创建初始状态
    state = GameState()

    # 执行 MCTS 搜索
    moves, policy = mcts.search(state)

    print("Initial state:")
    print(state)
    print("\nMCTS policy:")
    for move, prob in zip(moves, policy):
        print(f"Move: {move}, Probability: {prob:.4f}")

    # 选择概率最高的动作
    best_move_idx = np.argmax(policy)
    best_move = moves[best_move_idx]

    print(f"\nBest move: {best_move}")

    # 应用最佳动作
    new_state = apply_move(state, best_move)

    print("\nState after best move:")
    print(new_state)
