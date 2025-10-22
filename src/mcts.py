import math
import time
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
        使用 PUCT 公式选择最佳子节点：
          PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
        其中：
          - Q(s,a) = child.value() = value_sum / visit_count（根执手视角）
          - P(s,a) = child.prior（先验概率，根处可能含Dirichlet）
          - N_parent = max(1, self.visit_count)  # 防止首轮 sqrt(0)
          - N(s,a) = child.visit_count
        """
        if not self.children:
            raise ValueError("Node has no children")

        parent_N = max(1, self.visit_count)

        best_child = None
        best_score = -float("inf")

        for child in self.children:
            Q = child.value()                 # 平均价值（根执手视角）
            P = child.prior                   # 先验
            U = exploration_weight * P * math.sqrt(parent_N) / (1 + child.visit_count)
            puct = Q + U
            if puct > best_score:
                best_score = puct
                best_child = child

        return best_child

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
        batch_K: int = 16,
        verbose: bool = False,
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
        self.batch_K = max(1, batch_K)
        self.verbose = verbose
        self.root: Optional[MCTSNode] = None
        self.last_profile: Dict[str, Any] = {}
        self._reset_profile_counters()

    def _reset_profile_counters(self) -> None:
        self._profile_forward_batches = 0
        self._profile_forward_samples = 0
        self._profile_forward_time = 0.0

    # def search(self, state: GameState) -> Tuple[List[MoveType], np.ndarray]:
    #     """
    #     执行 AlphaZero 式 MCTS，返回基于访问次数的策略（根执手视角）。
    #     支持 root 复用（若 self.root 与 state 匹配则复用）与 verbose 诊断打印。
    #     """
    #     # ========== 1) 建立或复用根节点 ==========
    #     if getattr(self, "root", None) is None or not self._same_state(self.root.state, state):
    #         root = MCTSNode(state)
    #         if not root.is_fully_expanded():
    #             self._expand_and_evaluate(root)
    #         self.root = root
    #     else:
    #         root = self.root

    #     if not root.is_fully_expanded():
    #         self._expand_and_evaluate(root)

    #     # 若已是终局（或无合法着法在 _expand_and_evaluate 中被判终局），直接返回空
    #     if root.is_terminal() or not root.children:
    #         return [], np.array([], dtype=float)

    #     root_player = root.state.current_player

    #     # ========== 2) 进行指定次数的模拟 ==========
    #     for sim in range(self.num_simulations):
    #         path = [root]
    #         node = root

    #         # selection：按 PUCT 向下走到叶子
    #         while node.is_fully_expanded() and not node.is_terminal():
    #             node = node.get_best_child(self.exploration_weight)
    #             path.append(node)

    #         leaf = node

    #         # expansion + evaluation：扩展并用网络估值（非终局）
    #         if not leaf.is_terminal():
    #             value_estimate = self._expand_and_evaluate(leaf)
    #             # 回传（leaf 可能在扩展时判成了终局，但 backpropagate 一样成立）
    #             leaf.backpropagate(value_estimate if not leaf.is_terminal() else leaf.terminal_value)
    #         else:
    #             # 已是终局，直接回传终局值
    #             leaf.backpropagate(leaf.terminal_value)

    #     # ========== 3) 基于访问次数得到根策略（分母保护） ==========
    #     moves = [child.move for child in root.children]
    #     visit_counts = np.array([child.visit_count for child in root.children], dtype=float)
    #     total = visit_counts.sum()
    #     if total <= 0:
    #         policy = np.ones_like(visit_counts) / len(visit_counts)
    #     else:
    #         policy = visit_counts / total

    #     # ========== 4) 温度（需要时） ==========
    #     if self.temperature != 1.0:
    #         policy = np.power(policy, 1.0 / self.temperature)
    #         policy /= policy.sum()

    #     # ========== 5) verbose 模式：打印诊断信息 ==========
    #     if self.verbose:
    #         print(f"\n===== MCTS 诊断（根执手: {root_player.name}, 候选 {len(root.children)} 个） =====")
    #         parent_N = max(1, root.visit_count)  # 与选择时保持一致

    #         rows = []
    #         for idx, child in enumerate(root.children, start=1):
    #             Q = child.value()
    #             N = child.visit_count
    #             P = child.prior
    #             U = self.exploration_weight * P * math.sqrt(parent_N) / (1 + N)
    #             puct = Q + U
    #             rows.append((idx, child.move, N, Q, P, U, puct))

    #         # 按访问次数排序，方便观察主要分支
    #         rows.sort(key=lambda r: r[2], reverse=True)

    #         for idx, mv, N, Q, P, U, puct in rows:
    #             print(f"[{idx:02d}] N={N:4d} | Q={Q:+.3f} | P={P:.3f} | U={U:.3f} | Q+U={puct:+.3f} | move={mv}")

    #         print("\nPolicy (from visit counts):")
    #         sorted_idx = np.argsort(policy)[::-1]
    #         topk = min(10, len(sorted_idx))
    #         for rank in range(topk):
    #             i = sorted_idx[rank]
    #             print(f"  #{rank+1:02d} π={policy[i]:.3f}  move={moves[i]}")
    #         print("====================================\n")

    #     # ========== 6) 返回结果 ==========
    #     return moves, policy

    def search(self, state: GameState) -> Tuple[List[MoveType], np.ndarray]:
        """
        AlphaZero-style MCTS:
        - Root reuse: if self.root matches `state`, reuse it; otherwise rebuild once.
        - Batched inference: collect up to batch_K leaves per wave, evaluate NN once.
        - Verbose diagnostics (Q/N/P/U/PUCT + policy Top-K) when self.verbose is True.
        Requires: self.batch_K (int), self.verbose (bool), self.device (str / torch.device)
        """

        # ========== 0) 建立或复用根节点 ==========
        if getattr(self, "root", None) is None or not self._same_state(self.root.state, state):
            root = MCTSNode(state)
            if not root.is_fully_expanded():
                self._expand_and_evaluate(root)
            self.root = root
        else:
            root = self.root

        # 若已是终局（或无合法着法在 _expand_and_evaluate 中被判终局），直接返回空
        if root.is_terminal() or not root.children:
            return [], np.array([], dtype=float)

        self._reset_profile_counters()

        root_player = root.state.current_player
        sims_done = 0
        K = max(1, getattr(self, "batch_K", 1))

        # ========== 1) 模拟循环（批量 selection + 批量前向） ==========
        while sims_done < self.num_simulations:
            leaves = []
            paths = []

            # ---- 1.1 收集最多 K 个叶子（每个叶子一条 path）----
            to_collect = min(K, self.num_simulations - sims_done)
            for _ in range(to_collect):
                node = root
                path = [root]
                while node.is_fully_expanded() and not node.is_terminal():
                    node = node.get_best_child(self.exploration_weight)
                    path.append(node)
                leaves.append(node)
                paths.append(path)
                sims_done += 1

            # ---- 1.2 终局快速处理 + 准备批量前向的张量 ----
            tensors = []
            work_idx = []          # 需要 NN 评估的叶子索引（在 leaves 的下标）
            legal_moves_batch = [] # 与 work_idx 对齐

            for i, leaf in enumerate(leaves):
                if leaf.is_terminal():
                    leaf.backpropagate(leaf.terminal_value)
                    continue

                # 规则终局检查
                if leaf.state.is_game_over():
                    winner = leaf.state.get_winner()
                    v = 0.0 if winner is None else (1.0 if winner == leaf.player_to_act else -1.0)
                    leaf.set_terminal(v)
                    leaf.backpropagate(v)
                    continue

                legal_moves = generate_all_legal_moves(leaf.state)
                if not legal_moves:
                    # 无合法走法 -> 当前执手视角失败
                    leaf.set_terminal(-1.0)
                    leaf.backpropagate(-1.0)
                    continue

                t = state_to_tensor(leaf.state, leaf.player_to_act)  # (1,C,H,W) on CPU
                tensors.append(t)
                work_idx.append(i)
                legal_moves_batch.append(legal_moves)

            if not tensors:
                # 这一批全是终局或无合法走法，继续下一批
                continue

            batch = torch.cat(tensors, dim=0).to(self.device)  # (B,C,H,W)

            # ---- 1.3 批量前向 ----
            with torch.inference_mode():
                log_p1, log_p2, log_pmc, values = self.model(batch)  # shapes: (B,HW), (B,HW), (B,HW), (B,1)
                values = values.squeeze(1)  # (B,)

            # ---- 1.4 逐个叶子：展开 + 先验 + 回传 ----
            for bi, i_leaf in enumerate(work_idx):
                leaf = leaves[i_leaf]
                legal_moves = legal_moves_batch[bi]

                lp1 = log_p1[bi]   # (HW,)
                lp2 = log_p2[bi]   # (HW,)
                lpc = log_pmc[bi]  # (HW,)

                # 用“可反传”的组合器计算先验分布；此处处于 inference_mode，仅作 prior
                move_probs, _ = get_move_probabilities(
                    lp1, lp2, lpc, legal_moves, leaf.state.BOARD_SIZE, device=self.device
                )

                # 根处混 Dirichlet 噪声（如启用）
                if self.add_dirichlet_noise and leaf.parent is None:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * len(move_probs))
                    move_probs = (1 - self.dirichlet_epsilon) * np.array(move_probs) + self.dirichlet_epsilon * noise
                    move_probs = (move_probs / move_probs.sum()).tolist()

                leaf.expand(legal_moves, move_probs)

                v = float(values[bi].item())
                # （可选）数值裁剪，避免初始化早期的尖峰
                # v = max(-1.0, min(1.0, v))
                leaf.backpropagate(v)

        # ========== 2) 基于访问次数得到根策略 ==========
        moves = [child.move for child in root.children]
        visit_counts = np.array([child.visit_count for child in root.children], dtype=float)
        total = visit_counts.sum()
        policy = (np.ones_like(visit_counts) / len(visit_counts)) if total <= 0 else (visit_counts / total)

        # ========== 3) 温度（需要时；含零和保护） ==========
        if self.temperature != 1.0:
            policy = np.power(policy, 1.0 / self.temperature)
            s = policy.sum()
            policy = policy / s if s > 0 else np.ones_like(policy) / len(policy)

        # ========== 4) verbose 诊断打印 ==========
        if self.verbose:
            print(f"\n===== MCTS 诊断（根执手: {root_player.name}, 候选 {len(root.children)} 个） =====")
            parent_N = max(1, root.visit_count)
            rows = []
            for idx, child in enumerate(root.children, start=1):
                Q = child.value()
                N = child.visit_count
                P = child.prior
                U = self.exploration_weight * P * math.sqrt(parent_N) / (1 + N)
                puct = Q + U
                rows.append((idx, child.move, N, Q, P, U, puct))
            rows.sort(key=lambda r: r[2], reverse=True)
            for idx, mv, N, Q, P, U, puct in rows:
                print(f"[{idx:02d}] N={N:4d} | Q={Q:+.3f} | P={P:.3f} | U={U:.3f} | Q+U={puct:+.3f} | move={mv}")

            print("\nPolicy (from visit counts):")
            sorted_idx = np.argsort(policy)[::-1]
            topk = min(10, len(sorted_idx))
            for rank in range(topk):
                i = sorted_idx[rank]
                print(f"  #{rank+1:02d} π={policy[i]:.3f}  move={moves[i]}")
            print("====================================\n")

        return moves, policy


    def _same_state(self, a: GameState, b: GameState) -> bool:
        """比较两个 GameState 是否等价（用于 root 复用）。"""
        if a is b:
            return True

        if (
            a.phase != b.phase
            or a.current_player != b.current_player
            or a.forced_removals_done != b.forced_removals_done
            or a.move_count != b.move_count
            or a.pending_marks_required != b.pending_marks_required
            or a.pending_marks_remaining != b.pending_marks_remaining
            or a.pending_captures_required != b.pending_captures_required
            or a.pending_captures_remaining != b.pending_captures_remaining
        ):
            return False

        if a.marked_black != b.marked_black or a.marked_white != b.marked_white:
            return False

        if a.board != b.board:
            return False

        return True

    def advance_root(self, move: MoveType) -> None:
        """
        在执行完一个实际动作后，将对应的子节点提升为新的根节点，以便复用搜索树。
        """
        if self.root is None:
            return

        target_key = move_to_key(move)
        for child in self.root.children:
            if move_to_key(child.move) == target_key:
                child.parent = None
                self.root = child
                return

        # 未找到匹配的子节点，退化为重建
        self.root = None


    # def _expand_and_evaluate(self, node: MCTSNode) -> float:
    #     """
    #     扩展节点并使用神经网络评估其价值。
    #     """
    #     # 获取当前状态的所有合法动作
    #     # 检查是否游戏已结束
    #     if node.state.is_game_over():
    #         winner = node.state.get_winner()
    #         if winner is None:
    #             value = 0.0
    #         else:
    #             value = 1.0 if winner == node.player_to_act else -1.0
    #         node.set_terminal(value)
    #         return value

    #     legal_moves = generate_all_legal_moves(node.state)

    #     # 如果没有合法动作，则从规则上视为当前玩家失败
    #     if not legal_moves:
    #         node.set_terminal(-1.0)
    #         return -1.0

    #     # 使用神经网络评估当前状态
    #     with torch.no_grad():
    #         # 将状态转换为张量
    #         input_tensor = state_to_tensor(node.state, node.player_to_act).to(
    #             self.device
    #         )
    #         # 获取神经网络的输出
    #         log_p1, log_p2, log_pmc, value = self.model(input_tensor)
    #         # 将对数概率转换为概率
    #         log_p1 = log_p1.squeeze(0)
    #         log_p2 = log_p2.squeeze(0)
    #         log_pmc = log_pmc.squeeze(0)

    #         # 获取每个合法动作的概率
    #         # move_probs 是 softmax 后的概率， raw_log_probs 是 softmax 前的原始分数
    #         move_probs, _ = (
    #             get_move_probabilities(  # _ placeholder for raw_log_probs as we don't need it here now
    #                 log_p1,
    #                 log_p2,
    #                 log_pmc,
    #                 legal_moves,
    #                 node.state.BOARD_SIZE,
    #                 self.device,
    #             )
    #         )

    #     # 如果是根节点且启用了Dirichlet噪声，混合探索噪声
    #     if self.add_dirichlet_noise and node.parent is None:
    #         noise = np.random.dirichlet([self.dirichlet_alpha] * len(move_probs))
    #         move_probs = (1 - self.dirichlet_epsilon) * np.array(
    #             move_probs
    #         ) + self.dirichlet_epsilon * noise
    #         move_probs = (move_probs / move_probs.sum()).tolist()

    #     # 扩展节点
    #     node.expand(legal_moves, move_probs)

    #     # 返回神经网络评估值，但保持 visit_count 和 value_sum 为 0，
    #     # 让 backpropagate 负责第一次更新
    #     return value.item()
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        单节点扩展 + 评估：
        - 仅在根初始化/回退等需要单样本时使用；
        - 批量推理场景下，主要逻辑在 search() 里完成，这里只是备用通道。
        返回：value（根执手视角，float）
        """
        # 1) 终局判断（规则）
        if node.state.is_game_over():
            winner = node.state.get_winner()
            v = 0.0 if winner is None else (1.0 if winner == node.player_to_act else -1.0)
            node.set_terminal(v)
            return v

        # 2) 合法着法
        legal_moves = generate_all_legal_moves(node.state)
        if not legal_moves:
            # 无合法着法 -> 当前执手失败
            node.set_terminal(-1.0)
            return -1.0

        # 3) 单样本前向（推理友好）
        with torch.inference_mode():  # 比 no_grad 更省开销
            inp = state_to_tensor(node.state, node.player_to_act).to(self.device, non_blocking=True)
            log_p1, log_p2, log_pmc, value = self.model(inp)   # (1, HW), (1, HW), (1, HW), (1, 1)
            log_p1 = log_p1.squeeze(0)  # (HW,)
            log_p2 = log_p2.squeeze(0)
            log_pmc = log_pmc.squeeze(0)
            v = float(value.squeeze(0).squeeze(0).item())  # 转 python float，避免外面再 .item()

        # 4) 先验分布（对合法着法组合打分）
        move_probs, _ = get_move_probabilities(
            log_p1, log_p2, log_pmc, legal_moves, node.state.BOARD_SIZE, device=self.device
        )

        # 数值保护：全零/NaN 等退化情况
        if not move_probs or not np.isfinite(np.sum(move_probs)):
            move_probs = [1.0 / len(legal_moves)] * len(legal_moves)

        # 5) 根处混合 Dirichlet 噪声（如启用）
        if self.add_dirichlet_noise and node.parent is None and len(move_probs) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(move_probs))
            mix = (1 - self.dirichlet_epsilon) * np.array(move_probs) + self.dirichlet_epsilon * noise
            s = mix.sum()
            move_probs = (mix / s).tolist() if s > 0 else [1.0 / len(move_probs)] * len(move_probs)

        # 6) 扩展
        node.expand(legal_moves, move_probs)

        # 7) 返回估值（让调用方 backpropagate）
        # 可选：早期训练可进行轻微裁剪防爆
        # v = max(-1.0, min(1.0, v))
        return v



def self_play(
    model: ChessNet,
    num_games: int = 1,
    mcts_simulations: int = 800,
    temperature_init: float = 1.0,
    temperature_final: float = 0.1,
    temperature_threshold: int = 10,
    exploration_weight: float = 1.0,
    device: str = "cpu",
    add_dirichlet_noise: bool = True,
    mcts_verbose: Optional[bool] = None,
    verbose: bool = False,
) -> List[Tuple[List[GameState], List[np.ndarray], float]]:
    """
    执行自我对弈，生成训练数据。

    返回值是一个列表，每个元素是一个元组 (states, policies, value)，
    其中 states 是游戏中的所有状态，policies 是 MCTS 生成的策略，
    value 是游戏结果 (1 表示黑方胜，-1 表示白方胜)。
    """
    if mcts_verbose is None:
        mcts_verbose = verbose
    log = print if verbose else (lambda *args, **kwargs: None)
    training_data = []

    for game_idx in range(num_games):
        log(f"Starting self-play game {game_idx + 1}/{num_games}")

        # 创建 MCTS
        mcts = MCTS(
            model=model,
            num_simulations=mcts_simulations,
            exploration_weight=exploration_weight,
            device=device,
            add_dirichlet_noise=add_dirichlet_noise,
            verbose=mcts_verbose,
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

            if verbose:
                log(
                    f"\n--- Self-Play Move {move_count + 1} for Player {state.current_player} (Game {game_idx + 1}) ---"
                )
                log(
                    f"MCTS found {len(moves)} legal moves with the following policy distribution (temperature: {mcts.temperature:.2f}):"
                )
                if not moves:
                    log("  No legal moves found by MCTS from this state.")
                else:
                    sorted_moves_indices = np.argsort(policy)[::-1]
                    for i in range(len(sorted_moves_indices)):
                        idx = sorted_moves_indices[i]
                        log(f"  {i + 1}. Move: {moves[idx]}, Policy Prob: {policy[idx]:.4f}")



            # 记录当前状态和策略
            game_states.append(state.copy())
            game_policies.append(policy)

            # 根据策略选择动作
            move_idx = np.random.choice(len(moves), p=policy)
            move = moves[move_idx]

            # 应用动作
            state = apply_move(state, move)
            mcts.advance_root(move)

            move_count += 1

            if verbose:
                log(state)
            winner = state.get_winner()
            if winner is not None:
                result = 1.0 if winner == Player.BLACK else -1.0 if winner == Player.WHITE else 0.0
                training_data.append((game_states, game_policies, result))
                break

            # 如果游戏进行了太多步，也结束游戏
            if move_count > 500:
                if verbose:
                    log(f"Game {game_idx + 1} reached move limit ({move_count} moves), ending as a draw.")
                    log("Final state before ending due to move limit:")
                    log(state)
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
