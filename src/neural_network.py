import torch
import torch.nn as nn
import torch.nn.functional as F
from src.game_state import GameState, Player, Phase # 导入 Player 和 Phase
from src.move_generator import generate_all_legal_moves, MoveType # 导入走法生成器和类型

# 增加输入通道数量
NUM_INPUT_CHANNELS = 9 # 1.己方棋子, 2.对方棋子, 3.己方被标记, 4.对方被标记, 5-9.Phase(one-hot)

def state_to_tensor(state: GameState, player_to_act: Player) -> torch.Tensor:
    """
    将游戏状态转换为神经网络输入张量。
    
    Args:
        state: 游戏状态。
        player_to_act: 当前执行动作的玩家，用于确定视角。
        
    Returns:
        一个形状为 (1, NUM_INPUT_CHANNELS, board_size, board_size) 的张量。
    """
    board_size = state.BOARD_SIZE
    tensor = torch.zeros(NUM_INPUT_CHANNELS, board_size, board_size)
    
    player_val = player_to_act.value
    opponent_val = player_to_act.opponent().value
    
    # 通道 0: 当前玩家棋子
    # 通道 1: 对手玩家棋子
    for r in range(board_size):
        for c in range(board_size):
            if state.board[r][c] == player_val:
                tensor[0, r, c] = 1
            elif state.board[r][c] == opponent_val:
                tensor[1, r, c] = 1
    
    # 通道 2: 己方 (player_to_act) 被标记的棋子
    # 通道 3: 对方 (opponent) 被标记的棋子
    marked_self = state.marked_black if player_to_act == Player.BLACK else state.marked_white
    marked_opponent = state.marked_white if player_to_act == Player.BLACK else state.marked_black

    for r, c in marked_self:
        if 0 <= r < board_size and 0 <= c < board_size: # 确保在棋盘内
             tensor[2, r, c] = 1
    
    for r, c in marked_opponent:
        if 0 <= r < board_size and 0 <= c < board_size: # 确保在棋盘内
            tensor[3, r, c] = 1

    # 通道 4-7: 游戏阶段 (独热编码)
    if state.phase == Phase.PLACEMENT:
        tensor[4, :, :] = 1
    elif state.phase == Phase.REMOVAL:
        tensor[5, :, :] = 1
    elif state.phase == Phase.MOVEMENT:
        tensor[6, :, :] = 1
    elif state.phase == Phase.FORCED_REMOVAL:
        tensor[7, :, :] = 1
    elif state.phase == Phase.COUNTER_REMOVAL:
        tensor[8, :, :] = 1
        
    return tensor.unsqueeze(0) # 增加 batch 维度

class ChessNet(nn.Module):
    def __init__(self, board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS, hidden_conv_channels=64):
        super(ChessNet, self).__init__()
        self.board_size = board_size
        self.num_input_channels = num_input_channels

        # 共享的卷积层
        self.conv1 = nn.Conv2d(num_input_channels, hidden_conv_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_conv_channels)
        self.conv2 = nn.Conv2d(hidden_conv_channels, hidden_conv_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_conv_channels * 2)
        # 增加一个卷积层以获取更深层特征
        self.conv3 = nn.Conv2d(hidden_conv_channels * 2, hidden_conv_channels * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_conv_channels * 4)
        self.conv4 = nn.Conv2d(hidden_conv_channels * 4, hidden_conv_channels * 4, kernel_size=3, padding=1) # 再加一层
        self.bn4 = nn.BatchNorm2d(hidden_conv_channels * 4)

        # 策略头 (Policy Heads)
        # 共同的特征提取部分
        self.policy_common_conv = nn.Conv2d(hidden_conv_channels * 4, hidden_conv_channels, kernel_size=1)
        self.policy_common_bn = nn.BatchNorm2d(hidden_conv_channels)

        # Policy Head 1: 主要位置 (例如：落子位置，移动起始位置，移除位置)
        self.policy_pos1_conv = nn.Conv2d(hidden_conv_channels, 2, kernel_size=1) # 输出2个通道，展平后得到棋盘大小
        self.policy_pos1_bn = nn.BatchNorm2d(2)
        self.policy_pos1_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Policy Head 2: 次要位置 (例如：移动目标位置)
        self.policy_pos2_conv = nn.Conv2d(hidden_conv_channels, 2, kernel_size=1)
        self.policy_pos2_bn = nn.BatchNorm2d(2)
        self.policy_pos2_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Policy Head 3: 标记/捕获位置
        self.policy_mark_capture_conv = nn.Conv2d(hidden_conv_channels, 2, kernel_size=1)
        self.policy_mark_capture_bn = nn.BatchNorm2d(2)
        self.policy_mark_capture_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # 价值头 (Value Head)
        self.value_conv_common = nn.Conv2d(hidden_conv_channels * 4, hidden_conv_channels, kernel_size=1) # 价值头的共同卷积
        self.value_bn_common = nn.BatchNorm2d(hidden_conv_channels)
        self.value_conv1 = nn.Conv2d(hidden_conv_channels, 1, kernel_size=1) # 进一步降维
        self.value_bn1 = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x 的形状预期为 (batch_size, num_input_channels, board_size, board_size)

        # 共享卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        shared_features = F.relu(self.bn4(self.conv4(x))) # (batch_size, hidden_conv_channels*4, board_size, board_size)

        # --- 策略头 ---
        policy_latent = F.relu(self.policy_common_bn(self.policy_common_conv(shared_features))) # (batch_size, hidden_conv_channels, board_size, board_size)

        # policy_pos1
        p1 = F.relu(self.policy_pos1_bn(self.policy_pos1_conv(policy_latent)))
        p1 = p1.view(p1.size(0), -1)
        log_policy_pos1 = F.log_softmax(self.policy_pos1_fc(p1), dim=1) # (batch_size, board_size*board_size)

        # policy_pos2
        p2 = F.relu(self.policy_pos2_bn(self.policy_pos2_conv(policy_latent)))
        p2 = p2.view(p2.size(0), -1)
        log_policy_pos2 = F.log_softmax(self.policy_pos2_fc(p2), dim=1) # (batch_size, board_size*board_size)

        # policy_mark_capture
        pmc = F.relu(self.policy_mark_capture_bn(self.policy_mark_capture_conv(policy_latent)))
        pmc = pmc.view(pmc.size(0), -1)
        log_policy_mark_capture = F.log_softmax(self.policy_mark_capture_fc(pmc), dim=1) # (batch_size, board_size*board_size)

        # --- 价值头 ---
        value_latent = F.relu(self.value_bn_common(self.value_conv_common(shared_features))) # (batch_size, hidden_conv_channels, board_size, board_size)
        v = F.relu(self.value_bn1(self.value_conv1(value_latent))) # (batch_size, 1, board_size, board_size)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)) # 输出范围在 [-1, 1] 之间

        return log_policy_pos1, log_policy_pos2, log_policy_mark_capture, value

def get_move_probabilities(
    log_policy_pos1: torch.Tensor,
    log_policy_pos2: torch.Tensor,
    log_policy_mark_capture: torch.Tensor,
    legal_moves: list, # List of MoveType dicts
    board_size: int,
    device: str = 'cpu' # Added device parameter
) : # Returns list of probabilities and raw log_probs
    """
    Calculates a probability distribution over legal moves.

    Args:
        log_policy_pos1: Log probabilities for the first key position. Shape (board_size*board_size,).
        log_policy_pos2: Log probabilities for the second key position. Shape (board_size*board_size,).
        log_policy_mark_capture: Log probabilities for mark/capture positions. Shape (board_size*board_size,).
        legal_moves: A list of legal move dictionaries from generate_all_legal_moves.
        board_size: The size of the board.
        device: The device to perform tensor operations on ('cpu' or 'cuda').

    Returns:
        A tuple containing:
        - A list of probabilities for each legal move, normalized via softmax.
        - A tensor of the raw combined log-probabilities for each legal move.
    """
    if not legal_moves:
        return [], torch.empty(0, device=device)

    # Ensure policy tensors are 1D
    log_policy_pos1 = log_policy_pos1.view(-1)
    log_policy_pos2 = log_policy_pos2.view(-1)
    log_policy_mark_capture = log_policy_mark_capture.view(-1)

    combined_log_probs = []
    
    # 记录每个走法的信息，用于后处理
    move_types = []  # 用于标记每个走法是否包含标记/提子操作

    def flatten_index(r, c):
        return r * board_size + c

    for move in legal_moves:
        current_log_prob = 0.0
        has_mark_capture = False  # 标记这个走法是否包含标记/提子操作

        phase = move['phase']
        action_type = move['action_type']

        if phase == Phase.PLACEMENT and action_type == 'place':
            r, c = move['position']
            # 添加落子位置的对数概率
            current_log_prob += log_policy_pos1[flatten_index(r, c)].item()
            
            # 修改：如果有标记位置，记录并使用最大值
            if move['mark_positions']:
                has_mark_capture = True
                # 获取所有标记位置的对数概率
                mark_log_probs = []
                for mr, mc in move['mark_positions']:
                    mark_log_prob = log_policy_mark_capture[flatten_index(mr, mc)].item()
                    mark_log_probs.append(mark_log_prob)
                
                if mark_log_probs:
                    # 使用标记位置概率的最大值
                    current_log_prob += max(mark_log_probs)
        
        elif phase == Phase.MOVEMENT and action_type == 'move':
            r_from, c_from = move['from_position']
            r_to, c_to = move['to_position']
            # 添加起始位置和目标位置的对数概率
            current_log_prob += log_policy_pos1[flatten_index(r_from, c_from)].item()
            current_log_prob += log_policy_pos2[flatten_index(r_to, c_to)].item()
            
            # 修改：如果有提子位置，记录并使用最大值
            if move['capture_positions']:
                has_mark_capture = True
                # 获取所有提子位置的对数概率
                capture_log_probs = []
                for cr, cc in move['capture_positions']:
                    capture_log_prob = log_policy_mark_capture[flatten_index(cr, cc)].item()
                    capture_log_probs.append(capture_log_prob)
                
                if capture_log_probs:
                    # 使用提子位置概率的最大值
                    current_log_prob += max(capture_log_probs)

        elif phase == Phase.FORCED_REMOVAL and action_type == 'remove':
            r, c = move['position']
            # Use pos1 for forced removal as it's a primary action at a location
            current_log_prob += log_policy_pos1[flatten_index(r, c)].item()
            
        elif phase == Phase.REMOVAL and action_type == 'process_removal':
            # This is typically the only move in this phase.
            # Assign a neutral log probability (0.0) or a sum of all pos1 (needs thought).
            # For now, if it's the only move, its probability will be 1.0 after softmax.
            # If multiple moves were possible here, this would need refinement.
            # Let's give it a base score; if other moves existed, they'd compete.
            # A simple approach: sum of all log_policy_pos1, implies "any action is fine".
            # This is a placeholder; often for such deterministic single moves, policy isn't strictly needed.
            current_log_prob += 0.0 # Effectively neutral, relies on softmax for normalization if other moves existed.

        elif phase == Phase.MOVEMENT and action_type == 'no_moves_remove':
            has_mark_capture = True
            r, c = move['position'] # Position of opponent's piece to remove
            # Using mark_capture head as it's a removal action not part of a standard move sequence.
            no_moves_remove_log_prob = log_policy_mark_capture[flatten_index(r, c)].item()
            current_log_prob += no_moves_remove_log_prob
        
        elif phase == Phase.COUNTER_REMOVAL and action_type == 'counter_remove':
            r, c = move['position']
            # Counter removals mirror forced removals: use the primary position head
            current_log_prob += log_policy_pos1[flatten_index(r, c)].item()
        
        else:
            # Should not happen with current move generator if all cases are covered
            print(f"Warning: Unhandled move type: Phase {phase}, Action {action_type}")
            current_log_prob += -float('inf') # Penalize heavily

        combined_log_probs.append(current_log_prob)
        move_types.append(has_mark_capture)

    if not combined_log_probs: # Should be redundant due to earlier check
        return [], torch.empty(0, device=device)
    
    # 转换为张量
    combined_log_probs_tensor = torch.tensor(combined_log_probs, dtype=torch.float32, device=device)
    
    # 如果只有一个合法走法，概率为1.0
    if len(legal_moves) == 1:
        return [1.0], combined_log_probs_tensor
    
    # 特殊处理：确保带有标记/提子的走法不会有极低的概率
    # 检查是否有带标记/提子的走法，以及这些走法的概率是否都接近于0
    has_mark_capture_moves = any(move_types)
    if has_mark_capture_moves:
        # 计算softmax前的原始概率
        orig_probs = torch.softmax(combined_log_probs_tensor, dim=0)
        
        # 检查所有带标记/提子的走法，是否所有的概率都小于一个阈值（例如0.001）
        mark_capture_probs = [prob.item() for prob, has_mc in zip(orig_probs, move_types) if has_mc]
        
        # 如果所有带标记/提子的走法概率都很低，我们进行调整
        if mark_capture_probs and max(mark_capture_probs) < 0.001:
            # 对数概率空间的调整：找出不带标记/提子的走法和带标记/提子的走法的最大对数概率
            max_log_prob_no_mc = max([lp for lp, has_mc in zip(combined_log_probs, move_types) if not has_mc], default=-float('inf'))
            max_log_prob_mc = max([lp for lp, has_mc in zip(combined_log_probs, move_types) if has_mc], default=-float('inf'))
            
            # 计算调整因子，使得带标记/提子的走法至少有一定的机会被探索
            # 我们不希望调整太多，只是确保至少有一个带标记/提子的走法有合理的概率
            adjustment = max(0, max_log_prob_no_mc - max_log_prob_mc - 1.0)  # 确保log_prob差距不超过1.0
            
            # 应用调整
            adjusted_log_probs = [lp + adjustment if has_mc else lp for lp, has_mc in zip(combined_log_probs, move_types)]
            combined_log_probs_tensor = torch.tensor(adjusted_log_probs, dtype=torch.float32, device=device)

    # 计算最终概率
    probabilities = torch.softmax(combined_log_probs_tensor, dim=0).tolist()
    
    return probabilities, combined_log_probs_tensor

if __name__ == '__main__':
    board_size = GameState.BOARD_SIZE
    net = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
    
    # 模拟一个输入
    dummy_input = torch.randn(1, NUM_INPUT_CHANNELS, board_size, board_size)
    p1, p2, pmc, v = net(dummy_input)
    
    print("Policy_pos1 output shape:", p1.shape)
    print("Policy_pos2 output shape:", p2.shape)
    print("Policy_mark_capture output shape:", pmc.shape)
    print("Value output shape:", v.shape) 

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    initial_state = GameState()
    input_tensor_black = state_to_tensor(initial_state, Player.BLACK)
    print("Input tensor shape (Black's turn):", input_tensor_black.shape)
    p1_b, p2_b, pmc_b, v_b = net(input_tensor_black)
    print("P1 (B):", p1_b.shape, "P2 (B):", p2_b.shape, "PMC (B):", pmc_b.shape, "V (B):", v_b.item())

    # 模拟白方回合
    initial_state.current_player = Player.WHITE # 假设轮到白方
    # 假设黑方标记了白方的 (0,0) 和 (1,1)
    initial_state.marked_white.add((0,0))
    initial_state.marked_white.add((1,1))
    # 假设白方标记了黑方的 (2,2)
    initial_state.marked_black.add((2,2))
    initial_state.phase = Phase.MOVEMENT # 假设进入移动阶段

    input_tensor_white = state_to_tensor(initial_state, Player.WHITE)
    print("Input tensor shape (White's turn, with marks, movement phase):", input_tensor_white.shape)
    # 打印一些输入通道的内容以验证
    # print("Channel 0 (White pieces):\n", input_tensor_white[0,0,:,:])
    # print("Channel 1 (Black pieces):\n", input_tensor_white[0,1,:,:])
    # print("Channel 2 (White marked):\n", input_tensor_white[0,2,:,:]) # 白方视角，己方被标记
    # print("Channel 3 (Black marked):\n", input_tensor_white[0,3,:,:]) # 白方视角，对方被标记
    # print("Channel 6 (Movement Phase):\n", input_tensor_white[0,6,:,:])


    p1_w, p2_w, pmc_w, v_w = net(input_tensor_white)
    print("P1 (W):", p1_w.shape, "P2 (W):", p2_w.shape, "PMC (W):", pmc_w.shape, "V (W):", v_w.item()) 

    print("\n--- Testing get_move_probabilities ---")
    # Simulate network outputs (assuming batch size 1, so squeeze)
    # These are log_softmax outputs
    dummy_log_policy_pos1 = torch.randn(board_size * board_size)
    dummy_log_policy_pos2 = torch.randn(board_size * board_size)
    dummy_log_policy_mark_capture = torch.randn(board_size * board_size)

    # Example legal moves (manually crafted for testing)
    # Note: These moves might not be strictly "legal" in a real game state,
    # they are for testing the combination logic.
    example_legal_moves = [
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (0,0), 'mark_positions': None},
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (1,1), 'mark_positions': [(2,2)]},
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (0,1), 'mark_positions': [(3,3), (4,4)]},
        {'phase': Phase.MOVEMENT, 'action_type': 'move', 'from_position': (2,0), 'to_position': (2,1), 'capture_positions': None},
        {'phase': Phase.MOVEMENT, 'action_type': 'move', 'from_position': (3,0), 'to_position': (3,1), 'capture_positions': [(0,0)]},
        {'phase': Phase.FORCED_REMOVAL, 'action_type': 'remove', 'position': (5,5)},
        {'phase': Phase.REMOVAL, 'action_type': 'process_removal'}, # Only one of this type usually
        {'phase': Phase.MOVEMENT, 'action_type': 'no_moves_remove', 'position': (1,0)},
    ]

    if not example_legal_moves:
        print("No example legal moves to test.")
    else:
        probs, raw_log_probs = get_move_probabilities(
            dummy_log_policy_pos1,
            dummy_log_policy_pos2,
            dummy_log_policy_mark_capture,
            example_legal_moves,
            board_size
        )
        print(f"Number of legal moves: {len(example_legal_moves)}")
        print(f"Calculated probabilities ({len(probs)}): {probs}")
        print(f"Sum of probabilities: {sum(probs):.4f}") # Should be close to 1.0
        print(f"Raw combined log_probs ({raw_log_probs.shape[0]}): {raw_log_probs.tolist()}")

        # Test with a single legal move (e.g. process_removal)
        single_legal_move = [{'phase': Phase.REMOVAL, 'action_type': 'process_removal'}]
        probs_single, raw_single = get_move_probabilities(
            dummy_log_policy_pos1,
            dummy_log_policy_pos2,
            dummy_log_policy_mark_capture,
            single_legal_move,
            board_size
        )
        print(f"\nProbabilities for single move: {probs_single} (Sum: {sum(probs_single):.4f})")
        print(f"Raw log_prob for single move: {raw_single.tolist()}")

        # Test with no legal moves
        probs_none, raw_none = get_move_probabilities(
            dummy_log_policy_pos1,
            dummy_log_policy_pos2,
            dummy_log_policy_mark_capture,
            [],
            board_size
        )
        print(f"\nProbabilities for no legal moves: {probs_none}")
        print(f"Raw log_probs for no legal moves: {raw_none.tolist()}")

    print("\n--- Demo: Running a single step inference ---")
    # 1. Create an initial GameState
    current_state = GameState()
    player_to_act = current_state.current_player # Should be Player.BLACK
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device) # Ensure model is on the correct device

    print(f"Initial Game State (Player: {player_to_act}):")
    print(current_state)

    # 2. Convert GameState to tensor
    # Ensure state_to_tensor produces tensor on the same device as model if operations are done on it before model input
    # However, state_to_tensor currently returns a CPU tensor. Model input will move it.
    input_tensor = state_to_tensor(current_state, player_to_act).to(device)

    # 3. Get network outputs
    net.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations for inference
        log_p1, log_p2, log_pmc, value_estimate = net(input_tensor)
    
    # Squeeze batch dimension as we are processing a single state
    log_p1_squeezed = log_p1.squeeze(0)
    log_p2_squeezed = log_p2.squeeze(0)
    log_pmc_squeezed = log_pmc.squeeze(0)

    print(f"\nNetwork Value Estimate for {player_to_act}: {value_estimate.item():.4f}")

    # 4. Get legal moves
    legal_moves = generate_all_legal_moves(current_state)

    if not legal_moves:
        print("\nNo legal moves available for the current state.")
    else:
        print(f"\nNumber of legal moves available: {len(legal_moves)}")
        # 5. Get move probabilities
        move_probs, raw_log_probs_for_moves = get_move_probabilities(
            log_p1_squeezed,
            log_p2_squeezed,
            log_pmc_squeezed,
            legal_moves,
            board_size,
            device=device
        )

        # 6. Print top N moves with their probabilities
        print("\nLegal moves with their probabilities:")
        moves_with_probs = sorted(zip(legal_moves, move_probs), key=lambda x: x[1], reverse=True)
        
        for i, (move, prob) in enumerate(moves_with_probs):
            print(f"Move {i+1}: Prob={prob:.4f}, Details={move}")
            if i >= 4: # Print top 5 moves
                break

    # Optional: Show a slightly more complex state (e.g., after a few moves)
    print("\n--- Demo: Slightly more complex state (after a few hypothetical moves) ---")
    # Let's apply a few moves to change the state (these are not from the policy)
    # This requires `apply_move` which we should import
    from src.move_generator import apply_move # Import apply_move

    state_after_moves = GameState()
    # Example sequence of moves (manually chosen for diversity)
    # This is just to get a different board configuration quickly.
    # These moves are not necessarily good or from the policy.
    moves_to_apply = [
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (2,2), 'mark_positions': None}, # B
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (3,3), 'mark_positions': None}, # W
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (2,3), 'mark_positions': None}, # B
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (3,2), 'mark_positions': None}, # W
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (1,1), 'mark_positions': None}, # B
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (4,4), 'mark_positions': None}, # W
    ]
    
    temp_state_for_demo = GameState()
    print("Applying a sequence of moves for demo...")
    for move_dict in moves_to_apply:
        # Ensure the move is valid for the current player and phase of temp_state_for_demo
        # This basic loop doesn't check if the move is actually among `generate_all_legal_moves`
        # It assumes these are valid for demonstration.
        if temp_state_for_demo.phase == move_dict['phase']:
            try:
                temp_state_for_demo = apply_move(temp_state_for_demo, move_dict, quiet=True)
            except ValueError as e:
                print(f"Skipping invalid demo move {move_dict}: {e}")
                # If a move fails, the player/phase might not update correctly for subsequent moves
                # For a robust demo sequence, one might need to check generate_all_legal_moves first
                break # Stop if a move is bad
        else:
            # This can happen if a previous move changed phase unexpectedly or if sequence is ill-defined
            print(f"Phase mismatch for demo move. State: {temp_state_for_demo.phase}, Move: {move_dict['phase']}")
            break

    current_state_complex = temp_state_for_demo
    player_to_act_complex = current_state_complex.current_player
    print(f"\nComplex Game State (Player: {player_to_act_complex}):")
    print(current_state_complex)

    input_tensor_complex = state_to_tensor(current_state_complex, player_to_act_complex).to(device)
    with torch.no_grad():
        log_p1_c, log_p2_c, log_pmc_c, value_c = net(input_tensor_complex)
    
    log_p1_c_s = log_p1_c.squeeze(0)
    log_p2_c_s = log_p2_c.squeeze(0)
    log_pmc_c_s = log_pmc_c.squeeze(0)

    print(f"\nNetwork Value Estimate for {player_to_act_complex}: {value_c.item():.4f}")
    legal_moves_complex = generate_all_legal_moves(current_state_complex)

    if not legal_moves_complex:
        print("\nNo legal moves available for the complex state.")
    else:
        print(f"\nNumber of legal moves available: {len(legal_moves_complex)}")
        move_probs_c, _ = get_move_probabilities(
            log_p1_c_s, log_p2_c_s, log_pmc_c_s, 
            legal_moves_complex, board_size, device=device
        )
        moves_with_probs_c = sorted(zip(legal_moves_complex, move_probs_c), key=lambda x: x[1], reverse=True)
        print("\nTop 5 Legal moves for complex state:")
        for i, (move, prob) in enumerate(moves_with_probs_c):
            print(f"Move {i+1}: Prob={prob:.4f}, Details={move}")
            if i >= 4:
                break 
