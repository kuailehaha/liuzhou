import torch
import torch.nn as nn
import torch.nn.functional as F
from src.game_state import GameState, Player, Phase
from src.move_generator import generate_all_legal_moves, MoveType

# Number of input channels: player pieces, opponent pieces, mark indicators, and phase planes
NUM_INPUT_CHANNELS = 11

def state_to_tensor(state: GameState, player_to_act: Player) -> torch.Tensor:
    """Convert a GameState into the neural network input tensor."""
    board_size = state.BOARD_SIZE
    tensor = torch.zeros(NUM_INPUT_CHANNELS, board_size, board_size)

    player_val = player_to_act.value
    opponent_val = player_to_act.opponent().value

    # Channel 0: current player's pieces
    # Channel 1: opponent pieces
    for r in range(board_size):
        for c in range(board_size):
            if state.board[r][c] == player_val:
                tensor[0, r, c] = 1
            elif state.board[r][c] == opponent_val:
                tensor[1, r, c] = 1

    # Channel 2: current player's marked pieces
    # Channel 3: opponent's marked pieces
    marked_self = state.marked_black if player_to_act == Player.BLACK else state.marked_white
    marked_opponent = state.marked_white if player_to_act == Player.BLACK else state.marked_black

    for r, c in marked_self:
        if 0 <= r < board_size and 0 <= c < board_size:
            tensor[2, r, c] = 1

    for r, c in marked_opponent:
        if 0 <= r < board_size and 0 <= c < board_size:
            tensor[3, r, c] = 1

    # Phase one-hot channels
    if state.phase == Phase.PLACEMENT:
        tensor[4, :, :] = 1
    elif state.phase == Phase.MARK_SELECTION:
        tensor[5, :, :] = 1
    elif state.phase == Phase.REMOVAL:
        tensor[6, :, :] = 1
    elif state.phase == Phase.MOVEMENT:
        tensor[7, :, :] = 1
    elif state.phase == Phase.CAPTURE_SELECTION:
        tensor[8, :, :] = 1
    elif state.phase == Phase.FORCED_REMOVAL:
        tensor[9, :, :] = 1
    elif state.phase == Phase.COUNTER_REMOVAL:
        tensor[10, :, :] = 1
    return tensor.unsqueeze(0)  # add batch dimension

class ChessNet(nn.Module):
    def __init__(self, board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS, hidden_conv_channels=64):
        super(ChessNet, self).__init__()
        self.board_size = board_size
        self.num_input_channels = num_input_channels

        self.conv1 = nn.Conv2d(num_input_channels, hidden_conv_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_conv_channels)
        self.conv2 = nn.Conv2d(hidden_conv_channels, hidden_conv_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_conv_channels * 2)
        self.conv3 = nn.Conv2d(hidden_conv_channels * 2, hidden_conv_channels * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_conv_channels * 4)
        self.conv4 = nn.Conv2d(hidden_conv_channels * 4, hidden_conv_channels * 4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(hidden_conv_channels * 4)

        # Policy head shared trunk
        self.policy_common_conv = nn.Conv2d(hidden_conv_channels * 4, hidden_conv_channels, kernel_size=1)
        self.policy_common_bn = nn.BatchNorm2d(hidden_conv_channels)

        # Policy Head 1: primary position logits (placement, move source, removals)
        self.policy_pos1_conv = nn.Conv2d(hidden_conv_channels, 2, kernel_size=1)
        self.policy_pos1_bn = nn.BatchNorm2d(2)
        self.policy_pos1_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Policy Head 2: secondary position logits (move destination)
        self.policy_pos2_conv = nn.Conv2d(hidden_conv_channels, 2, kernel_size=1)
        self.policy_pos2_bn = nn.BatchNorm2d(2)
        self.policy_pos2_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Policy Head 3: mark/capture position logits
        self.policy_mark_capture_conv = nn.Conv2d(hidden_conv_channels, 2, kernel_size=1)
        self.policy_mark_capture_bn = nn.BatchNorm2d(2)
        self.policy_mark_capture_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head
        self.value_conv_common = nn.Conv2d(hidden_conv_channels * 4, hidden_conv_channels, kernel_size=1)
        self.value_bn_common = nn.BatchNorm2d(hidden_conv_channels)
        self.value_conv1 = nn.Conv2d(hidden_conv_channels, 1, kernel_size=1)
        self.value_bn1 = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch_size, num_input_channels, board_size, board_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        shared_features = F.relu(self.bn4(self.conv4(x)))

        # --- policy heads ---
        policy_latent = F.relu(self.policy_common_bn(self.policy_common_conv(shared_features)))

        p1 = F.relu(self.policy_pos1_bn(self.policy_pos1_conv(policy_latent)))
        p1 = p1.view(p1.size(0), -1)
        log_policy_pos1 = F.log_softmax(self.policy_pos1_fc(p1), dim=1)

        p2 = F.relu(self.policy_pos2_bn(self.policy_pos2_conv(policy_latent)))
        p2 = p2.view(p2.size(0), -1)
        log_policy_pos2 = F.log_softmax(self.policy_pos2_fc(p2), dim=1)

        pmc = F.relu(self.policy_mark_capture_bn(self.policy_mark_capture_conv(policy_latent)))
        pmc = pmc.view(pmc.size(0), -1)
        log_policy_mark_capture = F.log_softmax(self.policy_mark_capture_fc(pmc), dim=1)

        # --- value head ---
        value_latent = F.relu(self.value_bn_common(self.value_conv_common(shared_features)))
        v = F.relu(self.value_bn1(self.value_conv1(value_latent)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return log_policy_pos1, log_policy_pos2, log_policy_mark_capture, value

def get_move_probabilities(
    log_policy_pos1: torch.Tensor,
    log_policy_pos2: torch.Tensor,
    log_policy_mark_capture: torch.Tensor,
    legal_moves: list,
    board_size: int,
    device: str = 'cpu'
) -> tuple[list[float], torch.Tensor]:
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
    
    # ?    
    move_types = []  # /

    def flatten_index(r, c):
        return r * board_size + c

    for move in legal_moves:
        current_log_prob = 0.0
        has_mark_capture = False  # /

        phase = move['phase']
        action_type = move['action_type']

        if phase == Phase.PLACEMENT and action_type == 'place':
            r, c = move['position']
            current_log_prob += log_policy_pos1[flatten_index(r, c)].item()
        
        elif phase == Phase.MARK_SELECTION and action_type == 'mark':
            has_mark_capture = True
            r, c = move['position']
            current_log_prob += log_policy_mark_capture[flatten_index(r, c)].item()

        elif phase == Phase.MOVEMENT and action_type == 'move':
            r_from, c_from = move['from_position']
            r_to, c_to = move['to_position']
            # 
            current_log_prob += log_policy_pos1[flatten_index(r_from, c_from)].item()
            current_log_prob += log_policy_pos2[flatten_index(r_to, c_to)].item()

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
        
        elif phase == Phase.CAPTURE_SELECTION and action_type == 'capture':
            has_mark_capture = True
            r, c = move['position']
            current_log_prob += log_policy_mark_capture[flatten_index(r, c)].item()

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

    if not combined_log_probs:  # Should be redundant due to earlier check
        return [], torch.empty(0, device=device)
    
    # Convert to tensor for downstream processing
    combined_log_probs_tensor = torch.tensor(combined_log_probs, dtype=torch.float32, device=device)
    
    # If there is only one legal move, its probability is 1.0
    if len(legal_moves) == 1:
        return [1.0], combined_log_probs_tensor
    
    # Optional adjustment: prevent all mark/capture moves from collapsing to near-zero probability
    has_mark_capture_moves = any(move_types)
    if has_mark_capture_moves:
        orig_probs = torch.softmax(combined_log_probs_tensor, dim=0)
        mark_capture_probs = [prob.item() for prob, has_mc in zip(orig_probs, move_types) if has_mc]

        if mark_capture_probs and max(mark_capture_probs) < 0.001:
            max_log_prob_no_mc = max(
                [lp for lp, has_mc in zip(combined_log_probs, move_types) if not has_mc],
                default=-float("inf"),
            )
            max_log_prob_mc = max(
                [lp for lp, has_mc in zip(combined_log_probs, move_types) if has_mc],
                default=-float("inf"),
            )
            adjustment = max(0.0, max_log_prob_no_mc - max_log_prob_mc - 1.0)

            adjusted_log_probs = [
                lp + adjustment if has_mc else lp
                for lp, has_mc in zip(combined_log_probs, move_types)
            ]
            combined_log_probs_tensor = torch.tensor(
                adjusted_log_probs, dtype=torch.float32, device=device
            )

    # Final normalized probabilities
    probabilities = torch.softmax(combined_log_probs_tensor, dim=0).tolist()
    
    return probabilities, combined_log_probs_tensor

if __name__ == '__main__':
    board_size = GameState.BOARD_SIZE
    net = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
    
    # ?    
    # 
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

    # 
    initial_state.current_player = Player.WHITE # 
    #  (0,0) ?(1,1)
    initial_state.marked_white.add((0,0))
    initial_state.marked_white.add((1,1))
    #  (2,2)
    initial_state.marked_black.add((2,2))
    initial_state.phase = Phase.MOVEMENT # 

    input_tensor_white = state_to_tensor(initial_state, Player.WHITE)
    print("Input tensor shape (White's turn, with marks, movement phase):", input_tensor_white.shape)
    # 
    # print("Channel 0 (White pieces):\n", input_tensor_white[0,0,:,:])
    # print("Channel 1 (Black pieces):\n", input_tensor_white[0,1,:,:])
    # print("Channel 2 (White marked):\n", input_tensor_white[0,2,:,:]) # 
    # print("Channel 3 (Black marked):\n", input_tensor_white[0,3,:,:]) # 
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
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (0, 0)},
        {'phase': Phase.MARK_SELECTION, 'action_type': 'mark', 'position': (2, 2)},
        {'phase': Phase.MOVEMENT, 'action_type': 'move', 'from_position': (2, 0), 'to_position': (2, 1)},
        {'phase': Phase.CAPTURE_SELECTION, 'action_type': 'capture', 'position': (3, 3)},
        {'phase': Phase.FORCED_REMOVAL, 'action_type': 'remove', 'position': (5, 5)},
        {'phase': Phase.REMOVAL, 'action_type': 'process_removal'},
        {'phase': Phase.MOVEMENT, 'action_type': 'no_moves_remove', 'position': (1, 0)},
        {'phase': Phase.COUNTER_REMOVAL, 'action_type': 'counter_remove', 'position': (0, 1)},
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
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (2,2)}, # B
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (3,3)}, # W
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (2,3)}, # B
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (3,2)}, # W
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (1,1)}, # B
        {'phase': Phase.PLACEMENT, 'action_type': 'place', 'position': (4,4)}, # W
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
