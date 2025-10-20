import torch
from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from src.mcts import self_play

def test_self_play_generation():
    """
    Tests the self-play data generation process.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize a model
    board_size = GameState.BOARD_SIZE
    model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(device)
    model.eval()

    # Run a small number of self-play games
    num_games = 5
    mcts_simulations = 100
    print(f"Running {num_games} self-play games with {mcts_simulations} MCTS simulations...")

    training_data = self_play(
        model=model,
        num_games=num_games,
        mcts_simulations=mcts_simulations,
        device=device
    )

    # Print the generated training data
    if training_data:
        print(f"Successfully generated training data for {len(training_data)} games.")
        for i, (game_states, game_policies, result) in enumerate(training_data):
            print(f"  Game {i+1}: {len(game_states)} states, result: {result}")
    else:
        print("Failed to generate any training data.")

if __name__ == "__main__":
    test_self_play_generation()
