import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.game_state import GameState, Player, Phase
from src.move_generator import generate_all_legal_moves, apply_move, MoveType
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS # For type hinting
from src.mcts import MCTS
from src.random_agent import RandomAgent


def _normalize_eval_games(num_games: int, log) -> int:
    if num_games % 2 != 0:
        adjusted = max(2, (num_games // 2) * 2)
        if adjusted == 0 and num_games > 0:
            adjusted = 2
        log("Warning: num_games for evaluation should be even for fair comparison. Adjusting...")
        num_games = adjusted
    return num_games


def _split_game_indices(num_games: int, num_workers: int) -> List[List[int]]:
    if num_workers <= 0:
        return []
    buckets: List[List[int]] = [[] for _ in range(num_workers)]
    for idx in range(num_games):
        buckets[idx % num_workers].append(idx)
    return [bucket for bucket in buckets if bucket]


def _seed_worker(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _load_v0_mcts():
    try:
        from v0.python.mcts import MCTS as V0MCTS
    except Exception as exc:
        raise ImportError(
            "v0 MCTS backend is unavailable; build v0_core or use legacy evaluation."
        ) from exc
    return V0MCTS


def _load_model_from_checkpoint(checkpoint_path: str, device: str) -> ChessNet:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    board_size = checkpoint.get("board_size", GameState.BOARD_SIZE)
    num_inputs = checkpoint.get("num_input_channels", NUM_INPUT_CHANNELS)
    model = ChessNet(board_size=board_size, num_input_channels=num_inputs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _play_game_indices(
    game_indices: List[int],
    num_games: int,
    challenger_agent,
    opponent_agent,
    game_verbose: bool,
) -> Tuple[int, int, int]:
    wins = losses = draws = 0
    half = num_games / 2
    for idx in game_indices:
        if idx < half:
            result = play_single_game(challenger_agent, opponent_agent, verbose=game_verbose)
            if result == 1.0:
                wins += 1
            elif result == -1.0:
                losses += 1
            else:
                draws += 1
        else:
            result = play_single_game(opponent_agent, challenger_agent, verbose=game_verbose)
            if result == -1.0:
                wins += 1
            elif result == 1.0:
                losses += 1
            else:
                draws += 1
    return wins, losses, draws


def _eval_worker(
    worker_id: int,
    game_indices: List[int],
    num_games: int,
    device: str,
    mcts_simulations: int,
    temperature: float,
    add_dirichlet_noise: bool,
    challenger_checkpoint: str,
    opponent_checkpoint: Optional[str],
    game_verbose: bool,
    mcts_verbose: Optional[bool],
    seed: int,
    threads_per_worker: int,
    sample_moves: bool,
) -> Tuple[int, int, int]:
    torch.set_num_threads(max(1, int(threads_per_worker)))
    _seed_worker(seed + worker_id)

    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        torch.cuda.set_device(device_obj.index or 0)

    challenger_model = _load_model_from_checkpoint(challenger_checkpoint, device)
    challenger_agent = MCTSAgent(
        challenger_model,
        mcts_simulations=mcts_simulations,
        temperature=temperature,
        device=device,
        add_dirichlet_noise=add_dirichlet_noise,
        verbose=False,
        mcts_verbose=mcts_verbose,
        sample_moves=sample_moves,
    )

    if opponent_checkpoint:
        opponent_model = _load_model_from_checkpoint(opponent_checkpoint, device)
        opponent_agent = MCTSAgent(
            opponent_model,
            mcts_simulations=mcts_simulations,
            temperature=temperature,
            device=device,
            add_dirichlet_noise=add_dirichlet_noise,
            verbose=False,
            mcts_verbose=mcts_verbose,
            sample_moves=sample_moves,
        )
    else:
        opponent_agent = RandomAgent()

    return _play_game_indices(game_indices, num_games, challenger_agent, opponent_agent, game_verbose)


def _eval_worker_v0(
    worker_id: int,
    game_indices: List[int],
    num_games: int,
    device: str,
    mcts_simulations: int,
    temperature: float,
    add_dirichlet_noise: bool,
    challenger_checkpoint: str,
    opponent_checkpoint: Optional[str],
    game_verbose: bool,
    mcts_verbose: Optional[bool],
    seed: int,
    threads_per_worker: int,
    batch_leaves: int,
    inference_backend: str,
    torchscript_path: Optional[str],
    torchscript_dtype: Optional[str],
    inference_batch_size: int,
    inference_warmup_iters: int,
    sample_moves: bool,
) -> Tuple[int, int, int]:
    torch.set_num_threads(max(1, int(threads_per_worker)))
    _seed_worker(seed + worker_id)

    device_obj = torch.device(device)
    if device_obj.type == "cuda":
        torch.cuda.set_device(device_obj.index or 0)

    challenger_model = _load_model_from_checkpoint(challenger_checkpoint, device)
    challenger_agent = V0MCTSAgent(
        challenger_model,
        mcts_simulations=mcts_simulations,
        temperature=temperature,
        device=device,
        add_dirichlet_noise=add_dirichlet_noise,
        verbose=False,
        mcts_verbose=mcts_verbose,
        batch_leaves=batch_leaves,
        inference_backend=inference_backend,
        torchscript_path=torchscript_path,
        torchscript_dtype=torchscript_dtype,
        inference_batch_size=inference_batch_size,
        inference_warmup_iters=inference_warmup_iters,
        sample_moves=sample_moves,
    )

    if opponent_checkpoint:
        opponent_model = _load_model_from_checkpoint(opponent_checkpoint, device)
        opponent_agent = V0MCTSAgent(
            opponent_model,
            mcts_simulations=mcts_simulations,
            temperature=temperature,
            device=device,
            add_dirichlet_noise=add_dirichlet_noise,
            verbose=False,
            mcts_verbose=mcts_verbose,
            batch_leaves=batch_leaves,
            inference_backend=inference_backend,
            torchscript_path=torchscript_path,
            torchscript_dtype=torchscript_dtype,
            inference_batch_size=inference_batch_size,
            inference_warmup_iters=inference_warmup_iters,
            sample_moves=sample_moves,
        )
    else:
        opponent_agent = RandomAgent()

    return _play_game_indices(game_indices, num_games, challenger_agent, opponent_agent, game_verbose)

@dataclass
class EvaluationStats:
    """Aggregate win/draw/loss information for a head-to-head evaluation."""
    wins: int
    losses: int
    draws: int
    total_games: int

    @property
    def win_rate(self) -> float:
        return self._safe_rate(self.wins)

    @property
    def loss_rate(self) -> float:
        return self._safe_rate(self.losses)

    @property
    def draw_rate(self) -> float:
        return self._safe_rate(self.draws)

    def _safe_rate(self, value: int) -> float:
        return 0.0 if self.total_games == 0 else value / self.total_games

class MCTSAgent:
    """Agent that selects moves via Monte Carlo Tree Search guided by a neural network."""

    def __init__(
        self,
        model: ChessNet,
        mcts_simulations: int,
        temperature: float = 0.05,
        device: str = "cpu",
        add_dirichlet_noise: bool = False,
        verbose: bool = False,
        mcts_verbose: Optional[bool] = None,
        sample_moves: bool = False,
    ):
        self.model = model
        self.model.eval()  # Keep the model in inference mode during evaluation
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature  # Controls how deterministic the visit distribution is
        self.device = device
        self.verbose = verbose
        if mcts_verbose is None:
            mcts_verbose = verbose
        self.sample_moves = bool(sample_moves)
        self.mcts = MCTS(
            model=self.model,
            num_simulations=self.mcts_simulations,
            exploration_weight=1.0,  # Exploration constant for UCT scoring
            temperature=self.temperature,
            device=self.device,
            add_dirichlet_noise=add_dirichlet_noise,
            verbose=mcts_verbose,
        )

    def select_move(self, state: GameState) -> MoveType:
        """Use MCTS to choose a move for the given state."""
        if not generate_all_legal_moves(state):  # Fail fast if the state unexpectedly has no legal moves
            # print(f"MCTSAgent: No legal moves for state:\n{state}")
            raise ValueError("MCTSAgent: No legal moves available for the current state to select from.")

        moves, policy = self.mcts.search(state)
        if not moves:
            # print(f"MCTSAgent: MCTS search returned no moves for state:\n{state}")
            raise ValueError("MCTSAgent: MCTS search returned no moves.")
        
        policy = np.asarray(policy, dtype=float)
        if self.sample_moves:
            if not np.all(np.isfinite(policy)) or policy.sum() <= 0:
                policy = np.ones_like(policy, dtype=float) / len(policy)
            else:
                policy = policy / policy.sum()
            move_idx = int(np.random.choice(len(moves), p=policy))
        else:
            # In evaluation we simply pick the move with the highest visit probability
            move_idx = int(np.argmax(policy))
        chosen_move = moves[move_idx]
        self.mcts.advance_root(chosen_move)
        return chosen_move


class V0MCTSAgent:
    """Agent that selects moves via the v0 C++-backed MCTS."""

    def __init__(
        self,
        model: ChessNet,
        mcts_simulations: int,
        temperature: float = 0.05,
        device: str = "cpu",
        add_dirichlet_noise: bool = False,
        verbose: bool = False,
        mcts_verbose: Optional[bool] = None,
        batch_leaves: int = 256,
        inference_backend: str = "graph",
        torchscript_path: Optional[str] = None,
        torchscript_dtype: Optional[str] = None,
        inference_batch_size: int = 512,
        inference_warmup_iters: int = 5,
        sample_moves: bool = False,
    ):
        self.model = model
        self.model.eval()
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.device = device
        self.verbose = verbose
        if mcts_verbose is None:
            mcts_verbose = verbose
        self.sample_moves = bool(sample_moves)
        V0MCTS = _load_v0_mcts()
        self.mcts = V0MCTS(
            model=self.model,
            num_simulations=self.mcts_simulations,
            exploration_weight=1.0,
            temperature=self.temperature,
            device=self.device,
            add_dirichlet_noise=add_dirichlet_noise,
            verbose=mcts_verbose,
            batch_K=max(1, int(batch_leaves)),
            virtual_loss=1.0,
            inference_backend=inference_backend,
            torchscript_path=torchscript_path,
            torchscript_dtype=torchscript_dtype,
            inference_batch_size=max(1, int(inference_batch_size)),
            inference_warmup_iters=max(0, int(inference_warmup_iters)),
        )

    def select_move(self, state: GameState) -> MoveType:
        if not generate_all_legal_moves(state):
            raise ValueError("V0MCTSAgent: No legal moves available for the current state to select from.")

        moves, policy = self.mcts.search(state)
        if not moves:
            raise ValueError("V0MCTSAgent: MCTS search returned no moves.")

        policy = np.asarray(policy, dtype=float)
        if self.sample_moves:
            if not np.all(np.isfinite(policy)) or policy.sum() <= 0:
                policy = np.ones_like(policy, dtype=float) / len(policy)
            else:
                policy = policy / policy.sum()
            move_idx = int(np.random.choice(len(moves), p=policy))
        else:
            move_idx = int(np.argmax(policy))
        chosen_move = moves[move_idx]
        self.mcts.advance_root(chosen_move)
        return chosen_move

def play_single_game(
    agent_black,
    agent_white,
    initial_state: GameState = None,
    max_moves: int = 200,
    verbose: bool = False,
) -> float:
    """Play a single game and return the outcome from Black's perspective."""
    state = initial_state if initial_state else GameState()
    move_count = 0
    log = print if verbose else (lambda *args, **kwargs: None)

    while move_count < max_moves:
        current_player = state.current_player
        active_agent = agent_black if current_player == Player.BLACK else agent_white

        legal_moves = generate_all_legal_moves(state)
        if not legal_moves:
            log(f"Game ended: Player {current_player} has no legal moves.")
            return -1.0 if current_player == Player.BLACK else 1.0

        try:
            selected_move = active_agent.select_move(state)
        except ValueError:
            log(f"Game ended: Agent for {current_player} could not select a move.")
            return -1.0 if current_player == Player.BLACK else 1.0

        state = apply_move(state, selected_move, quiet=True)
        move_count += 1

        winner = state.get_winner()
        if winner is not None:
            return 1.0 if winner == Player.BLACK else -1.0 if winner == Player.WHITE else 0.0

    return 0.0

def evaluate_against_agent(
    challenger_agent,
    opponent_agent,
    num_games: int,
    device: str,
    verbose: bool = False,
    game_verbose: Optional[bool] = None,
) -> EvaluationStats:
    """Pit the challenger against the opponent and return aggregated results."""
    log = print if verbose else (lambda *args, **kwargs: None)
    if num_games % 2 != 0:
        adjusted = max(2, (num_games // 2) * 2)
        if adjusted == 0 and num_games > 0:
            adjusted = 2
        log("Warning: num_games for evaluation should be even for fair comparison. Adjusting...")
        num_games = adjusted

    if game_verbose is None:
        game_verbose = verbose

    challenger_wins = 0
    challenger_losses = 0
    challenger_draws = 0

    for i in range(num_games):
        log(f"  Playing evaluation game {i + 1}/{num_games}...")
        if i < num_games / 2:
            result = play_single_game(challenger_agent, opponent_agent, verbose=game_verbose)
            if result == 1.0:
                challenger_wins += 1
            elif result == -1.0:
                challenger_losses += 1
            else:
                challenger_draws += 1
        else:
            result = play_single_game(opponent_agent, challenger_agent, verbose=game_verbose)
            if result == -1.0:
                challenger_wins += 1
            elif result == 1.0:
                challenger_losses += 1
            else:
                challenger_draws += 1

    return EvaluationStats(
        wins=challenger_wins,
        losses=challenger_losses,
        draws=challenger_draws,
        total_games=num_games
    )


def evaluate_against_agent_parallel(
    challenger_checkpoint: str,
    opponent_checkpoint: Optional[str],
    num_games: int,
    device: str,
    mcts_simulations: int,
    temperature: float = 0.05,
    add_dirichlet_noise: bool = False,
    num_workers: int = 1,
    verbose: bool = False,
    game_verbose: Optional[bool] = None,
    mcts_verbose: Optional[bool] = None,
    sample_moves: bool = False,
) -> EvaluationStats:
    """Parallel evaluation using multiple worker processes."""
    log = print if verbose else (lambda *args, **kwargs: None)
    num_games = _normalize_eval_games(num_games, log)
    if game_verbose is None:
        game_verbose = verbose

    num_workers = max(1, int(num_workers))
    if num_games == 0:
        return EvaluationStats(wins=0, losses=0, draws=0, total_games=0)

    if num_workers == 1:
        challenger_model = _load_model_from_checkpoint(challenger_checkpoint, device)
        challenger_agent = MCTSAgent(
            challenger_model,
            mcts_simulations=mcts_simulations,
            temperature=temperature,
            device=device,
            add_dirichlet_noise=add_dirichlet_noise,
            verbose=verbose,
            mcts_verbose=mcts_verbose,
            sample_moves=sample_moves,
        )
        if opponent_checkpoint:
            opponent_model = _load_model_from_checkpoint(opponent_checkpoint, device)
            opponent_agent = MCTSAgent(
                opponent_model,
                mcts_simulations=mcts_simulations,
                temperature=temperature,
                device=device,
                add_dirichlet_noise=add_dirichlet_noise,
                verbose=verbose,
                mcts_verbose=mcts_verbose,
                sample_moves=sample_moves,
            )
        else:
            opponent_agent = RandomAgent()

        wins, losses, draws = _play_game_indices(
            list(range(num_games)),
            num_games,
            challenger_agent,
            opponent_agent,
            game_verbose,
        )
        return EvaluationStats(wins=wins, losses=losses, draws=draws, total_games=num_games)

    chunks = _split_game_indices(num_games, num_workers)
    if not chunks:
        return EvaluationStats(wins=0, losses=0, draws=0, total_games=num_games)

    base_seed = int(time.time() * 1e6) & 0x7FFFFFFF
    threads_per_worker = max(1, (os.cpu_count() or 1) // len(chunks))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(chunks)) as pool:
        results = pool.starmap(
            _eval_worker,
            [
                (
                    worker_id,
                    chunks[worker_id],
                    num_games,
                    device,
                    mcts_simulations,
                    temperature,
                    add_dirichlet_noise,
                    challenger_checkpoint,
                    opponent_checkpoint,
                    game_verbose,
                    mcts_verbose,
                    base_seed,
                    threads_per_worker,
                    sample_moves,
                )
                for worker_id in range(len(chunks))
            ],
        )

    wins = sum(result[0] for result in results)
    losses = sum(result[1] for result in results)
    draws = sum(result[2] for result in results)
    return EvaluationStats(wins=wins, losses=losses, draws=draws, total_games=num_games)


def evaluate_against_agent_parallel_v0(
    challenger_checkpoint: str,
    opponent_checkpoint: Optional[str],
    num_games: int,
    device: str,
    mcts_simulations: int,
    temperature: float = 0.05,
    add_dirichlet_noise: bool = False,
    num_workers: int = 1,
    verbose: bool = False,
    game_verbose: Optional[bool] = None,
    mcts_verbose: Optional[bool] = None,
    batch_leaves: int = 256,
    inference_backend: str = "graph",
    torchscript_path: Optional[str] = None,
    torchscript_dtype: Optional[str] = None,
    inference_batch_size: int = 512,
    inference_warmup_iters: int = 5,
    sample_moves: bool = False,
) -> EvaluationStats:
    """Parallel evaluation using v0 C++ MCTS workers."""
    log = print if verbose else (lambda *args, **kwargs: None)
    num_games = _normalize_eval_games(num_games, log)
    if game_verbose is None:
        game_verbose = verbose

    num_workers = max(1, int(num_workers))
    if num_games == 0:
        return EvaluationStats(wins=0, losses=0, draws=0, total_games=0)

    batch_leaves = max(1, int(batch_leaves))
    inference_batch_size = max(1, int(inference_batch_size))
    inference_warmup_iters = max(0, int(inference_warmup_iters))

    if num_workers == 1:
        challenger_model = _load_model_from_checkpoint(challenger_checkpoint, device)
        challenger_agent = V0MCTSAgent(
            challenger_model,
            mcts_simulations=mcts_simulations,
            temperature=temperature,
            device=device,
            add_dirichlet_noise=add_dirichlet_noise,
            verbose=verbose,
            mcts_verbose=mcts_verbose,
            batch_leaves=batch_leaves,
            inference_backend=inference_backend,
            torchscript_path=torchscript_path,
            torchscript_dtype=torchscript_dtype,
            inference_batch_size=inference_batch_size,
            inference_warmup_iters=inference_warmup_iters,
            sample_moves=sample_moves,
        )
        if opponent_checkpoint:
            opponent_model = _load_model_from_checkpoint(opponent_checkpoint, device)
            opponent_agent = V0MCTSAgent(
                opponent_model,
                mcts_simulations=mcts_simulations,
                temperature=temperature,
                device=device,
                add_dirichlet_noise=add_dirichlet_noise,
                verbose=verbose,
                mcts_verbose=mcts_verbose,
                batch_leaves=batch_leaves,
                inference_backend=inference_backend,
                torchscript_path=torchscript_path,
                torchscript_dtype=torchscript_dtype,
                inference_batch_size=inference_batch_size,
                inference_warmup_iters=inference_warmup_iters,
                sample_moves=sample_moves,
            )
        else:
            opponent_agent = RandomAgent()

        wins, losses, draws = _play_game_indices(
            list(range(num_games)),
            num_games,
            challenger_agent,
            opponent_agent,
            game_verbose,
        )
        return EvaluationStats(wins=wins, losses=losses, draws=draws, total_games=num_games)

    chunks = _split_game_indices(num_games, num_workers)
    if not chunks:
        return EvaluationStats(wins=0, losses=0, draws=0, total_games=num_games)

    base_seed = int(time.time() * 1e6) & 0x7FFFFFFF
    threads_per_worker = max(1, (os.cpu_count() or 1) // len(chunks))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(chunks)) as pool:
        results = pool.starmap(
            _eval_worker_v0,
            [
                (
                    worker_id,
                    chunks[worker_id],
                    num_games,
                    device,
                    mcts_simulations,
                    temperature,
                    add_dirichlet_noise,
                    challenger_checkpoint,
                    opponent_checkpoint,
                    game_verbose,
                    mcts_verbose,
                    base_seed,
                    threads_per_worker,
                    batch_leaves,
                    inference_backend,
                    torchscript_path,
                    torchscript_dtype,
                    inference_batch_size,
                    inference_warmup_iters,
                    sample_moves,
                )
                for worker_id in range(len(chunks))
            ],
        )

    wins = sum(result[0] for result in results)
    losses = sum(result[1] for result in results)
    draws = sum(result[2] for result in results)
    return EvaluationStats(wins=wins, losses=losses, draws=draws, total_games=num_games)

if __name__ == '__main__':
    # Basic smoke test for the evaluation utilities
    print("Testing evaluation module...")
    current_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create an untrained ChessNet model as a lightweight placeholder
    from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
    from src.game_state import GameState
    board_size = GameState.BOARD_SIZE
    dummy_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS).to(current_device)

    # Build the MCTS-based agent used for evaluation
    mcts_agent = MCTSAgent(dummy_model, mcts_simulations=20, device=current_device)  # Low simulation count keeps this quick
    
    # Random baseline agent for comparison
    random_agent = RandomAgent()

    num_eval_games = 4  # Even number so each agent gets both colors
    print(f"\nEvaluating MCTSAgent vs RandomAgent over {num_eval_games} games...")
    stats_vs_random = evaluate_against_agent(mcts_agent, random_agent, num_eval_games, current_device)
    print(f"MCTSAgent record against RandomAgent: {stats_vs_random.wins}-{stats_vs_random.losses}-{stats_vs_random.draws} "
          f"(win {stats_vs_random.win_rate:.2%} / loss {stats_vs_random.loss_rate:.2%} / draw {stats_vs_random.draw_rate:.2%})")

    # Example: evaluate two MCTS agents (e.g., new model vs. previous best)
    # Create another placeholder model to stand in for the previous best
    # old_best_model = ChessNet(board_size=board_size, num_input_channels=NUM_INPUT_CHANNELS).to(current_device)
    # best_model_agent = MCTSAgent(old_best_model, mcts_simulations=20, device=current_device)
    
    # print(f"\nEvaluating MCTSAgent vs Another MCTSAgent (dummy old_best_model) over {num_eval_games} games...")
    # win_rate_vs_best = evaluate_against_agent(mcts_agent, best_model_agent, num_eval_games, current_device)
    # print(f"MCTSAgent win rate against Another MCTSAgent: {win_rate_vs_best:.2%}")

    print("\nTesting play_single_game directly (Random vs Random for quick check)")
    res = play_single_game(RandomAgent(), RandomAgent())
    print(f"Random vs Random game result (1.0 Black win, -1.0 White win, 0.0 Draw): {res}") 
