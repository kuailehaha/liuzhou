"""Cross-implementation contracts for the portable C++ MCTS backend."""

from __future__ import annotations

import math

import pytest
import torch

from src.game_state import GameState, Phase, Player
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS, state_to_tensor
from src.policy_batch import TOTAL_DIM, action_to_index


def _small_model() -> ChessNet:
    torch.manual_seed(7)
    model = ChessNet(
        board_size=GameState.BOARD_SIZE,
        num_input_channels=NUM_INPUT_CHANNELS,
        trunk_channels=8,
        num_blocks=1,
        policy_channels=4,
        value_channels=4,
        value_mlp_channels=8,
    )
    model.eval()
    return model


def _cpp_module():
    from v1.python.portable_cpp_loader import load_portable_cpp

    module = load_portable_cpp(required=False)
    if module is None:
        pytest.skip("portable C++ extension is not built")
    return module


def _assert_state_payload_matches(payload: dict, state: GameState) -> None:
    assert payload["board"] == state.board
    assert payload["phase"] == state.phase.value
    assert payload["current_player"] == state.current_player.value
    assert set(map(tuple, payload["marked_black"])) == state.marked_black
    assert set(map(tuple, payload["marked_white"])) == state.marked_white
    for field in (
        "forced_removals_done",
        "move_count",
        "pending_marks_required",
        "pending_marks_remaining",
        "pending_captures_required",
        "pending_captures_remaining",
        "moves_since_capture",
    ):
        assert payload[field] == getattr(state, field)


def _representative_states() -> list[GameState]:
    states = [GameState()]

    line_board = [[0] * GameState.BOARD_SIZE for _ in range(GameState.BOARD_SIZE)]
    for col in range(5):
        line_board[0][col] = Player.BLACK.value
    for position in ((2, 0), (2, 2), (3, 4), (4, 1), (5, 3), (5, 5)):
        line_board[position[0]][position[1]] = Player.WHITE.value
    state = GameState(
        board=line_board,
        phase=Phase.PLACEMENT,
        current_player=Player.BLACK,
    )
    state = apply_move(
        state,
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (0, 5)},
        quiet=True,
    )
    states.append(state)
    state = apply_move(state, generate_all_legal_moves(state)[0], quiet=True)
    states.append(state)

    removal_board = [[0] * GameState.BOARD_SIZE for _ in range(GameState.BOARD_SIZE)]
    for position in ((0, 0), (0, 2), (1, 4), (3, 1), (5, 5)):
        removal_board[position[0]][position[1]] = Player.BLACK.value
    for position in ((0, 5), (2, 1), (3, 4), (4, 0), (5, 2)):
        removal_board[position[0]][position[1]] = Player.WHITE.value
    states.extend(
        [
            GameState(
                board=[row[:] for row in removal_board],
                phase=Phase.REMOVAL,
                current_player=Player.WHITE,
                marked_black={(0, 0)},
            ),
            GameState(
                board=[row[:] for row in removal_board],
                phase=Phase.FORCED_REMOVAL,
                current_player=Player.WHITE,
                forced_removals_done=0,
            ),
            GameState(
                board=[row[:] for row in removal_board],
                phase=Phase.COUNTER_REMOVAL,
                current_player=Player.WHITE,
            ),
        ]
    )

    movement_board = [[0] * GameState.BOARD_SIZE for _ in range(GameState.BOARD_SIZE)]
    for position in ((0, 0), (0, 1), (1, 0), (2, 1)):
        movement_board[position[0]][position[1]] = Player.BLACK.value
    for position in ((2, 3), (2, 5), (3, 4), (4, 0), (5, 2)):
        movement_board[position[0]][position[1]] = Player.WHITE.value
    movement = GameState(
        board=movement_board,
        phase=Phase.MOVEMENT,
        current_player=Player.BLACK,
        moves_since_capture=7,
    )
    states.append(movement)
    states.append(
        apply_move(
            movement,
            {
                "phase": Phase.MOVEMENT,
                "action_type": "move",
                "from_position": (2, 1),
                "to_position": (1, 1),
            },
            quiet=True,
        )
    )
    terminal = GameState(
        board=[row[:] for row in removal_board],
        phase=Phase.MOVEMENT,
        current_player=Player.BLACK,
    )
    for row, col in ((2, 1), (3, 4)):
        terminal.board[row][col] = 0
    assert terminal.get_winner() == Player.BLACK
    states.append(terminal)
    states.append(
        GameState(
            board=[row[:] for row in removal_board],
            phase=Phase.MOVEMENT,
            current_player=Player.WHITE,
            move_count=GameState.MAX_MOVE_COUNT,
        )
    )
    return states


def test_cpp_rules_actions_encoding_and_transitions_match_python() -> None:
    cpp = _cpp_module()
    states = _representative_states()
    assert {state.phase for state in states} == {
        Phase.PLACEMENT,
        Phase.MARK_SELECTION,
        Phase.REMOVAL,
        Phase.MOVEMENT,
        Phase.CAPTURE_SELECTION,
        Phase.FORCED_REMOVAL,
        Phase.COUNTER_REMOVAL,
    }

    for state in states:
        inspected = cpp.inspect_state(state)
        python_moves = generate_all_legal_moves(state)
        python_indices = sorted(
            int(action_to_index(move, GameState.BOARD_SIZE)) for move in python_moves
        )

        assert inspected["legal_action_indices"] == python_indices
        assert bool(inspected["game_over"]) == state.is_game_over()
        winner = state.get_winner()
        assert int(inspected["winner"]) == (
            int(winner.value) if winner is not None else 0
        )
        assert torch.equal(
            torch.from_numpy(inspected["model_input"]),
            state_to_tensor(state, state.current_player)[0],
        )
        _assert_state_payload_matches(inspected["state"], state)

        move_by_index = {
            int(action_to_index(move, GameState.BOARD_SIZE)): move
            for move in python_moves
        }
        for index in python_indices:
            cpp_next = cpp.apply_action(state, index)
            python_next = apply_move(state, move_by_index[index], quiet=True)
            _assert_state_payload_matches(cpp_next, python_next)


def test_cpp_backup_flips_only_on_actual_player_change() -> None:
    cpp = _cpp_module()

    same_player = cpp.debug_backup_values([1, 1, 1], 0.75)
    switched_player = cpp.debug_backup_values([1, 1, -1], 0.75)

    assert same_player == pytest.approx([0.75, 0.75, 0.75])
    assert switched_player == pytest.approx([-0.75, -0.75, 0.75])


def test_cpp_and_python_deterministic_search_match() -> None:
    _cpp_module()
    from v1.python.portable_cpp_mcts import PortableCppMCTS
    from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig, PortableTree

    python_model = _small_model()
    cpp_model = _small_model()
    cpp_model.load_state_dict(python_model.state_dict())
    config = PortableMCTSConfig(
        num_simulations=8,
        exploration_weight=1.0,
        temperature=1.0,
        add_dirichlet_noise=False,
        sample_moves=False,
    )
    states = [GameState(), GameState(current_player=Player.WHITE)]
    python_search = PortableMCTS(python_model, config, "cpu")
    python_outputs = python_search.search_batch(
        [PortableTree(state) for state in states],
        add_dirichlet_noise=False,
    )
    cpp_search = PortableCppMCTS(
        cpp_model,
        config,
        device="cpu",
        num_threads=2,
        initial_states=states,
    )
    cpp_outputs = cpp_search.search_batch(
        temperatures=[1.0, 1.0],
        add_dirichlet_noise=False,
    )

    for python_output, cpp_output in zip(python_outputs, cpp_outputs):
        assert torch.equal(cpp_output.legal_mask, python_output.legal_mask)
        assert cpp_output.visit_counts == python_output.visit_counts
        assert torch.equal(cpp_output.policy_dense, python_output.policy_dense)
        assert cpp_output.root_value == pytest.approx(
            python_output.root_value, abs=1e-6
        )
        assert cpp_output.chosen_action_index == python_output.chosen_action_index


def test_cpp_self_play_preserves_tensor_contract_and_audit_fields() -> None:
    _cpp_module()
    from v1.python.portable_cpp_self_play import self_play_v1_portable_cpp

    samples, stats = self_play_v1_portable_cpp(
        model=_small_model(),
        num_games=2,
        mcts_simulations=2,
        temperature_init=1.0,
        temperature_final=1.0,
        temperature_threshold=4,
        exploration_weight=1.0,
        device="cpu",
        add_dirichlet_noise=False,
        max_game_plies=8,
        sample_moves=False,
        concurrent_games=2,
        cpu_threads=2,
    )

    assert samples.state_tensors.shape == (samples.num_samples, 11, 6, 6)
    assert samples.legal_masks.shape == (samples.num_samples, TOTAL_DIM)
    assert samples.policy_targets.shape == (samples.num_samples, TOTAL_DIM)
    assert samples.value_targets.shape == (samples.num_samples,)
    assert samples.soft_value_targets.shape == (samples.num_samples,)
    assert stats.num_games == 2
    assert stats.num_positions == samples.num_samples
    assert stats.device == "cpu"
    assert stats.fallback_count == 0
    assert stats.mcts_counters["portable_cpp_threads"] == 2
    assert stats.mcts_counters["portable_cpp_illegal_actions"] == 0
    assert stats.mcts_counters["portable_cpp_non_finite"] == 0
    assert all(
        math.isfinite(float(value))
        for tensor in (
            samples.state_tensors,
            samples.policy_targets,
            samples.value_targets,
            samples.soft_value_targets,
        )
        for value in tensor.flatten()
    )


def test_cpp_self_play_is_reproducible_across_thread_counts() -> None:
    _cpp_module()
    from v1.python.portable_cpp_self_play import self_play_v1_portable_cpp

    model_state = _small_model().state_dict()
    reference = None
    for threads in (1, 2, 4, 8):
        model = _small_model()
        model.load_state_dict(model_state)
        torch.manual_seed(20260723)
        samples, stats = self_play_v1_portable_cpp(
            model=model,
            num_games=4,
            mcts_simulations=2,
            temperature_init=1.0,
            temperature_final=1.0,
            temperature_threshold=8,
            exploration_weight=1.0,
            device="cpu",
            add_dirichlet_noise=False,
            max_game_plies=6,
            sample_moves=False,
            concurrent_games=4,
            cpu_threads=threads,
        )
        snapshot = (
            samples.state_tensors,
            samples.legal_masks,
            samples.policy_targets,
            samples.value_targets,
            samples.soft_value_targets,
        )
        if reference is None:
            reference = tuple(tensor.clone() for tensor in snapshot)
        else:
            assert all(
                torch.equal(actual, expected)
                for actual, expected in zip(snapshot, reference)
            )
        assert stats.mcts_counters["portable_cpp_threads"] == threads
        assert stats.fallback_count == 0


def test_cpp_process_worker_emits_existing_training_manifest(tmp_path) -> None:
    _cpp_module()
    from v1.python.self_play_worker import run_self_play_worker

    model = ChessNet(
        board_size=GameState.BOARD_SIZE,
        num_input_channels=NUM_INPUT_CHANNELS,
    )
    model_state_path = tmp_path / "model_state.pt"
    manifest_path = tmp_path / "worker_manifest.pt"
    torch.save(model.state_dict(), model_state_path)

    row = run_self_play_worker(
        worker_idx=0,
        shard_device="cpu",
        shard_games=1,
        seed=20260723,
        model_state_path=str(model_state_path),
        output_path=str(manifest_path),
        mcts_simulations=1,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_threshold=4,
        exploration_weight=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        soft_value_k=2.0,
        opening_random_moves=0,
        max_game_plies=4,
        concurrent_games_per_device=1,
        chunk_output_dir=str(tmp_path),
        chunk_file_prefix="portable_cpp_worker",
        search_backend="portable",
        portable_mcts_backend="cpp",
        portable_cpp_threads=2,
    )

    payload = torch.load(manifest_path, map_location="cpu")
    assert row["output_path"] == str(manifest_path)
    assert payload["payload_format"] == "v1_worker_chunk_manifest"
    assert payload["metadata"]["search_backend"] == "portable"
    assert payload["metadata"]["portable_mcts_backend"] == "cpp"
    assert payload["metadata"]["portable_cpp_threads"] == 2
    assert payload["stats"]["fallback_count"] == 0
    assert payload["stats"]["mcts_counters"]["portable_cpp_illegal_actions"] == 0
    assert payload["stats"]["mcts_counters"]["portable_cpp_non_finite"] == 0
    assert payload["num_samples"] > 0
