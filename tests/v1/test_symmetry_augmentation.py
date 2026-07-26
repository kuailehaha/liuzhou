from __future__ import annotations

import pytest
import torch

from src.game_state import GameState, Phase, Player
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet, MARKOV_NUM_INPUT_CHANNELS
from src.policy_batch import TOTAL_DIM, action_to_index
from v1.python.train_bridge import train_network_from_tensors
from v1.python.trajectory_buffer import TensorSelfPlayBatch
from v1.python.symmetry import (
    D4_TRANSFORM_COUNT,
    d4_action_permutation,
    transform_game_state,
    transform_training_batch,
)


def _state_signature(state: GameState) -> tuple:
    return (
        tuple(tuple(row) for row in state.board),
        state.phase,
        state.current_player,
        frozenset(state.marked_black),
        frozenset(state.marked_white),
        state.forced_removals_done,
        state.move_count,
        state.pending_marks_required,
        state.pending_marks_remaining,
        state.pending_captures_required,
        state.pending_captures_remaining,
        state.moves_since_capture,
    )


def _representative_states() -> list[GameState]:
    placement = GameState()
    board = [[0] * 6 for _ in range(6)]
    for col in range(5):
        board[0][col] = Player.BLACK.value
    for row, col in ((2, 0), (2, 2), (3, 4), (4, 1), (5, 3), (5, 5)):
        board[row][col] = Player.WHITE.value
    mark = apply_move(
        GameState(board=board, phase=Phase.PLACEMENT, current_player=Player.BLACK),
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (0, 5)},
        quiet=True,
    )

    movement_board = [[0] * 6 for _ in range(6)]
    for position in ((0, 0), (0, 1), (1, 0), (2, 1), (4, 4)):
        movement_board[position[0]][position[1]] = Player.BLACK.value
    for position in ((2, 3), (2, 5), (3, 4), (4, 0), (5, 2), (5, 5)):
        movement_board[position[0]][position[1]] = Player.WHITE.value
    movement = GameState(
        board=movement_board,
        phase=Phase.MOVEMENT,
        current_player=Player.BLACK,
        move_count=51,
        moves_since_capture=11,
    )
    capture = apply_move(
        movement,
        {
            "phase": Phase.MOVEMENT,
            "action_type": "move",
            "from_position": (2, 1),
            "to_position": (1, 1),
        },
        quiet=True,
    )
    removal = GameState(
        board=[row[:] for row in movement_board],
        phase=Phase.REMOVAL,
        current_player=Player.WHITE,
        marked_black={(0, 0)},
    )
    forced = GameState(
        board=[row[:] for row in movement_board],
        phase=Phase.FORCED_REMOVAL,
        current_player=Player.WHITE,
        forced_removals_done=0,
    )
    counter = GameState(
        board=[row[:] for row in movement_board],
        phase=Phase.COUNTER_REMOVAL,
        current_player=Player.WHITE,
    )
    return [placement, mark, removal, movement, capture, forced, counter]


def test_all_d4_action_permutations_are_bijections() -> None:
    for transform in range(D4_TRANSFORM_COUNT):
        permutation = d4_action_permutation(transform)
        assert len(permutation) == TOTAL_DIM
        assert sorted(permutation) == list(range(TOTAL_DIM))
        assert permutation[-4:] == tuple(range(TOTAL_DIM - 4, TOTAL_DIM))
        assert any(
            all(
                d4_action_permutation(inverse)[permutation[index]] == index
                for index in range(TOTAL_DIM)
            )
            for inverse in range(D4_TRANSFORM_COUNT)
        )


def test_d4_transform_commutes_with_all_atomic_phase_transitions() -> None:
    states = _representative_states()
    assert {state.phase for state in states} == set(Phase)
    for state in states:
        moves = generate_all_legal_moves(state)
        for transform in range(D4_TRANSFORM_COUNT):
            transformed_state = transform_game_state(state, transform)
            transformed_moves = {
                int(action_to_index(move, 6)): move
                for move in generate_all_legal_moves(transformed_state)
            }
            permutation = d4_action_permutation(transform)
            assert sorted(transformed_moves) == sorted(
                permutation[int(action_to_index(move, 6))] for move in moves
            )
            for move in moves:
                transformed_action = permutation[int(action_to_index(move, 6))]
                expected = transform_game_state(
                    apply_move(state, move, quiet=True),
                    transform,
                )
                observed = apply_move(
                    transformed_state,
                    transformed_moves[transformed_action],
                    quiet=True,
                )
                assert _state_signature(observed) == _state_signature(expected)


def test_training_batch_transform_preserves_masks_targets_and_normalization() -> None:
    states = torch.arange(2 * 19 * 36, dtype=torch.float32).view(2, 19, 6, 6)
    masks = torch.zeros((2, TOTAL_DIM), dtype=torch.bool)
    targets = torch.zeros((2, TOTAL_DIM), dtype=torch.float32)
    masks[0, [0, 36, 180, 216]] = True
    masks[1, [35, 179, 215, 219]] = True
    targets[0, [0, 36, 180, 216]] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    targets[1, [35, 179, 215, 219]] = torch.tensor([0.4, 0.3, 0.2, 0.1])

    out_states, out_masks, out_targets = transform_training_batch(
        states,
        masks,
        targets,
        torch.tensor([1, 7]),
    )
    assert out_states.shape == states.shape
    assert torch.equal(out_masks.sum(dim=1), masks.sum(dim=1))
    assert torch.allclose(out_targets.sum(dim=1), torch.ones(2))
    assert torch.count_nonzero(out_targets.masked_fill(out_masks, 0.0)) == 0


def test_train_bridge_random_d4_is_audited() -> None:
    torch.manual_seed(21)
    states = torch.randn(4, MARKOV_NUM_INPUT_CHANNELS, 6, 6)
    masks = torch.zeros((4, TOTAL_DIM), dtype=torch.bool)
    targets = torch.zeros((4, TOTAL_DIM), dtype=torch.float32)
    for row in range(4):
        actions = torch.tensor([row, 36 + row * 4, 180 + row, 216])
        masks[row, actions] = True
        targets[row, actions] = 0.25
    samples = TensorSelfPlayBatch(
        state_tensors=states,
        legal_masks=masks,
        policy_targets=targets,
        value_targets=torch.tensor([1.0, -1.0, 0.0, 1.0]),
        soft_value_targets=torch.zeros(4),
    )
    model = ChessNet(
        num_input_channels=MARKOV_NUM_INPUT_CHANNELS,
        trunk_channels=8,
        num_blocks=1,
        policy_channels=4,
        value_channels=4,
        value_mlp_channels=8,
    )
    _model, metrics = train_network_from_tensors(
        model,
        samples,
        batch_size=2,
        epochs=2,
        device="cpu",
        use_amp=False,
        parallel_strategy="none",
        symmetry_augmentation="random_d4",
    )
    assert metrics["symmetry_augmentation"] == "random_d4"
    assert sum(metrics["symmetry_transform_counts"]) == 8
    assert torch.isfinite(
        torch.tensor(metrics["epoch_stats"][-1]["avg_loss"])
    )


def test_phase3_strict_optimizer_restore_does_not_fresh_start(
    tmp_path,
) -> None:
    samples = TensorSelfPlayBatch(
        state_tensors=torch.zeros(1, MARKOV_NUM_INPUT_CHANNELS, 6, 6),
        legal_masks=torch.nn.functional.one_hot(
            torch.tensor([0]), TOTAL_DIM
        ).bool(),
        policy_targets=torch.nn.functional.one_hot(
            torch.tensor([0]), TOTAL_DIM
        ).float(),
        value_targets=torch.ones(1),
        soft_value_targets=torch.ones(1),
    )
    model = ChessNet(
        num_input_channels=MARKOV_NUM_INPUT_CHANNELS,
        trunk_channels=8,
        num_blocks=1,
        policy_channels=4,
        value_channels=4,
        value_mlp_channels=8,
    )
    with pytest.raises(FileNotFoundError, match="Required optimizer state"):
        train_network_from_tensors(
            model,
            samples,
            batch_size=1,
            epochs=1,
            device="cpu",
            use_amp=False,
            parallel_strategy="none",
            optimizer_state_path=str(tmp_path / "missing_optimizer.pt"),
            optimizer_name="adamw",
            strict_optimizer_restore=True,
        )
