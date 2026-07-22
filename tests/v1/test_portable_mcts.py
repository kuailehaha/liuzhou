"""Correctness contracts for the portable CPU/MPS reference search."""

from __future__ import annotations

import importlib
import math
import sys

import pytest
import torch

from src.game_state import GameState, Phase, Player
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from src.policy_batch import TOTAL_DIM


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


def _state(*, phase: Phase, player: Player) -> GameState:
    return GameState(phase=phase, current_player=player)


def test_portable_modules_import_without_v0_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "v0_core", None)
    for name in (
        "v1.python",
        "v1.python.portable_device",
        "v1.python.portable_mcts",
        "v1.python.portable_self_play",
        "v1.train",
        "scripts.eval_checkpoint",
    ):
        sys.modules.pop(name, None)
    package = importlib.import_module("v1.python")
    assert package is not None
    assert importlib.import_module("v1.python.portable_mcts") is not None
    assert importlib.import_module("v1.python.portable_self_play") is not None
    assert importlib.import_module("v1.train") is not None
    assert importlib.import_module("scripts.eval_checkpoint") is not None


@pytest.mark.parametrize(
    ("parent_phase", "child_phase"),
    [
        (Phase.MARK_SELECTION, Phase.MARK_SELECTION),
        (Phase.MARK_SELECTION, Phase.REMOVAL),
        (Phase.CAPTURE_SELECTION, Phase.CAPTURE_SELECTION),
        (Phase.COUNTER_REMOVAL, Phase.MOVEMENT),
    ],
)
def test_backup_keeps_sign_across_same_player_atomic_phases(
    parent_phase: Phase,
    child_phase: Phase,
) -> None:
    from v1.python.portable_mcts import PortableNode, backup_path

    parent = PortableNode(state=_state(phase=parent_phase, player=Player.BLACK))
    child = PortableNode(
        state=_state(phase=child_phase, player=Player.BLACK),
        parent=parent,
        prior=1.0,
        action_index=0,
    )
    backup_path([parent, child], 0.75)
    assert child.mean_value == pytest.approx(0.75)
    assert parent.mean_value == pytest.approx(0.75)


def test_backup_flips_only_when_side_to_move_changes() -> None:
    from v1.python.portable_mcts import PortableNode, backup_path

    parent = PortableNode(state=_state(phase=Phase.MOVEMENT, player=Player.BLACK))
    child = PortableNode(
        state=_state(phase=Phase.MOVEMENT, player=Player.WHITE),
        parent=parent,
        prior=1.0,
        action_index=0,
    )
    backup_path([parent, child], 0.5)
    assert child.mean_value == pytest.approx(0.5)
    assert parent.mean_value == pytest.approx(-0.5)


def _assert_backup_matches_real_transition(parent_state: GameState, move: dict) -> GameState:
    """Exercise rules first, then verify backup from the resulting player transition."""

    from v1.python.portable_mcts import PortableNode, backup_path

    child_state = apply_move(parent_state, move, quiet=True)
    parent = PortableNode(state=parent_state)
    child = PortableNode(state=child_state, parent=parent, prior=1.0, action_index=0)
    backup_path([parent, child], 0.75)
    expected = 0.75 if parent_state.current_player == child_state.current_player else -0.75
    assert child.mean_value == pytest.approx(0.75)
    assert parent.mean_value == pytest.approx(expected)
    return child_state


def test_line_creation_keeps_player_for_two_real_mark_actions_then_switches() -> None:
    board = [[0] * GameState.BOARD_SIZE for _ in range(GameState.BOARD_SIZE)]
    for col in range(5):
        board[0][col] = Player.BLACK.value
    for position in ((2, 0), (2, 2), (3, 4), (4, 1), (5, 3)):
        board[position[0]][position[1]] = Player.WHITE.value
    state = GameState(board=board, phase=Phase.PLACEMENT, current_player=Player.BLACK)

    state = _assert_backup_matches_real_transition(
        state,
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (0, 5)},
    )
    assert state.phase == Phase.MARK_SELECTION
    assert state.current_player == Player.BLACK
    assert state.pending_marks_remaining == 2

    first_mark = generate_all_legal_moves(state)[0]
    state = _assert_backup_matches_real_transition(state, first_mark)
    assert state.phase == Phase.MARK_SELECTION
    assert state.current_player == Player.BLACK
    assert state.pending_marks_remaining == 1

    second_mark = generate_all_legal_moves(state)[0]
    state = _assert_backup_matches_real_transition(state, second_mark)
    assert state.phase == Phase.PLACEMENT
    assert state.current_player == Player.WHITE


def test_line_creation_keeps_player_for_two_real_capture_actions_then_switches() -> None:
    board = [[0] * GameState.BOARD_SIZE for _ in range(GameState.BOARD_SIZE)]
    for col in range(5):
        board[0][col] = Player.BLACK.value
    board[1][5] = Player.BLACK.value
    for position in ((2, 0), (2, 2), (3, 4), (4, 1), (4, 5), (5, 3)):
        board[position[0]][position[1]] = Player.WHITE.value
    state = GameState(board=board, phase=Phase.MOVEMENT, current_player=Player.BLACK)

    state = _assert_backup_matches_real_transition(
        state,
        {
            "phase": Phase.MOVEMENT,
            "action_type": "move",
            "from_position": (1, 5),
            "to_position": (0, 5),
        },
    )
    assert state.phase == Phase.CAPTURE_SELECTION
    assert state.current_player == Player.BLACK
    assert state.pending_captures_remaining == 2

    first_capture = generate_all_legal_moves(state)[0]
    state = _assert_backup_matches_real_transition(state, first_capture)
    assert state.phase == Phase.CAPTURE_SELECTION
    assert state.current_player == Player.BLACK
    assert state.pending_captures_remaining == 1

    second_capture = generate_all_legal_moves(state)[0]
    state = _assert_backup_matches_real_transition(state, second_capture)
    assert state.phase == Phase.MOVEMENT
    assert state.current_player == Player.WHITE


def test_square_creation_real_mark_and_capture_edges_keep_then_switch_player() -> None:
    placement_board = [[0] * GameState.BOARD_SIZE for _ in range(GameState.BOARD_SIZE)]
    for position in ((0, 0), (0, 1), (1, 0)):
        placement_board[position[0]][position[1]] = Player.BLACK.value
    for position in ((2, 0), (2, 2), (3, 4), (4, 1), (5, 3)):
        placement_board[position[0]][position[1]] = Player.WHITE.value
    placement = GameState(
        board=placement_board,
        phase=Phase.PLACEMENT,
        current_player=Player.BLACK,
    )
    placement = _assert_backup_matches_real_transition(
        placement,
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (1, 1)},
    )
    assert placement.phase == Phase.MARK_SELECTION
    assert placement.current_player == Player.BLACK
    assert placement.pending_marks_remaining == 1
    placement = _assert_backup_matches_real_transition(
        placement, generate_all_legal_moves(placement)[0]
    )
    assert placement.phase == Phase.PLACEMENT
    assert placement.current_player == Player.WHITE

    movement_board = [[0] * GameState.BOARD_SIZE for _ in range(GameState.BOARD_SIZE)]
    for position in ((0, 0), (0, 1), (1, 0), (2, 1)):
        movement_board[position[0]][position[1]] = Player.BLACK.value
    for position in ((2, 3), (2, 5), (3, 4), (4, 0), (5, 2)):
        movement_board[position[0]][position[1]] = Player.WHITE.value
    movement = GameState(
        board=movement_board,
        phase=Phase.MOVEMENT,
        current_player=Player.BLACK,
    )
    movement = _assert_backup_matches_real_transition(
        movement,
        {
            "phase": Phase.MOVEMENT,
            "action_type": "move",
            "from_position": (2, 1),
            "to_position": (1, 1),
        },
    )
    assert movement.phase == Phase.CAPTURE_SELECTION
    assert movement.current_player == Player.BLACK
    assert movement.pending_captures_remaining == 1
    movement = _assert_backup_matches_real_transition(
        movement, generate_all_legal_moves(movement)[0]
    )
    assert movement.phase == Phase.MOVEMENT
    assert movement.current_player == Player.WHITE


def test_real_removal_forced_removal_and_counter_removal_turn_semantics() -> None:
    board = [[0] * GameState.BOARD_SIZE for _ in range(GameState.BOARD_SIZE)]
    for position in ((0, 0), (0, 2), (1, 4), (3, 1), (5, 5)):
        board[position[0]][position[1]] = Player.BLACK.value
    for position in ((0, 5), (2, 1), (3, 4), (4, 0), (5, 2)):
        board[position[0]][position[1]] = Player.WHITE.value

    removal = GameState(
        board=[row[:] for row in board],
        phase=Phase.REMOVAL,
        current_player=Player.WHITE,
        marked_black={(0, 0)},
    )
    after_removal = _assert_backup_matches_real_transition(
        removal,
        {"phase": Phase.REMOVAL, "action_type": "process_removal"},
    )
    assert after_removal.phase == Phase.MOVEMENT
    assert after_removal.current_player == Player.WHITE

    forced = GameState(
        board=[row[:] for row in board],
        phase=Phase.FORCED_REMOVAL,
        current_player=Player.WHITE,
        forced_removals_done=0,
    )
    first_forced = generate_all_legal_moves(forced)[0]
    forced = _assert_backup_matches_real_transition(forced, first_forced)
    assert forced.phase == Phase.FORCED_REMOVAL
    assert forced.current_player == Player.BLACK
    second_forced = generate_all_legal_moves(forced)[0]
    forced = _assert_backup_matches_real_transition(forced, second_forced)
    assert forced.phase == Phase.MOVEMENT
    assert forced.current_player == Player.WHITE

    counter = GameState(
        board=[row[:] for row in board],
        phase=Phase.COUNTER_REMOVAL,
        current_player=Player.WHITE,
    )
    counter_move = generate_all_legal_moves(counter)[0]
    after_counter = _assert_backup_matches_real_transition(counter, counter_move)
    assert after_counter.phase == Phase.MOVEMENT
    assert after_counter.current_player == Player.BLACK


def test_deep_negative_evidence_can_lower_a_shallow_action_value() -> None:
    from v1.python.portable_mcts import PortableNode, backup_path, value_for_parent

    root = PortableNode(state=_state(phase=Phase.MOVEMENT, player=Player.BLACK))
    action = PortableNode(
        state=_state(phase=Phase.CAPTURE_SELECTION, player=Player.BLACK),
        parent=root,
        prior=1.0,
        action_index=0,
    )
    deep_leaf = PortableNode(
        state=_state(phase=Phase.MOVEMENT, player=Player.WHITE),
        parent=action,
        prior=1.0,
        action_index=1,
    )
    backup_path([root, action], 0.8)
    shallow_q = value_for_parent(root, action)
    backup_path([root, action, deep_leaf], 1.0)
    deep_q = value_for_parent(root, action)
    assert shallow_q == pytest.approx(0.8)
    assert deep_q < shallow_q
    assert deep_q == pytest.approx(-0.1)


def test_policy_is_220d_normalized_visit_distribution() -> None:
    from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig, PortableTree

    search = PortableMCTS(
        model=_small_model(),
        config=PortableMCTSConfig(
            num_simulations=8,
            temperature=1.0,
            add_dirichlet_noise=False,
            sample_moves=False,
        ),
        device="cpu",
    )
    output = search.search_batch([PortableTree(GameState())])[0]
    assert output.legal_mask.shape == (TOTAL_DIM,)
    assert output.policy_dense.shape == (TOTAL_DIM,)
    assert output.policy_dense[~output.legal_mask].count_nonzero().item() == 0
    assert float(output.policy_dense.sum().item()) == pytest.approx(1.0, abs=1e-6)
    assert sum(output.visit_counts.values()) == 8


def test_root_puct_reuses_a_fixed_q_after_first_visit() -> None:
    from v1.python.portable_root_puct import allocate_fixed_q_visits

    priors = torch.tensor([0.6, 0.4], dtype=torch.float32)
    fixed_q = torch.tensor([0.2, -0.1], dtype=torch.float32)
    visits, value_sum = allocate_fixed_q_visits(
        priors,
        fixed_q,
        torch.tensor([True, True]),
        num_simulations=32,
        exploration_weight=1.0,
    )
    visited = visits.gt(0)
    assert torch.allclose(value_sum[visited] / visits[visited], fixed_q[visited])


def test_partial_terminal_batch_keeps_tree_row_mapping() -> None:
    from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig, PortableTree

    terminal = GameState(phase=Phase.MOVEMENT, current_player=Player.BLACK)
    terminal.board[0][0] = Player.BLACK.value
    terminal.board[0][1] = Player.BLACK.value
    terminal.board[0][2] = Player.BLACK.value
    terminal.board[0][3] = Player.BLACK.value
    terminal.board[5][0] = Player.WHITE.value
    terminal.board[5][1] = Player.WHITE.value
    terminal.board[5][2] = Player.WHITE.value
    assert terminal.is_game_over()

    search = PortableMCTS(
        model=_small_model(),
        config=PortableMCTSConfig(num_simulations=4, add_dirichlet_noise=False),
        device="cpu",
    )
    outputs = search.search_batch([PortableTree(terminal), PortableTree(GameState())])
    assert len(outputs) == 2
    assert outputs[0].terminal
    assert outputs[0].chosen_action_index is None
    assert not outputs[1].terminal
    assert outputs[1].chosen_action_index is not None


def test_no_legal_nonterminal_state_is_an_explicit_loss() -> None:
    from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig, PortableTree

    state = GameState(
        phase=Phase.MARK_SELECTION,
        current_player=Player.BLACK,
        pending_marks_required=1,
        pending_marks_remaining=1,
    )
    assert not state.is_game_over()
    assert generate_all_legal_moves(state) == []
    tree = PortableTree(state)
    output = PortableMCTS(
        _small_model(),
        PortableMCTSConfig(num_simulations=2, add_dirichlet_noise=False),
        "cpu",
    ).search_batch([tree])[0]
    assert output.terminal
    assert output.chosen_action_index is None
    assert tree.root.no_legal_terminal
    assert tree.root.initial_value == pytest.approx(-1.0)


def test_advance_root_reuses_selected_subtree() -> None:
    from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig, PortableTree

    tree = PortableTree(GameState())
    search = PortableMCTS(
        model=_small_model(),
        config=PortableMCTSConfig(num_simulations=4, add_dirichlet_noise=False),
        device="cpu",
    )
    output = search.search_batch([tree])[0]
    chosen = tree.root.children[int(output.chosen_action_index)]
    old_visits = chosen.visit_count
    assert tree.advance_root(int(output.chosen_action_index))
    assert tree.root is chosen
    assert tree.root.parent is None
    assert tree.root.visit_count == old_visits


def test_reused_expanded_root_receives_fresh_dirichlet_noise() -> None:
    from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig, PortableTree

    tree = PortableTree(GameState())
    search = PortableMCTS(
        model=_small_model(),
        config=PortableMCTSConfig(
            num_simulations=4,
            add_dirichlet_noise=False,
            sample_moves=False,
            dirichlet_epsilon=1.0,
        ),
        device="cpu",
    )
    output = search.search_batch([tree], add_dirichlet_noise=False)[0]
    assert tree.advance_root(int(output.chosen_action_index))
    assert tree.root.expanded
    before = [tree.root.children[index].prior for index in sorted(tree.root.children)]
    torch.manual_seed(123)
    search.search_batch([tree], add_dirichlet_noise=True)
    after = [tree.root.children[index].prior for index in sorted(tree.root.children)]
    assert after != pytest.approx(before)
    assert sum(after) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_cpu_mps_policy_value_parity_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    cpu_model = _small_model()
    mps_model = _small_model()
    mps_model.load_state_dict(cpu_model.state_dict())
    config = PortableMCTSConfig(num_simulations=2, add_dirichlet_noise=False)
    states = [GameState(), GameState(current_player=Player.WHITE)]
    cpu_eval = PortableMCTS(cpu_model, config, "cpu").evaluate_states(states)
    mps_eval = PortableMCTS(mps_model, config, "mps").evaluate_states(states)
    assert torch.equal(cpu_eval.legal_masks, mps_eval.legal_masks)
    assert torch.allclose(cpu_eval.priors, mps_eval.priors, atol=2e-4, rtol=2e-4)
    assert torch.allclose(cpu_eval.values, mps_eval.values, atol=2e-4, rtol=2e-4)
    assert mps_eval.fallback_count == 0


def test_portable_self_play_tensor_contract_cpu() -> None:
    from v1.python.portable_self_play import self_play_v1_portable

    samples, stats = self_play_v1_portable(
        model=_small_model(),
        num_games=1,
        mcts_simulations=2,
        temperature_init=1.0,
        temperature_final=1.0,
        temperature_threshold=4,
        exploration_weight=1.0,
        device="cpu",
        add_dirichlet_noise=False,
        max_game_plies=8,
        sample_moves=False,
        concurrent_games=1,
    )
    assert samples.state_tensors.shape == (samples.num_samples, 11, 6, 6)
    assert samples.legal_masks.shape == (samples.num_samples, TOTAL_DIM)
    assert samples.policy_targets.shape == (samples.num_samples, TOTAL_DIM)
    assert samples.value_targets.shape == (samples.num_samples,)
    assert samples.soft_value_targets.shape == (samples.num_samples,)
    assert stats.num_games == 1
    assert stats.device == "cpu"
    assert stats.fallback_count == 0
    assert all(math.isfinite(float(x)) for x in samples.policy_targets.flatten())


def test_mps_fallback_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from v1.python.portable_device import resolve_portable_device

    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="silently execute unsupported operators"):
            resolve_portable_device("mps")
    else:
        with pytest.raises(RuntimeError, match="MPS was explicitly requested"):
            resolve_portable_device("mps")


def test_auto_cpu_fallback_is_explicitly_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    from v1.python.portable_device import resolve_portable_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    resolution = resolve_portable_device("auto")
    assert resolution.device == torch.device("cpu")
    assert resolution.fallback_count == 1
    assert resolution.fallback_reasons == (
        "auto selected CPU because torch.backends.mps.is_available() is False",
    )


def test_optimizer_state_load_failure_is_reported(tmp_path) -> None:
    from v1.python.portable_self_play import self_play_v1_portable
    from v1.python.train_bridge import train_network_from_tensors

    model = _small_model()
    samples, _stats = self_play_v1_portable(
        model=model,
        num_games=1,
        mcts_simulations=1,
        temperature_init=1.0,
        temperature_final=1.0,
        temperature_threshold=2,
        exploration_weight=1.0,
        device="cpu",
        add_dirichlet_noise=False,
        max_game_plies=2,
        sample_moves=False,
        concurrent_games=1,
    )
    invalid_state = tmp_path / "invalid_optimizer.pt"
    torch.save({"not_an_optimizer": True}, invalid_state)
    _model, metrics = train_network_from_tensors(
        model=model,
        samples=samples,
        batch_size=2,
        epochs=1,
        device="cpu",
        use_amp=False,
        parallel_strategy="none",
        optimizer_state_path=str(invalid_state),
    )
    assert not metrics["optimizer_loaded"]
    assert metrics["optimizer_load_error"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_train_bridge_uses_mps_local_batch_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    from v1.python.portable_self_play import self_play_v1_portable
    from v1.python.train_bridge import train_network_from_tensors

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    model = _small_model()
    samples, _stats = self_play_v1_portable(
        model=model,
        num_games=1,
        mcts_simulations=1,
        temperature_init=1.0,
        temperature_final=1.0,
        temperature_threshold=2,
        exploration_weight=1.0,
        device="mps",
        add_dirichlet_noise=False,
        max_game_plies=2,
        sample_moves=False,
        concurrent_games=1,
    )
    _model, metrics = train_network_from_tensors(
        model=model,
        samples=samples,
        batch_size=2,
        epochs=1,
        device="mps",
        use_amp=False,
        parallel_strategy="none",
    )
    assert metrics["device"] == "mps"
    assert metrics["device_fallback_count"] == 0
    assert math.isfinite(float(metrics["epoch_stats"][0]["avg_loss"]))
