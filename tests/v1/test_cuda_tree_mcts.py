from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _UniformModel(torch.nn.Module):
    num_input_channels = 19

    def __init__(self) -> None:
        super().__init__()
        # PortableCppMCTS currently evaluates its getattr default eagerly.
        self.stem_conv = torch.nn.Conv2d(19, 1, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor):
        batch = int(inputs.shape[0])
        options = {"dtype": inputs.dtype, "device": inputs.device}
        policy = torch.zeros((batch, 36), **options)
        value = torch.zeros((batch, 1), **options)
        return policy, policy.clone(), policy.clone(), value


def _gpu_state_from_game_state(state):
    from v1.python.mcts_gpu import GpuStateBatch

    marks_black = torch.zeros((1, 6, 6), dtype=torch.bool, device="cuda:0")
    marks_white = torch.zeros((1, 6, 6), dtype=torch.bool, device="cuda:0")
    for row, col in state.marked_black:
        marks_black[0, row, col] = True
    for row, col in state.marked_white:
        marks_white[0, row, col] = True

    def scalar(value: int) -> torch.Tensor:
        return torch.tensor([value], dtype=torch.int64, device="cuda:0")

    return GpuStateBatch(
        board=torch.tensor(state.board, dtype=torch.int8, device="cuda:0").unsqueeze(0),
        marks_black=marks_black,
        marks_white=marks_white,
        phase=scalar(int(state.phase.value)),
        current_player=scalar(int(state.current_player.value)),
        pending_marks_required=scalar(int(state.pending_marks_required)),
        pending_marks_remaining=scalar(int(state.pending_marks_remaining)),
        pending_captures_required=scalar(int(state.pending_captures_required)),
        pending_captures_remaining=scalar(int(state.pending_captures_remaining)),
        forced_removals_done=scalar(int(state.forced_removals_done)),
        move_count=scalar(int(state.move_count)),
        moves_since_capture=scalar(int(state.moves_since_capture)),
    )


def test_cuda_tree_two_simulations_visit_first_two_tied_actions():
    import v0_core

    if not hasattr(v0_core, "cuda_tree_select"):
        pytest.fail("test loaded a v0_core binary without cuda_tree kernels")

    from v1.python.mcts_cuda_tree import CudaTreeMCTS, CudaTreeMCTSConfig
    from v1.python.mcts_gpu import GpuStateBatch

    search = CudaTreeMCTS(
        _UniformModel().cuda().eval(),
        CudaTreeMCTSConfig(
            num_simulations=2,
            exploration_weight=1.0,
            temperature=1.0,
            add_dirichlet_noise=False,
            sample_moves=False,
            precision="fp32",
        ),
        "cuda:0",
    )
    output = search.search_batch(GpuStateBatch.initial("cuda:0", batch_size=2))

    expected = torch.zeros((2, 220), dtype=torch.float32)
    expected[:, :2] = 0.5
    torch.testing.assert_close(output.policy_dense.cpu(), expected, atol=0.0, rtol=0.0)
    assert output.chosen_action_indices.cpu().tolist() == [0, 0]
    assert output.chosen_valid_mask.cpu().tolist() == [True, True]
    assert output.terminal_mask.cpu().tolist() == [False, False]
    assert tuple(output.model_input.shape) == (2, 19, 6, 6)
    assert output.model_input[:, 11].eq(1.0).all().item()
    assert output.model_input[:, 12:].eq(0.0).all().item()
    assert output.legal_mask[:, :36].all().item()
    assert not output.legal_mask[:, 36:].any().item()
    assert search.metrics()["arena_overflow"] == 0


def test_cuda_tree_accepts_smaller_repeated_simulation_chunks() -> None:
    from v1.python.mcts_cuda_tree import CudaTreeMCTS, CudaTreeMCTSConfig
    from v1.python.mcts_gpu import GpuStateBatch

    search = CudaTreeMCTS(
        _UniformModel().cuda().eval(),
        CudaTreeMCTSConfig(
            num_simulations=8,
            add_dirichlet_noise=False,
            sample_moves=False,
            precision="fp32",
        ),
        "cuda:0",
    )
    states = GpuStateBatch.initial("cuda:0", batch_size=1)

    search.search_batch(states, num_simulations=2)
    assert search.active_root_visit_counts().cpu().tolist() == [2]
    search.search_batch(states, num_simulations=3)
    assert search.active_root_visit_counts().cpu().tolist() == [5]

    metrics = search.metrics()
    assert int(metrics["leaf_evaluations"]) == 7
    assert int(metrics["inference_batches"]) == 7


def test_cuda_tree_padding_rows_do_not_report_reuse_or_state_mismatch() -> None:
    from v1.python.mcts_cuda_tree import CudaTreeMCTS, CudaTreeMCTSConfig
    from v1.python.mcts_gpu import GpuStateBatch, batch_apply_moves_compat
    from v1.python.self_play_gpu_runner import _pad_state_batch

    search = CudaTreeMCTS(
        _UniformModel().cuda().eval(),
        CudaTreeMCTSConfig(
            num_simulations=2,
            add_dirichlet_noise=False,
            sample_moves=False,
            precision="fp32",
        ),
        "cuda:0",
    )
    padded = GpuStateBatch.initial("cuda:0", batch_size=2)
    tree_ids = torch.tensor([7, -1], dtype=torch.int64, device="cuda:0")
    first = search.search_batch(padded, tree_ids=tree_ids)
    search.advance_roots(first.chosen_action_indices)

    real_next = batch_apply_moves_compat(
        padded.select(torch.tensor([0], dtype=torch.int64, device="cuda:0")),
        first.chosen_action_codes[:1],
        torch.zeros((1,), dtype=torch.int64, device="cuda:0"),
    )
    search.search_batch(_pad_state_batch(real_next, 2), tree_ids=tree_ids)

    metrics = search.metrics()
    assert int(metrics["subtree_reuse_hits"]) == 1
    assert int(metrics["subtree_reuse_misses"]) == 0
    assert int(metrics["subtree_state_mismatches"]) == 0


@pytest.mark.parametrize("simulations", [1, 2, 8, 64])
def test_cuda_tree_uniform_full_search_matches_portable_cpp(simulations: int):
    import v0_core

    if not hasattr(v0_core, "cuda_tree_select"):
        pytest.fail("test loaded a v0_core binary without cuda_tree kernels")

    from src.game_state import GameState
    from v1.python.mcts_cuda_tree import CudaTreeMCTS, CudaTreeMCTSConfig
    from v1.python.mcts_gpu import GpuStateBatch
    from v1.python.portable_cpp_loader import load_portable_cpp
    from v1.python.portable_cpp_mcts import PortableCppMCTS
    from v1.python.portable_mcts import PortableMCTSConfig

    load_portable_cpp(required=True)
    cuda_search = CudaTreeMCTS(
        _UniformModel().cuda().eval(),
        CudaTreeMCTSConfig(
            num_simulations=simulations,
            exploration_weight=1.0,
            temperature=1.0,
            add_dirichlet_noise=False,
            sample_moves=False,
            precision="fp32",
        ),
        "cuda:0",
    )
    cuda_output = cuda_search.search_batch(GpuStateBatch.initial("cuda:0", batch_size=1))

    portable = PortableCppMCTS(
        _UniformModel(),
        PortableMCTSConfig(
            num_simulations=simulations,
            exploration_weight=1.0,
            temperature=1.0,
            add_dirichlet_noise=False,
            sample_moves=False,
        ),
        "cuda:0",
        num_threads=1,
        initial_states=[GameState()],
    )
    portable_output = portable.search_batch(
        temperatures=1.0,
        add_dirichlet_noise=False,
    )[0]

    portable_visits = torch.zeros((220,), dtype=torch.int32)
    for action, count in portable_output.visit_counts.items():
        portable_visits[int(action)] = int(count)
    assert cuda_search.last_root_visits is not None
    assert torch.equal(cuda_search.last_root_visits[0].cpu(), portable_visits)
    torch.testing.assert_close(
        cuda_output.policy_dense[0].cpu(),
        portable_output.policy_dense,
        atol=1e-7,
        rtol=0.0,
    )
    assert int(cuda_output.chosen_action_indices[0].item()) == int(
        portable_output.chosen_action_index
    )


def test_cuda_tree_advance_root_reuses_full_selected_subtree_like_portable_cpp():
    from src.game_state import GameState
    from v1.python.mcts_cuda_tree import CudaTreeMCTS, CudaTreeMCTSConfig
    from v1.python.mcts_gpu import GpuStateBatch, batch_apply_moves_compat
    from v1.python.portable_cpp_loader import load_portable_cpp
    from v1.python.portable_cpp_mcts import PortableCppMCTS
    from v1.python.portable_mcts import PortableMCTSConfig

    load_portable_cpp(required=True)
    simulations = 8
    config = CudaTreeMCTSConfig(
        num_simulations=simulations,
        exploration_weight=1.0,
        temperature=1.0,
        add_dirichlet_noise=False,
        sample_moves=False,
        precision="fp32",
    )
    cuda_search = CudaTreeMCTS(_UniformModel().cuda().eval(), config, "cuda:0")
    cuda_states = GpuStateBatch.initial("cuda:0", batch_size=1)
    first_cuda = cuda_search.search_batch(cuda_states)

    portable = PortableCppMCTS(
        _UniformModel(),
        PortableMCTSConfig(
            num_simulations=simulations,
            exploration_weight=1.0,
            temperature=1.0,
            add_dirichlet_noise=False,
            sample_moves=False,
        ),
        "cuda:0",
        num_threads=1,
        initial_states=[GameState()],
    )
    first_portable = portable.search_batch(
        temperatures=1.0,
        add_dirichlet_noise=False,
    )[0]
    assert int(first_cuda.chosen_action_indices[0].item()) == int(
        first_portable.chosen_action_index
    )

    cuda_search.advance_roots(first_cuda.chosen_action_indices)
    portable.advance_roots([int(first_portable.chosen_action_index)])
    cuda_states = batch_apply_moves_compat(
        cuda_states,
        first_cuda.chosen_action_codes,
        torch.zeros((1,), dtype=torch.int64, device="cuda:0"),
    )

    second_cuda = cuda_search.search_batch(cuda_states)
    second_portable = portable.search_batch(
        temperatures=1.0,
        add_dirichlet_noise=False,
    )[0]
    portable_visits = torch.zeros((220,), dtype=torch.int32)
    for action, count in second_portable.visit_counts.items():
        portable_visits[int(action)] = int(count)

    assert cuda_search.last_root_visits is not None
    assert torch.equal(cuda_search.last_root_visits[0].cpu(), portable_visits)
    torch.testing.assert_close(
        second_cuda.policy_dense[0].cpu(),
        second_portable.policy_dense,
        atol=1e-7,
        rtol=0.0,
    )
    assert int(second_cuda.chosen_action_indices[0].item()) == int(
        second_portable.chosen_action_index
    )
    metrics = cuda_search.metrics()
    assert int(metrics["subtree_reuse_hits"]) == 1
    assert int(metrics["subtree_reuse_misses"]) == 0
    assert int(metrics["subtree_retained_nodes"]) > 0
    assert int(metrics["subtree_state_mismatches"]) == 0


def test_cuda_tree_reused_subtree_matches_portable_across_same_player_mark_phase():
    from src.game_state import GameState, Phase, Player
    from src.move_generator import apply_move
    from v1.python.mcts_cuda_tree import CudaTreeMCTS, CudaTreeMCTSConfig
    from v1.python.mcts_gpu import batch_apply_moves_compat
    from v1.python.portable_cpp_loader import load_portable_cpp
    from v1.python.portable_cpp_mcts import PortableCppMCTS
    from v1.python.portable_mcts import PortableMCTSConfig

    board = [[0] * 6 for _ in range(6)]
    for col in range(5):
        board[0][col] = Player.BLACK.value
    for position in ((2, 0), (2, 2), (3, 4), (4, 1), (5, 3)):
        board[position[0]][position[1]] = Player.WHITE.value
    state = apply_move(
        GameState(board=board, phase=Phase.PLACEMENT, current_player=Player.BLACK),
        {"phase": Phase.PLACEMENT, "action_type": "place", "position": (0, 5)},
    )
    assert state.phase == Phase.MARK_SELECTION
    assert state.current_player == Player.BLACK
    assert state.pending_marks_remaining == 2

    simulations = 8
    cuda_search = CudaTreeMCTS(
        _UniformModel().cuda().eval(),
        CudaTreeMCTSConfig(
            num_simulations=simulations,
            add_dirichlet_noise=False,
            sample_moves=False,
            precision="fp32",
        ),
        "cuda:0",
    )
    cuda_states = _gpu_state_from_game_state(state)
    load_portable_cpp(required=True)
    portable = PortableCppMCTS(
        _UniformModel(),
        PortableMCTSConfig(
            num_simulations=simulations,
            add_dirichlet_noise=False,
            sample_moves=False,
        ),
        "cuda:0",
        num_threads=1,
        initial_states=[state],
    )

    first_cuda = cuda_search.search_batch(cuda_states)
    first_portable = portable.search_batch(
        temperatures=1.0, add_dirichlet_noise=False
    )[0]
    assert int(first_cuda.chosen_action_indices[0].item()) == int(
        first_portable.chosen_action_index
    )
    cuda_search.advance_roots(first_cuda.chosen_action_indices)
    portable.advance_roots([int(first_portable.chosen_action_index)])
    cuda_states = batch_apply_moves_compat(
        cuda_states,
        first_cuda.chosen_action_codes,
        torch.zeros((1,), dtype=torch.int64, device="cuda:0"),
    )
    assert int(cuda_states.current_player[0].item()) == Player.BLACK.value
    assert int(cuda_states.phase[0].item()) == Phase.MARK_SELECTION.value

    second_cuda = cuda_search.search_batch(cuda_states)
    second_portable = portable.search_batch(
        temperatures=1.0, add_dirichlet_noise=False
    )[0]
    portable_visits = torch.zeros((220,), dtype=torch.int32)
    for action, count in second_portable.visit_counts.items():
        portable_visits[int(action)] = int(count)
    assert cuda_search.last_root_visits is not None
    assert torch.equal(cuda_search.last_root_visits[0].cpu(), portable_visits)
    assert int(second_cuda.chosen_action_indices[0].item()) == int(
        second_portable.chosen_action_index
    )


def test_cuda_tree_observes_and_resets_external_root_state_mismatch():
    from v1.python.mcts_cuda_tree import CudaTreeMCTS, CudaTreeMCTSConfig
    from v1.python.mcts_gpu import GpuStateBatch

    search = CudaTreeMCTS(
        _UniformModel().cuda().eval(),
        CudaTreeMCTSConfig(
            num_simulations=2,
            add_dirichlet_noise=False,
            sample_moves=False,
            precision="fp32",
        ),
        "cuda:0",
    )
    initial = GpuStateBatch.initial("cuda:0", batch_size=1)
    first = search.search_batch(initial)
    search.advance_roots(first.chosen_action_indices)
    # Deliberately omit applying the selected action. The persistent child is
    # therefore not the external root and must be rejected observably.
    search.search_batch(initial)
    metrics = search.metrics()
    assert int(metrics["subtree_state_mismatches"]) == 1
    assert int(metrics["subtree_capacity_resets"]) == 0
    assert int(metrics["arena_overflow"]) == 0


def test_cuda_tree_self_play_emits_standard_markov_payload_and_metrics():
    from v1.python.self_play_gpu_runner import self_play_v1_gpu

    samples, stats = self_play_v1_gpu(
        model=_UniformModel().cuda().eval(),
        num_games=2,
        mcts_simulations=2,
        temperature_init=1.0,
        temperature_final=1.0,
        temperature_threshold=99,
        exploration_weight=1.0,
        device="cuda:0",
        add_dirichlet_noise=False,
        soft_value_k=2.0,
        opening_random_moves=0,
        max_game_plies=4,
        sample_moves=False,
        concurrent_games=2,
        search_backend="cuda_tree",
        cuda_tree_precision="fp32",
    )

    assert samples.num_samples == 8
    assert tuple(samples.state_tensors.shape) == (8, 19, 6, 6)
    assert tuple(samples.legal_masks.shape) == (8, 220)
    assert tuple(samples.policy_targets.shape) == (8, 220)
    assert torch.isfinite(samples.state_tensors).all().item()
    assert torch.isfinite(samples.policy_targets).all().item()
    assert stats.mcts_metrics["search_backend"] == "cuda_tree"
    assert stats.mcts_metrics["precision"] == "fp32"
    assert int(stats.mcts_metrics["arena_overflow"]) == 0
    assert int(stats.mcts_metrics["fallback"]) == 0


def test_cuda_tree_refills_fixed_slots_without_losing_trajectories():
    from v1.python.self_play_gpu_runner import self_play_v1_gpu

    samples, stats = self_play_v1_gpu(
        model=_UniformModel().cuda().eval(),
        num_games=5,
        mcts_simulations=2,
        temperature_init=1.0,
        temperature_final=1.0,
        temperature_threshold=99,
        exploration_weight=1.0,
        device="cuda:0",
        add_dirichlet_noise=False,
        max_game_plies=4,
        sample_moves=False,
        concurrent_games=2,
        search_backend="cuda_tree",
        cuda_tree_precision="fp32",
    )

    assert stats.num_games == 5
    assert stats.num_positions == 20
    assert stats.avg_game_length == 4.0
    assert samples.num_samples == 20
    # Batch 2 is used while slots can be refilled; the final single-game tail
    # uses its own batch-1 graph instead of evaluating an inactive second row.
    assert int(stats.mcts_metrics["graph_capture"]) == 2
    assert int(stats.mcts_metrics["subtree_layout_migrations"]) == 1
    assert int(stats.mcts_metrics["subtree_reuse_hits"]) == stats.num_positions
    assert int(stats.mcts_metrics["subtree_reuse_misses"]) == 0
    assert int(stats.mcts_metrics["subtree_capacity_resets"]) == 0
    assert int(stats.mcts_metrics["subtree_state_mismatches"]) == 0
    assert int(stats.mcts_metrics["arena_overflow"]) == 0
