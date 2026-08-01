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
    assert int(stats.mcts_metrics["arena_overflow"]) == 0
