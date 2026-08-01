from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _arena(batch: int = 1, nodes: int = 5, actions: int = 4):
    import v0_core

    device = torch.device("cuda")
    arena = {
        "priors": torch.zeros((batch, nodes, actions), dtype=torch.float32, device=device),
        "legal": torch.zeros((batch, nodes, actions), dtype=torch.bool, device=device),
        "children": torch.full((batch, nodes, actions), -1, dtype=torch.int32, device=device),
        "visits": torch.zeros((batch, nodes), dtype=torch.int32, device=device),
        "value_sum": torch.zeros((batch, nodes), dtype=torch.float64, device=device),
        "player": torch.zeros((batch, nodes), dtype=torch.int8, device=device),
        "expanded": torch.zeros((batch, nodes), dtype=torch.bool, device=device),
        "terminal": torch.zeros((batch, nodes), dtype=torch.bool, device=device),
        "parent": torch.full((batch, nodes), -1, dtype=torch.int32, device=device),
        "action": torch.full((batch, nodes), -1, dtype=torch.int32, device=device),
        "count": torch.ones((batch,), dtype=torch.int32, device=device),
    }
    assert hasattr(v0_core, "cuda_tree_select")
    return v0_core, arena


def _select(v0_core, arena, exploration: float = 1.0, max_path_length: int = 0):
    return v0_core.cuda_tree_select(
        arena["priors"],
        arena["legal"],
        arena["children"],
        arena["visits"],
        arena["value_sum"],
        arena["player"],
        arena["expanded"],
        arena["terminal"],
        arena["parent"],
        arena["action"],
        arena["count"],
        exploration,
        max_path_length,
    )


def test_cuda_tree_select_uses_action_index_tie_break_and_lazy_allocation():
    v0_core, arena = _arena()
    arena["player"][0, 0] = 1
    arena["expanded"][0, 0] = True
    arena["legal"][0, 0, :2] = True
    arena["priors"][0, 0, :2] = 0.5

    leaf, paths, lengths, actions, allocated, overflow = _select(v0_core, arena)

    assert leaf.cpu().tolist() == [1]
    assert paths[0, :2].cpu().tolist() == [0, 1]
    assert lengths.cpu().tolist() == [2]
    assert actions.cpu().tolist() == [0]
    assert allocated.cpu().tolist() == [True]
    assert overflow.cpu().tolist() == [False]
    assert arena["children"][0, 0, :2].cpu().tolist() == [1, -1]
    assert arena["parent"][0, 1].item() == 0
    assert arena["action"][0, 1].item() == 0
    assert arena["count"].item() == 2


def test_cuda_tree_backup_flips_only_when_side_to_move_changes():
    v0_core, arena = _arena(batch=2)
    for tree in range(2):
        arena["player"][tree, 0] = 1
        arena["expanded"][tree, 0] = True
        arena["legal"][tree, 0, 0] = True
        arena["priors"][tree, 0, 0] = 1.0

    leaf, paths, lengths, _actions, _allocated, _overflow = _select(v0_core, arena)
    arena["player"][0, leaf[0].to(torch.int64)] = -1
    arena["player"][1, leaf[1].to(torch.int64)] = 1
    v0_core.cuda_tree_backup(
        paths,
        lengths,
        arena["player"],
        torch.tensor([0.25, 0.5], dtype=torch.float64, device="cuda"),
        arena["visits"],
        arena["value_sum"],
    )

    torch.cuda.synchronize()
    assert arena["visits"].cpu().tolist() == [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]
    assert arena["value_sum"].cpu().tolist() == [
        [-0.25, 0.25, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
    ]


def test_cuda_tree_path_capacity_is_independent_from_arena_capacity():
    v0_core, arena = _arena(nodes=9)
    arena["player"][0, 0] = 1
    arena["expanded"][0, 0] = True
    arena["legal"][0, 0, 0] = True
    arena["priors"][0, 0, 0] = 1.0

    leaf, paths, lengths, _actions, _allocated, overflow = _select(
        v0_core, arena, max_path_length=3
    )
    assert tuple(paths.shape) == (1, 3)
    assert lengths.cpu().tolist() == [2]
    assert overflow.cpu().tolist() == [False]
    arena["player"][0, leaf[0].to(torch.int64)] = -1
    v0_core.cuda_tree_backup(
        paths,
        lengths,
        arena["player"],
        torch.tensor([0.5], dtype=torch.float64, device="cuda"),
        arena["visits"],
        arena["value_sum"],
    )
    assert arena["visits"][0, :2].cpu().tolist() == [1, 1]


def test_cuda_tree_reports_path_capacity_overflow_instead_of_truncating_silently():
    v0_core, arena = _arena(nodes=9)
    arena["count"][0] = 3
    for node in range(3):
        arena["player"][0, node] = 1
        arena["expanded"][0, node] = True
        arena["legal"][0, node, 0] = True
        arena["priors"][0, node, 0] = 1.0
    arena["children"][0, 0, 0] = 1
    arena["children"][0, 1, 0] = 2
    arena["parent"][0, 1] = 0
    arena["parent"][0, 2] = 1

    _leaf, paths, lengths, _actions, allocated, overflow = _select(
        v0_core, arena, max_path_length=2
    )
    assert tuple(paths.shape) == (1, 2)
    assert lengths.cpu().tolist() == [2]
    assert allocated.cpu().tolist() == [False]
    assert overflow.cpu().tolist() == [True]


def test_cuda_tree_visited_child_q_affects_next_puct_choice():
    v0_core, arena = _arena()
    arena["player"][0, 0] = 1
    arena["expanded"][0, 0] = True
    arena["legal"][0, 0, :2] = True
    arena["priors"][0, 0, :2] = 0.5

    leaf, paths, lengths, _actions, _allocated, _overflow = _select(v0_core, arena)
    arena["player"][0, 1] = -1
    v0_core.cuda_tree_backup(
        paths,
        lengths,
        arena["player"],
        torch.tensor([0.25], dtype=torch.float64, device="cuda"),
        arena["visits"],
        arena["value_sum"],
    )
    leaf2, paths2, lengths2, actions2, allocated2, overflow2 = _select(v0_core, arena)

    assert actions2.cpu().tolist() == [1]
    assert leaf2.cpu().tolist() == [2]
    assert paths2[0, :2].cpu().tolist() == [0, 2]
    assert lengths2.cpu().tolist() == [2]
    assert allocated2.cpu().tolist() == [True]
    assert overflow2.cpu().tolist() == [False]


def test_cuda_tree_terminal_root_is_inactive_for_backup():
    v0_core, arena = _arena()
    arena["player"][0, 0] = 1
    arena["expanded"][0, 0] = True
    arena["terminal"][0, 0] = True

    leaf, paths, lengths, actions, allocated, overflow = _select(v0_core, arena)
    v0_core.cuda_tree_backup(
        paths,
        lengths,
        arena["player"],
        torch.tensor([1.0], dtype=torch.float64, device="cuda"),
        arena["visits"],
        arena["value_sum"],
    )

    assert leaf.cpu().tolist() == [0]
    assert lengths.cpu().tolist() == [0]
    assert actions.cpu().tolist() == [-1]
    assert allocated.cpu().tolist() == [False]
    assert overflow.cpu().tolist() == [False]
    assert arena["visits"].count_nonzero().item() == 0
    assert arena["value_sum"].count_nonzero().item() == 0
