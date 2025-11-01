import pytest

torch = pytest.importorskip("torch")

from v1.common.tensor_utils import batched_index_select


def test_batched_index_select_single_index():
    tensor = torch.arange(12, dtype=torch.float32).view(3, 4)
    indices = torch.tensor([0, 2, 1])

    selected = batched_index_select(tensor, indices)
    expected = torch.tensor([0.0, 6.0, 9.0])
    torch.testing.assert_close(selected, expected)


def test_batched_index_select_multiple_indices():
    tensor = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    indices = torch.tensor([[0, 2], [1, 0]])

    selected = batched_index_select(tensor, indices)
    expected = torch.stack([tensor[0, [0, 2]], tensor[1, [1, 0]]], dim=0)
    torch.testing.assert_close(selected, expected)


def test_batched_index_select_invalid_batch():
    tensor = torch.zeros(2, 3)
    indices = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError):
        batched_index_select(tensor, indices)
