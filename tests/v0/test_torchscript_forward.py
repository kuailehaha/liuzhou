import os

import torch

import pytest

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS


v0_core = pytest.importorskip("v0_core")


def _export_torchscript(model: ChessNet, sample_input: torch.Tensor, path) -> None:
    scripted = torch.jit.trace(model, sample_input, strict=False)
    scripted.save(str(path))


def _fixed_input(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    board = GameState.BOARD_SIZE
    inputs = torch.zeros(
        batch_size,
        NUM_INPUT_CHANNELS,
        board,
        board,
        device=device,
        dtype=dtype,
    )
    for b in range(batch_size):
        r = b % board
        c = (b * 2) % board
        inputs[b, 0, r, c] = 1.0
        inputs[b, 1, (r + 1) % board, (c + 2) % board] = -0.5
        inputs[b, 2, (r + 2) % board, (c + 3) % board] = 0.25
    return inputs


def _dtype_from_string(value: str) -> torch.dtype:
    key = value.strip().lower()
    if key in ("float32", "fp32", "f32"):
        return torch.float32
    if key in ("float16", "fp16", "f16"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def _dtype_to_string(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return str(dtype)


def _tolerances_for_dtype(dtype: torch.dtype) -> tuple[float, float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2, 1e-2
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2, 3e-2
    return 1e-4, 1e-5, 1e-4


def _assert_argmax_matches(ref: torch.Tensor, out: torch.Tensor, min_margin: float) -> None:
    ref_cpu = ref.float().cpu()
    out_cpu = out.float().cpu()
    top2 = torch.topk(ref_cpu, k=2, dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1]
    stable = margin > min_margin
    if stable.any():
        ref_idx = top2.indices[:, 0]
        out_idx = out_cpu.argmax(dim=1)
        assert torch.equal(ref_idx[stable], out_idx[stable])


def _assert_outputs_match(
    ref: tuple[torch.Tensor, ...],
    out: tuple[torch.Tensor, ...],
    batch_size: int,
    dtype: torch.dtype,
) -> None:
    assert isinstance(out, tuple)
    assert len(out) == 4

    board_area = GameState.BOARD_SIZE * GameState.BOARD_SIZE
    expected_shapes = [
        (batch_size, board_area),
        (batch_size, board_area),
        (batch_size, board_area),
        (batch_size, 1),
    ]

    rtol, atol, margin = _tolerances_for_dtype(dtype)

    for idx, (ref_tensor, out_tensor) in enumerate(zip(ref, out)):
        assert tuple(out_tensor.shape) == expected_shapes[idx]
        ref_cmp = ref_tensor
        out_cmp = out_tensor
        if ref_cmp.dtype in (torch.float16, torch.bfloat16):
            ref_cmp = ref_cmp.float()
        if out_cmp.dtype in (torch.float16, torch.bfloat16):
            out_cmp = out_cmp.float()
        torch.testing.assert_close(out_cmp.cpu(), ref_cmp.cpu(), rtol=rtol, atol=atol)

    for head in range(3):
        _assert_argmax_matches(ref[head], out[head], margin)


def test_torchscript_runner_matches_eager(tmp_path) -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    model.to(device)
    model.eval()

    batch_size = 4
    random_inputs = torch.randn(
        batch_size,
        NUM_INPUT_CHANNELS,
        GameState.BOARD_SIZE,
        GameState.BOARD_SIZE,
        device=device,
        dtype=torch.float32,
    )
    fixed_inputs = _fixed_input(batch_size, device, torch.float32)

    model_path = tmp_path / "model.ts.pt"
    _export_torchscript(model, random_inputs, model_path)

    runner = v0_core.TorchScriptRunner(str(model_path), device="cpu", dtype="float32")

    with torch.inference_mode():
        ref_random = model(random_inputs)
        ref_fixed = model(fixed_inputs)

    out_random = runner.forward(random_inputs)
    out_fixed = runner.forward(fixed_inputs)

    _assert_outputs_match(ref_random, out_random, batch_size, torch.float32)
    _assert_outputs_match(ref_fixed, out_fixed, batch_size, torch.float32)


@pytest.mark.slow
def test_torchscript_runner_cuda_batch512_matches_eager(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dtype_str = os.environ.get("V0_TORCHSCRIPT_DTYPE", "float16")
    dtype = _dtype_from_string(dtype_str)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this CUDA device")

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    prev_det = torch.backends.cudnn.deterministic
    prev_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
        model.to(device=device, dtype=dtype)
        model.eval()

        batch_size = 512
        fixed_inputs = _fixed_input(batch_size, device, dtype)

        model_path = tmp_path / "model_cuda.ts.pt"
        _export_torchscript(model, fixed_inputs, model_path)

        runner = v0_core.TorchScriptRunner(
            str(model_path),
            device=str(device),
            dtype=_dtype_to_string(dtype),
        )

        with torch.inference_mode():
            ref = model(fixed_inputs)
        out = runner.forward(fixed_inputs)

        _assert_outputs_match(ref, out, batch_size, dtype)
    finally:
        torch.backends.cudnn.deterministic = prev_det
        torch.backends.cudnn.benchmark = prev_bench
