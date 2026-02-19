"""Multiprocess self-play worker entry for v1."""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict

import torch

from src.game_state import GameState
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from .self_play_gpu_runner import self_play_v1_gpu


def _is_stream_capture_error(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    return (
        "stream is capturing" in text
        or "cudaerrorstreamcaptureunsupported" in text
        or "operation not permitted when stream is capturing" in text
    )


def run_self_play_worker(
    *,
    worker_idx: int,
    shard_device: str,
    shard_games: int,
    seed: int,
    model_state_path: str,
    output_path: str,
    mcts_simulations: int,
    temperature_init: float,
    temperature_final: float,
    temperature_threshold: int,
    exploration_weight: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    soft_value_k: float,
    max_game_plies: int,
    concurrent_games_per_device: int,
) -> Dict[str, Any]:
    """Run one self-play shard inside a dedicated process and persist shard payload."""

    try:
        local_seed = int(seed)
        torch.manual_seed(local_seed)
        dev = torch.device(str(shard_device))
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
            torch.cuda.manual_seed(local_seed)

        state_payload = torch.load(str(model_state_path), map_location="cpu")
        if not isinstance(state_payload, dict):
            raise RuntimeError(
                f"Invalid model_state payload type: {type(state_payload)!r} ({model_state_path})"
            )

        shard_games_i = int(max(0, int(shard_games)))
        if shard_games_i <= 0:
            raise ValueError(f"shard_games must be positive in worker, got {shard_games_i}")
        shard_concurrent = max(1, min(shard_games_i, int(concurrent_games_per_device)))

        def _run_once() -> tuple[Any, Any]:
            model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
            model.load_state_dict(state_payload, strict=True)
            model.to(dev)
            model.eval()
            return self_play_v1_gpu(
                model=model,
                num_games=shard_games_i,
                mcts_simulations=int(mcts_simulations),
                temperature_init=float(temperature_init),
                temperature_final=float(temperature_final),
                temperature_threshold=int(temperature_threshold),
                exploration_weight=float(exploration_weight),
                device=str(dev),
                add_dirichlet_noise=True,
                dirichlet_alpha=float(dirichlet_alpha),
                dirichlet_epsilon=float(dirichlet_epsilon),
                soft_value_k=float(soft_value_k),
                max_game_plies=int(max_game_plies),
                sample_moves=True,
                concurrent_games=shard_concurrent,
                verbose=False,
            )

        graph_env_explicit = "V1_FINALIZE_GRAPH" in os.environ
        graph_retry_off = False
        try:
            samples, stats = _run_once()
        except Exception as first_exc:
            if (not graph_env_explicit) and _is_stream_capture_error(first_exc):
                os.environ["V1_FINALIZE_GRAPH"] = "off"
                graph_retry_off = True
                samples, stats = _run_once()
            else:
                raise

        payload = {
            "state_tensors": samples.state_tensors.detach().cpu(),
            "legal_masks": samples.legal_masks.detach().cpu(),
            "policy_targets": samples.policy_targets.detach().cpu(),
            "value_targets": samples.value_targets.detach().cpu(),
            "soft_value_targets": samples.soft_value_targets.detach().cpu(),
            "stats": stats.to_dict(),
            "metadata": {
                "worker_idx": int(worker_idx),
                "device": str(dev),
                "games": int(shard_games_i),
                "graph_retry_off": bool(graph_retry_off),
            },
        }
        os.makedirs(os.path.dirname(str(output_path)) or ".", exist_ok=True)
        torch.save(payload, str(output_path))

        return {
            "worker_idx": int(worker_idx),
            "device": str(dev),
            "games": int(shard_games_i),
            "output_path": str(output_path),
            "num_samples": int(samples.num_samples),
        }
    except Exception as exc:
        detail = traceback.format_exc()
        raise RuntimeError(
            "v1 self-play process worker failed: "
            f"worker={int(worker_idx)}, device={str(shard_device)}, games={int(shard_games)}\n{detail}"
        ) from exc
