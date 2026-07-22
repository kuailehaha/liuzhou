#!/usr/bin/env python
"""Fixed-model Root-PUCT versus portable full-MCTS comparison.

This tool intentionally performs no training. It runs the current fixed-q
root-only algorithm and the portable full tree against identical model weights
and fixed CPU-rule positions, then emits policy, sensitivity, throughput, and
optional head-to-head evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.game_state import GameState, Phase, Player
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS
from v1.python.portable_device import resolve_portable_device
from v1.python.portable_mcts import PortableMCTS, PortableMCTSConfig, PortableTree
from v1.python.portable_root_puct import (
    PortableRootPUCT,
    allocate_fixed_q_visits,
    policy_from_visits,
)


def _entropy(policy: torch.Tensor) -> float:
    p = policy.to(torch.float64)
    p = p[p > 0]
    return float((-(p * torch.log(p))).sum().item()) if int(p.numel()) else 0.0


def _kl(prior: torch.Tensor, policy: torch.Tensor) -> float:
    p = prior.to(torch.float64)
    q = policy.to(torch.float64)
    valid = p.gt(0)
    if not bool(valid.any().item()):
        return 0.0
    eps = 1e-12
    return float((p[valid] * (torch.log(p[valid] + eps) - torch.log(q[valid] + eps))).sum().item())


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(sum(rows) / len(rows)) if rows else 0.0


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _resource_metrics(*, device: str, elapsed: float, cpu_started: float) -> Dict[str, object]:
    cpu_sec = max(0.0, time.process_time() - cpu_started)
    logical_cpus = max(1, int(os.cpu_count() or 1))
    payload: Dict[str, object] = {
        "process_cpu_sec": cpu_sec,
        "cpu_utilization_percent_of_host": float(
            100.0 * cpu_sec / max(elapsed, 1e-9) / logical_cpus
        ),
        "logical_cpu_count": logical_cpus,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "mps_utilization_percent": None,
        "mps_utilization_status": (
            "not exposed by the installed PyTorch MPS API"
            if device == "mps"
            else "not applicable"
        ),
        "mps_current_allocated_bytes": None,
        "mps_driver_allocated_bytes": None,
    }
    if device == "mps":
        payload["mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        payload["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
    return payload


def _collect_positions(seed: int, count: int) -> List[GameState]:
    rng = random.Random(int(seed))
    candidates: List[GameState] = [GameState()]
    first_by_phase: Dict[Phase, GameState] = {Phase.PLACEMENT: candidates[0]}
    for game_idx in range(16):
        state = GameState()
        for ply in range(GameState.MAX_MOVE_COUNT):
            legal = generate_all_legal_moves(state)
            if not legal or state.is_game_over():
                break
            state = apply_move(state, rng.choice(legal), quiet=True)
            should_keep = state.phase not in first_by_phase or (ply + game_idx * 7) % 17 == 0
            if should_keep and not state.is_game_over():
                snapshot = state.copy()
                candidates.append(snapshot)
                first_by_phase.setdefault(snapshot.phase, snapshot)

    selected: List[GameState] = []
    selected_ids = set()
    for phase in Phase:
        snapshot = first_by_phase.get(phase)
        if snapshot is not None and len(selected) < int(count):
            selected.append(snapshot)
            selected_ids.add(id(snapshot))
    for snapshot in candidates:
        if len(selected) >= int(count):
            break
        if id(snapshot) not in selected_ids:
            selected.append(snapshot)
            selected_ids.add(id(snapshot))
    return selected


def _model_sha256(model: ChessNet) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_model(checkpoint: Optional[str], seed: int) -> Tuple[ChessNet, str, str]:
    torch.manual_seed(int(seed))
    model = ChessNet(board_size=GameState.BOARD_SIZE, num_input_channels=NUM_INPUT_CHANNELS)
    if not checkpoint:
        return model, "random_init", _model_sha256(model)
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    return model, str(checkpoint), _model_sha256(model)


def _parse_checkpoints(values: Sequence[str]) -> List[Tuple[str, Optional[str]]]:
    if not values:
        return [("random", None)]
    parsed: List[Tuple[str, Optional[str]]] = []
    for raw in values:
        if raw.strip().lower() in {"random", "random_init"}:
            parsed.append((raw.strip(), None))
            continue
        if "=" not in raw:
            raise ValueError(
                "--checkpoint must be 'random' or use label=/path/to/model.pt"
            )
        label, path = raw.split("=", 1)
        if not label.strip() or not path.strip():
            raise ValueError("--checkpoint must use non-empty label and path")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        parsed.append((label.strip(), path.strip()))
    return parsed


def _root_metrics(
    model: ChessNet,
    states: Sequence[GameState],
    *,
    device: str,
    simulations: int,
    temperature: float,
    perturbation: float,
    seed: int,
) -> Tuple[Dict[str, object], List[int]]:
    agent = PortableRootPUCT(
        model,
        device=device,
        num_simulations=int(simulations),
        exploration_weight=1.0,
        temperature=float(temperature),
        sample_moves=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(simulations))
    rows = []
    top1: List[int] = []
    started = time.perf_counter()
    cpu_started = time.process_time()
    for state in states:
        out = agent.search(state)
        action = int(out.chosen_action_index) if out.chosen_action_index is not None else -1
        top1.append(action)
        noise = torch.randn(out.q_values.shape, generator=generator) * float(perturbation)
        perturbed_visits, _ = allocate_fixed_q_visits(
            out.priors,
            out.q_values + noise,
            out.legal_mask,
            num_simulations=int(simulations),
            exploration_weight=1.0,
        )
        perturbed_policy = policy_from_visits(
            perturbed_visits, out.legal_mask, float(temperature)
        )
        perturbed_top1 = int(torch.argmax(perturbed_policy).item())
        rows.append(
            {
                "p_to_pi_kl": _kl(out.priors, out.policy_dense),
                "policy_entropy": _entropy(out.policy_dense),
                "effective_policy_actions": math.exp(_entropy(out.policy_dense)),
                "visited_actions": int(out.visits.gt(0).sum().item()),
                "top1": action,
                "perturbed_top1": perturbed_top1,
                "perturb_top1_changed": bool(action != perturbed_top1),
                "perturb_policy_l1": float(
                    torch.abs(out.policy_dense - perturbed_policy).sum().item()
                ),
            }
        )
    elapsed = max(1e-9, time.perf_counter() - started)
    return (
        {
            "simulations": int(simulations),
            "positions": len(rows),
            "elapsed_sec": elapsed,
            "positions_per_sec": float(len(rows) / elapsed),
            "resources": _resource_metrics(
                device=device, elapsed=elapsed, cpu_started=cpu_started
            ),
            "p_to_pi_kl_mean": _mean(row["p_to_pi_kl"] for row in rows),
            "policy_entropy_mean": _mean(row["policy_entropy"] for row in rows),
            "effective_policy_actions_mean": _mean(
                row["effective_policy_actions"] for row in rows
            ),
            "visited_actions_mean": _mean(row["visited_actions"] for row in rows),
            "perturb_top1_change_rate": _mean(
                1.0 if row["perturb_top1_changed"] else 0.0 for row in rows
            ),
            "perturb_policy_l1_mean": _mean(row["perturb_policy_l1"] for row in rows),
            "rows": rows,
        },
        top1,
    )


def _full_metrics(
    model: ChessNet,
    states: Sequence[GameState],
    *,
    device: str,
    simulations: int,
    temperature: float,
) -> Tuple[Dict[str, object], List[int]]:
    search = PortableMCTS(
        model,
        PortableMCTSConfig(
            num_simulations=int(simulations),
            exploration_weight=1.0,
            temperature=float(temperature),
            add_dirichlet_noise=False,
            sample_moves=False,
        ),
        device=device,
    )
    prior_eval = search.evaluate_states(states)
    trees = [PortableTree(state) for state in states]
    started = time.perf_counter()
    cpu_started = time.process_time()
    outputs = search.search_batch(
        trees,
        temperatures=float(temperature),
        add_dirichlet_noise=False,
    )
    elapsed = max(1e-9, time.perf_counter() - started)
    rows = []
    top1: List[int] = []
    for index, out in enumerate(outputs):
        action = int(out.chosen_action_index) if out.chosen_action_index is not None else -1
        top1.append(action)
        entropy = _entropy(out.policy_dense)
        rows.append(
            {
                "p_to_pi_kl": _kl(prior_eval.priors[index], out.policy_dense),
                "policy_entropy": entropy,
                "effective_policy_actions": math.exp(entropy),
                "visited_actions": sum(1 for value in out.visit_counts.values() if value > 0),
                "top1": action,
                "root_value": float(out.root_value),
            }
        )
    return (
        {
            "simulations": int(simulations),
            "positions": len(rows),
            "elapsed_sec": elapsed,
            "positions_per_sec": float(len(rows) / elapsed),
            "resources": _resource_metrics(
                device=device, elapsed=elapsed, cpu_started=cpu_started
            ),
            "p_to_pi_kl_mean": _mean(row["p_to_pi_kl"] for row in rows),
            "policy_entropy_mean": _mean(row["policy_entropy"] for row in rows),
            "effective_policy_actions_mean": _mean(
                row["effective_policy_actions"] for row in rows
            ),
            "visited_actions_mean": _mean(row["visited_actions"] for row in rows),
            "inference_batches": int(search.inference_batches),
            "fallback_count": int(search.device_resolution.fallback_count),
            "fallback_reasons": list(search.device_resolution.fallback_reasons),
            "rows": rows,
        },
        top1,
    )


def _head_to_head(
    model: ChessNet,
    *,
    device: str,
    games: int,
    root_simulations: int,
    full_simulations: int,
    temperature: float,
) -> Dict[str, object]:
    if int(games) <= 0:
        return {"games": 0, "status": "not_requested"}
    full = PortableMCTS(
        model,
        PortableMCTSConfig(
            num_simulations=int(full_simulations),
            temperature=float(temperature),
            add_dirichlet_noise=False,
            sample_moves=False,
        ),
        device=device,
    )
    root = PortableRootPUCT(
        model,
        device=device,
        num_simulations=int(root_simulations),
        temperature=float(temperature),
        sample_moves=False,
    )
    full_wins = root_wins = draws = 0
    elapsed_per_game: List[float] = []
    cpu_started = time.process_time()
    all_started = time.perf_counter()
    for game in range(int(games)):
        state = GameState()
        full_black = game < (int(games) / 2.0)
        started = time.perf_counter()
        while not state.is_game_over():
            legal = generate_all_legal_moves(state)
            if not legal:
                break
            full_to_move = (state.current_player == Player.BLACK) == full_black
            if full_to_move:
                move = full.search(PortableTree(state), add_dirichlet_noise=False).chosen_move
            else:
                move = root.search(state).chosen_move
            if move is None:
                break
            state = apply_move(state, move, quiet=True)
        elapsed_per_game.append(time.perf_counter() - started)
        winner = state.get_winner()
        if winner is None:
            draws += 1
        elif (winner == Player.BLACK) == full_black:
            full_wins += 1
        else:
            root_wins += 1
    elapsed = max(1e-9, time.perf_counter() - all_started)
    return {
        "games": int(games),
        "portable_full_wins": full_wins,
        "root_only_wins": root_wins,
        "draws": draws,
        "seconds_per_game_mean": _mean(elapsed_per_game),
        "resources": _resource_metrics(
            device=device, elapsed=elapsed, cpu_started=cpu_started
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--root_simulations", default="128,512,1536,65536")
    parser.add_argument("--full_simulations", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--q_perturbation", type=float, default=1e-3)
    parser.add_argument("--head_to_head_games", type=int, default=0)
    parser.add_argument("--head_to_head_root_simulations", type=int, default=128)
    parser.add_argument("--output_json", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    resolution = resolve_portable_device(args.device)
    device = str(resolution.device)
    checkpoints = _parse_checkpoints(args.checkpoint)
    simulations = [int(item.strip()) for item in str(args.root_simulations).split(",") if item.strip()]
    states = _collect_positions(int(args.seed), max(1, int(args.positions)))
    phase_counts = Counter(state.phase.name for state in states)
    report: Dict[str, object] = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "requested_device": str(args.device),
            "device": device,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "fallback_count": int(resolution.fallback_count),
            "fallback_reasons": list(resolution.fallback_reasons),
            "mps_driver_allocated_bytes_before": (
                int(torch.mps.driver_allocated_memory()) if device == "mps" else None
            ),
        },
        "conditions": {
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "q_perturbation": float(args.q_perturbation),
            "root_simulations": simulations,
            "full_simulations": int(args.full_simulations),
            "head_to_head_games": int(args.head_to_head_games),
            "head_to_head_root_simulations": int(args.head_to_head_root_simulations),
            "positions": len(states),
            "phase_counts": dict(phase_counts),
        },
        "method": {
            "root_only": (
                "portable semantic reproduction of current one-step child evaluation "
                "and fixed-q visit allocation; CUDA fused-kernel throughput is not measured"
            ),
            "portable_full_mcts": "CPU rules/tree with batched PyTorch inference",
            "checkpoint_labels": [label for label, _path in checkpoints],
            "checkpoint_coverage_limit": (
                "early/mid/strong project checkpoints were not available in this checkout"
            ),
        },
        "models": [],
        "tactical_accuracy": {
            "value": None,
            "status": "not_measured",
            "reason": "repository contains no labeled tactical position suite",
        },
    }
    for label, checkpoint in checkpoints:
        model, source, model_sha256 = _load_model(checkpoint, int(args.seed))
        root_rows = []
        top_by_sim: Dict[int, List[int]] = {}
        for sims in simulations:
            metrics, top1 = _root_metrics(
                model,
                states,
                device=device,
                simulations=sims,
                temperature=float(args.temperature),
                perturbation=float(args.q_perturbation),
                seed=int(args.seed),
            )
            root_rows.append(metrics)
            top_by_sim[sims] = top1
        full_metrics, full_top1 = _full_metrics(
            model,
            states,
            device=device,
            simulations=int(args.full_simulations),
            temperature=float(args.temperature),
        )
        max_root_sims = max(simulations)
        agreement = _mean(
            1.0 if left == right else 0.0
            for left, right in zip(top_by_sim[max_root_sims], full_top1)
        )
        stability_reference = top_by_sim[max_root_sims]
        for row in root_rows:
            row["top1_stability_vs_max"] = _mean(
                1.0 if left == right else 0.0
                for left, right in zip(top_by_sim[int(row["simulations"])], stability_reference)
            )
        h2h = _head_to_head(
            model,
            device=device,
            games=int(args.head_to_head_games),
            root_simulations=max(1, int(args.head_to_head_root_simulations)),
            full_simulations=int(args.full_simulations),
            temperature=float(args.temperature),
        )
        report["models"].append(
            {
                "label": label,
                "source": source,
                "model_sha256": model_sha256,
                "root_only": root_rows,
                "portable_full_mcts": full_metrics,
                "top1_agreement_max_root_vs_full": agreement,
                "head_to_head": h2h,
            }
        )
    report["environment"]["mps_driver_allocated_bytes_after"] = (
        int(torch.mps.driver_allocated_memory()) if device == "mps" else None
    )
    report["search_quality_conclusion"] = {
        "status": "inconclusive",
        "claim": "no search-quality or training-effect improvement is claimed",
        "reason": (
            "No labeled tactical suite or early/mid/strong project checkpoints were available; "
            "throughput and policy-shape evidence alone do not establish playing strength."
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[ab_portable_search] report saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
