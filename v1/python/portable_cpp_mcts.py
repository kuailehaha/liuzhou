"""PyTorch/MPS inference bridge for the threaded portable C++ tree search."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.game_state import GameState
from src.neural_network import bucket_logits_to_scalar
from src.policy_batch import TOTAL_DIM, build_combined_logits

from .portable_cpp_loader import load_portable_cpp
from .portable_device import PortableDeviceResolution, resolve_portable_device
from .portable_mcts import (
    PortableMCTSConfig,
    PortableSearchOutput,
    policy_from_visits_and_priors,
)


class PortableCppMCTS:
    """Own C++ trees while keeping all model execution in ordinary PyTorch."""

    def __init__(
        self,
        model,
        config: PortableMCTSConfig,
        device: str | torch.device = "cpu",
        *,
        num_threads: int = 1,
        initial_states: Optional[Sequence[GameState]] = None,
    ) -> None:
        if int(num_threads) <= 0:
            raise ValueError("num_threads must be positive")
        states = list(initial_states) if initial_states is not None else [GameState()]
        if not states:
            raise ValueError("initial_states must not be empty")
        self.config = config
        self.device_resolution: PortableDeviceResolution = resolve_portable_device(device)
        self.device = self.device_resolution.device
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.cpp = load_portable_cpp(required=True)
        self.trees = self.cpp.PortableTreeBatch(
            states,
            exploration_weight=float(config.exploration_weight),
            num_threads=int(num_threads),
        )
        self.inference_batches = 0
        self.timing_seconds: Dict[str, float] = {}
        self.timing_calls: Dict[str, int] = {}
        self.last_current_players = [int(state.current_player.value) for state in states]

    @property
    def num_trees(self) -> int:
        return int(self.trees.num_trees)

    @property
    def num_threads(self) -> int:
        return int(self.trees.num_threads)

    @property
    def illegal_action_count(self) -> int:
        return int(self.trees.illegal_action_count)

    @property
    def non_finite_count(self) -> int:
        return int(self.trees.non_finite_count)

    def record_timing(self, name: str, elapsed_sec: float) -> None:
        key = str(name)
        self.timing_seconds[key] = self.timing_seconds.get(key, 0.0) + max(
            0.0, float(elapsed_sec)
        )
        self.timing_calls[key] = self.timing_calls.get(key, 0) + 1

    def timing_snapshot(
        self, total_elapsed_sec: float
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        denominator = max(1e-9, float(total_elapsed_sec))
        timing_ms = {
            key: float(value * 1000.0) for key, value in self.timing_seconds.items()
        }
        timing_ratio = {
            key: min(1.0, max(0.0, float(value / denominator)))
            for key, value in self.timing_seconds.items()
        }
        return timing_ms, timing_ratio, dict(self.timing_calls)

    def _evaluate_pending(
        self, pending: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        model_inputs = torch.from_numpy(pending["model_inputs"]).to(torch.float32)
        legal_masks = torch.from_numpy(pending["legal_masks"]).to(torch.bool)
        count = int(model_inputs.size(0))
        if tuple(model_inputs.shape) != (count, 11, 6, 6):
            raise ValueError(
                f"Unexpected C++ model input shape: {tuple(model_inputs.shape)}"
            )
        if tuple(legal_masks.shape) != (count, TOTAL_DIM):
            raise ValueError(
                f"Unexpected C++ legal mask shape: {tuple(legal_masks.shape)}"
            )
        if count == 0:
            return (
                np.empty((0, TOTAL_DIM), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )

        started = time.perf_counter()
        device_inputs = model_inputs.to(self.device, dtype=torch.float32)
        device_masks = legal_masks.to(self.device)
        try:
            with torch.inference_mode():
                log_p1, log_p2, log_pmc, raw_values = self.model(device_inputs)
                combined = build_combined_logits(
                    log_p1.view(count, -1),
                    log_p2.view(count, -1),
                    log_pmc.view(count, -1),
                    board_size=GameState.BOARD_SIZE,
                ).to(torch.float32)
                masked = combined.masked_fill(~device_masks, float("-inf"))
                priors_device = torch.zeros_like(masked, dtype=torch.float32)
                valid_rows = device_masks.any(dim=1)
                if bool(valid_rows.any().item()):
                    priors_device[valid_rows] = torch.softmax(
                        masked[valid_rows], dim=1
                    )
                if raw_values.dim() == 2 and int(raw_values.size(1)) == 1:
                    values_device = raw_values[:, 0].to(torch.float32)
                else:
                    values_device = bucket_logits_to_scalar(
                        raw_values, num_bins=int(raw_values.size(1))
                    ).to(torch.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Portable C++ inference failed on device={self.device}; "
                "no device fallback was attempted."
            ) from exc
        self.inference_batches += 1
        priors = priors_device.to("cpu")
        values = values_device.view(-1).to("cpu")
        self.record_timing("device_inference", time.perf_counter() - started)

        started = time.perf_counter()
        if not bool(torch.isfinite(priors).all().item()):
            raise ValueError("Portable C++ policy inference produced NaN/Inf.")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("Portable C++ value inference produced NaN/Inf.")
        legal_prob_sum = (priors * legal_masks.to(torch.float32)).sum(dim=1)
        has_legal = legal_masks.any(dim=1)
        if bool(has_legal.any().item()) and not torch.allclose(
            legal_prob_sum[has_legal],
            torch.ones_like(legal_prob_sum[has_legal]),
            atol=1e-5,
            rtol=0.0,
        ):
            raise ValueError(
                "Portable C++ policy is not normalized over legal actions."
            )
        self.record_timing("eval_validate", time.perf_counter() - started)
        return (
            priors.contiguous().numpy(),
            values.contiguous().numpy(),
        )

    def _complete(self, pending: dict) -> None:
        priors, values = self._evaluate_pending(pending)
        started = time.perf_counter()
        self.trees.complete_pending(priors, values)
        self.record_timing(
            "cpp_tree_expand_backup", time.perf_counter() - started
        )

    def _apply_root_noise(self) -> None:
        root_batch = self.trees.root_priors()
        priors = torch.from_numpy(root_batch["priors"]).to(torch.float32)
        legal_masks = torch.from_numpy(root_batch["legal_masks"]).to(torch.bool)
        active = torch.from_numpy(root_batch["active"]).to(torch.bool)
        alpha = max(float(self.config.dirichlet_alpha), 1e-8)
        epsilon = min(max(float(self.config.dirichlet_epsilon), 0.0), 1.0)
        for row in range(int(priors.size(0))):
            indices = torch.where(legal_masks[row])[0]
            if not bool(active[row].item()) or int(indices.numel()) <= 1:
                continue
            noise = torch.distributions.Dirichlet(
                torch.full((int(indices.numel()),), alpha, dtype=torch.float32)
            ).sample()
            mixed = ((1.0 - epsilon) * priors[row, indices]) + (
                epsilon * noise
            )
            mixed.div_(mixed.sum().clamp_min(1e-8))
            priors[row, indices] = mixed
        self.trees.set_root_priors(priors.contiguous().numpy())

    @staticmethod
    def _normalize_temperatures(
        temperatures: Optional[float | Sequence[float]],
        count: int,
        default: float,
    ) -> List[float]:
        if temperatures is None:
            return [float(default)] * count
        if isinstance(temperatures, (float, int)):
            return [float(temperatures)] * count
        values = [float(value) for value in temperatures]
        if len(values) != count:
            raise ValueError(f"Expected {count} temperatures, got {len(values)}.")
        return values

    @staticmethod
    def _policy_from_visits(
        visits: torch.Tensor,
        legal_mask: torch.Tensor,
        temperature: float,
        priors: Optional[torch.Tensor] = None,
        prior_pseudocount: float = 0.0,
    ) -> torch.Tensor:
        policy = torch.zeros((TOTAL_DIM,), dtype=torch.float32)
        indices = torch.where(legal_mask)[0]
        if int(indices.numel()) == 0:
            return policy
        legal_visits = visits[indices].to(torch.float32)
        legal_priors = (
            torch.zeros_like(legal_visits)
            if priors is None
            else priors[indices].to(torch.float32)
        )
        probabilities = policy_from_visits_and_priors(
            legal_visits,
            legal_priors,
            temperature=float(temperature),
            prior_pseudocount=float(prior_pseudocount),
        )
        policy[indices] = probabilities
        return policy

    def search_batch(
        self,
        *,
        temperatures: Optional[float | Sequence[float]] = None,
        add_dirichlet_noise: Optional[bool] = None,
        force_uniform_random: Optional[Sequence[bool]] = None,
    ) -> List[PortableSearchOutput]:
        add_noise = (
            bool(self.config.add_dirichlet_noise)
            if add_dirichlet_noise is None
            else bool(add_dirichlet_noise)
        )
        temperature_values = self._normalize_temperatures(
            temperatures, self.num_trees, float(self.config.temperature)
        )
        force_uniform = (
            [False] * self.num_trees
            if force_uniform_random is None
            else [bool(value) for value in force_uniform_random]
        )
        if len(force_uniform) != self.num_trees:
            raise ValueError(
                f"Expected {self.num_trees} force-uniform flags, "
                f"got {len(force_uniform)}."
            )

        started = time.perf_counter()
        pending = self.trees.prepare_roots()
        self.record_timing("cpp_prepare_roots", time.perf_counter() - started)
        self._complete(pending)
        if add_noise:
            started = time.perf_counter()
            self._apply_root_noise()
            self.record_timing("root_noise", time.perf_counter() - started)

        for _simulation in range(max(1, int(self.config.num_simulations))):
            started = time.perf_counter()
            pending = self.trees.select_leaves()
            self.record_timing("cpp_tree_select", time.perf_counter() - started)
            self._complete(pending)

        started = time.perf_counter()
        raw = self.trees.root_outputs()
        root_prior_batch = self.trees.root_priors()
        self.record_timing("cpp_root_output", time.perf_counter() - started)
        model_inputs = torch.from_numpy(raw["model_inputs"]).to(torch.float32)
        legal_masks = torch.from_numpy(raw["legal_masks"]).to(torch.bool)
        visit_matrix = torch.from_numpy(raw["visit_counts"]).to(torch.int32)
        root_action_values = torch.from_numpy(
            raw["root_action_values"]
        ).to(torch.float32)
        root_values = torch.from_numpy(raw["root_values"]).to(torch.float32)
        terminals = torch.from_numpy(raw["terminal"]).to(torch.bool)
        active = torch.from_numpy(raw["active"]).to(torch.bool)
        current_players = torch.from_numpy(raw["current_players"]).to(torch.int32)
        root_priors = torch.from_numpy(root_prior_batch["priors"]).to(torch.float32)
        self.last_current_players = [
            int(value) for value in current_players.tolist()
        ]

        outputs: List[PortableSearchOutput] = []
        selection_started = time.perf_counter()
        for row in range(self.num_trees):
            legal_mask = legal_masks[row].clone()
            visit_counts = {
                int(index): int(visit_matrix[row, index].item())
                for index in torch.where(legal_mask)[0].tolist()
            }
            terminal = bool(terminals[row].item()) or not bool(active[row].item())
            if terminal or not visit_counts:
                outputs.append(
                    PortableSearchOutput(
                        model_input=model_inputs[row].clone(),
                        legal_mask=legal_mask,
                        policy_dense=torch.zeros(
                            (TOTAL_DIM,), dtype=torch.float32
                        ),
                        selection_policy_dense=torch.zeros(
                            (TOTAL_DIM,), dtype=torch.float32
                        ),
                        root_priors=root_priors[row].clone(),
                        root_action_values=root_action_values[row].clone(),
                        root_value=float(root_values[row].item()),
                        terminal=True,
                        chosen_action_index=None,
                        chosen_move=None,
                        visit_counts=visit_counts,
                    )
                )
                continue
            selection_policy = self._policy_from_visits(
                visit_matrix[row], legal_mask, temperature_values[row]
            )
            target_temperature = self.config.policy_target_temperature
            if target_temperature is None:
                target_temperature = temperature_values[row]
            policy = self._policy_from_visits(
                visit_matrix[row],
                legal_mask,
                float(target_temperature),
                priors=root_priors[row],
                prior_pseudocount=float(
                    self.config.policy_target_prior_pseudocount
                ),
            )
            policy_sum = float(policy.sum().item())
            if not math.isfinite(policy_sum) or abs(policy_sum - 1.0) > 1e-5:
                raise RuntimeError(
                    f"Invalid portable C++ root policy sum: {policy_sum}"
                )
            if int(policy[~legal_mask].count_nonzero().item()) != 0:
                raise RuntimeError(
                    "Portable C++ root policy assigned mass to an illegal action."
                )
            if force_uniform[row]:
                legal_indices = torch.where(legal_mask)[0]
                local = int(
                    torch.randint(int(legal_indices.numel()), (1,)).item()
                )
                chosen_index = int(legal_indices[local].item())
            elif self.config.sample_moves:
                chosen_index = int(
                    torch.multinomial(selection_policy, num_samples=1).item()
                )
            else:
                chosen_index = int(torch.argmax(selection_policy).item())
            outputs.append(
                PortableSearchOutput(
                    model_input=model_inputs[row].clone(),
                    legal_mask=legal_mask,
                    policy_dense=policy,
                    selection_policy_dense=selection_policy,
                    root_priors=root_priors[row].clone(),
                    root_action_values=root_action_values[row].clone(),
                    root_value=float(root_values[row].item()),
                    terminal=False,
                    chosen_action_index=chosen_index,
                    chosen_move=None,
                    visit_counts=visit_counts,
                )
            )
        self.record_timing("policy_select", time.perf_counter() - selection_started)
        return outputs

    def advance_roots(self, actions: Sequence[int]) -> None:
        started = time.perf_counter()
        self.trees.advance_roots([int(action) for action in actions])
        self.record_timing("cpp_advance_roots", time.perf_counter() - started)

    def deactivate(self, tree_indices: Sequence[int]) -> None:
        self.trees.deactivate([int(index) for index in tree_indices])

    def root_status(self) -> dict:
        return self.trees.root_status()
