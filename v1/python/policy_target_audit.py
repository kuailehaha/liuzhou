"""Additive health metrics for portable policy targets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

import torch

from .portable_mcts import PortableSearchOutput


def _empty_bucket() -> Dict[str, float | int]:
    return {
        "positions": 0,
        "legal_actions_sum": 0,
        "positive_support_sum": 0,
        "visit_support_sum": 0,
        "entropy_sum": 0.0,
        "effective_support_sum": 0.0,
        "unvisited_prior_mass_sum": 0.0,
        "one_hot_count": 0,
    }


def _bucket_name(ply: int) -> str:
    if int(ply) < 10:
        return "ply_0_9"
    if int(ply) < 30:
        return "ply_10_29"
    return "ply_30_plus"


def _finalize_bucket(raw: Mapping[str, Any]) -> Dict[str, float | int]:
    positions = int(raw.get("positions", 0) or 0)
    denominator = max(1, positions)
    return {
        "positions": positions,
        "mean_legal_actions": float(
            int(raw.get("legal_actions_sum", 0) or 0) / denominator
        ),
        "mean_positive_support": float(
            int(raw.get("positive_support_sum", 0) or 0) / denominator
        ),
        "mean_visit_support": float(
            int(raw.get("visit_support_sum", 0) or 0) / denominator
        ),
        "mean_entropy": float(
            float(raw.get("entropy_sum", 0.0) or 0.0) / denominator
        ),
        "mean_effective_support": float(
            float(raw.get("effective_support_sum", 0.0) or 0.0) / denominator
        ),
        "mean_unvisited_prior_mass": float(
            float(raw.get("unvisited_prior_mass_sum", 0.0) or 0.0)
            / denominator
        ),
        "one_hot_count": int(raw.get("one_hot_count", 0) or 0),
        "one_hot_ratio": float(
            int(raw.get("one_hot_count", 0) or 0) / denominator
        ),
    }


@dataclass
class PolicyTargetAudit:
    totals: Dict[str, float | int] = field(default_factory=_empty_bucket)
    buckets: Dict[str, Dict[str, float | int]] = field(
        default_factory=lambda: {
            "ply_0_9": _empty_bucket(),
            "ply_10_29": _empty_bucket(),
            "ply_30_plus": _empty_bucket(),
        }
    )

    @staticmethod
    def _observe_bucket(
        bucket: Dict[str, float | int],
        output: PortableSearchOutput,
    ) -> None:
        legal_mask = output.legal_mask.to(torch.bool)
        legal_policy = output.policy_dense[legal_mask].to(torch.float32)
        legal_priors = output.root_priors[legal_mask].to(torch.float32)
        legal_indices = torch.where(legal_mask)[0].tolist()
        visits = torch.tensor(
            [int(output.visit_counts.get(int(index), 0)) for index in legal_indices],
            dtype=torch.int64,
        )
        if int(legal_policy.numel()) <= 0:
            return
        if not bool(torch.isfinite(legal_policy).all().item()):
            raise ValueError("Policy target audit received NaN/Inf target.")
        if not bool(torch.isfinite(legal_priors).all().item()):
            raise ValueError("Policy target audit received NaN/Inf priors.")

        positive_support = int(torch.count_nonzero(legal_policy > 0).item())
        visit_support = int(torch.count_nonzero(visits > 0).item())
        entropy = float(
            -(
                legal_policy
                * torch.log(legal_policy.clamp_min(1e-12))
            ).sum().item()
        )
        effective_support = float(math.exp(entropy))
        unvisited_prior_mass = float(legal_priors[visits.eq(0)].sum().item())

        bucket["positions"] = int(bucket["positions"]) + 1
        bucket["legal_actions_sum"] = int(bucket["legal_actions_sum"]) + int(
            legal_policy.numel()
        )
        bucket["positive_support_sum"] = int(
            bucket["positive_support_sum"]
        ) + positive_support
        bucket["visit_support_sum"] = int(bucket["visit_support_sum"]) + visit_support
        bucket["entropy_sum"] = float(bucket["entropy_sum"]) + entropy
        bucket["effective_support_sum"] = float(
            bucket["effective_support_sum"]
        ) + effective_support
        bucket["unvisited_prior_mass_sum"] = float(
            bucket["unvisited_prior_mass_sum"]
        ) + unvisited_prior_mass
        if positive_support <= 1:
            bucket["one_hot_count"] = int(bucket["one_hot_count"]) + 1

    def observe(self, output: PortableSearchOutput, *, ply: int) -> None:
        self._observe_bucket(self.totals, output)
        self._observe_bucket(self.buckets[_bucket_name(ply)], output)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **_finalize_bucket(self.totals),
            "ply_buckets": {
                name: _finalize_bucket(bucket)
                for name, bucket in self.buckets.items()
            },
            "_raw": {
                "totals": dict(self.totals),
                "buckets": {
                    name: dict(bucket) for name, bucket in self.buckets.items()
                },
            },
        }


def merge_policy_target_audits(
    audits: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = PolicyTargetAudit()
    for audit in audits:
        raw = audit.get("_raw") if isinstance(audit, Mapping) else None
        if not isinstance(raw, Mapping):
            continue
        totals = raw.get("totals")
        if isinstance(totals, Mapping):
            _merge_raw_bucket(merged.totals, totals)
        buckets = raw.get("buckets")
        if isinstance(buckets, Mapping):
            for name, bucket in buckets.items():
                if name in merged.buckets and isinstance(bucket, Mapping):
                    _merge_raw_bucket(merged.buckets[name], bucket)
    return merged.to_dict()


def _merge_raw_bucket(
    destination: Dict[str, float | int],
    source: Mapping[str, Any],
) -> None:
    for key in (
        "positions",
        "legal_actions_sum",
        "positive_support_sum",
        "visit_support_sum",
        "one_hot_count",
    ):
        destination[key] = int(destination[key]) + int(source.get(key, 0) or 0)
    for key in (
        "entropy_sum",
        "effective_support_sum",
        "unvisited_prior_mass_sum",
    ):
        destination[key] = float(destination[key]) + float(
            source.get(key, 0.0) or 0.0
        )
