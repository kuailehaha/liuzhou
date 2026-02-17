#!/usr/bin/env python3
"""Benchmark fused root_puct op against reference ATen loop."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import v0_core


def _reference_root_puct(
    priors: torch.Tensor,
    leaf_values: torch.Tensor,
    valid_mask: torch.Tensor,
    num_simulations: int,
    exploration_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    priors_f = priors.to(torch.float32)
    leaf_f = leaf_values.to(torch.float32)
    mask_b = valid_mask.to(torch.bool)

    visits = torch.zeros_like(priors_f)
    value_sum = torch.zeros_like(priors_f)
    total_visit = torch.zeros((priors_f.size(0),), dtype=torch.float32, device=priors_f.device)
    neg_inf = float("-inf")

    for _ in range(int(num_simulations)):
        q = torch.where(
            visits > 0,
            value_sum / visits.clamp_min(1e-8),
            torch.zeros_like(value_sum),
        )
        u = float(exploration_weight) * priors_f * torch.sqrt(total_visit + 1.0).unsqueeze(1) / (1.0 + visits)
        scores = (q + u).masked_fill(~mask_b, neg_inf)
        selected = torch.argmax(scores, dim=1)
        selected_col = selected.unsqueeze(1)
        one = torch.ones((selected_col.size(0), 1), dtype=torch.float32, device=priors_f.device)
        visits = visits.scatter_add(1, selected_col, one)
        selected_leaf = leaf_f.gather(1, selected_col)
        value_sum = value_sum.scatter_add(1, selected_col, selected_leaf)
        total_visit = total_visit + 1.0

    root_values = value_sum.sum(1) / visits.sum(1).clamp_min(1.0)
    return visits, value_sum, root_values


def _time_one(fn, *, iters: int, sync: bool) -> list[float]:
    out: list[float] = []
    for _ in range(int(iters)):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _random_case(
    *,
    roots: int,
    actions: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.rand((roots, actions), generator=gen, device=device) > 0.25
    mask[:, 0] = True
    priors = torch.rand((roots, actions), generator=gen, device=device, dtype=torch.float32)
    priors = priors * mask.to(torch.float32)
    priors = priors / priors.sum(dim=1, keepdim=True).clamp_min(1e-8)
    leaf = torch.randn((roots, actions), generator=gen, device=device, dtype=torch.float32)
    return priors, leaf, mask


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark root_puct fused CUDA op.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--roots", type=int, default=16)
    ap.add_argument("--actions", type=int, default=220)
    ap.add_argument("--simulations", type=str, default="128,256")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--output-json", type=str, default="")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type != "cuda":
        raise ValueError("This benchmark is intended for CUDA device.")
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(args.seed))

    sims_list = [int(s.strip()) for s in str(args.simulations).split(",") if s.strip()]
    report: dict[str, object] = {
        "device": str(dev),
        "seed": int(args.seed),
        "roots": int(args.roots),
        "actions": int(args.actions),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "cases": [],
    }

    for sims in sims_list:
        priors, leaf, mask = _random_case(
            roots=int(args.roots),
            actions=int(args.actions),
            device=dev,
            gen=gen,
        )

        for _ in range(int(args.warmup)):
            _ = v0_core.root_puct_allocate_visits(priors, leaf, mask, int(sims), 1.25)
            _ = _reference_root_puct(priors, leaf, mask, int(sims), 1.25)

        fused_ms = _time_one(
            lambda: v0_core.root_puct_allocate_visits(priors, leaf, mask, int(sims), 1.25),
            iters=int(args.iters),
            sync=True,
        )
        ref_ms = _time_one(
            lambda: _reference_root_puct(priors, leaf, mask, int(sims), 1.25),
            iters=int(args.iters),
            sync=True,
        )

        fused_visits, fused_value_sum, fused_root = v0_core.root_puct_allocate_visits(
            priors, leaf, mask, int(sims), 1.25
        )
        ref_visits, ref_value_sum, ref_root = _reference_root_puct(
            priors, leaf, mask, int(sims), 1.25
        )

        visits_max_abs_diff = float((fused_visits - ref_visits).abs().max().item())
        value_sum_max_abs_diff = float((fused_value_sum - ref_value_sum).abs().max().item())
        root_max_abs_diff = float((fused_root - ref_root).abs().max().item())
        root_mean_abs_diff = float((fused_root - ref_root).abs().mean().item())
        speedup = float(statistics.median(ref_ms) / max(statistics.median(fused_ms), 1e-9))
        wall_drop_ratio = float(1.0 - (statistics.median(fused_ms) / max(statistics.median(ref_ms), 1e-9)))

        row = {
            "simulations": int(sims),
            "fused_ms_median": float(statistics.median(fused_ms)),
            "fused_ms_p95": float(statistics.quantiles(fused_ms, n=20)[18]) if len(fused_ms) >= 20 else float(max(fused_ms)),
            "reference_ms_median": float(statistics.median(ref_ms)),
            "reference_ms_p95": float(statistics.quantiles(ref_ms, n=20)[18]) if len(ref_ms) >= 20 else float(max(ref_ms)),
            "speedup_vs_reference": speedup,
            "wall_drop_ratio_vs_reference": wall_drop_ratio,
            "visits_max_abs_diff": visits_max_abs_diff,
            "value_sum_max_abs_diff": value_sum_max_abs_diff,
            "root_value_max_abs_diff": root_max_abs_diff,
            "root_value_mean_abs_diff": root_mean_abs_diff,
        }
        report["cases"].append(row)
        print(
            f"sims={sims} fused_median={row['fused_ms_median']:.3f}ms "
            f"ref_median={row['reference_ms_median']:.3f}ms "
            f"speedup={row['speedup_vs_reference']:.2f}x "
            f"drop={row['wall_drop_ratio_vs_reference']*100.0:.1f}% "
            f"root_diff(max/mean)=({row['root_value_max_abs_diff']:.3g}/{row['root_value_mean_abs_diff']:.3g})"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report_saved={out_path.as_posix()}")


if __name__ == "__main__":
    main()
