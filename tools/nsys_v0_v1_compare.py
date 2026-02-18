#!/usr/bin/env python
"""Profile v0/v1 workloads with Nsight Systems and generate timeline/fragmentation summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _parse_number(text: str) -> float:
    value = str(text).strip().replace(",", "")
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _unit_scale_from_header(header: str) -> float:
    h = str(header).lower()
    if "(ns" in h:
        return 1.0
    if "(us" in h:
        return 1_000.0
    if "(ms" in h:
        return 1_000_000.0
    if "(s" in h:
        return 1_000_000_000.0
    return 1.0


def _find_col(columns: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in columns}
    for pat in patterns:
        p = pat.lower()
        for lc, raw in lower_cols.items():
            if p in lc:
                return raw
    return None


@dataclass
class TraceOp:
    start_ns: float
    end_ns: float
    kind: str
    name: str


def _classify_gpu_op(name: str) -> str:
    n = str(name).strip().lower()
    if "memcpy" in n:
        return "memcpy"
    if "memset" in n:
        return "memset"
    return "kernel"


def _parse_gputrace_csv(path: Path) -> Tuple[List[TraceOp], Dict[str, float]]:
    ops: List[TraceOp] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        start_col = _find_col(cols, ["start"])
        dur_col = _find_col(cols, ["duration"])
        name_col = _find_col(cols, ["name", "kernel", "operation"])
        if not start_col or not dur_col or not name_col:
            return [], {
                "kernel_count": 0.0,
                "kernel_total_ms": 0.0,
                "kernel_avg_us": 0.0,
                "kernel_median_us": 0.0,
                "kernels_per_ms": 0.0,
                "kernel_gap_mean_us": 0.0,
                "kernel_gap_p95_us": 0.0,
                "kernel_gap_total_ms": 0.0,
                "kernel_idle_ratio": 0.0,
                "memcpy_count": 0.0,
                "memcpy_total_ms": 0.0,
                "memset_count": 0.0,
                "memset_total_ms": 0.0,
            }

        start_scale = _unit_scale_from_header(start_col)
        dur_scale = _unit_scale_from_header(dur_col)

        for row in reader:
            start_ns = _parse_number(row.get(start_col, "0")) * start_scale
            dur_ns = _parse_number(row.get(dur_col, "0")) * dur_scale
            if dur_ns <= 0.0:
                continue
            end_ns = start_ns + dur_ns
            name = str(row.get(name_col, ""))
            kind = _classify_gpu_op(name)
            ops.append(TraceOp(start_ns=float(start_ns), end_ns=float(end_ns), kind=kind, name=name))

    kernel = [op for op in ops if op.kind == "kernel"]
    memcpy = [op for op in ops if op.kind == "memcpy"]
    memset = [op for op in ops if op.kind == "memset"]

    kernel_dur_ns = [max(0.0, op.end_ns - op.start_ns) for op in kernel]
    kernel_total_ms = sum(kernel_dur_ns) / 1_000_000.0
    kernel_avg_us = (statistics.fmean(kernel_dur_ns) / 1_000.0) if kernel_dur_ns else 0.0
    kernel_median_us = (statistics.median(kernel_dur_ns) / 1_000.0) if kernel_dur_ns else 0.0
    kernels_per_ms = (len(kernel) / kernel_total_ms) if kernel_total_ms > 0 else 0.0

    kernel_sorted = sorted(kernel, key=lambda x: x.start_ns)
    gaps_ns: List[float] = []
    prev_end = None
    for op in kernel_sorted:
        if prev_end is not None:
            gaps_ns.append(max(0.0, op.start_ns - prev_end))
        prev_end = max(float(prev_end) if prev_end is not None else 0.0, op.end_ns)
    gap_mean_us = (statistics.fmean(gaps_ns) / 1_000.0) if gaps_ns else 0.0
    if gaps_ns:
        gaps_sorted = sorted(gaps_ns)
        idx = min(len(gaps_sorted) - 1, int(math.ceil(0.95 * len(gaps_sorted))) - 1)
        gap_p95_us = gaps_sorted[idx] / 1_000.0
    else:
        gap_p95_us = 0.0
    gap_total_ms = sum(gaps_ns) / 1_000_000.0

    span_ns = 0.0
    if kernel_sorted:
        span_ns = max(0.0, kernel_sorted[-1].end_ns - kernel_sorted[0].start_ns)
    idle_ratio = (sum(gaps_ns) / span_ns) if span_ns > 0 else 0.0

    metrics = {
        "kernel_count": float(len(kernel)),
        "kernel_total_ms": float(kernel_total_ms),
        "kernel_avg_us": float(kernel_avg_us),
        "kernel_median_us": float(kernel_median_us),
        "kernels_per_ms": float(kernels_per_ms),
        "kernel_gap_mean_us": float(gap_mean_us),
        "kernel_gap_p95_us": float(gap_p95_us),
        "kernel_gap_total_ms": float(gap_total_ms),
        "kernel_idle_ratio": float(idle_ratio),
        "memcpy_count": float(len(memcpy)),
        "memcpy_total_ms": float(sum(max(0.0, op.end_ns - op.start_ns) for op in memcpy) / 1_000_000.0),
        "memset_count": float(len(memset)),
        "memset_total_ms": float(sum(max(0.0, op.end_ns - op.start_ns) for op in memset) / 1_000_000.0),
    }
    return ops, metrics


def _parse_cudaapisum_csv(path: Path) -> Dict[str, float]:
    sync_count = 0.0
    sync_total_ns = 0.0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        name_col = _find_col(cols, ["name"])
        calls_col = _find_col(cols, ["calls"])
        total_col = _find_col(cols, ["total"])
        if not name_col:
            return {"sync_api_calls": 0.0, "sync_api_total_ms": 0.0}
        total_scale = _unit_scale_from_header(total_col) if total_col else 1.0

        for row in reader:
            name = str(row.get(name_col, "")).lower()
            is_sync = (
                "synchronize" in name
                or "streamwait" in name
                or "eventquery" in name
                or "devicewait" in name
            )
            if not is_sync:
                continue
            if calls_col:
                sync_count += _parse_number(row.get(calls_col, "0"))
            else:
                sync_count += 1.0
            if total_col:
                sync_total_ns += _parse_number(row.get(total_col, "0")) * total_scale
    return {
        "sync_api_calls": float(sync_count),
        "sync_api_total_ms": float(sync_total_ns / 1_000_000.0),
    }


def _find_stats_csv(prefix: Path, report: str) -> Path:
    candidates = [
        prefix.with_name(prefix.name + f"_{report}.csv"),
        prefix.with_name(prefix.name + f".{report}.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    globbed = sorted(prefix.parent.glob(prefix.name + f"*{report}*.csv"))
    if not globbed:
        raise FileNotFoundError(f"Could not locate {report} csv for prefix: {prefix}")
    return globbed[0]


def _list_nsys_reports(nsys_bin: str, env: Dict[str, str]) -> List[str]:
    cmd = [nsys_bin, "stats", "--help-reports"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    reports: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("The following built-in reports"):
            continue
        if stripped.startswith("For more information"):
            continue
        if stripped.startswith("usage:"):
            continue
        token = stripped.split()[0]
        if token and token[0].isalpha():
            clean = token.split("[")[0].strip()
            clean = clean.split(":")[0].strip()
            if clean:
                reports.append(clean)
    dedup: List[str] = []
    seen = set()
    for name in reports:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(name)
    return dedup


def _pick_report(available: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {name.lower(): name for name in available}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _plot_timeline(v0_ops: List[TraceOp], v1_ops: List[TraceOp], output_path: Path, max_ops: int) -> None:
    import matplotlib.pyplot as plt

    def _draw(ax, ops: List[TraceOp], title: str) -> None:
        if not ops:
            ax.text(0.5, 0.5, "No GPU ops", ha="center", va="center")
            ax.set_title(title)
            ax.axis("off")
            return
        t0 = ops[0].start_ns
        lanes = {"kernel": (2, "#4C78A8"), "memcpy": (1, "#E45756"), "memset": (0, "#72B7B2")}
        for op in ops:
            lane, color = lanes.get(op.kind, (2, "#4C78A8"))
            start_ms = (op.start_ns - t0) / 1_000_000.0
            dur_ms = max(1e-6, (op.end_ns - op.start_ns) / 1_000_000.0)
            ax.broken_barh([(start_ms, dur_ms)], (lane - 0.38, 0.76), facecolors=color, alpha=0.9)
        ax.set_yticks([0, 1, 2], ["memset", "memcpy", "kernel"])
        ax.set_xlabel("Time (ms)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

    def _prep(ops: List[TraceOp]) -> List[TraceOp]:
        ordered = sorted(ops, key=lambda x: x.start_ns)
        if max_ops > 0:
            ordered = ordered[:max_ops]
        return ordered

    v0_sorted = _prep(v0_ops)
    v1_sorted = _prep(v1_ops)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=120, sharex=False)
    _draw(axes[0], v0_sorted, "v0 Nsight Timeline (GPU ops)")
    _draw(axes[1], v1_sorted, "v1 Nsight Timeline (GPU ops)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_timeline_single(ops: List[TraceOp], output_path: Path, max_ops: int, title: str) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(ops, key=lambda x: x.start_ns)
    if max_ops > 0:
        ordered = ordered[:max_ops]

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=120)
    if not ordered:
        ax.text(0.5, 0.5, "No GPU ops", ha="center", va="center")
        ax.axis("off")
    else:
        t0 = ordered[0].start_ns
        lanes = {"kernel": (2, "#4C78A8"), "memcpy": (1, "#E45756"), "memset": (0, "#72B7B2")}
        for op in ordered:
            lane, color = lanes.get(op.kind, (2, "#4C78A8"))
            start_ms = (op.start_ns - t0) / 1_000_000.0
            dur_ms = max(1e-6, (op.end_ns - op.start_ns) / 1_000_000.0)
            ax.broken_barh([(start_ms, dur_ms)], (lane - 0.38, 0.76), facecolors=color, alpha=0.9)
        ax.set_yticks([0, 1, 2], ["memset", "memcpy", "kernel"])
        ax.set_xlabel("Time (ms)")
        ax.grid(axis="x", alpha=0.25)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_summary(v0_metrics: Dict[str, float], v1_metrics: Dict[str, float], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    keys = ["kernel_count", "memcpy_count", "sync_api_calls", "kernel_gap_mean_us", "kernels_per_ms"]
    labels = ["kernel_count", "memcpy_count", "sync_calls", "kernel_gap_mean_us", "kernels_per_ms"]
    x = list(range(len(keys)))
    v0_vals = [float(v0_metrics.get(k, 0.0)) for k in keys]
    v1_vals = [float(v1_metrics.get(k, 0.0)) for k in keys]

    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=120)
    ax.bar([i - width / 2 for i in x], v0_vals, width=width, label="v0", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], v1_vals, width=width, label="v1", color="#F58518")
    ax.set_xticks(x, labels)
    ax.set_title("Nsight Summary: memcpy/sync/kernel fragmentation")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_summary_single(metrics: Dict[str, float], output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    keys = ["kernel_count", "memcpy_count", "sync_api_calls", "kernel_gap_mean_us", "kernels_per_ms"]
    labels = ["kernel_count", "memcpy_count", "sync_calls", "kernel_gap_mean_us", "kernels_per_ms"]
    vals = [float(metrics.get(k, 0.0)) for k in keys]

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
    ax.bar(range(len(vals)), vals, color="#4C78A8")
    ax.set_xticks(list(range(len(vals))), labels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _profile_mode(
    *,
    nsys_bin: str,
    mode: str,
    output_prefix: Path,
    env: Dict[str, str],
    args: argparse.Namespace,
) -> Path:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    workload_json = output_prefix.with_name(output_prefix.name + "_workload.json")

    workload_cmd = [
        sys.executable,
        "tools/run_selfplay_workload.py",
        "--mode",
        mode,
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--num-games-per-iter",
        str(args.num_games_per_iter),
        "--duration-sec",
        str(args.duration_sec),
        "--mcts-simulations",
        str(args.mcts_simulations),
        "--output-json",
        str(workload_json),
    ]

    if mode == "v0":
        workload_cmd.extend(
            [
                "--v0-workers",
                str(args.v0_workers),
                "--v0-batch-leaves",
                str(args.v0_batch_leaves),
                "--v0-inference-backend",
                str(args.v0_inference_backend),
                "--v0-inference-batch-size",
                str(args.v0_inference_batch_size),
                "--v0-inference-warmup-iters",
                str(args.v0_inference_warmup_iters),
            ]
        )
    else:
        workload_cmd.extend(
            [
                "--v1-threads",
                str(args.v1_threads),
                "--v1-concurrent-games",
                str(args.v1_concurrent_games),
                "--v1-child-eval-mode",
                str(args.v1_child_eval_mode),
                "--v1-inference-backend",
                str(args.v1_inference_backend),
                "--v1-inference-batch-size",
                str(args.v1_inference_batch_size),
                "--v1-inference-warmup-iters",
                str(args.v1_inference_warmup_iters),
            ]
        )

    profile_cmd = [
        nsys_bin,
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-memory-usage=true",
        "-o",
        str(output_prefix),
    ] + workload_cmd

    _run(profile_cmd, env=env)
    rep_path = output_prefix.with_suffix(".nsys-rep")
    if not rep_path.exists():
        raise FileNotFoundError(f"nsys report not found: {rep_path}")
    return rep_path


def _export_stats(nsys_bin: str, rep_path: Path, output_prefix: Path, env: Dict[str, str]) -> Tuple[Path, Path]:
    stats_prefix = output_prefix.with_name(output_prefix.name + "_stats")
    available = _list_nsys_reports(nsys_bin, env)
    gpu_report = _pick_report(available, ["cuda_gpu_trace", "gputrace"])
    api_report = _pick_report(available, ["cuda_api_sum", "cudaapisum"])
    if not gpu_report or not api_report:
        raise RuntimeError(
            f"Unsupported nsys stats reports. available={available[:16]}..., "
            f"required one of gpu=[cuda_gpu_trace|gputrace], api=[cuda_api_sum|cudaapisum]"
        )
    cmd = [
        nsys_bin,
        "stats",
        "--report",
        f"{gpu_report},{api_report}",
        "--format",
        "csv",
        "--output",
        str(stats_prefix),
        str(rep_path),
    ]
    _run(cmd, env=env)
    gputrace_csv = _find_stats_csv(stats_prefix, gpu_report)
    cudaapisum_csv = _find_stats_csv(stats_prefix, api_report)
    return gputrace_csv, cudaapisum_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nsight Systems v0/v1 timeline compare")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--num-games-per-iter", type=int, default=8)
    parser.add_argument("--mcts-simulations", type=int, default=128)

    parser.add_argument("--v0-workers", type=int, default=1)
    parser.add_argument("--v0-batch-leaves", type=int, default=512)
    parser.add_argument("--v0-inference-backend", type=str, default="graph", choices=["graph", "py", "ts"])
    parser.add_argument("--v0-inference-batch-size", type=int, default=512)
    parser.add_argument("--v0-inference-warmup-iters", type=int, default=5)

    parser.add_argument("--v1-threads", type=int, default=1)
    parser.add_argument("--v1-concurrent-games", type=int, default=8)
    parser.add_argument("--v1-child-eval-mode", type=str, default="value_only", choices=["value_only", "full"])
    parser.add_argument("--v1-inference-backend", type=str, default="py", choices=["py", "graph"])
    parser.add_argument("--v1-inference-batch-size", type=int, default=512)
    parser.add_argument("--v1-inference-warmup-iters", type=int, default=5)
    parser.add_argument(
        "--profile-modes",
        type=str,
        default="v0,v1",
        help="Comma-separated modes to profile: v0, v1",
    )

    parser.add_argument("--max-timeline-ops", type=int, default=4000)
    parser.add_argument("--nsys-bin", type=str, default="nsys")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", f"nsys_v0_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    return parser


def _parse_profile_modes(raw: str) -> List[str]:
    allowed = {"v0", "v1"}
    parts = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("profile_modes cannot be empty.")
    out: List[str] = []
    seen = set()
    for p in parts:
        if p not in allowed:
            raise ValueError(f"Unsupported profile mode: {p}")
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def main() -> None:
    args = build_parser().parse_args()
    nsys_bin = shutil.which(str(args.nsys_bin))
    if not nsys_bin:
        raise RuntimeError(f"Nsight Systems binary not found: {args.nsys_bin}")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    py_path_parts = [str(repo_root / "build" / "v0" / "src"), str(repo_root)]
    current_py_path = env.get("PYTHONPATH", "")
    if current_py_path:
        py_path_parts.append(current_py_path)
    env["PYTHONPATH"] = os.pathsep.join(py_path_parts)

    modes = _parse_profile_modes(args.profile_modes)
    reports: Dict[str, str] = {}
    csv_paths: Dict[str, Dict[str, str]] = {}
    ops_by_mode: Dict[str, List[TraceOp]] = {}
    metrics_by_mode: Dict[str, Dict[str, float]] = {}

    for mode in modes:
        prefix = output_dir / f"{mode}_trace"
        rep = _profile_mode(nsys_bin=nsys_bin, mode=mode, output_prefix=prefix, env=env, args=args)
        gputrace_csv, cudaapisum_csv = _export_stats(nsys_bin, rep, prefix, env)
        mode_ops, mode_gpu_metrics = _parse_gputrace_csv(gputrace_csv)
        mode_sync_metrics = _parse_cudaapisum_csv(cudaapisum_csv)
        reports[f"{mode}_nsys_rep"] = str(rep)
        csv_paths[f"{mode}_gputrace_csv"] = {"path": str(gputrace_csv)}
        csv_paths[f"{mode}_cudaapisum_csv"] = {"path": str(cudaapisum_csv)}
        ops_by_mode[mode] = mode_ops
        metrics_by_mode[mode] = {**mode_gpu_metrics, **mode_sync_metrics}

    artifacts: Dict[str, str] = {}
    if "v0" in modes and "v1" in modes:
        timeline_png = output_dir / "nsys_timeline_v0_vs_v1.png"
        summary_png = output_dir / "nsys_summary_v0_vs_v1.png"
        _plot_timeline(
            ops_by_mode.get("v0", []),
            ops_by_mode.get("v1", []),
            timeline_png,
            max_ops=int(args.max_timeline_ops),
        )
        _plot_summary(
            metrics_by_mode.get("v0", {}),
            metrics_by_mode.get("v1", {}),
            summary_png,
        )
        artifacts["timeline_png"] = str(timeline_png)
        artifacts["summary_png"] = str(summary_png)
    elif len(modes) == 1:
        mode = modes[0]
        timeline_png = output_dir / f"nsys_timeline_{mode}.png"
        summary_png = output_dir / f"nsys_summary_{mode}.png"
        _plot_timeline_single(
            ops_by_mode.get(mode, []),
            timeline_png,
            max_ops=int(args.max_timeline_ops),
            title=f"{mode} Nsight Timeline (GPU ops)",
        )
        _plot_summary_single(
            metrics_by_mode.get(mode, {}),
            summary_png,
            title=f"{mode} Nsight Summary: memcpy/sync/kernel fragmentation",
        )
        artifacts["timeline_png"] = str(timeline_png)
        artifacts["summary_png"] = str(summary_png)

    delta = {}
    if "v0" in modes and "v1" in modes:
        v0_metrics = metrics_by_mode.get("v0", {})
        v1_metrics = metrics_by_mode.get("v1", {})
        for key in sorted(set(v0_metrics.keys()) | set(v1_metrics.keys())):
            a = float(v0_metrics.get(key, 0.0))
            b = float(v1_metrics.get(key, 0.0))
            delta[key] = {
                "v0": a,
                "v1": b,
                "delta": b - a,
                "ratio_v1_over_v0": (b / a) if a not in (0.0, -0.0) else float("nan"),
            }

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "profile_modes": modes,
            "device": args.device,
            "seed": int(args.seed),
            "duration_sec": float(args.duration_sec),
            "num_games_per_iter": int(args.num_games_per_iter),
            "mcts_simulations": int(args.mcts_simulations),
            "v0_workers": int(args.v0_workers),
            "v0_batch_leaves": int(args.v0_batch_leaves),
            "v1_threads": int(args.v1_threads),
            "v1_concurrent_games": int(args.v1_concurrent_games),
            "v1_inference_backend": str(args.v1_inference_backend),
        },
        "reports": {**reports, **{k: v["path"] for k, v in csv_paths.items()}},
        "artifacts": artifacts,
        "metrics": {
            **metrics_by_mode,
            "delta": delta,
        },
    }

    out_json = output_dir / "nsys_compare_summary.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"summary_saved={out_json}")
    if "timeline_png" in artifacts:
        print(f"timeline_png={artifacts['timeline_png']}")
    if "summary_png" in artifacts:
        print(f"summary_png={artifacts['summary_png']}")


if __name__ == "__main__":
    main()
