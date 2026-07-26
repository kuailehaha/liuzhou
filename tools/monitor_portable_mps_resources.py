#!/usr/bin/env python
"""Audit CPU, memory and Apple GPU utilization for a process tree."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

STOP = False


def _stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def _process_tree(root_pid: int) -> tuple[float, int, list[int]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,%cpu=,rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: dict[int, tuple[int, float, int]] = {}
    children: dict[int, list[int]] = {}
    for raw in result.stdout.splitlines():
        fields = raw.split()
        if len(fields) != 4:
            continue
        pid, ppid = int(fields[0]), int(fields[1])
        rows[pid] = (ppid, float(fields[2]), int(fields[3]))
        children.setdefault(ppid, []).append(pid)
    pending = [int(root_pid)]
    descendants: list[int] = []
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.append(pid)
        pending.extend(children.get(pid, []))
    active = [pid for pid in descendants if pid in rows]
    cpu = sum(rows[pid][1] for pid in active)
    rss_kib = sum(rows[pid][2] for pid in active)
    return cpu, rss_kib, active


def _gpu_utilization() -> float | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', result.stdout)
    return float(match.group(1)) if match else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((len(ordered) - 1) * float(fraction)))),
    )
    return float(ordered[index])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    prefix = Path(args.output_prefix).expanduser().resolve()
    jsonl = prefix.with_suffix(".jsonl")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    cpu_values: list[float] = []
    rss_values: list[float] = []
    gpu_values: list[float] = []
    started = time.time()
    with jsonl.open("a", encoding="utf-8") as output:
        while not STOP:
            try:
                cpu, rss_kib, pids = _process_tree(int(args.root_pid))
            except (OSError, subprocess.SubprocessError):
                cpu, rss_kib, pids = 0.0, 0, []
            if not pids:
                break
            gpu = _gpu_utilization()
            row: dict[str, Any] = {
                "epoch": time.time(),
                "root_pid": int(args.root_pid),
                "process_count": len(pids),
                "pids": pids,
                "process_tree_cpu_percent": float(cpu),
                "process_tree_rss_mib": float(rss_kib) / 1024.0,
                "gpu_device_utilization_percent": gpu,
            }
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            cpu_values.append(float(cpu))
            rss_values.append(float(rss_kib) / 1024.0)
            if gpu is not None:
                gpu_values.append(float(gpu))
            time.sleep(max(0.25, float(args.interval)))
    summary = {
        "root_pid": int(args.root_pid),
        "started_epoch": started,
        "ended_epoch": time.time(),
        "samples": len(cpu_values),
        "process_tree_cpu_percent_median": (
            statistics.median(cpu_values) if cpu_values else None
        ),
        "process_tree_cpu_percent_p95": _percentile(cpu_values, 0.95),
        "process_tree_cpu_percent_max": max(cpu_values) if cpu_values else None,
        "process_tree_rss_mib_median": (
            statistics.median(rss_values) if rss_values else None
        ),
        "process_tree_rss_mib_max": max(rss_values) if rss_values else None,
        "gpu_device_utilization_percent_median": (
            statistics.median(gpu_values) if gpu_values else None
        ),
        "gpu_device_utilization_percent_p95": _percentile(gpu_values, 0.95),
        "gpu_device_utilization_percent_max": (
            max(gpu_values) if gpu_values else None
        ),
        "gpu_samples": len(gpu_values),
        "host_cpu_count": os.cpu_count(),
        "jsonl": str(jsonl),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"resource_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
