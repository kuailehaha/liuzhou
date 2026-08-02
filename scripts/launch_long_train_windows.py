#!/usr/bin/env python3
"""Run a long-training command on Windows while preventing automatic sleep."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_system_awake(enabled: bool) -> None:
    if os.name != "nt":
        raise RuntimeError("launch_long_train_windows.py is only supported on Windows")
    flags = ES_CONTINUOUS
    if enabled:
        flags |= ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    if result == 0:
        raise OSError("SetThreadExecutionState failed")


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument(
        "--command-file",
        type=Path,
        default=None,
        help="JSON array containing the exact child command; avoids Task Scheduler quoting limits.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if args.command_file is not None:
        if command:
            raise ValueError("--command-file cannot be combined with a command after --")
        command_path = args.command_file.expanduser().resolve()
        payload = json.loads(command_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(
            isinstance(value, str) and value for value in payload
        ):
            raise ValueError("--command-file must contain a non-empty JSON string array")
        command = list(payload)
    if not command:
        raise ValueError("a training command is required after --")

    log_path = args.log_file.expanduser().resolve()
    state_path = args.state_file.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "launcher_pid": os.getpid(),
        "started_utc": utc_now(),
        "command": command,
        "cwd": str(ROOT_DIR),
        "log_file": str(log_path),
        "sleep_inhibition": "ES_SYSTEM_REQUIRED|ES_AWAYMODE_REQUIRED",
        "status": "starting",
    }
    atomic_write_json(state_path, state)

    set_system_awake(True)
    try:
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        python_paths = [str(ROOT_DIR)]
        v0_build = ROOT_DIR / "build" / "v0" / "src"
        if v0_build.is_dir():
            python_paths.insert(0, str(v0_build))
        inherited_pythonpath = child_env.get("PYTHONPATH", "").strip()
        if inherited_pythonpath:
            python_paths.append(inherited_pythonpath)
        child_env["PYTHONPATH"] = os.pathsep.join(python_paths)
        state["child_pythonpath"] = child_env["PYTHONPATH"]
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(
                f"[{utc_now()}] [windows_launcher] pid={os.getpid()} command={command!r}\n"
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                env=child_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            state.update({"status": "running", "child_pid": process.pid})
            atomic_write_json(state_path, state)
            return_code = int(process.wait())
            state.update(
                {
                    "status": "completed" if return_code == 0 else "failed",
                    "return_code": return_code,
                    "ended_utc": utc_now(),
                }
            )
            atomic_write_json(state_path, state)
            log_handle.write(
                f"[{utc_now()}] [windows_launcher] child_exit={return_code}\n"
            )
            return return_code
    finally:
        set_system_awake(False)


if __name__ == "__main__":
    raise SystemExit(main())
