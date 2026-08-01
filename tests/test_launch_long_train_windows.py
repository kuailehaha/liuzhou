from __future__ import annotations

import json
import os
import sys

import pytest

from scripts import launch_long_train_windows


@pytest.mark.skipif(os.name != "nt", reason="Windows power request is platform-specific")
def test_windows_launcher_runs_child_and_records_completion(tmp_path) -> None:
    log_path = tmp_path / "launcher.log"
    state_path = tmp_path / "launcher.json"

    return_code = launch_long_train_windows.main(
        [
            "--log-file",
            str(log_path),
            "--state-file",
            str(state_path),
            "--",
            sys.executable,
            "-c",
            "print('launcher-child-ok')",
        ]
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert return_code == 0
    assert state["status"] == "completed"
    assert state["return_code"] == 0
    assert state["sleep_inhibition"] == "ES_SYSTEM_REQUIRED|ES_AWAYMODE_REQUIRED"
    assert "launcher-child-ok" in log_path.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "nt", reason="Windows power request is platform-specific")
def test_windows_launcher_reads_exact_command_from_json(tmp_path) -> None:
    log_path = tmp_path / "launcher.log"
    state_path = tmp_path / "launcher.json"
    command_path = tmp_path / "command.json"
    command_path.write_text(
        json.dumps([sys.executable, "-c", "print('command-file-ok')"]),
        encoding="utf-8",
    )

    assert (
        launch_long_train_windows.main(
            [
                "--log-file",
                str(log_path),
                "--state-file",
                str(state_path),
                "--command-file",
                str(command_path),
            ]
        )
        == 0
    )
    assert "command-file-ok" in log_path.read_text(encoding="utf-8")
