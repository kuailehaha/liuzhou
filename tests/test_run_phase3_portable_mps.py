import argparse
from pathlib import Path

from scripts.run_phase3_portable_mps import _training_command


def test_phase3_training_command_freezes_requested_contract(tmp_path: Path):
    checkpoint = tmp_path / "start.pt"
    optimizer = tmp_path / "optimizer.pt"
    anchors = [tmp_path / f"anchor_{index}.pt" for index in range(3)]
    args = argparse.Namespace(
        initial_checkpoint=checkpoint,
        initial_optimizer=optimizer,
        phase2_parent_run=tmp_path / "phase2",
        anchors=anchors,
        seed=123,
    )
    command = _training_command(
        args=args,
        run_dir=tmp_path / "formal",
        simulations=32,
        hours=40.0,
        formal=True,
    )
    joined = " ".join(command)
    assert "--state-encoding-schema markov_v2" in joined
    assert "--lineage-parent-iteration 2478" in joined
    assert "--symmetry-augmentation random_d4" in joined
    assert "--optimizer-name adamw" in joined
    assert "--strict-optimizer-restore" in command
    assert "--mcts-simulations 32" in joined
    assert "--no-sample-moves" in command
    assert "--policy-target-temperature 1" in joined
    assert "--policy-target-prior-pseudocount 1" in joined
    assert "--opening-random-start 3" in joined
    assert "--opening-random-anneal-fraction 0.25" in joined
    assert "--soft-label-alpha 0" in joined
    assert "--replay-window 16" in joined
    assert "--eval-simulations 64" in joined
    assert command.count("--promotion-anchor") == 3
    assert "--quality-strength-eval" in command
    assert "--strength-screen-simulations 256" in joined
