from __future__ import annotations

import time

import pytest
import torch

from src.game_state import GameState
from src.move_generator import apply_move, generate_all_legal_moves
from src.neural_network import NUM_INPUT_CHANNELS, ChessNet
from src.policy_batch import action_to_index


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _model() -> ChessNet:
    torch.manual_seed(123)
    return ChessNet(
        board_size=GameState.BOARD_SIZE,
        num_input_channels=NUM_INPUT_CHANNELS,
    ).cuda().eval()


def test_cuda_tree_gameplay_advances_ai_and_external_human_moves() -> None:
    from v1.python.cuda_tree_gameplay_agent import (
        CudaTreeGameplayAgent,
        state_fingerprint,
    )

    state = GameState()
    agent = CudaTreeGameplayAgent(
        _model(),
        mcts_simulations=2,
        temperature=0.0,
        device="cuda:0",
        precision="fp32",
    )
    agent.sync_state(state)

    ai_move = agent.select_move(state)
    after_ai = apply_move(state, ai_move, quiet=True)
    assert agent.expected_fingerprint == state_fingerprint(after_ai)

    human_move = generate_all_legal_moves(after_ai)[0]
    after_human = apply_move(after_ai, human_move, quiet=True)
    agent.observe_external_move(after_ai, human_move, after_human)
    assert agent.expected_fingerprint == state_fingerprint(after_human)

    agent.select_move(after_human)
    audit = agent.audit_metadata()
    assert audit["searchBackend"] == "cuda_tree"
    assert audit["device"] == "cuda:0"
    assert audit["precision"] == "fp32"
    assert audit["lastSearch"]["backend"] == "cuda_tree"
    assert audit["lastSearch"]["top"]
    assert audit["lastSearch"]["fallbackCount"] == 0
    assert audit["lastSearch"]["illegalActionCount"] == 0
    assert audit["lastSearch"]["nonFiniteCount"] == 0
    assert audit["lastSearch"]["subtreeStateMismatches"] == 0


def test_cuda_tree_gameplay_ponders_in_interruptible_chunks() -> None:
    from v1.python.cuda_tree_gameplay_agent import CudaTreeGameplayAgent

    state = GameState()
    agent = CudaTreeGameplayAgent(
        _model(),
        mcts_simulations=8,
        temperature=0.0,
        device="cuda:0",
        precision="fp32",
        ponder_enabled=True,
        ponder_chunk_simulations=2,
        ponder_max_simulations=4,
    )
    agent.sync_state(state)
    agent.start_pondering(state)

    deadline = time.monotonic() + 10.0
    while agent.audit_metadata()["ponderRounds"] < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    agent.stop_pondering()

    audit = agent.audit_metadata()
    assert audit["ponderActive"] is False
    assert audit["ponderRounds"] >= 1
    assert 2 <= audit["ponderSimulations"] <= 4
    assert audit["ponderError"] is None

    assert agent.mcts.last_root_visits is not None
    ponder_action = int(agent.mcts.last_root_visits[0].argmax().item())
    move_by_index = {
        action_to_index(move, GameState.BOARD_SIZE): move
        for move in generate_all_legal_moves(state)
    }
    human_move = move_by_index[ponder_action]
    after_human = apply_move(state, human_move, quiet=True)
    agent.observe_external_move(state, human_move, after_human)
    assert int(agent.mcts.active_root_visit_counts()[0].item()) > 0

    agent.select_move(after_human)
    last = agent.audit_metadata()["lastSearch"]
    assert last["rootVisitsBeforeSearch"] > 0
    assert 2 <= last["simulationsRun"] <= 8
    assert last["simulationsRun"] % 2 == 0
