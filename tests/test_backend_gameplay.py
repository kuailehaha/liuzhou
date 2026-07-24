from __future__ import annotations

import pytest
import torch
from fastapi import HTTPException

from backend.model_loader import clear_model_cache, get_model_with_metadata
from src.game_state import GameState, Phase, Player
from src.move_generator import apply_move
from src.neural_network import ChessNet, NUM_INPUT_CHANNELS


def _model() -> ChessNet:
    torch.manual_seed(123)
    return ChessNet(
        board_size=GameState.BOARD_SIZE,
        num_input_channels=NUM_INPUT_CHANNELS,
    )


def test_mutable_checkpoint_alias_is_reloaded_by_sha(tmp_path) -> None:
    clear_model_cache()
    checkpoint = tmp_path / "best_model.pt"
    first_state = _model().state_dict()
    torch.save({"model_state_dict": first_state}, checkpoint)
    first_model, first_metadata = get_model_with_metadata(str(checkpoint), "cpu")

    second_state = {key: value.clone() for key, value in first_state.items()}
    first_key = next(iter(second_state))
    second_state[first_key].add_(1.0)
    torch.save({"model_state_dict": second_state}, checkpoint)
    second_model, second_metadata = get_model_with_metadata(str(checkpoint), "cpu")

    assert first_metadata["modelSha256"] != second_metadata["modelSha256"]
    assert first_model is not second_model
    assert not torch.equal(
        first_model.state_dict()[first_key],
        second_model.state_dict()[first_key],
    )


def test_portable_gameplay_reuses_tree_across_same_player_atomic_phases() -> None:
    from v1.python.portable_gameplay_agent import PortableGameplayAgent, state_fingerprint

    state = GameState(
        phase=Phase.MARK_SELECTION,
        current_player=Player.BLACK,
        pending_marks_required=2,
        pending_marks_remaining=2,
    )
    state.board[0][0] = Player.WHITE.value
    state.board[1][1] = Player.WHITE.value
    state.board[2][2] = Player.WHITE.value
    agent = PortableGameplayAgent(
        _model(),
        mcts_simulations=2,
        temperature=0.0,
        device="cpu",
        portable_mcts_backend="python",
    )
    agent.sync_state(state)
    tree = agent._python_tree

    first_move = agent.select_move(state)
    after_first = apply_move(state, first_move, quiet=True)
    assert after_first.current_player == Player.BLACK
    assert agent._python_tree is tree
    assert state_fingerprint(agent._python_tree.root.state) == state_fingerprint(after_first)

    agent.select_move(after_first)
    assert agent.last_search["treeReused"] is True
    assert agent.last_search["fallbackCount"] == 0
    assert agent.last_search["illegalActionCount"] == 0
    assert agent.last_search["nonFiniteCount"] == 0
    assert agent.last_search["top"]
    assert "q" in agent.last_search["top"][0]


def test_backend_refuses_missing_model_without_random_fallback() -> None:
    from backend.main import _create_ai_agent

    with pytest.raises(HTTPException, match="No model checkpoint"):
        _create_ai_agent(
            model_path=None,
            simulations=2,
            temperature=0.0,
            device="cpu",
            search_backend="portable_cpp",
        )


def test_backend_game_record_keeps_move_states_and_root_audit() -> None:
    from backend.game_manager import GameManager
    from backend.main import _maybe_run_ai_turn, _prepare_payload, _sync_agent
    from v1.python.portable_gameplay_agent import PortableGameplayAgent

    agent = PortableGameplayAgent(
        _model(),
        mcts_simulations=2,
        temperature=0.0,
        device="cpu",
        portable_mcts_backend="python",
    )
    session = GameManager().create_session(
        human_player=Player.WHITE,
        ai_agent=agent,
        ai_metadata={
            "modelSha256": "test-sha",
            "searchBackend": "portable_python",
        },
    )
    _sync_agent(agent, session.state)

    moves = _maybe_run_ai_turn(session)
    payload = _prepare_payload(session, ai_moves=moves)

    assert len(moves) == 1
    assert len(payload["gameRecord"]) == 1
    row = payload["gameRecord"][0]
    assert row["sequence"] == 1
    assert row["actor"] == "AI"
    assert row["player"] == "BLACK"
    assert row["phase"] == "PLACEMENT"
    assert row["move"]["action_type"] == "place"
    assert row["stateBefore"]["currentPlayer"] == "BLACK"
    assert row["stateAfter"]["currentPlayer"] == "WHITE"
    assert row["search"]["backend"] == "portable_python"
    assert row["search"]["chosenActionIndex"] is not None
    assert row["search"]["top"]
    assert {"actionIndex", "visits", "prior", "q"} <= set(
        row["search"]["top"][0]
    )
    assert row["search"]["fallbackCount"] == 0
    assert row["search"]["illegalActionCount"] == 0
    assert row["search"]["nonFiniteCount"] == 0
