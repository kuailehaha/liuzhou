from __future__ import annotations

import pytest
import torch

from src.game_state import GameState, Phase, Player
from src.neural_network import (
    LEGACY_NUM_INPUT_CHANNELS,
    LEGACY_STATE_ENCODING_SCHEMA,
    MARKOV_NUM_INPUT_CHANNELS,
    MARKOV_STATE_ENCODING_SCHEMA,
    state_encoding_feature_names,
    state_to_tensor,
)
from v1.python.trajectory_buffer import TensorSelfPlayBatch
from v1.train import (
    _distribute_replay_file_budgets,
    _validate_self_play_schema,
)


def _state(**overrides: int) -> GameState:
    values = {
        "phase": Phase.MARK_SELECTION,
        "current_player": Player.BLACK,
        "forced_removals_done": 1,
        "move_count": 72,
        "pending_marks_required": 2,
        "pending_marks_remaining": 1,
        "pending_captures_required": 2,
        "pending_captures_remaining": 1,
        "moves_since_capture": 18,
    }
    values.update(overrides)
    return GameState(**values)


def test_markov_v2_distinguishes_rule_relevant_scalar_state() -> None:
    left = _state(move_count=71, moves_since_capture=17)
    right = _state(move_count=72, moves_since_capture=18)

    legacy_left = state_to_tensor(
        left, left.current_player, schema=LEGACY_STATE_ENCODING_SCHEMA
    )
    legacy_right = state_to_tensor(
        right, right.current_player, schema=LEGACY_STATE_ENCODING_SCHEMA
    )
    assert torch.equal(legacy_left, legacy_right)

    markov_left = state_to_tensor(
        left, left.current_player, schema=MARKOV_STATE_ENCODING_SCHEMA
    )
    markov_right = state_to_tensor(
        right, right.current_player, schema=MARKOV_STATE_ENCODING_SCHEMA
    )
    assert markov_left.shape == (1, MARKOV_NUM_INPUT_CHANNELS, 6, 6)
    assert not torch.equal(markov_left, markov_right)


def test_markov_v2_scalar_planes_are_normalized_and_constant() -> None:
    state = _state()
    encoded = state_to_tensor(
        state, state.current_player, schema=MARKOV_STATE_ENCODING_SCHEMA
    )[0]

    expected = (1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5)
    for channel, value in enumerate(expected, start=LEGACY_NUM_INPUT_CHANNELS):
        assert torch.allclose(
            encoded[channel],
            torch.full_like(encoded[channel], value),
            atol=0.0,
            rtol=0.0,
        )

    white = _state(current_player=Player.WHITE)
    encoded_white = state_to_tensor(
        white, white.current_player, schema=MARKOV_STATE_ENCODING_SCHEMA
    )[0]
    assert torch.count_nonzero(encoded_white[LEGACY_NUM_INPUT_CHANNELS]) == 0


def test_state_encoding_feature_metadata_matches_channel_count() -> None:
    assert len(state_encoding_feature_names(LEGACY_STATE_ENCODING_SCHEMA)) == 11
    assert len(state_encoding_feature_names(MARKOV_STATE_ENCODING_SCHEMA)) == 19
    assert LEGACY_NUM_INPUT_CHANNELS == 11
    assert MARKOV_NUM_INPUT_CHANNELS == 19


def test_markov_training_rejects_legacy_replay_without_silent_fallback() -> None:
    samples = TensorSelfPlayBatch(
        state_tensors=torch.zeros(2, LEGACY_NUM_INPUT_CHANNELS, 6, 6),
        legal_masks=torch.zeros(2, 220, dtype=torch.bool),
        policy_targets=torch.zeros(2, 220),
        value_targets=torch.zeros(2),
        soft_value_targets=torch.zeros(2),
    )
    with pytest.raises(RuntimeError, match="tensor channel mismatch"):
        _validate_self_play_schema(
            samples=samples,
            metadata={},
            expected_schema=MARKOV_STATE_ENCODING_SCHEMA,
            expected_channels=MARKOV_NUM_INPUT_CHANNELS,
            source="legacy_replay.pt",
        )


def test_markov_streaming_manifest_requires_schema_metadata() -> None:
    with pytest.raises(RuntimeError, match="metadata is required"):
        _validate_self_play_schema(
            samples=None,
            metadata={},
            expected_schema=MARKOV_STATE_ENCODING_SCHEMA,
            expected_channels=MARKOV_NUM_INPUT_CHANNELS,
            source="unknown_manifest.pt",
        )


def test_replay_history_budget_equals_primary_sample_budget() -> None:
    budgets = _distribute_replay_file_budgets(10, 16)
    assert len(budgets) == 16
    assert sum(budgets) == 10
    assert budgets[:10] == [1] * 10
    assert budgets[10:] == [0] * 6
