from __future__ import annotations

import torch

from scripts.upgrade_v1_state_schema import (
    migrate_model_checkpoint,
    migrate_optimizer_state,
)
from src.game_state import GameState
from src.neural_network import (
    ChessNet,
    LEGACY_NUM_INPUT_CHANNELS,
    MARKOV_NUM_INPUT_CHANNELS,
    MARKOV_STATE_ENCODING_SCHEMA,
)


def _small_model(channels: int) -> ChessNet:
    return ChessNet(
        board_size=GameState.BOARD_SIZE,
        num_input_channels=channels,
        trunk_channels=8,
        num_blocks=1,
        policy_channels=4,
        value_channels=4,
        value_mlp_channels=8,
    )


def test_model_schema_migration_preserves_outputs_with_zero_new_planes() -> None:
    torch.manual_seed(11)
    old_model = _small_model(LEGACY_NUM_INPUT_CHANNELS).eval()
    payload = {
        "model_state_dict": old_model.state_dict(),
        "num_input_channels": LEGACY_NUM_INPUT_CHANNELS,
    }
    migrated = migrate_model_checkpoint(
        payload,
        source_sha256="abc",
        parent_iteration=2478,
    )
    new_model = _small_model(MARKOV_NUM_INPUT_CHANNELS).eval()
    new_model.load_state_dict(migrated["model_state_dict"], strict=True)

    old_input = torch.randn(4, LEGACY_NUM_INPUT_CHANNELS, 6, 6)
    new_input = torch.zeros(4, MARKOV_NUM_INPUT_CHANNELS, 6, 6)
    new_input[:, :LEGACY_NUM_INPUT_CHANNELS] = old_input
    with torch.inference_mode():
        old_outputs = old_model(old_input)
        new_outputs = new_model(new_input)
    for old, new in zip(old_outputs, new_outputs):
        assert torch.allclose(old, new, atol=1e-6, rtol=0.0)
    assert migrated["state_encoding_schema"] == MARKOV_STATE_ENCODING_SCHEMA
    assert migrated["num_input_channels"] == MARKOV_NUM_INPUT_CHANNELS
    assert torch.count_nonzero(
        migrated["model_state_dict"]["stem_conv.weight"][
            :, LEGACY_NUM_INPUT_CHANNELS:
        ]
    ) == 0


def test_optimizer_migration_zero_extends_stem_moments_and_loads() -> None:
    torch.manual_seed(12)
    old_model = _small_model(LEGACY_NUM_INPUT_CHANNELS)
    optimizer = torch.optim.Adam(old_model.parameters(), lr=1e-4)
    loss = sum(output.float().mean() for output in old_model(
        torch.randn(2, LEGACY_NUM_INPUT_CHANNELS, 6, 6)
    ))
    loss.backward()
    optimizer.step()

    migrated_model = migrate_model_checkpoint(
        {
            "model_state_dict": old_model.state_dict(),
            "num_input_channels": LEGACY_NUM_INPUT_CHANNELS,
        },
        source_sha256="abc",
        parent_iteration=2478,
    )
    migrated_optimizer = migrate_optimizer_state(optimizer.state_dict())
    new_model = _small_model(MARKOV_NUM_INPUT_CHANNELS)
    new_model.load_state_dict(migrated_model["model_state_dict"], strict=True)
    # Phase 3 changes the update rule to AdamW while preserving all Adam
    # first/second moments from the paired Phase-2 optimizer.
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
    new_optimizer.load_state_dict(migrated_optimizer)

    stem_state = new_optimizer.state[new_model.stem_conv.weight]
    assert stem_state["exp_avg"].shape[1] == MARKOV_NUM_INPUT_CHANNELS
    assert stem_state["exp_avg_sq"].shape[1] == MARKOV_NUM_INPUT_CHANNELS
    assert torch.count_nonzero(
        stem_state["exp_avg"][:, LEGACY_NUM_INPUT_CHANNELS:]
    ) == 0

    new_optimizer.zero_grad(set_to_none=True)
    next_loss = sum(output.float().mean() for output in new_model(
        torch.randn(2, MARKOV_NUM_INPUT_CHANNELS, 6, 6)
    ))
    next_loss.backward()
    new_optimizer.step()
