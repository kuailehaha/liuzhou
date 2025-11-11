#pragma once

#include <torch/extension.h>

#include <vector>

#include "v0/game_state.hpp"

namespace v0 {

struct TensorStateBatch {
    torch::Tensor board;
    torch::Tensor marks_black;
    torch::Tensor marks_white;
    torch::Tensor phase;
    torch::Tensor current_player;
    torch::Tensor pending_marks_required;
    torch::Tensor pending_marks_remaining;
    torch::Tensor pending_captures_required;
    torch::Tensor pending_captures_remaining;
    torch::Tensor forced_removals_done;
    torch::Tensor move_count;
    torch::Tensor mask_alive;
    int64_t board_size{kBoardSize};

    TensorStateBatch To(const torch::Device& device) const;
    TensorStateBatch Clone() const;
};

TensorStateBatch FromGameStates(
    const std::vector<GameState>& states,
    const torch::Device& device);

std::vector<GameState> ToGameStates(const TensorStateBatch& batch);

}  // namespace v0
