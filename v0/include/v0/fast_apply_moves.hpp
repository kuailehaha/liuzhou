#pragma once

#include <tuple>
#include <torch/extension.h>

namespace v0 {

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
batch_apply_moves(
    torch::Tensor board,
    torch::Tensor marks_black,
    torch::Tensor marks_white,
    torch::Tensor phase,
    torch::Tensor current_player,
    torch::Tensor pending_marks_required,
    torch::Tensor pending_marks_remaining,
    torch::Tensor pending_captures_required,
    torch::Tensor pending_captures_remaining,
    torch::Tensor forced_removals_done,
    torch::Tensor move_count,
    torch::Tensor moves_since_capture,
    torch::Tensor action_codes,
    torch::Tensor parent_indices);

}  // namespace v0
