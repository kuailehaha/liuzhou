#pragma once

#include <tuple>
#include <torch/extension.h>

namespace v0 {

std::tuple<torch::Tensor, torch::Tensor> encode_actions_fast(
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
    int64_t placement_dim,
    int64_t movement_dim,
    int64_t selection_dim,
    int64_t auxiliary_dim);

}  // namespace v0
