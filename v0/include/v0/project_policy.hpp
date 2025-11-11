#pragma once

#include <torch/extension.h>

namespace v0 {

std::tuple<torch::Tensor, torch::Tensor> project_policy_logits_fast(
    const torch::Tensor& log_p1,
    const torch::Tensor& log_p2,
    const torch::Tensor& log_pmc,
    const torch::Tensor& legal_mask,
    int64_t placement_dim,
    int64_t movement_dim,
    int64_t selection_dim,
    int64_t auxiliary_dim);

}  // namespace v0
