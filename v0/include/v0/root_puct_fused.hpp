#pragma once

#include <tuple>
#include <torch/extension.h>

namespace v0 {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> root_puct_allocate_visits_cuda(
    torch::Tensor priors,
    torch::Tensor leaf_values,
    torch::Tensor valid_mask,
    int64_t num_simulations,
    double exploration_weight);

}  // namespace v0
