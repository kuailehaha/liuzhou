#pragma once

#include <torch/extension.h>

namespace v0 {

torch::Tensor states_to_model_input(
    const torch::Tensor& board,
    const torch::Tensor& marks_black,
    const torch::Tensor& marks_white,
    const torch::Tensor& phase,
    const torch::Tensor& current_player);

torch::Tensor postprocess_value_head(const torch::Tensor& raw_values);

torch::Tensor apply_temperature_scaling(
    const torch::Tensor& probs,
    double temperature,
    int64_t dim = -1);

}  // namespace v0
