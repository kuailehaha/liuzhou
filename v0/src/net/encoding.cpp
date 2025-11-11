#include "v0/net_encoding.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace v0 {

namespace {

constexpr std::array<int64_t, 7> kPhaseOrder = {1, 2, 3, 4, 5, 6, 7};

torch::Tensor ensure_contiguous(const torch::Tensor& tensor) {
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

int64_t resolve_dim(int64_t requested_dim, int64_t total_dims) {
    if (requested_dim >= 0) {
        return requested_dim;
    }
    return total_dims + requested_dim;
}

}  // namespace

torch::Tensor states_to_model_input(
    const torch::Tensor& board_in,
    const torch::Tensor& marks_black_in,
    const torch::Tensor& marks_white_in,
    const torch::Tensor& phase_in,
    const torch::Tensor& current_player_in) {
    TORCH_CHECK(board_in.dim() == 3, "board must be (B, H, W)");
    auto board = ensure_contiguous(board_in);
    auto marks_black = ensure_contiguous(marks_black_in);
    auto marks_white = ensure_contiguous(marks_white_in);
    auto phase = ensure_contiguous(phase_in);
    auto current_player = ensure_contiguous(current_player_in);

    const int64_t batch_size = board.size(0);
    const int64_t height = board.size(1);
    const int64_t width = board.size(2);
    TORCH_CHECK(marks_black.sizes() == board.sizes(), "marks_black shape mismatch");
    TORCH_CHECK(marks_white.sizes() == board.sizes(), "marks_white shape mismatch");
    TORCH_CHECK(phase.numel() == batch_size, "phase length mismatch");
    TORCH_CHECK(current_player.numel() == batch_size, "current_player length mismatch");

    auto dtype = torch::kFloat32;
    auto device = board.device();

    auto board_dtype = board.scalar_type();
    auto current = current_player.view({batch_size, 1, 1}).to(board_dtype);
    auto self_mask = board.eq(current).to(dtype);
    auto opp_mask = board.eq(current.neg()).to(dtype);
    auto board_planes = torch::stack({self_mask, opp_mask}, 1);

    auto marks_black_f = marks_black.to(dtype);
    auto marks_white_f = marks_white.to(dtype);
    auto current_is_black = current_player.eq(1).view({batch_size, 1, 1});
    auto self_marks = torch::where(current_is_black, marks_black_f, marks_white_f);
    auto opp_marks = torch::where(current_is_black, marks_white_f, marks_black_f);
    auto mark_planes = torch::stack({self_marks, opp_marks}, 1);

    // auto phase_ids = torch::tensor(
    //     kPhaseOrder,
    //     torch::TensorOptions().dtype(torch::kInt64).device(phase.device()));

    auto phase_ids = torch::arange(
        /*start=*/1,
        /*end=*/static_cast<int64_t>(kPhaseOrder.size()) + 1,
        torch::TensorOptions().dtype(torch::kInt64)
    ).to(phase.device());
    auto matches =
        phase.view({batch_size, 1}).eq(phase_ids.view({1, static_cast<int64_t>(kPhaseOrder.size())}));
    auto phase_one_hot = matches.to(dtype).view({batch_size, static_cast<int64_t>(kPhaseOrder.size()), 1, 1});
    auto phase_planes = phase_one_hot.expand({batch_size, static_cast<int64_t>(kPhaseOrder.size()), height, width});

    auto stacked = torch::cat({board_planes, mark_planes, phase_planes}, 1).contiguous();
    return stacked;
}

torch::Tensor postprocess_value_head(const torch::Tensor& raw_values) {
    return torch::tanh(raw_values);
}

torch::Tensor apply_temperature_scaling(
    const torch::Tensor& probs_in,
    double temperature,
    int64_t dim) {
    auto probs = probs_in.contiguous();
    if (temperature <= 1e-6) {
        return probs.clone();
    }
    auto exponent = 1.0 / std::max(temperature, 1e-6);
    auto positive = torch::where(
        probs > 0,
        probs.pow(exponent),
        torch::zeros_like(probs));

    int64_t reduce_dim = resolve_dim(dim, probs.dim());
    TORCH_CHECK(reduce_dim >= 0 && reduce_dim < probs.dim(), "Invalid dimension for temperature scaling");
    auto sums = positive.sum(reduce_dim, /*keepdim=*/true);
    auto normalized = torch::where(
        sums > 0,
        positive / sums,
        torch::zeros_like(positive));
    return normalized;
}

}  // namespace v0
