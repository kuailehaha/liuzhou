#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace {

constexpr int kDirections[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

std::tuple<torch::Tensor, torch::Tensor> project_policy_logits_fast(
    const torch::Tensor& log_p1,
    const torch::Tensor& log_p2,
    const torch::Tensor& log_pmc,
    const torch::Tensor& legal_mask,
    int64_t placement_dim,
    int64_t movement_dim,
    int64_t selection_dim,
    int64_t auxiliary_dim) {
    TORCH_CHECK(log_p1.dim() == 2, "log_p1 must be 2D (B, placement_dim)");
    TORCH_CHECK(log_p2.dim() == 2, "log_p2 must be 2D (B, placement_dim)");
    TORCH_CHECK(log_pmc.dim() == 2, "log_pmc must be 2D (B, placement_dim)");
    TORCH_CHECK(
        log_p1.sizes() == log_p2.sizes() && log_p1.sizes() == log_pmc.sizes(),
        "All policy heads must share the same shape.");

    TORCH_CHECK(
        log_p1.options().device() == log_p2.options().device() &&
        log_p1.options().device() == log_pmc.options().device(),
        "All policy heads must be on the same device.");

    TORCH_CHECK(
        log_p1.scalar_type() == log_p2.scalar_type() &&
        log_p1.scalar_type() == log_pmc.scalar_type(),
        "All policy heads must share the same dtype.");

    TORCH_CHECK(legal_mask.dim() == 2, "legal_mask must be 2D (B, total_dim)");
    TORCH_CHECK(
        legal_mask.scalar_type() == torch::kBool,
        "legal_mask must be of dtype bool.");

    const int64_t batch_size = log_p1.size(0);
    const int64_t head_dim = log_p1.size(1);
    TORCH_CHECK(
        head_dim == placement_dim,
        "Policy head dimension mismatch: expected ",
        placement_dim,
        ", got ",
        head_dim,
        ".");

    const int64_t total_dim = placement_dim + movement_dim + selection_dim + auxiliary_dim;
    TORCH_CHECK(
        legal_mask.size(0) == batch_size && legal_mask.size(1) == total_dim,
        "legal_mask expected shape (",
        batch_size,
        ", ",
        total_dim,
        "), got (",
        legal_mask.size(0),
        ", ",
        legal_mask.size(1),
        ").");

    const double root = std::sqrt(static_cast<double>(placement_dim));
    const int64_t board_size = static_cast<int64_t>(std::llround(root));
    TORCH_CHECK(
        board_size * board_size == placement_dim,
        "placement_dim must be a perfect square representing the board area.");

    constexpr int kDirs = 4;
    TORCH_CHECK(
        movement_dim == placement_dim * kDirs,
        "movement_dim mismatch: expected ",
        placement_dim * kDirs,
        ", got ",
        movement_dim,
        ".");

    const auto options = log_p1.options();

    auto combined = torch::empty({batch_size, total_dim}, options);
    combined.narrow(1, 0, placement_dim).copy_(log_p1);

    // Precompute movement logits by gathering destination scores.
    auto device = log_p1.device();
    auto index_options = torch::TensorOptions().dtype(torch::kLong).device(device);

    auto indices = torch::arange(placement_dim, index_options);
    // auto rows = indices.div(board_size);
    auto rows = torch::floor_divide(indices, board_size); 
    auto cols = indices.remainder(board_size);

    std::vector<torch::Tensor> movement_chunks;
    movement_chunks.reserve(kDirs);

    const auto neginf = -std::numeric_limits<double>::infinity();

    for (int dir = 0; dir < kDirs; ++dir) {
        const int dr = kDirections[dir][0];
        const int dc = kDirections[dir][1];

        auto dest_rows = rows + dr;
        auto dest_cols = cols + dc;

        auto valid_rows = dest_rows.ge(0) & dest_rows.lt(board_size);
        auto valid_cols = dest_cols.ge(0) & dest_cols.lt(board_size);
        auto valid = valid_rows & valid_cols;

        auto dest_indices = dest_rows * board_size + dest_cols;
        // dest_indices = dest_indices.clamp(0, placement_dim - 1);
        dest_indices = dest_indices.clamp(0, placement_dim - 1);


        auto gathered = log_p1.index_select(1, dest_indices);
        auto movement_dir = log_p2 + gathered;

        auto invalid_mask = valid.logical_not().unsqueeze(0).expand_as(movement_dir);
        movement_dir.masked_fill_(invalid_mask, neginf);
        movement_chunks.push_back(movement_dir);
    }

    auto movement_concat = torch::stack(movement_chunks, 2).reshape({batch_size, movement_dim});
    combined.narrow(1, placement_dim, movement_dim).copy_(movement_concat);
    combined.narrow(1, placement_dim + movement_dim, selection_dim).copy_(log_pmc);

    if (auxiliary_dim > 0) {
        combined.narrow(1, placement_dim + movement_dim + selection_dim, auxiliary_dim).zero_();
    }

    auto masked_logits = combined.masked_fill(legal_mask.logical_not(), neginf);
    auto probs = torch::zeros_like(combined);

    using torch::indexing::Slice;

    for (int64_t b = 0; b < batch_size; ++b) {
        auto legal_row = legal_mask[b];
        const int64_t legal_count = legal_row.sum().item<int64_t>();
        if (legal_count == 0) {
            continue;
        }

        // auto legal_indices = legal_row.nonzero().squeeze(1)
        auto legal_indices = legal_row.nonzero().squeeze(1).to(torch::kLong).contiguous();
        auto row_logits = masked_logits.index({b, legal_indices});

        const bool any_finite = row_logits.isfinite().any().item<bool>();
        if (!any_finite) {
            auto zeros = torch::zeros_like(row_logits);
            masked_logits.index_put_({b, legal_indices}, zeros);
            row_logits = zeros;
        }

        auto row_probs = torch::softmax(row_logits, 0);
        probs.index_put_({b, legal_indices}, row_probs);
    }

    return {probs, masked_logits};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "project_policy_logits_fast",
        &project_policy_logits_fast,
        "Fused policy projection with masked softmax");
}
