#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <tuple>

namespace v0 {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__device__ __forceinline__ float LoadScore(
    const scalar_t* p1,
    const scalar_t* p2,
    const scalar_t* pmc,
    int64_t row,
    int action,
    int placement_dim,
    int movement_dim,
    int selection_dim,
    int board_size) {
    const int64_t head_base = row * placement_dim;
    if (action < placement_dim) {
        return static_cast<float>(p1[head_base + action]);
    }
    if (action < placement_dim + movement_dim) {
        const int movement = action - placement_dim;
        const int source = movement / 4;
        const int direction = movement % 4;
        const int row_index = source / board_size;
        const int col_index = source % board_size;
        const int dr[4] = {-1, 1, 0, 0};
        const int dc[4] = {0, 0, -1, 1};
        const int dest_row = row_index + dr[direction];
        const int dest_col = col_index + dc[direction];
        if (dest_row < 0 || dest_row >= board_size || dest_col < 0 || dest_col >= board_size) {
            return -INFINITY;
        }
        const int destination = dest_row * board_size + dest_col;
        return static_cast<float>(p2[head_base + source])
            + static_cast<float>(p1[head_base + destination]);
    }
    if (action < placement_dim + movement_dim + selection_dim) {
        return static_cast<float>(pmc[head_base + action - placement_dim - movement_dim]);
    }
    return 0.0F;
}

template <typename scalar_t>
__global__ void ProjectPolicyKernel(
    const scalar_t* __restrict__ p1,
    const scalar_t* __restrict__ p2,
    const scalar_t* __restrict__ pmc,
    const bool* __restrict__ legal,
    int64_t batch_size,
    int placement_dim,
    int movement_dim,
    int selection_dim,
    int total_dim,
    int board_size,
    scalar_t* __restrict__ probabilities,
    scalar_t* __restrict__ masked_logits) {
    const int64_t row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }
    __shared__ float reduction[kThreads];
    __shared__ int legal_reduction[kThreads];
    const int tid = threadIdx.x;
    const int64_t base = row * total_dim;
    float score = -INFINITY;
    const bool is_legal = tid < total_dim && legal[base + tid];
    if (tid < total_dim) {
        score = LoadScore(
            p1, p2, pmc, row, tid, placement_dim, movement_dim,
            selection_dim, board_size);
    }
    const bool finite_legal = is_legal && isfinite(score);
    reduction[tid] = finite_legal ? score : -INFINITY;
    const bool softmax_abnormal = is_legal && (isnan(score) || score == INFINITY);
    legal_reduction[tid] = (is_legal ? 1 : 0) | (softmax_abnormal ? 2 : 0);
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + offset]);
            legal_reduction[tid] |= legal_reduction[tid + offset];
        }
        __syncthreads();
    }
    const float maximum = reduction[0];
    const bool has_legal = legal_reduction[0] != 0;
    const bool has_finite = isfinite(maximum);
    const bool has_softmax_abnormal = (legal_reduction[0] & 2) != 0;
    const float exponential = finite_legal && has_finite ? __expf(score - maximum) : 0.0F;
    reduction[tid] = exponential;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduction[tid] += reduction[tid + offset];
        }
        __syncthreads();
    }
    if (tid < total_dim) {
        float output_logit = is_legal ? score : -INFINITY;
        if (has_legal && !has_finite && is_legal) {
            output_logit = 0.0F;
        }
        const float probability = has_finite && has_softmax_abnormal
            ? NAN
            : (has_finite && reduction[0] > 0.0F ? exponential / reduction[0] : 0.0F);
        masked_logits[base + tid] = static_cast<scalar_t>(output_logit);
        probabilities[base + tid] = static_cast<scalar_t>(probability);
    }
}

template <typename scalar_t>
void Launch(
    const torch::Tensor& p1,
    const torch::Tensor& p2,
    const torch::Tensor& pmc,
    const torch::Tensor& legal,
    int placement_dim,
    int movement_dim,
    int selection_dim,
    int total_dim,
    int board_size,
    torch::Tensor& probabilities,
    torch::Tensor& masked_logits) {
    ProjectPolicyKernel<scalar_t><<<
        static_cast<unsigned int>(p1.size(0)),
        kThreads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        p1.data_ptr<scalar_t>(),
        p2.data_ptr<scalar_t>(),
        pmc.data_ptr<scalar_t>(),
        legal.data_ptr<bool>(),
        p1.size(0),
        placement_dim,
        movement_dim,
        selection_dim,
        total_dim,
        board_size,
        probabilities.data_ptr<scalar_t>(),
        masked_logits.data_ptr<scalar_t>());
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> project_policy_logits_cuda(
    const torch::Tensor& log_p1_in,
    const torch::Tensor& log_p2_in,
    const torch::Tensor& log_pmc_in,
    const torch::Tensor& legal_mask_in,
    int64_t placement_dim,
    int64_t movement_dim,
    int64_t selection_dim,
    int64_t auxiliary_dim) {
    TORCH_CHECK(log_p1_in.is_cuda(), "project_policy_logits_cuda requires CUDA tensors");
    TORCH_CHECK(log_p1_in.dim() == 2, "log_p1 must be [B, placement_dim]");
    TORCH_CHECK(log_p1_in.sizes() == log_p2_in.sizes(), "policy head shape mismatch");
    TORCH_CHECK(log_p1_in.sizes() == log_pmc_in.sizes(), "policy head shape mismatch");
    TORCH_CHECK(log_p1_in.scalar_type() == log_p2_in.scalar_type(), "policy dtype mismatch");
    TORCH_CHECK(log_p1_in.scalar_type() == log_pmc_in.scalar_type(), "policy dtype mismatch");
    TORCH_CHECK(
        log_p1_in.scalar_type() == torch::kFloat32 || log_p1_in.scalar_type() == torch::kFloat16,
        "CUDA policy projection supports float32 and float16");
    const int64_t total_dim = placement_dim + movement_dim + selection_dim + auxiliary_dim;
    TORCH_CHECK(placement_dim == 36 && movement_dim == 144 && selection_dim == 36,
        "CUDA policy projection currently requires the 220-action 6x6 encoding");
    TORCH_CHECK(legal_mask_in.scalar_type() == torch::kBool, "legal mask must be bool");
    TORCH_CHECK(legal_mask_in.size(0) == log_p1_in.size(0), "legal batch mismatch");
    TORCH_CHECK(legal_mask_in.size(1) == total_dim, "legal action dimension mismatch");

    auto p1 = log_p1_in.contiguous();
    auto p2 = log_p2_in.contiguous();
    auto pmc = log_pmc_in.contiguous();
    auto legal = legal_mask_in.contiguous();
    auto probabilities = torch::empty({p1.size(0), total_dim}, p1.options());
    auto masked_logits = torch::empty_like(probabilities);
    at::cuda::OptionalCUDAGuard guard(p1.device());
    if (p1.scalar_type() == torch::kFloat32) {
        Launch<float>(p1, p2, pmc, legal, 36, 144, 36, 220, 6, probabilities, masked_logits);
    } else {
        Launch<at::Half>(p1, p2, pmc, legal, 36, 144, 36, 220, 6, probabilities, masked_logits);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {probabilities, masked_logits};
}

}  // namespace v0
