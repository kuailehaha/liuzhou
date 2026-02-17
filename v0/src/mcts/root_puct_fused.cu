#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <limits>
#include <tuple>

namespace v0 {
namespace {

__global__ void RootPuctAllocateVisitsKernel(
    const float* __restrict__ priors,
    const float* __restrict__ leaf_values,
    const bool* __restrict__ valid_mask,
    int64_t num_roots,
    int64_t num_actions,
    int64_t num_simulations,
    float exploration_weight,
    float* __restrict__ visits,
    float* __restrict__ value_sum,
    float* __restrict__ root_values) {
    const int64_t root = static_cast<int64_t>(blockIdx.x);
    if (root >= num_roots) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);
    const int64_t row_offset = root * num_actions;

    extern __shared__ unsigned char smem[];
    float* s_best = reinterpret_cast<float*>(smem);
    int* s_best_idx = reinterpret_cast<int*>(s_best + nthreads);
    float* s_sum_visits = reinterpret_cast<float*>(s_best_idx + nthreads);
    float* s_sum_values = s_sum_visits + nthreads;
    __shared__ float s_total_visit;

    if (tid == 0) {
        s_total_visit = 0.0f;
    }
    __syncthreads();

    for (int64_t sim = 0; sim < num_simulations; ++sim) {
        const float sqrt_total = sqrtf(s_total_visit + 1.0f);
        float local_best = -std::numeric_limits<float>::infinity();
        int local_best_idx = -1;

        for (int64_t action = tid; action < num_actions; action += nthreads) {
            const int64_t idx = row_offset + action;
            if (!valid_mask[idx]) {
                continue;
            }
            const float visit = visits[idx];
            const float q = visit > 0.0f ? (value_sum[idx] / fmaxf(visit, 1e-8f)) : 0.0f;
            const float u = exploration_weight * priors[idx] * sqrt_total / (1.0f + visit);
            const float score = q + u;
            if (score > local_best || (score == local_best && (local_best_idx < 0 || action < local_best_idx))) {
                local_best = score;
                local_best_idx = static_cast<int>(action);
            }
        }

        s_best[tid] = local_best;
        s_best_idx[tid] = local_best_idx;
        __syncthreads();

        for (int offset = nthreads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float other_best = s_best[tid + offset];
                const int other_idx = s_best_idx[tid + offset];
                const float cur_best = s_best[tid];
                const int cur_idx = s_best_idx[tid];
                if (other_best > cur_best || (other_best == cur_best && other_idx >= 0 && (cur_idx < 0 || other_idx < cur_idx))) {
                    s_best[tid] = other_best;
                    s_best_idx[tid] = other_idx;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int chosen = s_best_idx[0];
            if (chosen >= 0) {
                const int64_t chosen_idx = row_offset + static_cast<int64_t>(chosen);
                visits[chosen_idx] += 1.0f;
                value_sum[chosen_idx] += leaf_values[chosen_idx];
                s_total_visit += 1.0f;
            }
        }
        __syncthreads();
    }

    float partial_visit = 0.0f;
    float partial_value = 0.0f;
    for (int64_t action = tid; action < num_actions; action += nthreads) {
        const int64_t idx = row_offset + action;
        partial_visit += visits[idx];
        partial_value += value_sum[idx];
    }
    s_sum_visits[tid] = partial_visit;
    s_sum_values[tid] = partial_value;
    __syncthreads();

    for (int offset = nthreads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum_visits[tid] += s_sum_visits[tid + offset];
            s_sum_values[tid] += s_sum_values[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float denom = fmaxf(s_sum_visits[0], 1.0f);
        root_values[root] = s_sum_values[0] / denom;
    }
}

int NextPow2AtMost1024(int64_t n) {
    int t = 1;
    while (t < static_cast<int>(n) && t < 1024) {
        t <<= 1;
    }
    if (t < 32) {
        return 32;
    }
    return t;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> root_puct_allocate_visits_cuda(
    torch::Tensor priors,
    torch::Tensor leaf_values,
    torch::Tensor valid_mask,
    int64_t num_simulations,
    double exploration_weight) {
    TORCH_CHECK(priors.is_cuda(), "priors must be CUDA tensor.");
    TORCH_CHECK(leaf_values.is_cuda(), "leaf_values must be CUDA tensor.");
    TORCH_CHECK(valid_mask.is_cuda(), "valid_mask must be CUDA tensor.");
    TORCH_CHECK(priors.dim() == 2, "priors must be 2D [R, A].");
    TORCH_CHECK(leaf_values.dim() == 2, "leaf_values must be 2D [R, A].");
    TORCH_CHECK(valid_mask.dim() == 2, "valid_mask must be 2D [R, A].");
    TORCH_CHECK(priors.sizes() == leaf_values.sizes(), "priors/leaf_values shape mismatch.");
    TORCH_CHECK(priors.sizes() == valid_mask.sizes(), "priors/valid_mask shape mismatch.");
    TORCH_CHECK(num_simulations > 0, "num_simulations must be positive.");

    auto priors_f = priors.to(torch::kFloat32).contiguous();
    auto leaf_f = leaf_values.to(torch::kFloat32).contiguous();
    auto mask_b = valid_mask.to(torch::kBool).contiguous();

    const int64_t num_roots = priors_f.size(0);
    const int64_t num_actions = priors_f.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(priors_f.device());
    auto visits = torch::zeros({num_roots, num_actions}, options);
    auto value_sum = torch::zeros({num_roots, num_actions}, options);
    auto root_values = torch::zeros({num_roots}, options);

    if (num_roots == 0 || num_actions == 0) {
        return std::make_tuple(visits, value_sum, root_values);
    }

    at::cuda::OptionalCUDAGuard guard(priors_f.device());
    const int threads = NextPow2AtMost1024(num_actions);
    const dim3 blocks(static_cast<unsigned int>(num_roots));
    const size_t shared_bytes = static_cast<size_t>(threads) * (
        sizeof(float) + sizeof(int) + sizeof(float) + sizeof(float));

    RootPuctAllocateVisitsKernel<<<blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        priors_f.data_ptr<float>(),
        leaf_f.data_ptr<float>(),
        mask_b.data_ptr<bool>(),
        num_roots,
        num_actions,
        num_simulations,
        static_cast<float>(exploration_weight),
        visits.data_ptr<float>(),
        value_sum.data_ptr<float>(),
        root_values.data_ptr<float>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(visits, value_sum, root_values);
}

}  // namespace v0
