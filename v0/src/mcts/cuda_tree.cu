#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <tuple>

#include "v0/cuda_tree.hpp"

namespace v0 {
namespace {

__global__ void CudaTreeSelectKernel(
    const float* __restrict__ priors,
    const bool* __restrict__ legal_mask,
    int32_t* __restrict__ child_index,
    const int32_t* __restrict__ node_visits,
    const double* __restrict__ node_value_sum,
    const int8_t* __restrict__ node_player,
    const bool* __restrict__ expanded,
    const bool* __restrict__ terminal,
    int32_t* __restrict__ node_parent,
    int32_t* __restrict__ node_action,
    int32_t* __restrict__ node_count,
    int64_t batch_size,
    int64_t max_nodes,
    int64_t path_capacity,
    int64_t action_dim,
    double exploration_weight,
    int32_t* __restrict__ leaf_nodes,
    int32_t* __restrict__ paths,
    int32_t* __restrict__ path_lengths,
    int32_t* __restrict__ selected_actions,
    bool* __restrict__ allocated,
    bool* __restrict__ overflow) {
    const int64_t tree = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tree >= batch_size) {
        return;
    }

    const int64_t node_base = tree * max_nodes;
    const int64_t edge_base = node_base * action_dim;
    const int64_t path_base = tree * path_capacity;
    int32_t node = 0;
    int32_t depth = 0;
    int32_t picked_action = -1;
    bool did_allocate = false;
    bool did_overflow = false;
    bool finished_selection = false;

    while (depth < path_capacity) {
        paths[path_base + depth] = node;
        ++depth;
        const int64_t global_node = node_base + node;
        if (terminal[global_node] || !expanded[global_node]) {
            // A terminal root is an inactive tree: portable full-MCTS skips it
            // instead of repeatedly backing up the already finished position.
            if (node == 0 && depth == 1 && terminal[global_node]) {
                depth = 0;
            }
            finished_selection = true;
            break;
        }

        const double sqrt_total = sqrt(static_cast<double>(max(1, node_visits[global_node])));
        double best_score = -std::numeric_limits<double>::infinity();
        int32_t best_action = -1;
        for (int32_t action = 0; action < action_dim; ++action) {
            const int64_t edge = edge_base + static_cast<int64_t>(node) * action_dim + action;
            if (!legal_mask[edge]) {
                continue;
            }
            const int32_t child = child_index[edge];
            int32_t visits = 0;
            double q = 0.0;
            if (child >= 0) {
                const int64_t global_child = node_base + child;
                visits = node_visits[global_child];
                if (visits > 0) {
                    q = node_value_sum[global_child] / static_cast<double>(visits);
                    if (node_player[global_node] != node_player[global_child]) {
                        q = -q;
                    }
                }
            }
            const double u = exploration_weight * static_cast<double>(priors[edge]) *
                sqrt_total / static_cast<double>(1 + visits);
            const double score = q + u;
            if (score > best_score ||
                (score == best_score && (best_action < 0 || action < best_action))) {
                best_score = score;
                best_action = action;
            }
        }

        if (best_action < 0) {
            finished_selection = true;
            break;
        }
        picked_action = best_action;
        const int64_t edge = edge_base + static_cast<int64_t>(node) * action_dim + best_action;
        int32_t child = child_index[edge];
        if (child < 0) {
            const int32_t next = node_count[tree];
            if (next >= max_nodes) {
                did_overflow = true;
                break;
            }
            child = next;
            node_count[tree] = next + 1;
            child_index[edge] = child;
            node_parent[node_base + child] = node;
            node_action[node_base + child] = best_action;
            did_allocate = true;
        }
        node = child;
        if (did_allocate) {
            if (depth < path_capacity) {
                paths[path_base + depth] = node;
                ++depth;
            } else {
                did_overflow = true;
            }
            finished_selection = true;
            break;
        }
    }
    if (!finished_selection) {
        did_overflow = true;
    }

    leaf_nodes[tree] = node;
    path_lengths[tree] = depth;
    selected_actions[tree] = picked_action;
    allocated[tree] = did_allocate;
    overflow[tree] = did_overflow;
}

__global__ void CudaTreeBackupKernel(
    const int32_t* __restrict__ paths,
    const int32_t* __restrict__ path_lengths,
    const int8_t* __restrict__ node_player,
    const double* __restrict__ leaf_values,
    int32_t* __restrict__ node_visits,
    double* __restrict__ node_value_sum,
    int64_t batch_size,
    int64_t path_capacity,
    int64_t arena_max_nodes) {
    const int64_t tree = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tree >= batch_size) {
        return;
    }
    const int32_t length = path_lengths[tree];
    if (length <= 0 || length > path_capacity) {
        return;
    }
    const double input_value = leaf_values[tree];
    if (!isfinite(input_value)) {
        return;
    }
    const int64_t path_base = tree * path_capacity;
    const int64_t node_base = tree * arena_max_nodes;
    double value = input_value;
    for (int32_t reverse = length; reverse > 0; --reverse) {
        const int32_t node = paths[path_base + reverse - 1];
        if (node < 0 || node >= arena_max_nodes) {
            return;
        }
        const int64_t global_node = node_base + node;
        node_visits[global_node] += 1;
        node_value_sum[global_node] += value;
        if (reverse > 1) {
            const int32_t parent = paths[path_base + reverse - 2];
            if (parent < 0 || parent >= arena_max_nodes) {
                return;
            }
            if (node_player[node_base + parent] != node_player[global_node]) {
                value = -value;
            }
        }
    }
}

void CheckCudaArenaTensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
cuda_tree_select(
    torch::Tensor priors,
    torch::Tensor legal_mask,
    torch::Tensor child_index,
    torch::Tensor node_visits,
    torch::Tensor node_value_sum,
    torch::Tensor node_player,
    torch::Tensor expanded,
    torch::Tensor terminal,
    torch::Tensor node_parent,
    torch::Tensor node_action,
    torch::Tensor node_count,
    double exploration_weight,
    int64_t max_path_length) {
    CheckCudaArenaTensor(priors, "priors");
#define CHECK_ARENA_INPUT(tensor)                                                \
    CheckCudaArenaTensor(tensor, #tensor);                                      \
    TORCH_CHECK(tensor.device() == priors.device(), #tensor " device mismatch")
    CHECK_ARENA_INPUT(legal_mask);
    CHECK_ARENA_INPUT(child_index);
    CHECK_ARENA_INPUT(node_visits);
    CHECK_ARENA_INPUT(node_value_sum);
    CHECK_ARENA_INPUT(node_player);
    CHECK_ARENA_INPUT(expanded);
    CHECK_ARENA_INPUT(terminal);
    CHECK_ARENA_INPUT(node_parent);
    CHECK_ARENA_INPUT(node_action);
    CHECK_ARENA_INPUT(node_count);
#undef CHECK_ARENA_INPUT
    TORCH_CHECK(priors.scalar_type() == torch::kFloat32, "priors must be float32");
    TORCH_CHECK(legal_mask.scalar_type() == torch::kBool, "legal_mask must be bool");
    TORCH_CHECK(child_index.scalar_type() == torch::kInt32, "child_index must be int32");
    TORCH_CHECK(node_visits.scalar_type() == torch::kInt32, "node_visits must be int32");
    TORCH_CHECK(node_value_sum.scalar_type() == torch::kFloat64, "node_value_sum must be float64");
    TORCH_CHECK(node_player.scalar_type() == torch::kInt8, "node_player must be int8");
    TORCH_CHECK(expanded.scalar_type() == torch::kBool, "expanded must be bool");
    TORCH_CHECK(terminal.scalar_type() == torch::kBool, "terminal must be bool");
    TORCH_CHECK(node_parent.scalar_type() == torch::kInt32, "node_parent must be int32");
    TORCH_CHECK(node_action.scalar_type() == torch::kInt32, "node_action must be int32");
    TORCH_CHECK(node_count.scalar_type() == torch::kInt32, "node_count must be int32");
    TORCH_CHECK(priors.dim() == 3, "priors must be [B, N, A]");
    const int64_t batch_size = priors.size(0);
    const int64_t max_nodes = priors.size(1);
    const int64_t action_dim = priors.size(2);
    const int64_t path_capacity = max_path_length > 0 ? max_path_length : max_nodes;
    TORCH_CHECK(max_nodes > 0 && action_dim > 0, "arena dimensions must be positive");
    TORCH_CHECK(path_capacity > 0 && path_capacity <= max_nodes,
                "max_path_length must be in [1, max_nodes]");
    TORCH_CHECK(legal_mask.sizes() == priors.sizes(), "legal_mask shape mismatch");
    TORCH_CHECK(child_index.sizes() == priors.sizes(), "child_index shape mismatch");
    const auto node_shape = std::vector<int64_t>{batch_size, max_nodes};
    TORCH_CHECK(node_visits.sizes().vec() == node_shape, "node_visits shape mismatch");
    TORCH_CHECK(node_value_sum.sizes().vec() == node_shape, "node_value_sum shape mismatch");
    TORCH_CHECK(node_player.sizes().vec() == node_shape, "node_player shape mismatch");
    TORCH_CHECK(expanded.sizes().vec() == node_shape, "expanded shape mismatch");
    TORCH_CHECK(terminal.sizes().vec() == node_shape, "terminal shape mismatch");
    TORCH_CHECK(node_parent.sizes().vec() == node_shape, "node_parent shape mismatch");
    TORCH_CHECK(node_action.sizes().vec() == node_shape, "node_action shape mismatch");
    TORCH_CHECK(node_count.sizes().vec() == std::vector<int64_t>{batch_size}, "node_count shape mismatch");
    TORCH_CHECK(std::isfinite(exploration_weight) && exploration_weight >= 0.0,
                "exploration_weight must be finite and non-negative");

    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(priors.device());
    auto bool_options = torch::TensorOptions().dtype(torch::kBool).device(priors.device());
    auto leaf_nodes = torch::zeros({batch_size}, int_options);
    auto paths = torch::full({batch_size, path_capacity}, -1, int_options);
    auto path_lengths = torch::zeros({batch_size}, int_options);
    auto selected_actions = torch::full({batch_size}, -1, int_options);
    auto allocated = torch::zeros({batch_size}, bool_options);
    auto overflow = torch::zeros({batch_size}, bool_options);
    if (batch_size == 0) {
        return std::make_tuple(
            leaf_nodes, paths, path_lengths, selected_actions, allocated, overflow);
    }

    at::cuda::OptionalCUDAGuard guard(priors.device());
    constexpr int threads = 128;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    CudaTreeSelectKernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        priors.data_ptr<float>(),
        legal_mask.data_ptr<bool>(),
        child_index.data_ptr<int32_t>(),
        node_visits.data_ptr<int32_t>(),
        node_value_sum.data_ptr<double>(),
        node_player.data_ptr<int8_t>(),
        expanded.data_ptr<bool>(),
        terminal.data_ptr<bool>(),
        node_parent.data_ptr<int32_t>(),
        node_action.data_ptr<int32_t>(),
        node_count.data_ptr<int32_t>(),
        batch_size,
        max_nodes,
        path_capacity,
        action_dim,
        exploration_weight,
        leaf_nodes.data_ptr<int32_t>(),
        paths.data_ptr<int32_t>(),
        path_lengths.data_ptr<int32_t>(),
        selected_actions.data_ptr<int32_t>(),
        allocated.data_ptr<bool>(),
        overflow.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(
        leaf_nodes, paths, path_lengths, selected_actions, allocated, overflow);
}

void cuda_tree_backup(
    torch::Tensor paths,
    torch::Tensor path_lengths,
    torch::Tensor node_player,
    torch::Tensor leaf_values,
    torch::Tensor node_visits,
    torch::Tensor node_value_sum) {
    CheckCudaArenaTensor(paths, "paths");
#define CHECK_BACKUP_INPUT(tensor)                                               \
    CheckCudaArenaTensor(tensor, #tensor);                                      \
    TORCH_CHECK(tensor.device() == paths.device(), #tensor " device mismatch")
    CHECK_BACKUP_INPUT(path_lengths);
    CHECK_BACKUP_INPUT(node_player);
    CHECK_BACKUP_INPUT(leaf_values);
    CHECK_BACKUP_INPUT(node_visits);
    CHECK_BACKUP_INPUT(node_value_sum);
#undef CHECK_BACKUP_INPUT
    TORCH_CHECK(paths.scalar_type() == torch::kInt32 && paths.dim() == 2,
                "paths must be int32 [B, N]");
    const int64_t batch_size = paths.size(0);
    const int64_t path_capacity = paths.size(1);
    TORCH_CHECK(path_lengths.scalar_type() == torch::kInt32 && path_lengths.numel() == batch_size,
                "path_lengths must be int32 [B]");
    TORCH_CHECK(node_player.scalar_type() == torch::kInt8 && node_player.dim() == 2 &&
                node_player.size(0) == batch_size,
                "node_player must be int8 [B, arena_nodes]");
    const int64_t arena_max_nodes = node_player.size(1);
    TORCH_CHECK(leaf_values.scalar_type() == torch::kFloat64 && leaf_values.numel() == batch_size,
                "leaf_values must be float64 [B]");
    TORCH_CHECK(node_visits.scalar_type() == torch::kInt32 &&
                node_visits.sizes() == node_player.sizes(),
                "node_visits must be int32 [B, arena_nodes]");
    TORCH_CHECK(node_value_sum.scalar_type() == torch::kFloat64 &&
                node_value_sum.sizes() == node_player.sizes(),
                "node_value_sum must be float64 [B, arena_nodes]");
    if (batch_size == 0) {
        return;
    }
    at::cuda::OptionalCUDAGuard guard(paths.device());
    constexpr int threads = 128;
    const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
    CudaTreeBackupKernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        paths.data_ptr<int32_t>(),
        path_lengths.data_ptr<int32_t>(),
        node_player.data_ptr<int8_t>(),
        leaf_values.data_ptr<double>(),
        node_visits.data_ptr<int32_t>(),
        node_value_sum.data_ptr<double>(),
        batch_size,
        path_capacity,
        arena_max_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace v0
