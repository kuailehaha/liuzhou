#pragma once

#include <tuple>

#include <torch/extension.h>

namespace v0 {

// Select one leaf per tree from a dense GPU-resident arena.  The arena uses
// node 0 as the root of every tree and allocates at most one lazy child during
// each call.  All tensors remain on CUDA; no host-visible counts are needed.
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
    double exploration_weight);

// Back up leaf values along paths produced by cuda_tree_select.  Values are
// interpreted from the selected leaf's side-to-move perspective and are only
// negated across an edge whose side-to-move changes.
void cuda_tree_backup(
    torch::Tensor paths,
    torch::Tensor path_lengths,
    torch::Tensor node_player,
    torch::Tensor leaf_values,
    torch::Tensor node_visits,
    torch::Tensor node_value_sum);

}  // namespace v0
