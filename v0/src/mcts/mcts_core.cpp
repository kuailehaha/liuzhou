#include "v0/mcts_core.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>

#include <pybind11/pybind11.h>

#include "v0/fast_apply_moves.hpp"
#include "v0/fast_legal_mask.hpp"
#include "v0/net_encoding.hpp"
#include "v0/project_policy.hpp"

namespace v0 {

namespace py = pybind11;

namespace {

bool DebugEnabled() {
    static bool enabled = []() {
        const char* env = std::getenv("V0_MCTS_DEBUG");
        if (!env) {
            return false;
        }
        if (env[0] == '\0') {
            return false;
        }
        if (env[0] == '0' && env[1] == '\0') {
            return false;
        }
        return true;
    }();
    return enabled;
}

void DebugLog(const std::string& msg) {
    if (DebugEnabled()) {
        std::cerr << "[MCTSCore] " << msg << std::endl;
    }
}

std::string ShapeToString(const torch::Tensor& t) {
    std::ostringstream oss;
    oss << "(";
    for (int64_t i = 0; i < t.dim(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << t.size(i);
    }
    oss << ")";
    return oss.str();
}

torch::TensorOptions BoolCPU() {
    return torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
}

torch::TensorOptions IntCPU() {
    return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
}

torch::TensorOptions LongCPU() {
    return torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
}

// double WinnerValue(const GameState& state, Player to_act) {
//     auto winner = state.GetWinner();
//     if (!winner.has_value()) {
//         return 0.0;
//     }
//     return winner == to_act ? 1.0 : -1.0;
// }

double WinnerValue(const GameState& state, Player to_act) {
    const auto w = state.GetWinner();           // or state.Winner()
    if (!w.has_value()) return 0.0;
    return (*w == to_act) ? 1.0 : -1.0;
}

}  // namespace

namespace {

constexpr int kPlacementDim = kCellCount;
constexpr int kMovementDim = kCellCount * 4;
constexpr int kSelectionDim = kCellCount;
constexpr int kAuxiliaryDim = 4;

}  // namespace

MCTSCore::MCTSCore(MCTSConfig config)
    : config_(std::move(config)), rng_(config_.seed) {}

void MCTSCore::SetForwardCallback(ForwardCallback cb) {
    forward_cb_ = std::move(cb);
}

void MCTSCore::Reset() {
    nodes_.clear();
    root_index_ = -1;
}

void MCTSCore::SetRootState(const GameState& state) {
    Reset();
    try {
        root_index_ = AllocateNode(state);
    } catch (const std::exception& err) {
        DebugLog(std::string("SetRootState: AllocateNode failed: ") + err.what());
        throw;
    }
}

int MCTSCore::AllocateNode(const GameState& state) {
    nodes_.push_back(Node{state});
    Node& node = nodes_.back();
    node.parent = -1;
    node.action_index = -1;
    node.children.clear();
    node.prior = 1.0;
    node.value_sum = 0.0;
    node.visit_count = 0.0;
    node.virtual_loss = 0.0;
    node.is_expanded = false;
    node.is_terminal = state.IsGameOver();
    node.terminal_value = WinnerValue(state, state.current_player);
    return static_cast<int>(nodes_.size() - 1);
}

const GameState& MCTSCore::RootState() const {
    return nodes_.at(root_index_).state;
}

double MCTSCore::RootValue() const {
    if (root_index_ < 0) {
        return 0.0;
    }
    const Node& root = nodes_[root_index_];
    if (root.visit_count <= 0) {
        return 0.0;
    }
    return root.value_sum / root.visit_count;
}

std::vector<int> MCTSCore::SelectPath() {
    std::vector<int> path;
    if (root_index_ < 0) {
        return path;
    }
    int node_idx = root_index_;
    path.push_back(node_idx);
    while (true) {
        Node& node = nodes_[node_idx];
        if (node.is_terminal || !node.is_expanded || node.children.empty()) {
            return path;
        }
        double sqrt_total = std::sqrt(std::max(1.0, node.visit_count));
        double best_score = -std::numeric_limits<double>::infinity();
        int best_child = -1;
        for (int child_idx : node.children) {
            Node& child = nodes_[child_idx];
            double q = child.visit_count > 0 ? (child.value_sum / child.visit_count) : 0.0;
            double u = config_.exploration_weight * child.prior * sqrt_total / (1.0 + child.visit_count);
            double score = q + u;
            if (score > best_score) {
                best_score = score;
                best_child = child_idx;
            }
        }
        if (best_child < 0) {
            return path;
        }
        node_idx = best_child;
        path.push_back(node_idx);
    }
}

void MCTSCore::ApplyVirtualLoss(const std::vector<int>& path) {
    for (int idx : path) {
        Node& node = nodes_[idx];
        node.virtual_loss += config_.virtual_loss;
        node.visit_count += config_.virtual_loss;
        node.value_sum -= config_.virtual_loss;
    }
}

void MCTSCore::RevertVirtualLoss(const std::vector<int>& path) {
    for (int idx : path) {
        Node& node = nodes_[idx];
        node.virtual_loss -= config_.virtual_loss;
        node.visit_count = std::max(0.0, node.visit_count - config_.virtual_loss);
        node.value_sum += config_.virtual_loss;
    }
}

void MCTSCore::Backpropagate(const std::vector<int>& path, double value) {
    double current = value;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Node& node = nodes_[*it];
        node.visit_count += 1.0;
        node.value_sum += current;
        current = -current;
    }
}

torch::Tensor MCTSCore::BuildModelInputs(const TensorStateBatch& batch) {
    return states_to_model_input(
        batch.board,
        batch.marks_black,
        batch.marks_white,
        batch.phase,
        batch.current_player);
}

std::vector<double> MCTSCore::SampleDirichlet(int count, double alpha) {
    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> samples(count);
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        double val = gamma(rng_);
        samples[i] = val;
        sum += val;
    }
    if (sum <= 0) {
        return std::vector<double>(count, 1.0 / count);
    }
    for (double& v : samples) {
        v /= sum;
    }
    return samples;
}

void MCTSCore::ExpandBatch(const std::vector<int>& leaves, const std::vector<std::vector<int>>& paths) {
    try {
        if (leaves.empty()) {
            return;
        }
        DebugLog("ExpandBatch start: leaves=" + std::to_string(leaves.size()) +
            ", nodes=" + std::to_string(nodes_.size()));

    std::vector<GameState> eval_states;
    eval_states.reserve(leaves.size());
        for (int idx : leaves) {
            if (idx < 0 || idx >= static_cast<int>(nodes_.size())) {
                std::ostringstream oss;
                oss << "ExpandBatch: invalid leaf index " << idx
                    << " with nodes_.size()=" << nodes_.size();
                DebugLog(oss.str());
                throw std::runtime_error(oss.str());
            }
            eval_states.push_back(nodes_[idx].state);
        }
        if (eval_states.size() != leaves.size()) {
            std::ostringstream oss;
            oss << "ExpandBatch: eval_states mismatch leaves=" << leaves.size()
                << " actual=" << eval_states.size();
            DebugLog(oss.str());
            throw std::runtime_error(oss.str());
        }

        DebugLog("ExpandBatch: converting states to tensors (CPU), eval_states=" + std::to_string(eval_states.size()));
        TensorStateBatch batch_cpu;
        try {
            batch_cpu = FromGameStates(eval_states, torch::Device(torch::kCPU));
        } catch (const std::exception& e) {
            DebugLog(std::string("ExpandBatch: FromGameStates failed: ") + e.what());
            throw;
        }
        if (batch_cpu.board.size(0) != static_cast<int64_t>(leaves.size())) {
            std::ostringstream oss;
            oss << "ExpandBatch: tensor batch size mismatch, expected "
                << leaves.size() << " got " << batch_cpu.board.size(0);
            DebugLog(oss.str());
            throw std::runtime_error(oss.str());
        }
        DebugLog("ExpandBatch: batch_cpu board shape " + ShapeToString(batch_cpu.board));
        TensorStateBatch batch_device =
            config_.device.is_cpu() ? batch_cpu : batch_cpu.To(config_.device);
        torch::Tensor inputs = BuildModelInputs(batch_device);
        if (!forward_cb_) {
            throw std::runtime_error("Forward callback not set for MCTSCore.");
        }
        DebugLog("ExpandBatch: inputs shape " + ShapeToString(inputs));

        torch::Tensor log_p1, log_p2, log_pmc, values;
        {
            py::gil_scoped_acquire gil;
            std::tie(log_p1, log_p2, log_pmc, values) = forward_cb_(inputs);
        }
        DebugLog("ExpandBatch: NN outputs log_p shape " + ShapeToString(log_p1) +
            ", values shape " + ShapeToString(values));

        auto [legal_mask_device, metadata_device] = encode_actions_fast(
            batch_device.board,
            batch_device.marks_black.to(torch::kBool),
            batch_device.marks_white.to(torch::kBool),
            batch_device.phase,
            batch_device.current_player,
            batch_device.pending_marks_required,
            batch_device.pending_marks_remaining,
            batch_device.pending_captures_required,
            batch_device.pending_captures_remaining,
            batch_device.forced_removals_done,
            kPlacementDim,
            kMovementDim,
            kSelectionDim,
            kAuxiliaryDim);
        DebugLog("ExpandBatch: legal mask shape " + ShapeToString(legal_mask_device));

        auto [probs, _] = project_policy_logits_fast(
            log_p1,
            log_p2,
            log_pmc,
            legal_mask_device,
            kPlacementDim,
            kMovementDim,
            kSelectionDim,
            kAuxiliaryDim);
        auto probs_cpu = probs.to(torch::kCPU);
        auto values_cpu = values.to(torch::kCPU);
        auto legal_mask_cpu = legal_mask_device.to(torch::kCPU);
        auto metadata_cpu = metadata_device.to(torch::kCPU);
        DebugLog("ExpandBatch: projected policy shape " + ShapeToString(probs_cpu));

    std::vector<int> parent_indices;
    std::vector<std::array<int32_t, 4>> action_codes;
    std::vector<int> child_counts(leaves.size(), 0);
    std::vector<std::vector<int>> action_indices(leaves.size());
    std::vector<std::vector<double>> priors(leaves.size());

    int total_actions = 0;
    for (size_t bi = 0; bi < leaves.size(); ++bi) {
        auto mask_row = legal_mask_cpu[static_cast<int64_t>(bi)];
        auto legal_indices = torch::nonzero(mask_row).view(-1);
        if (legal_indices.size(0) == 0) {
            child_counts[bi] = 0;
            continue;
        }
        auto legal_indices_cpu = legal_indices.to(torch::kCPU);
        torch::Tensor meta_row = metadata_cpu[static_cast<int64_t>(bi)];
        torch::Tensor probs_row = probs_cpu[static_cast<int64_t>(bi)];

        std::vector<std::pair<int, int>> indexed;
        indexed.reserve(legal_indices_cpu.size(0));
        for (int64_t j = 0; j < legal_indices_cpu.size(0); ++j) {
            int action_idx = static_cast<int>(legal_indices_cpu[j].item<int64_t>());
            indexed.emplace_back(action_idx, static_cast<int>(j));
        }
        std::sort(indexed.begin(), indexed.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        child_counts[bi] = static_cast<int>(indexed.size());
        action_indices[bi].reserve(indexed.size());
        priors[bi].reserve(indexed.size());

        for (const auto& [action_idx, meta_idx] : indexed) {
            torch::Tensor meta = meta_row[legal_indices_cpu[meta_idx]];
            std::array<int32_t, 4> code{};
            for (int k = 0; k < 4; ++k) {
                code[k] = static_cast<int32_t>(meta[k].item<int64_t>());
            }
            action_codes.push_back(code);
            parent_indices.push_back(static_cast<int>(bi));

            action_indices[bi].push_back(action_idx);
            priors[bi].push_back(probs_row[action_idx].item<double>());
            total_actions += 1;
        }
    }

        std::vector<GameState> child_states;
        if (!action_codes.empty()) {
            torch::Tensor action_tensor = torch::empty({static_cast<int64_t>(action_codes.size()), 4}, IntCPU());
            auto action_ptr = action_tensor.data_ptr<int32_t>();
            for (size_t i = 0; i < action_codes.size(); ++i) {
                for (int k = 0; k < 4; ++k) {
                    action_ptr[i * 4 + k] = action_codes[i][k];
                }
            }
            torch::Tensor parent_tensor = torch::empty(
                {static_cast<int64_t>(parent_indices.size())}, LongCPU());
            auto parent_ptr = parent_tensor.data_ptr<int64_t>();
            for (size_t i = 0; i < parent_indices.size(); ++i) {
                parent_ptr[i] = parent_indices[i];
            }

            DebugLog("ExpandBatch: applying moves for total_actions=" + std::to_string(action_codes.size()));
            const auto tensor_device = batch_device.board.device();
            torch::Tensor action_tensor_device =
                tensor_device.is_cpu() ? action_tensor : action_tensor.to(tensor_device);
            torch::Tensor parent_tensor_device =
                tensor_device.is_cpu() ? parent_tensor : parent_tensor.to(tensor_device);
            auto next_batch = batch_apply_moves(
                batch_device.board,
                batch_device.marks_black,
                batch_device.marks_white,
                batch_device.phase,
                batch_device.current_player,
                batch_device.pending_marks_required,
                batch_device.pending_marks_remaining,
                batch_device.pending_captures_required,
                batch_device.pending_captures_remaining,
                batch_device.forced_removals_done,
                batch_device.move_count,
                action_tensor_device,
                parent_tensor_device);

            TensorStateBatch child_batch{
                std::get<0>(next_batch),
                std::get<1>(next_batch).to(torch::kBool),
                std::get<2>(next_batch).to(torch::kBool),
                std::get<3>(next_batch),
                std::get<4>(next_batch),
                std::get<5>(next_batch),
                std::get<6>(next_batch),
                std::get<7>(next_batch),
                std::get<8>(next_batch),
                std::get<9>(next_batch),
                std::get<10>(next_batch),
                torch::ones(std::get<0>(next_batch).size(0), BoolCPU()),
                batch_cpu.board_size};

            child_states = ToGameStates(child_batch);
            DebugLog("ExpandBatch: child_states size=" + std::to_string(child_states.size()));
        }

        DebugLog("ExpandBatch prepared actions=" + std::to_string(total_actions));

        size_t state_cursor = 0;
        for (size_t bi = 0; bi < leaves.size(); ++bi) {
            int node_idx = leaves[bi];
            RevertVirtualLoss(paths[bi]);

            // torch::Scalar value_scalar = values_cpu[static_cast<int64_t>(bi)];
            // double leaf_value = value_scalar.item<double>();
            double leaf_value = values_cpu[static_cast<int64_t>(bi)].item<double>();

            torch::Tensor mask_row = legal_mask_cpu[static_cast<int64_t>(bi)];
            auto legal_indices = torch::nonzero(mask_row).view(-1);
            if (legal_indices.size(0) == 0) {
                nodes_[node_idx].is_terminal = true;
                nodes_[node_idx].terminal_value = -1.0;
                Backpropagate(paths[bi], -1.0);
                continue;
            }

            nodes_[node_idx].is_expanded = true;
            nodes_[node_idx].children.clear();

            std::vector<double> node_priors = priors[bi];
            int parent_idx = nodes_[node_idx].parent;
            if (config_.add_dirichlet_noise && parent_idx == -1 && node_priors.size() > 1) {
                auto noise = SampleDirichlet(static_cast<int>(node_priors.size()), config_.dirichlet_alpha);
                for (size_t i = 0; i < node_priors.size(); ++i) {
                    node_priors[i] = (1.0 - config_.dirichlet_epsilon) * node_priors[i] +
                    config_.dirichlet_epsilon * noise[i];
            }
        }

        double prior_sum = 0.0;
        for (double p : node_priors) {
            prior_sum += p;
        }
        if (prior_sum <= 0) {
            for (double& p : node_priors) {
                p = 1.0 / node_priors.size();
            }
        } else {
            double inv = 1.0 / prior_sum;
            for (double& p : node_priors) {
                p *= inv;
            }
            }

            for (int j = 0; j < child_counts[bi]; ++j) {
                GameState child_state = child_states[state_cursor++];
                int child_idx = AllocateNode(child_state);
                Node& child = nodes_[child_idx];
                child.parent = node_idx;
                child.action_index = action_indices[bi][j];
                child.prior = node_priors[j];
                nodes_[node_idx].children.push_back(child_idx);
            }

            auto& children = nodes_[node_idx].children;
            std::sort(children.begin(), children.end(), [&](int lhs, int rhs) {
                return nodes_[lhs].action_index < nodes_[rhs].action_index;
            });

        Backpropagate(paths[bi], leaf_value);
    }
    } catch (const c10::Error& err) {
        std::ostringstream oss;
        oss << "MCTSCore::ExpandBatch c10::Error (leaves=" << leaves.size()
            << ", batch_size=" << config_.batch_size << "): " << err.what();
        DebugLog(oss.str());
        throw std::runtime_error(oss.str());
    } catch (const std::exception& err) {
        std::ostringstream oss;
        oss << "MCTSCore::ExpandBatch std::exception (leaves=" << leaves.size()
            << ", batch_size=" << config_.batch_size << "): " << err.what();
        DebugLog(oss.str());
        throw std::runtime_error(oss.str());
    }
}

void MCTSCore::RunSimulations(int num_simulations) {
    if (root_index_ < 0) {
        throw std::runtime_error("MCTS root not set.");
    }
    DebugLog("RunSimulations start: num_simulations=" + std::to_string(num_simulations) +
        ", batch_size=" + std::to_string(config_.batch_size));
    for (int sim = 0; sim < num_simulations; ) {
        if (DebugEnabled() && sim % 16 == 0) {
            DebugLog("RunSimulations progress: sim=" + std::to_string(sim) + "/" +
                std::to_string(num_simulations));
        }
        std::vector<int> leaves;
        std::vector<std::vector<int>> paths;

        for (int batch = 0; batch < config_.batch_size && sim < num_simulations; ++batch) {
            std::vector<int> path = SelectPath();
            if (path.empty()) {
                continue;
            }
            int leaf = path.back();
            Node& node = nodes_[leaf];
            if (node.is_terminal) {
                Backpropagate(path, node.terminal_value);
                ++sim;
                continue;
            }
            ApplyVirtualLoss(path);
            leaves.push_back(leaf);
            paths.push_back(path);
            ++sim;
        }

        if (!leaves.empty()) {
            DebugLog("RunSimulations expanding batch with leaves=" + std::to_string(leaves.size()));
            try {
                ExpandBatch(leaves, paths);
            } catch (const std::exception& err) {
                DebugLog(std::string("RunSimulations: ExpandBatch failed: ") + err.what());
                throw;
            }
        }
    }
}

std::vector<std::pair<int, double>> MCTSCore::GetPolicy(double temperature) const {
    std::vector<std::pair<int, double>> result;
    if (root_index_ < 0) {
        return result;
    }
    const Node& root = nodes_[root_index_];
    if (root.children.empty()) {
        return result;
    }

    std::vector<double> visits;
    visits.reserve(root.children.size());
    for (int child_idx : root.children) {
        const Node& child = nodes_[child_idx];
        visits.push_back(child.visit_count);
    }

    double temp = std::max(temperature, 1e-6);
    std::vector<double> scaled(visits.size());
    if (temp <= 1e-6) {
        int best = static_cast<int>(std::distance(
            visits.begin(), std::max_element(visits.begin(), visits.end())));
        scaled.assign(visits.size(), 0.0);
        if (!visits.empty()) {
            scaled[best] = 1.0;
        }
    } else {
        double sum = 0.0;
        for (size_t i = 0; i < visits.size(); ++i) {
            scaled[i] = std::pow(visits[i], 1.0 / temp);
            sum += scaled[i];
        }
        if (sum <= 0) {
            double uniform = 1.0 / visits.size();
            for (double& v : scaled) {
                v = uniform;
            }
        } else {
            double inv = 1.0 / sum;
            for (double& v : scaled) {
                v *= inv;
            }
        }
    }

    for (size_t i = 0; i < root.children.size(); ++i) {
        result.emplace_back(nodes_[root.children[i]].action_index, scaled[i]);
    }
    return result;
}

void MCTSCore::AdvanceRoot(int action_index) {
    if (root_index_ < 0) {
        return;
    }
    Node& root = nodes_[root_index_];
    for (int child_idx : root.children) {
        if (nodes_[child_idx].action_index == action_index) {
            nodes_[child_idx].parent = -1;
            root_index_ = child_idx;
            return;
        }
    }
    Reset();
}

}  // namespace v0
