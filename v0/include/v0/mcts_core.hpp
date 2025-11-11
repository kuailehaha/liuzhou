#pragma once

#include <functional>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "v0/game_state.hpp"
#include "v0/tensor_state_batch.hpp"

namespace v0 {

struct MCTSConfig {
    int num_simulations{800};
    double exploration_weight{1.0};
    double temperature{1.0};
    bool add_dirichlet_noise{false};
    double dirichlet_alpha{0.3};
    double dirichlet_epsilon{0.25};
    int batch_size{16};
    double virtual_loss{1.0};
    torch::Device device{torch::kCPU};
    uint64_t seed{0xC0FFEE};
};

using ForwardCallback = std::function<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>;

class MCTSCore {
   public:
    explicit MCTSCore(MCTSConfig config);

    void SetForwardCallback(ForwardCallback cb);

    void SetRootState(const GameState& state);
    void Reset();

    void RunSimulations(int num_simulations);

    std::vector<std::pair<int, double>> GetPolicy(double temperature) const;

    void AdvanceRoot(int action_index);

    double RootValue() const;

    const GameState& RootState() const;

   private:
    struct Node {
        GameState state;
        int parent{-1};
        int action_index{-1};
        std::vector<int> children;
        double prior{0.0};
        double value_sum{0.0};
        double visit_count{0.0};
        double virtual_loss{0.0};
        bool is_expanded{false};
        bool is_terminal{false};
        double terminal_value{0.0};
    };

    int AllocateNode(const GameState& state);
    int RootIndex() const { return root_index_; }

    std::vector<int> SelectPath();
    void ApplyVirtualLoss(const std::vector<int>& path);
    void RevertVirtualLoss(const std::vector<int>& path);
    void Backpropagate(const std::vector<int>& path, double value);

    void ExpandBatch(const std::vector<int>& leaves, const std::vector<std::vector<int>>& paths);

    std::vector<double> SampleDirichlet(int count, double alpha);

    torch::Tensor BuildModelInputs(const TensorStateBatch& batch);

    MCTSConfig config_;
    ForwardCallback forward_cb_;
    std::vector<Node> nodes_;
    int root_index_{-1};
    std::mt19937 rng_;
};

}  // namespace v0
