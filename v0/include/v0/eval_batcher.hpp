#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "v0/inference_engine.hpp"

namespace v0 {

class EvalBatcher {
   public:
    struct EvalStats {
        static constexpr int kHistBuckets = 17;
        uint64_t eval_calls{0};
        uint64_t eval_leaves{0};
        uint64_t full512_calls{0};
        std::array<uint64_t, kHistBuckets> hist{};
    };

    EvalBatcher(
        std::shared_ptr<InferenceEngine> engine,
        int64_t batch_size,
        int64_t input_channels,
        int64_t height,
        int64_t width,
        int64_t timeout_ms = 2);

    ~EvalBatcher();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Forward(
        const torch::Tensor& input,
        int64_t n_valid);

    void Shutdown();

    void ResetStats();
    EvalStats GetStats() const;

    int64_t BatchSize() const { return batch_size_; }
    int64_t TimeoutMs() const { return timeout_ms_; }

   private:
    using OutputTuple = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

    struct Request {
        torch::Tensor input;
        int64_t n_valid{0};
        std::promise<OutputTuple> promise;
    };

    void WorkerLoop();
    void UpdateStats(int64_t n_valid);
    int EvalHistBucket(int64_t n_valid) const;

    std::shared_ptr<InferenceEngine> engine_;
    int64_t batch_size_{0};
    int64_t input_channels_{0};
    int64_t height_{0};
    int64_t width_{0};
    int64_t timeout_ms_{0};

    torch::Device device_{torch::kCPU};
    torch::ScalarType dtype_{torch::kFloat32};
    torch::Tensor input_buf_;

    std::deque<std::shared_ptr<Request>> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    std::atomic<bool> stop_{false};

    mutable std::mutex stats_mutex_;
    EvalStats stats_;
};

}  // namespace v0
