#include "v0/eval_batcher.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <sstream>
#include <stdexcept>

namespace v0 {

namespace {

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

}  // namespace

EvalBatcher::EvalBatcher(
    std::shared_ptr<InferenceEngine> engine,
    int64_t batch_size,
    int64_t input_channels,
    int64_t height,
    int64_t width,
    int64_t timeout_ms)
    : engine_(std::move(engine)),
      batch_size_(batch_size),
      input_channels_(input_channels),
      height_(height),
      width_(width),
      timeout_ms_(timeout_ms) {
    if (!engine_) {
        throw std::runtime_error("EvalBatcher requires a valid InferenceEngine.");
    }
    if (batch_size_ <= 0) {
        throw std::runtime_error("EvalBatcher batch_size must be positive.");
    }
    if (input_channels_ <= 0 || height_ <= 0 || width_ <= 0) {
        throw std::runtime_error("EvalBatcher input shape must be positive.");
    }
    if (engine_->BatchSize() != batch_size_) {
        std::ostringstream oss;
        oss << "EvalBatcher batch_size mismatch: engine=" << engine_->BatchSize()
            << " batcher=" << batch_size_;
        throw std::runtime_error(oss.str());
    }

    device_ = engine_->Device();
    dtype_ = engine_->DType();
    worker_ = std::thread([this]() { WorkerLoop(); });
}

EvalBatcher::~EvalBatcher() {
    Shutdown();
}

void EvalBatcher::Shutdown() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true)) {
        return;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void EvalBatcher::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = EvalStats{};
}

EvalBatcher::EvalStats EvalBatcher::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> EvalBatcher::Forward(
    const torch::Tensor& input,
    int64_t n_valid) {
    if (stop_.load()) {
        throw std::runtime_error("EvalBatcher is shut down.");
    }
    auto request = std::make_shared<Request>();
    request->input = input;
    request->n_valid = n_valid;
    auto future = request->promise.get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(request);
    }
    cv_.notify_one();
    return future.get();
}

int EvalBatcher::EvalHistBucket(int64_t n_valid) const {
    if (n_valid <= 0) {
        return 0;
    }
    if (n_valid > batch_size_) {
        return EvalStats::kHistBuckets - 1;
    }
    int64_t bucket_width = std::max<int64_t>(1, batch_size_ / (EvalStats::kHistBuckets - 1));
    int bucket = static_cast<int>((n_valid - 1) / bucket_width);
    if (bucket < 0) {
        bucket = 0;
    }
    if (bucket >= EvalStats::kHistBuckets) {
        bucket = EvalStats::kHistBuckets - 1;
    }
    return bucket;
}

void EvalBatcher::UpdateStats(int64_t n_valid) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.eval_calls += 1;
    stats_.eval_leaves += static_cast<uint64_t>(n_valid);
    if (n_valid == batch_size_) {
        stats_.full512_calls += 1;
    }
    int bucket = EvalHistBucket(n_valid);
    stats_.hist[static_cast<size_t>(bucket)] += 1;
}

void EvalBatcher::WorkerLoop() {
    while (true) {
        std::vector<std::shared_ptr<Request>> batch;
        int64_t total = 0;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stop_.load() || !queue_.empty(); });
            if (stop_.load() && queue_.empty()) {
                break;
            }

            auto deadline = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(std::max<int64_t>(0, timeout_ms_));

            while (total < batch_size_) {
                bool reached_limit = false;
                while (!queue_.empty() && total < batch_size_) {
                    auto request = queue_.front();
                    queue_.pop_front();

                    int64_t n_valid = request->n_valid;
                    if (n_valid <= 0) {
                        n_valid = request->input.size(0);
                    }
                    request->n_valid = n_valid;

                    if (n_valid <= 0 || n_valid > batch_size_) {
                        request->promise.set_exception(
                            std::make_exception_ptr(std::runtime_error("EvalBatcher n_valid out of range.")));
                        continue;
                    }
                    if (request->input.dim() != 4) {
                        request->promise.set_exception(
                            std::make_exception_ptr(std::runtime_error("EvalBatcher input must be 4D.")));
                        continue;
                    }
                    if (request->input.size(1) != input_channels_ ||
                        request->input.size(2) != height_ ||
                        request->input.size(3) != width_) {
                        std::ostringstream oss;
                        oss << "EvalBatcher input shape mismatch, expected (B, "
                            << input_channels_ << ", " << height_ << ", " << width_
                            << ") got " << ShapeToString(request->input);
                        request->promise.set_exception(
                            std::make_exception_ptr(std::runtime_error(oss.str())));
                        continue;
                    }
                    if (request->input.size(0) < n_valid) {
                        request->promise.set_exception(
                            std::make_exception_ptr(std::runtime_error("EvalBatcher input batch smaller than n_valid.")));
                        continue;
                    }

                    if (total + n_valid > batch_size_) {
                        queue_.push_front(request);
                        reached_limit = true;
                        break;
                    }

                    batch.push_back(request);
                    total += n_valid;
                }

                if (total >= batch_size_ || reached_limit || timeout_ms_ <= 0) {
                    break;
                }
                if (queue_.empty()) {
                    if (cv_.wait_until(lock, deadline, [this]() { return stop_.load() || !queue_.empty(); })) {
                        if (stop_.load() && queue_.empty()) {
                            break;
                        }
                        continue;
                    }
                    break;
                }
            }
        }

        if (batch.empty()) {
            if (stop_.load()) {
                break;
            }
            continue;
        }

        if (!input_buf_.defined()) {
            auto options = torch::TensorOptions().device(device_).dtype(dtype_);
            input_buf_ = torch::zeros({batch_size_, input_channels_, height_, width_}, options);
        }

        input_buf_.zero_();

        std::vector<bool> failed(batch.size(), false);
        int64_t offset = 0;
        for (size_t i = 0; i < batch.size(); ++i) {
            auto& request = batch[i];
            try {
                torch::Tensor src = request->input;
                if (src.device() != device_ || src.scalar_type() != dtype_) {
                    src = src.to(device_, dtype_, /*non_blocking=*/true);
                }
                if (!src.is_contiguous()) {
                    src = src.contiguous();
                }
                input_buf_.narrow(0, offset, request->n_valid)
                    .copy_(src.narrow(0, 0, request->n_valid));
            } catch (const std::exception&) {
                request->promise.set_exception(std::current_exception());
                failed[i] = true;
            }
            offset += request->n_valid;
        }

        try {
            auto outputs = engine_->Forward(input_buf_, total);
            UpdateStats(total);
            int64_t out_offset = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                auto& request = batch[i];
                if (failed[i]) {
                    out_offset += request->n_valid;
                    continue;
                }
                auto log_p1 = std::get<0>(outputs).narrow(0, out_offset, request->n_valid).clone();
                auto log_p2 = std::get<1>(outputs).narrow(0, out_offset, request->n_valid).clone();
                auto log_pmc = std::get<2>(outputs).narrow(0, out_offset, request->n_valid).clone();
                auto values = std::get<3>(outputs).narrow(0, out_offset, request->n_valid).clone();
                request->promise.set_value(std::make_tuple(log_p1, log_p2, log_pmc, values));
                out_offset += request->n_valid;
            }
        } catch (const std::exception&) {
            for (size_t i = 0; i < batch.size(); ++i) {
                if (failed[i]) {
                    continue;
                }
                batch[i]->promise.set_exception(std::current_exception());
            }
        }
    }
}

}  // namespace v0
