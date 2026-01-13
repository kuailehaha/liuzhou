#pragma once

#include <memory>
#include <string>
#include <tuple>

#include <torch/extension.h>
#include <torch/script.h>

#if defined(TORCH_CUDA_AVAILABLE)
#include <ATen/cuda/CUDAGraph.h>
#endif

namespace v0 {

class InferenceEngine {
   public:
    InferenceEngine(
        const std::string& model_path,
        const std::string& device,
        const std::string& dtype,
        int64_t batch_size,
        int64_t input_channels,
        int64_t height,
        int64_t width,
        int64_t warmup_iters,
        bool use_inference_mode = true);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Forward(
        const torch::Tensor& input,
        int64_t n_valid);

    int64_t BatchSize() const { return batch_size_; }
    bool GraphEnabled() const { return graph_enabled_; }
    std::string DeviceString() const;
    std::string DTypeString() const;

   private:
    torch::Tensor PrepareInput(const torch::Tensor& input, int64_t n_valid);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> RunForward();

    torch::jit::Module module_;
    torch::Device device_{torch::kCPU};
    torch::ScalarType dtype_{torch::kFloat32};
    bool use_inference_mode_{true};
    int64_t batch_size_{0};
    int64_t input_channels_{0};
    int64_t height_{0};
    int64_t width_{0};
    bool graph_enabled_{false};

    torch::Tensor input_buf_;
    torch::Tensor out_p1_;
    torch::Tensor out_p2_;
    torch::Tensor out_pmc_;
    torch::Tensor out_value_;

#if defined(TORCH_CUDA_AVAILABLE)
    std::unique_ptr<at::cuda::CUDAGraph> graph_;
#endif
};

}  // namespace v0
