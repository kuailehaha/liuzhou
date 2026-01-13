#include "v0/inference_engine.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <c10/core/InferenceMode.h>

#if defined(TORCH_CUDA_AVAILABLE)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>
#endif

namespace v0 {

namespace {

torch::Device ParseDevice(const std::string& device) {
    return torch::Device(device);
}

torch::ScalarType ParseExplicitDType(const std::string& dtype) {
    if (dtype == "float32" || dtype == "fp32" || dtype == "f32") {
        return torch::kFloat32;
    }
    if (dtype == "float16" || dtype == "fp16" || dtype == "f16") {
        return torch::kFloat16;
    }
    if (dtype == "bfloat16" || dtype == "bf16") {
        return torch::kBFloat16;
    }
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

std::string DTypeToString(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32:
            return "float32";
        case torch::kFloat16:
            return "float16";
        case torch::kBFloat16:
            return "bfloat16";
        default:
            return "unknown";
    }
}

torch::ScalarType InferModuleDType(const torch::jit::Module& module, torch::ScalarType fallback) {
    for (const auto& param : module.parameters()) {
        return param.scalar_type();
    }
    return fallback;
}

}  // namespace

InferenceEngine::InferenceEngine(
    const std::string& model_path,
    const std::string& device,
    const std::string& dtype,
    int64_t batch_size,
    int64_t input_channels,
    int64_t height,
    int64_t width,
    int64_t warmup_iters,
    bool use_inference_mode)
    : device_(ParseDevice(device)),
      dtype_(torch::kFloat32),
      use_inference_mode_(use_inference_mode),
      batch_size_(batch_size),
      input_channels_(input_channels),
      height_(height),
      width_(width) {
    if (batch_size_ <= 0) {
        throw std::runtime_error("InferenceEngine batch_size must be positive.");
    }
    if (input_channels_ <= 0 || height_ <= 0 || width_ <= 0) {
        throw std::runtime_error("InferenceEngine input shape must be positive.");
    }
    if (device_.is_cpu() && dtype_ == torch::kFloat16) {
        throw std::runtime_error("float16 is not supported on CPU for InferenceEngine.");
    }

    module_ = torch::jit::load(model_path, device_);
    const std::string dtype_key = dtype.empty() ? "auto" : dtype;
    const bool dtype_auto = dtype_key == "auto" || dtype_key == "none";
    if (dtype_auto) {
        dtype_ = InferModuleDType(module_, torch::kFloat32);
    } else {
        dtype_ = ParseExplicitDType(dtype_key);
    }
    module_.to(device_, dtype_);
    module_.eval();

    auto options = torch::TensorOptions().device(device_).dtype(dtype_);
    input_buf_ = torch::zeros({batch_size_, input_channels_, height_, width_}, options);

#if defined(TORCH_CUDA_AVAILABLE)
    if (device_.is_cuda()) {
        at::cuda::CUDAGuard guard(device_);
        c10::InferenceMode guard_inference(use_inference_mode_);
        int64_t device_index = device_.has_index() ? device_.index() : -1;
        auto capture_stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, device_index);
        {
            c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
            for (int64_t i = 0; i < warmup_iters; ++i) {
                RunForward();
            }
            torch::cuda::synchronize(device_index);

            graph_ = std::make_unique<at::cuda::CUDAGraph>();
            graph_->capture_begin();
            std::tie(out_p1_, out_p2_, out_pmc_, out_value_) = RunForward();
            graph_->capture_end();
            graph_enabled_ = true;
        }
    }
#else
    (void)warmup_iters;
#endif
}

torch::Tensor InferenceEngine::PrepareInput(const torch::Tensor& input, int64_t n_valid) {
    if (input.dim() != 4) {
        throw std::runtime_error("InferenceEngine input must be 4D (B, C, H, W).");
    }
    if (input.size(1) != input_channels_ || input.size(2) != height_ || input.size(3) != width_) {
        throw std::runtime_error("InferenceEngine input shape mismatch.");
    }
    if (n_valid <= 0) {
        n_valid = input.size(0);
    }
    if (n_valid > batch_size_) {
        throw std::runtime_error("InferenceEngine n_valid exceeds batch_size.");
    }
    if (input.size(0) < n_valid) {
        throw std::runtime_error("InferenceEngine input batch smaller than n_valid.");
    }

    torch::Tensor src = input;
    if (src.device() != device_ || src.scalar_type() != dtype_) {
        src = src.to(device_, dtype_, /*non_blocking=*/true);
    }
    if (!src.is_contiguous()) {
        src = src.contiguous();
    }

    if (n_valid < batch_size_) {
        input_buf_.zero_();
    }
    input_buf_.narrow(0, 0, n_valid).copy_(src.narrow(0, 0, n_valid));
    return input_buf_;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> InferenceEngine::RunForward() {
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input_buf_);
    torch::jit::IValue output = module_.forward(inputs);
    auto tuple_ptr = output.toTuple();
    if (!tuple_ptr || tuple_ptr->elements().size() != 4) {
        throw std::runtime_error("InferenceEngine expected 4 outputs (log_p1, log_p2, log_pmc, value).");
    }
    const auto& elems = tuple_ptr->elements();
    return std::make_tuple(
        elems[0].toTensor(),
        elems[1].toTensor(),
        elems[2].toTensor(),
        elems[3].toTensor());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
InferenceEngine::Forward(const torch::Tensor& input, int64_t n_valid) {
#if defined(TORCH_CUDA_AVAILABLE)
    c10::optional<at::cuda::CUDAGuard> guard;
    if (device_.is_cuda()) {
        guard.emplace(device_);
    }
#endif
    c10::InferenceMode guard_inference(use_inference_mode_);
    int64_t valid = n_valid > 0 ? n_valid : input.size(0);
    PrepareInput(input, valid);
    if (graph_enabled_) {
#if defined(TORCH_CUDA_AVAILABLE)
        graph_->replay();
        return std::make_tuple(
            out_p1_.narrow(0, 0, valid),
            out_p2_.narrow(0, 0, valid),
            out_pmc_.narrow(0, 0, valid),
            out_value_.narrow(0, 0, valid));
#else
        throw std::runtime_error("InferenceEngine graph replay requested but CUDA is unavailable.");
#endif
    }

    auto outputs = RunForward();
    return std::make_tuple(
        std::get<0>(outputs).narrow(0, 0, valid),
        std::get<1>(outputs).narrow(0, 0, valid),
        std::get<2>(outputs).narrow(0, 0, valid),
        std::get<3>(outputs).narrow(0, 0, valid));
}

std::string InferenceEngine::DeviceString() const {
    return device_.str();
}

std::string InferenceEngine::DTypeString() const {
    return DTypeToString(dtype_);
}

}  // namespace v0
