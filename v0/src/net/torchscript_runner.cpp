#include "v0/torchscript_runner.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <c10/core/InferenceMode.h>
#if defined(TORCH_CUDA_AVAILABLE)
#include <c10/cuda/CUDAGuard.h>
#endif

namespace v0 {

namespace {

torch::Device ParseDevice(const std::string& device) {
    return torch::Device(device);
}

c10::optional<torch::ScalarType> ParseDType(const std::string& dtype) {
    if (dtype.empty() || dtype == "auto" || dtype == "none") {
        return c10::nullopt;
    }
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

std::string DTypeToString(const c10::optional<torch::ScalarType>& dtype) {
    if (!dtype.has_value()) {
        return "auto";
    }
    switch (*dtype) {
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

}  // namespace

TorchScriptRunner::TorchScriptRunner(
    const std::string& model_path,
    const std::string& device,
    const std::string& dtype,
    bool use_inference_mode)
    : device_(ParseDevice(device)),
      dtype_(ParseDType(dtype)),
      use_inference_mode_(use_inference_mode) {
    if (device_.is_cpu() && dtype_.has_value() && *dtype_ == torch::kFloat16) {
        throw std::runtime_error("float16 is not supported on CPU for TorchScriptRunner.");
    }

    module_ = torch::jit::load(model_path, device_);
    if (dtype_.has_value()) {
        module_.to(device_, *dtype_);
    }
    module_.eval();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TorchScriptRunner::Forward(const torch::Tensor& input) {
#if defined(TORCH_CUDA_AVAILABLE)
    c10::optional<at::cuda::CUDAGuard> guard;
    if (device_.is_cuda()) {
        guard.emplace(device_);
    }
#endif

    auto target_dtype = dtype_.has_value() ? *dtype_ : input.scalar_type();
    torch::Tensor prepared = input;
    if (prepared.device() != device_ || prepared.scalar_type() != target_dtype) {
        prepared = prepared.to(device_, target_dtype, /*non_blocking=*/true);
    }
    if (!prepared.is_contiguous()) {
        prepared = prepared.contiguous();
    }

    c10::InferenceMode guard_inference(use_inference_mode_);
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(prepared);
    torch::jit::IValue output = module_.forward(inputs);

    auto tuple_ptr = output.toTuple();
    if (!tuple_ptr || tuple_ptr->elements().size() != 4) {
        throw std::runtime_error("TorchScriptRunner expected 4 outputs (log_p1, log_p2, log_pmc, value).");
    }

    const auto& elems = tuple_ptr->elements();
    return std::make_tuple(
        elems[0].toTensor(),
        elems[1].toTensor(),
        elems[2].toTensor(),
        elems[3].toTensor());
}

std::string TorchScriptRunner::DeviceString() const {
    return device_.str();
}

std::string TorchScriptRunner::DTypeString() const {
    return DTypeToString(dtype_);
}

}  // namespace v0
