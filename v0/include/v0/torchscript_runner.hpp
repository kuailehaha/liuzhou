#pragma once

#include <string>
#include <tuple>

#include <c10/util/Optional.h>
#include <torch/script.h>

namespace v0 {

class TorchScriptRunner {
   public:
    TorchScriptRunner(
        const std::string& model_path,
        const std::string& device,
        const std::string& dtype,
        bool use_inference_mode = true);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Forward(
        const torch::Tensor& input);

    std::string DeviceString() const;
    std::string DTypeString() const;

   private:
    torch::jit::Module module_;
    torch::Device device_{torch::kCPU};
    c10::optional<torch::ScalarType> dtype_;
    bool use_inference_mode_{true};
};

}  // namespace v0
