// Flash-Next PDL extension ops (VLLM_PDL_GEMV), registered as torch.ops.flashnext_pdl.*
#include <string>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

void flashnext_gdn_post_conv_pdl(
    torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_log,
    torch::stable::Tensor const& dt_bias,
    torch::stable::Tensor const& state_indices,
    torch::stable::Tensor const& cu_seqlens,
    torch::stable::Tensor const& num_accepted_tokens,
    torch::stable::Tensor& state, torch::stable::Tensor const& output_gate,
    torch::stable::Tensor const& norm_weight, torch::stable::Tensor& out,
    double scale, double norm_eps, const std::string& output_gate_activation);

STABLE_TORCH_LIBRARY(flashnext_pdl, ops) {
  ops.def(
      "gdn_post_conv_mtp("
      "Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
      "Tensor! state, Tensor output_gate, Tensor norm_weight, Tensor! out, "
      "float scale, float norm_eps=1e-5, "
      "str output_gate_activation='silu') -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(flashnext_pdl, CUDA, ops) {
  ops.impl("gdn_post_conv_mtp", TORCH_BOX(&flashnext_gdn_post_conv_pdl));
}
