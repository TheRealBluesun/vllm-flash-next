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

void flashnext_gemv_tma(torch::stable::Tensor const& x, torch::stable::Tensor const& w,
                        torch::stable::Tensor const& s, torch::stable::Tensor& acc,
                        torch::stable::Tensor& cnt, torch::stable::Tensor& y, int64_t k_cta,
                        int64_t evict_first, int64_t rows_per_cta);

void flashnext_moe_route(torch::stable::Tensor const& logits, torch::stable::Tensor& w_out,
                         torch::stable::Tensor& id_out, torch::stable::Tensor& src_out,
                         torch::stable::Tensor& sorted_ids, torch::stable::Tensor& expert_ids,
                         torch::stable::Tensor& num_post_pad, int64_t topk, bool renormalize,
                         int64_t block_size, torch::stable::Tensor const& is_padding,
                         bool has_padding);

STABLE_TORCH_LIBRARY(flashnext_pdl, ops) {
  ops.def(
      "moe_route(Tensor logits, Tensor! w, Tensor! ids, Tensor! src, Tensor! sorted_ids, "
      "Tensor! expert_ids, Tensor! num_post_pad, int topk, bool renormalize, int block_size, "
      "Tensor is_padding, bool has_padding) -> ()");
  ops.def(
      "gemv_tma(Tensor x, Tensor w, Tensor s, Tensor! acc, Tensor! cnt, Tensor! y, "
      "int k_cta, int evict_first, int rows_per_cta) -> ()");
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
  ops.impl("gemv_tma", TORCH_BOX(&flashnext_gemv_tma));
  ops.impl("moe_route", TORCH_BOX(&flashnext_moe_route));
}
