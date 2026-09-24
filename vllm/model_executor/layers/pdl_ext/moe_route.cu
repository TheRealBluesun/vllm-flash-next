// Flash-Next fused MoE routing for decode (VLLM_FUSED_ROUTE): one CTA replaces
// topk_softmax (topkGating) + moe_align_block_size + count_and_sort_expert_tokens
// (+ the sorted_ids fill) for M <= 64 tokens.
//
//  1. softmax over E router logits per token (one warp per token), top-k by warp arg-max
//     (ties -> lower expert id), optional renormalization: same outputs as topk_softmax
//     (weights fp32, ids int32, source rows k * M + t).
//  2. moe_align_block_size semantics: sorted_ids filled with numel, each (token, k) pair i
//     placed at cumsum[e] + rank inside its expert's block-padded segment; expert_ids per
//     block (-1 after the last used block); num_tokens_post_pad = cumsum[E].
//     Order within an expert is arbitrary (as in the reference, which uses atomics).
// Launched with programmatic dependent launch: waits for the router GEMV, then lets the
// next kernel start early.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "libtorch_stable/torch_utils.h"

#ifndef FN_ROUTE_PHASE
#define FN_ROUTE_PHASE 9
#endif

namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxE = 1024;
constexpr int kMaxPerLane = kMaxE / 32;
constexpr int kMaxPairs = 64 * 32;

template <typename T>
__device__ __forceinline__ float to_f(T v) { return static_cast<float>(v); }
template <>
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T, int PER>
__global__ void __launch_bounds__(kThreads) moe_route_kernel(
    const T* __restrict__ logits, int stride_l, int M, int E, int topk, bool renormalize,
    int block_size, float* __restrict__ w_out, int* __restrict__ id_out,
    int* __restrict__ src_out, int* __restrict__ sorted_ids, int* __restrict__ expert_ids,
    int* __restrict__ num_post_pad, int max_pad, int max_blocks,
    const bool* __restrict__ is_padding) {
  __shared__ int counts[kMaxE];
  __shared__ int ranks[kMaxE];
  __shared__ int cumsum[kMaxE + 1];
  __shared__ int pairs[kMaxPairs];
  __shared__ int warp_tot[kWarps];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int numel = M * topk;

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  for (int e = tid; e < E; e += kThreads) { counts[e] = 0; ranks[e] = 0; }
  for (int i = tid; i < max_pad; i += kThreads) sorted_ids[i] = numel;
  __syncthreads();
  if (FN_ROUTE_PHASE == 0) return;

  // 1) softmax + top-k, one warp per token. Lane owns experts [lane * PER, lane * PER + PER).
  for (int t = warp; t < M; t += kWarps) {
    float v[PER];
    const T* row = logits + static_cast<size_t>(t) * stride_l + lane * PER;
    if (sizeof(T) == 2 && PER % 8 == 0 && E == 32 * PER) {  // host guarantees 16 B alignment
#pragma unroll
      for (int j = 0; j < PER; j += 8) {
        const uint4 raw = *reinterpret_cast<const uint4*>(row + j);
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int u = 0; u < 8; ++u) v[j + u] = __bfloat162float(h[u]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < PER; ++j) v[j] = lane * PER + j < E ? to_f(row[j]) : -INFINITY;
    }
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < PER; ++j) mx = fmaxf(mx, v[j]);
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      v[j] = lane * PER + j < E ? __expf(v[j] - mx) : 0.f;
      sum += v[j];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float inv = 1.f / sum;

    // Probabilities are >= 0, so their float bits order like unsigned ints: key = bits + 1
    // (0 = removed / out of range). Each round: redux max over lane bests, then redux min
    // over the expert ids holding that key (lowest id wins ties, like topk_softmax).
    // Only the winning lane rescans.
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j)
      key[j] = lane * PER + j < E ? __float_as_uint(v[j] * inv) + 1u : 0u;
    uint32_t bk = 0;
    int be = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < PER; ++j)
      if (key[j] > bk) { bk = key[j]; be = lane * PER + j; }
    float my_w = 0.f;
    int my_id = 0;
    float selected = 0.f;
    for (int r = 0; r < topk; ++r) {
      const uint32_t wk = __reduce_max_sync(0xffffffffu, bk);
      const int we = static_cast<int>(
          __reduce_min_sync(0xffffffffu, bk == wk ? static_cast<uint32_t>(be) : 0x7fffffffu));
      const float wv = __uint_as_float(wk - 1u);
      if (lane == r) { my_w = wv; my_id = we; }
      selected += wv;
      if (be == we) {
        bk = 0; be = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
          if (lane * PER + j == we) key[j] = 0u;
          if (key[j] > bk) { bk = key[j]; be = lane * PER + j; }
        }
      }
    }
    if (lane < topk) {
      // Padding rows of a CUDA-graph batch: id -1, not routed (as topk_softmax + align).
      const bool pad = is_padding != nullptr && is_padding[t];
      const float denom = selected > 0.f ? selected : 1.f;
      const int idx = t * topk + lane;
      w_out[idx] = renormalize ? my_w / denom : my_w;
      id_out[idx] = pad ? -1 : my_id;
      src_out[idx] = lane * M + t;
      pairs[idx] = pad ? -1 : my_id;
      if (!pad) atomicAdd(&counts[my_id], 1);
    }
  }
  __syncthreads();
  if (FN_ROUTE_PHASE == 1) return;

  // 2) exclusive scan of block-padded counts (each thread owns E / kThreads experts).
  constexpr int kPer = kMaxE / kThreads;
  int local[kPer];
  int run = 0;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int e = tid * kPer + j;
    const int c = e < E ? counts[e] : 0;
    local[j] = (c + block_size - 1) / block_size * block_size;
    run += local[j];
  }
  int incl = run;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += n;
  }
  if (lane == 31) warp_tot[warp] = incl;
  __syncthreads();
  int base = 0;
  for (int w = 0; w < warp; ++w) base += warp_tot[w];
  int pos = base + incl - run;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int e = tid * kPer + j;
    if (e < E) cumsum[e] = pos;
    pos += local[j];
  }
  if (tid == kThreads - 1) {
    cumsum[E] = pos;
    num_post_pad[0] = pos;
  }
  __syncthreads();

  if (FN_ROUTE_PHASE == 2) return;
  // 3) expert id per block (binary search in cumsum), then scatter the pairs.
  const int total = cumsum[E];
  for (int b = tid; b < max_blocks; b += kThreads) {
    const int p = b * block_size;
    int e = -1;
    if (p < total) {
      int lo = 0, hi = E - 1;  // last e with cumsum[e] <= p
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (cumsum[mid] <= p) lo = mid; else hi = mid - 1;
      }
      e = lo;
    }
    expert_ids[b] = e;
  }
  for (int i = tid; i < numel; i += kThreads) {
    const int e = pairs[i];
    if (e < 0) continue;
    const int r = atomicAdd(&ranks[e], 1);
    sorted_ids[cumsum[e] + r] = i;
  }
}

}  // namespace

void flashnext_moe_route(torch::stable::Tensor const& logits, torch::stable::Tensor& w_out,
                         torch::stable::Tensor& id_out, torch::stable::Tensor& src_out,
                         torch::stable::Tensor& sorted_ids, torch::stable::Tensor& expert_ids,
                         torch::stable::Tensor& num_post_pad, int64_t topk, bool renormalize,
                         int64_t block_size, torch::stable::Tensor const& is_padding,
                         bool has_padding) {
  using torch::headeronly::ScalarType;
  const int M = static_cast<int>(logits.size(0));
  const int E = static_cast<int>(logits.size(1));
  STD_TORCH_CHECK((!has_padding || (is_padding.scalar_type() == ScalarType::Bool && is_padding.numel() >= M)) &&
                      M >= 1 && M <= 64 && E >= 1 && E <= kMaxE && topk >= 1 && topk <= 32 &&
                      topk <= E && logits.stride(1) == 1,
                  "moe_route: unsupported shape");
  torch::stable::accelerator::DeviceGuard const device_guard(logits.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(logits.get_device_index());
  (void)cudaGetLastError();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(1);
  cfg.blockDim = dim3(kThreads);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const int sl = static_cast<int>(logits.stride(0));
#define FN_ROUTE_P(T, P)                                                                          \
  cudaLaunchKernelEx(&cfg, moe_route_kernel<T, P>, static_cast<const T*>(logits.data_ptr()), sl, \
                     M, E, static_cast<int>(topk), renormalize, static_cast<int>(block_size),  \
                     static_cast<float*>(w_out.data_ptr()), static_cast<int*>(id_out.data_ptr()), \
                     static_cast<int*>(src_out.data_ptr()),                                   \
                     static_cast<int*>(sorted_ids.data_ptr()),                                \
                     static_cast<int*>(expert_ids.data_ptr()),                                \
                     static_cast<int*>(num_post_pad.data_ptr()),                              \
                     static_cast<int>(sorted_ids.size(0)), static_cast<int>(expert_ids.size(0)), \
                     has_padding ? static_cast<const bool*>(is_padding.data_ptr()) : nullptr)
#define FN_ROUTE(T)                                               \
  do {                                                            \
    if (E == 512 && sl % 8 == 0 && reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0) \
      FN_ROUTE_P(T, 16);               \
    else FN_ROUTE_P(T, kMaxPerLane);                              \
  } while (0)
  if (logits.scalar_type() == ScalarType::BFloat16) {
    FN_ROUTE(__nv_bfloat16);
  } else if (logits.scalar_type() == ScalarType::Float) {
    FN_ROUTE(float);
  } else {
    STD_TORCH_CHECK(false, "moe_route: logits must be bf16 or fp32");
  }
#undef FN_ROUTE
#undef FN_ROUTE_P
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "moe_route launch failed: ", cudaGetErrorString(err));
}
