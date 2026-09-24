// Flash-Next decode GEMV (VLLM_PDL_GEMV_CUDA): y[M, N] = x[M, K] @ W[N, K]^T for M <= 16.
//
// - W is FP8 e4m3 with 128x128 block scales, or BF16, row-major [N, K].
// - A CTA owns 64 rows x a K slice. Weights stream through a small shared-memory
//   ring filled by TMA bulk copies (cp.async.bulk + mbarrier); the first stages
//   are issued *before* griddepcontrol.wait, so they overlap the previous kernel.
// - Math: mma.m16n8k16 bf16 (FP8 converted exactly to bf16 in registers). Each
//   warp owns 8 rows (the MMA n dimension); tokens are the MMA m dimension.
//   K is permuted so each thread reads 32 contiguous elements per 128-wide
//   K block (the dot product is order-independent; x and W use the same order).
// - Split-K partials go to an fp32 workspace with atomics; the last CTA of each
//   row tile writes y and re-zeroes its slice (same protocol as pdl_gemv.py).

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "libtorch_stable/torch_utils.h"

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
#ifndef FN_BOXES
#define FN_BOXES 2
#endif
#ifndef FN_STAGES
#define FN_STAGES 2
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

// One 2D TMA box [rows][128 B] (128B swizzle), OOB rows/cols zero-filled.
// Destination is .shared::cta, not .shared::cluster: on sm_120 ptxas guards cluster
// destinations with a software fallback call (__cuda_syscall_cp_async_bulk_*), which
// makes the driver reserve ~14.6 KB of stack per thread (~3.6 GB of VRAM).
template <bool EVICT_FIRST>
__device__ __forceinline__ void tma_box(void* dst, const CUtensorMap* map, int col, int row,
                                        uint64_t* bar, uint64_t policy) {
  if constexpr (EVICT_FIRST) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(smem_u32(dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(col), "r"(row), "r"(smem_u32(bar)), "l"(policy)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
        "l"(reinterpret_cast<uint64_t>(map)), "r"(col), "r"(row), "r"(smem_u32(bar))
        : "memory");
  }
}

__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Two e4m3 values (low 16 bits of v) -> bf16x2, exact (e4m3 is a subset of bf16).
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
  __nv_bfloat162 b = __float22bfloat162_rn(f);
  return *reinterpret_cast<uint32_t*>(&b);
}

// RG = row groups (of 8 rows) per CTA; the 8 warps form RG row groups x KG = 8 / RG
// K groups. KG > 1 splits K inside the CTA (reduced through shared memory), which
// is much cheaper than global split-K for layers with few rows (out_proj, HC down).
template <bool FP8, bool M16, bool EVICT_FIRST, int RG>
__global__ void __launch_bounds__(kThreads) gemv_tma_kernel(
    const __nv_bfloat16* __restrict__ x, int stride_x, const __grid_constant__ CUtensorMap wmap,
    const float* __restrict__ s, int stride_s, float* __restrict__ acc, int* __restrict__ cnt,
    __nv_bfloat16* __restrict__ y, int stride_y, int M, int N, int K, int k_cta, int splits) {
  constexpr int ES = FP8 ? 1 : 2;
  constexpr int STAGES = FN_STAGES;
  constexpr int KG = kWarps / RG;
  constexpr int kRows = 8 * RG;
  constexpr int BOX_BYTES = kRows * 128;         // [kRows][128 B]
  constexpr int BOX_K = 128 / ES;                // K elements per box
  constexpr int BOXES = FN_BOXES * KG;           // boxes per stage (~16 KB)
  constexpr int kKS = BOXES * BOX_K;             // K elements per stage
  constexpr int STAGE_BYTES = BOXES * BOX_BYTES;

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // 128B-swizzled TMA boxes need 1024 B alignment.
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* wbuf = smem;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + STAGES * STAGE_BYTES);
  __shared__ int s_ticket;

  const int tile = blockIdx.x;
  const int row0 = tile * kRows;
  const int k0 = blockIdx.y * k_cta;
  const int kc = min(k_cta, K - k0);
  const int nst = (kc + kKS - 1) / kKS;
  const int rows = min(kRows, N - row0);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  uint64_t policy = 0;
  if constexpr (EVICT_FIRST) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  }

  auto issue = [&](int j) {
    const int slot = j % STAGES;
    if (tid == 0) {
      mbar_expect_tx(&bars[slot], STAGE_BYTES);
#pragma unroll
      for (int bx = 0; bx < BOXES; ++bx)
        tma_box<EVICT_FIRST>(wbuf + slot * STAGE_BYTES + bx * BOX_BYTES, &wmap,
                             k0 + j * kKS + bx * BOX_K, row0, &bars[slot], policy);
    }
  };

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) mbar_init(&bars[i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // 1) weights first: they don't depend on the previous kernel.
  for (int j = 0; j < min(STAGES, nst); ++j) issue(j);
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // 2) activations are read straight from global (tiny, L1-resident) after the wait.

  const int g = lane >> 2, q = lane & 3;
  const int rg = warp % RG, kg = warp / RG;
  const int wrow = rg * 8 + g;
  const bool has_a = g < M;
  const bool has_b = M16 && (g + 8) < M;
  float c[4] = {0.f, 0.f, 0.f, 0.f};

  for (int j = 0; j < nst; ++j) {
    const int slot = j % STAGES;
    mbar_wait(&bars[slot], (j / STAGES) & 1);
    const int kk0 = j * kKS;  // relative to k0
#pragma unroll 1
    for (int kb = kg; kb < kKS / 128; kb += KG) {
    const int kk = kk0 + kb * 128;
    const bool kvalid = q * 32 < kc - kk;

    uint32_t wv[8 * ES];
    uint32_t xa[16], xb[16];
    if (kvalid) {
      // element e = kb*128 + 32q + i  ->  box e / BOX_K, byte (e % BOX_K) * ES, 16 B chunk c,
      // stored at chunk c ^ (row & 7) (128B swizzle).
#pragma unroll
      for (int i = 0; i < 2 * ES; ++i) {
        const int byte = (kb * 128 + q * 32) * ES + i * 16;  // byte offset within the stage row
        const int box = byte / 128, chunk = (byte % 128) / 16;
        const uint8_t* p = wbuf + slot * STAGE_BYTES + box * BOX_BYTES + wrow * 128 +
                           ((chunk ^ (wrow & 7)) * 16);
        *reinterpret_cast<uint4*>(&wv[4 * i]) = *reinterpret_cast<const uint4*>(p);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8 * ES; ++i) wv[i] = 0;
    }
    if (kvalid && has_a) {
      const uint4* xp = reinterpret_cast<const uint4*>(x + static_cast<size_t>(g) * stride_x + k0 + kk + q * 32);
#pragma unroll
      for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(&xa[4 * i]) = __ldg(xp + i);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) xa[i] = 0;
    }
    if constexpr (M16) {
      if (kvalid && has_b) {
        const uint4* xp = reinterpret_cast<const uint4*>(x + static_cast<size_t>(g + 8) * stride_x + k0 + kk + q * 32);
#pragma unroll
        for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(&xb[4 * i]) = __ldg(xp + i);
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) xb[i] = 0;
      }
    }

    float t[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      uint32_t b0, b1;
      if constexpr (FP8) {
        b0 = e4m3x2_to_bf16x2(static_cast<uint16_t>(wv[st] & 0xFFFFu));
        b1 = e4m3x2_to_bf16x2(static_cast<uint16_t>(wv[st] >> 16));
      } else {
        b0 = wv[2 * st];
        b1 = wv[2 * st + 1];
      }
      const uint32_t a1 = M16 ? xb[2 * st] : 0u;
      const uint32_t a3 = M16 ? xb[2 * st + 1] : 0u;
      mma_bf16(t, xa[2 * st], a1, xa[2 * st + 1], a3, b0, b1);
    }
    if constexpr (FP8) {
      const float sc = kvalid || true ? s[(row0 / 128) * stride_s + min(k0 + kk, K - 1) / 128] : 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] += sc * t[i];
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] += t[i];
    }
    }

    __syncthreads();  // all warps are done reading this slot
    if (j + STAGES < nst) {
      if (tid == 0) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      issue(j + STAGES);
    }
  }

  if constexpr (KG > 1) {  // reduce the K groups (the ring is free now)
    float* red = reinterpret_cast<float*>(wbuf);
#pragma unroll
    for (int i = 0; i < 4; ++i) red[(warp * 32 + lane) * 4 + i] = c[i];
    __syncthreads();
    if (kg == 0) {
      for (int o = 1; o < KG; ++o)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] += red[((rg + o * RG) * 32 + lane) * 4 + i];
    }
  }
  const bool writer = kg == 0;

  // Epilogue. c0,c1: token g, features 2q,2q+1 of this warp's 8 rows; c2,c3: token g+8.
  const int n = row0 + rg * 8 + 2 * q;
  if (splits == 1) {
    if (!writer) return;
#pragma unroll
    for (int h = 0; h < (M16 ? 2 : 1); ++h) {
      const int m = g + 8 * h;
      if (m < M) {
        if (n < N) y[static_cast<size_t>(m) * stride_y + n] = __float2bfloat16(c[2 * h]);
        if (n + 1 < N) y[static_cast<size_t>(m) * stride_y + n + 1] = __float2bfloat16(c[2 * h + 1]);
      }
    }
    return;
  }
#pragma unroll
  for (int h = 0; h < (M16 ? 2 : 1); ++h) {
    const int m = g + 8 * h;
    if (writer && m < M) {
      if (n < N) atomicAdd(&acc[static_cast<size_t>(m) * N + n], c[2 * h]);
      if (n + 1 < N) atomicAdd(&acc[static_cast<size_t>(m) * N + n + 1], c[2 * h + 1]);
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_ticket = atomicAdd(&cnt[tile], 1);
  __syncthreads();
  if (s_ticket == splits - 1) {
    __threadfence();
    for (int i = tid; i < M * kRows; i += kThreads) {
      const int m = i / kRows, r = i - m * kRows, nn = row0 + r;
      if (nn < N) {
        float* p = &acc[static_cast<size_t>(m) * N + nn];
        y[static_cast<size_t>(m) * stride_y + nn] = __float2bfloat16(__ldcg(p));
        *p = 0.f;
      }
    }
    if (tid == 0) cnt[tile] = 0;
  }
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                  const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                  const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                  CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiledFn encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    STD_TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeTiledFn>(p);
  }
  return fn;
}

template <bool FP8, bool M16, bool EVICT_FIRST, int RG>
void launch(const void* x, int stride_x, const void* w, const float* s, int stride_s, float* acc,
            int* cnt, void* y, int stride_y, int M, int N, int K, int k_cta, cudaStream_t stream) {
  constexpr int ES = FP8 ? 1 : 2;
  constexpr int kRows = 8 * RG;
  CUtensorMap map;
  const cuuint64_t gdim[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(N)};
  const cuuint64_t gstride[1] = {static_cast<cuuint64_t>(K) * ES};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(128 / ES), static_cast<cuuint32_t>(kRows)};
  const cuuint32_t estride[2] = {1, 1};
  const CUresult r = encode_fn()(
      &map, FP8 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      const_cast<void*>(w), gdim, gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
  constexpr int STAGE_BYTES = FN_BOXES * (kWarps / RG) * kRows * 128;
  const int smem = 1024 + FN_STAGES * STAGE_BYTES + 64;
  auto kernel = gemv_tma_kernel<FP8, M16, EVICT_FIRST, RG>;
  static int configured_smem = 0;
  if (smem > configured_smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured_smem = smem;
  }
  const int tiles = (N + kRows - 1) / kRows;
  const int splits = (K + k_cta - 1) / k_cta;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(tiles, splits);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kernel, static_cast<const __nv_bfloat16*>(x), stride_x, map, s,
                     stride_s, acc, cnt, static_cast<__nv_bfloat16*>(y), stride_y, M, N, K,
                     k_cta, splits);
}

}  // namespace

// x [M, K] bf16 (row stride % 8 == 0), w [N, K] fp8_e4m3 or bf16 contiguous, s [ceil(N/128),
// ceil(K/128)] fp32 (FP8 only; any tensor otherwise), acc/cnt workspace, y [M, N] bf16.
// Requires M <= 16, K % 64 == 0, k_cta % 128 == 0 (FP8) or % 64 == 0 (BF16).
void flashnext_gemv_tma(torch::stable::Tensor const& x, torch::stable::Tensor const& w,
                        torch::stable::Tensor const& s, torch::stable::Tensor& acc,
                        torch::stable::Tensor& cnt, torch::stable::Tensor& y, int64_t k_cta,
                        int64_t evict_first, int64_t rows_per_cta) {
  using torch::headeronly::ScalarType;
  const bool fp8 = w.scalar_type() == ScalarType::Float8_e4m3fn;
  STD_TORCH_CHECK(fp8 || w.scalar_type() == ScalarType::BFloat16, "w must be fp8_e4m3fn or bf16");
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::BFloat16 && y.scalar_type() == ScalarType::BFloat16,
                  "x and y must be bf16");
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  STD_TORCH_CHECK(M >= 1 && M <= 16 && w.size(1) == K && K % 64 == 0 && x.stride(1) == 1 &&
                      x.stride(0) % 8 == 0 && w.is_contiguous(),
                  "unsupported gemv_tma shape/layout");
  torch::stable::accelerator::DeviceGuard const device_guard(x.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(x.get_device_index());
  const float* sp = fp8 ? static_cast<const float*>(s.data_ptr()) : nullptr;
  const int ss = fp8 ? static_cast<int>(s.stride(0)) : 0;
  const bool m16 = M > 8;
  const bool ef = evict_first != 0;
  (void)cudaGetLastError();  // clear stale errors (e.g. a caching-allocator OOM that was retried)
  STD_TORCH_CHECK(rows_per_cta == 8 || rows_per_cta == 16 || rows_per_cta == 64,
                  "rows_per_cta must be 8, 16 or 64");
#define FN_LAUNCH_RG(F, MM, E, R)                                                                \
  launch<F, MM, E, R>(x.data_ptr(), static_cast<int>(x.stride(0)), w.data_ptr(), sp, ss,         \
                      static_cast<float*>(acc.data_ptr()), static_cast<int*>(cnt.data_ptr()),    \
                      y.data_ptr(), static_cast<int>(y.stride(0)), M, N, K,                      \
                      static_cast<int>(k_cta), stream)
#define FN_LAUNCH(F, MM, E)                                                   \
  do {                                                                        \
    if (rows_per_cta == 64) FN_LAUNCH_RG(F, MM, E, 8);                        \
    else if (rows_per_cta == 16) FN_LAUNCH_RG(F, MM, E, 2);                   \
    else FN_LAUNCH_RG(F, MM, E, 1);                                           \
  } while (0)
  if (fp8) {
    if (m16) { if (ef) FN_LAUNCH(true, true, true); else FN_LAUNCH(true, true, false); }
    else { if (ef) FN_LAUNCH(true, false, true); else FN_LAUNCH(true, false, false); }
  } else {
    if (m16) { if (ef) FN_LAUNCH(false, true, true); else FN_LAUNCH(false, true, false); }
    else { if (ef) FN_LAUNCH(false, false, true); else FN_LAUNCH(false, false, false); }
  }
#undef FN_LAUNCH_RG
#undef FN_LAUNCH
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "gemv_tma launch failed: ", cudaGetErrorString(err));
}
