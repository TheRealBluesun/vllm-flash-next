"""Triton W8A16 skinny GEMM: y[M,N] = x[M,K] @ (W_fp8[N,K] * scale[N])^T.

Memory-bound decode path (M <= 64): split-K over a 2-D grid so small-N layers
still cover every SM, FP8 weights dequantized in registers, BF16 tensor-core
dot, fp32 atomics for the split-K reduction.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                  stride_xm, stride_wn, stride_ym,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  K_PER_SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k0 = pid_k * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k0 + kk + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * s[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(y_ptrs, acc, mask=mask, sem="relaxed")


_NUM_SMS = None


def pick_config(M, N, K):
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    block_m = 16 if M <= 16 else (32 if M <= 32 else 64)
    block_n = 64
    block_k = 128
    n_tiles = triton.cdiv(N, block_n)
    # enough CTAs for ~4 per SM, each doing >= 2 K-blocks
    split = 1
    while n_tiles * split < 4 * _NUM_SMS and K // (block_k * split * 2) >= 2:
        split *= 2
    k_per_split = triton.cdiv(triton.cdiv(K, split), block_k) * block_k
    split = triton.cdiv(K, k_per_split)
    return block_m, block_n, block_k, split, k_per_split


def w8a16_gemm(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor,
               out: torch.Tensor | None = None) -> torch.Tensor:
    """x [M,K] bf16, w [N,K] float8_e4m3fn, scale [N] fp32 -> y [M,N] bf16."""
    M, K = x.shape
    N = w.shape[0]
    bm, bn, bk, split, kps = pick_config(M, N, K)
    acc = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    _w8a16_kernel[(triton.cdiv(N, bn), split)](
        x, w, scale, acc, M, N, K, x.stride(0), w.stride(0), acc.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, K_PER_SPLIT=kps, num_warps=4, num_stages=4)
    return acc.to(torch.bfloat16) if out is None else out.copy_(acc)


def quantize_per_channel(w_bf16: torch.Tensor):
    amax = w_bf16.float().abs().amax(dim=1).clamp(min=1e-12)
    scale = amax / 448.0
    wq = (w_bf16.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return wq, scale.float()
