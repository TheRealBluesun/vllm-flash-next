# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 (per-row scale) copy of the lm_head for greedy drafting.

Greedy drafters only need the argmax of the draft logits, and every draft
token is verified by the target model, so drafting with an FP8 copy of the
lm_head halves the bytes each draft step reads without affecting outputs
(it can only nudge the acceptance rate). The GEMV is memory-bound; this Triton
kernel reaches ~90% of DRAM bandwidth for [M<=16, 2560] x [248320, 2560].
"""

import torch

import vllm.envs as envs

from vllm.triton_utils import tl, triton


@triton.jit
def _w8a16_logits_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
    stride_xm, stride_wn, stride_ym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVICT_FIRST: tl.constexpr = False,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
            eviction_policy="evict_first" if EVICT_FIRST else "",
        )
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * s[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


class Fp8DraftHead:
    def __init__(self, weight: torch.Tensor) -> None:
        w = weight.detach()
        n, chunk = w.shape[0], 16384  # bounded fp32 temporaries (~170 MB)
        self.scale = torch.empty(n, device=w.device, dtype=torch.float32)
        q = torch.empty_like(w, dtype=torch.float8_e4m3fn)
        for i in range(0, n, chunk):
            wc = w[i : i + chunk].float()
            sc = wc.abs().amax(dim=1).clamp(min=1e-12) / 448.0
            self.scale[i : i + chunk] = sc
            q[i : i + chunk] = (wc / sc[:, None]).to(torch.float8_e4m3fn)
        self.weight = q
        self.n, self.k = q.shape

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """x [M, K] bf16 -> fp32 logits [M, N] (M small)."""
        x = x.contiguous()
        m = x.shape[0]
        block_m = 16 if m <= 16 else 32 if m <= 32 else 64
        y = torch.empty((m, self.n), device=x.device, dtype=torch.float32)
        block_n = 64
        _w8a16_logits_kernel[(triton.cdiv(self.n, block_n),)](
            x, self.weight, self.scale, y, m, self.n, self.k,
            x.stride(0), self.weight.stride(0), y.stride(0),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128,
            EVICT_FIRST=envs.VLLM_L2_DRAFT,  # heads stream 250-630 MB: keep L2 for the draft layer
            num_warps=4, num_stages=4,
        )
        return y
