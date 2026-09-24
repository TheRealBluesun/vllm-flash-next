# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PDL weight-prefetch split-K GEMV for decode-sized linears (opt-in: VLLM_PDL_GEMV=1).

Each CTA loads its whole weight tile *before* ``griddepcontrol.wait`` and
triggers its dependents right away, so when consecutive kernels are launched
with programmatic dependent launch, kernel N+1 streams its weights while
kernel N is still finishing. Weights don't depend on the previous kernel, so
only the activation load waits.

Two ops, both falling back to the regular path for M > _MAX_M (prefill):
- ``pdl_fp8_block_linear``: W8A16 with 128x128 block scales (online
  ``fp8_per_block_static``). Keeps a row-major FP8 copy next to Marlin's
  repacked weight, which prefill still uses.
- ``pdl_bf16_gemv``: unquantized BF16 weights.

Split-K partials are reduced with fp32 atomics; the last CTA of each N tile
writes the output and zeroes its workspace slice, so a workspace is shared by
all linears on the same stream (a dependent kernel only touches it after
``gdc_wait``, i.e. after the previous one has completed).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_MAX_M = 64
_BLOCK = 128  # FP8 block-scale granularity


@triton.jit
def _pdl_gemv_kernel(
    x_ptr, w_ptr, s_ptr, acc_ptr, cnt_ptr, y_ptr, M, N, K, stride_xm, stride_ym, stride_s,
    FP8: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, SPLIT: tl.constexpr, SBLK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    wmask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    # 1) weights (and scales) first: independent of the previous kernel
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=wmask, other=0.0)
    if FP8:
        # one scale per (row block, 128-wide K block): [BLOCK_N, BLOCK_K // SBLK]
        offs_kb = pid_k * (BLOCK_K // SBLK) + tl.arange(0, BLOCK_K // SBLK)
        s = tl.load(s_ptr + (offs_n[:, None] // SBLK) * stride_s + offs_kb[None, :],
                    mask=(offs_n[:, None] < N) & (offs_kb[None, :] * SBLK < K), other=0.0)
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    # 2) activations
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    if FP8:
        wb = (tl.reshape(w.to(tl.float32), (BLOCK_N, BLOCK_K // SBLK, SBLK)) * s[:, :, None])
        wb = tl.reshape(wb, (BLOCK_N, BLOCK_K)).to(tl.bfloat16)
    else:
        wb = w
    acc = tl.dot(x, tl.trans(wb))
    tile = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT == 1:
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16), mask=mask)
    else:
        tl.atomic_add(acc_ptr + tile, acc, mask=mask, sem="relaxed")
        tl.debug_barrier()
        ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
        if ticket == SPLIT - 1:
            full = tl.load(acc_ptr + tile, mask=mask, other=0.0, volatile=True)
            tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
                     full.to(tl.bfloat16), mask=mask)
            tl.store(acc_ptr + tile, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0)


def _config(n: int, k: int, fp8: bool) -> tuple[int, int, int]:
    """(BLOCK_N, BLOCK_K, num_warps); one K chunk per CTA. From tools/pdl_chain_sweep.py."""
    if fp8:
        return (64, 512, 8) if n >= 1024 else (32, 256, 4)
    if n >= 1024:
        return (32, 256, 4) if k <= 1024 else (64, 256, 8)
    return (32, 128, 4)


# (device, stream) -> (acc fp32, cnt int32), fixed size so a tensor captured in a
# CUDA graph is never replaced. Zeroed once; finalizing CTAs keep it zeroed.
_MAX_N = 32768
_WS: dict = {}


def _workspace(device: torch.device):
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    ws = _WS.get(key)
    if ws is None:
        ws = (torch.zeros(_MAX_M * _MAX_N, device=device, dtype=torch.float32),
              torch.zeros(_MAX_N // 32, device=device, dtype=torch.int32))
        _WS[key] = ws
    return ws


def _launch(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor | None) -> torch.Tensor:
    m, k = x.shape
    n = w.shape[0]
    fp8 = s is not None
    bn, bk, warps = _config(n, k, fp8)
    split = triton.cdiv(k, bk)
    tiles = triton.cdiv(n, bn)
    block_m = 16 if m <= 16 else 32 if m <= 32 else 64
    acc, cnt = _workspace(x.device)
    y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _pdl_gemv_kernel[(tiles, split)](
        x, w, s if fp8 else w, acc, cnt, y, m, n, k, x.stride(0), y.stride(0),
        s.stride(0) if fp8 else 0,
        FP8=fp8, BLOCK_M=block_m, BLOCK_N=bn, BLOCK_K=bk, SPLIT=split, SBLK=_BLOCK,
        num_warps=warps, num_stages=1, launch_pdl=True,
    )
    return y


def _usable(x: torch.Tensor, n: int) -> bool:
    return 0 < x.shape[0] <= _MAX_M and n <= _MAX_N and x.dtype == torch.bfloat16


# --------------------------------------------------------------------------- FP8
def _pdl_fp8_impl(
    x: torch.Tensor, w8: torch.Tensor, s: torch.Tensor, marlin_w: torch.Tensor,
    marlin_s: torch.Tensor, workspace: torch.Tensor, size_n: int, size_k: int,
) -> torch.Tensor:
    if _usable(x, size_n):
        if x.stride(1) != 1:
            x = x.contiguous()
        return _launch(x, w8, s)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear)
    return apply_fp8_marlin_linear(input=x, weight=marlin_w, weight_scale=marlin_s,
                                   workspace=workspace, size_n=size_n, size_k=size_k, bias=None)


def _pdl_fp8_fake(x, w8, s, marlin_w, marlin_s, workspace, size_n, size_k):
    return x.new_empty((x.shape[0], size_n))


direct_register_custom_op(op_name="pdl_fp8_block_linear", op_func=_pdl_fp8_impl,
                          fake_impl=_pdl_fp8_fake)


def keep_fp8_block_copy(layer: torch.nn.Module, weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Before Marlin repacking: keep the row-major [N, K] FP8 weight and its block scales."""
    if weight.dtype != torch.float8_e4m3fn or weight.dim() != 2 or scale.dim() != 2:
        return
    layer.register_buffer("_pdl_w8", weight.detach().clone().contiguous(), persistent=False)
    layer.register_buffer("_pdl_s", scale.detach().float().clone().contiguous(), persistent=False)


def pdl_fp8_apply(layer, x, marlin_scale, bias):
    lead = x.shape[:-1]
    y = torch.ops.vllm.pdl_fp8_block_linear(
        x.reshape(-1, x.shape[-1]), layer._pdl_w8, layer._pdl_s, layer.weight, marlin_scale,
        layer.workspace, layer.output_size_per_partition, layer.input_size_per_partition)
    y = y.reshape(*lead, y.shape[-1])
    return y if bias is None else y + bias


# --------------------------------------------------------------------------- BF16
def _pdl_bf16_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if _usable(x, weight.shape[0]):
        if x.stride(1) != 1:
            x = x.contiguous()
        return _launch(x, weight, None)
    return torch.nn.functional.linear(x, weight)


def _pdl_bf16_fake(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0]))


direct_register_custom_op(op_name="pdl_bf16_gemv", op_func=_pdl_bf16_impl,
                          fake_impl=_pdl_bf16_fake)


def pdl_bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    lead = x.shape[:-1]
    y = torch.ops.vllm.pdl_bf16_gemv(x.reshape(-1, x.shape[-1]), weight)
    return y.reshape(*lead, weight.shape[0])
