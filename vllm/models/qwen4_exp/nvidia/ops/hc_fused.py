# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused HyperConnection mix for decode-sized batches (opt-in: VLLM_HC_FUSED=1).

Replaces cuBLAS down+inject GEMM -> split-K reduce -> hc_silu -> cuBLAS up GEMM
-> hc_gate_mix with two kernels:

- K1: split-K down+inject GEMM (xn @ W_di^T). The last CTA of each N tile applies
  silu(x / HC) to the low-rank columns and emits the raw injection logits.
- K2: up GEMM split across HC streams; each CTA computes one stream's gate for an
  H tile, applies sigmoid, multiplies by xn and atomically accumulates the gated
  mean. The last stream CTA per tile writes the BF16 block input.

Both kernels round intermediates to BF16 where the unfused path does, so results
match it to BF16 rounding order. Accumulator/counter workspaces are zeroed once
and reset by the finalizing CTA, so layers can share them (same stream).
Batches with more than _MAX_M tokens use the unfused path.
"""

import torch

from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_gate_mix, _hc_silu
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_MAX_M = 64
_K1 = dict(BLOCK_N=16, BLOCK_K=64, SPLIT=80, num_warps=4, num_stages=3)
_K2 = dict(BLOCK_H=32, BLOCK_R=64, num_warps=4, num_stages=2)


@triton.jit
def _hc_down_silu_kernel(
    xn_ptr, w_ptr, acc_ptr, cnt_ptr, lora_ptr, inj_ptr, M, stride_xm,
    K: tl.constexpr, R: tl.constexpr, HC: tl.constexpr, N_TOT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr, SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = pid_k * K_PER_SPLIT + kk + tl.arange(0, BLOCK_K)
        x = tl.load(
            xn_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=(offs_n[:, None] < N_TOT) & (offs_k[None, :] < K), other=0.0,
        )
        acc += tl.dot(x, tl.trans(w))
    tile = offs_m[:, None] * N_TOT + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_TOT)
    tl.atomic_add(acc_ptr + tile, acc, mask=mask, sem="relaxed")
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
    if ticket == SPLIT - 1:
        full = tl.load(acc_ptr + tile, mask=mask, other=0.0, volatile=True)
        full = full.to(tl.bfloat16).to(tl.float32)
        xs = full / HC
        silu = xs * tl.sigmoid(xs)
        tl.store(
            lora_ptr + offs_m[:, None] * R + offs_n[None, :], silu.to(tl.bfloat16),
            mask=mask & (offs_n[None, :] < R),
        )
        is_inj = (offs_n[None, :] >= R) & (offs_n[None, :] < R + HC)
        tl.store(
            inj_ptr + offs_m[:, None] * HC + (offs_n[None, :] - R), full.to(tl.bfloat16),
            mask=mask & is_inj,
        )
        tl.store(acc_ptr + tile, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=mask)
        tl.atomic_xchg(cnt_ptr + pid_n, 0)


@triton.jit
def _hc_up_gate_mix_kernel(
    lora_ptr, wu_ptr, xn_ptr, acc_ptr, cnt_ptr, out_ptr, M, stride_xm, stride_om,
    H: tl.constexpr, R: tl.constexpr, HC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    s = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_h = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    rows = s * H + offs_h
    g = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    for r0 in tl.static_range(0, R, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        lv = tl.load(lora_ptr + offs_m[:, None] * R + offs_r[None, :],
                     mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(wu_ptr + rows[:, None] * R + offs_r[None, :])
        g += tl.dot(lv, tl.trans(w))
    g = g.to(tl.bfloat16).to(tl.float32)
    msk = offs_m[:, None] < M
    x = tl.load(xn_ptr + offs_m[:, None] * stride_xm + rows[None, :], mask=msk, other=0.0)
    part = tl.sigmoid(g) * x.to(tl.float32)
    tile = offs_m[:, None] * H + offs_h[None, :]
    tl.atomic_add(acc_ptr + tile, part, mask=msk, sem="relaxed")
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
    if ticket == HC - 1:
        full = tl.load(acc_ptr + tile, mask=msk, other=0.0, volatile=True)
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_h[None, :],
                 (full / HC).to(tl.bfloat16), mask=msk)
        tl.store(acc_ptr + tile, tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32), mask=msk)
        tl.atomic_xchg(cnt_ptr + pid, 0)


_WORKSPACES: dict = {}


def _workspace(device, n_tot: int, h: int, r: int, hc: int):
    key = (device, n_tot, h, r, hc)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = dict(
            acc1=torch.zeros(_MAX_M, n_tot, device=device, dtype=torch.float32),
            cnt1=torch.zeros(triton.cdiv(n_tot, _K1["BLOCK_N"]), device=device, dtype=torch.int32),
            lora=torch.empty(_MAX_M, r, device=device, dtype=torch.bfloat16),
            acc2=torch.zeros(_MAX_M, h, device=device, dtype=torch.float32),
            cnt2=torch.zeros(triton.cdiv(h, _K2["BLOCK_H"]), device=device, dtype=torch.int32),
        )
        _WORKSPACES[key] = ws
    return ws


def prepare_workspace(device, n_tot: int, h: int, r: int, hc: int) -> None:
    """Allocate the shared workspace eagerly (call at model init, before capture)."""
    _workspace(device, n_tot, h, r, hc)


def _hc_fused_mix_impl(
    xn: torch.Tensor, w_down_inject: torch.Tensor, w_up: torch.Tensor,
    block_input: torch.Tensor, injection: torch.Tensor, hc: int, rank: int,
) -> None:
    m, d = xn.shape
    h = d // hc
    n_tot = w_down_inject.shape[0]
    if m == 0:
        return
    if m > _MAX_M or rank % _K2["BLOCK_R"] or h % _K2["BLOCK_H"]:
        down = torch.nn.functional.linear(xn, w_down_inject)
        lora = _hc_silu(down[:, :rank], hc)
        injection.copy_(down[:, rank:rank + hc])
        block_input.copy_(_hc_gate_mix(xn, torch.nn.functional.linear(lora, w_up), hc))
        return
    ws = _workspace(xn.device, n_tot, h, rank, hc)
    block_m = 16 if m <= 16 else 32 if m <= 32 else 64
    kps = triton.cdiv(triton.cdiv(d, _K1["SPLIT"]), _K1["BLOCK_K"]) * _K1["BLOCK_K"]
    split = triton.cdiv(d, kps)
    _hc_down_silu_kernel[(triton.cdiv(n_tot, _K1["BLOCK_N"]), split)](
        xn, w_down_inject, ws["acc1"], ws["cnt1"], ws["lora"], injection, m, xn.stride(0),
        K=d, R=rank, HC=hc, N_TOT=n_tot, BLOCK_M=block_m, BLOCK_N=_K1["BLOCK_N"],
        BLOCK_K=_K1["BLOCK_K"], K_PER_SPLIT=kps, SPLIT=split,
        num_warps=_K1["num_warps"], num_stages=_K1["num_stages"],
    )
    _hc_up_gate_mix_kernel[(triton.cdiv(h, _K2["BLOCK_H"]), hc)](
        ws["lora"], w_up, xn, ws["acc2"], ws["cnt2"], block_input, m, xn.stride(0),
        block_input.stride(0), H=h, R=rank, HC=hc, BLOCK_M=block_m,
        BLOCK_H=_K2["BLOCK_H"], BLOCK_R=_K2["BLOCK_R"],
        num_warps=_K2["num_warps"], num_stages=_K2["num_stages"],
    )


def _hc_fused_mix_fake(xn, w_down_inject, w_up, block_input, injection, hc, rank) -> None:
    return


direct_register_custom_op(
    op_name="qwen4_exp_hc_fused_mix",
    op_func=_hc_fused_mix_impl,
    mutates_args=["block_input", "injection"],
    fake_impl=_hc_fused_mix_fake,
)


def hc_fused_mix(
    xn: torch.Tensor, w_down_inject: torch.Tensor, w_up: torch.Tensor, hc: int, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (block_input [M, D/hc], injection logits [M, hc])."""
    m, d = xn.shape
    block_input = xn.new_empty(m, d // hc)
    injection = xn.new_empty(m, hc)
    torch.ops.vllm.qwen4_exp_hc_fused_mix(xn, w_down_inject, w_up, block_input, injection, hc, rank)
    return block_input, injection
