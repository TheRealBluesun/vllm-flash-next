# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_FUSED_ROUTE: one kernel (pdl_ext/moe_route.cu) computes softmax top-k *and* the
moe_align_block_size layout for decode-sized batches. fused_topk() runs it and stashes the
align outputs; the next moe_align_block_size() call with the same topk_ids buffer, block
size and expert count takes them instead of launching its own kernels. Any call clears the
stash, so a stale entry can never match a later (address-reused) buffer."""

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_STASH: tuple | None = None
_OK: bool | None = None


def _enabled() -> bool:
    global _OK
    if _OK is None:
        _OK = False
        if envs.VLLM_FUSED_ROUTE:
            from vllm.model_executor.layers import pdl_ext

            _OK = pdl_ext.load()
    return _OK


def marlin_block_size(m: int, topk: int, num_experts: int) -> int:
    """Same choice as fused_marlin_moe (W4A16 / W8A16: no activation quantization)."""
    for bs in (8, 16, 32, 48, 64):
        if m * topk / num_experts / bs < 0.9:
            break
    return bs


def try_fused_topk_softmax(
    gating_output: torch.Tensor, topk: int, renormalize: bool, indices_type
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    global _STASH
    _STASH = None
    m, e = gating_output.shape
    if (
        not _enabled()
        or torch.compiler.is_compiling()
        or not (1 <= m <= 64 and e <= 1024 and topk <= 32)
        or indices_type not in (None, torch.int32)
        or gating_output.dtype not in (torch.bfloat16, torch.float32)
        or gating_output.stride(1) != 1
    ):
        return None
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import _get_padding_mask

    pad = _get_padding_mask(m)
    if pad is not None and (pad.dtype != torch.bool or not pad.is_cuda):
        return None
    dev = gating_output.device
    bs = marlin_block_size(m, topk, e)
    numel = m * topk
    max_pad = numel + e * (bs - 1)
    if numel < e:
        max_pad = min(numel * bs, max_pad)
    w = torch.empty(m, topk, dtype=torch.float32, device=dev)
    ids = torch.empty(m, topk, dtype=torch.int32, device=dev)
    src = torch.empty(m, topk, dtype=torch.int32, device=dev)
    sorted_ids = torch.empty(max_pad, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(-(-max_pad // bs), dtype=torch.int32, device=dev)
    npp = torch.empty(1, dtype=torch.int32, device=dev)
    torch.ops.flashnext_pdl.moe_route(gating_output, w, ids, src, sorted_ids, expert_ids, npp,
                                      topk, renormalize, bs,
                                      pad if pad is not None else gating_output, pad is not None)
    _STASH = (ids.data_ptr(), tuple(ids.shape), bs, e, sorted_ids, expert_ids, npp)
    return w, ids, src


def take_aligned(topk_ids: torch.Tensor, block_size: int, num_experts: int, expert_map,
                 pad_sorted_ids: bool):
    global _STASH
    st, _STASH = _STASH, None
    if (
        st is None
        or expert_map is not None
        or pad_sorted_ids
        or st[0] != topk_ids.data_ptr()
        or st[1] != tuple(topk_ids.shape)
        or st[2] != block_size
        or st[3] != num_experts
    ):
        return None
    logger.info_once("VLLM_FUSED_ROUTE: using fused top-k + align (block size %d)", block_size)
    return st[4], st[5], st[6]
