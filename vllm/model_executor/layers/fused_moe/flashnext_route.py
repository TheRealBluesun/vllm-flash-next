# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_FUSED_ROUTE: one kernel (pdl_ext/moe_route.cu) computes softmax top-k *and* the
moe_align_block_size layout for decode-sized batches. fused_topk() runs it and stashes the
align outputs in a table keyed by the topk_ids buffer address (views of it keep the
address). moe_align_block_size() on that buffer removes the entry and uses it if block size,
shape and expert count match. The non-fused top-k path clears its buffer's address, so a
reused address can never pick up a stale layout."""

import torch

import vllm.envs as envs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.logger import init_logger

logger = init_logger(__name__)

_OK: bool | None = None
_PENDING: dict[int, tuple] = {}
_LOGGED = False


def _enabled() -> bool:
    global _OK
    if _OK is None:
        _OK = False
        if envs.VLLM_FUSED_ROUTE:
            from vllm.model_executor.layers import pdl_ext

            _OK = pdl_ext.load()
        logger.info("VLLM_FUSED_ROUTE: fused routing %s", "on" if _OK else "off")
    return _OK


def marlin_block_size(m: int, topk: int, num_experts: int) -> int:
    """Same choice as fused_marlin_moe (W4A16 / W8A16: no activation quantization)."""
    for bs in (8, 16, 32, 48, 64):
        if m * topk / num_experts / bs < 0.9:
            break
    return bs


def _fused_topk_impl(gating_output: torch.Tensor, topk: int, renormalize: bool,
                     w: torch.Tensor, ids: torch.Tensor, src: torch.Tensor) -> None:
    """Runs eagerly (custom op): decode-sized batches get the fused kernel and a pending
    align layout; anything else uses the reference topk_softmax."""
    m, e = gating_output.shape
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        _get_padding_mask, vllm_topk_softmax)

    pad = _get_padding_mask(m)
    if (
        not _enabled()
        or not (1 <= m <= 64 and e <= 1024 and topk <= 32)
        or gating_output.stride(1) != 1
        or (pad is not None and (pad.dtype != torch.bool or not pad.is_cuda))
    ):
        _PENDING.pop(ids.data_ptr(), None)
        vllm_topk_softmax(w, ids, src, gating_output, renormalize)
        return
    dev = gating_output.device
    bs = marlin_block_size(m, topk, e)
    numel = m * topk
    max_pad = numel + e * (bs - 1)
    if numel < e:
        max_pad = min(numel * bs, max_pad)
    sorted_ids = torch.empty(max_pad, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(-(-max_pad // bs), dtype=torch.int32, device=dev)
    npp = torch.empty(1, dtype=torch.int32, device=dev)
    torch.ops.flashnext_pdl.moe_route(gating_output, w, ids, src, sorted_ids, expert_ids, npp,
                                      topk, renormalize, bs,
                                      pad if pad is not None else gating_output, pad is not None)
    _PENDING[ids.data_ptr()] = (tuple(ids.shape), bs, e, sorted_ids, expert_ids, npp)


def _fused_topk_fake(gating_output, topk, renormalize, w, ids, src) -> None:
    return


direct_register_custom_op(
    op_name="flashnext_fused_topk_softmax",
    op_func=_fused_topk_impl,
    mutates_args=["w", "ids", "src"],
    fake_impl=_fused_topk_fake,
)


def try_fused_topk_softmax(
    gating_output: torch.Tensor, topk: int, renormalize: bool, indices_type
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Called from fused_topk (possibly while torch.compile traces it): the routing runs in
    a custom op, so the fused path is decided at execution time, never at trace time."""
    if (
        not envs.VLLM_FUSED_ROUTE
        or indices_type not in (None, torch.int32)
        or gating_output.dtype not in (torch.bfloat16, torch.float32)
        or gating_output.dim() != 2
    ):
        return None
    m = gating_output.shape[0]
    dev = gating_output.device
    w = torch.empty(m, topk, dtype=torch.float32, device=dev)
    ids = torch.empty(m, topk, dtype=torch.int32, device=dev)
    src = torch.empty(m, topk, dtype=torch.int32, device=dev)
    torch.ops.vllm.flashnext_fused_topk_softmax(gating_output, topk, renormalize, w, ids, src)
    return w, ids, src


def forget(topk_ids: torch.Tensor) -> None:
    """Called by the non-fused top-k path for its freshly allocated buffer."""
    if _PENDING:
        _PENDING.pop(topk_ids.data_ptr(), None)


def take_aligned(topk_ids: torch.Tensor, block_size: int, num_experts: int, expert_map,
                 pad_sorted_ids: bool):
    if not _PENDING:
        return None
    st = _PENDING.pop(topk_ids.data_ptr(), None)
    if (
        st is None
        or expert_map is not None
        or pad_sorted_ids
        or st[0] != tuple(topk_ids.shape)
        or st[1] != block_size
        or st[2] != num_experts
    ):
        return None
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.info("VLLM_FUSED_ROUTE: using fused top-k + align (block size %d)", block_size)
    return st[3], st[4], st[5]
