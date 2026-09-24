# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Swap-aware row prefetch for the CPU-resident PLE table.

When the host is short on RAM the n-gram table is partly paged out to swap.
``torch.index_select`` then takes one major fault per cold row, serially
(small gathers run on a single thread), so the GPU idles for milliseconds per
decode step. Advising the kernel with ``MADV_WILLNEED`` on every page a gather
will touch starts all swap-ins up front, in parallel; the gather then waits
only for the slowest one. Pages already resident cost a page-table walk.

Enable with ``VLLM_PLE_PREFETCH=1``. ``VLLM_PLE_TIMING=1`` logs gather timing.
"""

import ctypes
import os
import resource
import time

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PREFETCH_ENABLED = os.environ.get("VLLM_PLE_PREFETCH", "0") == "1"
TIMING_ENABLED = os.environ.get("VLLM_PLE_TIMING", "0") == "1"
# Prefetch rows for whole prompts as soon as a request is admitted, so later
# prefill chunks find their rows already swapped in.
PREFILL_HINT_ENABLED = os.environ.get("VLLM_PLE_PREFILL_HINT", "0") == "1"
PREFILL_HINT_MIN_TOKENS = 2048

_MADV_WILLNEED = 3
_SYS_PROCESS_MADVISE = 440  # x86_64
_IOV_MAX = 1024
_PAGE_SHIFT = (os.sysconf("SC_PAGE_SIZE") - 1).bit_length()

_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long
_pidfd: int | None = None
_disabled_reason: str | None = None


def _process_madvise(iov: np.ndarray) -> None:
    global _pidfd, _disabled_reason
    if _pidfd is None:
        _pidfd = os.pidfd_open(os.getpid())
    for i in range(0, len(iov), _IOV_MAX):
        chunk = iov[i : i + _IOV_MAX]
        r = _libc.syscall(
            _SYS_PROCESS_MADVISE,
            _pidfd,
            ctypes.c_void_p(chunk.ctypes.data),
            ctypes.c_ulong(len(chunk)),
            _MADV_WILLNEED,
            0,
        )
        if r < 0:
            err = ctypes.get_errno()
            _disabled_reason = f"process_madvise failed: {os.strerror(err)}"
            logger.warning("PLE prefetch disabled (%s)", _disabled_reason)
            return


def prefetch_rows(weight: torch.Tensor, row_ids: torch.Tensor) -> None:
    """Start swap-in of every page backing ``weight[row_ids]`` (CPU, 2-D)."""
    if not PREFETCH_ENABLED or _disabled_reason is not None:
        return
    row_bytes = weight.stride(0) * weight.element_size()
    ids = row_ids.numpy().astype(np.uint64, copy=False)
    start = np.uint64(weight.data_ptr()) + ids * np.uint64(row_bytes)
    first = start >> np.uint64(_PAGE_SHIFT)
    last = (start + np.uint64(row_bytes - 1)) >> np.uint64(_PAGE_SHIFT)
    pages = np.unique(np.concatenate([first, last]))
    iov = np.empty((len(pages), 2), dtype=np.uint64)
    iov[:, 0] = pages << np.uint64(_PAGE_SHIFT)
    iov[:, 1] = 1 << _PAGE_SHIFT
    _process_madvise(iov)


class GatherTimer:
    """Aggregate per-gather timing; logs a summary every ``every`` calls."""

    def __init__(self, every: int = 200) -> None:
        self.every = every
        self._reset()

    def _reset(self) -> None:
        self.n = 0
        self.rows = 0
        self.t_ids = self.t_prefetch = self.t_gather = 0.0
        self.faults = 0
        self.max_gather = 0.0

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    @staticmethod
    def majflt() -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_majflt

    def add(self, rows: int, t_ids: float, t_prefetch: float, t_gather: float,
            faults: int) -> None:
        self.n += 1
        self.rows += rows
        self.t_ids += t_ids
        self.t_prefetch += t_prefetch
        self.t_gather += t_gather
        self.faults += faults
        self.max_gather = max(self.max_gather, t_gather)
        if self.n >= self.every:
            n = self.n
            logger.info(
                "PLE gather x%d (prefetch=%s): rows/call %.0f | ids %.3f ms | "
                "prefetch %.3f ms | index_select %.3f ms (max %.2f) | majflt/call %.1f",
                n, PREFETCH_ENABLED, self.rows / n, self.t_ids / n * 1e3,
                self.t_prefetch / n * 1e3, self.t_gather / n * 1e3,
                self.max_gather * 1e3, self.faults / n,
            )
            self._reset()


gather_timer = GatherTimer() if TIMING_ENABLED else None
