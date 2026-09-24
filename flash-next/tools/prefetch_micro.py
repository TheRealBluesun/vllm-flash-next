"""Does MADV_WILLNEED (via process_madvise) turn serial swap-in faults into
parallel I/O? Simulates PLE row gathers on an anonymous buffer whose pages
are pushed to swap with MADV_PAGEOUT."""
import ctypes, os, time
import numpy as np
import torch

libc = ctypes.CDLL(None, use_errno=True)
MADV_WILLNEED, MADV_PAGEOUT = 3, 21
SYS_process_madvise = 440
PAGE = 4096
pidfd = os.pidfd_open(os.getpid())
libc.syscall.restype = ctypes.c_long


def pvec_madvise(pages: np.ndarray, advice: int) -> None:
    """pages: uint64 page-aligned addresses."""
    for i in range(0, len(pages), 1024):
        chunk = pages[i:i + 1024]
        iov = np.empty((len(chunk), 2), dtype=np.uint64)
        iov[:, 0] = chunk
        iov[:, 1] = PAGE
        r = libc.syscall(SYS_process_madvise, pidfd, ctypes.c_void_p(iov.ctypes.data),
                         ctypes.c_ulong(len(chunk)), advice, 0)
        if r < 0:
            raise OSError(ctypes.get_errno(), "process_madvise")


ROW = 160
N_ROWS = (2 << 30) // ROW  # 2 GiB table
w = torch.empty((N_ROWS, ROW), dtype=torch.uint8)
w.fill_(1)
base = w.data_ptr()
torch.set_num_threads(16)


def pages_for(ids: np.ndarray) -> np.ndarray:
    a = base + ids.astype(np.uint64) * ROW
    return np.unique(np.concatenate([a >> 12, (a + ROW - 1) >> 12]) << 12).astype(np.uint64)


def trial(n_rows: int, prefetch: bool, rng) -> tuple[float, float]:
    ids = rng.integers(0, N_ROWS, n_rows)
    pg = pages_for(ids)
    pvec_madvise(pg, MADV_PAGEOUT)  # evict exactly the pages we will touch
    time.sleep(0.2)
    idx = torch.from_numpy(ids)
    t0 = time.perf_counter()
    if prefetch:
        pvec_madvise(pg, MADV_WILLNEED)
    t1 = time.perf_counter()
    torch.index_select(w, 0, idx)
    t2 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t0) * 1e3


rng = np.random.default_rng(0)
for n in (64, 256, 32768):
    for pf in (False, True):
        res = [trial(n, pf, rng) for _ in range(3)]
        print(f"rows={n:6d} prefetch={pf!s:5}  madvise {np.mean([r[0] for r in res]):7.2f} ms  "
              f"total {np.mean([r[1] for r in res]):8.2f} ms", flush=True)
