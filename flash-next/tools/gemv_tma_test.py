"""B: CUDA TMA+mma GEMV (pdl_ext gemv_tma) vs Triton PDL GEMV vs production (Marlin / cuBLAS)."""
import sys, torch
sys.path.insert(0, "/opt/d/vllm-flash-next-0906")
import vllm.model_executor.layers.pdl_gemv as pg
from vllm.model_executor.layers import pdl_ext
pdl_ext.load()
BW, SMS = 1.79e12, 188
QUICK = len(sys.argv) > 1
SHAPES = [("qkvz", 16384, 2560, 1), ("out", 2560, 6144, 1), ("hcdn", 336, 10240, 0), ("hcup", 10240, 320, 0)]

import os
TARGET = int(os.environ.get("FN_TARGET", "564"))
def rows_for(N):
    return 64 if -(-N // 64) >= 128 else 16 if -(-N // 16) >= 128 else 8

def kcta(M, N, K, fp8):
    """Split K across CTAs only for few-row layers: aim for ~2 CTAs/SM, K slices >= 1024."""
    gran = 128 if fp8 else 64
    tiles = -(-N // rows_for(N))
    splits = max(1, min(2 * SMS // tiles, K // 1024))  # <= resident CTAs (2/SM at ~34 KB smem)
    return -(-(-(-K // splits)) // gran) * gran

def mk(N, K, fp8):
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    if not fp8:
        return w, None, w
    wb = w.float().reshape(N // 128 if N % 128 == 0 else -1, 128, K // 128, 128) if N % 128 == 0 else None
    s = (wb.abs().amax((1, 3)) / 448.0).clamp(min=1e-12)
    w8 = (wb / s[:, None, :, None]).reshape(N, K).to(torch.float8_e4m3fn)
    wref = (w8.float().reshape(N // 128, 128, K // 128, 128) * s[:, None, :, None]).reshape(N, K)
    return w8, s.contiguous(), wref

acc, cnt = pg._workspace(torch.device("cuda"))
def cuda_gemv(x, w, s, k, ev=0):
    y = torch.empty(x.shape[0], w.shape[0], device="cuda", dtype=torch.bfloat16)
    torch.ops.flashnext_pdl.gemv_tma(x, w, s if s is not None else w, acc, cnt, y, k, ev, rows_for(w.shape[0]))
    return y

def bench(fn, reps=5):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3

rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
torch.manual_seed(0)
for name, N, K, fp8 in SHAPES:
    w, s, wref = mk(N, K, fp8)
    for M in ((5, 16) if QUICK else (1, 5, 8, 9, 16)):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        exact = x.float() @ wref.float().T
        k = kcta(M, N, K, fp8)
        yc = cuda_gemv(x, w, s, k); yt = pg._launch(x, w, s)
        print(f"{name} M={M:2d} rows={rows_for(N)} k_cta={k:5d} splits={-(-K//k):2d}  cuda {rel(yc, exact):.1e}  triton {rel(yt, exact):.1e}")

print("\nper shape, 24 distinct layers back to back (PDL chain), us:")
for name, N, K, fp8 in SHAPES:
    Ls = [mk(N, K, fp8)[:2] for _ in range(24)]
    roof = sum(w.numel() * w.element_size() for w, _ in Ls) / BW * 1e6
    for M in ((5,) if QUICK else (1, 5, 16)):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        k = kcta(M, N, K, fp8)
        tt = bench(lambda: [pg._launch(x, w, s) for w, s in Ls])
        tc = bench(lambda: [cuda_gemv(x, w, s, k) for w, s in Ls])
        print(f"  {name:5s} M={M:2d} roof {roof:5.0f} | triton {tt:5.0f} ({roof/tt*100:4.1f}%) | cuda {tc:5.0f} ({roof/tc*100:4.1f}%)")
    del Ls; torch.cuda.empty_cache()
