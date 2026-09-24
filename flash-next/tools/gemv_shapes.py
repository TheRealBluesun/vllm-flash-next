"""Triton PDL GEMV vs CUDA TMA GEMV on every real decode shape (enough distinct copies to exceed L2)."""
import os, sys, torch
sys.path.insert(0, "/opt/d/vllm-flash-next-0906")
import vllm.model_executor.layers.pdl_gemv as pg
from vllm.model_executor.layers import pdl_ext
pdl_ext.load()
BW = 1.79e12
# name, N, K, fp8, launches per step (target M=5 + draft)
SHAPES = [("gdn qkvz", 16384, 2560, 1, 36), ("gdn out / qsa o", 2560, 6144, 1, 48), ("qsa qkv", 13312, 2560, 1, 12),
          ("hc down+inj", 336, 10240, 0, 96), ("hc up", 10240, 320, 0, 96), ("shared gate_up", 1280, 2560, 0, 48),
          ("shared down", 2560, 640, 0, 48), ("router", 512, 2560, 0, 48), ("indexer", 640, 2560, 0, 12),
          ("mtp fc", 2560, 2560, 1, 0)]
def mk(N, K, fp8):
    if fp8:
        return torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn), torch.rand(-(-N // 128), -(-K // 128), device="cuda") * 1e-3
    return torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02, None
def bench(fn, reps=5):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3
tot = {"tri": 0, "cuda": 0, "best": 0}
print(f"{'shape':16s} {'M':>2} {'us/call tri':>11} {'cuda':>6} {'%roof tri':>9} {'cuda':>6}")
for name, N, K, fp8, per_step in SHAPES:
    nbytes = N * K * (1 if fp8 else 2)
    copies = max(8, int(300e6 // nbytes))
    Ls = [mk(N, K, fp8) for _ in range(copies)]
    for M in (5, 1):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        pg._CUDA_OK = False
        t = bench(lambda: [pg._launch(x, w, s, 1) for w, s in Ls]) / copies
        pg._CUDA_OK = None; os.environ["VLLM_PDL_GEMV_CUDA"] = "1"
        import vllm.envs as envs
        c = bench(lambda: [pg._launch(x, w, s, 1) for w, s in Ls]) / copies
        roof = nbytes / BW * 1e6
        print(f"{name:16s} {M:2d} {t:11.2f} {c:6.2f} {roof/t*100:8.0f}% {roof/c*100:5.0f}%")
        if M == 5:
            tot["tri"] += t * per_step; tot["cuda"] += c * per_step; tot["best"] += min(t, c) * per_step
    del Ls; torch.cuda.empty_cache()
print({k: f"{v:.0f} us/step (target M=5)" for k, v in tot.items()})
