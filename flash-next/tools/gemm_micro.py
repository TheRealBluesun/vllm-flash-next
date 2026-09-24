"""Skinny BF16 GEMM microbenchmark on Flash-Next's real dense shapes. Rotates
through enough weight copies (>=640 MB) that nothing is served from the 128 MB L2."""
import torch
dev = "cuda"
shapes = [  # (name, N_out, K_in, count_per_target_step)
    ("gdn.in_proj_qkv", 10240, 2560, 36), ("gdn.in_proj_z", 6144, 2560, 36), ("gdn.out_proj", 2560, 6144, 36),
    ("qsa.q_proj", 12288, 2560, 12), ("qsa.o_proj", 2560, 6144, 12), ("hc.mix_down", 320, 10240, 96),
    ("hc.mix_up", 10240, 320, 96), ("shared.gate/up", 640, 2560, 96), ("lm_head", 248320, 2560, 1)]
BW = 1792e9
def bench(fns, reps=20):
    for f in fns: f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for f in fns: f()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    g.replay(); torch.cuda.synchronize()
    s.record(); g.replay(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / (reps * len(fns)) * 1e3
for lib in ("cublas", "cublaslt"):
    torch.backends.cuda.preferred_blas_library(lib)
    print(f"== {lib}")
    for name, N, K, cnt in shapes:
        mb = N * K * 2
        copies = max(1, int(640e6 // mb))
        Ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) for _ in range(copies)]
        for M in (1, 4):
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            t = bench([lambda W=W: torch.nn.functional.linear(x, W) for W in Ws])
            ideal = mb / BW * 1e6
            print(f"  {name:15s} M={M}  {t:7.1f} us  ideal {ideal:6.1f} us  {ideal/t*100:5.1f}% BW   x{cnt}/step -> {t*cnt/1e3:5.2f} ms")
        del Ws; torch.cuda.empty_cache()
