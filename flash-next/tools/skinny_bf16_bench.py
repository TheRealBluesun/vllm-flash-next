"""Split-K Triton skinny GEMM for small BF16 layers (decode M<=16) vs cuBLAS."""
import torch, triton, triton.language as tl
@triton.jit
def _k(x_ptr, w_ptr, y_ptr, M, N, K, sxm, swn, sym, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, KPS: tl.constexpr, SPLIT: tl.constexpr):
    pn = tl.program_id(0); pk = tl.program_id(1)
    om = tl.arange(0, BM); on = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, KPS, BK):
        ok = pk * KPS + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + om[:, None] * sxm + ok[None, :], mask=(om[:, None] < M) & (ok[None, :] < K), other=0.0)
        w = tl.load(w_ptr + on[:, None] * swn + ok[None, :], mask=(on[:, None] < N) & (ok[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    ptrs = y_ptr + om[:, None] * sym + on[None, :]; msk = (om[:, None] < M) & (on[None, :] < N)
    if SPLIT == 1: tl.store(ptrs, acc.to(y_ptr.dtype.element_ty), mask=msk)
    else: tl.atomic_add(ptrs, acc, mask=msk, sem="relaxed")
NSM = torch.cuda.get_device_properties(0).multi_processor_count
def cfg(M, N, K, BN=None, BK=None):
    BN = BN or (32 if N <= 1024 else 64); BK = BK or 128
    tiles = triton.cdiv(N, BN); split = 1
    while tiles * split < 2 * NSM and K // (BK * split * 2) >= 1: split *= 2
    kps = triton.cdiv(triton.cdiv(K, split), BK) * BK; split = triton.cdiv(K, kps)
    return BN, BK, split, kps
def lin(x, w, BN=None, BK=None, warps=4):
    M, K = x.shape; N = w.shape[0]; BN, BK, split, kps = cfg(M, N, K, BN, BK)
    if split == 1:
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    else:
        y = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    _k[(triton.cdiv(N, BN), split)](x, w, y, M, N, K, x.stride(0), w.stride(0), y.stride(0), BM=16, BN=BN, BK=BK, KPS=kps, SPLIT=split, num_warps=warps, num_stages=3)
    return y if split == 1 else y.to(x.dtype)
def bench(fns, reps=20):
    for f in fns: f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for f in fns: f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); g.replay(); e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / (reps * len(fns)) * 1e3
shapes = [("hc.mix_down", 320, 10240), ("hc.mix_up", 10240, 320), ("shared.gate_up", 1280, 2560), ("shared.down", 2560, 640),
          ("gdn.in_proj_ba", 96, 2560), ("hc.block_inject", 4, 10240), ("router.gate", 512, 2560), ("qsa.qkv", 13312, 2560)]
for name, N, K in shapes:
    copies = max(1, int(400e6 // (N * K * 2)))
    Ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
    x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
    err = ((lin(x, Ws[0]).float() - torch.nn.functional.linear(x, Ws[0]).float()).norm() / torch.nn.functional.linear(x, Ws[0]).float().norm()).item()
    tc = bench([lambda w=w: torch.nn.functional.linear(x, w) for w in Ws])
    best = min(((bench([lambda w=w: lin(x, w, bn, bk, nw) for w in Ws]), bn, bk, nw) for bn in (16, 32, 64) for bk in (64, 128, 256) for nw in (2, 4)), key=lambda r: r[0])
    ideal = N * K * 2 / 1792e9 * 1e6
    print(f"{name:16s} [{N}x{K}] cuBLAS {tc:6.1f}us ({ideal/tc*100:3.0f}%) | triton {best[0]:6.1f}us ({ideal/best[0]*100:3.0f}%) BN={best[1]} BK={best[2]} warps={best[3]} err {err:.1e}", flush=True)
    del Ws; torch.cuda.empty_cache()
