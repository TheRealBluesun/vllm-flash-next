"""Sweep per-shape tile configs for the PDL GEMV chain."""
import sys, torch, triton
sys.path.insert(0, "/opt/d/flash-next-perf")
import pdl_chain_proto as P
torch.manual_seed(0)
n_layers = 24
layers = [P.Layer(N, K, kind) for _ in range(n_layers) for (_, N, K, kind) in P.SHAPES]
bufs = [torch.randn(P.M, 16384, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2)]
W = {}  # per-shape (bn, bk, warps)
def apply(shape_idx, bn, bk, warps):
    for i, L in enumerate(layers):
        if i % 4 == shape_idx:
            L.bn, L.bk, L.split = bn, bk, triton.cdiv(L.K, bk)
            L.cnt = torch.zeros(triton.cdiv(L.N, bn), device="cuda", dtype=torch.int32)
            L.warps = warps
orig_tri = P.Layer.tri
def tri(self, x, y, pdl):
    grid = (triton.cdiv(self.N, self.bn), self.split)
    P._gemv_splitk[grid](x, self.w, self.scale if self.scale is not None else self.w, self.acc, self.cnt, y,
                         P.M, self.N, self.K, x.stride(0), y.stride(0), HAS_SCALE=self.scale is not None,
                         BLOCK_M=16, BLOCK_N=self.bn, BLOCK_K=self.bk, SPLIT=self.split, USE_PDL=pdl,
                         num_warps=getattr(self, "warps", 4), num_stages=1, launch_pdl=pdl)
P.Layer.tri = tri
def chain():
    for i, L in enumerate(layers):
        L.tri(bufs[i % 2], bufs[(i + 1) % 2], True)
total = sum(L.w.numel() * L.w.element_size() for L in layers); roof = total / P.BW * 1e6
cands = {"fp8": [(16, 512, 4), (32, 512, 4), (32, 256, 4), (64, 256, 8), (16, 1024, 8), (32, 1024, 8), (64, 512, 8)],
         "bf16": [(16, 256, 4), (32, 256, 4), (16, 512, 8), (32, 128, 4), (64, 128, 4), (32, 512, 8)]}
best = {}
for si, (name, N, K, kind) in enumerate(P.SHAPES):
    res = []
    for c in cands[kind]:
        if c[1] > K and K % c[1]:
            continue
        apply(si, *c)
        res.append((P.bench(chain), c))
    res.sort()
    best[si] = res[0][1]; apply(si, *res[0][1])
    print(f"{name:5s} [{N}x{K}] best {res[0][1]} -> chain {res[0][0]:.0f} us | worst {res[-1][1]} {res[-1][0]:.0f} us", flush=True)
t = P.bench(chain)
print(f"tuned tri_pdl chain: {t:.0f} us = {roof/t*100:.1f}% of roofline ({roof:.0f} us); per-shape configs {best}")
