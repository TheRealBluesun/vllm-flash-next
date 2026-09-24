"""Level-1 prototype: PDL weight-prefetch GEMV chain vs today's Marlin/cuBLAS chain.

Per "layer" (decode M=5), four dependent skinny GEMMs shaped like Flash-Next's decode path:
  qkvz  FP8  [16384 x 2560]    (GDN in_proj_qkvz)
  out   FP8  [2560  x 6144]    (GDN out_proj)
  hcdn  BF16 [336   x 10240]   (HC down+inject)
  hcup  BF16 [10240 x 320]     (HC up)
Each GEMM reads its input from the previous GEMM's output buffer (a real dependency).

Variants:
  ref      : Marlin W8A16 (FP8 layers) + cuBLAS (BF16 layers)       -- production today
  tri      : split-K Triton GEMV, no PDL
  tri_pdl  : same, but each CTA loads its whole weight tile BEFORE griddepcontrol.wait
             and triggers dependents early, so kernel N+1 streams weights while N finishes.
"""
import sys, types, torch, triton, triton.language as tl

sys.path.insert(0, "/opt/d/vllm-flash-next-0906")
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
import vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 as mu

M = 5
BW = 1.79e12
SHAPES = [("qkvz", 16384, 2560, "fp8"), ("out", 2560, 6144, "fp8"),
          ("hcdn", 336, 10240, "bf16"), ("hcup", 10240, 320, "bf16")]


@triton.jit
def _gemv_splitk(x_ptr, w_ptr, s_ptr, acc_ptr, cnt_ptr, y_ptr, M, N, K, stride_x, stride_y,
                 HAS_SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, SPLIT: tl.constexpr, USE_PDL: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    # 1) weights first: independent of the previous kernel's output
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # let the next kernel start prefetching
        tl.extra.cuda.gdc_wait()                # now wait for our input to be complete
    # 2) activations
    x = tl.load(x_ptr + offs_m[:, None] * stride_x + offs_k[None, :],
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
    acc = tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    if HAS_SCALE:
        acc = acc * tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
    tile = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(acc_ptr + tile, acc, mask=mask, sem="relaxed")
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
    if ticket == SPLIT - 1:
        full = tl.load(acc_ptr + tile, mask=mask, other=0.0, volatile=True)
        tl.store(y_ptr + offs_m[:, None] * stride_y + offs_n[None, :], full.to(tl.bfloat16), mask=mask)
        tl.store(acc_ptr + tile, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=mask)
        tl.atomic_xchg(cnt_ptr + pid_n, 0)


def cfg_for(N, K, kind):
    # one K-chunk per CTA so the whole weight tile is loaded before the wait (~8-16 KB/CTA)
    bk = 512 if kind == "fp8" else 256
    bn = 16 if N >= 1024 else 16
    return bn, bk


class Layer:
    def __init__(self, N, K, kind, dev="cuda"):
        self.N, self.K, self.kind = N, K, kind
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        if kind == "fp8":
            amax = w.float().abs().amax(1).clamp(min=1e-12)
            self.scale = (amax / 448.0).float().contiguous()
            self.w = (w.float() / self.scale[:, None]).to(torch.float8_e4m3fn).contiguous()
            L = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K,
                                      weight=torch.nn.Parameter(self.w.clone(), requires_grad=False),
                                      weight_scale=torch.nn.Parameter(self.scale.clone(), requires_grad=False),
                                      orig_dtype=torch.bfloat16, logical_widths=[])
            mu.replace_parameter = lambda layer, name, val: setattr(layer, name, torch.nn.Parameter(val, requires_grad=False))
            prepare_fp8_layer_for_marlin(L, size_k_first=False)
            self.marlin = L
        else:
            self.w, self.scale, self.marlin = w.contiguous(), None, None
        bn, bk = cfg_for(N, K, kind)
        self.bn, self.bk = bn, bk
        self.split = triton.cdiv(K, bk)
        self.acc = torch.zeros(16, N, device=dev, dtype=torch.float32)
        self.cnt = torch.zeros(triton.cdiv(N, bn), device=dev, dtype=torch.int32)

    def ref(self, x, y):
        xin = x[:, :self.K]
        if self.kind == "fp8":
            out = apply_fp8_marlin_linear(xin, self.marlin.weight, self.marlin.weight_scale,
                                          self.marlin.workspace, self.N, self.K, None)
        else:
            out = torch.nn.functional.linear(xin, self.w)
        y[:, :self.N].copy_(out)

    def tri(self, x, y, pdl):
        grid = (triton.cdiv(self.N, self.bn), self.split)
        _gemv_splitk[grid](x, self.w, self.scale if self.scale is not None else self.w, self.acc, self.cnt, y,
                           M, self.N, self.K, x.stride(0), y.stride(0),
                           HAS_SCALE=self.scale is not None, BLOCK_M=16, BLOCK_N=self.bn, BLOCK_K=self.bk,
                           SPLIT=self.split, USE_PDL=pdl, num_warps=4, num_stages=1, launch_pdl=pdl)


def bench(fn, reps=5):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); g.replay(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1e3


def main():
    torch.manual_seed(0)
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    layers = [Layer(N, K, kind) for _ in range(n_layers) for (_, N, K, kind) in SHAPES]
    bufs = [torch.randn(M, 16384, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2)]
    # correctness: triton vs ref per shape
    rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
    for L in layers[:4]:
        ya = torch.zeros_like(bufs[1]); yb = torch.zeros_like(bufs[1])
        L.ref(bufs[0], ya); L.tri(bufs[0], yb, pdl=True); torch.cuda.synchronize()
        print(f"{L.kind:4s} [{L.N}x{L.K}] rel err vs ref {rel(yb[:, :L.N], ya[:, :L.N]):.1e}")

    def chain(mode):
        def run():
            for i, L in enumerate(layers):
                x, y = bufs[i % 2], bufs[(i + 1) % 2]
                if mode == "ref":
                    L.ref(x, y)
                else:
                    L.tri(x, y, pdl=(mode == "tri_pdl"))
        return run

    total_bytes = sum(L.w.numel() * L.w.element_size() for L in layers)
    roof = total_bytes / BW * 1e6
    res = {m: bench(chain(m)) for m in ("ref", "tri", "tri_pdl")}
    n = len(layers)
    print(f"\nchain of {n} GEMVs ({total_bytes/1e9:.2f} GB weights, M={M}); roofline {roof:.0f} us")
    for m, t in res.items():
        print(f"  {m:8s} {t:8.0f} us  ({roof/t*100:4.1f}% of DRAM roofline, {t/n:5.2f} us/GEMV)")
    print(f"  tri_pdl vs ref: {res['ref']/res['tri_pdl']:.2f}x   tri_pdl vs tri: {res['tri']/res['tri_pdl']:.2f}x")


if __name__ == "__main__":
    main()
