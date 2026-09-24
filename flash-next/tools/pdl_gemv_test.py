"""Test the integrated PDL GEMV ops (vllm/model_executor/layers/pdl_gemv.py) vs Marlin block-FP8 / cuBLAS."""
import sys, types, torch
sys.path.insert(0, "/opt/d/vllm-flash-next-0906")
import vllm.model_executor.layers.pdl_gemv as pg
import vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 as mu
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin

BW = 1.79e12
SHAPES = [("qkvz", 16384, 2560, "fp8"), ("out", 2560, 6144, "fp8"),
          ("hcdn", 336, 10240, "bf16"), ("hcup", 10240, 320, "bf16")]
mu.replace_parameter = lambda layer, name, val: setattr(layer, name, torch.nn.Parameter(val, requires_grad=False))


class L:
    def __init__(self, N, K, kind, keep_ref=False):
        self.N, self.K, self.kind = N, K, kind
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        if kind == "fp8":
            wb = w.float().reshape(N // 128, 128, K // 128, 128)
            s = (wb.abs().amax((1, 3)) / 448.0).clamp(min=1e-12)
            w8 = (wb / s[:, None, :, None]).reshape(N, K).to(torch.float8_e4m3fn)
            self.layer = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K,
                                               weight=torch.nn.Parameter(w8, requires_grad=False),
                                               weight_scale_inv=torch.nn.Parameter(s, requires_grad=False),
                                               weight_block_size=[128, 128], orig_dtype=torch.bfloat16, logical_widths=[])
            pg.keep_fp8_block_copy(self.layer_reg(), w8, s)
            prepare_fp8_layer_for_marlin(self.layer, size_k_first=False)
            self.bytes = N * K
            self.wref = None if not keep_ref else (w8.float().reshape(N // 128, 128, K // 128, 128) * s[:, None, :, None]).reshape(N, K).bfloat16()
        else:
            self.w = w
            self.bytes = N * K * 2
            self.wref = w

    def layer_reg(self):
        mod = torch.nn.Module()
        self._mod = mod
        return mod

    def ref(self, x):
        if self.kind == "fp8":
            Lr = self.layer
            return apply_fp8_marlin_linear(input=x, weight=Lr.weight, weight_scale=Lr.weight_scale_inv,
                                           workspace=Lr.workspace, size_n=self.N, size_k=self.K, bias=None)
        return torch.nn.functional.linear(x, self.w)

    def pdl(self, x):
        if self.kind == "fp8":
            Lr = self.layer
            return torch.ops.vllm.pdl_fp8_block_linear(x, self._mod._pdl_w8, self._mod._pdl_s, Lr.weight,
                                                      Lr.weight_scale_inv, Lr.workspace, self.N, self.K)
        return torch.ops.vllm.pdl_bf16_gemv(x, self.w)


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


torch.manual_seed(0)
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
layers = [L(N, K, k, keep_ref=(i == 0)) for i in range(24) for (_, N, K, k) in SHAPES]
for Lx in layers[:4]:
    for m in (1, 5, 20, 64, 65):
        x = torch.randn(m, Lx.K, device="cuda", dtype=torch.bfloat16)
        a, b = Lx.pdl(x), Lx.ref(x)
        exact = x.float() @ Lx.wref.float().T
        print(f"{Lx.kind:4s} [{Lx.N}x{Lx.K}] M={m:2d}  pdl-vs-exact {rel(a, exact):.1e}  ref-vs-exact {rel(b, exact):.1e}")
for m in (5, 20):
    bufs = [torch.randn(m, 16384, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2)]
    def chain(mode):
        def run():
            for i, Lx in enumerate(layers):
                x, y = bufs[i % 2], bufs[(i + 1) % 2]
                y[:, :Lx.N].copy_(getattr(Lx, mode)(x[:, :Lx.K]))
        return run
    roof = sum(Lx.bytes for Lx in layers) / BW * 1e6
    t = {mo: bench(chain(mo)) for mo in ("ref", "pdl")}
    print(f"M={m}: roofline {roof:.0f} us | ref {t['ref']:.0f} us ({roof/t['ref']*100:.1f}%) | pdl {t['pdl']:.0f} us ({roof/t['pdl']*100:.1f}%) | {t['ref']/t['pdl']:.2f}x")
print("per-shape, 24 distinct layers, fixed output buffers (no copies):")
for si, (name, N, K, kind) in enumerate(SHAPES):
    Ls = layers[si::4]
    for m in (5, 20):
        x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
        roof = sum(Lx.bytes for Lx in Ls) / BW * 1e6
        t = {mo: bench(lambda: [getattr(Lx, mo)(x) for Lx in Ls]) for mo in ("ref", "pdl")}
        print(f"  {name:5s} M={m:2d} roof {roof:5.0f} | ref {t['ref']:5.0f} ({roof/t['ref']*100:4.1f}%) | pdl {t['pdl']:5.0f} ({roof/t['pdl']*100:4.1f}%)")
