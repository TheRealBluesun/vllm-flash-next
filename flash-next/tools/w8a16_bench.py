"""Compare cuBLAS BF16, Marlin FP8 (vLLM), and Triton W8A16 on Flash-Next dense shapes."""
import sys, types, torch
sys.path.insert(0, "/opt/d/flash-next-perf")
from w8a16 import w8a16_gemm, quantize_per_channel
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin, apply_fp8_marlin_linear
BW = 1792e9
shapes = [("shared.gate_up", 1280, 2560), ("shared.down", 2560, 640), ("router.gate", 512, 2560), ("qsa.indexer_qk", 640, 2560), ("qsa.k_or_v", 512, 2560), ("mtp.fc", 2560, 2560)]
def bench(fns, reps=10):
    for f in fns: f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            for f in fns: f()
    g.replay(); torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); g.replay(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / (reps * len(fns)) * 1e3
def marlin_layer(wq, scale):
    L = types.SimpleNamespace()
    L.output_size_per_partition, L.input_size_per_partition = wq.shape
    L.weight = torch.nn.Parameter(wq, requires_grad=False); L.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    L.orig_dtype = torch.bfloat16; L.logical_widths = []
    def rp(layer, name, val): setattr(layer, name, torch.nn.Parameter(val, requires_grad=False))
    import vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 as mu; mu.replace_parameter = rp
    prepare_fp8_layer_for_marlin(L, size_k_first=False); return L
torch.manual_seed(0)
for name, N, K in shapes:
    copies = max(1, int(600e6 // (N * K)))  # >=600MB of fp8 to defeat L2
    Wb = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(min(copies, max(1, int(1.2e9 // (N*K*2)))))]
    Q = [quantize_per_channel(w) for w in Wb]
    ML = [marlin_layer(wq.clone(), s.clone()) for wq, s in Q] if N * K < 1e8 else []
    for M in (1, 4, 16):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, Wb[0])
        tri = w8a16_gemm(x, *Q[0]); err = ((tri.float() - ref.float()).norm() / ref.float().norm()).item()
        t_bf = bench([lambda w=w: torch.nn.functional.linear(x, w) for w in Wb])
        t_tr = bench([lambda q=q: w8a16_gemm(x, *q) for q in Q])
        t_ma = bench([lambda L=L: apply_fp8_marlin_linear(x, L.weight, L.weight_scale, L.workspace, N, K, None) for L in ML]) if ML else float("nan")
        ideal8 = N * K / BW * 1e6
        print(f"{name:15s} M={M:2d} bf16 {t_bf:7.1f}us | marlin-fp8 {t_ma:7.1f}us ({ideal8/t_ma*100:4.0f}%BW) | triton-fp8 {t_tr:7.1f}us ({ideal8/t_tr*100:4.0f}%BW) relerr {err:.4f}", flush=True)
    del Wb, Q, ML; torch.cuda.empty_cache()
