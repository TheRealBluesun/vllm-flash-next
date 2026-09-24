"""HC prototype v2 kernels: unpadded-rank K2b, stream-split K2c, and K1 sweeps."""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, "/opt/d/flash-next-perf")
from hc_fused_proto import load_layer, bench, reference, Fused, _k1_down, _hc_combine_norm, HC, H, R, D, N_TOT, EPS

@triton.jit
def _k2b(lora_ptr, wu_ptr, xn_ptr, out_ptr, M, stride_xm, stride_om, H: tl.constexpr, R: tl.constexpr,
         HC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M); offs_h = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    for s in tl.static_range(HC):
        rows = s * H + offs_h
        g = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
        for r0 in tl.static_range(0, R, BLOCK_R):
            offs_r = r0 + tl.arange(0, BLOCK_R)
            l = tl.load(lora_ptr + offs_m[:, None] * R + offs_r[None, :], mask=offs_m[:, None] < M, other=0.0)
            w = tl.load(wu_ptr + rows[:, None] * R + offs_r[None, :])
            g += tl.dot(l, tl.trans(w))
        g = g.to(tl.bfloat16).to(tl.float32)
        x = tl.load(xn_ptr + offs_m[:, None] * stride_xm + rows[None, :], mask=offs_m[:, None] < M, other=0.0)
        acc += tl.sigmoid(g) * x.to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_h[None, :], (acc / HC).to(tl.bfloat16), mask=offs_m[:, None] < M)

@triton.jit
def _k2c(lora_ptr, wu_ptr, xn_ptr, acc_ptr, cnt_ptr, out_ptr, M, stride_xm, stride_om, H: tl.constexpr, R: tl.constexpr,
         HC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_R: tl.constexpr):
    pid = tl.program_id(0); s = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M); offs_h = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    rows = s * H + offs_h
    g = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    for r0 in tl.static_range(0, R, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        l = tl.load(lora_ptr + offs_m[:, None] * R + offs_r[None, :], mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(wu_ptr + rows[:, None] * R + offs_r[None, :])
        g += tl.dot(l, tl.trans(w))
    g = g.to(tl.bfloat16).to(tl.float32)
    x = tl.load(xn_ptr + offs_m[:, None] * stride_xm + rows[None, :], mask=offs_m[:, None] < M, other=0.0)
    part = tl.sigmoid(g) * x.to(tl.float32)
    tile = offs_m[:, None] * H + offs_h[None, :]; msk = offs_m[:, None] < M
    tl.atomic_add(acc_ptr + tile, part, mask=msk, sem="relaxed")
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
    if ticket == HC - 1:
        full = tl.load(acc_ptr + tile, mask=msk, other=0.0, volatile=True)
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_h[None, :], (full / HC).to(tl.bfloat16), mask=msk)
        tl.store(acc_ptr + tile, tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32), mask=msk)
        tl.atomic_xchg(cnt_ptr + pid, 0)

norm, wdi, wu = load_layer("/opt/d/models/Qwen3.8-Flash-Next-NVFP4/model-bf16-00012.safetensors", 5)
L = 48; Ws = [(norm.clone(), wdi.clone(), wu.clone()) for _ in range(L)]
m = 5
res = torch.randn(m, D, device="cuda", dtype=torch.bfloat16); blk = torch.randn(m, H, device="cuda", dtype=torch.bfloat16) * .5
inj = torch.randn(m, HC, device="cuda", dtype=torch.bfloat16)
out, xn = _hc_combine_norm(res, blk, inj, norm, EPS, HC)
lora = (torch.randn(m, R, device="cuda") * .3).to(torch.bfloat16)
ref = None
gate = torch.nn.functional.linear(lora, wu)
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_gate_mix
ref = _hc_gate_mix(xn, gate, HC)
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
bi = torch.empty(m, H, device="cuda", dtype=torch.bfloat16)
acc2 = torch.zeros(64, H, device="cuda", dtype=torch.float32); cnt2 = torch.zeros(4096, device="cuda", dtype=torch.int32)
print("--- K2 variants (up GEMM + gate mix), reference cuBLAS up + gate_mix:",
      f"{bench(lambda: [_hc_gate_mix(xn, torch.nn.functional.linear(lora, w[2]), HC) for w in Ws]) / L:.2f} us")
for BH, BR, W in [(16, 64, 4), (16, 64, 2), (32, 64, 4), (16, 32, 2), (32, 32, 4)]:
    f = lambda w: _k2b[(triton.cdiv(H, BH),)](lora, w, xn, bi, m, xn.stride(0), bi.stride(0), H=H, R=R, HC=HC, BLOCK_M=16, BLOCK_H=BH, BLOCK_R=BR, num_warps=W, num_stages=2)
    f(wu); e = rel(bi, ref)
    print(f"K2b BH={BH} BR={BR} W={W}: {bench(lambda: [f(w[2]) for w in Ws]) / L:6.2f} us  err {e:.1e}")
for BH, BR, W in [(16, 64, 4), (16, 64, 2), (32, 64, 4), (16, 32, 2)]:
    f = lambda w: _k2c[(triton.cdiv(H, BH), HC)](lora, w, xn, acc2, cnt2, bi, m, xn.stride(0), bi.stride(0), H=H, R=R, HC=HC, BLOCK_M=16, BLOCK_H=BH, BLOCK_R=BR, num_warps=W, num_stages=2)
    f(wu); e = rel(bi, ref)
    print(f"K2c BH={BH} BR={BR} W={W}: {bench(lambda: [f(w[2]) for w in Ws]) / L:6.2f} us  err {e:.1e}")
print("--- K1 variants (down+inject split-K + silu); reference cuBLAS down",
      f"{bench(lambda: [torch.nn.functional.linear(xn, w[1]) for w in Ws]) / L:.2f} us")
fz = Fused("cuda", dict(BN=16, BK=64, SPLIT=80, W1=4, BH=16, W2=4))
for BN, BK, SPLIT, W, S in [(16, 256, 20, 4, 3), (16, 256, 40, 4, 2), (16, 128, 40, 4, 4), (32, 256, 20, 8, 3), (16, 512, 20, 8, 2), (16, 256, 10, 8, 3)]:
    kps = triton.cdiv(triton.cdiv(D, SPLIT), BK) * BK; sp = triton.cdiv(D, kps); nt = triton.cdiv(N_TOT, BN)
    fz.cnt = torch.zeros(nt, device="cuda", dtype=torch.int32)
    f = lambda w: _k1_down[(nt, sp)](xn, w, fz.acc, fz.cnt, fz.lora, fz.inj, m, xn.stride(0), K=D, R=R, HC=HC, N_TOT=N_TOT, BLOCK_M=16, BLOCK_N=BN, BLOCK_K=BK, K_PER_SPLIT=kps, SPLIT=sp, num_warps=W, num_stages=S)
    print(f"K1 BN={BN} BK={BK} SPLIT={sp} W={W} S={S}: {bench(lambda: [f(w[1]) for w in Ws]) / L:6.2f} us")
