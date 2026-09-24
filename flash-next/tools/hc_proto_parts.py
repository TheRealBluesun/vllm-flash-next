"""Per-kernel timing for the HC prototype (M=5, 48 distinct weight copies)."""
import sys, torch, triton
sys.path.insert(0, "/opt/d/flash-next-perf")
from hc_fused_proto import *
from hc_fused_proto import _k1_down, _k2_up_mix, _hc_combine_norm
norm, wdi, wu = load_layer("/opt/d/models/Qwen3.8-Flash-Next-NVFP4/model-bf16-00012.safetensors", 5)
L = 48; Ws = [(norm.clone(), wdi.clone(), wu.clone()) for _ in range(L)]
m = 5
res = torch.randn(m, D, device="cuda", dtype=torch.bfloat16); blk = torch.randn(m, H, device="cuda", dtype=torch.bfloat16) * .5
inj = torch.randn(m, HC, device="cuda", dtype=torch.bfloat16)
out, xn = _hc_combine_norm(res, blk, inj, norm, EPS, HC)
f = Fused("cuda", dict(BN=16, BK=64, SPLIT=80, W1=4, BH=16, W2=4))
def k1(w, c):
    kps = triton.cdiv(triton.cdiv(D, c["SPLIT"]), c["BK"]) * c["BK"]; split = triton.cdiv(D, kps)
    _k1_down[(triton.cdiv(N_TOT, c["BN"]), split)](xn, w, f.acc, f.cnt, f.lora, f.inj, m, xn.stride(0), K=D, R=R, HC=HC,
        N_TOT=N_TOT, BLOCK_M=16, BLOCK_N=c["BN"], BLOCK_K=c["BK"], K_PER_SPLIT=kps, SPLIT=split, num_warps=c["W1"], num_stages=c.get("S1", 3))
bi = torch.empty(m, H, device="cuda", dtype=torch.bfloat16)
def k2(w, c):
    _k2_up_mix[(triton.cdiv(H, c["BH"]),)](f.lora, w, xn, bi, m, xn.stride(0), bi.stride(0), H=H, R=R, R_PAD=512, HC=HC,
        BLOCK_M=16, BLOCK_H=c["BH"], num_warps=c["W2"], num_stages=c.get("S2", 2))
roof = lambda t: t.numel() * 2 / 1.79e12 * 1e6
print(f"combine_norm      {bench(lambda: [_hc_combine_norm(res, blk, inj, w[0], EPS, HC) for w in Ws]) / L:6.2f} us")
print(f"cuBLAS down       {bench(lambda: [torch.nn.functional.linear(xn, w[1]) for w in Ws]) / L:6.2f} us (roofline {roof(wdi):.2f})")
lora = torch.randn(m, R, device="cuda", dtype=torch.bfloat16)
print(f"cuBLAS up         {bench(lambda: [torch.nn.functional.linear(lora, w[2]) for w in Ws]) / L:6.2f} us (roofline {roof(wu):.2f})")
for c in [dict(BN=16, BK=64, SPLIT=80, W1=4), dict(BN=16, BK=128, SPLIT=80, W1=4), dict(BN=16, BK=64, SPLIT=160, W1=4),
          dict(BN=16, BK=64, SPLIT=80, W1=2), dict(BN=16, BK=128, SPLIT=40, W1=8, S1=4)]:
    print(f"K1 {str(c):58s} {bench(lambda: [k1(w[1], c) for w in Ws]) / L:6.2f} us")
for c in [dict(BH=16, W2=4), dict(BH=16, W2=2), dict(BH=32, W2=4), dict(BH=16, W2=4, S2=1), dict(BH=64, W2=8)]:
    print(f"K2 {str(c):58s} {bench(lambda: [k2(w[2], c) for w in Ws]) / L:6.2f} us")
