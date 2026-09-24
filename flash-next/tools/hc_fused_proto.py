"""Prototype: fused hyper-connection mix for decode (roadmap #4).

Reference (production): hc_combine_norm -> cuBLAS down+inject GEMM (+split-K reduce)
-> hc_silu -> cuBLAS up GEMM -> hc_gate_mix              (~6 kernels)
Fused: hc_combine_norm -> K1 (split-K down+inject GEMM, last CTA applies silu)
-> K2 (up GEMM + sigmoid + gated mean over HC streams)   (3 kernels)
"""
import argparse
import torch
import triton
import triton.language as tl
from safetensors import safe_open

from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_combine_norm, _hc_gate_mix, _hc_silu

HC, H, R = 4, 2560, 320
D = HC * H          # 10240
N_TOT = 336         # R + HC + pad (as in vLLM's merged linear)
EPS = 1e-6


@triton.jit
def _k1_down(xn_ptr, w_ptr, acc_ptr, cnt_ptr, lora_ptr, inj_ptr, M, stride_xm,
             K: tl.constexpr, R: tl.constexpr, HC: tl.constexpr, N_TOT: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             K_PER_SPLIT: tl.constexpr, SPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = pid_k * K_PER_SPLIT + kk + tl.arange(0, BLOCK_K)
        x = tl.load(xn_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                    mask=(offs_n[:, None] < N_TOT) & (offs_k[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tile = offs_m[:, None] * N_TOT + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_TOT)
    tl.atomic_add(acc_ptr + tile, acc, mask=mask, sem="relaxed")
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
    if ticket == SPLIT - 1:
        full = tl.load(acc_ptr + tile, mask=mask, other=0.0, volatile=True)
        full = full.to(tl.bfloat16).to(tl.float32)          # match bf16 GEMM output
        xs = full / HC
        silu = xs * tl.sigmoid(xs)
        is_lora = offs_n[None, :] < R
        tl.store(lora_ptr + offs_m[:, None] * R + offs_n[None, :], silu.to(tl.bfloat16),
                 mask=mask & is_lora)
        is_inj = (offs_n[None, :] >= R) & (offs_n[None, :] < R + HC)
        tl.store(inj_ptr + offs_m[:, None] * HC + (offs_n[None, :] - R), full.to(tl.bfloat16),
                 mask=mask & is_inj)
        tl.store(acc_ptr + tile, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=mask)
        tl.atomic_xchg(cnt_ptr + pid_n, 0)


@triton.jit
def _k2_up_mix(lora_ptr, wu_ptr, xn_ptr, out_ptr, M, stride_xm, stride_om,
               H: tl.constexpr, R: tl.constexpr, R_PAD: tl.constexpr, HC: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_h = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_r = tl.arange(0, R_PAD)
    l = tl.load(lora_ptr + offs_m[:, None] * R + offs_r[None, :],
                mask=(offs_m[:, None] < M) & (offs_r[None, :] < R), other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    for s in tl.static_range(HC):
        rows = s * H + offs_h
        w = tl.load(wu_ptr + rows[:, None] * R + offs_r[None, :],
                    mask=(offs_h[:, None] < H) & (offs_r[None, :] < R), other=0.0)
        g = tl.dot(l, tl.trans(w)).to(tl.bfloat16).to(tl.float32)   # match bf16 GEMM output
        x = tl.load(xn_ptr + offs_m[:, None] * stride_xm + rows[None, :],
                    mask=(offs_m[:, None] < M) & (offs_h[None, :] < H), other=0.0)
        acc += tl.sigmoid(g) * x.to(tl.float32)
    acc = acc / HC
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_h[None, :], acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_h[None, :] < H))


class Fused:
    def __init__(self, dev, cfg):
        self.cfg = cfg
        bn = cfg["BN"]
        self.n_tiles = triton.cdiv(N_TOT, bn)
        self.acc = torch.zeros(64, N_TOT, device=dev, dtype=torch.float32)
        self.cnt = torch.zeros(self.n_tiles, device=dev, dtype=torch.int32)
        self.lora = torch.empty(64, R, device=dev, dtype=torch.bfloat16)
        self.inj = torch.empty(64, HC, device=dev, dtype=torch.bfloat16)

    def mix(self, res, block, inj_in, norm_w, wdi, wu):
        out, xn = _hc_combine_norm(res, block, inj_in, norm_w, EPS, HC)
        m = xn.shape[0]
        c = self.cfg
        bm = 16 if m <= 16 else 32 if m <= 32 else 64
        split = c["SPLIT"]
        kps = triton.cdiv(triton.cdiv(D, split), c["BK"]) * c["BK"]
        split = triton.cdiv(D, kps)
        _k1_down[(self.n_tiles, split)](xn, wdi, self.acc, self.cnt, self.lora, self.inj, m, xn.stride(0),
                                        K=D, R=R, HC=HC, N_TOT=N_TOT, BLOCK_M=bm, BLOCK_N=c["BN"],
                                        BLOCK_K=c["BK"], K_PER_SPLIT=kps, SPLIT=split,
                                        num_warps=c["W1"], num_stages=3)
        bi = torch.empty(m, H, device=xn.device, dtype=torch.bfloat16)
        _k2_up_mix[(triton.cdiv(H, c["BH"]),)](self.lora, wu, xn, bi, m, xn.stride(0), bi.stride(0),
                                               H=H, R=R, R_PAD=512, HC=HC, BLOCK_M=bm, BLOCK_H=c["BH"],
                                               num_warps=c["W2"], num_stages=2)
        return out, bi, self.inj[:m]


def reference(res, block, inj_in, norm_w, wdi, wu):
    out, xn = _hc_combine_norm(res, block, inj_in, norm_w, EPS, HC)
    d = torch.nn.functional.linear(xn, wdi)
    lora, inj, _ = d.split([R, HC, N_TOT - R - HC], dim=-1)
    lora = _hc_silu(lora, HC)
    gate = torch.nn.functional.linear(lora, wu)
    return out, _hc_gate_mix(xn, gate, HC), inj


def load_layer(path, layer, role="attn"):
    pre = f"model.language_model.layers.{layer}.{role}_hyper_connection."
    with safe_open(path, "pt", device="cuda") as f:
        down = f.get_tensor(pre + "input_mix_weight_down.weight")
        inj = f.get_tensor(pre + "block_inject_weight.weight")
        up = f.get_tensor(pre + "input_mix_weight_up.weight")
        norm = f.get_tensor(pre + "hc_norm.weight")
    wdi = torch.cat([down, inj, torch.zeros(N_TOT - R - HC, D, device="cuda", dtype=down.dtype)]).contiguous()
    return norm.contiguous(), wdi, up.contiguous()


def bench(fn, reps=10):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/opt/d/models/Qwen3.8-Flash-Next-NVFP4/model-bf16-00012.safetensors")
    a = ap.parse_args()
    torch.manual_seed(0)
    norm, wdi, wu = load_layer(a.ckpt, 5)
    configs = [dict(BN=16, BK=128, SPLIT=40, W1=4, BH=16, W2=4), dict(BN=16, BK=64, SPLIT=80, W1=4, BH=16, W2=4),
               dict(BN=32, BK=128, SPLIT=40, W1=4, BH=16, W2=4), dict(BN=16, BK=128, SPLIT=40, W1=4, BH=32, W2=4),
               dict(BN=16, BK=128, SPLIT=20, W1=4, BH=16, W2=8)]
    # correctness on real weights
    for m in (1, 5, 16):
        res = torch.randn(m, D, device="cuda", dtype=torch.bfloat16)
        blk = torch.randn(m, H, device="cuda", dtype=torch.bfloat16) * 0.5
        inj = torch.randn(m, HC, device="cuda", dtype=torch.bfloat16)
        r_out, r_bi, r_inj = reference(res, blk, inj, norm, wdi, wu)
        f_out, f_bi, f_inj = Fused("cuda", configs[0]).mix(res, blk, inj, norm, wdi, wu)
        rel = lambda x, y: ((x.float() - y.float()).norm() / y.float().norm()).item()
        print(f"M={m:2d} correctness: block_input rel err {rel(f_bi, r_bi):.2e}, injection {rel(f_inj, r_inj):.2e}, "
              f"residual identical {torch.equal(f_out, r_out)}")
    # timing: 48 distinct layers' worth of weights (>L2), as in one decode pass
    L = 48
    Ws = [(norm.clone(), wdi.clone(), wu.clone()) for _ in range(L)]
    for m in (1, 5, 10, 20):
        res = torch.randn(m, D, device="cuda", dtype=torch.bfloat16)
        blk = torch.randn(m, H, device="cuda", dtype=torch.bfloat16) * 0.5
        inj = torch.randn(m, HC, device="cuda", dtype=torch.bfloat16)
        t_ref = bench(lambda: [reference(res, blk, inj, *w) for w in Ws]) / L
        best = None
        for c in configs:
            f = Fused("cuda", c)
            t = bench(lambda: [f.mix(res, blk, inj, *w) for w in Ws]) / L
            if best is None or t < best[0]:
                best = (t, c)
        roof = (wdi.numel() + wu.numel()) * 2 / 1.79e12 * 1e6
        print(f"M={m:2d}: reference {t_ref:6.2f} us/sublayer | fused {best[0]:6.2f} us ({t_ref/best[0]:.2f}x) best={best[1]} | weight-read roofline {roof:.2f} us")


if __name__ == "__main__":
    main()
