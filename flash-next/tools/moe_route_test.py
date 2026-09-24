"""Fused MoE routing (pdl_ext moe_route) vs topk_softmax + moe_align_block_size."""
import sys, torch
sys.path.insert(0, "/opt/d/vllm-flash-next-0906")
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers import pdl_ext
pdl_ext.load()
E, K = 512, 10

def ref(logits, bs):
    M = logits.shape[0]
    w = torch.empty(M, K, device="cuda", dtype=torch.float32); ids = torch.empty(M, K, device="cuda", dtype=torch.int32)
    src = torch.empty(M, K, device="cuda", dtype=torch.int32)
    ops.topk_softmax(w, ids, src, logits, True)
    s, e, n = moe_align_block_size(ids, bs, E, None, ignore_invalid_experts=True)
    return w, ids, src, s, e, n

def fused(logits, bs):
    M = logits.shape[0]
    w = torch.empty(M, K, device="cuda", dtype=torch.float32); ids = torch.empty(M, K, device="cuda", dtype=torch.int32)
    src = torch.empty(M, K, device="cuda", dtype=torch.int32)
    numel = M * K
    max_pad = numel + E * (bs - 1)
    if numel < E: max_pad = min(numel * bs, max_pad)
    s = torch.empty(max_pad, device="cuda", dtype=torch.int32); e = torch.empty(-(-max_pad // bs), device="cuda", dtype=torch.int32)
    n = torch.empty(1, device="cuda", dtype=torch.int32)
    torch.ops.flashnext_pdl.moe_route(logits, w, ids, src, s, e, n, K, True, bs, logits, False)
    return w, ids, src, s, e, n

def segs(ids, s, e, n, bs):
    """expert -> sorted list of pair indices, from the aligned layout."""
    out = {}
    tot = int(n.item()); s = s.tolist(); e = e.tolist(); numel = ids.numel()
    for b in range(tot // bs):
        for p in s[b * bs:(b + 1) * bs]:
            if p != numel: out.setdefault(e[b], []).append(p)
    return {k: sorted(v) for k, v in out.items()}, e[tot // bs:], s[tot:]

torch.manual_seed(0)
for dt in (torch.bfloat16, torch.float32):
    for M in (1, 3, 5, 10, 20, 37, 64):
        for bs in (8, 16):
            logits = (torch.randn(M, E, device="cuda") * 3).to(dt)
            a, b = ref(logits, bs), fused(logits, bs)
            assert torch.equal(a[1], b[1]), ("ids", dt, M)
            assert torch.allclose(a[0], b[0], rtol=1e-5, atol=1e-6), ("w", (a[0] - b[0]).abs().max())
            assert torch.equal(a[2], b[2]), "src"
            assert torch.equal(a[5], b[5]), ("npp", a[5], b[5])
            sa, ta, pa = segs(a[1], a[3], a[4], a[5], bs); sb, tb, pb = segs(b[1], b[3], b[4], b[5], bs)
            assert sa == sb, "segments"
            assert all(x == -1 for x in ta) and all(x == -1 for x in tb), "tail expert ids"
            assert a[3].shape == b[3].shape and a[4].shape == b[4].shape
print("all equal (ids, weights, src rows, padded segments, tails)")

def bench(fn, reps=48):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    g.replay(); torch.cuda.synchronize()
    x, y = torch.cuda.Event(True), torch.cuda.Event(True)
    x.record(); g.replay(); y.record(); torch.cuda.synchronize()
    return x.elapsed_time(y) / reps * 1e3
for M in (1, 5, 20):
    logits = torch.randn(M, E, device="cuda").to(torch.bfloat16)
    print(f"M={M:2d}: reference {bench(lambda: ref(logits, 8)):5.1f} us/layer  fused {bench(lambda: fused(logits, 8)):5.1f} us/layer")
# padding rows
for M in (5, 8, 16):
    logits = torch.randn(M, E, device="cuda").to(torch.bfloat16)
    pad = torch.zeros(M, dtype=torch.bool, device="cuda"); pad[M // 2:] = True
    w = torch.empty(M, K, device="cuda"); ids = torch.empty(M, K, device="cuda", dtype=torch.int32); src = torch.empty_like(ids)
    ops.topk_softmax(w, ids, src, logits, True, is_padding=pad)
    s0, e0, n0 = moe_align_block_size(ids, 8, E, None, ignore_invalid_experts=True)
    w2 = torch.empty_like(w); ids2 = torch.empty_like(ids); src2 = torch.empty_like(ids)
    s = torch.empty_like(s0); e = torch.empty_like(e0); n = torch.empty_like(n0)
    torch.ops.flashnext_pdl.moe_route(logits, w2, ids2, src2, s, e, n, K, True, 8, pad, True)
    assert torch.equal(ids, ids2) and torch.allclose(w, w2, rtol=1e-5, atol=1e-6) and torch.equal(n0, n)
    assert segs(ids, s0, e0, n0, 8)[0] == segs(ids2, s, e, n, 8)[0]
print("padding rows: equal")
