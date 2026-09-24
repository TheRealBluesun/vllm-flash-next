"""Level 2 sim: can eviction hints keep the MTP draft layer's dense weights in L2?
Per 'step': stream a target weight set T (GEMVs, > L2), then 4 draft passes over D (the draft
layer's dense shapes, ~118 MB). Time the draft passes under different L2 policies."""
import sys, torch
sys.path.insert(0, "/opt/d/vllm-flash-next-0906")
import vllm.model_executor.layers.pdl_gemv as pg
BW = 1.79e12
DRAFT = [(2560, 2560, 1), (2560, 2560, 1), (336, 10240, 0), (10240, 320, 0), (336, 10240, 0), (10240, 320, 0),
         (12288, 2560, 1), (512, 2560, 1), (512, 2560, 1), (2560, 6144, 1), (336, 10240, 0), (10240, 320, 0),
         (512, 2560, 0), (1280, 2560, 0), (2560, 640, 0), (640, 2560, 0)]
def mk(N, K, fp8):
    if fp8:
        return (torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn), torch.rand(N // 128 + 1, K // 128 + 1, device="cuda") * 1e-2)
    return (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02, None)
D = [mk(*s) for s in DRAFT]
dbytes = sum(w.numel() * w.element_size() for w, _ in D)
T = [mk(16384, 2560, 1) for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 40)]  # ~42 MB each
tbytes = sum(w.numel() for w, _ in T)
H = [mk(12288, 2560, 1) for _ in range(8)]
E = [mk(12288, 2560, 1) for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 2)]  # draft experts, normal policy  # ~283 MB: draft head (251) + experts, between passes
x = torch.randn(5, 16384, device="cuda", dtype=torch.bfloat16)
x1 = x[:1]
def gemv(xx, w, s, ev): return pg._launch(xx[:, :w.shape[1]].contiguous(), w, s, ev)
def run(t_ev, d_ev, h_ev, e_ev=0, passes=4):
    g = torch.cuda.CUDAGraph(); ev = [torch.cuda.Event(True) for _ in range(passes + 2)]
    for w, s in T: gemv(x, w, s, t_ev)
    for w, s in D: gemv(x1, w, s, d_ev)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for w, s in T: gemv(x, w, s, t_ev)
    gd = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gd):
        for w, s in D: gemv(x1, w, s, d_ev)
    gh = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gh):
        for w, s in H: gemv(x1, w, s, h_ev)
        for w, s in E: gemv(x1, w, s, e_ev)
    res = []
    evs = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(passes)]
    for rep in range(5):
        g.replay()
        for p in range(passes):
            evs[p][0].record(); gd.replay(); evs[p][1].record(); gh.replay()
        torch.cuda.synchronize()
        res.append([a.elapsed_time(b) for a, b in evs])
    return [sorted(r[i] for r in res)[2] * 1e3 for i in range(passes)]
print(f"draft set {dbytes/1e6:.0f} MB ({dbytes/BW*1e6:.0f} us at DRAM roofline); target stream {tbytes/1e9:.2f} GB")
for name, t_ev, d_ev, h_ev, *rest in [("all normal", 0, 0, 0), ("target first", 1, 0, 0),
                         ("target first, draft last", 1, 2, 0), ("target+head first", 1, 0, 1),
                         ("target+head first, draft last", 1, 2, 1), ("all others first, draft last", 1, 2, 1, 1)]:
    r = run(t_ev, d_ev, h_ev, *rest)
    print(f"  {name:40s} draft pass us: " + "  ".join(f"{v:6.1f}" for v in r) + f"   (sum {sum(r):.0f})")
