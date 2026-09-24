# Flash-Next optimization roadmap (as of 2026-09-24)

Where we are: a decode step is ~12.5–13.5 ms and reads ~12 GB, so ~6.7 ms at the
1.79 TB/s DRAM roofline. That's ~50% of the bandwidth limit. Tuned engines reach
~70–80% on models like this, so the realistic target is ~9–10 ms/step (+30–40%).
Prefill is ~12.7K tok/s and compute-bound (MoE 27%, dense 32%).

Per-step budget (approx.): MoE 2.8 ms (at roofline) · FP8 dense 2.2 · small BF16
layers (HC mixers, shared experts, router) 2.6 · elementwise/routing/norms/other small
kernels ~3.5 (~2,350 kernels/step) · 4 draft passes 2.1 · PLE stall ~1.0 ·
sampling + inter-graph gaps ~1.5.

## Items (gains are estimates for single-stream decode)

| # | Item | What it involves | Est. gain | Effort | Quality risk | Status |
|---|---|---|---|---|---|---|
| 1 | `vm.page-cluster=0` | sysctl: swap-in reads 1 page instead of 8 (random 160 B rows) | 2–4% | minutes | none | **done 2026-09-24** (not persistent: add to /etc/sysctl.d/ to keep it across reboots) |
| 2 | MTP k=5 | `BLOCK_SIZE=48` makes the hybrid attention block 1632 so ring capacity 12 divides it | 0–4% | hours | none | **tested 2026-09-24: no net gain on chat** (accept +8%, step +9%; code +4%). Kept k=4; `BLOCK_SIZE` knob stays in the serve script |
| 3 | 4-bit dense layers (online) | online MXFP4 (`DENSE_QUANT_SCHEME=mxfp4`; the only online 4-bit scheme) instead of FP8 | +7% measured | hours | **+7–9% ppl measured** | **rejected 2026-09-24** |
| 3b | Calibrated NVFP4 dense layers (offline) | ModelOpt calibration of the dense projections to NVFP4, like the experts; new checkpoint | ~5–7% | days | unknown; likely ~1% ppl, needs NLL | idea |
| 4 | Fused hyper-connection block | K1: split-K down+inject GEMM with last-CTA silu; K2c: stream-split up GEMM + sigmoid + gated mean (atomics) | **prototype: 1.21× on HC → ~0.3 ms/step (~2–3%)**; hand-written CUDA at ~80% BW might reach ~5% | days | low (numerics match: 0 at M=1, ~2e-3 at M=5–16) | **prototyped 2026-09-24** (`tools/hc_fused_proto.py`, `hc_proto_v2.py`); not integrated |
| 5 | Fused MoE routing | top-k + align + sort + sum (~236 kernels/step) | ~1% (re-profile: only 0.17 ms critical; it overlaps the GEMMs) | days | low | deprioritized |
| 6 | VRAM hot-row cache for PLE | 1–2 GB of the most-used rows on GPU; misses go via CPU/swap | 6–8% (re-profile: PLE stall still 1.12 ms/step) | days | none (costs KV) | **second priority** |
| 6b | FP8 experts for the MTP draft layer | online FP8 MoE for `mtp.*` experts only (ignore the NVFP4 target experts); draft-only | ~2% (draft MoE is 0.5 ms/step) | hours | none (drafts are verified) | cheap, try first |
| 7 | Sampling/rejection in CUDA graphs | cut ~1.5 ms of eager work + gaps | 3–5% | days | none | |
| 8 | Better drafter (EAGLE-3/DFlash-style, trained on own traffic) | +0.3 accepted tokens/step ≈ +9%; prose accepts only ~2.4 today | 10–30% | weeks | none | |
| 9 | Megakernel (persistent decoder-layer kernel) | route to 70–80% of roofline | up to 30–40% | months | none | |

Prefill: FlashInfer CUTLASS NVFP4 MoE gives +12% prefill but −11% decode, and both
expert layouts (63 GB) can't coexist, so expect only ~5–10% more without new kernels.

Stacking 1–7 realistically takes decode from ~13 → ~10–11 ms/step (+20–30%).

## Re-profile 2026-09-24 (prof6, current config)
Critical-path attribution (GPU-busy time that disappears if a category costs 0; summed kernel
times double-count overlapped streams): dense FP8 2.16 · idle 2.01 (PLE stall 1.12) · MoE 1.92 ·
HC mixers 1.34 + helpers 0.34 + split-K reduce ~0.18 · FP8 lm_head 1.04 · GDN 0.55 ·
draft MoE 0.50 · other BF16 0.45 · shared experts 0.38 · elementwise 0.35 · QSA 0.26 ·
MoE routing 0.17 · norms 0.12 (ms/step, under nsys: 13.8 ms/step vs ~12.6 normally).
The big "elementwise" item (a MulFunctor at 18 µs) is the shared-expert gate multiply running in
parallel with the MoE GEMM, so it's ~free.

## #4 prototype details (2026-09-24, M=5, 48 distinct weight copies, CUDA graph)
Production path 16.7 µs/sublayer: combine_norm 1.3 + cuBLAS down 7.44 (roofline 3.84) + split-K
reduce + silu + cuBLAS up 5.94 (roofline 3.66) + gate_mix ~1.1, with the small kernels already
overlapped by PDL. Fused best: combine_norm 1.3 + K1 6.4 + K2c 6.1 ≈ 13.8 µs. K2 without stream
splitting was *slower* than cuBLAS (rank 320 pads to 512 in `tl.arange`, and only 160 CTAs).
The Triton GEMVs top out ~55–60% of DRAM BW on these ~7 MB matrices.

## Side fixes
- 2026-09-24: online FP8 was also quantizing the **vision tower** (Marlin pads its K=4304).
  `*visual*` is now in `DENSE_QUANT_IGNORE`, so vision is back to BF16 (+0.6 GB). Image test OK.

## Method (what worked)
- Restart with `tools/restart_wait.sh`. It resets systemd's start limit (3 starts/h).
- Warm up after every restart (`ab.py` once), then use the 2nd+ run. The PLE table starts cold.
- Speed: `accept.py` (greedy) and `accept.py --sampled`, 2–3 runs each.
- Quality: `quality.py compare bf16` (40 × 512-token passages; noise ±0.1%) **and**
  `QUALITY_SHORT=1 quality.py compare bf16head` (300 × 60-token passages; runs the
  decode-sized paths, M ≤ 64; noise ±0.3%, so repeat it).
- Toggle experiments via a `zz-experiment.conf` drop-in, then delete it.
  New env vars that change the traced graph must be registered in `vllm/envs.py`,
  or the compile cache goes stale.
