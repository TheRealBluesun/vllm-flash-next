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
| 4 | Fused hyper-connection block | K1: split-K down+inject GEMM with last-CTA silu; K2c: stream-split up GEMM + sigmoid + gated mean (atomics) | **prototype: 1.21× on HC → ~0.3 ms/step (~2–3%)**; hand-written CUDA at ~80% BW might reach ~5% | days | low (numerics match: 0 at M=1, ~2e-3 at M=5–16) | **integrated as `VLLM_HC_FUSED` (off)**: 1.14× in isolation, but no measurable gain in the server, within noise → left off |
| 5 | Fused MoE routing | `pdl_ext/moe_route.cu`: one 1024-thread CTA does softmax top-k (warp per token, `redux.sync` arg-max on float-bit keys) + the full `moe_align_block_size` layout (counts, padded scan, expert ids, scatter), padding-row aware, PDL; `fused_topk` stashes the align outputs and `moe_align_block_size` takes them on an exact (buffer, block size, E) match | 7.4 → 3.7 µs/layer isolated; **server −0.2 ms/step (~2%)**: corpus 11.71–11.87 → 11.57–11.69, chat 11.50–11.54 → 11.25–11.34 | ~1.5 h | none: ids/weights/src rows/segments match the reference exactly; short NLL +0.13/+0.28% (unchanged) | **adopted 2026-09-24** (`VLLM_FUSED_ROUTE=1`) |
| 6 | VRAM hot-row cache for PLE | 1–2 GB of the most-used rows on GPU; misses go via CPU/swap | **~0 for chat**: a step skips the CPU only if all ~80 rows hit, and even an unbounded seen-before cache gives zero-miss steps 4.6% (chat) / 36% (code); novel 2/3-grams dominate (`tools/ple_cache_sim.py`) | — | none | **rejected 2026-09-24** |
| 6c | Progressive PLE prefetch hints during drafting | `VLLM_PLE_DRAFT_HINT=1`: after each draft pass, hint the next step's known tokens to the PLE worker | **no gain measured**: gather wait didn't drop (0.54–0.57 vs 0.23–0.48 ms), probably GIL contention with the real request; the remainder is likely DRAM/TLB misses, not swap | ~1 h | none (advisory) | **off 2026-09-24** |
| 6b | FP8 for the MTP draft layer (dense + experts) | the draft's quant config never got --quantization-config; `VLLM_DRAFT_ONLINE_QUANT=1` attaches the target's online spec + FP8-block experts (target NVFP4 untouched) | **+3% measured** (12.6 → 12.3 ms/step; sampled chat 240–254 → 252–266) | ~30 min | none (drafts are verified) | **adopted 2026-09-24** |
| 7 | Sampling/rejection in CUDA graphs | prof6: ~0.70 ms/step of launch gaps inside eager sections + 0.21 ms between graphs; vLLM has no sampler-capture option, so it means per-shape graphs around MRV2 sampling (CPU-side branching on sampling params) | ≤5–7% if all gaps vanish | several hours, intrusive | none, but correctness-sensitive | **scoped 2026-09-24, deferred** |
| 8 | Better drafter (EAGLE-3/DFlash-style) | **Evidence against (2026-09-24):** PixelML trained a 5-layer DFlash drafter on ~98.5K on-policy Flash-Next conversations (`PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash`): only +3.87% aggregate vs native MTP k=4 (CI +2.1..+5.8%), "a maths drafter, a code wash, and it makes chat slower at every block size". Native MTP is "trained with multi-steps" and already accepts ~3.37 at k=4. | ≤ ~4%, negative on chat | many hours of GPU time | none | **not pursued** (only upside left: fine-tune MTP on own chat traffic, which needs a training implementation of the full MTP layer) |
| 10 | **A. PDL weight-prefetch GEMV** (hand-written kernels, level 1) | split-K W8A16/BF16 GEMV that loads its weight tile *before* `griddepcontrol.wait` and triggers dependents early, so kernel N+1 streams weights while N finishes; replaces Marlin FP8 dense + cuBLAS HC/small BF16 GEMMs at decode sizes (M ≤ 64) | prototype chain 1.22× (78.2% vs 64.2% of DRAM roofline) on ~4.1 ms/step of critical path → **~0.7 ms (~5–6%)**, less if neighbours don't trigger PDL | ~1–1.5 h | low (numerics = FP8/BF16 rounding) | **adopted 2026-09-24 as `VLLM_PDL_GEMV=1`: −0.5 ms/step corpus, −0.8 chat (≈ −4…−6%)**; C=1/2/4 270/427/595 tok/s; NLL within noise; costs 2.7 GB KV (280K → 200K tokens) |
| 10b | **B. CUDA C++ version of 10** | `pdl_ext/gemv_tma.cu`: 2D tensor-map TMA ring (2 stages × 2 boxes, 128B swizzle, `.shared::cta`), `mma.m16n8k16` bf16 with exact e4m3→bf16, intra-CTA split-K for few-row layers, PDL | isolated chains (M=5): qkvz 89% of roofline (Triton 78%), out 80% (77), HC down 77% (71), HC up 97% (74) → −9% GEMV time; **server: no measurable gain** (neighbours break the PDL chain, and unchained it's no faster than Triton) | ~2 h | none (numerics better than Triton) | **built 2026-09-24, off** (`VLLM_PDL_GEMV_CUDA=1`) |
| 10c | **C. L2-resident draft layer** (level 2) | eviction hints instead of the persistence API: target GEMVs, both FP8 heads and the draft's Triton MoE load weights with `evict_first`, so the draft layer's ~118 MB stays in L2 across its 4 passes | sim: 381 → 215 µs of draft GEMVs per step; **server: −0.07 ms/step (~0.6%)** | ~45 min | none (same math) | **adopted 2026-09-24** (`VLLM_L2_DRAFT=1`) |
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

## Hand-written kernel prototype (2026-09-24, `tools/pdl_chain_proto.py`, `tools/pdl_chain_sweep.py`)
Chain of 24 × {qkvz FP8 16384×2560, out FP8 2560×6144, hcdn BF16 336×10240, hcup BF16 10240×320}
= 96 dependent GEMVs, 1.71 GB weights, M=5, CUDA graph; roofline 953 µs at 1.79 TB/s.

| variant | µs | % of roofline |
|---|---|---|
| ref: Marlin W8A16 + cuBLAS (production) | 1484 | 64.2% |
| Triton split-K GEMV, no PDL | 1409 | 67.7% |
| same + PDL weight prefetch (default tiles) | 1305 | 73.0% |
| same + per-shape tiles | **1218** | **78.2%** |

Best tiles (BLOCK_N, BLOCK_K, warps), one K-chunk per CTA: qkvz (64,512,8), out (64,512,8),
hcdn (32,128,4), hcup (32,256,4). The win is the overlap: weights for kernel N+1 are in flight
while kernel N's tail CTAs and split-K finalize run. Levels beyond: 10b (CUDA/TMA), 10c (L2
residency), then #9 (megakernel) — each moves the step closer to one continuous weight stream.

### #10 integration (2026-09-24, `VLLM_PDL_GEMV=1`, adopted)
- `vllm/model_executor/layers/pdl_gemv.py`: two custom ops. `pdl_fp8_block_linear` (W8A16, 128×128
  block scales, one scale per 128-wide K block per row block, broadcast via `tl.reshape`) and
  `pdl_bf16_gemv`; both fall back (Marlin / F.linear) for M > 64. The Marlin FP8 kernel class keeps
  a row-major FP8 copy before repacking (+2.7 GB). Split-K workspaces are per stream (the shared
  experts run on an aux stream), fixed size so CUDA graphs never see them replaced.
- Isolated, per shape (24 distinct layers, M=5): out 353→272 µs, hcdn 176→127, hcup 134→119,
  qkvz 705→722. But in a chain with a non-PDL kernel between GEMVs the gain vanished (1.01×):
  every non-PDL kernel is a barrier. So the neighbours matter:
  - HC kernels (`ops/hc.py`) now call `gdc_launch_dependents` right after `gdc_wait` (was: before
    the store), so the HC chain + qkvz pipeline.
  - GDN decode post-conv (CUDA, 12 µs, only requests×HV CTAs) gets an early
    `griddepcontrol.launch_dependents` via a runtime-built copy
    (`layers/pdl_ext/`, `torch.ops.flashnext_pdl.gdn_post_conv_mtp`, bit-identical output+state),
    so out_proj streams its weights during the recurrence.
- A/B same session (accept.py, medians): corpus 12.39 → 11.88 ms/step, chat 12.39 → 11.60.
  Sampled chat 259–269 → 272–277 tok/s. Quality: short NLL +0.27/+0.35% (noise ±0.3%),
  long +0.80% (noise ±0.2%, earlier same-math runs +0.58/+0.87).
- The first version was 0.84× — the block-scale gather loaded a fp32 scale per weight element.

### #10c details (L2 residency, `tools/l2_draft_sim.py`)
- The draft layer's dense weights are ~118 MB per pass (63 MB FP8 + 55 MB BF16), plus ~50 MB of experts
  and the 251 MB FP8 draft head. Passes 2–4 already hit L2 when nothing runs between them.
- `evict_last` on the draft weights gives no protection. What matters is that everything else streams
  with `evict_first`: even 31 MB of normal-policy traffic between passes undoes most of the benefit.
- Implemented without the persistence API (which CUDA graphs don't capture): per-load eviction hints in
  `pdl_gemv` (target layers `evict_first`, draft modules tagged `_pdl_draft` keep the default),
  `fp8_draft_head.py` (both heads) and the Triton `fused_moe` B loads (only the draft uses Triton MoE).
  The server gain is smaller than the sim's, since some remaining kernels between passes still use the
  default policy.

### #10b details (CUDA GEMV, `tools/gemv_tma_test.py`)
- Path to 89%: 1D per-row bulk copies (128–256 B) were limited by the TMA *op rate* (~1 copy / 130 clk / SM),
  giving 19–25% of roofline → 2D tensor-map boxes [rows][128 B] with 128B swizzle (conflict-free
  fragment reads) → x read from global/L1 instead of smem (3× occupancy) → 2 stages × 16 KB →
  fewest global K-splits (atomics + ticket + finalize cost more than the lost parallelism), grid capped
  at resident CTAs (2/SM), and intra-CTA split-K (8/16-row tiles, smem reduction) for few-row layers.
- **sm_120 gotcha:** `cp.async.bulk{.tensor}` with a `.shared::cluster` destination compiles to a native
  copy plus a guarded call to `__cuda_syscall_cp_async_bulk_*` (the remote-CTA fallback). Its presence
  makes the driver raise the per-thread stack to 14.6 KB → **3.6 GB of VRAM reserved** at the first
  launch. Use `.shared::cta` destinations (PTX 8.6).
- Why no server gain: in a chain where a non-PDL kernel follows each GEMV (tools/pdl_gemv_test.py),
  CUDA and Triton tie (1476 vs 1481 µs). The advantage only shows when neighbours chain via PDL, and
  in the decode path most GEMVs border inductor elementwise kernels or Marlin MoE. Next lever: fuse or
  PDL-enable those neighbours (or go to #9, the megakernel), then turn this kernel on.

## Re-profile 2026-09-24 afternoon (prof7 current config, prof8 with 10b on)
Serialized main-stream time per decode step (each kernel charged its non-overlapped part; 12.6 ms/step under nsys):
PDL GEMVs 3.52 (~406 launches) · Marlin MoE 2.94 (near roofline for the experts touched) · idle 1.96
(PLE stall + launch gaps) · FP8 heads 1.06 (~87% of roofline) · GDN recurrence 0.72 · MoE routing 0.68
(topkGating 0.26, align 0.18, count/sort 0.07, act_and_mul 0.08, moe_sum 0.09) · hc_combine_norm 0.45 ·
small elementwise ~0.2.
- **The "neighbours break the PDL chain" theory was mostly wrong** (`tools/chain_breaks.py`): 96–100% of GEMVs
  already start before their predecessor ends, including after inductor kernels. GEMV time is exposed
  because it streams weights, not because of chain breaks. Remaining breaks: a non-PDL `elementwise_kernel`
  before ~47 GEMVs/step (~0.4 ms exposed), and non-PDL successors (vectorized_elementwise, topkGating,
  QSA norm/indexer).
- 10b on (prof8): all ~500 GEMVs/step dispatched to the CUDA kernel. GEMV time +0.35 ms but hc_combine_norm
  −0.2 and GDN −0.1 → net ~0. On the real shapes in isolation (`tools/gemv_shapes.py`) CUDA saves ~0.2 ms/step
  (qkvz 91% vs 82%, qsa qkv 93 vs 84, shared down 87 vs 63; loses on router/indexer), but that doesn't
  survive in context. Stays off.
- Bigger remaining levers: MoE routing fusion (~0.68 ms of 5 serialized kernels × 48), idle (PLE ~1.1 +
  launch gaps ~0.9, see #7), hc_combine_norm (0.45, fuse into the preceding kernel).

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
