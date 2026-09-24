# Flash-Next performance project: full report (2026-09-23 → 2026-09-24)

This is the entry point if you come back to this work. It covers the goal, the
starting point, every change (what it is, how it's toggled, what it measured),
what didn't work and why, where the time goes now, how to operate and measure the
setup, and the traps we hit. Deeper logs: `FINDINGS.md` (chronological lab notes),
`ROADMAP.md` (item table with estimates, status and per-item details), `SUMMARY.md`
(overnight snapshot, 2026-09-24 morning).

---

## 1. Goal and constraints

- **Model:** Qwen3.8-Flash-Next, NVFP4 checkpoint (`/opt/d/models/Qwen3.8-Flash-Next-NVFP4`).
  48 layers: 36 GDN linear-attention + 12 QSA sparse-attention. Hyper-connections (4 streams).
  MoE with 512 experts, top-10 (NVFP4 experts; dense projections are BF16 in the checkpoint).
  Per-layer n-gram embedding table ("PLE", layer 1): ~320M rows × 160 B, ~48 GB in FP8.
  A native MTP draft layer is used for speculative decoding (k=4).
- **Hardware:** one RTX PRO 6000 Blackwell (sm_120, 96 GB, ~1.79 TB/s, 188 SMs, 128 MB L2,
  ~100 KB shared memory per SM, no tcgen05/TMEM). An RTX 2080 SUPER drives the desktop.
  91 GB RAM + 64 GB swap, 32 CPUs.
- **Hard constraint (user's choice):** the PLE table stays in **pageable memory / swap**, not
  pinned, so the workstation stays usable. This rules out upstream vLLM's pinned-memory UVA
  offload (vllm-project/vllm#54371) and keeps us on the fork's CPU-worker offload design.
- **Engine:** vLLM fork `peakcrosser7/vllm`, branch `release/qwen38next_offload` @ `357e054`,
  plus this branch (`flash-next-tuning`). The fork's author has moved on from the CPU-worker
  design; upstream main can't page the table out.
- **Goal:** make decode and prefill as fast as possible on this card without giving up
  noticeable quality, with the table in swap.

## 2. Results (same benchmark, `tools/bench.py`, median of 3 runs)

| | Baseline (tree untuned) | After overnight tuning | Final | Final vs baseline |
|---|---|---|---|---|
| Prefill 1K / 8K / 32K / 100K (tok/s) | 5.8K / 7.6K / 7.8K / 7.9K | 8.6K / 12.2K / 12.5K / 12.7K | 10.1K / 12.1K / 12.9K / 12.9K | **+59–75%** |
| Time to first token, 100K prompt | 13.0 s | 8.1 s | 8.0 s | −39% |
| Chat decode, short answer / code / prose (tok/s) | 195 / 197 / 125 | 298 / 335 / 198 | 311 / 367 / 217 | **+59–86%** |
| Time per decode step | 19.0 ms | 13.0 ms | ~11.4–11.6 ms | **−39%** |
| 2 / 4 parallel 8K requests, total (tok/s) | 127 / 135 | 209 / 253 | 225 / 259 | +78% / +91% |
| Pure decode, 1 / 2 / 4 streams (`conc.py`) | — | 245 / 372 / 558 | 270 / 427 / 595 | |

Raw runs: `results/20260923-220156-baseline-quiet.json`, `results/20260924-063917-share.json`
(overnight), `results/20260924-152137-route.json` (final). Decode tok/s depends on how many
draft tokens are accepted on a given text; **ms per step** is the stable metric.

**Quality:** +0.4–0.65% perplexity on long passages and ~+0.3% on short (decode-path)
passages vs the BF16-dense model, all from the FP8 dense weights (+ ~0.1% from the FP8 target
lm_head). Every other change is bit-exact or only affects draft tokens (which the target verifies).

For reference, published SGLang numbers on the same card (Pennyroyal: 171 tok/s single-stream,
427 at 4 streams, 10.1K tok/s prefill at 64K; SSHdotCodes 155–180 tok/s) are below these.

## 3. What's deployed, and how to toggle it

The service is the systemd **user** unit `vllm-flash-next.service`
(`~/.config/systemd/user/`, copy in `deploy/`), launching `serve-qwen38-flash-next.sh`.
All tuning lives in the drop-in `vllm-flash-next.service.d/tuning.conf` (copy in
`deploy/tuning.conf`). Every code change is inert unless its env var is set, so deleting
`tuning.conf` restores stock behavior. Each toggle is registered in `vllm/envs.py` (see §8).

| Toggle | What it does | Effect | Code |
|---|---|---|---|
| `VLLM_PLE_PREFETCH=1` | PLE worker requests all pages of a gather up front with `process_madvise(MADV_WILLNEED)`, so swap-ins run in parallel | PLE stall ~4.3 → ~1 ms/step | `vllm/v1/ple_offload/prefetch.py`, `ple_layer.py` |
| `VLLM_PLE_PREFILL_HINT=1` | On admission, send the rest of the prompt to the PLE worker to prefetch its rows while earlier chunks compute | prefill TTFT down | `connector.py`, `worker.py` |
| `VLLM_PLE_NP_IDS=1` | NumPy n-gram ids in the worker (≤256 tokens) | 2× faster than torch; bit-identical | `ple_layer.py` |
| `VLLM_PLE_PY_IDS=1` | Pure-Python n-gram ids for ≤32 tokens (decode); only computes the context positions each token needs | ids 0.16 → 0.07 ms/request; bit-identical (6000-layout fuzz) | `ple_layer.py` |
| `DENSE_QUANT=fp8_per_block_static` + `DENSE_QUANT_IGNORE=…` | Online FP8 (weight-only, 128×128 blocks) for the BF16 dense layers; tiny layers, router, shared experts, indexer, HC inject and the vision tower stay BF16 | biggest single decode win; **the only quality-affecting change** (+~0.5% ppl) | serve script / `--quantization-config` |
| `VLLM_QSA_QKV_ONLINE_QUANT=1` | Extends online FP8 to the QSA fused qkv projection | small decode gain | `qsa.py` |
| `VLLM_DRAFT_HEAD_FP8=1` + `VLLM_DRAFT_VOCAB_FILE=…npy` | Draft lm_head as an FP8 Triton GEMV over the 98K most-used tokens (99.6% chat coverage) | draft phase ~2.9 → ~2.1 ms/step, acceptance flat; English-centric | `fp8_draft_head.py`, `speculator.py`, `data/draft_vocab_chat96k.npy` |
| `VLLM_LM_HEAD_FP8=1` | FP8 target lm_head for decode-sized batches | −0.5 ms/step, ~+0.1% ppl | `logits_processor.py` |
| `VLLM_NARROW_GEMM=1` | Split-K Triton GEMM for BF16 linears with ≤256 outputs (cuBLAS used 8 CTAs) | 14 → 3 µs per such layer | `narrow_gemm.py` |
| `VLLM_DRAFT_ONLINE_QUANT=1` | The MTP draft layer never received the online quant config; attach FP8 dense + FP8-block experts to it | −0.3–0.4 ms/step (~3%) | `mtp.py` |
| `VLLM_PDL_GEMV=1` | **Hand-written PDL weight-prefetch GEMV** (see §4) for FP8 dense + wide BF16 linears at M ≤ 64; plus early PDL triggers in the HC kernels and a PDL copy of the GDN post-conv CUDA kernel | −0.5 (corpus) / −0.8 (chat) ms/step; costs 2.7 GB of KV cache (280K → 200K tokens, still > 184K max context) | `pdl_gemv.py`, `pdl_ext/`, `ops/hc.py`, `qwen_gdn_linear_attn.py`, `scaled_mm/marlin.py` |
| `VLLM_L2_DRAFT=1` | L2 eviction hints: target GEMVs, both FP8 heads and the draft's Triton MoE load weights `evict_first`, so the draft layer's ~118 MB stays L2-resident across its 4 passes | −0.07 ms/step | `pdl_gemv.py`, `fp8_draft_head.py`, `fused_moe.py`, `mtp.py` |
| `VLLM_FUSED_ROUTE=1` | **Fused MoE routing kernel**: one CTA does softmax top-k + the whole `moe_align_block_size` layout for M ≤ 64 | 7.4 → 3.7 µs/layer; −0.2 ms/step; exact routing | `pdl_ext/moe_route.cu`, `fused_moe/flashnext_route.py`, router + align hooks |
| `MAX_NUM_BATCHED_TOKENS=8192` | Bigger prefill chunks (was 2048) | +7% prefill | serve script |
| `SPEC_TOKENS=4` | MTP k=4 (was 3) | +7–9% chat | serve script |
| Built but **off**: `VLLM_PDL_GEMV_CUDA`, `VLLM_HC_FUSED`, `VLLM_PLE_DRAFT_HINT`, `VLLM_SMALL_FP8_LAYERS`, `VLLM_PLE_PREFETCH_MIN_ROWS` | see §5 | | |

System setting: `vm.page-cluster=0` (swap-in reads 1 page instead of 8 for random 160-byte
rows) was set by hand and is **not persistent**; add it to `/etc/sysctl.d/` to keep it.

## 4. The hand-written kernel work (day 2)

**A. PDL weight-prefetch GEMV** (`vllm/model_executor/layers/pdl_gemv.py`). Decode linears at
M ≈ 5 are pure weight streaming. Each CTA loads its whole weight tile *before*
`griddepcontrol.wait` and triggers dependents right away, so with programmatic dependent
launch the next kernel streams its weights while the current one finishes. Split-K partials
reduce with fp32 atomics; the last CTA per tile writes the output and re-zeroes its
workspace (per-stream, fixed size so CUDA graphs never see it replaced). FP8 uses one scale
per 128×128 block, broadcast via `tl.reshape`. The first version was *slower* (0.84×)
because it gathered one fp32 scale per weight element.
- Prototype (`tools/pdl_chain_proto.py`, `pdl_chain_sweep.py`): a 96-GEMV chain went from
  64% (Marlin + cuBLAS) to 78% of DRAM roofline.
- In the server, gains appeared only after making neighbours PDL-friendly: the HC Triton
  kernels now trigger right after their wait, and the GDN decode post-conv CUDA kernel got an
  early `griddepcontrol.launch_dependents` via a runtime-built copy
  (`pdl_ext/gdn_post_conv_pdl.cu`, bit-identical output and state).

**B. CUDA TMA GEMV** (`pdl_ext/gemv_tma.cu`, `VLLM_PDL_GEMV_CUDA`, **off**). 2D tensor-map
TMA ring (boxes of [rows][128 B], 128B swizzle), `mma.m16n8k16` bf16 with exact e4m3→bf16
conversion, intra-CTA split-K for few-row layers, fewest global splits, grid capped at resident
CTAs. On the real shapes in isolation: qkvz 91% of roofline vs Triton 82%, qsa qkv 93 vs 84,
shared-down 87 vs 63 (`tools/gemv_shapes.py`). In the server: no net gain (GEMV time went up by
about what the neighbours saved), so it stays off. Path to 89% in `ROADMAP.md` (#10b).

**C. L2-resident draft layer** (`VLLM_L2_DRAFT`). The persistence API isn't captured by
CUDA graphs, so it uses per-load eviction hints. Finding from `tools/l2_draft_sim.py`:
`evict_last` on the draft weights protects nothing; everything *else* must stream
`evict_first` (even 31 MB of default-policy traffic between passes undoes it).

**Fused MoE routing** (`pdl_ext/moe_route.cu`, `VLLM_FUSED_ROUTE`). Replaces `topkGating` +
`moe_align_block_size` + `count_and_sort_expert_tokens` (+ fill) with one 1024-thread CTA:
warp-per-token softmax, top-k by `redux.sync` arg-max on float-bit keys (lowest id wins ties,
like the reference), then counts, padded scan, expert ids and scatter; padding-row aware.
Phase timing showed top-k was the cost (3.2 µs) until per-lane contiguous experts + vector
loads + redux cut it to ~1.9 µs. Integration: top-k runs as a registered custom op
(`vllm.flashnext_fused_topk_softmax`) that records the align layout under the `topk_ids`
buffer address; `moe_align_block_size` takes it on an exact (address, shape, block size, E)
match; the non-fused top-k paths invalidate their buffer's address. Verified in nsys:
`moe_route_kernel` 52.6/step, reference routing kernels 0/step.

The runtime-built extension (`pdl_ext/__init__.py`) compiles `bindings.cpp`,
`gdn_post_conv_pdl.cu`, `gemv_tma.cu` and `moe_route.cu` with nvcc into
`~/.cache/vllm/flashnext_pdl` on first use (~10 s) and registers `torch.ops.flashnext_pdl.*`.

## 5. Tried and not adopted (why)

| Idea | Result |
|---|---|
| Online MXFP4 dense layers | +7% speed but +7–9% ppl: rejected |
| W8A8 FP8 dense (b12x / CUTLASS) | CUTLASS fails on sm_120; b12x slower and +1.35% ppl |
| b12x NVFP4 MoE | crashes in CUDA-graph capture, not faster eager |
| FlashInfer CUTLASS NVFP4 MoE | +12% prefill, −11% decode, +1.6% ppl; both expert layouts don't fit |
| 64K / 70K draft vocab | acceptance loss ate the gain; 98K is the sweet spot |
| FP8 copies of HC mixers + shared experts (`VLLM_SMALL_FP8_LAYERS`) | +3.5% speed for +0.6–0.85% ppl: off |
| MTP k=5 (`BLOCK_SIZE=48`) | acceptance +8% but step +9%: no chat gain |
| Fused hyper-connection op (`VLLM_HC_FUSED`) | 1.14× isolated, no server gain: off |
| VRAM hot-row cache for PLE | simulation: steps with zero misses only 4.6% (chat) — every step still needs the CPU: rejected |
| Draft-pass PLE hints (`VLLM_PLE_DRAFT_HINT`) | no gain (advisory prefetch only): off |
| Trained drafter (PixelML DFlash for this model) | +3.9% aggregate, slower on chat: not pursued |
| Sampling in CUDA graphs (#7) | scoped at ≤0.9 ms of gaps; intrusive: deferred (and later profiling showed most idle is the PLE stall, below) |
| CUDA TMA GEMV (B) | faster in isolation, ~0 in server: off |
| Skip PLE prefetch for small requests | helps warm chat, but cold corpus n-grams then fault serially (gather ~1.7 ms): off |
| "Fix the non-PDL `elementwise_kernel` break" | it's on the shared-expert side stream, off the critical path: no gain |
| SGLang | published numbers on this card already below ours: not set up |

## 6. Where the time goes now (prof9, ~12.6 ms/step under nsys)

Serialized main-stream time per decode step: PDL GEMVs 3.5 ms (~80% of roofline) · Marlin
MoE 2.9 (near roofline for the experts touched) · **GPU idle ~1.9** · FP8 heads 1.1 (~87% of
roofline) · GDN recurrence 0.7 · MoE routing ~0.3 after fusion · `hc_combine_norm` 0.45.

Idle breakdown: **~0.93 ms is one gap per step, the PLE stall** — the forward graph waits in
`cuStreamWaitValue32` for the CPU worker's rows. Worker time per decode request is now
~0.33 ms (IPC 0.07, ids 0.07, madvise 0.10, gather 0.07); the rest is hand-off latency
(D2H + event → connector thread → zmq → worker → H2D + semaphore). The other ~0.95 ms is
dozens of small gaps.

## 7. If you resume: remaining levers (each ~1–3% unless noted)

1. **Cut PLE hand-offs**: have the worker poll a GPU-written host flag and read inputs from
   pinned memory directly, dropping the connector thread and zmq (−0.1 to −0.3 ms).
2. **Fold `hc_combine_norm`** into its predecessor (−0.2 to −0.3 ms). After an out/o-proj
   GEMV it finishes 5.25 µs after the GEMV; after the MoE add ~3.7 µs. Needs a cross-CTA RMS
   reduction in the GEMV epilogue, or residual prefetch before the PDL wait (only safe for
   chains where every earlier kernel is known complete).
3. **Turn on B** only after 2 (it needs back-to-back PDL chains to pay off).
4. **Calibrated NVFP4 dense checkpoint** (offline ModelOpt): ~5–7% decode, quality unknown.
5. **Megakernel** (persistent decoder-layer kernel): the only big jump left (up to 30–40%),
   a large project.

## 8. Operating and measuring

- **Restart:** `tools/restart_wait.sh` (daemon-reload, resets the start limit of 3/h, restarts,
  waits for the API). Startup ~4–6 min; the PLE table pages back in over the first runs, so
  warm up (`tools/ab.py` once, then discard 2–3 `accept.py` runs) before measuring.
- **Experiments:** put toggles in a `zz-experiment.conf` drop-in, then delete it.
- **Speed:** `tools/accept.py` (greedy, fixed prompts; prints accept length and **ms/step**),
  `--sampled` for default sampling; `tools/bench.py` (full suite → `results/`);
  `tools/conc.py` (1/2/4 streams). Compare medians of ≥4 warm runs; noise is ~±0.2 ms.
  Stop Hermes (`systemctl --user stop hermes-gateway hermes-serve-tailnet`) while benchmarking.
- **Quality:** `tools/quality.py compare bf16` (40 × 512-token passages, runs prefill paths;
  noise ±0.2%) and `QUALITY_SHORT=1 tools/quality.py compare bf16head` (300 × 60-token
  passages, runs the decode-size paths; noise ±0.3%, repeat it).
- **Profiling:** stop the service, `tools/prof_launch.sh <dir>` (nsys with the drop-in env),
  `tools/capture.py` (decode range + 16K prefill range), then stop the vLLM server process
  (SIGTERM its PID, don't `pgrep -f` from a shell whose command line contains the pattern),
  wait for nsys to write, restart the service. Analysis: `tools/analyze.py` (categories),
  `tools/stream_time.py` (serialized time per kernel on the main stream),
  `tools/chain_breaks.py` (PDL overlap per predecessor), PLE timing via `VLLM_PLE_TIMING=1`.
- **Kernel tests:** `tools/pdl_gemv_test.py`, `tools/gemv_tma_test.py`, `tools/gemv_shapes.py`,
  `tools/moe_route_test.py`, `tools/test_ple_np_ids.py`, `tools/l2_draft_sim.py`.
- **Repo:** private `TheRealBluesun/vllm-flash-next`, branch `flash-next-tuning`. This
  workstation has no credentials for that account: `git bundle create X <remote-sha>..flash-next-tuning`,
  copy it to the inference server that has them, shallow-clone there, `git fetch` the bundle
  and push. Commits use the TheRealBluesun noreply identity.

## 9. Gotchas learned (read before changing things)

- **Compile cache ignores source edits.** vLLM's AOT/torch.compile cache is keyed on config +
  *registered* env vars. New toggles must be registered in `vllm/envs.py`; after editing code
  that gets traced, move `/opt/d/caches/vllm/torch_compile_cache` aside before restarting.
- **Python in the router can be traced.** An `is_compiling()` guard silently selects the
  reference path inside the compiled graph. Make such decisions inside a custom op.
- **INFO logs emitted during CUDA-graph capture don't appear.** "Never logged" ≠ "never ran":
  confirm kernel paths with an nsys profile.
- **sm_120 + TMA:** `cp.async.bulk{.tensor}` to a `.shared::cluster` destination compiles with a
  guarded software-fallback call, which makes the driver reserve ~14.6 KB of stack per thread
  (**~3.6 GB of VRAM**) at first launch. Use `.shared::cta` destinations. Many small (128–256 B)
  bulk copies are limited by TMA op rate; use 2D tensor-map boxes.
- **PDL:** a kernel only overlaps a predecessor that triggers early; any non-PDL kernel is a
  barrier. Early triggers are safe only if every PDL-launched dependent waits before reading.
- **Environment in workers:** `/proc/<pid>/environ` of EngineCore/Worker doesn't show all
  `VLLM_*` vars, but registered ones do reach the worker.
- **Two GPUs:** without pinning by UUID (`CUDA_VISIBLE_DEVICES` + `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  in the serve script), FlashInfer JITs for sm_75+sm_120 during startup and OOMs the host.
- **Restarts:** systemd allows 3 starts/hour (`restart_wait.sh` resets it); SIGINT is ignored
  mid-startup; a clean stop takes ~16 s.
- **Benchmark noise:** the machine is a desktop; Hermes, the PLE worker's swap activity and a
  busy `cinnamon` add noise. The first 2–3 runs after a restart are slow (cold table).

## 10. Upstream and sharing

- Posted the swap-prefetch findings on vllm-project/vllm#54070 (disk-backed PLE tables) as
  TheRealBluesun. The `/start_profile` crash fix was already upstream (#55237).
- Candidates if the fork design ever lands upstream: PLE swap prefetch, FP8 draft head with a
  vocab subset, online quant for the MTP draft layer, the fused routing kernel (generic for
  softmax top-k at small M), and the narrow split-K GEMM.

## 11. Timeline

- **2026-09-23:** updated the fork tree, made it a systemd user service, fixed the 2080 SUPER
  startup OOM, baseline benchmarks + nsys profiles.
- **Overnight 09-23 → 24:** PLE prefetch/hints/NumPy ids, online FP8 dense + QSA qkv, FP8 draft
  head with 98K vocab, FP8 target head, narrow GEMM, batch 8192, MTP k=4 (`SUMMARY.md`).
- **09-24 morning:** repo created, upstream comment posted, roadmap; `vm.page-cluster=0`;
  k=5 and MXFP4 tested and dropped; re-profile; HC fusion prototype; FP8 draft layer (6b);
  PLE cache sim; draft hints; trained-drafter research.
- **09-24 afternoon/evening:** hand-written kernels — PDL GEMV (A, adopted), L2 draft (C,
  adopted), CUDA TMA GEMV (B, off); re-profiles; fused MoE routing (adopted); PLE worker
  breakdown and Python ids (adopted). Stopped here by choice.
