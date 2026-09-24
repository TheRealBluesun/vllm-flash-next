# Flash-Next on the RTX PRO 6000: overnight tuning summary (2026-09-24)

The service is running the tuned config now: `vllm-flash-next.service` plus the
drop-in `~/.config/systemd/user/vllm-flash-next.service.d/tuning.conf`.
The PLE table is still in swap, as you asked.

## Results: quiet-endpoint baseline → tuned (same benchmark, `bench.py`)

| | baseline | tuned | change |
|---|---|---|---|
| TTFT 8K prompt | 1.08 s | 0.67 s | −38% |
| TTFT 32K | 4.23 s | 2.58 s | −39% |
| TTFT 100K | 12.6 s | 8.0 s | −36% |
| prefill throughput | ~7.6–8.1K tok/s | ~12.3–12.8K tok/s | +60% |
| decode, code | 190 tok/s | 310 tok/s | +64% |
| decode, prose essay | 125 tok/s | 174 tok/s | +39% |
| decode, short answer | 187 tok/s | 267 tok/s | +42% |
| decode at 32K / 100K context | 153 / 196 | 270 / 302 tok/s | +77% / +54% |
| 4× parallel 8K requests (incl. prefill) | 135 tok/s | 257 tok/s | +90% |
| pure decode, 1 / 2 / 4 streams (chat) | — | 245 / 372 / 558 tok/s aggregate | |
| chat, greedy / default sampling (T=1) | — | 266–279 / 238–258 tok/s | |

Quality: +0.4–0.65% perplexity vs the BF16-dense model (40 fixed passages), plus
~0.1% from the FP8 target lm_head. It all comes from the FP8 weights; every other change is exact, or affects only
MTP drafts, which the target verifies. Tool calling checked OK.

For comparison, the published SGLang setups on this same card: Pennyroyal 171
tok/s single-stream (207 with online FP8), 427 aggregate at 4 streams, 10.1K
tok/s prefill at 64K; SSHdotCodes 155–180 tok/s. We're ahead on every axis, so I
didn't set up SGLang.

## What changed (each can be toggled in tuning.conf)

1. **PLE swap prefetch** (`VLLM_PLE_PREFETCH`): the GPU was stalling ~4.3 ms per
   step while the PLE worker faulted rows in from swap one at a time. Now all
   the pages for a gather are requested up front with `process_madvise(WILLNEED)`,
   so they swap in in parallel. The stall is now ~1 ms. The table stays in swap.
2. **Prefill hints** (`VLLM_PLE_PREFILL_HINT`): on admission, the rest of the
   prompt is sent to the PLE worker, which prefetches its rows in the background
   while earlier chunks compute.
3. **NumPy n-gram ids** (`VLLM_PLE_NP_IDS`): 2× faster decode-path id
   computation, fuzz-verified bit-identical (`test_ple_np_ids.py`).
4. **Online FP8 for the BF16 dense layers** (`DENSE_QUANT=fp8_per_block_static`,
   weight-only, Marlin): the checkpoint only quantizes the experts, so ~5 GB of
   BF16 projections were read every step. Tiny layers and the MoE router stay
   BF16 (`DENSE_QUANT_IGNORE`), because Marlin FP8 is slower than BF16 on them.
   `VLLM_QSA_QKV_ONLINE_QUANT` extends FP8 to the attention qkv projection. **This
   is the one change that affects quality (+~0.5% ppl).** Remove these three lines
   if you'd rather have exact BF16 (costs ~15% decode speed).
5. **FP8 draft lm_head over a 98K hot-token subset** (`VLLM_DRAFT_HEAD_FP8` + `VLLM_DRAFT_VOCAB_FILE`): MTP read the 1.27 GB BF16
   lm_head on every draft pass. Now it reads 252 MB, via a Triton FP8 GEMV at ~1.5 TB/s over
   the 98K most-used tokens (99.6% coverage, ranked from the model's own chat output).
   The draft phase went from ~2.9 to ~2.1 ms/step with acceptance unchanged. It's
   English-centric; drop the VOCAB line if you mostly use other languages.
6. **Narrow-GEMM kernel** (`VLLM_NARROW_GEMM`): cuBLAS spent 14 µs on a 96-wide
   layer using 8 CTAs; a split-K Triton kernel does it in 3 µs.
6b. **FP8 target lm_head** (`VLLM_LM_HEAD_FP8`) for decode-sized batches: −0.5 ms/step.
   Verified with 300 short passages (so the FP8 path runs): +0.21% ppl vs a
   +0.13% noise floor, i.e. about +0.1%.
7. **`MAX_NUM_BATCHED_TOKENS=8192`** (was 2048): +7% prefill. KV cache is still
   220K tokens, above the 184K context limit.
8. **MTP k=4** (was 3): +7–9% on chat, greedy and sampled. k=5 isn't supported
   by the QSA kernels.

## Also fixed along the way
- The 2080 SUPER made FlashInfer rebuild for sm_75 → OOM. The serve script now
  pins vLLM to the PRO 6000 by UUID.
- The unit now waits for GPU memory to free before starting. A slow previous
  exit, with 48 GB swapped plus the CUDA context, used to make restarts fail.
- An upstream bug: `/start_profile` crashes with `profiler=cuda`
  (`AsyncLLM.profiler` unset). Fixed locally; worth a PR.

## Tried and rejected (details in FINDINGS.md)
- b12x NVFP4 MoE: crashes during CUDA-graph capture, and wasn't faster.
- W8A8 FP8 dense (b12x/CUTLASS): CUTLASS fails on sm_120; b12x was slower and +1.35% ppl.
- FlashInfer CUTLASS NVFP4 MoE: works now; +12% prefill but −11% decode, +1.6% ppl.
- A 64K hot-token draft vocab lost 3.5% acceptance (no net gain). The adopted 98K
  vocab keeps acceptance flat.
- Triton for the draft's BF16 MoE: slower than FlashInfer.
- FP8 copies of the hyper-connection mixers and shared experts (`VLLM_SMALL_FP8_LAYERS`,
  code kept, off). HC+shared was +3.5% speed for about +0.6% ppl; shared-only was within noise.

## Where the remaining time goes
A decode step is ~14 ms and reads ~14.5 GB, which would take ~8 ms at DRAM
bandwidth, so we're at ~55% of the memory-bandwidth ceiling. The rest is:
- ~2,100 small launch-bound kernels per step (MoE is already at roofline);
- the ~1 ms swap-latency floor of the PLE table;
- ~2.5 ms of sampling/rejection and scheduling between graphs.

The CPU isn't the bottleneck: the GPU worker is idle waiting for work 82% of the
time. Going further means kernel fusion (hyper-connections, shared experts,
norms) or megakernel-style work.

## Important if you toggle settings
vLLM's torch.compile cache only hashes *registered* env vars. I've registered all
the new toggles in `vllm/envs.py`, so changing them now triggers a recompile
(before that, a stale graph once crashed a start). If you ever see a start fail
with an odd `AttributeError` after changing env vars, move
`/opt/d/caches/vllm/torch_compile_cache` aside and restart.
The two old caches I moved aside, `/opt/d/caches/vllm/torch_compile_cache.old-*`, can be deleted.

## Things only you can decide or do
- `sudo sysctl vm.page-cluster=0` would cut swap-in readahead from 8 pages to 1 for
  these random 160-byte row reads, probably shaving part of the remaining ~1 ms PLE
  stall. It's a system-wide setting, so I didn't change it.
- `cinnamon` sat at ~100% of one CPU core all night (possibly related to the
  2080 SUPER setup). It adds noise to vLLM's CPU-side work.
- Rollback: `rm ~/.config/systemd/user/vllm-flash-next.service.d/tuning.conf &&
  systemctl --user daemon-reload && systemctl --user restart vllm-flash-next`
  restores the stock behavior of this tree. The code patches are all inert unless
  their env var is set. The full diff is in `patches/`.

Files: `FINDINGS.md` (full log), `results/*.json` (bench runs), `prof*/` (nsys),
`bench.py` `ab.py` `accept.py` `quality.py` `conc.py` `stress.py` (harness),
`patches/` (vLLM diff vs 357e054, serve script, unit, drop-in).
