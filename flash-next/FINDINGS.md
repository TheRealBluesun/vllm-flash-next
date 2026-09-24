# Flash-Next perf findings (2026-09-23, tree 357e054 + local patches)

Tools here: `bench.py` (suite; results in `results/`), `capture.py` (nsys ranges),
`gemm_micro.py` (skinny BF16 GEMMs, L2-busting), `ple_threads.py` (PLE worker
thread states + major faults), `prof/fn.{1,2}.nsys-rep` (decode / 16K prefill).

Caveat: until Hermes cron jobs moved to .15, :8000 had outside traffic; the
baseline has some rep-to-rep noise, and acceptance numbers (from global
counters) may include other requests. Re-baseline on a quiet endpoint.

## Baseline (results/20260923-213540-baseline.json)
- Prefill ~7.5–8.6K tok/s flat from 8K to 100K ctx (TTFT 1.1 s @8K, 12–13 s @100K).
- Decode 112–229 tok/s depending on MTP acceptance; step rate ~48–59/s at every
  context length, so context depth is nearly free.
- C=4 × 8K: 150 tok/s aggregate (includes the 4 prefills).

## Decode step (~21 ms = target + MTP draft)
| bucket | ms/step |
|---|---|
| BF16 dense GEMMs (GDN/QSA projections, hyper-connection mixers, shared expert) + BF16 lm_head GEMV (target 1× + MTP 3×) | ~10.4 |
| GPU idle, one stall per step at the PLE layer (after `_hc_combine_kernel`) | ~4.3 |
| Marlin MoE (NVFP4 routed experts) | ~3.3 |
| rest (GDN core, QSA, norms, routing, sampling) | ~2.5 |

- cuBLAS isn't mis-picking kernels: standalone the same shapes reach 70–83% of
  1.79 TB/s (small HC/shared-expert shapes 35–58%). The cost is BF16 bytes:
  ~8.5 GB target + ~4.3 GB for 3 MTP passes (lm_head 1.27 GB each) per step.
- PLE stall = page faults on swapped table rows, served serially: the worker
  has 1 thread in D-state during decode (~17 major faults/token).

## Prefill (16K prompt, 2.4 s)
- ~101 ms GPU idle per 2048-token chunk at the same PLE point, about 1.0 s of 2.4 s.
- 61K major faults; gather runs up to 16 threads in D-state, but isn't
  overlapped with GPU work.
- Of GPU time: Marlin MoE 36%, BF16 GEMMs 29%, QSA sparse attention 11%.

## Candidate fixes (PLE table stays in swap, per user)
1. PLE decode: prefault the needed rows in parallel (thread pool / higher
   in-flight I/O) before index_select. Expected: ~4 ms → sub-ms per step (~15–20% decode).
2. PLE prefill: prefetch the next chunk's rows while the current chunk computes,
   and raise fault concurrency past 16. Expected: up to ~40% prefill.
3. FP8 (weight-only) for BF16 dense layers + lm_head: ~halves ~10 ms of GEMM time.
   Needs a quality check. Truncated-vocab lm_head for the MTP drafter is an
   alternative for the 3×1.27 GB draft reads.
4. Config: max-num-batched-tokens (prefill), MTP k, async scheduling w/ spec decode.

## Side issues
- Upstream bug fixed locally: `AsyncLLM.profiler` unset when profiler != torch
  (vllm/v1/engine/async_llm.py) → /start_profile 500s. Worth a PR.
- `systemctl --user stop` hit the 90 s timeout → SIGKILL once; direct SIGTERM to
  each process exits cleanly in <60 s. Investigate.

---
# Overnight log (2026-09-23 → 24)

Quiet-endpoint baseline: `results/20260923-220156-baseline-quiet.json`.
A/B harness: `ab.py` (fresh text: 3× 16K prefill + 4× 1K-ctx decode),
`accept.py` (fixed prompts: MTP acceptance), `quality.py` (NLL on 40 fixed
passages + greedy chats vs saved BF16 reference `quality-bf16.json`),
`conc.py` (pure-decode concurrency, counts tokens via logprobs).
Noise floor of quality.py (BF16 vs itself): mean NLL delta −0.0007; greedy
chats aren't deterministic (2/8 identical), so NLL is the metric.

| change | 16K prefill | decode ms/step | quality |
|---|---|---|---|
| baseline (BF16 dense, PLE unchanged) | 2.06 s | 19.8 | ref |
| + PLE swap prefetch (`VLLM_PLE_PREFETCH=1`) | 1.68 s | 18.0 | identical math |
| + prefill hints (`VLLM_PLE_PREFILL_HINT=1`) | 1.52 s | 18.0 | identical math |
| + FP8 dense W8A16 Marlin (`DENSE_QUANT=fp8_per_block_static`) | 1.43 s | 15.8 | +0.60% ppl |

Rejected:
- `fp8_per_channel_static`: online PTPC needs W8A8; Marlin refused.
- `fp8_per_tensor_static`: same speed as per-block, +0.61% ppl.
- b12x NVFP4 MoE (`MOE_BACKEND=b12x`): illegal memory access during CUDA-graph
  capture; in eager mode, prefill was no faster than Marlin → not pursued.
- b12x / CUTLASS block-FP8 W8A8 dense: CUTLASS fails on SM120 ("Invalid status");
  b12x+Marlin mix (VLLM_DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel) was
  slower (16.35 vs 15.45 ms/step chat) and +1.35% ppl.
- 70K-token draft vocab subset (`VLLM_DRAFT_VOCAB_FILE`): +9% on code, 0% on chat
  (acceptance fell 2.8%) → off; the code stays in, disabled.

Current decode step (FP8 dense, nsys `prof2/`): lm_head 3.3 ms (3 draft GEMV + 1
target), FP8-Marlin dense 3.2 ms (only ~39% of BW at M=4!), MoE 2.6 ms (near
roofline), idle 2.65 ms (4 small gaps/step, was 1× 4.3 ms PLE stall), HC mixers
~1.25 ms (BF16, not quantized), QSA fused qkv 0.63 ms (still BF16).

Community numbers on 1× RTX PRO 6000 (SGLang forks): Pennyroyal v2.1 C1 171 tok/s,
C4 427 aggregate, 64K prefill 10.1K tok/s; v2.5 online FP8 207 tok/s short ctx.
SSHdotCodes SGLang 0.5.20: 155–180 tok/s (sampled, T=1). Both use online FP8 +
a 65.8K "hot-token" draft vocab.
Ours now (FP8 dense config): C1 217 tok/s, C2 270, C4 466 aggregate (chat, greedy,
512 out); 16K prefill 11.5K tok/s.

## Overnight continued (after 23:30)

More wins (all in `tuning.conf`):
- Online FP8 had also hit tiny layers (router 512-wide, GDN in_proj_ba 96-wide,
  block_inject 4-wide, shared experts), where Marlin FP8 is 2–4× *slower* than BF16
  (e.g. router 16.4 µs vs 4.8). `DENSE_QUANT_IGNORE` keeps them BF16: FP8-Marlin
  time fell from 3.23 → 1.83 ms/step, and quality improved (router stays BF16).
- FP8 draft lm_head (`VLLM_DRAFT_HEAD_FP8`, Triton GEMV at ~1.5 TB/s): chat 15.45 → 14.9 ms/step.
- Narrow BF16 GEMM (`VLLM_NARROW_GEMM`, split-K Triton, N ≤ 256): in_proj_ba 13.8 → 2.8 µs.
- `MAX_NUM_BATCHED_TOKENS=8192`: 16K prefill 1.43 → 1.33 s (KV 267K → 220K tokens).
- MTP k=4 (k=5 unsupported: "QSA ring capacity 12 must divide block size"):
  greedy chat 208 → 226, sampled chat 190–195 → 206–209 tok/s. k=2 worse.
- NumPy n-gram ids (`VLLM_PLE_NP_IDS`, fuzz-verified bit-identical over 3000
  layouts, `test_ple_np_ids.py`): ids 0.26 → 0.14 ms per decode step.
- QSA fused qkv_proj to FP8 (`VLLM_QSA_QKV_ONLINE_QUANT`; my earlier ignore pattern
  `*v_proj` glob-matched the fused name `...qkv_proj` itself, since it ends in "v_proj", so vLLM
  correctly ignored the whole layer. My mistake, not a vLLM bug: a partial *shard* match
  raises an error instead): chat ~227 → ~244 tok/s.

Tried and rejected:
- 64K hot-token draft vocab ranked from the model's own chat output (98.9% coverage):
  acceptance −3.5%, step time only −0.2 ms → no net gain on chat.
- FlashInfer CUTLASS NVFP4 MoE (works on SM120 now): prefill +12% (16K 1.19 s) but
  decode −11% and +1.6% ppl (W4A4) → off. Option for prefill-heavy use.
- FP8 target lm_head (`VLLM_LM_HEAD_FP8`, shares the draft copy): ~+1–2%, but my
  NLL check can't see it (prompt logprobs run M=512 > 64), so it's off.
- Async scheduling: already on by default with MTP.

Final state (FP8 dense config, k=4), warm:
- accept.py greedy: chat ~244 tok/s (3.4 tok/step, 14.0 ms/step), corpus 226–259.
- accept.py sampled (T=1, top_p .95, top_k 20): chat 216–229 tok/s.
- bench.py: see `results/20260924-012333-tuned-k4.json` (before the qkv-FP8 and np-ids changes).
- Quality: +0.4–0.65% perplexity vs BF16-dense (from FP8 dense only; everything else
  is exact or draft-only).

Remaining per-step budget (~14–15 ms): target forward ~8.5 ms (MoE 2.6, FP8 dense
~1.5, BF16 small layers + HC ~2.5, PLE wait ~1, misc), drafts ~2.9 ms (4 passes),
eager sampling/rejection ~1.8 ms, idle ~0.75 ms. About 14.5 GB read per step,
so ~8 ms at DRAM roofline → ~55% of bandwidth. The rest is launch-bound small
kernels (~2,100/step), CPU gaps, and the swap latency floor.

Machine note: `cinnamon` was using ~100% of one core all night, which adds noise to
CPU-sensitive steps.

## Late addition: 96K hot-token draft vocab (adopted)
nsys (prof4 vs prof5): draft phases 2.96 → 2.06 ms/step (head GEMV 392 → 160 µs/pass).
The 64K subset lost 3.5% acceptance, but 98,320 tokens (99.6% held-out chat coverage)
keep acceptance flat: greedy chat 239–247 → 252–262 tok/s, sampled chat 227–232 →
232–246. Vocab = top-98,304 by (3 × chat freq from `gen_chat_corpus.py` output +
code/doc freq) ∪ special tokens → `draft_vocab_chat96k.npy`. English-centric: other
languages would see lower acceptance (outputs are unaffected).
Stop timing: a clean `systemctl --user stop` takes ~16 s; the earlier 90 s timeouts
happened only while an instance was still starting up or had crashed.
- FP8 target lm_head (`VLLM_LM_HEAD_FP8`) adopted after a proper test: `QUALITY_SHORT=1
  quality.py` (300 × 60-token passages, so prompt logprobs run with M ≤ 64 and use the
  FP8 head): +0.21% ppl vs a +0.13% self-compare noise floor. Chat greedy 252–262 →
  261–282, sampled 232–246 → 239–258 tok/s.
- Small-layer FP8 (`VLLM_SMALL_FP8_LAYERS`) rejected: HC+shared was faster (~3.5%) but
  short-passage NLL +0.85%; shared-only was within noise (the short test's noise is ~±0.3%;
  back-to-back runs of one config gave +0.24/+0.51).
- Found: vLLM's compile cache ignores unregistered env vars, so an env-only toggle reused
  a stale graph (crash: AttributeError '_w8a16_w'). Fixed by registering the toggles
  in `vllm/envs.py`. The final config was verified on a fresh cache:
  greedy chat 266–279, sampled 238–258, NLL long +0.41% / short −0.02%, stress OK,
  pure decode 245/372/558 tok/s at C=1/2/4.

## 2026-09-24 follow-ups
- `vm.page-cluster=0` (user): 16K prefill 1.33 → 1.27 s; decode change within noise.
- MTP k=5 via `BLOCK_SIZE=48` (attention block 1616 → 1632): accepted tokens +8% but step
  time +9%, so chat is flat (greedy 262–274 vs 267–276, sampled 235–252 vs 240–254), code +4%,
  prefill slightly worse. Kept k=4.
- Online MXFP4 dense (`DENSE_QUANT_SCHEME=mxfp4`, Marlin W4A16): chat 293–295 tok/s (+7%) but
  NLL +9.1% long / +7% short. Rejected. The weight-key form `{"weight":"mxfp4"}` fails
  (it maps to the dynamic key); the scheme shorthand works. The vision tower needs `*visual*` in
  the ignore list (K=4304 isn't a multiple of 32).
- Found: the FP8 config had been quantizing the vision tower too. Now excluded; image test OK.

## 2026-09-24 round 2 (6b, 4, 6c, 7)
- 6b adopted: the draft (MTP) layer had no online quant at all (its quant config is rebuilt from
  the draft model config without --quantization-config). `VLLM_DRAFT_ONLINE_QUANT=1` gives it FP8
  dense + FP8-block experts (Triton FP8 MoE). Warm: greedy chat 275–288, sampled 259–269 tok/s.
- 4: `VLLM_HC_FUSED` op (ops/hc_fused.py) is correct (bf16-rounding-level diffs; atomics make
  it non-bit-deterministic) and 1.14× in isolation, but no measurable server gain. Off.
- 6 rejected by simulation (`tools/ple_cache_sim.py`): chat steps are almost never all-hit.
- 6c `VLLM_PLE_DRAFT_HINT`: hints fire, but no gain. Off.
- 7 scoped only: ~0.9 ms/step of CPU-launch gaps in eager sampling/rejection/draft-prep sections.
- Long-passage NLL noise is ~±0.2% run to run (seen +0.58 and +0.87 on configs with the same
  target math).
