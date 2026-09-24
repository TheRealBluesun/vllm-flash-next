# Flash-Next tuning branch

**Start with [`REPORT.md`](REPORT.md)**: full context (goal, every change and toggle, results,
dead ends, where time goes, remaining levers, how to operate and measure, gotchas).

Qwen3.8-Flash-Next (NVFP4) serving on a single RTX PRO 6000 (SM120), with the
~48 GB FP8 n-gram (PLE) table in **pageable** host memory: partly in swap, not pinned.

This branch is `peakcrosser7/vllm` `release/qwen38next_offload` @ `357e054`, plus
the commits below. That fork's CPU-worker PLE offload is the only upstream-derived
design that works with a swappable table. Upstream main moved to pinned-memory UVA
offload (vllm-project/vllm#54371), which can't page out.

Results vs. the same tree untuned (single stream, `tools/bench.py`, medians):

| | Baseline | Final |
|---|---|---|
| Prefill 1K / 8K / 32K / 100K (tok/s) | 5.8K / 7.6K / 7.8K / 7.9K | 10.1K / 12.1K / 12.9K / 12.9K |
| Time to first token, 100K | 13.0 s | 8.0 s |
| Chat decode, short / code / prose (tok/s) | 195 / 197 / 125 | 311 / 367 / 217 |
| Time per decode step | 19.0 ms | ~11.4–11.6 ms |

Quality: ~+0.4–0.6% perplexity, from the FP8 weights only.

## Commits (all behavior is opt-in via env vars; off = stock behavior)

| Commit | Toggle(s) |
|---|---|
| PLE offload: ptrace opt-in; draft `moe_backend` | (always on) |
| AsyncLLM.profiler init (also upstream #55237) | (always on) |
| PLE swap-aware prefetch, prompt hints, NumPy ids | `VLLM_PLE_PREFETCH`, `VLLM_PLE_PREFILL_HINT`, `VLLM_PLE_NP_IDS`, `VLLM_PLE_TIMING` |
| FP8 draft/target lm_head, hot-token draft vocab | `VLLM_DRAFT_HEAD_FP8`, `VLLM_DRAFT_VOCAB_FILE`, `VLLM_LM_HEAD_FP8` |
| Narrow split-K GEMM; small-layer W8A16 | `VLLM_NARROW_GEMM`, `VLLM_SMALL_FP8_LAYERS` (not recommended) |
| QSA qkv_proj online quant | `VLLM_QSA_QKV_ONLINE_QUANT` |
| Register the toggles in the compile cache key | — |
| Online quant for the MTP draft layer | `VLLM_DRAFT_ONLINE_QUANT` |
| Fused HC mix op (off), draft-pass PLE hints (off) | `VLLM_HC_FUSED`, `VLLM_PLE_DRAFT_HINT` |
| PDL weight-prefetch GEMV + PDL-friendly neighbours | `VLLM_PDL_GEMV` |
| L2-resident draft layer (eviction hints); CUDA TMA GEMV (off) | `VLLM_L2_DRAFT`, `VLLM_PDL_GEMV_CUDA` |
| Fused MoE routing kernel | `VLLM_FUSED_ROUTE` |
| Pure-Python PLE n-gram ids; prefetch threshold (off) | `VLLM_PLE_PY_IDS`, `VLLM_PLE_PREFETCH_MIN_ROWS` |

## Layout

- `deploy/`: `serve-qwen38-flash-next.sh` (launcher; also takes `DENSE_QUANT`,
  `DENSE_QUANT_IGNORE`, `SPEC_TOKENS`, `MAX_NUM_BATCHED_TOKENS`, ...), a systemd
  *user* unit, and `tuning.conf` (the drop-in with the tuned settings).
- `tools/`: benchmark (`bench.py`, `ab.py`, `accept.py`, `conc.py`, `stress.py`),
  quality (`quality.py`; `QUALITY_SHORT=1` for ≤64-token passages), nsys capture and
  analysis (`prof_launch.sh`, `capture.py`, `analyze.py`), microbenchmarks, the
  chat-corpus generator for the hot-token vocab, and the NumPy-ids fuzz test.
- `data/`: the 98K hot-token draft vocab and quality reference outputs.
- `results/`: raw `bench.py` runs (baseline, tuned, final).
- `REPORT.md` (full report, start here), `ROADMAP.md` (items, estimates, status, details),
  `SUMMARY.md` (overnight snapshot) and `FINDINGS.md` (chronological log, including rejected ideas).

## Reproducing

1. Install this tree: `VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=<main
   commit merged into the base> uv pip install -e .`. The fork's changes are Python-only.
   Also `flashinfer-cubin` matching `flashinfer-python`, from the FlashInfer index.
2. Put the table in swap-friendly conditions: ~50 GB of free RAM plus swap is enough;
   it doesn't need to fit in RAM.
3. Copy `deploy/` into place and set your GPU UUID (`nvidia-smi -L`) in the serve
   script and the unit. Point `VLLM_DRAFT_VOCAB_FILE` at `data/draft_vocab_chat96k.npy`.
4. Tools assume the paths at the top of `tools/bench.py` (model dir, tree, output dir).
   Edit them for your layout.

Gotchas: see "Important if you toggle settings" in `SUMMARY.md`. In short, clear the
torch.compile cache if a start fails with an odd `AttributeError` after changing env
vars. Also warm up after every restart (the PLE table starts cold), and CUDA 13.1.
