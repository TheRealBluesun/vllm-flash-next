"""Would a VRAM cache of PLE rows remove the per-step PLE stall?

Computes the model's real n-gram row ids for token streams and simulates caches.
The stall disappears only for decode steps where *all* rows hit (any miss needs
the CPU round trip), so the key metric is P(step has zero misses).
"""
import json, os, sys, collections
import numpy as np, torch
os.environ["VLLM_PLE_NP_IDS"] = "1"
from vllm.model_executor.layers.ple_offload_layer import mark_as_offload_worker
mark_as_offload_worker()
import vllm.models.qwen4_exp.nvidia.ple_layer as pl

cls = next(c for c in vars(pl).values() if isinstance(c, type) and hasattr(c, "_compute_ngram_ids_np"))
cfg = json.load(open("/opt/d/models/Qwen3.8-Flash-Next-NVFP4/config.json")); tc = cfg.get("text_config", cfg)
o = object.__new__(cls); torch.nn.Module.__init__(o)
o.ngram_size = int(tc["ngram_size"]); o.heads_per_ngram = int(tc["heads_per_ngram"]); o.ngram_heads = (o.ngram_size - 1) * o.heads_per_ngram
eos = tc.get("eos_token_id", cfg.get("eos_token_id")); o.eos_token_id = int(eos[0] if isinstance(eos, list) else eos)
o.layer_multipliers = torch.tensor(cls._make_layer_multipliers(ngram_size=o.ngram_size, unigram_vocab_size=int(tc["vocab_size"]), seed=int(tc.get("seed", 1234)), ple_dense_layer_id=0))
sizes, offs, total_rows = cls._make_vocab_layout(ngram_vocab_size_base=int(tc["ngram_vocab_size_base"]), ngram_heads=o.ngram_heads, ple_dense_layer_id=0)
o.ngram_heads_vocab_sizes = torch.tensor(sizes); o.ngram_heads_offsets = torch.tensor(offs)
ROW_BYTES = 160
N_CTX = o.ngram_size - 1


def row_ids(tokens: np.ndarray) -> np.ndarray:
    """[T, heads] row ids for a token stream (one request, continuous context)."""
    out = []
    t = torch.from_numpy(tokens.astype(np.int64))
    for s in range(0, len(t), 4096):
        w = t[s:s + 4096]
        ctx = t[s - N_CTX:s].reshape(1, N_CTX) if s >= N_CTX else torch.full((1, N_CTX), o.eos_token_id)
        out.append(o.compute_ngram_ids(w, torch.tensor([0, len(w)]), ctx).numpy())
    return np.concatenate(out)


def simulate(ids: np.ndarray, warm_frac: float, cap_rows: int, tokens_per_step=5):
    """'Seen-before' cache (LRU at these capacities never evicts: distinct rows << cap).
    Every row touched (prefill or decode) is inserted. Measure on the post-warmup part."""
    seen = set(); warm = int(len(ids) * warm_frac)
    for r in ids[:warm].ravel():
        seen.add(int(r))
    assert len(seen) < cap_rows, "cache would evict; simulation assumption broken"
    row_hits = row_total = steps = clean = 0; misses_per_step = []
    for s in range(warm, len(ids) - tokens_per_step, tokens_per_step):
        rows = ids[s:s + tokens_per_step].ravel()
        m = sum(1 for r in rows if int(r) not in seen)
        row_hits += len(rows) - m; row_total += len(rows); steps += 1; clean += (m == 0); misses_per_step.append(m)
        for r in rows:
            seen.add(int(r))
    return row_hits / row_total, clean / steps, float(np.mean(misses_per_step)), len(seen)


chat = np.array(json.load(open("/opt/d/flash-next-perf/chat_corpus_ids.json")), dtype=np.int64)
code = np.array(json.load(open("/opt/d/flash-next-perf/corpus_ids_v2.json")), dtype=np.int64)[9_000_000:9_400_000]
print(f"total table rows {total_rows:,} ({total_rows * ROW_BYTES / 2**30:.1f} GiB); 1 GiB cache = {2**30 // ROW_BYTES:,} rows")
for name, toks in (("chat (model output, 122K tok)", chat), ("code/docs (400K tok)", code)):
    ids = row_ids(toks)
    for warm in (0.5,):
        hit, clean, mps, nseen = simulate(ids, warm, cap_rows=2**31 // ROW_BYTES)
        print(f"{name}: per-row hit {hit*100:5.1f}% | steps with ZERO misses {clean*100:5.1f}% | "
              f"avg misses/step {mps:4.1f} of {5*o.ngram_heads} | distinct rows {nseen:,} ({nseen*ROW_BYTES/2**20:.0f} MiB)")
    # per n-gram order: which heads miss?
    warm = len(ids) // 2; seen = set(ids[:warm].ravel().tolist())
    for h0, label in ((0, "2-gram heads"), (o.heads_per_ngram, "3-gram heads")):
        sub = ids[warm:, h0:h0 + o.heads_per_ngram].ravel()
        print(f"   {label}: hit {np.mean([int(r) in seen for r in sub])*100:5.1f}%")
