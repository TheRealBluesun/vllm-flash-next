"""Fuzz: NumPy decode-path n-gram ids must equal the torch path bit-for-bit."""
import json, os, time, random
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
mults = cls._make_layer_multipliers(ngram_size=o.ngram_size, unigram_vocab_size=int(tc["vocab_size"]), seed=int(tc.get("seed", 1234)), ple_dense_layer_id=0)
sizes, offs, _ = cls._make_vocab_layout(ngram_vocab_size_base=int(tc["ngram_vocab_size_base"]), ngram_heads=o.ngram_heads, ple_dense_layer_id=0)
o.layer_multipliers = torch.tensor(mults); o.ngram_heads_vocab_sizes = torch.tensor(sizes); o.ngram_heads_offsets = torch.tensor(offs)
V = int(tc["vocab_size"]); rng = random.Random(0); n_ctx = o.ngram_size - 1
def rand_tokens(n):
    return [o.eos_token_id if rng.random() < 0.08 else rng.randrange(V) for _ in range(n)]
t_np = t_torch = 0.0; cases = 0
for trial in range(3000):
    num_reqs = rng.randint(1, 4); lens = [rng.randint(1, 8) for _ in range(num_reqs)]
    qsl = [0]; [qsl.append(qsl[-1] + l) for l in lens]
    pad = rng.choice([0, 0, rng.randint(1, 6)])  # CUDA-graph padding past the valid tokens
    ids = torch.tensor(rand_tokens(qsl[-1] + pad), dtype=torch.int32)
    ctx = torch.tensor([rand_tokens(n_ctx) for _ in range(num_reqs + rng.randint(0, 2))], dtype=torch.int32)
    q = torch.tensor(qsl, dtype=torch.int32)
    os.environ["VLLM_PLE_NP_IDS"] = "0"; t0 = time.perf_counter(); ref = o.compute_ngram_ids(ids, q, ctx); t_torch += time.perf_counter() - t0
    os.environ["VLLM_PLE_NP_IDS"] = "1"; t0 = time.perf_counter(); got = o.compute_ngram_ids(ids, q, ctx); t_np += time.perf_counter() - t0
    assert ref.shape == got.shape and ref.dtype == got.dtype, (ref.shape, got.shape, ref.dtype, got.dtype)
    assert torch.equal(ref, got), f"mismatch trial {trial}: lens={lens} pad={pad}"
    cases += 1
print(f"{cases} random layouts identical; torch {t_torch/cases*1e3:.3f} ms/call vs numpy {t_np/cases*1e3:.3f} ms/call")
