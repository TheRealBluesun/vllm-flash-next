#!/usr/bin/env python3
"""Flash-Next serving benchmark: prefill (TTFT) + decode at depth, short-prompt
decode, and concurrency, against the running vLLM on :8000.

Prompts are exact-length token-id slices of a real text+code corpus. Every
request uses a fresh corpus offset so prefix caching never hits (and PLE rows
are a realistic mix of warm and cold). Run with the vLLM venv python (needs
`transformers` for the tokenizer):

    /opt/d/vllm-flash-next-0906/.venv/bin/python /opt/d/flash-next-perf/bench.py [--quick] [--label NAME]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import time
import urllib.request

BASE = "http://localhost:8000"
MODEL = "Qwen3.8-Flash-Next"
MODEL_DIR = "/opt/d/models/Qwen3.8-Flash-Next-NVFP4"
TREE = "/opt/d/vllm-flash-next-0906"
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus_ids_v2.json")
OFFSET_STATE = os.path.join(HERE, ".next_offset")  # each run starts on unseen text

SHORT = {
    "code": "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "prose": "Write a long essay about rivers.",
    "short": "What is the capital of France? Explain briefly.",
}


def build_corpus():
    if os.path.exists(CORPUS):
        return json.load(open(CORPUS))
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    files = sorted(
        os.path.join(r, f)
        for r, _, fs in os.walk(os.path.join(TREE, "docs"))
        for f in fs
        if f.endswith(".md")
    )
    files += sorted(
        os.path.join(r, f)
        for r, _, fs in os.walk(os.path.join(TREE, "vllm"))
        for f in fs
        if f.endswith(".py")
    )
    text = "\n\n".join(open(f, errors="ignore").read() for f in files)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    json.dump(ids, open(CORPUS, "w"))
    return ids


class Slicer:
    """Hands out non-overlapping corpus slices (wraps when exhausted)."""

    def __init__(self, ids):
        self.ids, self.pos = ids, 0

    def take(self, n):
        if self.pos + n > len(self.ids):
            self.pos = 0
        s = self.ids[self.pos : self.pos + n]
        self.pos += n
        return s


def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    out = {}
    for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        m = re.search(rf"^vllm:spec_decode_{k}_total\{{[^}}]*\}} (\S+)", txt, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def accept_len(m0, m1):
    d = m1["num_drafts"] - m0["num_drafts"]
    return 1 + (m1["num_accepted_tokens"] - m0["num_accepted_tokens"]) / d if d else None


def stream(path, body):
    body = dict({"temperature": 0}, **body, model=MODEL, stream=True,
                stream_options={"include_usage": True})
    if body["temperature"] is None:  # None -> server/model default sampling
        del body["temperature"]
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and first is None:
                piece = ch[0].get("text") or (ch[0].get("delta") or {}).get("content")
                if piece:
                    first = time.time()
    t1 = time.time()
    n_out = usage["completion_tokens"]
    ttft = first - t0
    dec = (n_out - 1) / (t1 - first) if n_out > 1 and t1 > first else None
    return {"prompt_tokens": usage["prompt_tokens"], "out_tokens": n_out,
            "ttft_s": ttft, "decode_tps": dec, "total_s": t1 - t0}


def step_ms(x):
    return 1e3 * (x["accept_len"] or 1) / x["decode_tps"] if x.get("decode_tps") else 0.0


def depth_run(sl, n_ctx, n_gen):
    ids = sl.take(n_ctx)
    return stream("/v1/completions", {"prompt": ids, "max_tokens": n_gen,
                                       "ignore_eos": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 rep, skip 100K")
    ap.add_argument("--label", default="")
    ap.add_argument("--offset", type=int, default=None,
                    help="corpus start (default: continue after the previous run)")
    a = ap.parse_args()
    reps = 1 if a.quick else 3
    depths = [1024, 8192, 32768] + ([] if a.quick else [102400])
    gen = 256

    ids = build_corpus()
    sl = Slicer(ids)
    sl.pos = a.offset if a.offset is not None else int(open(OFFSET_STATE).read()) if os.path.exists(OFFSET_STATE) else 0
    rev = subprocess.run(["git", "-C", TREE, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    res = {"label": a.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "tree_rev": rev, "corpus_tokens": len(ids), "depth": [], "short": [],
           "concurrency": []}
    print(f"corpus {len(ids)} tokens, tree {rev}, label={a.label!r}")

    # warm-up: first request after start pays Triton JIT / cold paths
    depth_run(sl, 512, 16)

    print("\n== prefill + decode at depth (gen 256, ignore_eos) ==")
    print(f"{'ctx':>7} {'rep':>3} {'TTFT s':>8} {'prefill tok/s':>14} {'decode tok/s':>13} {'accept':>7} {'ms/step':>8}")
    for n in depths:
        for r in range(reps):
            m0 = metrics()
            x = depth_run(sl, n, gen)
            x.update(ctx=n, rep=r, accept_len=accept_len(m0, metrics()),
                     prefill_tps=x["prompt_tokens"] / x["ttft_s"])
            res["depth"].append(x)
            print(f"{n:>7} {r:>3} {x['ttft_s']:>8.2f} {x['prefill_tps']:>14.0f} "
                  f"{x['decode_tps']:>13.1f} {x['accept_len'] or 0:>7.2f} {step_ms(x):>8.2f}", flush=True)

    print("\n== short-prompt chat decode (400 tok, thinking off) ==")
    for name, p in SHORT.items():
        for r in range(reps):
            m0 = metrics()
            x = stream("/v1/chat/completions", {
                "messages": [{"role": "user", "content": p}],
                "max_tokens": 128 if name == "short" else 400,
                "chat_template_kwargs": {"enable_thinking": False}})
            x.update(name=name, rep=r, accept_len=accept_len(m0, metrics()))
            res["short"].append(x)
            print(f"{name:>6} rep{r} decode {x['decode_tps']:6.1f} tok/s  "
                  f"accept {x['accept_len'] or 0:.2f}  {step_ms(x):5.2f} ms/step  (n={x['out_tokens']})", flush=True)

    print("\n== concurrency: N parallel 8K-ctx requests, gen 256 ==")
    for conc in ([4] if a.quick else [2, 4]):
        prompts = [sl.take(8192) for _ in range(conc)]
        m0 = metrics()
        t0 = time.time()
        with cf.ThreadPoolExecutor(conc) as ex:
            outs = list(ex.map(lambda ids: stream("/v1/completions", {
                "prompt": ids, "max_tokens": gen, "ignore_eos": True}), prompts))
        wall = time.time() - t0
        agg = sum(o["out_tokens"] for o in outs) / wall
        x = {"conc": conc, "wall_s": wall, "agg_out_tps": agg,
             "mean_ttft_s": sum(o["ttft_s"] for o in outs) / conc,
             "mean_decode_tps": sum(o["decode_tps"] for o in outs) / conc,
             "accept_len": accept_len(m0, metrics())}
        res["concurrency"].append(x)
        print(f"C={conc}: aggregate {agg:6.1f} out tok/s, per-stream decode "
              f"{x['mean_decode_tps']:.1f}, mean TTFT {x['mean_ttft_s']:.2f}s, "
              f"accept {x['accept_len'] or 0:.2f}", flush=True)

    res["corpus_end"] = sl.pos
    open(OFFSET_STATE, "w").write(str(sl.pos))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    fn = os.path.join(HERE, "results",
                      time.strftime("%Y%m%d-%H%M%S") + (f"-{a.label}" if a.label else "") + ".json")
    json.dump(res, open(fn, "w"), indent=1)
    print(f"\nsaved {fn}")


if __name__ == "__main__":
    main()
