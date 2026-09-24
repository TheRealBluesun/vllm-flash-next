#!/usr/bin/env python3
"""Quality check for numerics changes (e.g. FP8 dense layers).

  quality.py save NAME      # record reference from the running server
  quality.py compare NAME   # compare running server against saved NAME

Measures (1) mean NLL / perplexity of fixed corpus passages via
prompt_logprobs, and (2) greedy chat generations: exact-match rate and
first-divergence token index.
"""
import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import BASE, MODEL, build_corpus  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_PASSAGES, PASSAGE_LEN, OFFSET = 40, 512, 9_000_000
if os.environ.get("QUALITY_SHORT") == "1":  # <=64-token prompts: exercises decode-sized lm_head paths
    N_PASSAGES, PASSAGE_LEN, OFFSET = 300, 60, 9_100_000
CHATS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a haiku about autumn rain, then explain its imagery.",
    "What are the main causes of the French Revolution? Answer in 5 bullet points.",
    "Write a bash one-liner that finds the 10 largest files under /var/log.",
    "Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed in m/s?",
    "Translate to French and German: 'The meeting has been moved to Thursday afternoon.'",
    "Write a Rust function that reverses the words in a string, with a unit test.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(dict(body, model=MODEL)).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def measure():
    ids = build_corpus()
    nlls = []
    for i in range(N_PASSAGES):
        p = ids[OFFSET + i * PASSAGE_LEN: OFFSET + (i + 1) * PASSAGE_LEN]
        r = post("/v1/completions", {"prompt": p, "max_tokens": 1, "temperature": 0,
                                      "prompt_logprobs": 0})
        pl = r["choices"][0]["prompt_logprobs"]
        vals = []
        for tok, d in zip(p[1:], pl[1:]):
            e = d.get(str(tok))
            vals.append(e["logprob"] if isinstance(e, dict) else e)
        nlls.append(-sum(vals) / len(vals))
    gens = []
    for c in CHATS:
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": c}],
                                          "max_tokens": 200, "temperature": 0, "logprobs": True,
                                          "chat_template_kwargs": {"enable_thinking": False}})
        toks = [t["token"] for t in r["choices"][0]["logprobs"]["content"]]
        gens.append({"text": r["choices"][0]["message"]["content"], "tokens": toks})
    return {"nll": nlls, "gens": gens}


def main():
    mode, name = sys.argv[1], sys.argv[2]
    fn = os.path.join(HERE, f"quality-{name}{'-short' if os.environ.get('QUALITY_SHORT') == '1' else ''}.json")
    cur = measure()
    if mode == "save":
        json.dump(cur, open(fn, "w"), indent=1)
        m = sum(cur["nll"]) / len(cur["nll"])
        print(f"saved {fn}: mean NLL {m:.4f} (ppl {math.exp(m):.3f})")
        return
    ref = json.load(open(fn))
    a, b = sum(ref["nll"]) / len(ref["nll"]), sum(cur["nll"]) / len(cur["nll"])
    d = [y - x for x, y in zip(ref["nll"], cur["nll"])]
    print(f"NLL ref {a:.4f} (ppl {math.exp(a):.3f}) -> now {b:.4f} (ppl {math.exp(b):.3f}); "
          f"delta {b - a:+.4f} ({(math.exp(b) / math.exp(a) - 1) * 100:+.2f}% ppl), "
          f"per-passage delta range [{min(d):+.4f}, {max(d):+.4f}]")
    same = 0
    for i, (r, c) in enumerate(zip(ref["gens"], cur["gens"])):
        n = next((k for k, (x, y) in enumerate(zip(r["tokens"], c["tokens"])) if x != y), None)
        if n is None and len(r["tokens"]) == len(c["tokens"]):
            same += 1
            print(f"  chat {i}: identical ({len(r['tokens'])} tok)")
        else:
            n = n if n is not None else min(len(r["tokens"]), len(c["tokens"]))
            print(f"  chat {i}: diverges at token {n}/{len(r['tokens'])}: "
                  f"...{''.join(r['tokens'][max(0, n - 5):n + 5])!r} vs {''.join(c['tokens'][max(0, n - 5):n + 5])!r}")
    print(f"greedy identical: {same}/{len(ref['gens'])}")


if __name__ == "__main__":
    main()
