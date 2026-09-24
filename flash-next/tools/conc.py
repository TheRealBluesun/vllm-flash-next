"""Pure-decode concurrency: N parallel chat requests (short prompts, 512 out).
Aggregate = total tokens generated while all N streams are decoding."""
import os, sys, time, json, threading, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import BASE, MODEL
from quality import CHATS
def run(i, n_out, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": CHATS[i % len(CHATS)] + " Be thorough."}],
            "max_tokens": n_out, "ignore_eos": True, "temperature": 0, "stream": True, "logprobs": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    ts = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]": continue
            d = json.loads(line[5:]); ch = d.get("choices") or []
            if ch and ch[0].get("logprobs"):
                now = time.time(); ts.extend([now] * len(ch[0]["logprobs"]["content"] or []))
    out[i] = ts
for n in [int(a) for a in (sys.argv[1:] or ["1", "2", "4"])]:
    out = {}; th = [threading.Thread(target=run, args=(i, 512, out)) for i in range(n)]
    [t.start() for t in th]; [t.join() for t in th]
    start = max(ts[0] for ts in out.values()); end = min(ts[-1] for ts in out.values())
    toks = sum(sum(1 for t in ts if start <= t <= end) for ts in out.values())
    print(f"C={n}: aggregate {toks/(end-start):6.1f} tok/s over the all-decoding window ({end-start:.1f}s), per-stream {toks/(end-start)/n:.1f}", flush=True)
