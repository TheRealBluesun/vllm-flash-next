"""Fixed-prompt MTP acceptance test (same prompts every run): corpus passages
(code/docs) + chat prompts. Prints mean accepted tokens/step and ms/step."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import build_corpus, stream, metrics, accept_len, step_ms
from quality import CHATS
SAMPLED = "--sampled" in sys.argv  # use the model's default sampling (T=1, top_p .95, top_k 20)
ids = build_corpus()
rows = []
for i in range(6):
    p = ids[9_500_000 + i * 1024: 9_500_000 + (i + 1) * 1024]
    m0 = metrics(); x = stream("/v1/completions", dict({"prompt": p, "max_tokens": 256, "ignore_eos": True}, **({"temperature": None} if SAMPLED else {})))
    x["accept_len"] = accept_len(m0, metrics()); rows.append(("corpus", x))
for c in CHATS:
    m0 = metrics(); x = stream("/v1/chat/completions", {"messages": [{"role": "user", "content": c}], "max_tokens": 256,
                                                         "chat_template_kwargs": {"enable_thinking": False}, **({"temperature": None} if SAMPLED else {})})
    x["accept_len"] = accept_len(m0, metrics()); rows.append(("chat", x))
for kind in ("corpus", "chat"):
    xs = [x for k, x in rows if k == kind]
    print(f"{kind:6s}: accept {sum(x['accept_len'] for x in xs)/len(xs):.3f}  ms/step {sum(step_ms(x) for x in xs)/len(xs):.2f}  decode {sum(x['decode_tps'] for x in xs)/len(xs):.1f} tok/s")
