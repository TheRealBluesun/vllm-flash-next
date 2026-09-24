"""Stress: 4 concurrent 32K-token prompts + 2 chats, checks every request completes with sane text."""
import os, sys, json, concurrent.futures as cf, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import BASE, MODEL, build_corpus
ids = build_corpus(); base = 8_000_000
def comp(i):
    body = {"model": MODEL, "prompt": ids[base + i * 32768: base + (i + 1) * 32768] , "max_tokens": 200, "temperature": 0}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return "comp", r["usage"], r["choices"][0]["text"][:80]
def chat(q):
    body = {"model": MODEL, "messages": [{"role": "user", "content": q}], "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return "chat", r["usage"], r["choices"][0]["message"]["content"][:80]
with cf.ThreadPoolExecutor(6) as ex:
    fs = [ex.submit(comp, i) for i in range(4)] + [ex.submit(chat, q) for q in ("What is 17*23? Show your work.", "Name three primary colors and explain why they are primary.")]
    for f in fs:
        kind, u, t = f.result(); print(kind, u["prompt_tokens"], u["completion_tokens"], repr(t))
