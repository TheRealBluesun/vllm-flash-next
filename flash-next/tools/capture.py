#!/usr/bin/env python3
"""Drive two nsys capture ranges on a server launched with
--profiler-config '{"profiler":"cuda"}' under
`nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat`.

Range 1: steady-state decode (single request, ~200 tokens, short ctx).
Range 2: one 16K-token prefill (max_tokens=1).
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import BASE, Slicer, build_corpus, stream  # noqa: E402


def post(path):
    urllib.request.urlopen(urllib.request.Request(BASE + path, b"", method="POST"),
                           timeout=600).read()


sl = Slicer(build_corpus())
sl.pos = 9_700_000  # region the baseline run did not touch
# warm up: JIT, cold paths, a decode of similar shape
stream("/v1/completions", {"prompt": sl.take(1024), "max_tokens": 64, "ignore_eos": True})
stream("/v1/completions", {"prompt": sl.take(4096), "max_tokens": 1})

chat = {"messages": [{"role": "user", "content": "Explain how the immune system distinguishes self from non-self, in detail."}],
        "max_tokens": 300, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
stream("/v1/chat/completions", dict(chat, max_tokens=32))
post("/start_profile")
t = time.time()
r = stream("/v1/chat/completions", chat)
post("/stop_profile")
print("decode range:", json.dumps(r), f"{time.time()-t:.1f}s", flush=True)

time.sleep(5)
prompt = sl.take(16384)
post("/start_profile")
r = stream("/v1/completions", {"prompt": prompt, "max_tokens": 1})
post("/stop_profile")
print("prefill range:", json.dumps(r), flush=True)
