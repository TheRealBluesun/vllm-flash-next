"""Quick A/B: fresh-text 16K prefills + 1K-ctx decodes. Prints TTFT, decode tok/s, ms/step."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import Slicer, build_corpus, stream, metrics, accept_len, step_ms, OFFSET_STATE
sl = Slicer(build_corpus()); sl.pos = int(open(OFFSET_STATE).read())
stream("/v1/completions", {"prompt": sl.take(512), "max_tokens": 16})
pre, dec = [], []
for i in range(3):
    x = stream("/v1/completions", {"prompt": sl.take(16384), "max_tokens": 1}); pre.append(x["ttft_s"])
    print(f"prefill16K ttft {x['ttft_s']:.2f}s  ({16384/x['ttft_s']:.0f} tok/s)", flush=True)
for i in range(4):
    m0 = metrics(); x = stream("/v1/completions", {"prompt": sl.take(1024), "max_tokens": 400, "ignore_eos": True})
    x["accept_len"] = accept_len(m0, metrics()); dec.append(step_ms(x))
    print(f"decode {x['decode_tps']:6.1f} tok/s accept {x['accept_len']:.2f}  {step_ms(x):.2f} ms/step", flush=True)
open(OFFSET_STATE, "w").write(str(sl.pos))
print(f"MEAN prefill16K {sum(pre)/len(pre):.2f}s | decode {sum(dec)/len(dec):.2f} ms/step")
