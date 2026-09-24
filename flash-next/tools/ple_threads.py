"""Sample PLE-offload worker thread states while one request runs; reports how
many threads are Running vs in D (uninterruptible: swap-in) and major faults."""
import os, sys, time, threading, collections, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import Slicer, build_corpus, stream
pid = int(subprocess.check_output("ps -eo pid,args | awk '/spawn_main/ && /vllm-flash-next-0906/ && !/awk/ {print $1}' | head -1", shell=True))
def majflt(): return int(open(f"/proc/{pid}/stat").read().rsplit(")",1)[1].split()[9])
def sample(stop, hist):
    tdir = f"/proc/{pid}/task"
    while not stop.is_set():
        r = d = 0
        for t in os.listdir(tdir):
            try: st = open(f"{tdir}/{t}/stat").read().rsplit(")",1)[1].split()[0]
            except OSError: continue
            r += st == "R"; d += st == "D"
        hist[(r, d)] += 1
        time.sleep(0.0005)
sl = Slicer(build_corpus()); sl.pos = 1_800_000
for label, body in [("prefill 16K", {"prompt": sl.take(16384), "max_tokens": 1}),
                    ("decode 1K+300", {"prompt": sl.take(1024), "max_tokens": 300, "ignore_eos": True})]:
    hist = collections.Counter(); stop = threading.Event()
    th = threading.Thread(target=sample, args=(stop, hist)); f0 = majflt(); th.start()
    r = stream("/v1/completions", body); stop.set(); th.join()
    n = sum(hist.values())
    busy = {k: v for k, v in hist.items() if k != (0, 0)}
    print(f"{label}: ttft {r['ttft_s']:.2f}s decode {r['decode_tps'] or 0:.0f} tok/s, PLE major faults {majflt()-f0}")
    print(f"   samples {n}; idle {hist[(0,0)]/n*100:.0f}%; busy states (Running, D-state) -> share of samples:")
    for (rr, dd), v in sorted(busy.items(), key=lambda x: -x[1])[:8]: print(f"     R={rr:2d} D={dd:2d}: {v/n*100:5.1f}%")
