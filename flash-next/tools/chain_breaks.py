"""For each PDL GEMV in a decode capture: which kernel precedes it on the stream, did it
start before that kernel ended (PDL overlap), and how long is it exposed afterwards."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
rows = db.execute("""select k.start, k.end, s.value, k.streamId, k.gridX*k.gridY*k.gridZ
  from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName order by k.start""").fetchall()
by_stream = collections.defaultdict(list)
for r in rows:
    by_stream[r[3]].append(r)
GEMV = ("_pdl_gemv_kernel",)
pred = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])   # n, overlapped, exposed us, gemv us
succ = collections.defaultdict(lambda: [0, 0, 0.0])        # n, succ started before gemv end, gap us
tot_exposed = tot_gemv = 0.0
for sid, ks in by_stream.items():
    for i in range(1, len(ks) - 1):
        s, e, name, _, grid = ks[i]
        if name not in GEMV:
            continue
        ps, pe, pname = ks[i - 1][0], ks[i - 1][1], ks[i - 1][2]
        ss, se, sname = ks[i + 1][0], ks[i + 1][1], ks[i + 1][2]
        exposed = (e - max(s, pe)) / 1e3
        p = pred[pname]; p[0] += 1; p[1] += s < pe; p[2] += exposed; p[3] += (e - s) / 1e3
        q = succ[sname]; q[0] += 1; q[1] += ss < e; q[2] += max(0, ss - e) / 1e3
        tot_exposed += exposed; tot_gemv += (e - s) / 1e3
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
print(f"PDL GEMVs: {sum(p[0] for p in pred.values())}, busy {tot_gemv/steps:.0f} us/step, exposed after predecessor {tot_exposed/steps:.0f} us/step")
print(f"\n{'predecessor':45s} {'n/step':>7} {'overlap%':>8} {'exposed us/step':>15} {'avg gemv us':>11}")
for k, (n, ov, ex, g) in sorted(pred.items(), key=lambda kv: -kv[1][2])[:18]:
    print(f"{k[:45]:45s} {n/steps:7.1f} {ov/n*100:7.0f}% {ex/steps:15.0f} {g/n:11.1f}")
print(f"\n{'successor':45s} {'n/step':>7} {'early%':>7} {'gap us/step':>11}")
for k, (n, ov, gap) in sorted(succ.items(), key=lambda kv: -kv[1][0])[:18]:
    print(f"{k[:45]:45s} {n/steps:7.1f} {ov/n*100:6.0f}% {gap/steps:11.0f}")
