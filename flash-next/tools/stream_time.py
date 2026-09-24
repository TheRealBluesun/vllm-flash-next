"""Serialized time per kernel name on the busiest stream: each kernel is charged
end - max(start, previous end) (its non-overlapped part); gaps are charged to 'idle'."""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); steps = int(sys.argv[2])
rows = db.execute("""select k.start, k.end, s.value, k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k
  join StringIds s on s.id = k.shortName order by k.start""").fetchall()
cnt = collections.Counter(r[3] for r in rows)
for sid, n in cnt.most_common(4):
    ks = [r for r in rows if r[3] == sid]
    t = collections.Counter(); c = collections.Counter(); idle = 0; prev = ks[0][0]
    for s, e, name, _ in ks:
        if s > prev: idle += s - prev
        t[name] += max(0, e - max(s, prev)); c[name] += 1; prev = max(prev, e)
    span = ks[-1][1] - ks[0][0]
    print(f"stream {sid}: {n} kernels, span {span/1e6/steps:.2f} ms/step, idle {idle/1e6/steps:.2f} ms/step")
    if sid != cnt.most_common(1)[0][0]:
        continue
    for name, v in t.most_common(28):
        print(f"   {name[:50]:50s} {v/1e3/steps:7.0f} us/step  n/step {c[name]/steps:6.1f}  avg {v/1e3/c[name]:6.1f}")
