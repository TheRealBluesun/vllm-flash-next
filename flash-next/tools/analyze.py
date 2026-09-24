"""Summarize an nsys decode/prefill capture: kernel categories, idle gaps, steps.
Usage: analyze.py REPORT.nsys-rep [steps_hint]"""
import re, sqlite3, subprocess, sys, collections, os
rep = sys.argv[1]; db_path = rep.replace(".nsys-rep", ".sqlite")
if not os.path.exists(db_path):
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db_path, rep], check=True, capture_output=True)
c = sqlite3.connect(db_path).cursor()
ks = c.execute("""select k.start,k.end,s.value,d.value from CUPTI_ACTIVITY_KIND_KERNEL k
   join StringIds s on s.id=k.shortName join StringIds d on d.id=k.demangledName order by k.start""").fetchall()
CATS = [("moe_marlin", r"marlin_moe"), ("dense_fp8_marlin", r"marlin"),
        ("bf16_gemm_cutlass", r"cutlass_80_wmma|cutlass::Kernel2|sm90_xmma|sm80_xmma|gemm"), ("bf16_gemv_cublas", r"gemvx|gemv"),
        ("moe_other", r"moe_|topk|expert|count_and_sort|MoeFCGemm|fused_moe"), ("gdn", r"gdn|chunk_|delta_rule|recompute_w_u|kkt|conv1d|merge_16x16"),
        ("qsa_attn", r"qsa|indexer|flash|attn|attention"), ("hyperconn", r"_hc_"), ("ple", r"ple|ngram"),
        ("sampling", r"sampl|rejection|argmax|gumbel|logits"), ("norm_act", r"norm|silu|act_and_mul|gelu"),
        ("elementwise", r"elementwise|triton_poi|triton_red|copy|index|scatter|gather|fill|cat")]
cat_t = collections.Counter(); cat_n = collections.Counter(); unk = collections.Counter()
for s, e, short, dem in ks:
    for name, pat in CATS:
        if re.search(pat, dem, re.I) or re.search(pat, short, re.I):
            cat_t[name] += e - s; cat_n[name] += 1; break
    else:
        cat_t["other"] += e - s; cat_n["other"] += 1; unk[short] += e - s
ev = sorted((s, e) for s, e, _, _ in ks)
busy = 0; cs, ce = ev[0]; gaps = []
for s, e in ev[1:]:
    if s > ce: busy += ce - cs; gaps.append((ce, s)); cs, ce = s, e
    else: ce = max(ce, e)
busy += ce - cs; span = ev[-1][1] - ev[0][0]
syncs = c.execute("""select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cudaEventSynchronize%'""").fetchone()[0]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
print(f"{os.path.basename(rep)}: span {span/1e6:.1f} ms, GPU busy {busy/1e6:.1f} ms ({busy/span*100:.0f}%), kernels {len(ks)}, eventSyncs {syncs}")
big = sorted((e - s for s, e in gaps if e - s > 50_000), reverse=True)
print(f"  idle gaps >50us: {len(big)} totaling {sum(big)/1e6:.1f} ms (median {big[len(big)//2]/1e3 if big else 0:.0f} us); small gaps {sum(e-s for s,e in gaps if e-s<=50_000)/1e6:.1f} ms")
div = steps or 1
print(f"  {'category':20s} {'ms total':>9} {'%':>6} {'ms/step' if steps else '':>8}")
for name, t in cat_t.most_common():
    print(f"  {name:20s} {t/1e6:9.2f} {t/busy*100:6.1f} {t/1e6/div:8.2f}" if steps else f"  {name:20s} {t/1e6:9.2f} {t/busy*100:6.1f}")
if unk: print("  other top:", [(k[:40], round(v/1e6, 2)) for k, v in unk.most_common(6)])
