#!/usr/bin/env python3
"""
diagnose_entropy.py — is token_entropy detecting STALLING or just TEMPLATE STYLE?

token_entropy scored 0.89 AUPRC on verification stalls but only 0.35 on exact-copy.
Exact-copy duplicates GENUINE reasoning (varied perplexity); verification inserts
canned "let me double-check" text. That asymmetry is the tell that entropy may be
detecting the injected genre rather than stalling. This quantifies it.

Run locally (needs the .npz):
    python diagnose_entropy.py
"""
import json
import numpy as np
import statistics as st

acts = np.load("data/acts_injected.npz", allow_pickle=True)
tid = np.array([str(t) for t in acts["trace_id"]])
sidx = acts["step_idx"]
ent = acts["entropy"]
rows = [json.loads(l) for l in open("data/injected_traces.jsonl") if l.strip()]

byt = {}
for i, t in enumerate(tid):
    byt.setdefault(t, []).append(i)
for t in byt:
    byt[t] = sorted(byt[t], key=lambda i: sidx[i])

genuine, copy_ins, verif_ins = [], [], []
for r in rows:
    t = r["trace_id"]
    if t not in byt or len(byt[t]) != len(r["steps"]):
        continue
    for s, lab in enumerate(r["stall_labels"]):
        e = float(ent[byt[t][s]])
        if lab == 0:
            genuine.append(e)
        elif r["condition"] == "verification":
            verif_ins.append(e)
        else:
            copy_ins.append(e)

def d(x):
    return f"mean {st.mean(x):.3f}  sd {st.pstdev(x):.3f}  n={len(x)}"

print("entropy distribution by step type:")
print(f"  genuine reasoning (neg) : {d(genuine)}")
print(f"  exact-copy inserts (pos): {d(copy_ins)}")
print(f"  verification inserts(pos): {d(verif_ins)}")
print()
gsd = st.pstdev(genuine)
vsd = st.pstdev(verif_ins)
sep = (st.mean(genuine) - st.mean(verif_ins)) / gsd
print(f"verification-insert entropy sd / genuine sd = {vsd/gsd:.2f}")
print(f"  (<< 1 means verification inserts are entropy-UNIFORM => template-style artefact)")
print(f"separation: genuine mean is {sep:.2f} genuine-SDs above verification-insert mean")
print()
print("VERDICT:")
if vsd / gsd < 0.5 and sep > 1.0:
    print("  token_entropy is largely detecting TEMPLATE STYLE, not stalling. The 0.89")
    print("  is inflated by synthetic uniformity. Report it as a confounded baseline;")
    print("  the honest comparison is against exact-copy entropy (0.35) and the natural")
    print("  half of the story is that entropy cannot see stalls it wasn't handed as")
    print("  canned text. Consider a natural-stall follow-up (v0.2).")
else:
    print("  token_entropy is a GENUINE competitor: verification-insert entropy overlaps")
    print("  the genuine distribution, so the 0.89 is real detection, not template")
    print("  uniformity. Report entropy as beating the latent signal on this task. That")
    print("  is a legitimate, if deflating, headline. Say it plainly.")
