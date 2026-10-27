"""Per-step kernel breakdown of a spec_step nsys capture: each step starts at k_df_noise_tokens.
Groups kernels into draft / verify-forward / accept+commit / inject by their position in the step."""

import sqlite3
import sys
from collections import defaultdict

import numpy as np

db = sys.argv[1] if len(sys.argv) > 1 else r"E:\ABU\FAIE2\bench\profiles\spec_step.sqlite"
con = sqlite3.connect(db)
names = dict(con.execute("SELECT id, value FROM StringIds"))
rows = con.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
ks = [(s, e, names[n]) for s, e, n in rows]


def short(n):
    n = n.split("(")[0]
    return n.replace("void ", "")


starts = [i for i, k in enumerate(ks) if "k_df_noise_tokens" in k[2]]
steps = []
for a, b in zip(starts[5:], starts[6:]):   # skip warm-up steps
    seg = ks[a:b]
    phase = "draft"
    per = defaultdict(float)
    nl = 0
    for s, e, n in seg:
        sn = short(n)
        if phase == "draft" and "k_embed_q6k" in sn and nl > 3:
            phase = "verify"
        if phase == "verify" and "k_argmax_part" in sn:
            phase = "accept_commit"
        if phase == "accept_commit" and "k_q8_mma" in sn:
            phase = "inject"
        if phase == "inject" and "k_next_anchor" in sn:
            pass
        nl += 1
        per[(phase, sn)] += (e - s) / 1e6
    wall = (ks[b][0] - ks[a][0]) / 1e6
    busy = sum((e - s) for s, e, _ in seg) / 1e6
    steps.append((wall, busy, per))

wall = np.median([w for w, _, _ in steps])
busy = np.median([b for _, b, _ in steps])
print(f"steps analysed: {len(steps)}  median step wall (kernel start to start) {wall:.3f} ms, kernel busy {busy:.3f} ms, "
      f"gaps {wall - busy:.3f} ms")
agg = defaultdict(list)
for _, _, per in steps:
    for k, v in per.items():
        agg[k].append(v)
by_phase = defaultdict(float)
for (ph, n), v in agg.items():
    by_phase[ph] += np.median(v)
for ph in ["draft", "verify", "accept_commit", "inject"]:
    print(f"\n== {ph}: {by_phase[ph]:.3f} ms")
    items = sorted(((np.median(v), n) for (p, n), v in agg.items() if p == ph), reverse=True)
    for v, n in items[:14]:
        print(f"   {v:7.3f} ms  {n}")
