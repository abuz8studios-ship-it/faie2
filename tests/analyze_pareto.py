"""Pareto map of one steady-state speculative step: time per category (median over steps), bytes moved, share.
usage: analyze_pareto.py <nsys sqlite of profile_spec.py nopdl>"""

import sqlite3
import sys
from collections import defaultdict

import numpy as np

db = sys.argv[1]
con = sqlite3.connect(db)
names = dict(con.execute("SELECT id, value FROM StringIds"))
rows = con.execute("SELECT start, end, demangledName, gridX FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
starts = [i for i, r in enumerate(rows) if "k_df_noise_tokens" in names[r[2]]]


def category(n, gx, phase):
    if "k_q6k_mma" in n or "k_gemv" in n or "k_q6k_gemm" in n:
        if gx == 15520:
            return "LM head (target vocab, 1.04 GB) - verify" if phase == "verify" else "LM head (target vocab, 1.04 GB) - draft"
        return "target weight GEMVs (20.0 GB Q6_K)"
    if "k_q8_mma" in n:
        return "drafter weight GEMVs (1.9 GB Q8_0)" if phase != "inject" else "drafter KV inject GEMVs (0.2 GB)"
    if "k_fa_" in n or "k_attn" in n or "k_df_attn" in n:
        return "attention (target + drafter)"
    if "k_gdn" in n or "k_conv_commit" in n or "k_l2norm" in n:
        return "DeltaNet recurrence + conv + commit"
    if "rmsnorm" in n or "gated_norm" in n or "swiglu" in n or "k_df_conv" in n or "qk_norm" in n or "k_df_qk" in n:
        return "norms, activations, rope (small kernels)"
    return "selection, accept, embed, misc"


steps = []
for a, b in zip(starts[5:], starts[6:]):
    seg = rows[a:b]
    per = defaultdict(float)
    phase = "draft"
    seen_df = 0
    for s, e, n, gx in seg:
        nm = names[n]
        if phase == "draft" and "k_embed_q6k" in nm and seen_df:
            phase = "verify"
        if phase == "verify" and "k_argmax_part" in nm:
            phase = "accept"
        if phase == "accept" and "k_q8_mma" in nm:
            phase = "inject"
        seen_df += 1
        per[category(nm, gx, phase)] += (e - s) / 1e6
    busy = sum(per.values())
    wall = (rows[b][0] - rows[a][0]) / 1e6
    per["idle gaps between kernels"] = wall - busy
    steps.append(per)
cats = set().union(*steps)
med = {c: float(np.median([s.get(c, 0.0) for s in steps])) for c in cats}
total = sum(med.values())
print(f"steps analysed: {len(steps)}, median step {total:.2f} ms")
cum = 0.0
for c, v in sorted(med.items(), key=lambda kv: -kv[1]):
    cum += v
    print(f"{v:7.3f} ms  {100 * v / total:5.1f}%  cum {100 * cum / total:5.1f}%  {c}")
