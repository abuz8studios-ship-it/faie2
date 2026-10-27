"""A/B of engine kernel options on the real speculative step: median GPU ms per step on the code and prose
prompts (same prompts and step counts for every config), plus the tokens produced, so a faster config that
changed the output is visible.  usage: ab_step.py [steps]"""

import ctypes
import json
import sys

import numpy as np

from faie2 import Faie2
from test_spec_e2e import DRAFT, MODEL, PROMPTS, prefill

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
NAMES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["v3+pdl+a1", "v3+pdl+a2"]
# name = v<gemv>[+pdl (PDL on every kernel) | +pdg (PDL on GEMVs only)][+a<attn>]
CONFIGS = [(n, int(n[1]), 1 if "pdl" in n else 2 if "pdg" in n else 3 if "pdx" in n else 0, int(n.split("+a")[1][0]) if "+a" in n else 2) for n in NAMES]

eng = Faie2()
eng.load(MODEL, max_ctx=4096)
eng.load_drafter(DRAFT)
eng.use_graphs(True)
L = eng.lib
L.faie2_set_pdl.argtypes = [ctypes.c_int]
L.faie2_set_gemv_version.argtypes = [ctypes.c_int]
L.faie2_set_attn_version.argtypes = [ctypes.c_int]

res = {}
for name, gv, pdl, av in CONFIGS:
    eng._check(L.faie2_set_gemv_version(gv))
    eng._check(L.faie2_set_attn_version(av))
    eng._check(L.faie2_set_pdl(pdl))
    row = {}
    for cat in ("code", "prose"):
        for rep in range(2):   # first pass captures graphs
            t = prefill(eng, PROMPTS["cases"][cat]["prompt_tokens"])
            eng.spec_begin(t)
            out, ms = [], []
            for _ in range(STEPS):
                emit, _ = eng.spec_step()
                out += emit
                ms.append(eng.gpu_ms())
        row[cat] = {"ms_per_step": round(float(np.median(ms)), 3), "tok_per_step": round(len(out) / STEPS, 2),
                    "tok_s_gpu": round(len(out) / STEPS / (float(np.median(ms)) / 1e3), 1), "tokens": out}
    res[name] = row
    print(f"{name:<8} code {row['code']['ms_per_step']:7.3f} ms/step {row['code']['tok_per_step']:.2f} tok/step "
          f"({row['code']['tok_s_gpu']:6.1f} tok/s GPU)   prose {row['prose']['ms_per_step']:7.3f} ms/step "
          f"{row['prose']['tok_per_step']:.2f} tok/step ({row['prose']['tok_s_gpu']:6.1f} tok/s GPU)", flush=True)

base = res[NAMES[0]]
for name in res:
    same = {c: res[name][c]["tokens"] == base[c]["tokens"] for c in ("code", "prose")}
    print(f"{name:<8} same tokens as first config: {same}")
json.dump({k: {c: {kk: vv for kk, vv in v.items() if kk != "tokens"} for c, v in r.items()} for k, r in res.items()},
          open(r"E:\ABU\FAIE2\tests\ab_step_result.json", "w"), indent=1)
