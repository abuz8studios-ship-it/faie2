"""Gate: FAIE2 logits must be bit-identical whether tokens go through 1, 3 or 8 at a time.

This is the property speculative decoding needs to be lossless: an 8-token verify must produce exactly
what 8 single-token decode steps would have produced.
"""

import json
import sys

import numpy as np

from faie2 import Faie2
from gate_oracle import MODEL, ORACLE, run_case

sys.stdout.reconfigure(encoding="utf-8")
oracle = json.load(open(ORACLE, encoding="utf-8"))
eng = Faie2()
eng.load(MODEL, max_ctx=4096)
ok = True
for case in oracle["cases"]:
    if case["name"] not in ("json", "code", "long"):
        continue
    ref, _ = run_case(eng, case, block=8)
    for block in (1, 3):
        lp, _ = run_case(eng, case, block=block)
        same = np.array_equal(lp, ref)
        ok &= same
        print(f"{case['name']:<5} block {block} vs 8: bit-identical={same}  max|diff|={np.abs(lp - ref).max():.3e}", flush=True)
print("BATCH-INVARIANT" if ok else "NOT batch-invariant")
sys.exit(0 if ok else 1)
