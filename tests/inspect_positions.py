"""Print FAIE2 vs oracle log-probs at chosen (case, position) pairs."""

import json
import sys

import numpy as np

from faie2 import Faie2
from gate_oracle import MODEL, ORACLE, log_softmax, run_case

sys.stdout.reconfigure(encoding="utf-8")
WANT = {c.split(":")[0]: [int(p) for p in c.split(":")[1].split(",")] for c in sys.argv[1:]}

oracle = json.load(open(ORACLE, encoding="utf-8"))
eng = Faie2()
eng.load(MODEL, max_ctx=4096)
for case in oracle["cases"]:
    if case["name"] not in WANT:
        continue
    lp, _ = run_case(eng, case)
    for i in WANT[case["name"]]:
        step = case["steps"][i]
        f = lp[i]
        f_order = list(np.argsort(-f)[:10])
        print(f"== {case['name']} pos {i}: oracle chose {step['id']}  FAIE2 argmax {int(np.argmax(f))}")
        print(f"   {'token':>8} {'oracle lp':>10} {'faie2 lp':>10} {'diff':>8}  faie2 rank")
        for tid, olp in step["top"]:
            print(f"   {tid:>8} {olp:>10.4f} {f[tid]:>10.4f} {f[tid]-olp:>8.4f}  {f_order.index(tid) if tid in f_order else '>10'}")
