"""Time FAIE2 speculative steps (forward + device argmax + accept + commit) vs plain greedy steps.

Drafts here are deliberately wrong (k = 0), so every step costs a full 8-token verify but commits one token
and the context grows slowly. The cost of a step does not depend on k. Tok/s at an acceptance length L is
then L / step time; drafter cost is NOT included (the drafter is not in the engine yet).
"""

import json
import statistics as st
import sys

from faie2 import Faie2
from gate_oracle import MODEL, ORACLE

sys.stdout.reconfigure(encoding="utf-8")
eng = Faie2()
eng.load(MODEL, max_ctx=4096)
eng.use_graphs(True)
prompt = {c["name"]: c for c in json.load(open(ORACLE, encoding="utf-8"))["cases"]}["long"]["prompt_tokens"]
res = {}
for nt in (1, 8):
    eng.reset()
    for i in range(0, len(prompt) - 1, 8):
        eng.forward(prompt[i:min(i + 8, len(prompt) - 1)], want_logits=False)
    t = prompt[-1]
    gpu, host = [], []
    for step in range(60):
        emit, _, _ = eng.verify([t] + [0] * (nt - 1))
        if step >= 10:   # skip graph capture and warm-up
            gpu.append(eng.gpu_ms())
            host.append(eng.host_ms())
        t = emit[-1]
    res[nt] = {"gpu_ms": round(st.median(gpu), 3), "host_ms": round(st.median(host), 3)}
    print(f"verify nt={nt}: GPU {res[nt]['gpu_ms']:.3f} ms  host {res[nt]['host_ms']:.3f} ms per step", flush=True)

plain = 1000 / res[1]["host_ms"]
step = res[8]["host_ms"]
print(f"plain greedy decode: {plain:.1f} tok/s (host-measured)")
print("speculative, 8-token blocks, drafter cost excluded (acceptance lengths from the llama.cpp DFlash2 runs):")
table = {}
for label, L in (("prose", 2.57), ("code", 5.75), ("translation", 5.86), ("structured", 6.83), ("perfect drafter", 8.0)):
    table[label] = round(L * 1000 / step, 1)
    print(f"   {label:<16} L={L:<5} -> {table[label]:7.1f} tok/s")
json.dump({"steps": res, "plain_tok_s": round(plain, 1), "spec_tok_s_no_drafter": table},
          open(r"E:\ABU\FAIE2\tests\bench_spec_result.json", "w"), indent=1)
