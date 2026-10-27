"""Speculative steps (graphs on) on the code prompt, for an nsys per-kernel breakdown of one spec step.
Run under: nsys profile --cuda-graph-trace=node ... python profile_spec.py"""

import ctypes
import json
import sys

from faie2 import Faie2
from test_spec_e2e import DRAFT, MODEL, PROMPTS, prefill

STEPS = 60
eng = Faie2()
eng.load(MODEL, max_ctx=4096)
eng.load_drafter(DRAFT)
eng.use_graphs(True)
if "nopdl" in sys.argv:
    eng.lib.faie2_set_pdl(ctypes.c_int(0))
for a in sys.argv[1:]:
    if a.startswith("attn="):
        eng.lib.faie2_set_attn_version(ctypes.c_int(int(a[5:])))
t = prefill(eng, PROMPTS["cases"]["code"]["prompt_tokens"])
eng.spec_begin(t)
n = 0
for _ in range(STEPS):
    emit, _ = eng.spec_step()
    n += len(emit)
print(json.dumps({"steps": STEPS, "tokens": n, "last_gpu_ms": round(eng.gpu_ms(), 3)}))
