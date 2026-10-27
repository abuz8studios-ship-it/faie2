"""A few 64-token prefill forwards (graphs on) for an nsys per-kernel breakdown."""

import json

from faie2 import Faie2
from test_spec_e2e import DRAFT, MODEL

data = json.load(open(r"E:\ABU\FAIE2\tests\load_prompts.json", encoding="utf-8"))
prompt = []
for case in data["cases"].values():
    prompt += case["prompt_tokens"]
eng = Faie2()
eng.load(MODEL, max_ctx=4096)
eng.load_drafter(DRAFT)
eng.use_graphs(True)
import ctypes, sys
if "nopdl" in sys.argv:
    eng.lib.faie2_set_pdl(ctypes.c_int(0))
for rep in range(2):
    eng.reset()
    for i in range(0, 64 * 12, 64):
        eng.forward(prompt[i:i + 64], want_logits=False)
print("done")
