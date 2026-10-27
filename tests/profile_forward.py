"""Run a few 1-token and 8-token forwards (no graphs) for an nsys kernel breakdown."""

import sys

from faie2 import Faie2
from gate_oracle import MODEL

nt = int(sys.argv[1]) if len(sys.argv) > 1 else 8
eng = Faie2()
eng.load(MODEL, max_ctx=4096)
eng.use_graphs(False)
eng.reset()
prompt = list(range(1000, 1120))
for i in range(0, len(prompt), 8):
    eng.forward(prompt[i:i + 8], want_logits=False)
for _ in range(10):
    eng.forward(list(range(2000, 2000 + nt)))
print("done", eng.gpu_ms())
