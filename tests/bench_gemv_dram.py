"""DRAM-bound GEMV speed: the 1.04 GB LM head (248320 x 5120 Q6_K) cannot sit in the 5090's L2 cache.
Correctness of the same kernels is covered by test_q6k_gemv.py; this file only times them."""

import sys

import numpy as np

from faie2 import Faie2, gguf_reader
from test_q6k_gemv import MODEL

versions = [int(v) for v in sys.argv[1:]] or [1, 2]
r = gguf_reader(MODEL)
t = {x.name: x for x in r.tensors}["output.weight"]
cols, rows = int(t.shape[0]), int(t.shape[1])
raw = np.ascontiguousarray(np.asarray(t.data))
eng = Faie2()
for v in versions:
    for nt in (1, 2, 4, 8):
        x = np.random.default_rng(0).standard_normal((nt, cols)).astype(np.float32)
        _, ms = eng.test_q6k_gemv(raw, rows, cols, x, version=v)
        print(f"v{v} lm_head {rows}x{cols} nt={nt}: {ms*1000:8.1f} us  {raw.nbytes / (ms * 1e-3) / 1e9:7.1f} GB/s", flush=True)
