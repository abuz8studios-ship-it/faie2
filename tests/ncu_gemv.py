"""Run one GEMV case for Nsight Compute: python ncu_gemv.py <tensor> <nt> <version>"""

import sys

import numpy as np

from faie2 import Faie2, gguf_reader
from test_q6k_gemv import MODEL

name, nt, version = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = gguf_reader(MODEL)
t = {x.name: x for x in r.tensors}[name]
cols, rows = int(t.shape[0]), int(t.shape[1])
x = np.random.default_rng(0).standard_normal((nt, cols)).astype(np.float32)
y, ms = Faie2().test_q6k_gemv(np.asarray(t.data), rows, cols, x, version=version)
print(f"{name} nt={nt} v{version}: {ms * 1000:.1f} us")
