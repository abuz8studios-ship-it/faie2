"""Two FAIE2 builds diverge on a load-test output: which one does exact float64 math side with?
usage: check_flip.py <case> <result_a.json> <result_b.json>"""

import json
import sys

import numpy as np

from detok import decode

sys.stdout.reconfigure(encoding="utf-8")
case, fa, fb = sys.argv[1:4]
prompt = json.load(open(r"E:\ABU\FAIE2\tests\load_prompts.json", encoding="utf-8"))["cases"][case]["prompt_tokens"]
a = json.load(open(fa, encoding="utf-8"))["outputs_round0"][case]
b = json.load(open(fb, encoding="utf-8"))["outputs_round0"][case]
k = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
print(f"{case}: first difference at generated token {k}: A={a[k]} {decode([a[k]])!r}  B={b[k]} {decode([b[k]])!r}")
print("context tail:", repr(decode(a[max(0, k - 12):k])))

import ref_forward  # noqa: E402

lp = np.asarray(ref_forward.forward(prompt + a[:k], 1), dtype=np.float64).reshape(-1)
idx = np.argsort(-lp)[:5]
print("float64 top:", [(int(i), decode([int(i)]), round(float(lp[i]), 4)) for i in idx])
print(f"float64: log p(A) - log p(B) = {lp[a[k]] - lp[b[k]]:+.4f} nats -> exact math picks {'A' if lp[a[k]] > lp[b[k]] else 'B'}")
