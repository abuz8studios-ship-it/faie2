"""Where FAIE2 and llama.cpp pick different tokens on a load-test prompt, which one does exact math pick?
FAIE2 log-probs, then the independent float64 reference (ref_forward.py) at the same position.
usage: check_first_token.py <case> [n_generated_prefix]   (prefix = how many shared generated tokens to feed first)"""

import json
import sys

import numpy as np

from detok import decode
from faie2 import Faie2

sys.stdout.reconfigure(encoding="utf-8")
case = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
d = json.load(open(r"E:\ABU\FAIE2\tests\load_prompts.json", encoding="utf-8"))["cases"][case]
fa = json.load(open(r"E:\ABU\FAIE2\tests\load_result_faie2.json", encoding="utf-8"))["outputs_round0"][case]
llama = d["llama_greedy"]
seq = d["prompt_tokens"] + llama[:k]
print(f"position after {k} shared generated tokens; llama.cpp next = {llama[k]} {decode([llama[k]])!r}, "
      f"FAIE2 next = {fa[k]} {decode([fa[k]])!r}")


def top(lp, n=6):
    idx = np.argsort(-lp)[:n]
    return [(int(i), decode([int(i)]), round(float(lp[i]), 4)) for i in idx]


def logsm(z):
    z = z.astype(np.float64)
    m = z.max()
    return z - m - np.log(np.exp(z - m).sum())


eng = Faie2()
eng.load(r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf", max_ctx=4096)
pre = seq[:-1]
for i in range(0, len(pre), 8):
    eng.forward(pre[i:i + 8], want_logits=False)
lp_f = logsm(eng.forward([seq[-1]])[0])
print("FAIE2  top:", top(lp_f))
eng.lib.faie2_free()

import ref_forward  # noqa: E402  float64, independent of FAIE2 code
lp_r = ref_forward.forward(seq, 1)
lp_r = np.asarray(lp_r.cpu() if hasattr(lp_r, "cpu") else lp_r, dtype=np.float64).reshape(-1)
print("float64 top:", top(lp_r))
a, b = llama[k], fa[k]
print(f"float64 log-prob gap  llama.cpp's pick - FAIE2's pick = {lp_r[a] - lp_r[b]:+.4f} nats")
print(f"FAIE2   log-prob gap  llama.cpp's pick - FAIE2's pick = {lp_f[a] - lp_f[b]:+.4f} nats")
