"""Gate + speed: prefill in 64-token forwards must leave exactly the state 8-token forwards leave.

The same ~1.5K-token prompt is prefilled three ways (8-token chunks, 64-token chunks, ragged chunks); the logits of
the next forward must be bit-identical, and so must a 128-token speculative continuation. Prefill tok/s is timed
for each (graphs on, after a warm-up that captures them)."""

import json
import sys
import time

import numpy as np

from faie2 import Faie2
from test_spec_e2e import DRAFT, MODEL

sys.stdout.reconfigure(encoding="utf-8")
data = json.load(open(r"E:\ABU\FAIE2\tests\load_prompts.json", encoding="utf-8"))
prompt = []
for case in data["cases"].values():
    prompt += case["prompt_tokens"]
prompt = prompt[:1537]
eng = Faie2()
eng.load(MODEL, max_ctx=4096)
eng.load_drafter(DRAFT)
eng.use_graphs(True)


def prefill(chunks):
    eng.reset()
    pos = 0
    t0 = time.perf_counter()
    for c in chunks:
        eng.forward(prompt[pos:pos + c], want_logits=False)
        pos += c
    dt = time.perf_counter() - t0
    assert pos == len(prompt) - 1
    return dt


def chunks_of(n, total):
    out = []
    while total > 0:
        out.append(min(n, total))
        total -= out[-1]
    return out


def ragged(total):
    pat, out = [37, 64, 13, 8, 64, 1, 50], []
    i = 0
    while total > 0:
        out.append(min(pat[i % len(pat)], total))
        total -= out[-1]
        i += 1
    return out


n = len(prompt) - 1
plans = {"8-token": chunks_of(8, n), "64-token": chunks_of(64, n), "ragged": ragged(n)}
for p in plans.values():          # warm-up: capture every graph size used
    prefill(p)
res, ok = {}, True
for name, plan in plans.items():
    dt = prefill(plan)
    logits = eng.forward([prompt[-1]])[0]
    eng.spec_begin(int(np.argmax(logits)))
    cont = []
    while len(cont) < 128:
        emit, _ = eng.spec_step()
        cont += emit
    res[name] = (dt, logits, cont[:128])
    print(f"{name:<9} prefill {n} tokens in {dt * 1e3:7.1f} ms = {n / dt:7.0f} tok/s", flush=True)
ref = res["8-token"]
for name in ("64-token", "ragged"):
    same_l = np.array_equal(res[name][1], ref[1])
    same_c = res[name][2] == ref[2]
    ok &= same_l and same_c
    print(f"{name:<9} next-token logits bit-identical to 8-token prefill: {same_l}; 128-token continuation identical: {same_c}")
print("PREFILL GATE PASS" if ok else "PREFILL GATE FAIL")
sys.exit(0 if ok else 1)
