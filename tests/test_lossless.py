"""Phase 3 gate: speculative decoding in FAIE2 is lossless.

Reference: plain greedy decoding, one token per step (faie2_verify with a 1-token block).
Speculative runs feed 8-token blocks [t0, d1..d7] from test drafters that know the reference continuation
and corrupt it on purpose, so every accept count k = 0..7 and every rollback point gets exercised:
  perfect  - the true continuation (expect k = number of drafts)
  corrupt  - true continuation with one wrong token at draft j (expect k = j - 1)
  random   - random tokens (expect k = 0, barring a lucky hit)
  mixed    - random draft length 1..7, corrupted or not
Pass = every emitted token equals the reference, every logits row behind an emitted token is
bit-identical to the reference row, and every k equals what the drafter forced.
"""

import collections
import json
import random
import sys
import time

import numpy as np

from faie2 import VOCAB, Faie2
from gate_oracle import MODEL, ORACLE

sys.stdout.reconfigure(encoding="utf-8")
N = 120


def prefill(eng, prompt):
    eng.reset()
    pre = prompt[:-1]
    for i in range(0, len(pre), 8):
        eng.forward(pre[i:i + 8], want_logits=False)
    return prompt[-1]


def plain(eng, prompt):
    t = prefill(eng, prompt)
    toks, rows = [], []
    for _ in range(N):
        emit, _, lg = eng.verify([t], want_logits=True)
        toks.append(emit[0])
        rows.append(lg[0].copy())
        t = emit[0]
    return toks, rows


def wrong(tok, rng):
    w = rng.randrange(VOCAB)
    return w if w != tok else (tok + 1) % VOCAB


def speculative(eng, prompt, G, R, mode, rng):
    t = prefill(eng, prompt)
    out, ks, bad = [], collections.Counter(), []
    while len(out) < N:
        pos = len(out)
        room = N - pos                     # tokens still wanted; a block of m drafts emits up to m + 1
        m = min(7, room - 1) if mode != "mixed" else min(rng.randint(1, 7), room - 1)
        truth = G[pos:pos + m]
        drafts, expect_k = list(truth), m
        if mode == "corrupt" or (mode == "mixed" and rng.random() < 0.6):
            if m > 0:
                j = rng.randint(1, m)
                drafts[j - 1] = wrong(truth[j - 1], rng)
                expect_k = j - 1
        elif mode == "random":
            drafts = [rng.randrange(VOCAB) for _ in range(m)]
            expect_k = next((i for i, (d, g) in enumerate(zip(drafts, truth)) if d != g), m)
        emit, _, lg = eng.verify([t] + drafts, want_logits=True)
        k = len(emit) - 1
        ks[k] += 1
        if k != expect_k:
            bad.append(f"pos {pos}: k={k} expected {expect_k}")
        for i, tok in enumerate(emit):
            if pos + i >= N:
                break
            if tok != G[pos + i]:
                bad.append(f"pos {pos + i}: token {tok} != reference {G[pos + i]}")
            elif not np.array_equal(lg[i], R[pos + i]):
                bad.append(f"pos {pos + i}: logits differ, max|d|={np.abs(lg[i] - R[pos + i]).max():.3e}")
        out += emit
        t = emit[-1]
    return out[:N], ks, bad


def main():
    graphs = "--no-graphs" not in sys.argv
    oracle = {c["name"]: c for c in json.load(open(ORACLE, encoding="utf-8"))["cases"]}
    eng = Faie2()
    eng.load(MODEL, max_ctx=4096)
    eng.use_graphs(graphs)
    rng = random.Random(1234)
    report, ok = {"graphs": graphs, "cases": {}}, True
    for name in ("code", "prose", "arabic"):
        prompt = oracle[name]["prompt_tokens"]
        t0 = time.time()
        G, R = plain(eng, prompt)
        llama16 = G[:16] == oracle[name]["generated"]
        case = {"plain_matches_llama_first16": llama16, "modes": {}}
        print(f"== {name}: plain greedy {N} tokens in {time.time() - t0:.1f} s; first 16 == llama.cpp greedy: {llama16}", flush=True)
        ok &= llama16
        for mode in ("perfect", "corrupt", "random", "mixed"):
            S, ks, bad = speculative(eng, prompt, G, R, mode, rng)
            same = S == G and not bad
            ok &= same
            hist = dict(sorted(ks.items()))
            case["modes"][mode] = {"lossless": same, "k_histogram": hist, "problems": bad[:5]}
            print(f"   {mode:<8} lossless={same}  steps={sum(ks.values()):3d}  k histogram {hist}"
                  + (f"  PROBLEMS {bad[:3]}" if bad else ""), flush=True)
        report["cases"][name] = case
    all_k = set()
    for c in report["cases"].values():
        for m in c["modes"].values():
            all_k |= set(m["k_histogram"])
    report["k_values_exercised"] = sorted(all_k)
    report["pass"] = ok and all_k >= set(range(8))
    print("k values exercised:", sorted(all_k))
    print("LOSSLESS GATE", "PASS" if report["pass"] else "FAIL")
    json.dump(report, open(r"E:\ABU\FAIE2\tests\lossless_result.json", "w"), indent=1)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
