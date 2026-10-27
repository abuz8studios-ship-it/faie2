"""Time FAIE2 forwards: GPU ms for 1-token decode and 8-token verify, with and without CUDA graphs.

llama.cpp reference on the same Q6_K file (oracle/llama_bench_q6k.md): tg 57.19 tok/s = 17.49 ms per token,
pp8 323.29 tok/s = 24.75 ms per 8-token batch.
"""

import json
import statistics as st
import sys
import time

from faie2 import Faie2
from gate_oracle import MODEL

sys.stdout.reconfigure(encoding="utf-8")
LLAMA = {"1": 17.49, "8": 24.75}


def measure(eng, nt, graphs, reps=30, ctx_tokens=120):
    eng.use_graphs(graphs)
    eng.reset()
    prompt = list(range(1000, 1000 + ctx_tokens))
    for i in range(0, ctx_tokens, 8):
        eng.forward(prompt[i:i + 8], want_logits=False)
    toks = list(range(2000, 2000 + nt))
    eng.forward(toks)            # warm-up (and graph capture)
    gpu, host = [], []
    for _ in range(reps):
        eng.forward(toks)
        gpu.append(eng.gpu_ms())
        host.append(eng.host_ms())
    return st.median(gpu), st.median(host)


def main():
    eng = Faie2()
    t0 = time.time()
    eng.load(MODEL, max_ctx=4096)
    print(f"loaded in {time.time() - t0:.1f} s", flush=True)
    res = {}
    for graphs in (False, True):
        for nt in (1, 8):
            g, h = measure(eng, nt, graphs)
            res[f"nt{nt}_graphs{int(graphs)}"] = {"gpu_ms": round(g, 3), "host_ms": round(h, 3)}
            print(f"graphs={graphs!s:<5} nt={nt}: GPU {g:7.3f} ms   host {h:7.3f} ms   "
                  f"(llama.cpp {LLAMA[str(nt)]} ms)  -> {nt * 1000 / g:7.1f} tok/s of forward", flush=True)
    for graphs in (0, 1):
        r1, r8 = res[f"nt1_graphs{graphs}"]["gpu_ms"], res[f"nt8_graphs{graphs}"]["gpu_ms"]
        print(f"graphs={graphs}: 8-token / 1-token cost = {r8 / r1:.3f}x   (llama.cpp {LLAMA['8'] / LLAMA['1']:.3f}x)")
    if len(sys.argv) > 1:
        json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
