"""Phase 4: end-to-end speculative decoding in FAIE2 (in-engine DFlash2 drafter), per bench category.

For each of bench/bench.py's chat prompts (exact token ids llama-server used, oracle/bench_prompts_q6k.json):
  plain  - FAIE2 greedy, one token per step (the reference output)
  spec   - FAIE2 faie2_spec_step loop: draft + verify + accept + commit + inject, one CUDA graph per step
Gate: spec output == plain output token for token (lossless). Speed: decode tok/s = tokens / wall time of the
decode loop (prefill excluded, as bench.py excludes time-to-first-token), stopping at <|im_end|> or 512.
"""

import json
import subprocess
import sys
import time

import numpy as np

from faie2 import Faie2

sys.stdout.reconfigure(encoding="utf-8")
MODEL = r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf"
DRAFT = r"E:\ABU\MODELS\llm\qwen3.8-27b-dflash\dflash-Qwen3.8-27B-Q8_0.gguf"
PROMPTS = json.load(open(r"E:\ABU\FAIE2\oracle\bench_prompts_q6k.json", encoding="utf-8"))
LLAMA_DFLASH2 = {"code": 213.2, "structured": 220.9, "translation": 145.6, "prose": 95.0}   # runs/q6k_dflash2_k7_lowmem.json
EOS = set(PROMPTS["eos"])
NMAX = 512


def prefill(eng, prompt):
    eng.reset()
    pre = prompt[:-1]
    for i in range(0, len(pre), 8):
        eng.forward(pre[i:i + 8], want_logits=False)
    return prompt[-1]


def run_plain(eng, prompt):
    t = prefill(eng, prompt)
    out = []
    t0 = time.perf_counter()
    while len(out) < NMAX:
        emit, _, _ = eng.verify([t])
        out.append(emit[0])
        t = emit[0]
        if t in EOS:
            break
    return out, time.perf_counter() - t0


def run_spec(eng, prompt):
    t = prefill(eng, prompt)
    eng.spec_begin(t)
    out, steps, gpu = [], 0, []
    t0 = time.perf_counter()
    while len(out) < NMAX:
        emit, _ = eng.spec_step()
        steps += 1
        gpu.append(eng.gpu_ms())
        stop = next((i for i, tok in enumerate(emit) if tok in EOS), None)
        out += emit if stop is None else emit[:stop + 1]
        if stop is not None:
            break
    dt = time.perf_counter() - t0
    return out[:NMAX], dt, steps, float(np.median(gpu))


def main():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("GPU state before:", smi.replace("\n", " | "))
    eng = Faie2()
    eng.load(MODEL, max_ctx=4096)
    eng.load_drafter(DRAFT)
    eng.use_graphs(True)
    run_spec(eng, PROMPTS["cases"]["code"]["prompt_tokens"])   # warm-up: graph capture
    rows, ok = {}, True
    for name, case in PROMPTS["cases"].items():
        P = case["prompt_tokens"]
        plain, t_plain = run_plain(eng, P)
        spec, t_spec, steps, gpu_ms = run_spec(eng, P)
        lossless = spec == plain
        ok &= lossless
        llama = case["llama_greedy"]
        agree = next((i for i, (a, b) in enumerate(zip(spec, llama)) if a != b), min(len(spec), len(llama)))
        r = {"tokens": len(spec), "spec_tok_s": round(len(spec) / t_spec, 1), "plain_tok_s": round(len(plain) / t_plain, 1),
             "steps": steps, "tokens_per_step": round(len(spec) / steps, 2), "gpu_ms_per_step": round(gpu_ms, 2),
             "lossless_vs_plain": lossless, "same_as_llama_greedy_for_first": agree,
             "llama_dflash2_tok_s": LLAMA_DFLASH2[name]}
        rows[name] = r
        print(f"{name:<12} {r['tokens']:3d} tok  FAIE2 spec {r['spec_tok_s']:6.1f} tok/s  (plain {r['plain_tok_s']:5.1f})  "
              f"{r['tokens_per_step']:.2f} tok/step  {r['gpu_ms_per_step']:.2f} ms/step  lossless={lossless}  "
              f"llama.cpp DFlash2 {r['llama_dflash2_tok_s']} tok/s  same as llama.cpp greedy for first {agree}", flush=True)
    mean_f = np.mean([r["spec_tok_s"] for r in rows.values()])
    mean_l = np.mean(list(LLAMA_DFLASH2.values()))
    print(f"mean over categories: FAIE2 {mean_f:.1f} tok/s vs llama.cpp DFlash2 {mean_l:.1f} tok/s; lossless all: {ok}")
    json.dump({"gpu_before": smi, "rows": rows, "mean_faie2": round(float(mean_f), 1), "mean_llama": round(float(mean_l), 1),
               "lossless_all": ok}, open(r"E:\ABU\FAIE2\tests\spec_e2e_result.json", "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
