"""Sustained load test, thinking ON, same token ids for both engines.

  python load_test.py capture            # llama-server (DFlash2) must be up on :8191: tokenizes the prompts with
                                         # thinking enabled and records llama.cpp's own greedy run under load
  python load_test.py faie2              # FAIE2 (in-engine DFlash2) on the same prompt ids
  python load_test.py report             # side-by-side table from both result files

Load = every prompt back to back, ROUNDS times, each up to NMAX new tokens (the model thinks first, so outputs
are long and the context grows to thousands of tokens). Decode tok/s = generated tokens / decode wall time
(prefill excluded on both sides: llama.cpp's predicted_ms, FAIE2's loop after prefill). GPU0 clocks, power,
temperature and utilization are sampled every second for the whole run.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8191"
MODEL = r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf"
DRAFT = r"E:\ABU\MODELS\llm\qwen3.8-27b-dflash\dflash-Qwen3.8-27B-Q8_0.gguf"
PROMPT_FILE = os.path.join(HERE, "load_prompts.json")
NMAX = 2048
ROUNDS = 2
MAX_CTX = 4096

PROMPTS = {
    "code": "Implement an LRU cache class in Python with O(1) get and put, using only the standard library. "
            "Explain the design briefly, then give the code and a few unit tests.",
    "math": "Find all real numbers x such that sqrt(x + 3 - 4*sqrt(x - 1)) + sqrt(x + 8 - 6*sqrt(x - 1)) = 1. "
            "Show your reasoning and give the final answer as an interval.",
    "logic": "Five people (Ali, Bea, Chen, Dana, Eli) sit in a row. Ali is not at either end. Bea sits immediately "
             "right of Chen. Dana is somewhere left of Eli. Chen is not next to Ali. Eli is at an end. "
             "List every valid seating order and explain how you found them.",
    "structured": "Extract the entities from this text into JSON with keys people, organizations, places, dates: "
                  "'On 3 March 2025, Dr. Layla Haddad of the Amman Water Authority met engineers from Siemens in "
                  "Aqaba to review the desalination project, which the World Bank agreed to co-finance in June.'",
    "arabic": "اشرح بالتفصيل كيف تعمل الألواح الشمسية الكهروضوئية، ثم قارن بين الألواح أحادية البلورة ومتعددة "
              "البلورات من حيث الكفاءة والتكلفة والعمر الافتراضي.",
    "debug": "This Python function should return the k most frequent words but gives wrong results. Find every bug "
             "and fix it:\n\ndef top_k(text, k):\n    counts = {}\n    for w in text.split(' '):\n"
             "        counts[w] = counts.get(w, 1) + 1\n    return sorted(counts, key=counts.get)[:k]\n",
    "science": "Why is the sky blue during the day but red and orange at sunset? Explain Rayleigh scattering with "
               "the relevant physics and a rough quantitative argument.",
    "plan": "Plan a 3-day technical workshop for 40 engineers on deploying local LLMs: agenda per session, hardware "
            "needed, hands-on labs, and how to measure what participants learned.",
}


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class GpuMonitor:
    """nvidia-smi sampling of physical GPU 0 every second into a CSV."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "w")
        self.p = subprocess.Popen(["nvidia-smi", "-i", "0", "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,"
                                   "temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                                   "-lms", "1000"], stdout=self.f, stderr=subprocess.DEVNULL)

    def stop(self):
        self.p.terminate()
        self.p.wait()
        self.f.close()
        rows = []
        for line in open(self.path):
            parts = [x.strip() for x in line.split(",")]
            try:
                rows.append([float(x) for x in parts[1:]])
            except ValueError:
                pass
        a = np.array(rows) if rows else np.zeros((0, 6))
        busy = a[a[:, 4] > 50] if len(a) else a   # samples while the GPU was working
        def stat(col):
            return {"median": float(np.median(busy[:, col])), "min": float(busy[:, col].min()),
                    "max": float(busy[:, col].max())} if len(busy) else None
        return {"samples": len(a), "busy_samples": len(busy), "sm_mhz": stat(0), "mem_mhz": stat(1),
                "power_w": stat(2), "temp_c": stat(3), "util_pct": stat(4), "mem_used_mb": stat(5)}


def other_gpu0_processes():
    out = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return out.splitlines()


def capture():
    out = {"nmax": NMAX, "rounds": ROUNDS, "eos": [248046], "cases": {}}
    for name, text in PROMPTS.items():
        tmpl = post("/apply-template", {"messages": [{"role": "user", "content": text}],
                                        "chat_template_kwargs": {"enable_thinking": True}})["prompt"]
        toks = post("/tokenize", {"content": tmpl, "add_special": False, "parse_special": True})["tokens"]
        out["cases"][name] = {"prompt_tokens": toks}
    json.dump(out, open(PROMPT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    mon = GpuMonitor(os.path.join(HERE, "load_gpu_llama.csv"))
    runs, t_start = [], time.time()
    for rnd in range(ROUNDS):
        for name, case in out["cases"].items():
            t0 = time.perf_counter()
            res = post("/completion", {"prompt": case["prompt_tokens"], "n_predict": NMAX, "temperature": 0.0,
                                       "top_k": 1, "cache_prompt": False, "return_tokens": True})
            wall = time.perf_counter() - t0
            tm = res.get("timings", {})
            toks = res.get("tokens", [])
            r = {"round": rnd, "name": name, "prompt_n": len(case["prompt_tokens"]), "tokens": len(toks),
                 "decode_ms": tm.get("predicted_ms"), "decode_tok_s": tm.get("predicted_per_second"),
                 "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"), "wall_s": round(wall, 2),
                 "stop": res.get("stop_type")}
            if rnd == 0:
                case["llama_greedy"] = toks
            runs.append(r)
            print(f"llama r{rnd} {name:<10} {r['tokens']:5d} tok  {r['decode_tok_s']:7.1f} tok/s  "
                  f"accepted {r['draft_accepted']}/{r['draft_n']}  stop={r['stop']}", flush=True)
    gpu = mon.stop()
    json.dump(out, open(PROMPT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tot_tok = sum(r["tokens"] for r in runs)
    tot_ms = sum(r["decode_ms"] for r in runs)
    res = {"engine": "llama.cpp DFlash2 (b10510 dflash2 branch), Q6_K + Q8_0 drafter, KV q8_0, ub 256",
           "runs": runs, "total_tokens": tot_tok, "total_decode_s": round(tot_ms / 1e3, 2),
           "aggregate_tok_s": round(tot_tok / (tot_ms / 1e3), 1), "elapsed_s": round(time.time() - t_start, 1),
           "gpu": gpu, "gpu0_processes": other_gpu0_processes()}
    json.dump(res, open(os.path.join(HERE, "load_result_llama.json"), "w"), indent=1)
    print(f"llama.cpp aggregate {res['aggregate_tok_s']} tok/s over {tot_tok} tokens; GPU {gpu['sm_mhz']}")


def run_faie2():
    from faie2 import Faie2
    data = json.load(open(PROMPT_FILE, encoding="utf-8"))
    eos = set(data["eos"])
    dll = os.environ.get("FAIE2_DLL")   # e.g. an archived engine build, for before/after under the same load
    tag = os.environ.get("FAIE2_TAG", "faie2")
    eng = Faie2(dll) if dll else Faie2()
    eng.load(MODEL, max_ctx=MAX_CTX)
    eng.load_drafter(DRAFT)
    eng.use_graphs(True)
    procs = other_gpu0_processes()

    def prefill(prompt):
        eng.reset()
        pre = prompt[:-1]
        for i in range(0, len(pre), 8):
            eng.forward(pre[i:i + 8], want_logits=False)
        return prompt[-1]

    def spec(prompt):
        t = prefill(prompt)
        eng.spec_begin(t)
        out, steps = [], 0
        t0 = time.perf_counter()
        while len(out) < NMAX:
            emit, _ = eng.spec_step()
            steps += 1
            stop = next((i for i, tok in enumerate(emit) if tok in eos), None)
            out += emit if stop is None else emit[:stop + 1]
            if stop is not None:
                break
        return out[:NMAX], time.perf_counter() - t0, steps

    spec(data["cases"]["code"]["prompt_tokens"][:64])   # warm-up: graph capture
    mon = GpuMonitor(os.path.join(HERE, f"load_gpu_{tag}.csv"))
    runs, outs, t_start = [], {}, time.time()
    for rnd in range(data["rounds"]):
        for name, case in data["cases"].items():
            toks, dt, steps = spec(case["prompt_tokens"])
            llama = case.get("llama_greedy", [])
            agree = next((i for i, (a, b) in enumerate(zip(toks, llama)) if a != b), min(len(toks), len(llama)))
            r = {"round": rnd, "name": name, "prompt_n": len(case["prompt_tokens"]), "tokens": len(toks),
                 "decode_s": round(dt, 3), "decode_tok_s": round(len(toks) / dt, 1), "steps": steps,
                 "tok_per_step": round(len(toks) / steps, 2), "same_as_llama_for_first": agree}
            if rnd == 0:
                outs[name] = toks
            elif toks != outs[name]:
                r["round_mismatch"] = True   # greedy must repeat exactly
            runs.append(r)
            print(f"faie2 r{rnd} {name:<10} {r['tokens']:5d} tok  {r['decode_tok_s']:7.1f} tok/s  "
                  f"{r['tok_per_step']:.2f} tok/step  same as llama.cpp for first {agree}", flush=True)
    gpu = mon.stop()
    # losslessness spot check under load: plain greedy (one token per step) must give the same tokens
    lossless = {}
    for name in ("code", "arabic"):
        prompt = data["cases"][name]["prompt_tokens"]
        t = prefill(prompt)
        plain = []
        while len(plain) < len(outs[name]):
            emit, _, _ = eng.verify([t])
            plain.append(emit[0])
            t = emit[0]
            if t in eos:
                break
        lossless[name] = plain == outs[name]
        print(f"lossless check {name}: {lossless[name]} ({len(plain)} tokens)", flush=True)
    tot_tok = sum(r["tokens"] for r in runs)
    tot_s = sum(r["decode_s"] for r in runs)
    res = {"engine": "FAIE2 (in-engine DFlash2), Q6_K + Q8_0 drafter, fp16 KV, dll=" + (dll or "engine/faie2.dll"), "runs": runs, "total_tokens": tot_tok,
           "total_decode_s": round(tot_s, 2), "aggregate_tok_s": round(tot_tok / tot_s, 1),
           "elapsed_s": round(time.time() - t_start, 1), "gpu": gpu, "gpu0_processes_at_start": procs,
           "lossless_spot_check": lossless, "outputs_round0": outs}
    json.dump(res, open(os.path.join(HERE, f"load_result_{tag}.json"), "w"), indent=1)
    print(f"FAIE2 aggregate {res['aggregate_tok_s']} tok/s over {tot_tok} tokens; GPU {gpu['sm_mhz']}")


def report():
    la = json.load(open(os.path.join(HERE, "load_result_llama.json")))
    fa = json.load(open(os.path.join(HERE, "load_result_faie2.json")))
    print(f"{'prompt':<11} {'llama tok':>9} {'llama tok/s':>11} {'FAIE2 tok':>9} {'FAIE2 tok/s':>11} {'speedup':>8}")
    for name in PROMPTS:
        l = [r for r in la["runs"] if r["name"] == name]
        f = [r for r in fa["runs"] if r["name"] == name]
        lt = sum(r["tokens"] for r in l) / (sum(r["decode_ms"] for r in l) / 1e3)
        ft = sum(r["tokens"] for r in f) / sum(r["decode_s"] for r in f)
        print(f"{name:<11} {l[0]['tokens']:>9} {lt:>11.1f} {f[0]['tokens']:>9} {ft:>11.1f} {ft / lt:>7.2f}x")
    print(f"{'ALL':<11} {la['total_tokens']:>9} {la['aggregate_tok_s']:>11.1f} {fa['total_tokens']:>9} "
          f"{fa['aggregate_tok_s']:>11.1f} {fa['aggregate_tok_s'] / la['aggregate_tok_s']:>7.2f}x")
    print("GPU llama:", la["gpu"]["sm_mhz"], la["gpu"]["power_w"], la["gpu"]["temp_c"])
    print("GPU FAIE2:", fa["gpu"]["sm_mhz"], fa["gpu"]["power_w"], fa["gpu"]["temp_c"])


if __name__ == "__main__":
    {"capture": capture, "faie2": run_faie2, "report": report}[sys.argv[1]]()
