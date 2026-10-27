"""Gate: FAIE2's 8-token verify path vs the llama.cpp oracle (oracle/oracle_q6k.json).

Per case: prefill prompt[:-1] in chunks of <= 8 tokens, then two 8-token verify blocks
[prompt[-1], g0..g6] and [g7..g14]. Row i of a block predicts the next generated token; its
distribution is compared with the top-10 log-probs llama.cpp reported at that position.
"""

import argparse
import json
import sys
import time

import numpy as np

from faie2 import Faie2

sys.stdout.reconfigure(encoding="utf-8")
MODEL = r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf"
ORACLE = r"E:\ABU\FAIE2\oracle\oracle_q6k.json"
NEAR_TIE = 0.10   # nats; a top-1 flip inside this oracle margin is a rounding tie, not a bug


def log_softmax(z):
    z = z.astype(np.float64)
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def run_case(eng, case, block=8):
    eng.reset()
    p, g = case["prompt_tokens"], case["generated"]
    pre = p[:-1]
    for i in range(0, len(pre), block):
        eng.forward(pre[i:i + block], want_logits=False)
    rows, times = [], []
    seq = [p[-1]] + g[:-1]          # 16 tokens -> predicts g[0..15]
    for i in range(0, len(seq), block):
        lg = eng.forward(seq[i:i + block])
        times.append(eng.gpu_ms())
        rows.append(log_softmax(lg))
    return np.concatenate(rows), times


def compare(lp, case):
    out = []
    for i, step in enumerate(case["steps"]):
        top = step["top"]
        o_ids = [t[0] for t in top]
        o_lp = np.array([t[1] for t in top])
        f = lp[i]
        f_top5 = list(np.argsort(-f)[:5])
        margin = o_lp[0] - o_lp[1]
        out.append({
            "pos": i,
            "top1_match": int(np.argmax(f)) == step["id"],
            "near_tie": bool(margin < NEAR_TIE),
            "top5_overlap": len(set(f_top5) & set(o_ids[:5])),
            "d_lp_top1": float(abs(f[step["id"]] - step["logprob"])),
            "d_lp_top10_max": float(np.abs(f[o_ids] - o_lp).max()),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--out", default=r"E:\ABU\FAIE2\tests\gate_oracle_result.json")
    args = ap.parse_args()

    oracle = json.load(open(ORACLE, encoding="utf-8"))
    eng = Faie2()
    t0 = time.time()
    eng.load(MODEL, max_ctx=4096)
    print(f"loaded in {time.time() - t0:.1f} s", flush=True)
    eng.use_graphs(args.graphs)

    all_rows, summary = [], []
    for case in oracle["cases"]:
        lp, times = run_case(eng, case)
        res = compare(lp, case)
        all_rows += res
        bad = [r for r in res if not r["top1_match"] and not r["near_tie"]]
        ties = [r for r in res if not r["top1_match"] and r["near_tie"]]
        s = {"case": case["name"], "positions": len(res),
             "top1_match": sum(r["top1_match"] for r in res), "real_mismatch": len(bad), "tie_flips": len(ties),
             "mean_top5_overlap": round(float(np.mean([r["top5_overlap"] for r in res])), 2),
             "max_d_lp_top1": round(max(r["d_lp_top1"] for r in res), 4),
             "max_d_lp_top10": round(max(r["d_lp_top10_max"] for r in res), 4),
             "verify_gpu_ms": [round(t, 2) for t in times]}
        summary.append(s)
        print(f"{s['case']:<7} top1 {s['top1_match']}/{s['positions']} (real mismatches {s['real_mismatch']}, tie flips "
              f"{s['tie_flips']})  top5 overlap {s['mean_top5_overlap']}  max|dlp| top1 {s['max_d_lp_top1']} "
              f"top10 {s['max_d_lp_top10']}  verify ms {s['verify_gpu_ms']}", flush=True)

    n = len(all_rows)
    real = sum(1 for r in all_rows if not r["top1_match"] and not r["near_tie"])
    gate = {
        "positions": n,
        "top1_match": sum(r["top1_match"] for r in all_rows),
        "real_mismatches": real,
        "mean_top5_overlap": round(float(np.mean([r["top5_overlap"] for r in all_rows])), 3),
        "max_d_lp_top1": round(max(r["d_lp_top1"] for r in all_rows), 4),
        "p95_d_lp_top10": round(float(np.percentile([r["d_lp_top10_max"] for r in all_rows], 95)), 4),
    }
    # llama.cpp is not exact: it rounds activations to 8 bits inside its matmuls. A float64 reference
    # (ref_forward.py / compare_ref.py) showed llama.cpp off by up to 0.28 nats on top-1 log-probs where
    # FAIE2 is off by <= 0.002, so log-prob closeness is judged against that reference, not here. This
    # gate checks what llama.cpp can vouch for: the same greedy token everywhere and the same top-5.
    gate["pass"] = real == 0 and gate["mean_top5_overlap"] >= 4.5
    print("GATE", json.dumps(gate))
    json.dump({"graphs": args.graphs, "gate": gate, "cases": summary, "rows": all_rows},
              open(args.out, "w", encoding="utf-8"), indent=1)
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
