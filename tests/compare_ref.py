"""Three-way check at every oracle position: float64 reference vs FAIE2 vs llama.cpp (top-10 tokens)."""

import json
import sys

import numpy as np

from faie2 import Faie2
from gate_oracle import MODEL, ORACLE, run_case

sys.stdout.reconfigure(encoding="utf-8")
oracle = json.load(open(ORACLE, encoding="utf-8"))
eng = Faie2()
eng.load(MODEL, max_ctx=4096)
summary = {}
for case in oracle["cases"]:
    name = case["name"]
    try:
        ref = np.load(rf"E:\ABU\FAIE2\tests\ref_{name}.npy").astype(np.float64)
    except FileNotFoundError:
        continue
    fa, _ = run_case(eng, case)
    rows = []
    for i, step in enumerate(case["steps"]):
        ids = [t[0] for t in step["top"]]
        llama = np.array([t[1] for t in step["top"]])
        r, f = ref[i][ids], fa[i][ids]
        p_ref = np.exp(ref[i])
        kl_f = float((p_ref * (ref[i] - fa[i])).sum())
        rows.append({"pos": i, "faie2_err_top1": abs(f[0] - r[0]), "llama_err_top1": abs(llama[0] - r[0]),
                     "faie2_err_top10": np.abs(f - r).max(), "llama_err_top10": np.abs(llama - r).max(),
                     "kl_ref_faie2": kl_f, "ref_top1_is_oracle_top1": int(np.argmax(ref[i])) == step["id"]})
    print(f"== {name}")
    print(f"   pos  |faie2-ref| top1  |llama-ref| top1   |faie2-ref| top10  |llama-ref| top10   KL(ref||faie2)")
    for r_ in rows:
        print(f"   {r_['pos']:>3}  {r_['faie2_err_top1']:>14.4f}  {r_['llama_err_top1']:>15.4f}   {r_['faie2_err_top10']:>16.4f}"
              f"  {r_['llama_err_top10']:>16.4f}   {r_['kl_ref_faie2']:.2e}")
    s = {k: float(np.max([r_[k] for r_ in rows])) for k in ("faie2_err_top1", "llama_err_top1", "faie2_err_top10",
                                                              "llama_err_top10", "kl_ref_faie2")}
    s["ref_agrees_with_oracle_top1"] = sum(r_["ref_top1_is_oracle_top1"] for r_ in rows)
    summary[name] = s
    print("   MAX", {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
json.dump(summary, open(r"E:\ABU\FAIE2\tests\compare_ref_result.json", "w"), indent=1)
