"""Prompt-reading vs generation time per tool case, FAIE2 vs llama.cpp (tools_faie2.json / tools_llamacpp.json)."""

import json

f = json.load(open("tools_faie2.json", encoding="utf-8"))
l = json.load(open("tools_llamacpp.json", encoding="utf-8"))
print(f"{'case':<12} {'prompt':>6} {'cached':>6} {'FAIE2 pp ms':>11} {'gen':>5} {'gen ms':>7} | {'llama pp ms':>11} {'gen':>5} {'gen ms':>7}")
for a, b in zip(f["results"], l["results"]):
    ta, tb = a["timings"], b["timings"]
    print(f"{a['case']:<12} {ta['prompt_n']:>6} {ta['cached_n']:>6} {ta['prompt_ms']:>11.0f} {ta['predicted_n']:>5} "
          f"{ta['predicted_ms']:>7.0f} | {tb.get('prompt_ms', 0):>11.0f} {tb.get('predicted_n', 0):>5} {tb.get('predicted_ms', 0):>7.0f}")
pa = sum(r["timings"]["prefilled_n"] for r in f["results"])
ma = sum(r["timings"]["prompt_ms"] for r in f["results"])
pb = sum(r["timings"].get("prompt_n", 0) for r in l["results"])
mb = sum(r["timings"].get("prompt_ms", 0) for r in l["results"])
ga = sum(r["timings"]["predicted_n"] for r in f["results"])
gma = sum(r["timings"]["predicted_ms"] for r in f["results"])
gb = sum(r["timings"].get("predicted_n", 0) for r in l["results"])
gmb = sum(r["timings"].get("predicted_ms", 0) for r in l["results"])
print(f"prompt reading: FAIE2 {pa / ma * 1e3:.0f} tok/s ({pa} tokens read, the rest from cache), llama.cpp {pb / mb * 1e3:.0f} tok/s")
print(f"generation:     FAIE2 {ga / gma * 1e3:.0f} tok/s, llama.cpp {gb / gmb * 1e3:.0f} tok/s")
