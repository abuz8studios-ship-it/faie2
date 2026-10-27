"""Token ids of bench/bench.py's chat prompts (thinking off), plus llama.cpp's own greedy continuation,
so FAIE2's end-to-end speed is measured on exactly the inputs llama.cpp's DFlash2 runs used."""

import importlib.util
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
URL = "http://127.0.0.1:8191"
spec = importlib.util.spec_from_file_location("bench", r"E:\ABU\FAIE2\bench\bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


out = {"eos": [], "cases": {}}
props = json.loads(urllib.request.urlopen(URL + "/props", timeout=30).read())
for name, text in bench.PROMPTS.items():
    tmpl = post("/apply-template", {"messages": [{"role": "user", "content": text}],
                                    "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    toks = post("/tokenize", {"content": tmpl, "add_special": False, "parse_special": True})["tokens"]
    res = post("/completion", {"prompt": toks, "n_predict": 512, "temperature": 0.0, "top_k": 1,
                               "cache_prompt": False, "return_tokens": True})
    out["cases"][name] = {"prompt_tokens": toks, "llama_greedy": res.get("tokens", []), "stop_type": res.get("stop_type"),
                          "llama_text_head": res.get("content", "")[:120]}
    print(f"{name:<12} prompt {len(toks):3d} tokens, llama.cpp greedy {len(res.get('tokens', []))} tokens, "
          f"stop={res.get('stop_type')}", flush=True)
# the end-of-generation tokens: whatever ended the generations that stopped on EOS
for name, c in out["cases"].items():
    if c["stop_type"] == "eos" and c["llama_greedy"]:
        out["eos"].append(c["llama_greedy"][-1])
out["eos"] = sorted(set(out["eos"]))
print("eos ids seen:", out["eos"])
json.dump(out, open(r"E:\ABU\FAIE2\oracle\bench_prompts_q6k.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
