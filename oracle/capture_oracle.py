"""Capture the llama.cpp oracle for FAIE2's verify path.

For each prompt: tokenize with the server, then greedy-generate N tokens with
plain decoding (no speculation) and record the top-K log-probs the target
assigned at every generated position. FAIE2 later runs the same token
sequence through its verify path and must reproduce these distributions.

Usage: python capture_oracle.py --url http://127.0.0.1:8191 --out oracle_q6k.json
"""

import argparse
import datetime
import json
import sys
import urllib.request

# python -I ignores PYTHONIOENCODING, and the Windows console default cannot print Arabic
sys.stdout.reconfigure(encoding="utf-8")

PROMPTS = {
    "prose": "The lighthouse keeper climbed the spiral stairs one last time, and as the lamp",
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n",
    "json": "{\"planets\": [{\"name\": \"Mercury\", \"order\": 1}, {\"name\": \"Venus\", \"order\": 2}, {\"name\":",
    "arabic": "الذكاء الاصطناعي هو مجال من مجالات علوم الحاسوب يهدف إلى",
    "math": "Question: A train travels 120 km in 1.5 hours. What is its average speed?\nAnswer: The average speed is",
    "long": ("In the history of computing, few ideas have been as influential as the stored-program computer. "
             "Before it, machines were rewired for each new task, a process that could take days. "
             "The insight that instructions could live in the same memory as data changed everything: "
             "programs became data that other programs could read, write and transform. "
             "Compilers, operating systems and eventually the web all grew from that single idea. "
             "Today, a phone in your pocket runs billions of instructions per second, yet the basic "
             "architecture would be recognisable to the engineers of the 1940s. What has changed most is"),
}


def post(url, path, body):
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8191")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--n-probs", type=int, default=10)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cases = []
    for name, text in PROMPTS.items():
        tokens = post(args.url, "/tokenize", {"content": text, "add_special": False})["tokens"]
        res = post(args.url, "/completion", {
            "prompt": tokens, "n_predict": args.n_predict, "temperature": 0.0, "top_k": 1,
            "n_probs": args.n_probs, "post_sampling_probs": False, "cache_prompt": False,
            "return_tokens": True, "ignore_eos": True,
        })
        steps = []
        for p in res["completion_probabilities"]:
            steps.append({"id": p["id"], "logprob": p["logprob"],
                          "top": [[t["id"], t["logprob"]] for t in p["top_logprobs"]]})
        cases.append({"name": name, "prompt_tokens": tokens,
                      "generated": [s["id"] for s in steps], "steps": steps,
                      "text": res.get("content", "")})
        print(f"{name:<7} prompt {len(tokens):3d} tokens -> {res.get('content', '')!r}", flush=True)

    out = {"timestamp": datetime.datetime.now().isoformat(timespec="minutes"), "url": args.url,
           "n_probs": args.n_probs, "notes": args.notes, "cases": cases}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
