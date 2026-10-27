"""FAIE2 decode benchmark: engine-agnostic, OpenAI-compatible, proof-first.

Measures single-stream decode speed per prompt category against any
OpenAI-compatible server (llama-server, vLLM, SGLang, NInfer, FAIE2).

Decode tok/s is measured on the client from the first streamed token to the
last, so it excludes prefill and does not trust the server's own numbers.
Server-reported timings and speculative acceptance are recorded beside it.

Usage:
  python bench.py --url http://127.0.0.1:8191 --label plain --out runs/plain.json
  python bench.py --url ... --label mtp --out runs/mtp.json --compare runs/plain.json

--compare checks the greedy (temperature 0) output text against an earlier
run: speculative decoding is lossless, so any mismatch means the fast path is
doing different work and its speed number is not valid.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 date string into a datetime object "
            "without using any library besides the standard library. Include docstring and tests.",
    "structured": "Return a JSON array of 12 objects describing the planets and dwarf planets of the "
                  "solar system. Each object has keys: name, type, mean_radius_km, moons, order_from_sun.",
    "translation": "Translate this paragraph into formal English, sentence by sentence: "
                   "يعتبر الذكاء الاصطناعي من أهم التقنيات في العصر الحديث، حيث يساعد في تحسين الإنتاجية "
                   "وتسريع اتخاذ القرارات في مختلف المجالات مثل الطب والتعليم والصناعة والتجارة.",
    "prose": "Write an original short story of about 400 words about a lighthouse keeper who finds "
             "a message in a bottle written in her own handwriting.",
}


def gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,clocks.sm,clocks.mem,"
             "power.draw,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return out
    except Exception as e:  # nvidia-smi missing is not fatal for the bench
        return [f"unavailable: {e}"]


def run_one(url, model, prompt, max_tokens, temperature, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": 1 if temperature == 0 else 20,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_send = time.perf_counter()
    t_first = t_last = None
    chunks, text, usage, timings = 0, [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunks += 1
                    text.append(piece)
    t_end = time.perf_counter()
    n_tok = (usage or {}).get("completion_tokens")
    if n_tok is None and timings:
        n_tok = timings.get("predicted_n")
    decode_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else None
    rec = {
        "completion_tokens": n_tok,
        "stream_chunks": chunks,
        "ttft_ms": round((t_first - t_send) * 1000, 1) if t_first else None,
        "wall_s": round(t_end - t_send, 3),
        # first token arrives with prefill; count only the tokens after it
        "client_decode_tps": round((n_tok - 1) / decode_s, 2) if (n_tok and decode_s) else None,
        "text": "".join(text),
    }
    if timings:
        rec["server_decode_tps"] = round(timings.get("predicted_per_second") or 0, 2)
        dn, da = timings.get("draft_n"), timings.get("draft_n_accepted")
        if dn:
            rec["draft_n"], rec["draft_accepted"] = dn, da
            rec["draft_acceptance"] = round(da / dn, 4)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8191")
    ap.add_argument("--model", default="local")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--categories", default=",".join(PROMPTS))
    ap.add_argument("--compare", help="earlier run JSON whose greedy texts must match")
    ap.add_argument("--notes", default="")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    cats = [c for c in args.categories.split(",") if c]
    # warmup: first request pays graph capture / allocator growth
    run_one(args.url, args.model, "Say hello.", 16, 0.0, args.timeout)

    results = {}
    for cat in cats:
        runs = [run_one(args.url, args.model, PROMPTS[cat], args.max_tokens,
                        args.temperature, args.timeout) for _ in range(args.repeats)]
        tps = [r["client_decode_tps"] for r in runs if r["client_decode_tps"]]
        results[cat] = {
            "client_decode_tps_mean": round(sum(tps) / len(tps), 2) if tps else None,
            "runs": runs,
        }
        acc = runs[-1].get("draft_acceptance")
        print(f"{args.label:>10} {cat:<12} decode {results[cat]['client_decode_tps_mean']} tok/s"
              f"  ttft {runs[-1]['ttft_ms']} ms  tokens {runs[-1]['completion_tokens']}"
              + (f"  accept {acc}" if acc is not None else ""), flush=True)

    all_tps = [results[c]["client_decode_tps_mean"] for c in cats if results[c]["client_decode_tps_mean"]]
    proof = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip() or
                     datetime.datetime.now().isoformat(timespec="minutes"),
        "label": args.label,
        "url": args.url,
        "notes": args.notes,
        "settings": {"max_tokens": args.max_tokens, "temperature": args.temperature,
                     "repeats": args.repeats, "thinking": False, "concurrency": 1},
        "gpus_after": gpu_snapshot(),
        "mean_decode_tps_across_categories": round(sum(all_tps) / len(all_tps), 2) if all_tps else None,
        "categories": results,
    }

    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            ref = json.load(f)
        same = {}
        for cat in cats:
            a = ref["categories"].get(cat, {}).get("runs", [{}])[0].get("text")
            b = results[cat]["runs"][0]["text"]
            same[cat] = (a == b)
        proof["greedy_match_vs"] = {"file": args.compare, "per_category": same,
                                    "all_match": all(same.values())}
        print(f"greedy output identical to {args.compare}: {same}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(proof, f, ensure_ascii=False, indent=2)
    print(f"mean decode across categories: {proof['mean_decode_tps_across_categories']} tok/s -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
