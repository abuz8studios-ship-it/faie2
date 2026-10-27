"""Tool-calling gate over any OpenAI-compatible endpoint (FAIE2 or llama-server), thinking on, greedy.

  python test_tools.py --url http://127.0.0.1:8300 --label faie2

Each case checks what an agent needs: the right function(s), valid JSON arguments with the required keys and
types, a plain answer when no tool fits, and a correct final answer after a tool result comes back.
Writes tools_<label>.json with per-case results, timings and the raw messages.
"""

import argparse
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")


def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


WEATHER = fn("get_weather", "Current weather for a city.", {"city": {"type": "string"},
             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"])
CALC = fn("calculator", "Evaluate an arithmetic expression exactly.", {"expression": {"type": "string"}}, ["expression"])
SEARCH = fn("web_search", "Search the web.", {"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"])
EVENT = fn("create_event", "Create a calendar event.", {
    "title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601 datetime"},
    "duration_minutes": {"type": "integer"}, "attendees": {"type": "array", "items": {"type": "string"}}},
    ["title", "start", "duration_minutes", "attendees"])
LS = fn("list_dir", "List files in a directory.", {"path": {"type": "string"}}, ["path"])
READ = fn("read_file", "Read a text file.", {"path": {"type": "string"}}, ["path"])

CASES = [
    ("weather", [WEATHER, CALC, SEARCH], [{"role": "user", "content": "What's the weather in Amman right now, in celsius?"}],
     {"calls": [("get_weather", {"city": "Amman"})]}),
    ("calc", [WEATHER, CALC, SEARCH], [{"role": "user", "content": "Use a tool to compute 48213 * 977 exactly."}],
     {"calls": [("calculator", {})]}),
    ("event_types", [EVENT, WEATHER], [{"role": "user", "content":
        "Book a 45 minute meeting titled 'FAIE2 review' on 2026-10-28 at 14:00 with ahmad@abuz8.com and lina@abuz8.com."}],
     {"calls": [("create_event", {"title": "FAIE2 review", "duration_minutes": 45})], "list_arg": ("attendees", 2)}),
    ("parallel", [WEATHER, CALC], [{"role": "user", "content": "Get the weather in Amman and in Dubai."}],
     {"calls": [("get_weather", {"city": "Amman"}), ("get_weather", {"city": "Dubai"})]}),
    ("no_tool", [WEATHER, CALC], [{"role": "user", "content": "Reply with exactly the word: ready"}],
     {"calls": [], "content_has": "ready"}),
    ("arabic", [WEATHER, SEARCH], [{"role": "user", "content": "ما هو الطقس الآن في عمّان؟"}],
     {"calls": [("get_weather", {})]}),
    ("tool_result", [WEATHER], [
        {"role": "user", "content": "What's the weather in Amman?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": json.dumps({"city": "Amman"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"city": "Amman", "temp_c": 23, "sky": "clear"})}],
     {"calls": [], "content_has": "23"}),
    ("agent_step", [LS, READ], [
        {"role": "system", "content": "You are a coding agent. Use the tools to inspect the project before answering."},
        {"role": "user", "content": "What Python files are in E:/proj/src?"}],
     {"calls": [("list_dir", {})]}),
]


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check(msg, finish, exp):
    problems = []
    calls = msg.get("tool_calls") or []
    got = []
    for c in calls:
        try:
            args = json.loads(c["function"]["arguments"])
        except (ValueError, KeyError):
            problems.append(f"arguments not JSON: {c}")
            args = {}
        got.append((c["function"]["name"], args))
    want = exp["calls"]
    if len(got) != len(want):
        problems.append(f"expected {len(want)} call(s), got {len(got)}: {[g[0] for g in got]}")
    for (wn, wa), (gn, ga) in zip(want, got):
        if wn != gn:
            problems.append(f"expected {wn}, got {gn}")
        for k, v in wa.items():
            if str(ga.get(k, "")).strip().lower() != str(v).strip().lower():
                problems.append(f"{gn}.{k} = {ga.get(k)!r}, expected {v!r}")
    if want and finish != "tool_calls":
        problems.append(f"finish_reason {finish!r}, expected 'tool_calls'")
    if "list_arg" in exp and got:
        k, n = exp["list_arg"]
        v = got[0][1].get(k)
        if not isinstance(v, list) or len(v) != n:
            problems.append(f"{k} should be a list of {n}, got {v!r}")
        if not isinstance(got[0][1].get("duration_minutes"), int):
            problems.append(f"duration_minutes should be an integer, got {got[0][1].get('duration_minutes')!r}")
    if "content_has" in exp and exp["content_has"].lower() not in (msg.get("content") or "").lower():
        problems.append(f"content lacks {exp['content_has']!r}: {(msg.get('content') or '')[:120]!r}")
    return problems, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8300")
    ap.add_argument("--label", default="faie2")
    ap.add_argument("--effort", default=None, help="reasoning_effort (xhigh/medium/low); default = template default")
    a = ap.parse_args()
    results, passed = [], 0
    for name, tools, messages, exp in CASES:
        body = {"model": "x", "messages": messages, "tools": tools, "temperature": 0, "max_tokens": 6000}
        if a.effort:
            body["chat_template_kwargs"] = {"reasoning_effort": a.effort}
        t0 = time.perf_counter()
        r = post(a.url + "/v1/chat/completions", body)
        dt = time.perf_counter() - t0
        ch = r["choices"][0]
        problems, got = check(ch["message"], ch.get("finish_reason"), exp)
        ok = not problems
        passed += ok
        tm = r.get("timings") or {}
        results.append({"case": name, "pass": ok, "problems": problems, "calls": got, "seconds": round(dt, 2),
                        "finish": ch.get("finish_reason"), "usage": r.get("usage"), "timings": tm,
                        "content": ch["message"].get("content"), "reasoning_chars": len(ch["message"].get("reasoning_content") or "")})
        print(f"{name:<12} {'PASS' if ok else 'FAIL'}  {dt:6.1f} s  calls={[(n, a_) for n, a_ in got]}  "
              f"{'' if ok else problems}", flush=True)
    print(f"{a.label}: {passed}/{len(CASES)} passed")
    json.dump({"label": a.label, "url": a.url, "passed": passed, "total": len(CASES), "results": results},
              open(f"tools_{a.label}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
