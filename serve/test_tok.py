"""Gate: FAIE2's own tokenizer + template vs llama.cpp (llama-server's /apply-template + /tokenize outputs saved
earlier, and llama-tokenize.exe on extra multilingual text). Every id must match."""

import json
import os
import subprocess
import sys

from faie2_tok import QwenTokenizer

sys.stdout.reconfigure(encoding="utf-8")
MODEL = r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf"
LLAMA_TOKENIZE = r"E:\ABU\MODELS\runtime\src\llama.cpp-dflash2\build\bin\llama-tokenize.exe"
tok = QwenTokenizer(MODEL)
ok = True

# 1) template + tokenizer reproduce llama-server's prompts exactly (thinking on and off)
sets = [(r"E:\ABU\FAIE2\tests\load_prompts.json", True), (r"E:\ABU\FAIE2\oracle\bench_prompts_q6k.json", False)]
sys.path.insert(0, r"E:\ABU\FAIE2\tests")
import importlib.util  # noqa: E402
lt = importlib.util.spec_from_file_location("lt", r"E:\ABU\FAIE2\tests\load_test.py")
load_test = importlib.util.module_from_spec(lt)
lt.loader.exec_module(load_test)
bs = importlib.util.spec_from_file_location("bench", r"E:\ABU\FAIE2\bench\bench.py")
bench = importlib.util.module_from_spec(bs)
bs.loader.exec_module(bench)
texts = {"load": load_test.PROMPTS, "bench": bench.PROMPTS}
for path, thinking in sets:
    data = json.load(open(path, encoding="utf-8"))
    src = texts["load" if thinking else "bench"]
    for name, case in data["cases"].items():
        ref = case["prompt_tokens"]
        text = tok.render([{"role": "user", "content": src[name]}], enable_thinking=thinking)
        ids = tok.encode(text)
        same = ids == ref
        ok &= same
        rt = tok.encode(tok.decode(ref)) == ref
        ok &= rt
        if not same:
            k = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), min(len(ids), len(ref)))
            print(f"  MISMATCH at {k}: ours {ids[k:k+5]} {tok.decode(ids[k:k+5])!r} ref {ref[k:k+5]} {tok.decode(ref[k:k+5])!r}")
        print(f"template+encode {name:<11} thinking={thinking!s:<5} {len(ids):4d} ids  identical={same}  roundtrip={rt}")
    # llama.cpp's generated outputs must also round-trip through decode/encode where they are canonical
    for name, case in data["cases"].items():
        g = case.get("llama_greedy", [])
        if g:
            print(f"  decode(llama output {name}) -> {len(tok.decode(g))} chars")

# 2) raw text vs llama-tokenize.exe (multilingual, code, numbers, whitespace, specials)
samples = ["Hello world!  It's 2026-10-26, and π≈3.14159.\n\n\tdef f(x): return x**2  # tabs\r\n",
           "اشرح بالتفصيل كيف تعمل الألواح الشمسية؟ ١٢٣٤ 中文测试，日本語のテキスト。 Ελληνικά καλημέρα",
           "<|im_start|>user\nCall <tool_call>\n<function=get_weather>\n<parameter=city>\nAmman\n</parameter>\n</function>\n</tool_call><|im_end|>",
           "    leading spaces   and   multiple    gaps\n\n\n\nnewlines 12345678901234567890 x' y's I'LL WE'RE",
           "emoji 😀🚀 combining é é Hindi हिन्दी Thai ภาษาไทย Korean 한국어 Russian Привет"]
if os.path.exists(LLAMA_TOKENIZE):
    for s in samples:
        out = subprocess.run([LLAMA_TOKENIZE, "-m", MODEL, "-p", s, "--ids", "--log-disable", "--no-bos"],
                             capture_output=True, text=True, encoding="utf-8")
        line = [l for l in out.stdout.splitlines() if l.startswith("[")]
        ref = json.loads(line[-1]) if line else None
        ids = tok.encode(s)
        same = ids == ref
        ok &= same
        print(f"llama-tokenize {len(ids):3d} ids identical={same}  {s[:40]!r}")
        if not same:
            print("   ours", ids[:40], "\n   ref ", (ref or [])[:40], "\n   stderr", out.stderr[-300:])
print("TOKENIZER GATE PASS" if ok else "TOKENIZER GATE FAIL")
sys.exit(0 if ok else 1)
