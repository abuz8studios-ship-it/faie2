# FAIE2

A from-scratch CUDA inference engine for **Qwen3.8-27B** (Q6_K GGUF), built for a dual RTX 5090 workstation on Windows — with its own OpenAI-style server, a tool-calling suite, and **lossless speculative decoding** (an in-engine DFlash2 drafter). No llama.cpp runtime and no PyTorch inference path in the engine itself; llama.cpp serves as the correctness oracle it is gated against.

The Flash-Next variant of this engine (DeltaNet-hybrid MoE, ~173 tok/s, bit-exact gates) lives in the sibling [`faie2-flashnext`](../faie2-flashnext) repository. The predecessor GLM engine is in [`faie`](../faie).

**Status: active work in progress.** Dated snapshot (2026-10-27), published under ABUZ8's post-everything policy — results with their evidence, unfinished parts stated as unfinished.

## Measured results (author's runs, 2026-10-26; result files in `tests/`)

| Test | Result |
|---|---|
| Oracle gate vs llama.cpp (`tests/gate_oracle.py`, `gate_oracle_result.json`) | **96/96 top-1 positions match, 0 real mismatches, PASS** |
| Speculative decoding, end-to-end (`spec_e2e_result.json`) | code **296.3 tok/s**, structured **301.0 tok/s**, translation 254.7 tok/s — vs ~53 tok/s plain decode; **lossless vs plain decode**, token-identical to llama.cpp greedy for the first 512 tokens |
| Spec bench (`bench_spec_result.json`) | up to **359.0 tok/s** (structured); 420.5 tok/s with a perfect drafter |
| Lossless suite (`lossless_result.json`) | speculative output lossless under perfect, corrupt, random and mixed drafters, CUDA graphs on |
| Load test (`load_result_faie2.json`) | 2,048-token generations at 187.7 (code), 306.1 (math), 202.1 (logic) tok/s |
| Tool suite (`serve/test_tools.py`, `compare_tools.py`) | **8/8** tool calls match the reference behavior |

An independent verification note for this snapshot is in `VERDICT.md`.

## Layout

- `engine/` — `faie2.cu` (the engine) and `build.ps1` (CUDA 13.1 + MSVC, sm_120; builds to a temp DLL and swaps it in, so a failed build never removes a working one).
- `serve/` — `faie2_server.py` (server, GPU0, default port 8300), tokenizer wrapper, tool definitions, tool tests, `serve_faie2.ps1` launcher.
- `tests/` — oracle gates, load/spec/lossless test suites and their captured result JSONs, prefill and batch-invariance tests.
- `oracle/` — the llama.cpp oracle capture (`oracle_q6k.json`) and the comparison bench write-up.
- `bench/` — early bench harness.
- `DESIGN.md` — the family's complete engineering record, including the Flash-Next chapters (§9).

## Not included

Model weights (the engine loads a local Q6_K GGUF) and the compiled `faie2.dll` — build from source.

## License

MIT — see `LICENSE`. Copyright (c) 2026 Ahmad A Odeh / ABUZ8 LLC.
