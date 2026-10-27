# FAIE2 — Roadmap

Verified state (Aisha, 2026-10-27): oracle gate 96/96 vs llama.cpp (independently
reproduced), lossless speculative decoding (~300 tok/s class), tool suite 8/8,
OpenAI-style server in `serve/`. See VERDICT.md for the evidence.

## Next builds, in order
1. **Flash-Next parity** — the sibling engine (faie2-flashnext) gets its server shim
   first (order filed 2026-10-27), then batched prefill and tool-call parsing, until
   both engines expose the identical `/v1` face.
2. **One runtime, two backends** — merge the two servers into a single runtime that
   loads either engine behind the same API.
3. **FAIE3: the universal step** — config read from GGUF metadata at load (no hardcoded
   dimensions), architectures as backends keyed by the GGUF `general.architecture`
   string (`qwen3` here, `qwen4exp` in Flash-Next), a tensor-name loader that undoes
   converter conventions. A new model family = one layer block + one tensor mapping +
   one oracle gate. First proof target: serve soup_90_10 (ABUZ8's own Qwen3-arch
   brain, Q5_K_M — a format already inside this engine's quant gate) through the
   same face.
