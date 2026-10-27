# FAIE2 (Qwen3.8-27B) — Independent Verification Verdict

**Tester:** Aisha (Muse, ABUZ8 fleet) — not the author. The engine was built by Claude
(Claude Desktop sessions on Pegasus) with Ahmad Odeh.
**Date:** 2026-10-27, ~01:10 EDT
**Machine:** Pegasus — 2× RTX 5090 32 GB, Ryzen 9 9950X, 128 GB DDR5, Windows 11.

## Artifact under test

- `engine/faie2.cu` → `engine/faie2.dll` (6,826,496 bytes, built 2026-10-26 06:05)
- Model: Qwen3.8-27B-ABLITERATED Q6_K GGUF (local disk)
- Oracle: llama.cpp captured log-probs, `oracle/oracle_q6k.json`

## Test run by the verifier (fresh execution)

**ORACLE GATE** — `tests/gate_oracle.py` (Python 3.13, Pegasus), comparing the engine's
8-token verify path against the llama.cpp oracle across six cases
(prose, code, json, arabic, math, long), 96 positions:

- loaded in 11.8 s
- **top-1 match 96/96, real mismatches 0, tie flips 0**
- mean top-5 overlap 4.927; max |Δlogprob| on top-1 = 0.2829
- **GATE PASS: true**

The verifier's numbers reproduce the author's captured `tests/gate_oracle_result.json`
exactly (96/96, 0 mismatches, 4.927, 0.2829). Raw log: `verification/aisha_oracle_gate_27b.log`.

## Author's captured results (read from the tree, not re-run by the verifier)

- Speculative decoding end-to-end: 254.7–301.0 tok/s by case vs ~53 tok/s plain,
  lossless vs plain decode (`tests/spec_e2e_result.json`).
- Lossless suite: lossless under perfect/corrupt/random/mixed drafters (`tests/lossless_result.json`).
- Load test: 187.7–306.1 tok/s on 2,048-token generations (`tests/load_result_faie2.json`).
- Tool suite: 8/8 (`serve/test_tools.py`).

## Verdict

**VERIFIED (gate level).** The FAIE2 engine's forward path matches the llama.cpp oracle
at every tested position. The speculative-decoding and load numbers above are the
author's captured artifacts — consistent, in-tree, and reproducible with the suites in
`tests/`; the verifier re-ran the correctness gate, not the full load suite.

The Flash-Next variant was verified separately (bench + two gates) in the
`faie2-flashnext` repository.
