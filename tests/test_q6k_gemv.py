"""Gate: FAIE2's Q6_K batched GEMV vs llama.cpp gguf-py dequantization (numpy, float64 matmul)."""

import sys

import numpy as np

from faie2 import Faie2, gguf_reader

sys.stdout.reconfigure(encoding="utf-8")
MODEL = r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf"
TENSORS = ["blk.0.ssm_out.weight", "blk.3.attn_q.weight", "blk.0.ffn_down.weight", "blk.0.ssm_beta.weight"]


def main():
    r = gguf_reader(MODEL)  # puts llama.cpp's gguf-py on sys.path
    from gguf import quants
    by_name = {t.name: t for t in r.tensors}
    eng = Faie2()
    rng = np.random.default_rng(0)
    ok = True
    for name in TENSORS:
        t = by_name[name]
        cols, rows = int(t.shape[0]), int(t.shape[1])
        raw = np.asarray(t.data)  # [rows, row_bytes] uint8
        W = quants.dequantize(raw, t.tensor_type).astype(np.float64).reshape(rows, cols)
        x8 = rng.standard_normal((8, cols)).astype(np.float32)
        for nt in (1, 2, 8):
            x = x8[:nt]
            ref = x.astype(np.float64) @ W.T
            for version in (1, 2, 3, 4, 5, 6):
                y, ms = eng.test_q6k_gemv(raw, rows, cols, x, version=version)
                rel = np.abs(y - ref).max() / np.abs(ref).max()
                cos = float((y * ref).sum() / np.sqrt((y * y).sum() * (ref * ref).sum()))
                gbps = raw.nbytes / (ms * 1e-3) / 1e9
                # v3/v4 round activations to fp16 (2^-11 relative), v1/v2 keep fp32
                passed = (rel < 3e-3 and cos > 0.99999) if version >= 3 else (rel < 1e-4 and cos > 0.999999)
                extra = ""
                if version >= 4:
                    # v4/v5 must be deterministic and give token 0 the same bits whatever nt is
                    y2, _ = eng.test_q6k_gemv(raw, rows, cols, x, version=version)
                    y8, _ = eng.test_q6k_gemv(raw, rows, cols, x8, version=version)
                    same = np.array_equal(y, y2) and np.array_equal(y[0], y8[0])
                    passed &= same
                    extra = f"  repeat+batch bit-identical={same}"
                ok &= passed
                print(f"v{version} {name:<24} {rows:>6}x{cols:<6} nt={nt}  max_rel_err={rel:.2e}  cos={cos:.8f}  "
                      f"{ms*1000:8.1f} us  {gbps:7.1f} GB/s  {'PASS' if passed else 'FAIL'}{extra}", flush=True)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
