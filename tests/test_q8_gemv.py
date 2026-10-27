"""Gate: FAIE2's Q8_0 tensor-core GEMV (drafter weights) vs gguf-py dequantization, float64 matmul."""

import ctypes
import sys

import numpy as np

from faie2 import Faie2, gguf_reader

sys.stdout.reconfigure(encoding="utf-8")
DRAFT = r"E:\ABU\MODELS\llm\qwen3.8-27b-dflash\dflash-Qwen3.8-27B-Q8_0.gguf"
TENSORS = ["fc.weight", "blk.0.attn_q.weight", "blk.0.attn_output.weight", "blk.0.ffn_down.weight",
           "blk.0.attn_conv_proj.weight", "selector_hidden.weight"]


def main():
    r = gguf_reader(DRAFT)
    from gguf import quants
    by = {t.name: t for t in r.tensors}
    eng = Faie2()
    eng.lib.faie2_test_q8_gemv_v.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    def run(raw, rows, cols, x, version):
        y = np.empty((x.shape[0], rows), dtype=np.float32)
        ms = ctypes.c_float()
        eng._check(eng.lib.faie2_test_q8_gemv_v(raw.ctypes.data, rows, cols, x.ctypes.data, x.shape[0], y.ctypes.data,
                                                ctypes.byref(ms), version))
        return y, ms.value

    rng = np.random.default_rng(0)
    ok = True
    for name in TENSORS:
        t = by[name]
        cols, rows = int(t.shape[0]), int(t.shape[1])
        raw = np.ascontiguousarray(np.asarray(t.data))
        W = quants.dequantize(raw, t.tensor_type).astype(np.float64).reshape(rows, cols)
        x8 = rng.standard_normal((8, cols)).astype(np.float32)
        for nt in (1, 8):
            x = np.ascontiguousarray(x8[:nt])
            ref = x.astype(np.float64) @ W.T
            for version in (3, 4, 5, 6):
                y, ms = run(raw, rows, cols, x, version)
                rel = np.abs(y - ref).max() / np.abs(ref).max()
                cos = float((y * ref).sum() / np.sqrt((y * y).sum() * (ref * ref).sum()))
                passed = rel < 3e-3 and cos > 0.99999    # fp16 activations, as in the Q6_K tensor-core path
                extra = ""
                if version >= 4:
                    y2, _ = run(raw, rows, cols, x, version)
                    y8, _ = run(raw, rows, cols, x8, version)
                    same = np.array_equal(y, y2) and np.array_equal(y[0], y8[0])
                    passed &= same
                    extra = f"  repeat+batch bit-identical={same}"
                ok &= passed
                print(f"v{version} {name:<28} {rows:>6}x{cols:<6} nt={nt}  max_rel_err={rel:.2e}  cos={cos:.8f}  "
                      f"{ms*1000:7.1f} us  {raw.nbytes / (ms * 1e-3) / 1e9:7.1f} GB/s  {'PASS' if passed else 'FAIL'}{extra}",
                      flush=True)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
