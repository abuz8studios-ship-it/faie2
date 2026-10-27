"""Gate: FAIE2's DFlash2 drafter vs an independent float64 numpy reference.

Reference math is written from llama.cpp src/models/dflash.cpp (encoder, KV injection, noise block with
the dynamic 2-tap conv, non-causal sliding-window attention, target LM head, selector lattice, greedy walk).
Weights come from gguf-py dequantization; no FAIE2 code. Input: the fp16 target-layer features FAIE2
captured during prefill, so only the drafter is under test.
"""

import json
import sys

import numpy as np

from faie2 import Faie2, gguf_reader

sys.stdout.reconfigure(encoding="utf-8")
MODEL = r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf"
DRAFT = r"E:\ABU\MODELS\llm\qwen3.8-27b-dflash\dflash-Qwen3.8-27B-Q8_0.gguf"
ORACLE = r"E:\ABU\FAIE2\oracle\oracle_q6k.json"
EPS, BASE, MASK, SWA = 1e-6, 1e7, 248070, 2048
D, NH, NKV, HD, NG = 5120, 32, 8, 128, 320


class Weights:
    def __init__(self):
        self.d = gguf_reader(DRAFT)
        self.t = gguf_reader(MODEL)
        from gguf import quants
        self.q = quants
        self.dt = {x.name: x for x in self.d.tensors}
        self.tt = {x.name: x for x in self.t.tensors}

    def W(self, name, target=False):
        t = (self.tt if target else self.dt)[name]
        a = self.q.dequantize(np.asarray(t.data), t.tensor_type)
        return np.ascontiguousarray(a, dtype=np.float64)

    def rows(self, name, ids, target=False):
        t = (self.tt if target else self.dt)[name]
        return self.q.dequantize(np.asarray(t.data)[ids], t.tensor_type).astype(np.float64)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def rope(x, pos):   # neox over all 128 dims; x [..., n, heads, 128], pos [n]
    p = np.arange(64, dtype=np.float64)
    th = pos[:, None].astype(np.float64) * BASE ** (-2.0 * p / HD)
    c, s = np.cos(th)[:, None, :], np.sin(th)[:, None, :]
    x0, x1 = x[..., :64], x[..., 64:]
    return np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], -1)


def conv(h, dyn, base, side):   # h [8, D], dyn [8, 1280], base [2, 2, D] flattened as e + D*tap + 2D*side
    g = np.arange(D) // 16
    w0 = dyn[:, g + 0 * NG + 2 * NG * side] + base[0 * D + 2 * D * side: 1 * D + 2 * D * side]
    w1 = dyn[:, g + 1 * NG + 2 * NG * side] + base[1 * D + 2 * D * side: 2 * D + 2 * D * side]
    prev = np.vstack([np.zeros((1, D)), h[:-1]])
    return w0 * h + w1 * prev


def silu(x):
    return x / (1 + np.exp(-x))


def reference(w, feats, anchor):
    P = feats.shape[0]
    g = rms(feats @ w.W("fc.weight").T, w.W("enc.output_norm.weight"))
    x = w.rows("token_embd.weight", [anchor] + [MASK] * 7, target=True)
    pos_ctx, pos_blk = np.arange(P), np.arange(P, P + 8)
    for l in range(5):
        p = f"blk.{l}."
        wk, wv = w.W(p + "attn_k.weight"), w.W(p + "attn_v.weight")
        kn, qn = w.W(p + "attn_k_norm.weight"), w.W(p + "attn_q_norm.weight")
        Kc = rope(rms((g @ wk.T).reshape(P, NKV, HD), kn), pos_ctx)
        Vc = (g @ wv.T).reshape(P, NKV, HD)
        ab = w.W(p + "attn_conv_base").reshape(-1)
        fb = w.W(p + "ffn_conv_base").reshape(-1)
        h = rms(x, w.W(p + "attn_norm.weight"))
        dyn = h @ w.W(p + "attn_conv_proj.weight").T
        hc = conv(h, dyn, ab, 0)
        q = rope(rms((hc @ w.W(p + "attn_q.weight").T).reshape(8, NH, HD), qn), pos_blk)
        kb = rope(rms((hc @ wk.T).reshape(8, NKV, HD), kn), pos_blk)
        vb = (hc @ wv.T).reshape(8, NKV, HD)
        K, V = np.concatenate([Kc, kb]), np.concatenate([Vc, vb])     # positions 0..P+7
        att = np.empty((8, NH, HD))
        for t in range(8):
            lo = max(0, P + t - (SWA - 1))
            for hh in range(NH):
                kv = hh // (NH // NKV)
                sc = K[lo:, kv] @ q[t, hh] / np.sqrt(HD)
                pr = np.exp(sc - sc.max())
                att[t, hh] = (pr / pr.sum()) @ V[lo:, kv]
        o = att.reshape(8, NH * HD) @ w.W(p + "attn_output.weight").T
        x = conv(o, dyn, ab, 1) + x
        h = rms(x, w.W(p + "ffn_norm.weight"))
        dyn = h @ w.W(p + "ffn_conv_proj.weight").T
        hc = conv(h, dyn, fb, 0)
        f = (silu(hc @ w.W(p + "ffn_gate.weight").T) * (hc @ w.W(p + "ffn_up.weight").T)) @ w.W(p + "ffn_down.weight").T
        x = conv(f, dyn, fb, 1) + x
    hn = rms(x, w.W("output_norm.weight"))
    out = w.tt["output.weight"]
    raw = np.asarray(out.data)
    logits = np.concatenate([hn @ w.q.dequantize(raw[r:r + 16384], out.tensor_type).astype(np.float64).T
                             for r in range(0, raw.shape[0], 16384)], 1)
    hid = hn @ w.W("selector_hidden.weight").T
    cand = np.argsort(-logits, axis=1, kind="stable")[:, :16]
    unary = np.take_along_axis(logits, cand, 1)
    prev_ids = sorted(set([anchor] + cand[1:7].reshape(-1).tolist()))
    nxt_ids = sorted(set(cand[1:8].reshape(-1).tolist()))
    prev = dict(zip(prev_ids, w.rows("selector_predecessor.weight", prev_ids)))
    nxt = dict(zip(nxt_ids, w.rows("selector_successor.weight", nxt_ids)))
    score = np.zeros((8, 16, 16))
    for pp in range(1, 8):
        for i in range(16):
            pred = anchor if pp == 1 else cand[pp - 1][i]
            cond = prev[pred] * hid[pp]
            score[pp, i] = [nxt[cand[pp][j]] @ cond + unary[pp][j] for j in range(16)]
    block, pred = [anchor], 0
    for pp in range(1, 8):
        best = int(np.argmax(score[pp, pred]))      # first maximum, like std::max_element
        block.append(int(cand[pp][best]))
        pred = best
    return block, cand, score


def main():
    oracle = {c["name"]: c for c in json.load(open(ORACLE, encoding="utf-8"))["cases"]}
    eng = Faie2()
    eng.load(MODEL, max_ctx=4096)
    eng.load_drafter(DRAFT)
    w = Weights()
    ok = True
    for name in ("code", "prose", "arabic"):
        prompt = oracle[name]["prompt_tokens"]
        eng.reset()
        pre, feats = prompt[:-1], []
        for i in range(0, len(pre), 8):
            chunk = pre[i:i + 8]
            eng.forward(chunk, want_logits=False)
            feats.append(eng.df_features()[:len(chunk)].astype(np.float64))
        feats = np.concatenate(feats)
        eng.spec_begin(prompt[-1])
        fb, fc, fs = eng.df_draft()
        rb, rc, rs = reference(w, feats, prompt[-1])
        same_block = fb == rb
        overlap = [len(set(fc[p]) & set(rc[p])) for p in range(1, 8)]
        top1 = [int(fc[p][0]) == int(rc[p][0]) for p in range(1, 8)]
        sdiff = max(float(np.abs(fs[p][i][j] - rs[p][i][j])) for p in range(1, 8) for i in range(16) for j in range(16)
                    if (p == 1 or fc[p - 1][i] == rc[p - 1][i]) and fc[p][j] == rc[p][j])
        passed = same_block and min(overlap) >= 15 and all(top1)
        ok &= passed
        print(f"{name:<7} P={len(pre):3d}  drafts FAIE2 {fb[1:]}  ref {rb[1:]}  identical={same_block}  "
              f"top16 overlap {overlap}  top1 same {sum(top1)}/7  max|score diff| {sdiff:.4f}  {'PASS' if passed else 'FAIL'}",
              flush=True)
    print("DRAFTER GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
