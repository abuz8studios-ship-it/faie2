"""Independent float64 reference forward for Qwen3.8-27B (qwen35) from the Q6_K GGUF.

No FAIE2 code: weights come from llama.cpp's gguf-py dequantizer, math is torch float64 written from
llama.cpp's graph (qwen35.cpp / delta-net-base.cpp). Used to judge which engine is closer to exact math
where FAIE2 and llama.cpp disagree.

Usage: python ref_forward.py json code   -> writes ref_<case>.npy (log-probs of the 16 generated positions)
"""

import json
import math
import sys
import time

import numpy as np
import torch

from faie2 import gguf_reader

sys.stdout.reconfigure(encoding="utf-8")
MODEL = r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf"
ORACLE = r"E:\ABU\FAIE2\oracle\oracle_q6k.json"
EPS = 1e-6
dev = torch.device("cuda")
f64 = torch.float64

r = gguf_reader(MODEL)
from gguf import quants  # noqa: E402  (gguf-py is on sys.path after gguf_reader)

T = {t.name: t for t in r.tensors}


def W(name):
    t = T[name]
    a = quants.dequantize(np.asarray(t.data), t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)).to(dev)


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def silu(x):
    return x * torch.sigmoid(x)


def forward(tokens, n_out):
    emb = T["token_embd.weight"]
    rows = np.asarray(emb.data)[tokens]
    x = torch.from_numpy(quants.dequantize(rows, emb.tensor_type).astype(np.float64)).to(dev)  # (n, 5120)
    n = x.shape[0]
    pos = torch.arange(n, device=dev, dtype=f64)
    for il in range(64):
        p = f"blk.{il}."
        xn = rms(x, W(p + "attn_norm.weight"))
        if (il + 1) % 4 != 0:
            qkv = xn @ W(p + "attn_qkv.weight").T
            z = xn @ W(p + "attn_gate.weight").T
            b = xn @ W(p + "ssm_beta.weight").T
            a = xn @ W(p + "ssm_alpha.weight").T
            cw = W(p + "ssm_conv1d.weight").reshape(10240, 4)
            inp = torch.cat([torch.zeros(3, 10240, device=dev, dtype=f64), qkv], 0)
            co = sum(inp[j:j + n] * cw[:, j] for j in range(4))
            co = silu(co)
            q = co[:, :2048].reshape(n, 16, 128)
            k = co[:, 2048:4096].reshape(n, 16, 128)
            v = co[:, 4096:].reshape(n, 48, 128)
            q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=EPS)
            k = k / torch.clamp(k.norm(dim=-1, keepdim=True), min=EPS)
            beta = torch.sigmoid(b)
            g = torch.nn.functional.softplus(a + W(p + "ssm_dt.bias")) * W(p + "ssm_a")
            hk = torch.arange(48, device=dev) % 16          # value head h reads key head h % 16
            S = torch.zeros(48, 128, 128, device=dev, dtype=f64)  # [h][i(k)][j(v)]
            o = torch.empty(n, 48, 128, device=dev, dtype=f64)
            for t in range(n):
                kt, qt = k[t][hk], q[t][hk]
                S = torch.exp(g[t])[:, None, None] * S
                kv = torch.einsum("hij,hi->hj", S, kt)
                delta = (v[t] - kv) * beta[t][:, None]
                S = S + torch.einsum("hi,hj->hij", kt, delta)
                o[t] = torch.einsum("hij,hi->hj", S, qt) / math.sqrt(128)
            on = rms(o, W(p + "ssm_norm.weight")) * silu(z.reshape(n, 48, 128))
            y = on.reshape(n, 6144) @ W(p + "ssm_out.weight").T
        else:
            qf = (xn @ W(p + "attn_q.weight").T).reshape(n, 24, 512)
            q, gate = qf[..., :256], qf[..., 256:]
            q = rms(q, W(p + "attn_q_norm.weight"))
            k = rms((xn @ W(p + "attn_k.weight").T).reshape(n, 4, 256), W(p + "attn_k_norm.weight"))
            v = (xn @ W(p + "attn_v.weight").T).reshape(n, 4, 256)
            pidx = torch.arange(32, device=dev, dtype=f64)
            theta = pos[:, None] * torch.pow(torch.tensor(1e7, dtype=f64, device=dev), -2.0 * pidx / 64)  # (n, 32)
            c, s = torch.cos(theta)[:, None, :], torch.sin(theta)[:, None, :]

            def rope(xh):
                x0, x1 = xh[..., :32], xh[..., 32:64]
                return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c, xh[..., 64:]], -1)

            q, k = rope(q), rope(k)
            kk = k.repeat_interleave(6, dim=1)   # q head h uses kv head h // 6
            vv = v.repeat_interleave(6, dim=1)
            sc = torch.einsum("thd,shd->hts", q, kk) / 16.0
            mask = torch.triu(torch.ones(n, n, device=dev, dtype=torch.bool), 1)
            sc = sc.masked_fill(mask, float("-inf"))
            att = torch.einsum("hts,shd->thd", torch.softmax(sc, -1), vv)
            att = att * torch.sigmoid(gate)
            y = att.reshape(n, 6144) @ W(p + "attn_output.weight").T
        x = x + y
        xn = rms(x, W(p + "post_attention_norm.weight"))
        y = (silu(xn @ W(p + "ffn_gate.weight").T) * (xn @ W(p + "ffn_up.weight").T)) @ W(p + "ffn_down.weight").T
        x = x + y
        if il % 8 == 7:
            print(f"  layer {il + 1}/64", flush=True)
    h = rms(x[-n_out:], W("output_norm.weight"))
    out = T["output.weight"]
    raw = np.asarray(out.data)
    logits = []
    for r0 in range(0, raw.shape[0], 16384):
        wch = torch.from_numpy(quants.dequantize(raw[r0:r0 + 16384], out.tensor_type).astype(np.float64)).to(dev)
        logits.append(h @ wch.T)
    lg = torch.cat(logits, -1)
    return torch.log_softmax(lg, -1).cpu().numpy()


if __name__ == "__main__":
    oracle = json.load(open(ORACLE, encoding="utf-8"))
    for case in oracle["cases"]:
        if case["name"] not in sys.argv[1:]:
            continue
        t0 = time.time()
        toks = case["prompt_tokens"] + case["generated"][:-1]
        lp = forward(toks, n_out=len(case["generated"]))
        np.save(rf"E:\ABU\FAIE2\tests\ref_{case['name']}.npy", lp.astype(np.float32))
        print(f"{case['name']}: {len(toks)} tokens in {time.time() - t0:.0f} s -> ref_{case['name']}.npy", flush=True)
