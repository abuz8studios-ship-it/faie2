"""Token ids -> text for Qwen3.8 (GPT-2 byte-level BPE vocabulary from the GGUF), for reading outputs."""

import functools

from faie2 import gguf_reader

MODEL = r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf"


def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


@functools.lru_cache(maxsize=1)
def vocab():
    r = gguf_reader(MODEL)
    f = r.fields["tokenizer.ggml.tokens"]
    return [bytes(f.parts[i]).decode("utf-8") for i in f.data]


def decode(ids):
    v, bd = vocab(), _byte_decoder()
    out = bytearray()
    for i in ids:
        tok = v[i]
        if tok.startswith("<|") or tok in ("<think>", "</think>"):
            out += tok.encode()
        else:
            out += bytes(bd.get(ch, ord("?")) for ch in tok)
    return out.decode("utf-8", errors="replace")
