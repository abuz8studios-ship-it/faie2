"""Qwen3.8 tokenizer and chat template, built from the GGUF itself (no llama.cpp at runtime).

Tokenizer: GPT-2 byte-level BPE (tokenizer.ggml.tokens / .merges) with the "qwen35" pre-tokenizer split regex
(llama.cpp src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_QWEN35). Control and user-defined tokens (types 3 and 4)
are added as special tokens, so "<|im_start|>" etc. in text become single ids (llama-server parse_special=true).
The built tokenizer is cached as tokenizer.json next to this file.

Template: tokenizer.chat_template rendered with jinja2 the way Hugging Face transformers does
(ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, tojson with ensure_ascii=False).
"""

import json
import os
import sys
import types

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF_PY = r"E:\ABU\MODELS\runtime\src\llama.cpp-dflash2\gguf-py"
QWEN35_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")


def _reader(path):
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))   # gguf-py's metadata module imports yaml, unused here
    if GGUF_PY not in sys.path:
        sys.path.insert(0, GGUF_PY)
    from gguf import GGUFReader
    return GGUFReader(path)


def _strings(field):
    return [bytes(field.parts[i]).decode("utf-8") for i in field.data]


class QwenTokenizer:
    def __init__(self, gguf_path, cache_dir=HERE):
        stem = os.path.splitext(os.path.basename(gguf_path))[0]
        tok_cache = os.path.join(cache_dir, f"{stem}.tokenizer.json")
        meta_cache = os.path.join(cache_dir, f"{stem}.meta.json")
        if os.path.exists(tok_cache) and os.path.exists(meta_cache):
            self.tk = Tokenizer.from_file(tok_cache)
            meta = json.load(open(meta_cache, encoding="utf-8"))
        else:
            r = _reader(gguf_path)
            f = r.fields
            if bytes(f["tokenizer.ggml.model"].parts[f["tokenizer.ggml.model"].data[0]]).decode() != "gpt2":
                raise ValueError("expected a gpt2 (byte-level BPE) vocabulary")
            pre = bytes(f["tokenizer.ggml.pre"].parts[f["tokenizer.ggml.pre"].data[0]]).decode()
            if pre != "qwen35":
                raise ValueError(f"pre-tokenizer {pre!r} is not qwen35")
            tokens = _strings(f["tokenizer.ggml.tokens"])
            merges = [tuple(m.split(" ", 1)) for m in _strings(f["tokenizer.ggml.merges"])]
            types_ = [int(f["tokenizer.ggml.token_type"].parts[i][0]) for i in f["tokenizer.ggml.token_type"].data]
            tk = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=merges, ignore_merges=False))
            tk.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(QWEN35_SPLIT), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
            tk.decoder = decoders.ByteLevel()
            specials = [tokens[i] for i, t in enumerate(types_) if t in (3, 4)]
            tk.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])
            def num(k):
                return int(f[k].parts[f[k].data[0]][0])
            meta = {"eos": num("tokenizer.ggml.eos_token_id"), "n_vocab": len(tokens),
                    "chat_template": bytes(f["tokenizer.chat_template"].parts[f["tokenizer.chat_template"].data[0]]).decode()}
            tk.save(tok_cache)
            json.dump(meta, open(meta_cache, "w", encoding="utf-8"), ensure_ascii=False)
            self.tk = tk
        self.eos = meta["eos"]
        self.n_vocab = meta["n_vocab"]
        self.template_src = meta["chat_template"]
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(self.template_src)
        self.im_end = self.tk.token_to_id("<|im_end|>")

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
