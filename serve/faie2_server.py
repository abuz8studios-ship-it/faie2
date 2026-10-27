"""FAIE2 server: OpenAI-compatible chat/completions over the FAIE2 engine, no llama.cpp anywhere.

  python faie2_server.py [--port 8300] [--ctx 16384] [--snapshots 4]

Endpoints
  POST /v1/chat/completions   messages, tools (Qwen <tool_call> XML <-> OpenAI tool_calls), stream (SSE),
                              reasoning split into reasoning_content, chat_template_kwargs / reasoning_effort
  POST /v1/completions        prompt as text or token ids
  POST /completion            llama-server style (prompt text or ids, n_predict, return_tokens), for existing tools
  POST /tokenize /detokenize /apply-template   (llama-server compatible)
  GET  /health /v1/models /props

Decoding is greedy (lossless DFlash2 speculation in the engine); requests that ask for temperature > 0 are served
greedily and the response says so in system_fingerprint. One engine, one request at a time (a lock queues others).

Prefix cache: attention K/V is positional and the engine can snapshot its recurrent state, so a new request reuses
the longest already-computed prefix: the live state when the new prompt extends it, else the best snapshot.
Snapshots are taken at the end of the system block, at the start of the generation prompt and at the prompt end.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
from faie2 import Faie2  # noqa: E402
from faie2_tok import QwenTokenizer  # noqa: E402

MODEL = r"E:\ABU\MODELS\llm\qwen3.8-27b-abliterated\Qwen3.8-27B-ABLITERATED-Q6_K.gguf"
DRAFT = r"E:\ABU\MODELS\llm\qwen3.8-27b-dflash\dflash-Qwen3.8-27B-Q8_0.gguf"
FINGERPRINT = "faie2-greedy-dflash2"


# ---------------------------------------------------------------------------------------------
# engine with prefix cache

class Engine:
    def __init__(self, model, draft, ctx, nsnap):
        import ctypes
        self.eng = Faie2()
        L = self.eng.lib
        for name, args in (("faie2_snapshot_alloc", [ctypes.c_int]), ("faie2_snapshot_save", [ctypes.c_int]),
                           ("faie2_snapshot_load", [ctypes.c_int])):
            getattr(L, name).argtypes = args
        t0 = time.time()
        self.eng.load(model, max_ctx=ctx)
        self.eng.load_drafter(draft)
        self.eng.use_graphs(True)
        self.eng._check(L.faie2_snapshot_alloc(nsnap))
        self.ctx = ctx
        self.nsnap = nsnap
        self.snaps = {}          # slot -> (tokens tuple, last_used)
        self.hist = []           # tokens committed in the engine, positions 0..n_past-1
        self.load_s = time.time() - t0
        self.lock = threading.Lock()

    def _save(self, toks):
        toks = tuple(toks)
        if not self.nsnap or any(v[0] == toks for v in self.snaps.values()):
            return
        free = [s for s in range(self.nsnap) if s not in self.snaps]
        slot = free[0] if free else min(self.snaps, key=lambda s: self.snaps[s][1])
        self.eng._check(self.eng.lib.faie2_snapshot_save(slot))
        self.snaps[slot] = (toks, time.time())

    def prepare(self, prompt, marks):
        """Bring the engine to state prompt[:-1]; returns (anchor token, tokens reused, tokens prefilled)."""
        target = prompt[:-1]
        best, src = -1, None
        if len(self.hist) <= len(target) and target[:len(self.hist)] == self.hist:
            best, src = len(self.hist), "live"
        for slot, (toks, _) in self.snaps.items():
            n = len(toks)
            if n > best and n <= len(target) and tuple(target[:n]) == toks:
                best, src = n, slot
        if src is None:
            self.eng.reset()
            self.hist = []
        elif src != "live":
            self.eng._check(self.eng.lib.faie2_snapshot_load(src))
            self.hist = list(self.snaps[src][0])
            self.snaps[src] = (self.snaps[src][0], time.time())
        reused = len(self.hist)
        pos = reused
        for b in sorted({m for m in marks if reused < m <= len(target)} | {len(target)}):
            while pos < b:
                chunk = target[pos:min(pos + 64, b)]   # 64-token prefill forwards (bit-identical to 8-token ones)
                self.eng.forward(chunk, want_logits=False)
                pos += len(chunk)
            self.hist = list(target[:pos])
            if b > reused:
                self._save(self.hist)
        return prompt[-1], reused, len(target) - reused

    def generate(self, prompt, max_new, marks, on_tokens):
        """Speculative greedy decode. on_tokens(new_ids) -> True to stop. Returns (tokens, finish, stats)."""
        t0 = time.perf_counter()
        anchor, reused, prefilled = self.prepare(prompt, marks)
        t1 = time.perf_counter()
        self.eng.spec_begin(anchor)
        pending, out, finish = anchor, [], "length"
        while len(out) < max_new:
            if self.eng.n_past() + 8 > self.ctx:
                break
            emit, _ = self.eng.spec_step()
            self.hist += [pending] + emit[:-1]
            pending = emit[-1]
            stop = next((i for i, t in enumerate(emit) if t == EOS), None)
            new = emit if stop is None else emit[:stop]
            new = new[:max_new - len(out)]
            out += new
            if stop is not None:
                finish = "stop"
                if new:
                    on_tokens(new)
                break
            if on_tokens(new):
                finish = "stop"
                break
        t2 = time.perf_counter()
        stats = {"prompt_n": len(prompt), "cached_n": reused, "prompt_ms": round((t1 - t0) * 1e3, 1),
                 "prefilled_n": prefilled, "predicted_n": len(out), "predicted_ms": round((t2 - t1) * 1e3, 1),
                 "predicted_per_second": round(len(out) / max(t2 - t1, 1e-9), 1)}
        return out, finish, stats


# ---------------------------------------------------------------------------------------------
# text: incremental decode, reasoning / content / tool-call split

class Stream:
    """Turns generated ids into (reasoning, content, tool-call) text with a hold-back for partial tags."""
    THINK_END = "</think>"
    TOOL_START = "<tool_call>"

    def __init__(self, tok, thinking, stops):
        self.tok, self.ids, self.done = tok, [], 0
        self.text = ""
        self.mode = "think" if thinking else "content"
        self.sent = {"reasoning": 0, "content": 0}
        self.reasoning, self.content, self.tool_text = "", "", ""
        self.stops = [s for s in (stops or []) if s]
        self.stopped = False

    def push(self, ids):
        self.ids += ids
        t = self.tok.decode(self.ids[self.done:])
        if t.endswith("\ufffd") and len(self.ids) - self.done < 8:   # likely a UTF-8 sequence split across ids
            return
        self.text += t
        self.done = len(self.ids)
        self._split()

    def _split(self):
        text = self.text
        if self.mode == "think":
            i = text.find(self.THINK_END)
            if i < 0:
                self.reasoning = text
                return
            self.reasoning = text[:i]
            self.mode = "content"
            self.body_start = i + len(self.THINK_END)
        body = text[getattr(self, "body_start", 0):].lstrip("\n")
        j = body.find(self.TOOL_START)
        if j >= 0:
            self.content, self.tool_text = body[:j], body[j:]
            self.mode = "tools"
        else:
            self.content = body
        for s in self.stops:
            k = self.content.find(s)
            if k >= 0:
                self.content = self.content[:k]
                self.stopped = True

    def deltas(self, final=False):
        """New reasoning/content text safe to send (holding back possible tag prefixes unless final)."""
        out = {}
        hold_r = 0 if final or self.mode != "think" else len(self.THINK_END) - 1
        r = self.reasoning[self.sent["reasoning"]:len(self.reasoning) - hold_r] if len(self.reasoning) > hold_r else ""
        if r:
            out["reasoning_content"] = r
            self.sent["reasoning"] += len(r)
        if self.mode != "think":
            hold_c = 0 if final or self.mode == "tools" else max(len(self.TOOL_START), max((len(s) for s in self.stops), default=0)) - 1
            c = self.content[self.sent["content"]:len(self.content) - hold_c] if len(self.content) > hold_c else ""
            if c:
                out["content"] = c
                self.sent["content"] += len(c)
        return out


TOOL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*(?:</tool_call>|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(tool_text, tools):
    props = {}
    for t in tools or []:
        f = t.get("function", t)
        props[f.get("name")] = (f.get("parameters") or {}).get("properties") or {}
    calls = []
    for m in TOOL_RE.finditer(tool_text):
        name = m.group(1).strip()
        args = {}
        for pm in PARAM_RE.finditer(m.group(2)):
            k, v = pm.group(1).strip(), pm.group(2)
            typ = props.get(name, {}).get(k, {}).get("type")
            if typ == "string":
                args[k] = v
            else:
                try:
                    args[k] = json.loads(v)
                except (ValueError, TypeError):
                    args[k] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


def normalize_messages(messages):
    """OpenAI wire format -> what the Qwen template expects (tool-call arguments as mappings)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                    except ValueError:
                        fn["arguments"] = {"input": fn["arguments"]}
                tcs.append(tc)
            m["tool_calls"] = tcs
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def prefix_marks(tok, prompt):
    """Snapshot points: end of the system block, start of the generation prompt (last <|im_start|>)."""
    im_start, im_end = tok.tk.token_to_id("<|im_start|>"), tok.im_end
    marks = set()
    if im_end in prompt:
        i = prompt.index(im_end)
        marks.add(i + 2)                                   # <|im_end|> "\n"
    starts = [i for i, t in enumerate(prompt) if t == im_start]
    if starts:
        marks.add(starts[-1])
    return marks


# ---------------------------------------------------------------------------------------------
# HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FAIE2"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            return self._json(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": ARGS.alias, "object": "model", "owned_by": "faie2"}]})
        if self.path == "/props":
            return self._json(200, {"model": ARGS.alias, "n_ctx": ENG.ctx, "snapshots": ENG.nsnap,
                                    "load_s": round(ENG.load_s, 1), "engine": "FAIE2"})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = self._body()
            route = {"/v1/chat/completions": self.chat, "/chat/completions": self.chat,
                     "/v1/completions": self.completions, "/completion": self.completion_llama,
                     "/tokenize": self.tokenize, "/detokenize": self.detokenize,
                     "/apply-template": self.apply_template}.get(self.path)
            if not route:
                return self._json(404, {"error": "not found"})
            route(req)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # report, keep serving
            traceback.print_exc()
            try:
                self._json(500, {"error": {"message": str(e), "type": type(e).__name__}})
            except Exception:
                pass

    # --- helpers ---
    def tokenize(self, req):
        self._json(200, {"tokens": TOK.encode(req.get("content", ""))})

    def detokenize(self, req):
        self._json(200, {"content": TOK.decode(req.get("tokens", []))})

    def _template_kwargs(self, req):
        kw = dict(req.get("chat_template_kwargs") or {})
        if "reasoning_effort" in req and "reasoning_effort" not in kw:
            kw["reasoning_effort"] = req["reasoning_effort"]
        return kw

    def apply_template(self, req):
        text = TOK.render(normalize_messages(req["messages"]), tools=req.get("tools"), **self._template_kwargs(req))
        self._json(200, {"prompt": text})

    def _max_new(self, req, prompt_len):
        m = req.get("max_completion_tokens") or req.get("max_tokens") or req.get("n_predict") or -1
        room = ENG.ctx - prompt_len - 8
        return room if m is None or m < 0 else min(m, room)

    # --- llama-server style /completion (used by the existing benchmarks) ---
    def completion_llama(self, req):
        p = req.get("prompt", "")
        prompt = p if isinstance(p, list) else TOK.encode(p)
        max_new = self._max_new(req, len(prompt))
        with ENG.lock:
            out, finish, stats = ENG.generate(prompt, max_new, prefix_marks(TOK, prompt), lambda ids: False)
        res = {"content": TOK.decode(out), "stop_type": "eos" if finish == "stop" else "limit", "timings": stats,
               "tokens_predicted": len(out), "tokens_evaluated": len(prompt)}
        if req.get("return_tokens"):
            res["tokens"] = out
        self._json(200, res)

    # --- OpenAI /v1/completions ---
    def completions(self, req):
        p = req.get("prompt", "")
        prompt = p if isinstance(p, list) and p and isinstance(p[0], int) else TOK.encode(p if isinstance(p, str) else p[0])
        max_new = self._max_new(req, len(prompt))
        stops = req.get("stop") or []
        stops = [stops] if isinstance(stops, str) else stops
        st = Stream(TOK, False, stops)
        with ENG.lock:
            out, finish, stats = ENG.generate(prompt, max_new, prefix_marks(TOK, prompt),
                                              lambda ids: (st.push(ids), st.stopped)[1])
        self._json(200, {"id": "cmpl-" + uuid.uuid4().hex[:24], "object": "text_completion", "created": int(time.time()),
                         "model": ARGS.alias, "system_fingerprint": FINGERPRINT,
                         "choices": [{"index": 0, "text": st.content, "finish_reason": finish}],
                         "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(out),
                                   "total_tokens": len(prompt) + len(out)}, "timings": stats})

    # --- OpenAI /v1/chat/completions ---
    def chat(self, req):
        tools = req.get("tools") if req.get("tool_choice") != "none" else None
        kw = self._template_kwargs(req)
        thinking = kw.get("enable_thinking", True) is not False
        text = TOK.render(normalize_messages(req["messages"]), tools=tools, **kw)
        prompt = TOK.encode(text)
        max_new = self._max_new(req, len(prompt))
        stops = req.get("stop") or []
        stops = [stops] if isinstance(stops, str) else stops
        st = Stream(TOK, thinking, stops)
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        base = {"id": cid, "created": created, "model": ARGS.alias, "system_fingerprint": FINGERPRINT}

        if not req.get("stream"):
            with ENG.lock:
                out, finish, stats = ENG.generate(prompt, max_new, prefix_marks(TOK, prompt),
                                                  lambda ids: (st.push(ids), st.stopped)[1])
            msg = {"role": "assistant", "content": st.content.strip() or None}
            if thinking:
                msg["reasoning_content"] = st.reasoning.strip()
            calls = parse_tool_calls(st.tool_text, tools) if st.tool_text else []
            if calls:
                msg["tool_calls"] = calls
                finish = "tool_calls"
            return self._json(200, dict(base, object="chat.completion",
                                        choices=[{"index": 0, "message": msg, "finish_reason": finish}],
                                        usage={"prompt_tokens": len(prompt), "completion_tokens": len(out),
                                               "total_tokens": len(prompt) + len(out),
                                               "prompt_tokens_details": {"cached_tokens": stats["cached_n"]}},
                                        timings=stats))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(delta, finish=None, extra=None):
            chunk = dict(base, object="chat.completion.chunk",
                         choices=[{"index": 0, "delta": delta, "finish_reason": finish}])
            if extra:
                chunk.update(extra)
            self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        send({"role": "assistant", "content": ""})

        def on_tokens(ids):
            st.push(ids)
            d = st.deltas()
            if d:
                send(d)
            return st.stopped

        with ENG.lock:
            out, finish, stats = ENG.generate(prompt, max_new, prefix_marks(TOK, prompt), on_tokens)
        d = st.deltas(final=True)
        if d:
            send(d)
        calls = parse_tool_calls(st.tool_text, tools) if st.tool_text else []
        if calls:
            send({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
            finish = "tool_calls"
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(out), "total_tokens": len(prompt) + len(out)}
        send({}, finish, {"usage": usage, "timings": stats})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def main():
    global ARGS, ENG, TOK, EOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--snapshots", type=int, default=4)
    ap.add_argument("--alias", default="qwen3.8-27b-abliterated-faie2")
    ARGS = ap.parse_args()
    TOK = QwenTokenizer(ARGS.model)
    EOS = TOK.eos
    ENG = Engine(ARGS.model, ARGS.draft, ARGS.ctx, ARGS.snapshots)
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"FAIE2 server ready on http://{ARGS.host}:{ARGS.port}  ctx={ARGS.ctx} snapshots={ARGS.snapshots} "
          f"load {ENG.load_s:.1f} s  GPU={os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
