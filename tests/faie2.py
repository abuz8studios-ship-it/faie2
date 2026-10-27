"""ctypes wrapper for engine/faie2.dll."""

import ctypes
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "..", "engine", "faie2.dll")
GGUF_PY = r"E:\ABU\MODELS\runtime\src\llama.cpp-dflash2\gguf-py"
VOCAB = 248320
D = 5120


def gguf_reader(path):
    """llama.cpp's gguf-py reader. Its metadata module imports yaml, which the reader never uses."""
    sys.modules.setdefault("yaml", types.ModuleType("yaml"))
    if GGUF_PY not in sys.path:
        sys.path.insert(0, GGUF_PY)
    from gguf import GGUFReader
    return GGUFReader(path)


class Faie2:
    def __init__(self, dll=DLL):
        os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")
        os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin")
        self.lib = ctypes.CDLL(os.path.abspath(dll))
        L = self.lib
        L.faie2_error.restype = ctypes.c_char_p
        L.faie2_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        L.faie2_forward.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
        L.faie2_last_gpu_ms.restype = ctypes.c_float
        L.faie2_last_host_ms.restype = ctypes.c_double
        L.faie2_get_hidden.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.faie2_verify.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                   ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]
        L.faie2_test_q6k_gemv.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    def _check(self, rc):
        if rc != 0:
            raise RuntimeError(self.lib.faie2_error().decode())

    def load(self, path, max_ctx=4096):
        self._check(self.lib.faie2_load(path.encode(), max_ctx))
        return self.lib.faie2_last_host_ms()

    def reset(self):
        self._check(self.lib.faie2_reset())

    def use_graphs(self, on):
        self._check(self.lib.faie2_use_graphs(int(on)))

    def set_layer_limit(self, n):
        self._check(self.lib.faie2_set_layer_limit(n))

    def forward(self, tokens, want_logits=True):
        arr = (ctypes.c_int * len(tokens))(*tokens)
        out = np.empty((len(tokens), VOCAB), dtype=np.float32) if want_logits else None
        self._check(self.lib.faie2_forward(arr, len(tokens), out.ctypes.data if want_logits else None))
        return out

    def verify(self, block, want_logits=False):
        """block = [t0, d1..d(n-1)]. Returns (emitted tokens, per-row argmax, logits or None).
        Emitted = accepted drafts + the target's own next token; that many tokens are committed."""
        n = len(block)
        arr = (ctypes.c_int * n)(*block)
        emit = (ctypes.c_int * 8)()
        n_emit = ctypes.c_int()
        am = (ctypes.c_int * n)()
        out = np.empty((n, VOCAB), dtype=np.float32) if want_logits else None
        self._check(self.lib.faie2_verify(arr, n, emit, ctypes.byref(n_emit), am,
                                          out.ctypes.data if want_logits else None))
        return list(emit[:n_emit.value]), list(am), out

    # ---- DFlash2 drafter ----
    def load_drafter(self, path):
        self.lib.faie2_load_drafter.argtypes = [ctypes.c_char_p]
        self._check(self.lib.faie2_load_drafter(path.encode()))
        return self.lib.faie2_last_host_ms()

    def spec_begin(self, anchor):
        self._check(self.lib.faie2_spec_begin(int(anchor)))

    def spec_step(self):
        """One on-GPU speculative step. Returns (emitted tokens, the 7 drafts)."""
        emit = (ctypes.c_int * 8)()
        n = ctypes.c_int()
        drafts = (ctypes.c_int * 7)()
        self._check(self.lib.faie2_spec_step(emit, ctypes.byref(n), drafts))
        return list(emit[:n.value]), list(drafts)

    def df_draft(self):
        block = (ctypes.c_int * 8)()
        cand = np.empty((8, 16), dtype=np.int32)
        score = np.empty((8, 16, 16), dtype=np.float32)
        self._check(self.lib.faie2_df_draft(block, cand.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                            score.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        return list(block), cand, score

    def df_features(self):
        out = np.empty((8, 5 * D), dtype=np.float16)
        self._check(self.lib.faie2_df_features(out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))))
        return out

    def hidden(self, nt):
        out = np.empty((nt, D), dtype=np.float32)
        self._check(self.lib.faie2_get_hidden(out.ctypes.data, nt))
        return out

    def gpu_ms(self):
        return self.lib.faie2_last_gpu_ms()

    def host_ms(self):
        return self.lib.faie2_last_host_ms()

    def n_past(self):
        return self.lib.faie2_n_past()

    def test_q6k_gemv(self, w_bytes, rows, cols, x, version=2):
        w = np.ascontiguousarray(w_bytes, dtype=np.uint8)
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.empty((x.shape[0], rows), dtype=np.float32)
        ms = ctypes.c_float()
        self._check(self.lib.faie2_test_q6k_gemv(w.ctypes.data, rows, cols, x.ctypes.data, x.shape[0],
                                                 y.ctypes.data, ctypes.byref(ms), version))
        return y, ms.value
