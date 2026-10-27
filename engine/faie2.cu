// FAIE2: decode/verify engine for Qwen3.8-27B (GGUF arch "qwen35"), Q6_K weights, RTX 5090 (sm_120).
//
// Phase 2 scope: an N <= 8 token forward ("verify path") that matches llama.cpp's math:
//   src/models/qwen35.cpp, src/models/delta-net-base.cpp, ggml-cuda/{gated_delta_net,rope,norm,ssm-conv}.cu
// Every 2D weight stays in its GGUF Q6_K blocks; one batched GEMV reads each weight once for all tokens.
// Positions live in device memory so a whole forward can be captured once and replayed as a CUDA graph.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#define FAIE2_API extern "C" __declspec(dllexport)

// ---------------------------------------------------------------------------------------------
// model constants (asserted against the GGUF at load time)

constexpr int D       = 5120;    // embedding
constexpr int NL      = 64;      // trunk layers (block 64 is the MTP head, not run here)
constexpr int FF      = 17408;   // ffn hidden
constexpr int NH      = 24;      // attention q heads
constexpr int NKV     = 4;       // attention kv heads
constexpr int HD      = 256;     // attention head dim
constexpr int NROT    = 64;      // rotary dims (partial rope)
constexpr int CONV_K  = 4;       // ssm conv kernel
constexpr int DS      = 128;     // ssm state size = k/v head dim
constexpr int NKH     = 16;      // gdn key heads
constexpr int NVH     = 48;      // gdn value heads
constexpr int DINNER  = NVH * DS;              // 6144
constexpr int CONVCH  = 2 * NKH * DS + DINNER; // 10240
constexpr int VOCAB   = 248320;
constexpr int MAXT    = 8;       // max tokens per verify forward (speculation block)
constexpr int MAXP    = 64;      // max tokens per prefill forward (8 groups of 8 through the same per-column math)
constexpr int N_RECR  = 48;
constexpr int N_ATTN  = 16;
constexpr float EPS        = 1e-6f;
constexpr float ROPE_BASE  = 10000000.0f;

constexpr int GGML_TYPE_F32  = 0;
constexpr int GGML_TYPE_Q6_K = 14;
constexpr int GGML_TYPE_Q8_0 = 8;
constexpr int Q6K_BYTES      = 210;  // ql[128] qh[64] scales[16] d(f16)
constexpr int Q80_BYTES      = 34;   // d(f16) q[32]

static bool is_attn_layer(int il) { return (il + 1) % 4 == 0; }

// ---------------------------------------------------------------------------------------------
// errors

static std::string g_err;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    g_err = std::string(#x) + " -> " + cudaGetErrorString(e_); return -1; } } while (0)
#define FAIL(msg) do { g_err = (msg); return -1; } while (0)

// ---------------------------------------------------------------------------------------------
// GGUF reader (memory-mapped)

struct GTensor {
    uint32_t type = 0;
    uint32_t nd = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    uint64_t offset = 0;
    uint64_t nbytes = 0;
    const uint8_t * host = nullptr;
};

struct Gguf {
    HANDLE hf = INVALID_HANDLE_VALUE;
    HANDLE hm = nullptr;
    const uint8_t * base = nullptr;
    uint64_t size = 0;
    uint64_t alignment = 32;
    std::unordered_map<std::string, GTensor> tensors;
    std::unordered_map<std::string, double> nums;
    std::unordered_map<std::string, std::vector<int64_t>> iarrays;  // integer arrays (e.g. dflash.target_layers)

    ~Gguf() {
        if (base) UnmapViewOfFile(base);
        if (hm) CloseHandle(hm);
        if (hf != INVALID_HANDLE_VALUE) CloseHandle(hf);
    }
};

struct Cursor {
    const uint8_t * p;
    const uint8_t * end;
    bool ok = true;
    template <typename T> T get() {
        T v{};
        if (p + sizeof(T) > end) { ok = false; return v; }
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (!ok || p + n > end) { ok = false; return {}; }
        std::string s((const char *) p, n);
        p += n;
        return s;
    }
};

static size_t gguf_scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}

static double gguf_scalar_value(Cursor & c, uint32_t t) {
    switch (t) {
        case 0: return c.get<uint8_t>();
        case 1: return c.get<int8_t>();
        case 2: return c.get<uint16_t>();
        case 3: return c.get<int16_t>();
        case 4: return c.get<uint32_t>();
        case 5: return c.get<int32_t>();
        case 6: return c.get<float>();
        case 7: return c.get<uint8_t>();
        case 10: return (double) c.get<uint64_t>();
        case 11: return (double) c.get<int64_t>();
        case 12: return c.get<double>();
        default: c.ok = false; return 0;
    }
}

static int gguf_open(Gguf & g, const char * path) {
    g.hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (g.hf == INVALID_HANDLE_VALUE) FAIL(std::string("cannot open ") + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(g.hf, &sz);
    g.size = (uint64_t) sz.QuadPart;
    g.hm = CreateFileMappingA(g.hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!g.hm) FAIL("CreateFileMapping failed");
    g.base = (const uint8_t *) MapViewOfFile(g.hm, FILE_MAP_READ, 0, 0, 0);
    if (!g.base) FAIL("MapViewOfFile failed");

    Cursor c{g.base, g.base + g.size};
    if (c.get<uint32_t>() != 0x46554747u) FAIL("not a GGUF file");
    const uint32_t version = c.get<uint32_t>();
    if (version < 2) FAIL("GGUF version < 2 not supported");
    const uint64_t n_tensors = c.get<uint64_t>();
    const uint64_t n_kv = c.get<uint64_t>();

    for (uint64_t i = 0; i < n_kv && c.ok; ++i) {
        std::string key = c.str();
        uint32_t t = c.get<uint32_t>();
        if (t == 8) {
            c.str();
        } else if (t == 9) {
            uint32_t et = c.get<uint32_t>();
            uint64_t n = c.get<uint64_t>();
            if (et == 8) {
                for (uint64_t j = 0; j < n && c.ok; ++j) c.str();
            } else if (et <= 5 || et == 10 || et == 11) {
                auto & v = g.iarrays[key];
                for (uint64_t j = 0; j < n && c.ok; ++j) v.push_back((int64_t) gguf_scalar_value(c, et));
            } else {
                size_t es = gguf_scalar_size(et);
                if (!es || c.p + es * n > c.end) FAIL("bad GGUF array");
                c.p += es * n;
            }
        } else {
            double v = gguf_scalar_value(c, t);
            g.nums[key] = v;
            if (key == "general.alignment") g.alignment = (uint64_t) v;
        }
    }
    std::vector<std::pair<std::string, GTensor>> infos;
    for (uint64_t i = 0; i < n_tensors && c.ok; ++i) {
        std::string name = c.str();
        GTensor t;
        t.nd = c.get<uint32_t>();
        for (uint32_t d = 0; d < t.nd; ++d) t.ne[d] = c.get<uint64_t>();
        t.type = c.get<uint32_t>();
        t.offset = c.get<uint64_t>();
        infos.emplace_back(std::move(name), t);
    }
    if (!c.ok) FAIL("truncated GGUF header");
    uint64_t pos = (uint64_t) (c.p - g.base);
    const uint64_t data_start = (pos + g.alignment - 1) / g.alignment * g.alignment;
    for (auto & [name, t] : infos) {
        uint64_t n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
        if (t.type == GGML_TYPE_F32) t.nbytes = n * 4;
        else if (t.type == GGML_TYPE_Q6_K) t.nbytes = n / 256 * Q6K_BYTES;
        else if (t.type == GGML_TYPE_Q8_0) t.nbytes = n / 32 * Q80_BYTES;
        else t.nbytes = 0;  // unsupported types are only an error if a needed tensor uses them
        t.host = g.base + data_start + t.offset;
        if (data_start + t.offset + t.nbytes > g.size) FAIL("tensor " + name + " runs past end of file");
        g.tensors[name] = t;
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// device helpers

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// result is returned to every thread of the block
template <int NW> __device__ __forceinline__ float block_sum(float v, float * red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float r = l < NW ? red[l] : 0.0f;
    return warp_sum(r);
}

template <int NW> __device__ __forceinline__ float block_max(float v, float * red) {
    v = warp_max(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float r = l < NW ? red[l] : -INFINITY;
    return warp_max(r);
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// Programmatic dependent launch. Every kernel the engine enqueues starts with PDL_ENTRY (or, in the GEMVs,
// fetches weights first): the trigger lets the next kernel be scheduled onto SMs this one frees, and the wait
// blocks until the previous kernel has completed and flushed, so nothing produced upstream is read early.
// Without the launch attribute both instructions are no-ops.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" :::); }
#define PDL_ENTRY() do { pdl_trigger(); pdl_wait(); } while (0)

// ---------------------------------------------------------------------------------------------
// kernels

// y[t] = rmsnorm(x[t]) * w      grid: nt blocks, 256 threads
__global__ void k_rmsnorm(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ y, int n) {
    __shared__ float red[32];
    const float * xr = x + (size_t) blockIdx.x * n;
    float * yr = y + (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum<8>(s, red);
    const float sc = rsqrtf(s / n + EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * sc * w[i];
}

__global__ void k_add(float * __restrict__ x, const float * __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void k_swiglu(const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}

// Q6_K batched GEMV: y[t][row] = sum_k W[row][k] * x[t][k], t < NT.
// One warp owns R consecutive rows; lane l handles block elements n*128 + l + 32*j (the dequantize_row_q6_K
// order), so every weight byte is read once per forward no matter how many tokens are verified.
template <int R, int NT>
__global__ void __launch_bounds__(128) k_q6k_gemv(const uint8_t * __restrict__ W, int rows, int cols,
                                                  const float * __restrict__ x, float * __restrict__ y) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= rows) return;

    const int nb = cols >> 8;
    const size_t rowbytes = (size_t) nb * Q6K_BYTES;
    const int is = lane >> 4;

    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        float xa[NT][8];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float * xp = x + (size_t) t * cols + ib * 256 + lane;
#pragma unroll
            for (int j = 0; j < 8; ++j) xa[t][j] = __ldg(xp + 32 * j);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= rows) break;
            const uint8_t * blk = W + (size_t) row * rowbytes + (size_t) ib * Q6K_BYTES;
            const float d = __half2float(*(const __half *) (blk + 208));
            const int8_t * sc = (const int8_t *) (blk + 192);
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int a  = __ldg(blk + n * 64 + lane);
                const int b  = __ldg(blk + n * 64 + lane + 32);
                const int hq = __ldg(blk + 128 + n * 32 + lane);
                const int8_t * s = sc + n * 8 + is;
                const float w0 = d * s[0] * (float) (((a & 0xF) | ((hq & 3) << 4)) - 32);
                const float w1 = d * s[2] * (float) (((b & 0xF) | (((hq >> 2) & 3) << 4)) - 32);
                const float w2 = d * s[4] * (float) (((a >> 4) | (((hq >> 4) & 3) << 4)) - 32);
                const float w3 = d * s[6] * (float) (((b >> 4) | (((hq >> 6) & 3) << 4)) - 32);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[r][t] += w0 * xa[t][n * 4 + 0] + w1 * xa[t][n * 4 + 1] + w2 * xa[t][n * 4 + 2] + w3 * xa[t][n * 4 + 3];
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0 && row < rows) y[(size_t) t * rows + row] = v;
        }
    }
}

// Q6_K v2. Weights are repacked once at load into a per-row SoA layout, same bytes and values:
//   row r: [nb x 128 ql][nb x 64 qh][nb x 16 scales][nb x 2 d]
// so every load below is 4-byte aligned. In each 256-element block, lane L takes ql word L: half n = L/16,
// l0 = 4*(L%8), jA = (L%16)/8. Its 8 elements are two runs of 4: n*128 + 32*jA + l0 + {0..3}, and the
// same + 64. Activations come in as float4. The 4 warps of a CTA split the row's blocks (split-K) and
// reduce through shared memory, so even 5120-row matrices fill the GPU. Summation order does not depend
// on NT, which keeps the forward bit-identical across batch sizes.
template <int R, int NT>
__global__ void __launch_bounds__(128) k_q6k_gemv2(const uint8_t * __restrict__ W, int rows, int cols,
                                                   const float * __restrict__ x, float * __restrict__ y) {
    __shared__ float part[4][R][NT];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * R;
    const int nb = cols >> 8;
    const size_t rowbytes = (size_t) nb * Q6K_BYTES;

    const int n = lane >> 4;
    const int jA = (lane >> 3) & 1;
    const int l0 = (lane & 7) * 4;
    const int e0 = n * 128 + 32 * jA + l0;   // first element of run A; run B = e0 + 64

    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

    for (int ib = warp; ib < nb; ib += 4) {
        float4 xa[NT], xb[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float * xp = x + (size_t) t * cols + ib * 256 + e0;
            xa[t] = __ldg((const float4 *) xp);
            xb[t] = __ldg((const float4 *) (xp + 64));
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            if (row >= rows) break;
            const uint8_t * base = W + (size_t) row * rowbytes;
            const uint32_t ql = __ldg((const uint32_t *) (base + (size_t) ib * 128) + lane);
            const uint32_t qh = __ldg((const uint32_t *) (base + (size_t) nb * 128 + (size_t) ib * 64) + n * 8 + (l0 >> 2));
            const int8_t * sc = (const int8_t *) (base + (size_t) nb * 192 + (size_t) ib * 16) + n * 8 + (l0 >> 4);
            const float d = __half2float(__ldg((const __half *) (base + (size_t) nb * 208) + ib));
            const float dA = d * (float) sc[2 * jA];
            const float dB = d * (float) sc[2 * jA + 4];
            float wa[4], wb[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int qlb = (ql >> (8 * k)) & 0xFF;
                const int qhb = (qh >> (8 * k)) & 0xFF;
                wa[k] = dA * (float) (((qlb & 0xF) | (((qhb >> (2 * jA)) & 3) << 4)) - 32);
                wb[k] = dB * (float) (((qlb >> 4) | (((qhb >> (2 * jA + 4)) & 3) << 4)) - 32);
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[r][t] += wa[0] * xa[t].x + wa[1] * xa[t].y + wa[2] * xa[t].z + wa[3] * xa[t].w
                           + wb[0] * xb[t].x + wb[1] * xb[t].y + wb[2] * xb[t].z + wb[3] * xb[t].w;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float v = warp_sum(acc[r][t]);
            if (lane == 0) part[warp][r][t] = v;
        }
    __syncthreads();
    if (threadIdx.x < R * NT) {
        const int r = threadIdx.x / NT, t = threadIdx.x % NT;
        const int row = row0 + r;
        if (row < rows) y[(size_t) t * rows + row] = (part[0][r][t] + part[1][r][t]) + (part[2][r][t] + part[3][r][t]);
    }
}

// raw GGUF Q6_K blocks (AoS, 210 B each) -> per-row SoA layout used by k_q6k_gemv2.   one thread per block
__global__ void k_q6k_repack(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int rows, int nb) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) rows * nb) return;
    const size_t row = i / nb, ib = i % nb;
    const uint8_t * s = src + i * Q6K_BYTES;
    uint8_t * base = dst + row * nb * Q6K_BYTES;
    for (int k = 0; k < 128; ++k) base[ib * 128 + k] = s[k];
    for (int k = 0; k < 64; ++k) base[nb * 128 + ib * 64 + k] = s[128 + k];
    for (int k = 0; k < 16; ++k) base[nb * 192 + ib * 16 + k] = s[192 + k];
    base[nb * 208 + ib * 2 + 0] = s[208];
    base[nb * 208 + ib * 2 + 1] = s[209];
}

// Q6_K v3: tensor-core GEMV, always 8 token columns (unused tokens are zero), so the result for a token
// does not depend on how many tokens share the call.
// mma.m16n8k16 (fp16 in, fp32 accumulate): M = 16 weight rows, N = 8 tokens, K = 16 inputs. Q6_K
// values q - 32 are integers in [-32, 31], exact in fp16, and each 16-element group has one scale d*sc,
// so every k-chunk runs an mma with zero C and is scaled into the fp32 accumulator per row.
// Load-time repack, per (16-row tile, 256-element block), 3360 bytes = 16 x 210:
//   [6 x 32 lanes x 16 B: the 96 bytes lane L needs for its A fragments, interleaved so each of the
//    6 warp-wide uint4 loads is one contiguous 512 B][16 rows x 16 B scales][16 rows x 2 B d]
// Lane L = 4g + q owns rows g, g+8 and k columns {2q, 2q+1, 2q+8, 2q+9} of every chunk. For
// combo c = 4rr + 2n + l0/16 (row g+8rr, half n, l0 in {0,16}) it stores 12 bytes:
//   ql[n*64 + l] (j=0/2 nibbles), ql[n*64 + 32 + l] (j=1/3 nibbles), qh[n*32 + l],
//   for l in {l0+2q, l0+2q+1, l0+2q+8, l0+2q+9}.
constexpr int Q6T_BYTES = 16 * Q6K_BYTES;  // 3360

__global__ void k_q6k_repack_mma(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int rows, int nb) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) (rows / 16) * nb * 32) return;
    const int L = (int) (idx & 31);
    const size_t tb = idx >> 5;                 // tile * nb + ib
    const int ib = (int) (tb % nb);
    const size_t tile = tb / nb;
    const int g = L >> 2, q = L & 3;
    uint8_t * out = dst + tb * Q6T_BYTES;
    // lane byte b (0..95) lives at uint4 slot b/16 of lane L
    auto put = [&](int b, uint8_t v) { out[((b >> 4) * 32 + L) * 16 + (b & 15)] = v; };
    for (int c = 0; c < 8; ++c) {
        const int rr = c >> 2, n = (c >> 1) & 1, l0 = (c & 1) * 16;
        const uint8_t * blk = src + ((tile * 16 + g + 8 * rr) * nb + ib) * Q6K_BYTES;
        const int ls[4] = {l0 + 2 * q, l0 + 2 * q + 1, l0 + 2 * q + 8, l0 + 2 * q + 9};
        for (int k = 0; k < 4; ++k) {
            put(12 * c + k, blk[n * 64 + ls[k]]);
            put(12 * c + 4 + k, blk[n * 64 + 32 + ls[k]]);
            put(12 * c + 8 + k, blk[128 + n * 32 + ls[k]]);
        }
    }
    if (L < 16) {
        const uint8_t * blk = src + ((tile * 16 + L) * nb + ib) * Q6K_BYTES;
        for (int k = 0; k < 16; ++k) out[3072 + L * 16 + k] = blk[192 + k];
        out[3328 + L * 2 + 0] = blk[208];
        out[3328 + L * 2 + 1] = blk[209];
    }
}

__device__ __forceinline__ void mma_16816(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%11,%12,%13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
}

// 4 bytes in [0, 63] -> two half2 of (byte - 32), exact: 0x64XX is fp16 1024 + XX.
__device__ __forceinline__ void q6_to_half2(uint32_t Q, uint32_t & lo, uint32_t & hi) {
    const __half2 magic = __halves2half2(__float2half(1056.0f), __float2half(1056.0f));
    uint32_t a = __byte_perm(Q, 0x64646464u, 0x5140);
    uint32_t b = __byte_perm(Q, 0x64646464u, 0x5342);
    __half2 ha = __hsub2(*(__half2 *) &a, magic);
    __half2 hb = __hsub2(*(__half2 *) &b, magic);
    lo = *(uint32_t *) &ha;
    hi = *(uint32_t *) &hb;
}

// Output segments: one launch can cover several matrices that share an input (weights concatenated by
// rows at load). Row r of the concatenation goes to segment s with row_end[s-1] <= r < row_end[s].
struct SegList {
    int n;
    int row_end[4];
    float * y[4];
};

// y_s[t][row] (+)= ... for t < nt.  x16: [8][cols] fp16, rows >= nt zero.  grid = rows/16 tiles, NW warps
// split the blocks. accumulate adds into y (fuses the residual add of the output projections).
template <int NW>
__global__ void __launch_bounds__(NW * 32) k_q6k_mma(const uint8_t * __restrict__ W, int rows, int cols,
                                                    const __half * __restrict__ x16, SegList segs, int nt,
                                                    int accumulate) {
    __shared__ float part[NW][32][4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int tile = blockIdx.x;
    const int nb = cols >> 8;
    const __half * xg = x16 + (size_t) g * cols + 2 * q;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // one block's operands; the next block's are fetched before the current one is computed
    struct Blk { uint4 v[6]; uint4 scg, scg8; __half dg, dg8; };
    auto fetch = [&](int ib, Blk & b) {
        const uint8_t * base = W + ((size_t) tile * nb + ib) * Q6T_BYTES;
        const uint4 * lp = (const uint4 *) base + lane;
#pragma unroll
        for (int i = 0; i < 6; ++i) b.v[i] = __ldg(lp + i * 32);
        b.scg = __ldg((const uint4 *) (base + 3072 + g * 16));
        b.scg8 = __ldg((const uint4 *) (base + 3072 + (g + 8) * 16));
        b.dg = __ldg((const __half *) (base + 3328) + g);
        b.dg8 = __ldg((const __half *) (base + 3328) + g + 8);
    };
    Blk cur, nxt;
    if (warp < nb) fetch(warp, cur);
    pdl_trigger();
    pdl_wait();

    for (int ib = warp; ib < nb; ib += NW) {
        if (ib + NW < nb) fetch(ib + NW, nxt);
        uint32_t w[24];
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            w[4 * i + 0] = cur.v[i].x; w[4 * i + 1] = cur.v[i].y; w[4 * i + 2] = cur.v[i].z; w[4 * i + 3] = cur.v[i].w;
        }
        const float dg = __half2float(cur.dg);
        const float dg8 = __half2float(cur.dg8);
        const uint32_t sg[4] = {cur.scg.x, cur.scg.y, cur.scg.z, cur.scg.w};
        const uint32_t sg8[4] = {cur.scg8.x, cur.scg8.y, cur.scg8.z, cur.scg8.w};
        const __half * xb = xg + ib * 256;
#pragma unroll
        for (int c = 0; c < 4; ++c) {          // (n, l0) combos; row g uses combo c, row g+8 uses c + 4
            const int n = c >> 1, l0 = (c & 1) * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t srcA = (j & 1) ? w[3 * c + 1] : w[3 * c];
                const uint32_t srcB = (j & 1) ? w[3 * (c + 4) + 1] : w[3 * (c + 4)];
                const uint32_t nibA = (j >> 1) ? ((srcA >> 4) & 0x0F0F0F0Fu) : (srcA & 0x0F0F0F0Fu);
                const uint32_t nibB = (j >> 1) ? ((srcB >> 4) & 0x0F0F0F0Fu) : (srcB & 0x0F0F0F0Fu);
                const uint32_t QA = nibA | (((w[3 * c + 2] >> (2 * j)) & 0x03030303u) << 4);
                const uint32_t QB = nibB | (((w[3 * (c + 4) + 2] >> (2 * j)) & 0x03030303u) << 4);
                uint32_t a[4];
                q6_to_half2(QA, a[0], a[2]);   // row g: cols (2q, 2q+1) -> a0a1, (2q+8, 2q+9) -> a4a5
                q6_to_half2(QB, a[1], a[3]);   // row g+8
                const int e0 = n * 128 + 32 * j + l0;
                uint32_t b[2];
                b[0] = __ldg((const uint32_t *) (xb + e0));
                b[1] = __ldg((const uint32_t *) (xb + e0 + 8));
                float dd[4];
                mma_16816(dd, a, b);
                const int si = n * 8 + l0 / 16 + 2 * j;   // scale index within the block, 0..15
                const float s0 = dg * (float) (int8_t) (sg[si >> 2] >> (8 * (si & 3)));
                const float s1 = dg8 * (float) (int8_t) (sg8[si >> 2] >> (8 * (si & 3)));
                acc[0] += s0 * dd[0];
                acc[1] += s0 * dd[1];
                acc[2] += s1 * dd[2];
                acc[3] += s1 * dd[3];
            }
        }
        cur = nxt;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) part[warp][lane][i] = acc[i];
    __syncthreads();
    if (threadIdx.x < 128) {
        const int L = threadIdx.x >> 2, i = threadIdx.x & 3;
        float s = 0.0f;
#pragma unroll
        for (int w2 = 0; w2 < NW; ++w2) s += part[w2][L][i];
        const int row = tile * 16 + (L >> 2) + 8 * (i >> 1);
        const int t = 2 * (L & 3) + (i & 1);
        if (t < nt) {
            int sg = 0;
            while (sg < segs.n - 1 && row >= segs.row_end[sg]) ++sg;
            const int r0 = sg ? segs.row_end[sg - 1] : 0;
            float * dst = segs.y[sg] + (size_t) t * (segs.row_end[sg] - r0) + (row - r0);
            *dst = accumulate ? *dst + s : s;
        }
    }
}

// Q8_0 (the DFlash2 drafter's format): 32-element blocks of {fp16 d, int8 q[32]}, 34 bytes.
// Same tensor-core scheme as Q6_K: int8 is exact in fp16 and each 16-element k-chunk lies inside one
// 32-element block, so its scale is applied after the mma. Repack per (16-row tile, 256-column
// superblock), 4352 bytes = 16 rows x 8 blocks x 34:
//   [8 x 32 lanes x 16 B][16 rows x 8 fp16 scales]
// Lane L = 4g+q stores, for chunk c = 0..15, 4 bytes of row g then 4 of row g+8, at chunk columns
// {2q, 2q+1, 2q+8, 2q+9}; byte b of a lane lives at uint4 slot b/16 (interleaved like Q6_K).
constexpr int Q8T_BYTES = 16 * 8 * Q80_BYTES;  // 4352

__global__ void k_q8_repack_mma(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int rows, int nsb) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) (rows / 16) * nsb * 32) return;
    const int L = (int) (idx & 31);
    const size_t tb = idx >> 5;                 // tile * nsb + sb
    const int sb = (int) (tb % nsb);
    const size_t tile = tb / nsb;
    const int g = L >> 2, q = L & 3;
    const int nb32 = nsb * 8;                   // Q8_0 blocks per row
    uint8_t * out = dst + tb * Q8T_BYTES;
    const int cols4[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
    for (int c = 0; c < 16; ++c) {
        for (int rr = 0; rr < 2; ++rr) {
            const size_t row = tile * 16 + g + 8 * rr;
            for (int k = 0; k < 4; ++k) {
                const int col = c * 16 + cols4[k];
                const uint8_t * blk = src + (row * nb32 + sb * 8 + col / 32) * Q80_BYTES;
                const int b = c * 8 + rr * 4 + k;
                out[((b >> 4) * 32 + L) * 16 + (b & 15)] = blk[2 + col % 32];
            }
        }
    }
    if (L < 16) {
        for (int k = 0; k < 8; ++k) {
            const uint8_t * blk = src + (((size_t) tile * 16 + L) * nb32 + sb * 8 + k) * Q80_BYTES;
            out[4096 + L * 16 + k * 2 + 0] = blk[0];
            out[4096 + L * 16 + k * 2 + 1] = blk[1];
        }
    }
}

// 4 signed bytes -> two half2, exact: (b ^ 0x80) = b + 128 in [0, 255], 0x64XX = 1024 + XX
__device__ __forceinline__ void q8_to_half2(uint32_t Q, uint32_t & lo, uint32_t & hi) {
    const __half2 magic = __halves2half2(__float2half(1152.0f), __float2half(1152.0f));
    const uint32_t u = Q ^ 0x80808080u;
    uint32_t a = __byte_perm(u, 0x64646464u, 0x5140);
    uint32_t b = __byte_perm(u, 0x64646464u, 0x5342);
    __half2 ha = __hsub2(*(__half2 *) &a, magic);
    __half2 hb = __hsub2(*(__half2 *) &b, magic);
    lo = *(uint32_t *) &ha;
    hi = *(uint32_t *) &hb;
}

// min 2 CTAs per SM: unbounded, ptxas used 130 registers, which left room for one 256-thread CTA per SM
template <int NW>
__global__ void __launch_bounds__(NW * 32, 2) k_q8_mma(const uint8_t * __restrict__ W, int rows, int cols,
                                                   const __half * __restrict__ x16, SegList segs, int nt, int accumulate) {
    __shared__ float part[NW][32][4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int tile = blockIdx.x;
    const int nsb = cols >> 8;
    const __half * xg = x16 + (size_t) g * cols + 2 * q;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    struct Blk { uint4 v[8]; uint4 sg, sg8; };
    auto fetch = [&](int sb, Blk & b) {
        const uint8_t * base = W + ((size_t) tile * nsb + sb) * Q8T_BYTES;
        const uint4 * lp = (const uint4 *) base + lane;
#pragma unroll
        for (int i = 0; i < 8; ++i) b.v[i] = __ldg(lp + i * 32);
        b.sg = __ldg((const uint4 *) (base + 4096 + g * 16));
        b.sg8 = __ldg((const uint4 *) (base + 4096 + (g + 8) * 16));
    };
    Blk cur, nxt;
    if (warp < nsb) fetch(warp, cur);
    pdl_trigger();
    pdl_wait();
    for (int sb = warp; sb < nsb; sb += NW) {
        if (sb + NW < nsb) fetch(sb + NW, nxt);
        uint32_t w[32];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[4 * i + 0] = cur.v[i].x; w[4 * i + 1] = cur.v[i].y; w[4 * i + 2] = cur.v[i].z; w[4 * i + 3] = cur.v[i].w;
        }
        const __half * dg = (const __half *) &cur.sg;
        const __half * dg8 = (const __half *) &cur.sg8;
        const __half * xb = xg + sb * 256;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uint32_t a[4];
            q8_to_half2(w[2 * c], a[0], a[2]);       // row g
            q8_to_half2(w[2 * c + 1], a[1], a[3]);   // row g+8
            uint32_t b[2];
            b[0] = __ldg((const uint32_t *) (xb + c * 16));
            b[1] = __ldg((const uint32_t *) (xb + c * 16 + 8));
            float dd[4];
            mma_16816(dd, a, b);
            const float s0 = __half2float(dg[c >> 1]);
            const float s1 = __half2float(dg8[c >> 1]);
            acc[0] += s0 * dd[0];
            acc[1] += s0 * dd[1];
            acc[2] += s1 * dd[2];
            acc[3] += s1 * dd[3];
        }
        cur = nxt;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) part[warp][lane][i] = acc[i];
    __syncthreads();
    if (threadIdx.x < 128) {
        const int L = threadIdx.x >> 2, i = threadIdx.x & 3;
        float s = 0.0f;
#pragma unroll
        for (int w2 = 0; w2 < NW; ++w2) s += part[w2][L][i];
        const int row = tile * 16 + (L >> 2) + 8 * (i >> 1);
        const int t = 2 * (L & 3) + (i & 1);
        if (t < nt) {
            int sg = 0;
            while (sg < segs.n - 1 && row >= segs.row_end[sg]) ++sg;
            const int r0 = sg ? segs.row_end[sg - 1] : 0;
            float * dst = segs.y[sg] + (size_t) t * (segs.row_end[sg] - r0) + (row - r0);
            *dst = accumulate ? *dst + s : s;
        }
    }
}

// ---- v4 GEMV: balanced split-K and programmatic dependent launch ----
// v3 gives a 16-row tile to one 8-warp CTA. A tile has 20 blocks for a 5120-column matrix, so warps ran 3 or
// 2 blocks (83% balance), and the 5120-row matrices had only 320 CTAs. v4 uses 4 warps of J blocks each and
// splits a tile's blocks over S = gridDim.y CTAs. Split partials are summed in split order by the tile's last
// CTA to finish (a per-tile counter), so results are deterministic and still independent of nt.
// Each kernel fetches its first weights before griddepcontrol.wait: weights never depend on the previous
// kernel, so under PDL that load overlaps the previous kernel's tail.

constexpr int GEMV_NW = 4;

struct SplitK {
    float * partial;   // [tiles][S][128]
    int * counter;     // [tiles], zero between launches
};

// 16 rows x 8 tokens per CTA: sum the 4 warps, then (S > 1) combine splits; thread i owns output i.
__device__ __forceinline__ void gemv_finish(float (*part)[32][4], const float acc[4], int tile, SegList segs, int nt,
                                            int accumulate, SplitK sk) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < 4; ++i) part[warp][lane][i] = acc[i];
    __syncthreads();
    const int L = threadIdx.x >> 2, i = threadIdx.x & 3;
    float s = 0.0f;
#pragma unroll
    for (int w2 = 0; w2 < GEMV_NW; ++w2) s += part[w2][L][i];
    const int S = gridDim.y;
    if (S > 1) {
        __shared__ int last;
        sk.partial[((size_t) tile * S + blockIdx.y) * 128 + threadIdx.x] = s;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) last = atomicAdd(sk.counter + tile, 1) == S - 1;
        __syncthreads();
        if (!last) return;
        s = 0.0f;
        for (int k = 0; k < S; ++k) s += __ldcg(sk.partial + ((size_t) tile * S + k) * 128 + threadIdx.x);
        if (threadIdx.x == 0) sk.counter[tile] = 0;
    }
    const int row = tile * 16 + (L >> 2) + 8 * (i >> 1);
    const int t = 2 * (L & 3) + (i & 1);
    if (t < nt) {
        int sg = 0;
        while (sg < segs.n - 1 && row >= segs.row_end[sg]) ++sg;
        const int r0 = sg ? segs.row_end[sg - 1] : 0;
        float * dst = segs.y[sg] + (size_t) t * (segs.row_end[sg] - r0) + (row - r0);
        *dst = accumulate ? *dst + s : s;
    }
}

// Q6_K, same repacked layout and per-block math as k_q6k_mma.  grid (tiles, S), 128 threads
__global__ void __launch_bounds__(128) k_q6k_mma4(const uint8_t * __restrict__ W, int cols, const __half * __restrict__ x16,
                                                 SegList segs, int nt, int accumulate, int J, SplitK sk) {
    __shared__ float part[GEMV_NW][32][4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int tile = blockIdx.x;
    const int nb = cols >> 8;
    const int ib0 = blockIdx.y * J * GEMV_NW + warp;
    const __half * xg = x16 + (size_t) g * cols + 2 * q;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    struct Blk { uint4 v[6]; uint4 scg, scg8; __half dg, dg8; };
    auto fetch = [&](int ib, Blk & b) {
        const uint8_t * base = W + ((size_t) tile * nb + ib) * Q6T_BYTES;
        const uint4 * lp = (const uint4 *) base + lane;
#pragma unroll
        for (int i = 0; i < 6; ++i) b.v[i] = __ldg(lp + i * 32);
        b.scg = __ldg((const uint4 *) (base + 3072 + g * 16));
        b.scg8 = __ldg((const uint4 *) (base + 3072 + (g + 8) * 16));
        b.dg = __ldg((const __half *) (base + 3328) + g);
        b.dg8 = __ldg((const __half *) (base + 3328) + g + 8);
    };
    Blk cur, nxt;
    fetch(ib0, cur);
    pdl_trigger();
    pdl_wait();
    for (int j = 0; j < J; ++j) {
        const int ib = ib0 + j * GEMV_NW;
        if (j + 1 < J) fetch(ib + GEMV_NW, nxt);
        uint32_t w[24];
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            w[4 * i + 0] = cur.v[i].x; w[4 * i + 1] = cur.v[i].y; w[4 * i + 2] = cur.v[i].z; w[4 * i + 3] = cur.v[i].w;
        }
        const float dg = __half2float(cur.dg);
        const float dg8 = __half2float(cur.dg8);
        const uint32_t sg[4] = {cur.scg.x, cur.scg.y, cur.scg.z, cur.scg.w};
        const uint32_t sg8[4] = {cur.scg8.x, cur.scg8.y, cur.scg8.z, cur.scg8.w};
        const __half * xb = xg + ib * 256;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int n = c >> 1, l0 = (c & 1) * 16;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const uint32_t srcA = (jj & 1) ? w[3 * c + 1] : w[3 * c];
                const uint32_t srcB = (jj & 1) ? w[3 * (c + 4) + 1] : w[3 * (c + 4)];
                const uint32_t nibA = (jj >> 1) ? ((srcA >> 4) & 0x0F0F0F0Fu) : (srcA & 0x0F0F0F0Fu);
                const uint32_t nibB = (jj >> 1) ? ((srcB >> 4) & 0x0F0F0F0Fu) : (srcB & 0x0F0F0F0Fu);
                const uint32_t QA = nibA | (((w[3 * c + 2] >> (2 * jj)) & 0x03030303u) << 4);
                const uint32_t QB = nibB | (((w[3 * (c + 4) + 2] >> (2 * jj)) & 0x03030303u) << 4);
                uint32_t a[4];
                q6_to_half2(QA, a[0], a[2]);
                q6_to_half2(QB, a[1], a[3]);
                const int e0 = n * 128 + 32 * jj + l0;
                uint32_t b[2];
                b[0] = __ldg((const uint32_t *) (xb + e0));
                b[1] = __ldg((const uint32_t *) (xb + e0 + 8));
                float dd[4];
                mma_16816(dd, a, b);
                const int si = n * 8 + l0 / 16 + 2 * jj;
                const float s0 = dg * (float) (int8_t) (sg[si >> 2] >> (8 * (si & 3)));
                const float s1 = dg8 * (float) (int8_t) (sg8[si >> 2] >> (8 * (si & 3)));
                acc[0] += s0 * dd[0];
                acc[1] += s0 * dd[1];
                acc[2] += s1 * dd[2];
                acc[3] += s1 * dd[3];
            }
        }
        cur = nxt;
    }
    gemv_finish(part, acc, tile, segs, nt, accumulate, sk);
}

// Q8_0, same repacked layout and math as k_q8_mma.  grid (tiles, S), 128 threads
__global__ void __launch_bounds__(128) k_q8_mma4(const uint8_t * __restrict__ W, int cols, const __half * __restrict__ x16,
                                                SegList segs, int nt, int accumulate, int J, SplitK sk) {
    __shared__ float part[GEMV_NW][32][4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int tile = blockIdx.x;
    const int nsb = cols >> 8;
    const int sb0 = blockIdx.y * J * GEMV_NW + warp;
    const __half * xg = x16 + (size_t) g * cols + 2 * q;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    struct Blk { uint4 v[8]; uint4 sg, sg8; };
    auto fetch = [&](int sb, Blk & b) {
        const uint8_t * base = W + ((size_t) tile * nsb + sb) * Q8T_BYTES;
        const uint4 * lp = (const uint4 *) base + lane;
#pragma unroll
        for (int i = 0; i < 8; ++i) b.v[i] = __ldg(lp + i * 32);
        b.sg = __ldg((const uint4 *) (base + 4096 + g * 16));
        b.sg8 = __ldg((const uint4 *) (base + 4096 + (g + 8) * 16));
    };
    Blk cur, nxt;
    fetch(sb0, cur);
    pdl_trigger();
    pdl_wait();
    for (int j = 0; j < J; ++j) {
        const int sb = sb0 + j * GEMV_NW;
        if (j + 1 < J) fetch(sb + GEMV_NW, nxt);
        uint32_t w[32];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[4 * i + 0] = cur.v[i].x; w[4 * i + 1] = cur.v[i].y; w[4 * i + 2] = cur.v[i].z; w[4 * i + 3] = cur.v[i].w;
        }
        const __half * dg = (const __half *) &cur.sg;
        const __half * dg8 = (const __half *) &cur.sg8;
        const __half * xb = xg + sb * 256;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uint32_t a[4];
            q8_to_half2(w[2 * c], a[0], a[2]);
            q8_to_half2(w[2 * c + 1], a[1], a[3]);
            uint32_t b[2];
            b[0] = __ldg((const uint32_t *) (xb + c * 16));
            b[1] = __ldg((const uint32_t *) (xb + c * 16 + 8));
            float dd[4];
            mma_16816(dd, a, b);
            const float s0 = __half2float(dg[c >> 1]);
            const float s1 = __half2float(dg8[c >> 1]);
            acc[0] += s0 * dd[0];
            acc[1] += s0 * dd[1];
            acc[2] += s1 * dd[2];
            acc[3] += s1 * dd[3];
        }
        cur = nxt;
    }
    gemv_finish(part, acc, tile, segs, nt, accumulate, sk);
}

// ---- v5 GEMV: warp-granular stream-K ----
// v3/v4 hand out whole tiles, so a launch runs in waves of CTAs and the last wave is partly empty (1030 tiles
// on 340 CTA slots = 3.03 waves, run as 4). v5 launches exactly 2 CTAs per SM and gives every warp an equal,
// contiguous range of the matrix's (tile, block) list: warp w of wtot owns blocks [w*W/wtot, (w+1)*W/wtot).
// A tile wholly inside one warp's range is written directly. A tile cut by range boundaries gets one partial
// per warp that touches it (slot 0 if it is that warp's first tile, else slot 1); the last of those warps to
// arrive (per-tile counter) sums them in warp order. Fixed partition + fixed order = deterministic, and the
// per-token math is unchanged, so results stay independent of nt.

template <bool Q8> struct GemvFmt;

template <> struct GemvFmt<false> {   // Q6_K, layout of k_q6k_repack_mma
    static constexpr int TILE_BYTES = Q6T_BYTES;
    struct Blk { uint4 v[6]; uint4 scg, scg8; __half dg, dg8; };
    static __device__ __forceinline__ void fetch(const uint8_t * W, long long b, int lane, int g, Blk & k) {
        const uint8_t * base = W + (size_t) b * TILE_BYTES;
        const uint4 * lp = (const uint4 *) base + lane;
#pragma unroll
        for (int i = 0; i < 6; ++i) k.v[i] = __ldg(lp + i * 32);
        k.scg = __ldg((const uint4 *) (base + 3072 + g * 16));
        k.scg8 = __ldg((const uint4 *) (base + 3072 + (g + 8) * 16));
        k.dg = __ldg((const __half *) (base + 3328) + g);
        k.dg8 = __ldg((const __half *) (base + 3328) + g + 8);
    }
    static __device__ __forceinline__ void compute(const Blk & cur, const __half * xb, float acc[4]) {
        uint32_t w[24];
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            w[4 * i + 0] = cur.v[i].x; w[4 * i + 1] = cur.v[i].y; w[4 * i + 2] = cur.v[i].z; w[4 * i + 3] = cur.v[i].w;
        }
        const float dg = __half2float(cur.dg);
        const float dg8 = __half2float(cur.dg8);
        const uint32_t sg[4] = {cur.scg.x, cur.scg.y, cur.scg.z, cur.scg.w};
        const uint32_t sg8[4] = {cur.scg8.x, cur.scg8.y, cur.scg8.z, cur.scg8.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int n = c >> 1, l0 = (c & 1) * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t srcA = (j & 1) ? w[3 * c + 1] : w[3 * c];
                const uint32_t srcB = (j & 1) ? w[3 * (c + 4) + 1] : w[3 * (c + 4)];
                const uint32_t nibA = (j >> 1) ? ((srcA >> 4) & 0x0F0F0F0Fu) : (srcA & 0x0F0F0F0Fu);
                const uint32_t nibB = (j >> 1) ? ((srcB >> 4) & 0x0F0F0F0Fu) : (srcB & 0x0F0F0F0Fu);
                const uint32_t QA = nibA | (((w[3 * c + 2] >> (2 * j)) & 0x03030303u) << 4);
                const uint32_t QB = nibB | (((w[3 * (c + 4) + 2] >> (2 * j)) & 0x03030303u) << 4);
                uint32_t a[4];
                q6_to_half2(QA, a[0], a[2]);
                q6_to_half2(QB, a[1], a[3]);
                const int e0 = n * 128 + 32 * j + l0;
                uint32_t b[2];
                b[0] = __ldg((const uint32_t *) (xb + e0));
                b[1] = __ldg((const uint32_t *) (xb + e0 + 8));
                float dd[4];
                mma_16816(dd, a, b);
                const int si = n * 8 + l0 / 16 + 2 * j;
                const float s0 = dg * (float) (int8_t) (sg[si >> 2] >> (8 * (si & 3)));
                const float s1 = dg8 * (float) (int8_t) (sg8[si >> 2] >> (8 * (si & 3)));
                acc[0] += s0 * dd[0];
                acc[1] += s0 * dd[1];
                acc[2] += s1 * dd[2];
                acc[3] += s1 * dd[3];
            }
        }
    }
};

template <> struct GemvFmt<true> {    // Q8_0, layout of k_q8_repack_mma
    static constexpr int TILE_BYTES = Q8T_BYTES;
    struct Blk { uint4 v[8]; uint4 sg, sg8; };
    static __device__ __forceinline__ void fetch(const uint8_t * W, long long b, int lane, int g, Blk & k) {
        const uint8_t * base = W + (size_t) b * TILE_BYTES;
        const uint4 * lp = (const uint4 *) base + lane;
#pragma unroll
        for (int i = 0; i < 8; ++i) k.v[i] = __ldg(lp + i * 32);
        k.sg = __ldg((const uint4 *) (base + 4096 + g * 16));
        k.sg8 = __ldg((const uint4 *) (base + 4096 + (g + 8) * 16));
    }
    static __device__ __forceinline__ void compute(const Blk & cur, const __half * xb, float acc[4]) {
        uint32_t w[32];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[4 * i + 0] = cur.v[i].x; w[4 * i + 1] = cur.v[i].y; w[4 * i + 2] = cur.v[i].z; w[4 * i + 3] = cur.v[i].w;
        }
        const __half * dg = (const __half *) &cur.sg;
        const __half * dg8 = (const __half *) &cur.sg8;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uint32_t a[4];
            q8_to_half2(w[2 * c], a[0], a[2]);
            q8_to_half2(w[2 * c + 1], a[1], a[3]);
            uint32_t b[2];
            b[0] = __ldg((const uint32_t *) (xb + c * 16));
            b[1] = __ldg((const uint32_t *) (xb + c * 16 + 8));
            float dd[4];
            mma_16816(dd, a, b);
            const float s0 = __half2float(dg[c >> 1]);
            const float s1 = __half2float(dg8[c >> 1]);
            acc[0] += s0 * dd[0];
            acc[1] += s0 * dd[1];
            acc[2] += s1 * dd[2];
            acc[3] += s1 * dd[3];
        }
    }
};

// lane holds rows (g, g+8) x tokens (2q, 2q+1) of a 16x8 tile: acc[i] -> row g + 8*(i>>1), token 2q + (i&1)
__device__ __forceinline__ void gemv_store_tile(const float v[4], int tile, int lane, SegList segs, int nt, int accumulate) {
    const int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = tile * 16 + g + 8 * (i >> 1);
        const int t = 2 * q + (i & 1);
        if (t >= nt) continue;
        int sg = 0;
        while (sg < segs.n - 1 && row >= segs.row_end[sg]) ++sg;
        const int r0 = sg ? segs.row_end[sg - 1] : 0;
        float * dst = segs.y[sg] + (size_t) t * (segs.row_end[sg] - r0) + (row - r0);
        *dst = accumulate ? *dst + v[i] : v[i];
    }
}

constexpr int GEMV5_WARPS = 8;   // warps per CTA; the grid is 2 CTAs per SM

template <bool Q8>
__global__ void __launch_bounds__(GEMV5_WARPS * 32, 2) k_gemv5(const uint8_t * __restrict__ W, int tiles, int cols,
                                                              const __half * __restrict__ x16, SegList segs, int nt,
                                                              int accumulate, SplitK sk) {
    using F = GemvFmt<Q8>;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, q = lane & 3;
    const int nb = cols >> 8;
    const long long WB = (long long) tiles * nb;
    const long long wtot = min((long long) gridDim.x * GEMV5_WARPS, WB);
    const long long wid = (long long) blockIdx.x * GEMV5_WARPS + (threadIdx.x >> 5);
    if (wid >= wtot) {
        pdl_trigger();
        return;
    }
    auto start = [&](long long w) { return w * WB / wtot; };
    auto owner = [&](long long blk) { return ((blk + 1) * wtot - 1) / WB; };
    const long long b0 = start(wid), b1 = start(wid + 1);
    const __half * xg = x16 + (size_t) g * cols + 2 * q;

    typename F::Blk cur, nxt;
    F::fetch(W, b0, lane, g, cur);
    pdl_trigger();
    pdl_wait();
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int tile = (int) (b0 / nb);
    for (long long b = b0; b < b1; ++b) {
        if (b + 1 < b1) F::fetch(W, b + 1, lane, g, nxt);
        F::compute(cur, xg + (b - (long long) tile * nb) * 256, acc);
        if (b + 1 == b1 || (b + 1) % nb == 0) {
            const long long tb0 = (long long) tile * nb, tb1 = tb0 + nb;
            if (b0 <= tb0 && b1 >= tb1) {
                gemv_store_tile(acc, tile, lane, segs, nt, accumulate);
            } else {
                const int slot = tile == (int) (b0 / nb) ? 0 : 1;
                *(float4 *) (sk.partial + ((size_t) wid * 2 + slot) * 128 + lane * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
                __threadfence();
                __syncwarp();
                const long long first = owner(tb0), last = owner(tb1 - 1);
                int is_last = 0;
                if (lane == 0) is_last = atomicAdd(sk.counter + tile, 1) == (int) (last - first);
                is_last = __shfl_sync(0xffffffffu, is_last, 0);
                if (is_last) {
                    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (long long j = first; j <= last; ++j) {
                        const int sl = tile == (int) (start(j) / nb) ? 0 : 1;
                        const float4 v = __ldcg((const float4 *) (sk.partial + ((size_t) j * 2 + sl) * 128 + lane * 4));
                        s[0] += v.x; s[1] += v.y; s[2] += v.z; s[3] += v.w;
                    }
                    gemv_store_tile(s, tile, lane, segs, nt, accumulate);
                    if (lane == 0) sk.counter[tile] = 0;
                }
            }
            acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
            ++tile;
        }
        cur = nxt;
    }
}

// ---- v6 GEMV: data-parallel rounds + stream-K remainder (CUTLASS "DP + SK" hybrid) ----
// v3 runs ceil(tiles / CTA slots) waves and the last one is mostly empty (in_proj: 1030 tiles on 340 slots
// = 3.03 waves). v6 keeps 2 persistent CTAs per SM: round r gives CTA c tile r*G + c with v3's in-tile layout
// (8 warps on adjacent blocks, DRAM-friendly), and only the tiles left over after the full rounds are cut
// into equal contiguous block ranges, one per CTA, so the tail costs about one block per warp instead of a tile.
// Warps never sync per tile: each deposits its per-tile sum in a 4-slot shared ring and the last warp to arrive
// reduces the slot in warp order. A remainder tile cut between CTAs is finished by the last CTA to arrive (global
// per-tile counter), summing the CTA partials in CTA order. Fixed partition + fixed orders = deterministic,
// and every token column is computed identically for any nt.
constexpr int G6_WARPS = 8;
constexpr int G6_RING = 4;

template <bool Q8>
__global__ void __launch_bounds__(G6_WARPS * 32, 2) k_gemv6(const uint8_t * __restrict__ W, int tiles, int cols,
                                                           const __half * __restrict__ x16, SegList segs, int nt,
                                                           int accumulate, SplitK sk) {
    using F = GemvFmt<Q8>;
    __shared__ float part[G6_RING][G6_WARPS][32][4];
    __shared__ int cnt[G6_RING], gen[G6_RING];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int nb = cols >> 8;                 // >= G6_WARPS for every matrix (checked on the host)
    const int G = gridDim.x, c = blockIdx.x;
    const int R = tiles / G, Tf = R * G;
    const long long Brem = (long long) (tiles - Tf) * nb;
    const long long A = c * Brem / G, B = (c + 1) * Brem / G;   // remainder blocks of this CTA
    const int tfirst = Tf + (int) (A / nb);                       // its first remainder tile
    if (threadIdx.x < G6_RING) { cnt[threadIdx.x] = 0; gen[threadIdx.x] = 0; }
    __syncthreads();

    const int n1 = (nb - warp + G6_WARPS - 1) / G6_WARPS;      // blocks per full tile for this warp
    const long long n2 = B - A > warp ? (B - A - warp + G6_WARPS - 1) / G6_WARPS : 0;
    const long long nitems = (long long) R * n1 + n2;
    auto item = [&](long long i, int & tile, int & ib, int & seg) {
        if (i < (long long) R * n1) {
            const int r = (int) (i / n1);
            tile = r * G + c;
            ib = warp + G6_WARPS * (int) (i % n1);
            seg = r;
        } else {
            const long long s = A + warp + G6_WARPS * (i - (long long) R * n1);
            tile = Tf + (int) (s / nb);
            ib = (int) (s % nb);
            seg = R + (tile - tfirst);
        }
    };
    auto startc = [&](long long cc) { return cc * Brem / G; };
    auto ownerc = [&](long long blk) { return ((blk + 1) * G - 1) / Brem; };

    // one warp's sum for (seg, tile) -> ring; the last contributor finishes the tile (or its CTA partial)
    auto deposit = [&](int seg, int tile, const float acc[4]) {
        const int slot = seg % G6_RING;
        while (*(volatile int *) &gen[slot] != seg / G6_RING) {}
        *(float4 *) &part[slot][warp][lane][0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        __threadfence_block();
        __syncwarp();
        long long s0 = 0, len = nb;
        bool whole = true;
        if (seg >= R) {
            const long long t0 = (long long) (tile - Tf) * nb;
            s0 = max(A, t0);
            len = min(B, t0 + nb) - s0;
            whole = s0 == t0 && len == nb;
        }
        const int expect = (int) min((long long) G6_WARPS, len);
        int last = 0;
        if (lane == 0) last = atomicAdd(&cnt[slot], 1) == expect - 1;
        last = __shfl_sync(0xffffffffu, last, 0);
        if (!last) return;
        __threadfence_block();
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < G6_WARPS; ++w) {
            if (seg >= R && len < G6_WARPS && ((w - (int) ((s0 - A) % G6_WARPS) + G6_WARPS) % G6_WARPS) >= len) continue;
            const float4 v = *(const float4 *) &part[slot][w][lane][0];
            s[0] += v.x; s[1] += v.y; s[2] += v.z; s[3] += v.w;
        }
        if (whole) {
            gemv_store_tile(s, tile, lane, segs, nt, accumulate);
        } else {
            const int pslot = tile == tfirst ? 0 : 1;
            *(float4 *) (sk.partial + ((size_t) c * 2 + pslot) * 128 + lane * 4) = make_float4(s[0], s[1], s[2], s[3]);
            __threadfence();
            __syncwarp();
            const long long t0 = (long long) (tile - Tf) * nb;
            const long long first = ownerc(t0), lastc = ownerc(t0 + nb - 1);
            // CTAs between first and lastc can have empty ranges (fewer remainder blocks than CTAs); they never
            // arrive and own no partial, so they are neither counted nor read
            int arrivals = 0;
            for (long long j = first; j <= lastc; ++j) arrivals += startc(j + 1) > startc(j);
            int lc = 0;
            if (lane == 0) lc = atomicAdd(sk.counter + tile, 1) == arrivals - 1;
            lc = __shfl_sync(0xffffffffu, lc, 0);
            if (lc) {
                float s2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (long long j = first; j <= lastc; ++j) {
                    if (startc(j + 1) == startc(j)) continue;
                    const int sl = tile == Tf + (int) (startc(j) / nb) ? 0 : 1;
                    const float4 v = __ldcg((const float4 *) (sk.partial + ((size_t) j * 2 + sl) * 128 + lane * 4));
                    s2[0] += v.x; s2[1] += v.y; s2[2] += v.z; s2[3] += v.w;
                }
                gemv_store_tile(s2, tile, lane, segs, nt, accumulate);
                if (lane == 0) sk.counter[tile] = 0;
            }
        }
        if (lane == 0) {
            cnt[slot] = 0;
            __threadfence_block();
            atomicAdd(&gen[slot], 1);
        }
        __syncwarp();
    };

    const __half * xg = x16 + (size_t) g * cols + 2 * q;
    typename F::Blk cur, nxt;
    int tile = 0, ib = 0, seg = 0;
    if (nitems > 0) {
        item(0, tile, ib, seg);
        F::fetch(W, (long long) tile * nb + ib, lane, g, cur);
    }
    pdl_trigger();
    pdl_wait();
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (long long i = 0; i < nitems; ++i) {
        int tile2 = -1, ib2 = 0, seg2 = -1;
        if (i + 1 < nitems) {
            item(i + 1, tile2, ib2, seg2);
            F::fetch(W, (long long) tile2 * nb + ib2, lane, g, nxt);
        }
        F::compute(cur, xg + ib * 256, acc);
        if (seg2 != seg) {
            deposit(seg, tile, acc);
            acc[0] = acc[1] = acc[2] = acc[3] = 0.0f;
        }
        cur = nxt;
        tile = tile2;
        ib = ib2;
        seg = seg2;
    }
}

// ---- prefill GEMM: NG groups of 8 tokens per weight read ----
// k_q6k_mma / k_q8_mma with the dequantized A fragments of each 16-element chunk multiplied into NG token groups
// before moving on. Warp -> block assignment, chunk order, scale application and the warp reduction order are
// exactly those of the 8-token kernels, so token t's column has the same rounding sequence as in an 8-token
// forward: a 64-token prefill leaves bit-identical state (outputs never depend on how a prompt was chunked).
template <int NG>
__global__ void __launch_bounds__(256) k_q6k_mma_n(const uint8_t * __restrict__ W, int rows, int cols,
                                                  const __half * __restrict__ x16, SegList segs, int nt, int accumulate) {
    constexpr int NW = 8;
    __shared__ float part[NW][32][4 * NG];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int tile = blockIdx.x;
    const int nb = cols >> 8;
    const __half * xg = x16 + (size_t) g * cols + 2 * q;
    float acc[NG][4];
#pragma unroll
    for (int j = 0; j < NG; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    struct Blk { uint4 v[6]; uint4 scg, scg8; __half dg, dg8; };
    auto fetch = [&](int ib, Blk & b) {
        const uint8_t * base = W + ((size_t) tile * nb + ib) * Q6T_BYTES;
        const uint4 * lp = (const uint4 *) base + lane;
#pragma unroll
        for (int i = 0; i < 6; ++i) b.v[i] = __ldg(lp + i * 32);
        b.scg = __ldg((const uint4 *) (base + 3072 + g * 16));
        b.scg8 = __ldg((const uint4 *) (base + 3072 + (g + 8) * 16));
        b.dg = __ldg((const __half *) (base + 3328) + g);
        b.dg8 = __ldg((const __half *) (base + 3328) + g + 8);
    };
    Blk cur, nxt;
    if (warp < nb) fetch(warp, cur);
    pdl_trigger();
    pdl_wait();
    for (int ib = warp; ib < nb; ib += NW) {
        if (ib + NW < nb) fetch(ib + NW, nxt);
        uint32_t w[24];
#pragma unroll
        for (int i = 0; i < 6; ++i) {
            w[4 * i + 0] = cur.v[i].x; w[4 * i + 1] = cur.v[i].y; w[4 * i + 2] = cur.v[i].z; w[4 * i + 3] = cur.v[i].w;
        }
        const float dg = __half2float(cur.dg);
        const float dg8 = __half2float(cur.dg8);
        const uint32_t sg[4] = {cur.scg.x, cur.scg.y, cur.scg.z, cur.scg.w};
        const uint32_t sg8[4] = {cur.scg8.x, cur.scg8.y, cur.scg8.z, cur.scg8.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int n = c >> 1, l0 = (c & 1) * 16;
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const uint32_t srcA = (jj & 1) ? w[3 * c + 1] : w[3 * c];
                const uint32_t srcB = (jj & 1) ? w[3 * (c + 4) + 1] : w[3 * (c + 4)];
                const uint32_t nibA = (jj >> 1) ? ((srcA >> 4) & 0x0F0F0F0Fu) : (srcA & 0x0F0F0F0Fu);
                const uint32_t nibB = (jj >> 1) ? ((srcB >> 4) & 0x0F0F0F0Fu) : (srcB & 0x0F0F0F0Fu);
                const uint32_t QA = nibA | (((w[3 * c + 2] >> (2 * jj)) & 0x03030303u) << 4);
                const uint32_t QB = nibB | (((w[3 * (c + 4) + 2] >> (2 * jj)) & 0x03030303u) << 4);
                uint32_t a[4];
                q6_to_half2(QA, a[0], a[2]);
                q6_to_half2(QB, a[1], a[3]);
                const int e0 = ib * 256 + n * 128 + 32 * jj + l0;
                const int si = n * 8 + l0 / 16 + 2 * jj;
                const float s0 = dg * (float) (int8_t) (sg[si >> 2] >> (8 * (si & 3)));
                const float s1 = dg8 * (float) (int8_t) (sg8[si >> 2] >> (8 * (si & 3)));
#pragma unroll
                for (int j = 0; j < NG; ++j) {
                    const __half * xb = xg + (size_t) j * 8 * cols + e0;
                    uint32_t b[2];
                    b[0] = __ldg((const uint32_t *) xb);
                    b[1] = __ldg((const uint32_t *) (xb + 8));
                    float dd[4];
                    mma_16816(dd, a, b);
                    acc[j][0] += s0 * dd[0];
                    acc[j][1] += s0 * dd[1];
                    acc[j][2] += s1 * dd[2];
                    acc[j][3] += s1 * dd[3];
                }
            }
        }
        cur = nxt;
    }
#pragma unroll
    for (int j = 0; j < NG; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) part[warp][lane][4 * j + i] = acc[j][i];
    __syncthreads();
    for (int o = threadIdx.x; o < 128 * NG; o += 256) {
        const int j = o >> 7, L = (o & 127) >> 2, i = o & 3;
        float s = 0.0f;
#pragma unroll
        for (int w2 = 0; w2 < NW; ++w2) s += part[w2][L][4 * j + i];
        const int row = tile * 16 + (L >> 2) + 8 * (i >> 1);
        const int t = 8 * j + 2 * (L & 3) + (i & 1);
        if (t < nt) {
            int sgi = 0;
            while (sgi < segs.n - 1 && row >= segs.row_end[sgi]) ++sgi;
            const int r0 = sgi ? segs.row_end[sgi - 1] : 0;
            float * dst = segs.y[sgi] + (size_t) t * (segs.row_end[sgi] - r0) + (row - r0);
            *dst = accumulate ? *dst + s : s;
        }
    }
}

template <int NG>
__global__ void __launch_bounds__(256) k_q8_mma_n(const uint8_t * __restrict__ W, int rows, int cols,
                                                 const __half * __restrict__ x16, SegList segs, int nt, int accumulate) {
    constexpr int NW = 8;
    __shared__ float part[NW][32][4 * NG];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int tile = blockIdx.x;
    const int nsb = cols >> 8;
    const __half * xg = x16 + (size_t) g * cols + 2 * q;
    float acc[NG][4];
#pragma unroll
    for (int j = 0; j < NG; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    struct Blk { uint4 v[8]; uint4 sg, sg8; };
    auto fetch = [&](int sb, Blk & b) {
        const uint8_t * base = W + ((size_t) tile * nsb + sb) * Q8T_BYTES;
        const uint4 * lp = (const uint4 *) base + lane;
#pragma unroll
        for (int i = 0; i < 8; ++i) b.v[i] = __ldg(lp + i * 32);
        b.sg = __ldg((const uint4 *) (base + 4096 + g * 16));
        b.sg8 = __ldg((const uint4 *) (base + 4096 + (g + 8) * 16));
    };
    Blk cur, nxt;
    if (warp < nsb) fetch(warp, cur);
    pdl_trigger();
    pdl_wait();
    for (int sb = warp; sb < nsb; sb += NW) {
        if (sb + NW < nsb) fetch(sb + NW, nxt);
        uint32_t w[32];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[4 * i + 0] = cur.v[i].x; w[4 * i + 1] = cur.v[i].y; w[4 * i + 2] = cur.v[i].z; w[4 * i + 3] = cur.v[i].w;
        }
        const __half * dg = (const __half *) &cur.sg;
        const __half * dg8 = (const __half *) &cur.sg8;
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            uint32_t a[4];
            q8_to_half2(w[2 * c], a[0], a[2]);
            q8_to_half2(w[2 * c + 1], a[1], a[3]);
            const float s0 = __half2float(dg[c >> 1]);
            const float s1 = __half2float(dg8[c >> 1]);
#pragma unroll
            for (int j = 0; j < NG; ++j) {
                const __half * xb = xg + (size_t) j * 8 * cols + sb * 256 + c * 16;
                uint32_t b[2];
                b[0] = __ldg((const uint32_t *) xb);
                b[1] = __ldg((const uint32_t *) (xb + 8));
                float dd[4];
                mma_16816(dd, a, b);
                acc[j][0] += s0 * dd[0];
                acc[j][1] += s0 * dd[1];
                acc[j][2] += s1 * dd[2];
                acc[j][3] += s1 * dd[3];
            }
        }
        cur = nxt;
    }
#pragma unroll
    for (int j = 0; j < NG; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) part[warp][lane][4 * j + i] = acc[j][i];
    __syncthreads();
    for (int o = threadIdx.x; o < 128 * NG; o += 256) {
        const int j = o >> 7, L = (o & 127) >> 2, i = o & 3;
        float s = 0.0f;
#pragma unroll
        for (int w2 = 0; w2 < NW; ++w2) s += part[w2][L][4 * j + i];
        const int row = tile * 16 + (L >> 2) + 8 * (i >> 1);
        const int t = 8 * j + 2 * (L & 3) + (i & 1);
        if (t < nt) {
            int sgi = 0;
            while (sgi < segs.n - 1 && row >= segs.row_end[sgi]) ++sgi;
            const int r0 = sgi ? segs.row_end[sgi - 1] : 0;
            float * dst = segs.y[sgi] + (size_t) t * (segs.row_end[sgi] - r0) + (row - r0);
            *dst = accumulate ? *dst + s : s;
        }
    }
}

// ---- prefill GEMM v2: residue CTAs with the activations shared through shared memory ----
// k_q6k_mma_n re-read each B fragment from L2 per warp: 32 KB of activations per 3.4 KB weight block (216 GB per
// 64-token forward) and ran at ~47 TFLOPS. In the 8-token kernel a 16-row tile's result is the sum, in warp order,
// of 8 partials: partial w accumulates blocks w, w+8, ... in order. Here CTA (group, w) computes partial w for 8
// adjacent row tiles (one per warp), so all 8 warps use the same activation block, staged once in shared memory
// (cp.async, double-buffered). k_gemm_reduce adds the 8 partials 0 + p0 + ... + p7 in that order, so every
// output is bit-identical to the 8-token kernel's.
constexpr int GR_BSTRIDE = 256 + 8;   // halves per staged activation row: lanes (g, q) hit banks 4g + q

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

// partial[w][t][row] for t < 8*NG.   grid (ceil(tiles / 8), 8), 256 threads, dynamic smem 2 * 8*NG * GR_BSTRIDE halves
template <int NG>
__global__ void __launch_bounds__(256) k_q6k_gemm_r(const uint8_t * __restrict__ W, int rows, int cols,
                                                   const __half * __restrict__ x16, float * __restrict__ partial) {
    extern __shared__ __align__(16) __half gr_smem[];
    constexpr int NT = 8 * NG;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, q = lane & 3;
    const int tiles = rows / 16;
    const int tile = blockIdx.x * 8 + warp;
    const bool live = tile < tiles;
    const int w = blockIdx.y;
    const int nb = cols >> 8;
    float acc[NG][4];
#pragma unroll
    for (int j = 0; j < NG; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    auto stage = [&](int ib, int buf) {   // activations [NT tokens][256 cols of block ib] -> smem
        __half * dst = gr_smem + (size_t) buf * NT * GR_BSTRIDE;
        for (int i = threadIdx.x; i < NT * 32; i += 256) {
            const int t = i >> 5, v = i & 31;
            cp_async16(dst + t * GR_BSTRIDE + v * 8, x16 + (size_t) t * cols + ib * 256 + v * 8);
        }
        cp_async_commit();
    };
    struct Blk { uint4 v[6]; uint4 scg, scg8; __half dg, dg8; };
    auto fetch = [&](int ib, Blk & b) {
        const uint8_t * base = W + ((size_t) tile * nb + ib) * Q6T_BYTES;
        const uint4 * lp = (const uint4 *) base + lane;
#pragma unroll
        for (int i = 0; i < 6; ++i) b.v[i] = __ldg(lp + i * 32);
        b.scg = __ldg((const uint4 *) (base + 3072 + g * 16));
        b.scg8 = __ldg((const uint4 *) (base + 3072 + (g + 8) * 16));
        b.dg = __ldg((const __half *) (base + 3328) + g);
        b.dg8 = __ldg((const __half *) (base + 3328) + g + 8);
    };
    Blk cur, nxt;
    if (live && w < nb) fetch(w, cur);
    pdl_trigger();
    pdl_wait();
    if (w < nb) stage(w, 0);
    int buf = 0;
    for (int ib = w; ib < nb; ib += 8) {
        const bool more = ib + 8 < nb;
        if (more) {
            stage(ib + 8, buf ^ 1);
            if (live) fetch(ib + 8, nxt);
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();
        if (live) {
            const __half * xs = gr_smem + (size_t) buf * NT * GR_BSTRIDE + g * GR_BSTRIDE + 2 * q;
            uint32_t wv[24];
#pragma unroll
            for (int i = 0; i < 6; ++i) {
                wv[4 * i + 0] = cur.v[i].x; wv[4 * i + 1] = cur.v[i].y; wv[4 * i + 2] = cur.v[i].z; wv[4 * i + 3] = cur.v[i].w;
            }
            const float dg = __half2float(cur.dg);
            const float dg8 = __half2float(cur.dg8);
            const uint32_t sg[4] = {cur.scg.x, cur.scg.y, cur.scg.z, cur.scg.w};
            const uint32_t sg8[4] = {cur.scg8.x, cur.scg8.y, cur.scg8.z, cur.scg8.w};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int n = c >> 1, l0 = (c & 1) * 16;
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const uint32_t srcA = (jj & 1) ? wv[3 * c + 1] : wv[3 * c];
                    const uint32_t srcB = (jj & 1) ? wv[3 * (c + 4) + 1] : wv[3 * (c + 4)];
                    const uint32_t nibA = (jj >> 1) ? ((srcA >> 4) & 0x0F0F0F0Fu) : (srcA & 0x0F0F0F0Fu);
                    const uint32_t nibB = (jj >> 1) ? ((srcB >> 4) & 0x0F0F0F0Fu) : (srcB & 0x0F0F0F0Fu);
                    const uint32_t QA = nibA | (((wv[3 * c + 2] >> (2 * jj)) & 0x03030303u) << 4);
                    const uint32_t QB = nibB | (((wv[3 * (c + 4) + 2] >> (2 * jj)) & 0x03030303u) << 4);
                    uint32_t a[4];
                    q6_to_half2(QA, a[0], a[2]);
                    q6_to_half2(QB, a[1], a[3]);
                    const int e0 = n * 128 + 32 * jj + l0;
                    const int si = n * 8 + l0 / 16 + 2 * jj;
                    const float s0 = dg * (float) (int8_t) (sg[si >> 2] >> (8 * (si & 3)));
                    const float s1 = dg8 * (float) (int8_t) (sg8[si >> 2] >> (8 * (si & 3)));
#pragma unroll
                    for (int j = 0; j < NG; ++j) {
                        const __half * xb = xs + j * 8 * GR_BSTRIDE + e0;
                        uint32_t b[2];
                        b[0] = *(const uint32_t *) xb;
                        b[1] = *(const uint32_t *) (xb + 8);
                        float dd[4];
                        mma_16816(dd, a, b);
                        acc[j][0] += s0 * dd[0];
                        acc[j][1] += s0 * dd[1];
                        acc[j][2] += s1 * dd[2];
                        acc[j][3] += s1 * dd[3];
                    }
                }
            }
            cur = nxt;
        }
        __syncthreads();   // everyone is done with this buffer before it is refilled
        buf ^= 1;
    }
    if (!live) return;
    // partial[w][t][row]
    float * pw = partial + (size_t) w * NT * rows;
#pragma unroll
    for (int j = 0; j < NG; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = tile * 16 + g + 8 * (i >> 1);
            const int t = 8 * j + 2 * q + (i & 1);
            pw[(size_t) t * rows + row] = acc[j][i];
        }
}

// out = 0 + p0 + ... + p7 in that order (the 8-token kernel's warp-reduction order), then the SegList epilogue.
// grid ceil(rows * nt / 256), 256 threads
__global__ void k_gemm_reduce(const float * __restrict__ partial, int rows, int ntp, int nt, SegList segs, int accumulate) {
    PDL_ENTRY();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * nt) return;
    const int t = idx / rows, row = idx % rows;
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += partial[((size_t) w * ntp + t) * rows + row];
    int sgi = 0;
    while (sgi < segs.n - 1 && row >= segs.row_end[sgi]) ++sgi;
    const int r0 = sgi ? segs.row_end[sgi - 1] : 0;
    float * dst = segs.y[sgi] + (size_t) t * (segs.row_end[sgi] - r0) + (row - r0);
    *dst = accumulate ? *dst + s : s;
}

// fp16 GEMV inputs below write all 8 token rows; rows t >= nt are zero so they add nothing to the mma.

// x16[t] = half(rmsnorm(x[t]) * w).   grid 8, 1024 threads
__global__ void k_rmsnorm_h(const float * __restrict__ x, const float * __restrict__ w, __half * __restrict__ x16,
                            int n, int nt) {
    PDL_ENTRY();
    __shared__ float red[32];
    const int t = blockIdx.x;
    __half * o = x16 + (size_t) t * n;
    if (t >= nt) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = __float2half(0.0f);
        return;
    }
    const float * xr = x + (size_t) t * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum<32>(s, red);
    const float sc = rsqrtf(s / n + EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = __float2half(xr[i] * sc * w[i]);
}

// x16[t][i] = half(silu(g) * u), n = FF, for the padded rows (rows t >= nt zero)
__global__ void k_swiglu_h(const float * __restrict__ g, const float * __restrict__ u, __half * __restrict__ x16,
                           int n, int nt, int rows) {
    PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    x16[i] = i < nt * n ? __float2half(silu(g[i]) * u[i]) : __float2half(0.0f);
}

// x (nt x n fp32) -> x16 (8 x n fp16), rows nt..7 zero
__global__ void k_to_half8(const float * __restrict__ x, __half * __restrict__ x16, int n, int nt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 8 * n) return;
    x16[i] = i < nt * n ? __float2half(x[i]) : __float2half(0.0f);
}

// --- gated delta net (linear attention) ---

// Recurrent state (conv taps, DeltaNet S) is never written by a forward. The forward reads the committed
// state and leaves per-layer history (qkv rows, conv outputs, g, beta); after the accept decision the
// commit kernels replay exactly the first n_commit tokens. Rejected drafts therefore never touch state,
// and plain decoding (commit everything) runs the same code as speculative decoding.

// causal depthwise conv over [state(3), x(nt)] then silu, from the committed state (read-only)
__global__ void k_gdn_conv(const float * __restrict__ qkv, const float * __restrict__ state, const float * __restrict__ w,
                           float * __restrict__ out, int nt) {
    PDL_ENTRY();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CONVCH) return;
    float h0 = state[c], h1 = state[CONVCH + c], h2 = state[2 * CONVCH + c];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < nt; ++t) {
        const float xin = qkv[(size_t) t * CONVCH + c];
        const float s = h0 * w0 + h1 * w1 + h2 * w2 + xin * w3;
        out[(size_t) t * CONVCH + c] = silu(s);
        h0 = h1; h1 = h2; h2 = xin;
    }
}

// conv taps after the first n_commit tokens of the block.   grid (CONVCH/256, N_RECR)
__global__ void k_conv_commit(float * __restrict__ conv_state, const float * __restrict__ qkv_hist,
                              const int * __restrict__ d_ncommit) {
    PDL_ENTRY();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CONVCH) return;
    float * st = conv_state + (size_t) blockIdx.y * 3 * CONVCH;
    const float * qh = qkv_hist + (size_t) blockIdx.y * MAXP * CONVCH;
    float h0 = st[c], h1 = st[CONVCH + c], h2 = st[2 * CONVCH + c];
    const int n = *d_ncommit;
    for (int t = 0; t < n; ++t) {
        h0 = h1; h1 = h2; h2 = qh[(size_t) t * CONVCH + c];
    }
    st[c] = h0; st[CONVCH + c] = h1; st[2 * CONVCH + c] = h2;
}

// l2-normalize the 16 q heads and 16 k heads of each token in place.   grid (nt, 32), 128 threads
__global__ void k_l2norm_qk(float * __restrict__ co) {
    PDL_ENTRY();
    __shared__ float red[32];
    float * p = co + (size_t) blockIdx.x * CONVCH + blockIdx.y * DS;
    const float v = p[threadIdx.x];
    const float ss = block_sum<4>(v * v, red);
    p[threadIdx.x] = v * rsqrtf(fmaxf(ss, EPS * EPS));
}

// beta = sigmoid(b), g = softplus(a + dt_bias) * ssm_a      n = nt * NVH
__global__ void k_gdn_gates(const float * __restrict__ a_raw, const float * __restrict__ b_raw,
                            const float * __restrict__ dt_bias, const float * __restrict__ ssm_a,
                            float * __restrict__ g, float * __restrict__ beta, int n) {
    PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % NVH;
    beta[i] = 1.0f / (1.0f + expf(-b_raw[i]));
    const float a = a_raw[i] + dt_bias[h];
    const float sp = a > 20.0f ? a : logf(1.0f + expf(a));
    g[i] = sp * ssm_a[h];
}

// One delta-rule step for value column col: S = g*S + k*delta, delta = beta*(v - g*(S_old^T k)).
// Shared by the forward and the commit replay; explicit _rn intrinsics pin the rounding so both produce
// bit-identical states (the compiler may not contract them into different FMAs in the two kernels).
template <bool WANT_OUT>
__device__ __forceinline__ float gdn_step(float (&s)[DS], const float * ks, const float * qs, float gv, float bv, float vv) {
    float kv = 0.0f;
#pragma unroll
    for (int i = 0; i < DS; ++i) kv = __fmaf_rn(s[i], ks[i], kv);
    const float delta = __fmul_rn(__fsub_rn(vv, __fmul_rn(gv, kv)), bv);
    float o = 0.0f;
#pragma unroll
    for (int i = 0; i < DS; ++i) {
        s[i] = __fmaf_rn(ks[i], delta, __fmul_rn(gv, s[i]));
        if (WANT_OUT) o = __fmaf_rn(s[i], qs[i], o);
    }
    return o;
}

constexpr float GDN_SCALE = 0.08838834764831845f;  // 1/sqrt(128)

// delta rule per value head over the block, from the committed state (read-only).
// grid NVH, 128 threads: thread = value column, holds S[:, col] in registers.
// S layout [head][i(k)][col(v)]; value head h reads key head h % NKH (ggml tiles k/q, see gated_delta_net.cu).
__global__ void __launch_bounds__(128) k_gdn_recur(const float * __restrict__ co, const float * __restrict__ g,
                                                   const float * __restrict__ beta, const float * __restrict__ S,
                                                   float * __restrict__ out, int nt) {
    PDL_ENTRY();
    const int h = blockIdx.x, col = threadIdx.x, kh = h % NKH;
    __shared__ float ks[DS], qs[DS];
    float s[DS];
    const float * Sh = S + (size_t) h * DS * DS;
#pragma unroll
    for (int i = 0; i < DS; ++i) s[i] = Sh[i * DS + col];
    for (int t = 0; t < nt; ++t) {
        const float * row = co + (size_t) t * CONVCH;
        __syncthreads();
        qs[col] = row[kh * DS + col];
        ks[col] = row[NKH * DS + kh * DS + col];
        __syncthreads();
        const float o = gdn_step<true>(s, ks, qs, expf(g[t * NVH + h]), beta[t * NVH + h], row[2 * NKH * DS + h * DS + col]);
        out[(size_t) t * DINNER + h * DS + col] = o * GDN_SCALE;
    }
}

// replay the first n_commit tokens of the block into the committed state.   grid (NVH, N_RECR), 128 threads
__global__ void __launch_bounds__(128) k_gdn_commit(const float * __restrict__ co_hist, const float * __restrict__ g_hist,
                                                    const float * __restrict__ beta_hist, float * __restrict__ S_all,
                                                    const int * __restrict__ d_ncommit) {
    PDL_ENTRY();
    const int h = blockIdx.x, layer = blockIdx.y, col = threadIdx.x, kh = h % NKH;
    __shared__ float ks[DS];
    const int n = *d_ncommit;
    float * Sh = S_all + ((size_t) layer * NVH + h) * DS * DS;
    const float * co = co_hist + (size_t) layer * MAXP * CONVCH;
    const float * g = g_hist + (size_t) layer * MAXP * NVH;
    const float * beta = beta_hist + (size_t) layer * MAXP * NVH;
    float s[DS];
#pragma unroll
    for (int i = 0; i < DS; ++i) s[i] = Sh[i * DS + col];
    for (int t = 0; t < n; ++t) {
        const float * row = co + (size_t) t * CONVCH;
        __syncthreads();
        ks[col] = row[NKH * DS + kh * DS + col];
        __syncthreads();
        gdn_step<false>(s, ks, nullptr, expf(g[t * NVH + h]), beta[t * NVH + h], row[2 * NKH * DS + h * DS + col]);
    }
#pragma unroll
    for (int i = 0; i < DS; ++i) Sh[i * DS + col] = s[i];
}

// --- DFlash2 drafter (math from llama.cpp src/models/dflash.cpp, common/speculative.cpp) ---

constexpr int DF_L = 5;                // drafter layers
constexpr int DF_NH = 32, DF_NKV = 8, DF_HD = 128;
constexpr int DF_QD = DF_NH * DF_HD;   // 4096
constexpr int DF_KD = DF_NKV * DF_HD;  // 1024
constexpr int DF_NG = D / 16;          // 320 conv groups of 16 channels
constexpr int DF_PROJ = 2 * 2 * DF_NG; // 1280 dynamic conv coefficients: [side][tap][group]
constexpr int DF_SEL_R = 256, DF_TOPK = 16, DF_SWA = 2048;
constexpr int DF_NFEAT = 5, DF_FEAT = DF_NFEAT * D;  // features: inputs of target layers 6, 20, 34, 48, 62
constexpr int DF_TARGET_LAYERS[DF_NFEAT] = {6, 20, 34, 48, 62};
constexpr int MASK_TOKEN = 248070;     // llama_vocab_mask() of the Qwen3.8 vocab, as llama.cpp logs it
constexpr int TOPK_PARTS = 64;

// feat16[t][k*D + i] = half(x[t][i]) for the k-th extracted target layer, padded rows; rows >= nt zero
__global__ void k_feat_capture(const float * __restrict__ x, __half * __restrict__ feat16, int k, int nt, int rows) {
    PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * D) return;
    const int t = i / D, c = i % D;
    feat16[(size_t) t * DF_FEAT + k * D + c] = t < nt ? __float2half(x[i]) : __float2half(0.0f);
}

// x[t] = dequantized Q6_K row ids[t] of the token embedding (device copy of token_embd).  grid nt, 256 threads
__global__ void k_embed_q6k(const int * __restrict__ ids, const uint8_t * __restrict__ tok, float * __restrict__ x) {
    PDL_ENTRY();
    const int t = blockIdx.x;
    const uint8_t * row = tok + (size_t) ids[t] * (D / 256) * Q6K_BYTES;
    for (int e = threadIdx.x; e < D; e += blockDim.x) {
        const uint8_t * blk = row + (e >> 8) * Q6K_BYTES;
        const int r = e & 255, n = r >> 7, j = (r >> 5) & 3, l = r & 31;
        const uint8_t qlb = blk[n * 64 + l + 32 * (j & 1)];
        const int nib = (j >> 1) ? (qlb >> 4) : (qlb & 0xF);
        const int q = (nib | (((blk[128 + n * 32 + l] >> (2 * j)) & 3) << 4)) - 32;
        const float d = __half2float(*(const __half *) (blk + 208));
        const int8_t sc = (int8_t) blk[192 + n * 8 + l / 16 + 2 * j];
        x[(size_t) t * D + e] = d * sc * q;   // dequantize_row_q6_K: d * sc * q
    }
}

// y32 / y16 = rmsnorm(x) * w for all 8 rows (drafter blocks are always full).  grid 8, 1024 threads
__global__ void k_rmsnorm_fh(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ y32,
                             __half * __restrict__ y16, int n) {
    PDL_ENTRY();
    __shared__ float red[32];
    const float * xr = x + (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum<32>(s, red);
    const float sc = rsqrtf(s / n + EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] * sc * w[i];
        if (y32) y32[(size_t) blockIdx.x * n + i] = v;
        y16[(size_t) blockIdx.x * n + i] = __float2half(v);
    }
}

// DFlash2 dynamic 2-tap block conv: out[t] = (dyn_tap0 + base_tap0) * h[t] + (dyn_tap1 + base_tap1) * h[t-1],
// h[-1] = 0, coefficients per (token, 16-channel group). Writes fp16 (next GEMV input) or adds into res.
__global__ void k_df_conv(const float * __restrict__ h, const float * __restrict__ dyn, const float * __restrict__ base,
                          int side, __half * __restrict__ out16, float * __restrict__ res) {
    PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 8 * D) return;
    const int t = i / D, e = i % D, g = e / 16;
    const float * dt = dyn + (size_t) t * DF_PROJ;
    const float w0 = __fadd_rn(dt[g + DF_NG * 0 + 2 * DF_NG * side], base[e + D * 0 + 2 * D * side]);
    const float w1 = __fadd_rn(dt[g + DF_NG * 1 + 2 * DF_NG * side], base[e + D * 1 + 2 * D * side]);
    const float v0 = h[i];
    const float v1 = t > 0 ? h[i - D] : 0.0f;
    const float r = __fadd_rn(__fmul_rn(w0, v0), __fmul_rn(w1, v1));
    if (res) res[i] = __fadd_rn(r, res[i]);
    else out16[i] = __float2half(r);
}

// per-head rmsnorm + neox rope over all 128 dims (MROPE with sections [64,0,0,0] == plain neox for text).
// q heads -> qout; k/v -> the drafter cache at pos0 + t for t < n (n from d_n if given, else 8).
// grid (8, DF_NH + DF_NKV), 128 threads
__global__ void k_df_qk(const float * __restrict__ qf, const float * __restrict__ kf, const float * __restrict__ vf,
                        const float * __restrict__ qn_w, const float * __restrict__ kn_w, float * __restrict__ qout,
                        __half * __restrict__ Kc, __half * __restrict__ Vc, const int * __restrict__ d_pos0,
                        const int * __restrict__ d_n) {
    PDL_ENTRY();
    __shared__ float red[32];
    __shared__ float buf[DF_HD];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const bool is_q = hh < DF_NH;
    if (!is_q && d_n && t >= *d_n) return;   // inject: only committed rows
    if (is_q && !qout) return;
    const int pos = *d_pos0 + t;
    const float * src = is_q ? qf + (size_t) t * DF_QD + hh * DF_HD : kf + (size_t) t * DF_KD + (hh - DF_NH) * DF_HD;
    const float v = src[d];
    const float ss = block_sum<4>(v * v, red);
    const float xn = v * rsqrtf(ss / DF_HD + EPS) * (is_q ? qn_w[d] : kn_w[d]);
    buf[d] = xn;
    __syncthreads();
    const int p = d < DF_HD / 2 ? d : d - DF_HD / 2;
    const float theta = (float) pos * powf(powf(ROPE_BASE, -2.0f / DF_HD), (float) p);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = buf[p], x1 = buf[p + DF_HD / 2];
    const float r = d < DF_HD / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
    if (is_q) {
        qout[(size_t) t * DF_QD + hh * DF_HD + d] = r;
    } else {
        const int kh = hh - DF_NH;
        Kc[((size_t) pos * DF_NKV + kh) * DF_HD + d] = __float2half(r);
        Vc[((size_t) pos * DF_NKV + kh) * DF_HD + d] = __float2half(vf[(size_t) t * DF_KD + kh * DF_HD + d]);
    }
}

// non-causal block attention: query t (pos P+t) over cache positions [max(0, P+t-2047), P+7].
// grid (8, DF_NH), 128 threads, dynamic smem (DF_HD + max_ctx + 8) floats.  Output fp16 for wo.
__global__ void k_df_attn(const float * __restrict__ q, const __half * __restrict__ Kc, const __half * __restrict__ Vc,
                          __half * __restrict__ out16, const int * __restrict__ d_pos0) {
    PDL_ENTRY();
    extern __shared__ float sm[];
    __shared__ float red[32];
    float * qs = sm;
    float * sc = sm + DF_HD;
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int P = *d_pos0;
    const int lo = max(0, P + t - (DF_SWA - 1)), hi = P + 8;   // keys in [lo, hi)
    const int n = hi - lo;
    const int kvh = h / (DF_NH / DF_NKV);
    qs[tid] = q[(size_t) t * DF_QD + h * DF_HD + tid];
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    const float scale = rsqrtf((float) DF_HD);
    for (int j = warp; j < n; j += 4) {
        const __half * kr = Kc + ((size_t) (lo + j) * DF_NKV + kvh) * DF_HD;
        float a = 0.0f;
        for (int dd = lane; dd < DF_HD; dd += 32) a += qs[dd] * __half2float(kr[dd]);
        a = warp_sum(a);
        if (lane == 0) sc[j] = a * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = tid; j < n; j += 128) m = fmaxf(m, sc[j]);
    m = block_max<4>(m, red);
    float s = 0.0f;
    for (int j = tid; j < n; j += 128) {
        const float e = expf(sc[j] - m);
        sc[j] = e;
        s += e;
    }
    s = block_sum<4>(s, red);
    __syncthreads();
    float o = 0.0f;
    for (int j = 0; j < n; ++j) o += sc[j] * __half2float(Vc[((size_t) (lo + j) * DF_NKV + kvh) * DF_HD + tid]);
    out16[(size_t) t * DF_QD + h * DF_HD + tid] = __float2half(o / s);
}

// x16[t] = half(silu(g) * u) for all 8 rows
__global__ void k_swiglu8_h(const float * __restrict__ g, const float * __restrict__ u, __half * __restrict__ x16, int n) {
    PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 8 * n) x16[i] = __float2half(silu(g[i]) * u[i]);
}

// top-16 of a logits row: stage 1 per (part, row) keeps each warp's best 16, then the block's best 16.
// Order: larger value first, ties to the smaller token id.   grid (TOPK_PARTS, 8), 256 threads
__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

__global__ void k_topk_part(const float * __restrict__ logits, float * __restrict__ pv, int * __restrict__ pi) {
    PDL_ENTRY();
    __shared__ float wv[8][DF_TOPK];
    __shared__ int wi[8][DF_TOPK];
    const int row = blockIdx.y, part = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int per = (VOCAB + TOPK_PARTS - 1) / TOPK_PARTS;
    const int beg = part * per, end = min(VOCAB, beg + per);
    const float * lr = logits + (size_t) row * VOCAB;
    constexpr int PER_LANE = 16;   // per = 3880 -> 3880 / 256 threads < 16
    float v[PER_LANE];
    int id[PER_LANE];
#pragma unroll
    for (int k = 0; k < PER_LANE; ++k) {
        const int i = beg + threadIdx.x + k * 256;
        v[k] = i < end ? lr[i] : -INFINITY;
        id[k] = i < end ? i : INT_MAX;
    }
    for (int r = 0; r < DF_TOPK; ++r) {
        float bv = -INFINITY;
        int bi = INT_MAX, bk = -1;
#pragma unroll
        for (int k = 0; k < PER_LANE; ++k)
            if (better(v[k], id[k], bv, bi)) { bv = v[k]; bi = id[k]; bk = k; }
        float wvv = bv;
        int wii = bi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, wvv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, wii, o);
            if (better(ov, oi, wvv, wii)) { wvv = ov; wii = oi; }
        }
        if (bi == wii && bk >= 0) {
#pragma unroll
            for (int k = 0; k < PER_LANE; ++k)
                if (k == bk) { v[k] = -INFINITY; id[k] = INT_MAX; }
        }
        if (lane == 0) { wv[warp][r] = wvv; wi[warp][r] = wii; }
    }
    __syncthreads();
    if (warp == 0) {
        float cv[4];
        int ci[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {   // 8 warps x 16 = 128 candidates, 4 per lane
            const int c = lane + 32 * k;
            cv[k] = wv[c / DF_TOPK][c % DF_TOPK];
            ci[k] = wi[c / DF_TOPK][c % DF_TOPK];
        }
        for (int r = 0; r < DF_TOPK; ++r) {
            float bv = -INFINITY;
            int bi = INT_MAX, bk = -1;
#pragma unroll
            for (int k = 0; k < 4; ++k)
                if (better(cv[k], ci[k], bv, bi)) { bv = cv[k]; bi = ci[k]; bk = k; }
            float wvv = bv;
            int wii = bi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, wvv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, wii, o);
                if (better(ov, oi, wvv, wii)) { wvv = ov; wii = oi; }
            }
            if (bi == wii && bk >= 0) {
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    if (k == bk) { cv[k] = -INFINITY; ci[k] = INT_MAX; }
            }
            if (lane == 0) {
                pv[((size_t) row * TOPK_PARTS + part) * DF_TOPK + r] = wvv;
                pi[((size_t) row * TOPK_PARTS + part) * DF_TOPK + r] = wii;
            }
        }
    }
}

// stage 2: merge TOPK_PARTS x 16 candidates per row into the sorted top-16.   grid 8, 32 threads
__global__ void k_topk_final(const float * __restrict__ pv, const int * __restrict__ pi, int * __restrict__ cand,
                             float * __restrict__ unary) {
    PDL_ENTRY();
    const int row = blockIdx.x, lane = threadIdx.x;
    constexpr int PER_LANE = TOPK_PARTS * DF_TOPK / 32;   // 32
    float v[PER_LANE];
    int id[PER_LANE];
#pragma unroll
    for (int k = 0; k < PER_LANE; ++k) {
        v[k] = pv[(size_t) row * TOPK_PARTS * DF_TOPK + lane + 32 * k];
        id[k] = pi[(size_t) row * TOPK_PARTS * DF_TOPK + lane + 32 * k];
    }
    for (int r = 0; r < DF_TOPK; ++r) {
        float bv = -INFINITY;
        int bi = INT_MAX, bk = -1;
#pragma unroll
        for (int k = 0; k < PER_LANE; ++k)
            if (better(v[k], id[k], bv, bi)) { bv = v[k]; bi = id[k]; bk = k; }
        float wvv = bv;
        int wii = bi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, wvv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, wii, o);
            if (better(ov, oi, wvv, wii)) { wvv = ov; wii = oi; }
        }
        if (bi == wii && bk >= 0) {
#pragma unroll
            for (int k = 0; k < PER_LANE; ++k)
                if (k == bk) { v[k] = -INFINITY; id[k] = INT_MAX; }
        }
        if (lane == 0) {
            cand[row * DF_TOPK + r] = wii;
            unary[row * DF_TOPK + r] = wvv;
        }
    }
}

// one row (DF_SEL_R = 256 values) of a raw Q8_0 table [VOCAB][256]: 8 blocks of 34 bytes
__device__ __forceinline__ float q8_row_elem(const uint8_t * table, int id, int e) {
    const uint8_t * blk = table + ((size_t) id * (DF_SEL_R / 32) + e / 32) * Q80_BYTES;
    return __half2float(*(const __half *) blk) * (float) (int8_t) blk[2 + e % 32];
}

// lattice scores. block (i, p), p = 1..7: predecessor = anchor (p == 1) or cand[p-1][i];
// score[p][i][j] = next[cand[p][j]] . (prev[pred] * hid[p]) + unary[p][j].   grid (16, 8), 256 threads
__global__ void k_df_scores(const uint8_t * __restrict__ sel_prev, const uint8_t * __restrict__ sel_next,
                            const float * __restrict__ hid, const int * __restrict__ cand, const float * __restrict__ unary,
                            const int * __restrict__ d_anchor, float * __restrict__ score) {
    PDL_ENTRY();
    __shared__ float cond[DF_SEL_R];
    const int i = blockIdx.x, p = blockIdx.y, tid = threadIdx.x;
    if (p == 0) return;
    const int pred = p == 1 ? *d_anchor : cand[(p - 1) * DF_TOPK + i];
    cond[tid] = q8_row_elem(sel_prev, pred, tid) * hid[p * DF_SEL_R + tid];
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    for (int j = warp; j < DF_TOPK; j += 8) {
        const int nid = cand[p * DF_TOPK + j];
        float a = 0.0f;
        for (int r = lane; r < DF_SEL_R; r += 32) a += q8_row_elem(sel_next, nid, r) * cond[r];
        a = warp_sum(a);
        if (lane == 0) score[((size_t) p * DF_TOPK + i) * DF_TOPK + j] = a + unary[p * DF_TOPK + j];
    }
}

// greedy walk through the lattice; writes the verify block [anchor, d1..d7] and the drafts
__global__ void k_df_walk(const float * __restrict__ score, const int * __restrict__ cand, const int * __restrict__ d_anchor,
                          int * __restrict__ block_tokens) {
    PDL_ENTRY();
    if (threadIdx.x) return;
    block_tokens[0] = *d_anchor;
    int pred = 0;
    for (int p = 1; p < 8; ++p) {
        const float * s = score + ((size_t) p * DF_TOPK + pred) * DF_TOPK;
        int best = 0;
        for (int j = 1; j < DF_TOPK; ++j)
            if (s[j] > s[best]) best = j;   // std::max_element: first maximum
        block_tokens[p] = cand[p * DF_TOPK + best];
        pred = best;
    }
}

__global__ void k_df_noise_tokens(const int * __restrict__ d_anchor, int * __restrict__ tok) {
    PDL_ENTRY();
    const int t = threadIdx.x;
    if (t < 8) tok[t] = t == 0 ? *d_anchor : MASK_TOKEN;
}

// after accept: the bonus token becomes the next anchor
__global__ void k_next_anchor(const int * __restrict__ out, int * __restrict__ d_anchor) {
    PDL_ENTRY();
    *d_anchor = out[out[0]];   // out = [n_emit, emit...]; emit[n_emit-1] is the bonus
}

// --- device-side greedy accept ---

constexpr int AM_PARTS = 64;   // argmax partial blocks per logits row

// per (row, part): max value and its smallest index over a slice of the vocab.   grid (AM_PARTS, nt), 256 threads
__global__ void k_argmax_part(const float * __restrict__ logits, float * __restrict__ pv, int * __restrict__ pi) {
    PDL_ENTRY();
    __shared__ float sv[256];
    __shared__ int si[256];
    const int row = blockIdx.y, part = blockIdx.x, tid = threadIdx.x;
    const int per = (VOCAB + AM_PARTS - 1) / AM_PARTS;
    const int beg = part * per, end = min(VOCAB, beg + per);
    const float * lr = logits + (size_t) row * VOCAB;
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int i = beg + tid; i < end; i += 256) {
        const float v = lr[i];
        if (v > bv) { bv = v; bi = i; }   // strided scan visits increasing i, so ties keep the first
    }
    sv[tid] = bv;
    si[tid] = bi;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (tid < o) {
            const float v2 = sv[tid + o];
            const int i2 = si[tid + o];
            if (v2 > sv[tid] || (v2 == sv[tid] && i2 < si[tid])) { sv[tid] = v2; si[tid] = i2; }
        }
        __syncthreads();
    }
    if (tid == 0) {
        pv[row * AM_PARTS + part] = sv[0];
        pi[row * AM_PARTS + part] = si[0];
    }
}

// Greedy accept for block tokens[0..nt-1] = [t0, d1..d(nt-1)]: row r of the logits predicts token r+1.
// k = number of leading drafts that equal the target's argmax; emits d1..dk plus the bonus argmax of row k,
// and commits k+1 tokens (t0, d1..dk). out = [n_emit, emit[0..7], argmax[0..7]].   1 block, 64 threads
__global__ void k_accept(const float * __restrict__ pv, const int * __restrict__ pi, const int * __restrict__ tokens,
                         int nt, int * __restrict__ d_ncommit, int * __restrict__ out) {
    PDL_ENTRY();
    __shared__ int am[MAXT];
    const int tid = threadIdx.x;
    if (tid < nt) {
        float bv = -INFINITY;
        int bi = INT_MAX;
        for (int p = 0; p < AM_PARTS; ++p) {
            const float v = pv[tid * AM_PARTS + p];
            const int i = pi[tid * AM_PARTS + p];
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
        am[tid] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        int k = 0;
        while (k < nt - 1 && tokens[k + 1] == am[k]) ++k;
        for (int i = 0; i < k; ++i) out[1 + i] = tokens[i + 1];
        out[1 + k] = am[k];
        out[0] = k + 1;
        for (int i = 0; i < MAXT; ++i) out[1 + MAXT + i] = i < nt ? am[i] : -1;
        *d_ncommit = k + 1;
    }
}

__global__ void k_set_int(int * __restrict__ p, int v) {
    PDL_ENTRY();
    *p = v;
}

// rmsnorm over each 128-wide head, times silu(z), as the fp16 input of ssm_out.   grid (8, NVH), 128 threads
__global__ void k_gated_norm_h(const float * __restrict__ o, const float * __restrict__ z, const float * __restrict__ w,
                               __half * __restrict__ x16, int nt) {
    PDL_ENTRY();
    __shared__ float red[32];
    const size_t idx = (size_t) blockIdx.x * DINNER + blockIdx.y * DS + threadIdx.x;
    if ((int) blockIdx.x >= nt) {
        x16[idx] = __float2half(0.0f);
        return;
    }
    const float v = o[idx];
    const float ss = block_sum<4>(v * v, red);
    x16[idx] = __float2half(v * rsqrtf(ss / DS + EPS) * w[threadIdx.x] * silu(z[idx]));
}

// --- full attention ---

// per-head rmsnorm + neox-style partial rope for q (24 heads) and k (4 heads); k, v go to the f16 cache.
// grid (nt, NH + NKV), 256 threads (= head dim)
__global__ void k_qk_norm_rope(const float * __restrict__ qf, const float * __restrict__ kf, const float * __restrict__ vf,
                               const float * __restrict__ qn_w, const float * __restrict__ kn_w, float * __restrict__ qout,
                               __half * __restrict__ Kc, __half * __restrict__ Vc, const int * __restrict__ d_pos0) {
    PDL_ENTRY();
    __shared__ float red[32];
    __shared__ float buf[HD];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const int pos = *d_pos0 + t;
    const bool is_q = hh < NH;
    const float * src = is_q ? qf + (size_t) t * (NH * 2 * HD) + hh * 2 * HD : kf + (size_t) t * (NKV * HD) + (hh - NH) * HD;
    const float v = src[d];
    const float ss = block_sum<8>(v * v, red);
    const float xn = v * rsqrtf(ss / HD + EPS) * (is_q ? qn_w[d] : kn_w[d]);
    buf[d] = xn;
    __syncthreads();
    float r = xn;
    if (d < NROT) {
        const int p = d < NROT / 2 ? d : d - NROT / 2;
        const float theta_scale = powf(ROPE_BASE, -2.0f / NROT);
        const float theta = (float) pos * powf(theta_scale, (float) p);
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = buf[p], x1 = buf[p + NROT / 2];
        r = d < NROT / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    if (is_q) {
        qout[(size_t) t * NH * HD + hh * HD + d] = r;
    } else {
        const int kh = hh - NH;
        Kc[((size_t) pos * NKV + kh) * HD + d] = __float2half(r);
        Vc[((size_t) pos * NKV + kh) * HD + d] = __float2half(vf[(size_t) t * NKV * HD + kh * HD + d]);
    }
}

// causal attention for each (token, q head) over cache positions [0, pos0 + t], then the sigmoid output gate,
// written as the fp16 input of wo.  grid (8, NH), 256 threads, dynamic smem (HD + max_ctx) floats
__global__ void k_attn(const float * __restrict__ q, const __half * __restrict__ Kc, const __half * __restrict__ Vc,
                       const float * __restrict__ qf, __half * __restrict__ out, const int * __restrict__ d_pos0, int nt) {
    PDL_ENTRY();
    extern __shared__ float sm[];
    __shared__ float red[32];
    float * qs = sm;
    float * sc = sm + HD;
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    if (t >= nt) {
        out[(size_t) t * NH * HD + h * HD + tid] = __float2half(0.0f);
        return;
    }
    const int npos = *d_pos0 + t + 1;
    const int kvh = h / (NH / NKV);
    qs[tid] = q[(size_t) t * NH * HD + h * HD + tid];
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    const float scale = rsqrtf((float) HD);
    for (int j = warp; j < npos; j += 8) {
        const __half * kr = Kc + ((size_t) j * NKV + kvh) * HD;
        float a = 0.0f;
        for (int dd = lane; dd < HD; dd += 32) a += qs[dd] * __half2float(kr[dd]);
        a = warp_sum(a);
        if (lane == 0) sc[j] = a * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = tid; j < npos; j += 256) m = fmaxf(m, sc[j]);
    m = block_max<8>(m, red);
    float s = 0.0f;
    for (int j = tid; j < npos; j += 256) {
        const float e = expf(sc[j] - m);
        sc[j] = e;
        s += e;
    }
    s = block_sum<8>(s, red);
    __syncthreads();
    float o = 0.0f;
    for (int j = 0; j < npos; ++j) o += sc[j] * __half2float(Vc[((size_t) j * NKV + kvh) * HD + tid]);
    o /= s;
    const float gate = qf[(size_t) t * NH * 2 * HD + h * 2 * HD + HD + tid];
    out[(size_t) t * NH * HD + h * HD + tid] = __float2half(o / (1.0f + expf(-gate)));
}

// --- split-KV tensor-core attention (target and drafter) ---
// k_attn / k_df_attn give each (token, head) one CTA that walks every key with a dependent V load per key:
// ~87 ns per key, so 52 us per target layer at 600 keys and ~0.7 ms at 8k. Here the 8 tokens x GROUP q heads
// that share a kv head form 16-row m-tiles; a CTA takes one (split, kv head, m-tile) and runs flash attention
// over 64-key chunks c = split, split + FA_SPLIT, ... with mma.m16n8k16. q and p are split into fp16 hi + lo
// parts (hi + lo carries ~22 bits), each product is exact in fp32 and accumulates in fp32, so the math keeps
// fp32 precision against the fp16 cache. Row r's result depends only on its own query and key range [lo, hi):
// chunks outside it are skipped or are exact no-ops (scale 1, add 0), and the chunk -> split map is fixed,
// so a token gets the same bits whatever nt is (lossless speculation). k_fa_combine merges the splits in order.

__device__ __forceinline__ void mma_16816_acc(float d[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

constexpr int FA_CH = 64;      // keys per chunk
constexpr int FA_SPLIT = 32;   // CTAs per (kv head, m-tile); chunk c is handled by split c % FA_SPLIT
constexpr int FA_W = 4;        // warps per CTA: 16 keys each for S, HD/4 dims each for O

template <int HDv> struct FaSmem {
    static constexpr int HDP = HDv + 8;    // padded rows: lanes (g, q) hit banks 4g + q
    static constexpr int PP = FA_CH + 8;
    static constexpr size_t BYTES = (size_t) (2 * FA_CH * HDP + 2 * 16 * HDP + 2 * 16 * PP) * sizeof(__half);
};

__device__ __forceinline__ uint32_t lds32(const __half * p) { return *(const uint32_t *) p; }
__device__ __forceinline__ uint32_t pack_h2(__half lo, __half hi) {
    return (uint32_t) __half_as_ushort(lo) | ((uint32_t) __half_as_ushort(hi) << 16);
}

// DRAFT = false: target, causal, key range of token t is [0, P + t + 1), rows t >= nt empty.
// DRAFT = true: drafter block, key range [max(0, P + t - (DF_SWA - 1)), P + 8) for all 8 tokens.
// q: [t][q_stride] fp32, head h at h * HDv.  K/V cache: [pos][nkv][HDv] fp16.
// part_o: [nkv][FA_SPLIT][ntok*GROUP][HDv], part_ml: [nkv][FA_SPLIT][ntok*GROUP][2] (running max, sum); grid z = ntok*GROUP/16
template <int HDv, int GROUP, bool DRAFT>
__global__ void __launch_bounds__(FA_W * 32) k_fa_split(const float * __restrict__ q, int q_stride,
                                                         const __half * __restrict__ Kc, const __half * __restrict__ Vc,
                                                         int nkv, const int * __restrict__ d_pos0, int nt, int ntok,
                                                         float * __restrict__ part_o, float * __restrict__ part_ml) {
    PDL_ENTRY();
    using SM = FaSmem<HDv>;
    constexpr int HDP = SM::HDP, PP = SM::PP, NTO = HDv / 32;
    const int R = ntok * GROUP;   // query rows per kv head: ntok tokens (8 per verify block, up to MAXP in prefill)
    extern __shared__ __align__(16) unsigned char fa_smem[];
    __half * Ks = (__half *) fa_smem;
    __half * Vs = Ks + FA_CH * HDP;
    __half * Qh = Vs + FA_CH * HDP;
    __half * Ql = Qh + 16 * HDP;
    __half * Ph = Ql + 16 * HDP;
    __half * Pl = Ph + 16 * PP;
    __shared__ float red[FA_W][16], red2[FA_W][16], m_row[16], l_row[16], alpha_row[16];
    __shared__ int row_lo[16], row_hi[16];

    const int split = blockIdx.x, kvh = blockIdx.y, mt = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, qd = lane & 3;
    const int P = *d_pos0;
    const float scale = rsqrtf((float) HDv);

    // per-row key ranges; CTA bounds
    int hi_max = 0, lo_min = INT_MAX;
#pragma unroll 1
    for (int i = 0; i < 16; ++i) {
        const int r = mt * 16 + i, t = r / GROUP;
        int lo, hi;
        if (DRAFT) { lo = max(0, P + t - (DF_SWA - 1)); hi = P + 8; }
        else if (t < nt) { lo = 0; hi = P + t + 1; }
        else { lo = 0; hi = 0; }
        if (tid == 0) { row_lo[i] = lo; row_hi[i] = hi; }
        if (hi > lo) { hi_max = max(hi_max, hi); lo_min = min(lo_min, lo); }
    }
    const int nch = (hi_max + FA_CH - 1) / FA_CH;
    if (split >= nch) return;   // k_fa_combine reads only splits below a row's own chunk count

    // q rows -> fp16 hi/lo in shared memory (rows with no keys are zero)
    for (int idx = tid; idx < 16 * HDv; idx += FA_W * 32) {
        const int i = idx / HDv, dd = idx % HDv;
        const int r = mt * 16 + i, t = r / GROUP, h = kvh * GROUP + r % GROUP;
        const bool live = DRAFT || t < nt;
        const float v = live ? q[(size_t) t * q_stride + h * HDv + dd] : 0.0f;
        const __half vh = __float2half_rn(v);
        Qh[i * HDP + dd] = vh;
        Ql[i * HDP + dd] = __float2half_rn(v - __half2float(vh));
    }
    if (tid < 16) { m_row[tid] = -INFINITY; l_row[tid] = 0.0f; }

    constexpr int NTW = NTO;      // this warp's HDv/4 dims = HDv/32 n-tiles of 8
    float o[NTW][4];
#pragma unroll
    for (int n = 0; n < NTW; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;

    for (int c = split; c < nch; c += FA_SPLIT) {
        const int k0 = c * FA_CH;
        if (k0 + FA_CH <= lo_min) continue;
        __syncthreads();   // previous chunk's K/V/P are no longer read
        // K/V chunk -> smem, keys >= hi_max zeroed (cache slots past the block are scratch)
        constexpr int V16 = HDv / 8;   // uint4 per key row
        for (int idx = tid; idx < FA_CH * V16; idx += FA_W * 32) {
            const int j = idx / V16, v = idx % V16;
            const int key = k0 + j;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (key < hi_max) {
                kv = *((const uint4 *) (Kc + ((size_t) key * nkv + kvh) * HDv) + v);
                vv = *((const uint4 *) (Vc + ((size_t) key * nkv + kvh) * HDv) + v);
            }
            *((uint4 *) (Ks + j * HDP) + v) = kv;
            *((uint4 *) (Vs + j * HDP) + v) = vv;
        }
        __syncthreads();

        // S = (q_hi + q_lo) K^T for this warp's 16 keys (2 n-tiles)
        float s[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
        for (int kk = 0; kk < HDv / 16; ++kk) {
            const int d0 = kk * 16 + 2 * qd;
            uint32_t ah[4], al[4];
            ah[0] = lds32(Qh + g * HDP + d0);       al[0] = lds32(Ql + g * HDP + d0);
            ah[1] = lds32(Qh + (g + 8) * HDP + d0); al[1] = lds32(Ql + (g + 8) * HDP + d0);
            ah[2] = lds32(Qh + g * HDP + d0 + 8);   al[2] = lds32(Ql + g * HDP + d0 + 8);
            ah[3] = lds32(Qh + (g + 8) * HDP + d0 + 8); al[3] = lds32(Ql + (g + 8) * HDP + d0 + 8);
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const __half * kr = Ks + (warp * 16 + nn * 8 + g) * HDP + d0;
                const uint32_t b[2] = {lds32(kr), lds32(kr + 8)};
                mma_16816_acc(s[nn], ah, b);
                mma_16816_acc(s[nn], al, b);
            }
        }
        // mask + scale; per-row chunk max over this warp's keys
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nn = 0; nn < 2; ++nn)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = g + 8 * (e >> 1);
                const int key = k0 + warp * 16 + nn * 8 + 2 * qd + (e & 1);
                const bool ok = key >= row_lo[row] && key < row_hi[row];
                s[nn][e] = ok ? s[nn][e] * scale : -INFINITY;
                if (e < 2) mx0 = fmaxf(mx0, s[nn][e]); else mx1 = fmaxf(mx1, s[nn][e]);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        if (qd == 0) { red[warp][g] = mx0; red[warp][g + 8] = mx1; }
        __syncthreads();
        if (tid < 16) {
            float mc = red[0][tid];
#pragma unroll
            for (int w = 1; w < FA_W; ++w) mc = fmaxf(mc, red[w][tid]);
            const float m_old = m_row[tid], m_new = fmaxf(m_old, mc);
            alpha_row[tid] = m_new == -INFINITY ? 1.0f : expf(m_old - m_new);
            m_row[tid] = m_new;
        }
        __syncthreads();
        // p = exp(s - m) as fp16 hi/lo; per-row sums
        float sm0 = 0.0f, sm1 = 0.0f;
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
            float p[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = g + 8 * (e >> 1);
                p[e] = s[nn][e] == -INFINITY ? 0.0f : expf(s[nn][e] - m_row[row]);
            }
            sm0 += p[0] + p[1];
            sm1 += p[2] + p[3];
            const int col = warp * 16 + nn * 8 + 2 * qd;
            const __half h0 = __float2half_rn(p[0]), h1 = __float2half_rn(p[1]);
            const __half h2 = __float2half_rn(p[2]), h3 = __float2half_rn(p[3]);
            *(uint32_t *) (Ph + g * PP + col) = pack_h2(h0, h1);
            *(uint32_t *) (Ph + (g + 8) * PP + col) = pack_h2(h2, h3);
            *(uint32_t *) (Pl + g * PP + col) = pack_h2(__float2half_rn(p[0] - __half2float(h0)), __float2half_rn(p[1] - __half2float(h1)));
            *(uint32_t *) (Pl + (g + 8) * PP + col) = pack_h2(__float2half_rn(p[2] - __half2float(h2)), __float2half_rn(p[3] - __half2float(h3)));
        }
        sm0 += __shfl_xor_sync(0xffffffffu, sm0, 1);
        sm0 += __shfl_xor_sync(0xffffffffu, sm0, 2);
        sm1 += __shfl_xor_sync(0xffffffffu, sm1, 1);
        sm1 += __shfl_xor_sync(0xffffffffu, sm1, 2);
        if (qd == 0) { red2[warp][g] = sm0; red2[warp][g + 8] = sm1; }
        __syncthreads();
        if (tid < 16) {
            float ssum = red2[0][tid];
#pragma unroll
            for (int w = 1; w < FA_W; ++w) ssum += red2[w][tid];
            l_row[tid] = l_row[tid] * alpha_row[tid] + ssum;
        }
        // O = O * alpha + (p_hi + p_lo) V for this warp's dims
        const float a0 = alpha_row[g], a1 = alpha_row[g + 8];
#pragma unroll
        for (int n = 0; n < NTW; ++n) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
#pragma unroll
        for (int kk = 0; kk < FA_CH / 16; ++kk) {
            const int c0 = kk * 16 + 2 * qd;
            uint32_t ah[4], al[4];
            ah[0] = lds32(Ph + g * PP + c0);       al[0] = lds32(Pl + g * PP + c0);
            ah[1] = lds32(Ph + (g + 8) * PP + c0); al[1] = lds32(Pl + (g + 8) * PP + c0);
            ah[2] = lds32(Ph + g * PP + c0 + 8);   al[2] = lds32(Pl + g * PP + c0 + 8);
            ah[3] = lds32(Ph + (g + 8) * PP + c0 + 8); al[3] = lds32(Pl + (g + 8) * PP + c0 + 8);
#pragma unroll
            for (int n = 0; n < NTW; ++n) {
                const int dim = warp * (HDv / 4) + n * 8 + g;
                const uint32_t b[2] = {pack_h2(Vs[c0 * HDP + dim], Vs[(c0 + 1) * HDP + dim]),
                                       pack_h2(Vs[(c0 + 8) * HDP + dim], Vs[(c0 + 9) * HDP + dim])};
                mma_16816_acc(o[n], ah, b);
                mma_16816_acc(o[n], al, b);
            }
        }
    }
    __syncthreads();
    const size_t rbase = ((size_t) kvh * FA_SPLIT + split) * R + mt * 16;
#pragma unroll
    for (int n = 0; n < NTW; ++n) {
        const int dim = warp * (HDv / 4) + n * 8 + 2 * qd;
        *(float2 *) (part_o + (rbase + g) * HDv + dim) = make_float2(o[n][0], o[n][1]);
        *(float2 *) (part_o + (rbase + g + 8) * HDv + dim) = make_float2(o[n][2], o[n][3]);
    }
    if (tid < 16) {
        part_ml[(rbase + tid) * 2 + 0] = m_row[tid];
        part_ml[(rbase + tid) * 2 + 1] = l_row[tid];
    }
}

// merge the splits of row (t, h) in split order; target: x the sigmoid gate (qf), rows t >= nt zero.
// grid (8, heads), HDv threads.  out: [t][out_stride] fp16, head h at h * HDv.
template <int HDv, int GROUP, bool DRAFT>
__global__ void k_fa_combine(const float * __restrict__ part_o, const float * __restrict__ part_ml,
                             const float * __restrict__ qf, __half * __restrict__ out, int out_stride,
                             const int * __restrict__ d_pos0, int nt, int ntok) {
    PDL_ENTRY();
    const int R = ntok * GROUP;
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    __half * dst = out + (size_t) t * out_stride + h * HDv + d;
    if (!DRAFT && t >= nt) {
        *dst = __float2half(0.0f);
        return;
    }
    const int P = *d_pos0;
    const int hi = DRAFT ? P + 8 : P + t + 1;
    const int ns = min(FA_SPLIT, (hi + FA_CH - 1) / FA_CH);
    const int kvh = h / GROUP, r = t * GROUP + h % GROUP;
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) {
        const size_t b = ((size_t) kvh * FA_SPLIT + s) * R + r;
        if (part_ml[b * 2 + 1] > 0.0f) M = fmaxf(M, part_ml[b * 2]);
    }
    float acc = 0.0f, L = 0.0f;
    for (int s = 0; s < ns; ++s) {
        const size_t b = ((size_t) kvh * FA_SPLIT + s) * R + r;
        const float l = part_ml[b * 2 + 1];
        if (l > 0.0f) {
            const float w = expf(part_ml[b * 2] - M);
            acc += w * part_o[b * HDv + d];
            L += w * l;
        }
    }
    float v = acc / L;
    if (!DRAFT) {
        const float gate = qf[(size_t) t * NH * 2 * HD + h * 2 * HD + HD + d];
        v = v / (1.0f + expf(-gate));
    }
    *dst = __float2half(v);
}

// ---------------------------------------------------------------------------------------------
// engine

struct Q6 {
    const uint8_t * w = nullptr;
    int rows = 0, cols = 0;
};

// DFlash2 drafter weights (Q8_0, tensor-core tile layout; Q6 is just {ptr, rows, cols}) and buffers
struct DLayerW {
    float * attn_norm = nullptr, * q_norm = nullptr, * k_norm = nullptr, * ffn_norm = nullptr;
    float * attn_base = nullptr, * ffn_base = nullptr;   // [side][tap][D]
    Q6 wq, wkv, wo, attn_proj, ffn_proj, gate_up, down;
};

struct Drafter {
    Gguf gguf;
    DLayerW L[DF_L];
    Q6 fc, sel_hidden;
    float * enc_norm = nullptr, * out_norm = nullptr;
    uint8_t * sel_prev = nullptr, * sel_next = nullptr;  // raw Q8_0 [VOCAB][256]
    __half * DK = nullptr, * DV = nullptr;              // [DF_L][max_ctx][DF_NKV][DF_HD]
    __half * feat16 = nullptr;                          // [8][DF_FEAT] target features of the last forward
    float * x = nullptr, * h = nullptr, * tmp = nullptr, * dyn = nullptr, * q = nullptr, * qr = nullptr;
    float * k = nullptr, * v = nullptr, * g = nullptr, * u = nullptr, * genc = nullptr, * hid = nullptr;
    float * unary = nullptr, * score = nullptr, * tk_pv = nullptr;
    int * cand = nullptr, * tk_pi = nullptr, * tok = nullptr;
};

// Matrices that share an input are concatenated by rows so one launch serves them all.
struct LayerW {
    bool attn = false;
    float * attn_norm = nullptr;
    float * post_norm = nullptr;
    Q6 in_proj;    // linear attn: [qkv 10240 | z 6144 | beta 48 | alpha 48];  full attn: [q 12288 | k 1024 | v 1024]
    Q6 out_proj;   // ssm_out or attn_output
    float * conv_w = nullptr, * dt_bias = nullptr, * ssm_a = nullptr, * ssm_norm = nullptr;
    float * q_norm = nullptr, * k_norm = nullptr;
    Q6 gate_up;    // [ffn_gate 17408 | ffn_up 17408]
    Q6 ffn_down;
};

struct Engine {
    Gguf gguf;
    std::vector<void *> allocs;
    LayerW L[NL];
    float * out_norm = nullptr;
    Q6 lm_head;
    const GTensor * tok_embd = nullptr;  // stays host-side, rows dequantized on the CPU like llama.cpp

    int max_ctx = 0;
    int n_past = 0;
    int layer_limit = NL;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    float last_ms = 0.0f;
    double last_host_ms = 0.0;

    // activations
    float * x = nullptr, * xn = nullptr, * y = nullptr;
    float * big1 = nullptr, * big2 = nullptr, * big3 = nullptr;
    float * a_raw = nullptr, * b_raw = nullptr, * g = nullptr, * beta = nullptr;
    float * o = nullptr, * o2 = nullptr;
    float * kbuf = nullptr, * vbuf = nullptr, * qbuf = nullptr, * attn_out = nullptr;
    float * logits = nullptr;
    __half * x16 = nullptr;     // [8][FF] fp16 input of the next GEMV; rows >= nt are zero
    int * d_pos0 = nullptr;
    float * h_embd = nullptr;   // pinned staging for embeddings
    int * h_pos0 = nullptr;     // pinned

    // recurrent / kv state (committed)
    float * conv_state = nullptr;  // [48][3][CONVCH]
    float * ssm_state = nullptr;   // [48][NVH][DS][DS]
    __half * Kc = nullptr;         // [16][max_ctx][NKV][HD]; entries past n_past are scratch
    __half * Vc = nullptr;

    // per-layer history of the last forward, replayed by the commit kernels
    float * qkv_hist = nullptr;    // [48][MAXP][CONVCH] conv inputs
    float * co_hist = nullptr;     // [48][MAXP][CONVCH] conv outputs, q/k l2-normalized
    float * g_hist = nullptr;      // [48][MAXP][NVH]
    float * beta_hist = nullptr;   // [48][MAXP][NVH]
    int n_recr_run = N_RECR;       // recurrent layers inside layer_limit

    // device-side accept
    int * d_tokens = nullptr;      // [MAXT] block tokens
    int * d_ncommit = nullptr;
    int * d_out = nullptr;         // [1 + 2*MAXT]: n_emit, emitted tokens, per-row argmax
    float * am_pv = nullptr;       // [MAXT][AM_PARTS]
    int * am_pi = nullptr;
    int * h_tokens = nullptr;      // pinned
    int * h_out = nullptr;         // pinned

    // CUDA graphs, one per token count (positions are read from d_pos0)
    cudaGraphExec_t graph[MAXP + 1][2] = {};   // faie2_forward, per token count (> MAXT: prefill, no logits)
    cudaGraphExec_t graph_verify[MAXT + 1] = {};
    cudaGraphExec_t graph_spec = nullptr;   // draft + verify + accept + commit + inject
    bool use_graphs = false;

    uint8_t * stage = nullptr;   // load-time staging for raw Q6_K before repacking
    size_t stage_bytes = 0;

    uint8_t * tok_dev = nullptr; // raw Q6_K token_embd on the GPU; every path embeds through k_embed_q6k
    Drafter * df = nullptr;      // DFlash2 drafter, once loaded
    int * d_anchor = nullptr;    // next token to feed (the last emitted one)

    std::vector<float *> snap_conv, snap_ssm;   // recurrent-state snapshots (faie2_snapshot_*)
    std::vector<int> snap_npast;
    float * gr_partial = nullptr;  // prefill GEMM residue partials
    float * fa_o = nullptr;        // split-KV attention partials (target and drafter share them)
    float * fa_ml = nullptr;
    float * sk_partial = nullptr;  // split-K partial tiles of the v4 GEMVs
    int * sk_counter = nullptr;    // per-tile arrival counters, back to zero after every launch

    void drop_graphs() {
        for (int n = 0; n <= MAXP; ++n)
            for (int l = 0; l < 2; ++l)
                if (graph[n][l]) { cudaGraphExecDestroy(graph[n][l]); graph[n][l] = nullptr; }
        for (int n = 0; n <= MAXT; ++n)
            if (graph_verify[n]) { cudaGraphExecDestroy(graph_verify[n]); graph_verify[n] = nullptr; }
        if (graph_spec) { cudaGraphExecDestroy(graph_spec); graph_spec = nullptr; }
    }

    ~Engine() {
        drop_graphs();
        delete df;
        for (void * p : allocs) cudaFree(p);
        if (h_embd) cudaFreeHost(h_embd);
        if (h_pos0) cudaFreeHost(h_pos0);
        if (h_tokens) cudaFreeHost(h_tokens);
        if (h_out) cudaFreeHost(h_out);
        if (ev0) cudaEventDestroy(ev0);
        if (ev1) cudaEventDestroy(ev1);
        if (stream) cudaStreamDestroy(stream);
    }
};

static Engine * g_eng = nullptr;

template <typename T> static int dalloc(Engine & e, T ** p, size_t n) {
    CK(cudaMalloc((void **) p, n * sizeof(T)));
    e.allocs.push_back(*p);
    return 0;
}

static int load_f32(Engine & e, const std::string & name, size_t n, float ** out) {
    auto it = e.gguf.tensors.find(name);
    if (it == e.gguf.tensors.end()) FAIL("missing tensor " + name);
    const GTensor & t = it->second;
    if (t.type != GGML_TYPE_F32) FAIL(name + " is not F32");
    if (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3] != n) FAIL(name + " has unexpected size");
    if (dalloc(e, out, n)) return -1;
    CK(cudaMemcpy(*out, t.host, t.nbytes, cudaMemcpyHostToDevice));
    return 0;
}

// Load Q6_K tensors with equal cols, concatenated by rows, in the tensor-core tile layout.
static int load_q6_group(Engine & e, const std::vector<std::string> & names, int cols, const std::vector<int> & rows, Q6 & q) {
    const int nb = cols / 256;
    int total = 0;
    for (int r : rows) {
        if (r % 16) FAIL(names[0] + ": rows not a multiple of 16");
        total += r;
    }
    uint8_t * d = nullptr;
    if (dalloc(e, &d, (size_t) total * nb * Q6K_BYTES)) return -1;
    int row0 = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        auto it = e.gguf.tensors.find(names[i]);
        if (it == e.gguf.tensors.end()) FAIL("missing tensor " + names[i]);
        const GTensor & t = it->second;
        if (t.type != GGML_TYPE_Q6_K) FAIL(names[i] + " is not Q6_K (type " + std::to_string(t.type) + ")");
        if ((int) t.ne[0] != cols || (int) t.ne[1] != rows[i]) FAIL(names[i] + " has unexpected shape");
        if (t.nbytes != (size_t) rows[i] * (cols / 256) * Q6K_BYTES) FAIL(names[i] + ": byte size mismatch");
        if (t.nbytes > e.stage_bytes) {
            if (e.stage) cudaFree(e.stage);
            CK(cudaMalloc(&e.stage, t.nbytes));
            e.stage_bytes = t.nbytes;
        }
        CK(cudaMemcpy(e.stage, t.host, t.nbytes, cudaMemcpyHostToDevice));
        const size_t nthr = (size_t) rows[i] / 16 * nb * 32;
        k_q6k_repack_mma<<<(unsigned) ((nthr + 255) / 256), 256>>>(e.stage, d + (size_t) row0 / 16 * nb * Q6T_BYTES, rows[i], nb);
        CK(cudaGetLastError());
        row0 += rows[i];
    }
    q.w = d;
    q.rows = total;
    q.cols = cols;
    return 0;
}

// Load Q8_0 tensors of the drafter with equal cols, concatenated by rows, in the Q8_0 tile layout.
static int load_q8_group(Engine & e, Gguf & gg, const std::vector<std::string> & names, int cols,
                         const std::vector<int> & rows, Q6 & q) {
    const int nsb = cols / 256;
    int total = 0;
    for (int r : rows) {
        if (r % 16) FAIL(names[0] + ": rows not a multiple of 16");
        total += r;
    }
    if (cols % 256) FAIL(names[0] + ": cols not a multiple of 256");
    uint8_t * d = nullptr;
    if (dalloc(e, &d, (size_t) total * (cols / 32) * Q80_BYTES)) return -1;
    int row0 = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        auto it = gg.tensors.find(names[i]);
        if (it == gg.tensors.end()) FAIL("drafter: missing tensor " + names[i]);
        const GTensor & t = it->second;
        if (t.type != GGML_TYPE_Q8_0) FAIL(names[i] + " is not Q8_0");
        if ((int) t.ne[0] != cols || (int) t.ne[1] != rows[i]) FAIL(names[i] + " has unexpected shape");
        if (t.nbytes != (size_t) rows[i] * (cols / 32) * Q80_BYTES) FAIL(names[i] + ": byte size mismatch");
        if (t.nbytes > e.stage_bytes) {
            if (e.stage) cudaFree(e.stage);
            CK(cudaMalloc(&e.stage, t.nbytes));
            e.stage_bytes = t.nbytes;
        }
        CK(cudaMemcpy(e.stage, t.host, t.nbytes, cudaMemcpyHostToDevice));
        const size_t nthr = (size_t) rows[i] / 16 * nsb * 32;
        k_q8_repack_mma<<<(unsigned) ((nthr + 255) / 256), 256>>>(e.stage, d + (size_t) row0 / 16 * nsb * Q8T_BYTES, rows[i], nsb);
        CK(cudaGetLastError());
        row0 += rows[i];
    }
    q.w = d;
    q.rows = total;
    q.cols = cols;
    return 0;
}

static int load_f32_from(Engine & e, Gguf & gg, const std::string & name, size_t n, float ** out) {
    auto it = gg.tensors.find(name);
    if (it == gg.tensors.end()) FAIL("drafter: missing tensor " + name);
    const GTensor & t = it->second;
    if (t.type != GGML_TYPE_F32) FAIL(name + " is not F32");
    if (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3] != n) FAIL(name + " has unexpected size");
    if (dalloc(e, out, n)) return -1;
    CK(cudaMemcpy(*out, t.host, t.nbytes, cudaMemcpyHostToDevice));
    return 0;
}

static int load_raw(Engine & e, Gguf & gg, const std::string & name, uint32_t type, uint8_t ** out) {
    auto it = gg.tensors.find(name);
    if (it == gg.tensors.end()) FAIL("missing tensor " + name);
    const GTensor & t = it->second;
    if (t.type != type || !t.nbytes) FAIL(name + " has an unexpected type");
    if (dalloc(e, out, t.nbytes)) return -1;
    CK(cudaMemcpy(*out, t.host, t.nbytes, cudaMemcpyHostToDevice));
    return 0;
}

// v1: raw GGUF layout (kept for the unit-test A/B)
template <int NT> static void launch_gemv1_nt(const Q6 & q, const float * x, float * y, cudaStream_t s) {
    constexpr int R = 4;
    const int blocks = (q.rows + 4 * R - 1) / (4 * R);
    k_q6k_gemv<R, NT><<<blocks, 128, 0, s>>>(q.w, q.rows, q.cols, x, y);
}

// v2: repacked layout. R rows per CTA so each activation load feeds R rows (with R=1 at nt=8 every 210-byte
// block re-read 8 KB of activations from L1); split-K keeps >= 1280 CTAs even for 5120-row matrices.
template <int NT> static void launch_gemv2_nt(const Q6 & q, const float * x, float * y, cudaStream_t s) {
    if (q.rows >= 2048) k_q6k_gemv2<4, NT><<<(q.rows + 3) / 4, 128, 0, s>>>(q.w, q.rows, q.cols, x, y);
    else if (q.rows >= 256) k_q6k_gemv2<2, NT><<<(q.rows + 1) / 2, 128, 0, s>>>(q.w, q.rows, q.cols, x, y);
    else k_q6k_gemv2<1, NT><<<q.rows, 128, 0, s>>>(q.w, q.rows, q.cols, x, y);
}

template <bool V2> static void gemv_v(const Q6 & q, const float * x, float * y, int nt, cudaStream_t s) {
    switch (nt) {
#define FAIE2_NT_CASE(N) case N: V2 ? launch_gemv2_nt<N>(q, x, y, s) : launch_gemv1_nt<N>(q, x, y, s); break;
        FAIE2_NT_CASE(1) FAIE2_NT_CASE(2) FAIE2_NT_CASE(3) FAIE2_NT_CASE(4)
        FAIE2_NT_CASE(5) FAIE2_NT_CASE(6) FAIE2_NT_CASE(7)
#undef FAIE2_NT_CASE
        default: V2 ? launch_gemv2_nt<8>(q, x, y, s) : launch_gemv1_nt<8>(q, x, y, s); break;
    }
}

static void gemv_mma(const Q6 & q, const __half * x16, float * y, int nt, cudaStream_t s) {
    SegList sl{};
    sl.n = 1;
    sl.row_end[0] = q.rows;
    sl.y[0] = y;
    k_q6k_mma<8><<<q.rows / 16, 256, 0, s>>>(q.w, q.rows, q.cols, x16, sl, nt, 0);
}

static SegList make_segs(std::initializer_list<std::pair<int, float *>> parts) {
    SegList sl{};
    int end = 0;
    for (const auto & p : parts) {
        end += p.first;
        sl.row_end[sl.n] = end;
        sl.y[sl.n] = p.second;
        ++sl.n;
    }
    return sl;
}

// Every enqueued kernel goes through launch(): with g_pdl it carries the programmatic-serialization
// attribute (see PDL_ENTRY). Launch errors are kept in g_launch_err and reported by run_maybe_graph.
// g_pdl: 0 off, 1 every kernel, 2 GEMVs only (their weight prefetch overlaps the previous kernel's tail;
// the other kernels keep plain stream order, so their waiting CTAs never hold SMs another kernel needs),
// 3 every kernel except the split-KV attention pair (launch_attn)
static int g_pdl = 3;
static cudaError_t g_launch_err = cudaSuccess;

template <typename... P, typename... A>
static void launch_x(bool pdl, void (*k)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, k, static_cast<P>(args)...);
    if (err != cudaSuccess && g_launch_err == cudaSuccess) g_launch_err = err;
}

template <typename... P, typename... A>
static void launch(void (*k)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A... args) {
    launch_x(g_pdl == 1 || g_pdl == 3, k, grid, block, smem, s, args...);
}

template <typename... P, typename... A>
static void launch_attn(void (*k)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A... args) {
    launch_x(g_pdl == 1, k, grid, block, smem, s, args...);
}

template <typename... P, typename... A>
static void launch_gemv(void (*k)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A... args) {
    launch_x(g_pdl != 0, k, grid, block, smem, s, args...);
}

// v4 work split: GEMV_NW-warp CTAs of J blocks per warp, S splits per tile. The smallest S that gives
// >= 2048 CTAs (or the most splits possible); every matrix here has a block count divisible by GEMV_NW.
constexpr int SK_TILES = 16384;
constexpr size_t SK_PARTIAL_FLOATS = (size_t) 2 << 20;

static void gemv_cfg(int tiles, int nb, int & S, int & J) {
    const int units = nb / GEMV_NW;
    S = 1;
    for (int s = 1; s <= units; ++s) {
        if (units % s) continue;
        if (s > 1 && (tiles > SK_TILES || (size_t) tiles * s * 128 > SK_PARTIAL_FLOATS)) break;
        S = s;
        if ((long long) tiles * s >= 2048) break;
    }
    J = units / S;
}

static void gemv4(const SplitK & sk, bool q8, const Q6 & q, const __half * x16, const SegList & sl, int nt, bool acc,
                  cudaStream_t s) {
    const int tiles = q.rows / 16, nb = q.cols >> 8;
    int S, J;
    gemv_cfg(tiles, nb, S, J);
    if (q8) launch_gemv(k_q8_mma4, dim3(tiles, S), dim3(32 * GEMV_NW), 0, s, q.w, q.cols, x16, sl, nt, acc ? 1 : 0, J, sk);
    else launch_gemv(k_q6k_mma4, dim3(tiles, S), dim3(32 * GEMV_NW), 0, s, q.w, q.cols, x16, sl, nt, acc ? 1 : 0, J, sk);
}

static int g_gemv_version = 3;   // 3..6, switchable for A/B (faie2_set_gemv_version); v3 measured fastest so far
static int g_nsm = 170;          // SM count, set at load

static void gemv4(const Engine & e, bool q8, const Q6 & q, const __half * x16, const SegList & sl, int nt, bool acc,
                  cudaStream_t s) {
    if (g_gemv_version == 6) {
        const SplitK sk{e.sk_partial, e.sk_counter};
        const dim3 grid(2 * g_nsm), block(G6_WARPS * 32);
        if (q8) launch_gemv(k_gemv6<true>, grid, block, 0, s, q.w, q.rows / 16, q.cols, x16, sl, nt, acc ? 1 : 0, sk);
        else launch_gemv(k_gemv6<false>, grid, block, 0, s, q.w, q.rows / 16, q.cols, x16, sl, nt, acc ? 1 : 0, sk);
        return;
    }
    if (g_gemv_version == 5) {
        const SplitK sk{e.sk_partial, e.sk_counter};
        const dim3 grid(2 * g_nsm), block(GEMV5_WARPS * 32);
        if (q8) launch_gemv(k_gemv5<true>, grid, block, 0, s, q.w, q.rows / 16, q.cols, x16, sl, nt, acc ? 1 : 0, sk);
        else launch_gemv(k_gemv5<false>, grid, block, 0, s, q.w, q.rows / 16, q.cols, x16, sl, nt, acc ? 1 : 0, sk);
        return;
    }
    if (g_gemv_version == 3) {
        if (q8) launch_gemv(k_q8_mma<8>, dim3(q.rows / 16), dim3(256), 0, s, q.w, q.rows, q.cols, x16, sl, nt, acc ? 1 : 0);
        else launch_gemv(k_q6k_mma<8>, dim3(q.rows / 16), dim3(256), 0, s, q.w, q.rows, q.cols, x16, sl, nt, acc ? 1 : 0);
        return;
    }
    gemv4(SplitK{e.sk_partial, e.sk_counter}, q8, q, x16, sl, nt, acc, s);
}

// prefill: ceil(nt / 8) token groups per weight read (k_q6k_mma_n / k_q8_mma_n)
static float * g_gr_partial = nullptr;   // [8][MAXP][GR_MAX_ROWS] residue partials of k_q6k_gemm_r (engine-owned)
constexpr int GR_MAX_ROWS = 2 * FF;      // widest prefill GEMM: gate_up
static bool g_gemm_r = true;             // prefill Q6_K GEMM: residue CTAs (true) or k_q6k_mma_n (false)

template <int NG> static size_t gr_smem_bytes() { return (size_t) 2 * 8 * NG * GR_BSTRIDE * sizeof(__half); }

template <int NG> static void gemm_n_ng(bool q8, const Q6 & q, const __half * x16, const SegList & sl, int nt, bool acc,
                                        cudaStream_t s) {
    if (q8) {
        launch_gemv(k_q8_mma_n<NG>, dim3(q.rows / 16), dim3(256), 0, s, q.w, q.rows, q.cols, x16, sl, nt, acc ? 1 : 0);
    } else if (g_gemm_r && q.rows <= GR_MAX_ROWS) {
        launch_gemv(k_q6k_gemm_r<NG>, dim3((q.rows / 16 + 7) / 8, 8), dim3(256), gr_smem_bytes<NG>(), s, q.w, q.rows, q.cols, x16, g_gr_partial);
        launch(k_gemm_reduce, dim3((q.rows * nt + 255) / 256), dim3(256), 0, s, (const float *) g_gr_partial, q.rows, 8 * NG, nt, sl, acc ? 1 : 0);
    } else {
        launch_gemv(k_q6k_mma_n<NG>, dim3(q.rows / 16), dim3(256), 0, s, q.w, q.rows, q.cols, x16, sl, nt, acc ? 1 : 0);
    }
}

template <int NG> static int gr_set_smem() {
    CK(cudaFuncSetAttribute(k_q6k_gemm_r<NG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gr_smem_bytes<NG>()));
    return 0;
}

static void gemm_n(bool q8, const Q6 & q, const __half * x16, const SegList & sl, int nt, bool acc, cudaStream_t s) {
    switch ((nt + 7) / 8) {
        case 2: gemm_n_ng<2>(q8, q, x16, sl, nt, acc, s); break;
        case 3: gemm_n_ng<3>(q8, q, x16, sl, nt, acc, s); break;
        case 4: gemm_n_ng<4>(q8, q, x16, sl, nt, acc, s); break;
        case 5: gemm_n_ng<5>(q8, q, x16, sl, nt, acc, s); break;
        case 6: gemm_n_ng<6>(q8, q, x16, sl, nt, acc, s); break;
        case 7: gemm_n_ng<7>(q8, q, x16, sl, nt, acc, s); break;
        default: gemm_n_ng<8>(q8, q, x16, sl, nt, acc, s); break;
    }
}

// one launch over a row-concatenated Q6 group; parts = {rows, output} in concatenation order
static void gemv_seg(const Engine & e, const Q6 & q, std::initializer_list<std::pair<int, float *>> parts, int nt,
                     bool accumulate, cudaStream_t s) {
    if (nt > MAXT) gemm_n(false, q, e.x16, make_segs(parts), nt, accumulate, s);
    else gemv4(e, false, q, e.x16, make_segs(parts), nt, accumulate, s);
}

// drafter GEMV (Q8_0); drafting runs full 8-row blocks, injection runs the committed (padded) rows
static void gemv_q8(const Engine & e, const Q6 & q, const __half * x16, std::initializer_list<std::pair<int, float *>> parts,
                    cudaStream_t s, int nt = 8) {
    if (nt > MAXT) gemm_n(true, q, x16, make_segs(parts), nt, false, s);
    else gemv4(e, true, q, x16, make_segs(parts), 8, false, s);
}

static size_t attn_smem(const Engine & e) { return (size_t) (HD + e.max_ctx) * sizeof(float); }

constexpr size_t FA_O_FLOATS = std::max((size_t) NKV * FA_SPLIT * MAXP * (NH / NKV) * HD, (size_t) DF_NKV * FA_SPLIT * 8 * (DF_NH / DF_NKV) * DF_HD);
constexpr size_t FA_ML_FLOATS = std::max((size_t) NKV * FA_SPLIT * MAXP * (NH / NKV) * 2, (size_t) DF_NKV * FA_SPLIT * 8 * (DF_NH / DF_NKV) * 2);
static int g_attn_version = 2;   // 1: per-(token, head) kernels, 2: split-KV tensor-core (faie2_set_attn_version)

// enqueue one forward of nt tokens whose embeddings are already in e.x; positions come from d_pos0.
// Each GEMV reads e.x16, which the previous kernel wrote; the output projections add into e.x directly.
static void enqueue_forward(Engine & e, int nt, bool want_logits) {
    cudaStream_t s = e.stream;
    const int rows = (nt + 7) / 8 * 8;   // padded token rows (8 for a verify block, up to MAXP in prefill)
    launch(k_embed_q6k, dim3(nt), dim3(256), 0, s, e.d_tokens, e.tok_dev, e.x);
    int ir = 0, ia = 0, ifeat = 0;
    for (int il = 0; il < e.layer_limit; ++il) {
        const LayerW & w = e.L[il];
        if (e.df && ifeat < DF_NFEAT && il == DF_TARGET_LAYERS[ifeat]) {
            launch(k_feat_capture, dim3((rows * D + 255) / 256), dim3(256), 0, s, e.x, e.df->feat16, ifeat, nt, rows);
            ++ifeat;
        }
        launch(k_rmsnorm_h, dim3(rows), dim3(1024), 0, s, e.x, w.attn_norm, e.x16, D, nt);
        if (!w.attn) {
            float * qkvh = e.qkv_hist + (size_t) ir * MAXP * CONVCH;
            float * coh = e.co_hist + (size_t) ir * MAXP * CONVCH;
            float * gh = e.g_hist + (size_t) ir * MAXP * NVH;
            float * bh = e.beta_hist + (size_t) ir * MAXP * NVH;
            gemv_seg(e, w.in_proj, {{CONVCH, qkvh}, {DINNER, e.big2}, {NVH, e.b_raw}, {NVH, e.a_raw}}, nt, false, s);
            launch(k_gdn_conv, dim3((CONVCH + 255) / 256), dim3(256), 0, s, qkvh, e.conv_state + (size_t) ir * 3 * CONVCH, w.conv_w, coh, nt);
            launch(k_l2norm_qk, dim3(nt, 2 * NKH), dim3(DS), 0, s, coh);
            launch(k_gdn_gates, dim3((nt * NVH + 255) / 256), dim3(256), 0, s, e.a_raw, e.b_raw, w.dt_bias, w.ssm_a, gh, bh, nt * NVH);
            launch(k_gdn_recur, dim3(NVH), dim3(DS), 0, s, coh, gh, bh, e.ssm_state + (size_t) ir * NVH * DS * DS, e.o, nt);
            launch(k_gated_norm_h, dim3(rows, NVH), dim3(DS), 0, s, e.o, e.big2, w.ssm_norm, e.x16, nt);
            gemv_seg(e, w.out_proj, {{D, e.x}}, nt, true, s);
            ++ir;
        } else {
            gemv_seg(e, w.in_proj, {{NH * 2 * HD, e.big1}, {NKV * HD, e.kbuf}, {NKV * HD, e.vbuf}}, nt, false, s);
            __half * Kc = e.Kc + (size_t) ia * e.max_ctx * NKV * HD;
            __half * Vc = e.Vc + (size_t) ia * e.max_ctx * NKV * HD;
            launch(k_qk_norm_rope, dim3(nt, NH + NKV), dim3(HD), 0, s, e.big1, e.kbuf, e.vbuf, w.q_norm, w.k_norm, e.qbuf, Kc, Vc, e.d_pos0);
            if (g_attn_version == 2) {
                launch_attn(k_fa_split<HD, NH / NKV, false>, dim3(FA_SPLIT, NKV, rows * (NH / NKV) / 16), dim3(FA_W * 32), FaSmem<HD>::BYTES, s,
                       e.qbuf, NH * HD, Kc, Vc, NKV, e.d_pos0, nt, rows, e.fa_o, e.fa_ml);
                launch_attn(k_fa_combine<HD, NH / NKV, false>, dim3(rows, NH), dim3(HD), 0, s, e.fa_o, e.fa_ml, e.big1, e.x16, NH * HD, e.d_pos0, nt, rows);
            } else {
                launch(k_attn, dim3(rows, NH), dim3(256), attn_smem(e), s, e.qbuf, Kc, Vc, e.big1, e.x16, e.d_pos0, nt);
            }
            gemv_seg(e, w.out_proj, {{D, e.x}}, nt, true, s);
            ++ia;
        }
        launch(k_rmsnorm_h, dim3(rows), dim3(1024), 0, s, e.x, w.post_norm, e.x16, D, nt);
        gemv_seg(e, w.gate_up, {{FF, e.big1}, {FF, e.big2}}, nt, false, s);
        launch(k_swiglu_h, dim3((rows * FF + 255) / 256), dim3(256), 0, s, e.big1, e.big2, e.x16, FF, nt, rows);
        gemv_seg(e, w.ffn_down, {{D, e.x}}, nt, true, s);
    }
    if (want_logits) {
        launch(k_rmsnorm_h, dim3(rows), dim3(1024), 0, s, e.x, e.out_norm, e.x16, D, nt);
        gemv_seg(e, e.lm_head, {{VOCAB, e.logits}}, nt, false, s);
    }
}

// fold the first *d_ncommit tokens of the last forward into the recurrent state
static void enqueue_commit(Engine & e) {
    launch(k_conv_commit, dim3((CONVCH + 255) / 256, e.n_recr_run), dim3(256), 0, e.stream, e.conv_state, e.qkv_hist, e.d_ncommit);
    launch(k_gdn_commit, dim3(NVH, e.n_recr_run), dim3(DS), 0, e.stream, e.co_hist, e.g_hist, e.beta_hist, e.ssm_state, e.d_ncommit);
}

// drafter KV for the committed rows: g = rmsnorm(fc(features)); per layer K = rope(knorm(Wk g)), V = Wv g
// at positions d_pos0 .. d_pos0 + n_commit - 1 (no attn_norm on g, as in dflash.cpp's injection path)
static void enqueue_inject(Engine & e, int rows = 8) {
    Drafter & d = *e.df;
    cudaStream_t s = e.stream;
    gemv_q8(e, d.fc, d.feat16, {{D, d.genc}}, s, rows);
    launch(k_rmsnorm_fh, dim3(rows), dim3(1024), 0, s, d.genc, d.enc_norm, nullptr, e.x16, D);
    for (int l = 0; l < DF_L; ++l) {
        const DLayerW & w = d.L[l];
        gemv_q8(e, w.wkv, e.x16, {{DF_KD, d.k}, {DF_KD, d.v}}, s, rows);
        launch(k_df_qk, dim3(rows, DF_NH + DF_NKV), dim3(DF_HD), 0, s, nullptr, d.k, d.v, nullptr, w.k_norm, nullptr,
               d.DK + (size_t) l * e.max_ctx * DF_KD, d.DV + (size_t) l * e.max_ctx * DF_KD, e.d_pos0, e.d_ncommit);
    }
}

// noise block [anchor, MASK x 7] at positions d_pos0 .. +7 -> logits -> top-16 lattice -> greedy walk.
// Writes the verify block [anchor, d1..d7] into e.d_tokens. Block K/V go to scratch cache slots >= n_past.
static void enqueue_draft(Engine & e) {
    Drafter & d = *e.df;
    cudaStream_t s = e.stream;
    const int ew = (8 * D + 255) / 256;
    launch(k_df_noise_tokens, dim3(1), dim3(32), 0, s, e.d_anchor, d.tok);
    launch(k_embed_q6k, dim3(8), dim3(256), 0, s, d.tok, e.tok_dev, d.x);
    for (int l = 0; l < DF_L; ++l) {
        const DLayerW & w = d.L[l];
        __half * DK = d.DK + (size_t) l * e.max_ctx * DF_KD;
        __half * DV = d.DV + (size_t) l * e.max_ctx * DF_KD;
        launch(k_rmsnorm_fh, dim3(8), dim3(1024), 0, s, d.x, w.attn_norm, d.h, e.x16, D);
        gemv_q8(e, w.attn_proj, e.x16, {{DF_PROJ, d.dyn}}, s);
        launch(k_df_conv, dim3(ew), dim3(256), 0, s, d.h, d.dyn, w.attn_base, 0, e.x16, nullptr);
        gemv_q8(e, w.wq, e.x16, {{DF_QD, d.q}}, s);
        gemv_q8(e, w.wkv, e.x16, {{DF_KD, d.k}, {DF_KD, d.v}}, s);
        launch(k_df_qk, dim3(8, DF_NH + DF_NKV), dim3(DF_HD), 0, s, d.q, d.k, d.v, w.q_norm, w.k_norm, d.qr, DK, DV, e.d_pos0, nullptr);
        if (g_attn_version == 2) {
            launch_attn(k_fa_split<DF_HD, DF_NH / DF_NKV, true>, dim3(FA_SPLIT, DF_NKV, 8 * (DF_NH / DF_NKV) / 16), dim3(FA_W * 32),
                   FaSmem<DF_HD>::BYTES, s, d.qr, DF_QD, DK, DV, DF_NKV, e.d_pos0, 8, 8, e.fa_o, e.fa_ml);
            launch_attn(k_fa_combine<DF_HD, DF_NH / DF_NKV, true>, dim3(8, DF_NH), dim3(DF_HD), 0, s, e.fa_o, e.fa_ml, (const float *) nullptr,
                   e.x16, DF_QD, e.d_pos0, 8, 8);
        } else {
            launch(k_df_attn, dim3(8, DF_NH), dim3(DF_HD), (DF_HD + DF_SWA + 8) * sizeof(float), s, d.qr, DK, DV, e.x16, e.d_pos0);
        }
        gemv_q8(e, w.wo, e.x16, {{D, d.tmp}}, s);
        launch(k_df_conv, dim3(ew), dim3(256), 0, s, d.tmp, d.dyn, w.attn_base, 1, nullptr, d.x);
        launch(k_rmsnorm_fh, dim3(8), dim3(1024), 0, s, d.x, w.ffn_norm, d.h, e.x16, D);
        gemv_q8(e, w.ffn_proj, e.x16, {{DF_PROJ, d.dyn}}, s);
        launch(k_df_conv, dim3(ew), dim3(256), 0, s, d.h, d.dyn, w.ffn_base, 0, e.x16, nullptr);
        gemv_q8(e, w.gate_up, e.x16, {{FF, d.g}, {FF, d.u}}, s);
        launch(k_swiglu8_h, dim3((8 * FF + 255) / 256), dim3(256), 0, s, d.g, d.u, e.x16, FF);
        gemv_q8(e, w.down, e.x16, {{D, d.tmp}}, s);
        launch(k_df_conv, dim3(ew), dim3(256), 0, s, d.tmp, d.dyn, w.ffn_base, 1, nullptr, d.x);
    }
    launch(k_rmsnorm_fh, dim3(8), dim3(1024), 0, s, d.x, d.out_norm, nullptr, e.x16, D);
    gemv_seg(e, e.lm_head, {{VOCAB, e.logits}}, 8, false, s);   // the target's LM head
    gemv_q8(e, d.sel_hidden, e.x16, {{DF_SEL_R, d.hid}}, s);
    launch(k_topk_part, dim3(TOPK_PARTS, 8), dim3(256), 0, s, e.logits, d.tk_pv, d.tk_pi);
    launch(k_topk_final, dim3(8), dim3(32), 0, s, d.tk_pv, d.tk_pi, d.cand, d.unary);
    launch(k_df_scores, dim3(DF_TOPK, 8), dim3(256), 0, s, d.sel_prev, d.sel_next, d.hid, d.cand, d.unary, e.d_anchor, d.score);
    launch(k_df_walk, dim3(1), dim3(32), 0, s, d.score, d.cand, e.d_anchor, e.d_tokens);
}

// forward + device-side greedy accept + commit of the accepted prefix (+ drafter KV inject, next anchor)
static void enqueue_verify(Engine & e, int nt) {
    enqueue_forward(e, nt, true);
    launch(k_argmax_part, dim3(AM_PARTS, nt), dim3(256), 0, e.stream, e.logits, e.am_pv, e.am_pi);
    launch(k_accept, dim3(1), dim3(64), 0, e.stream, e.am_pv, e.am_pi, e.d_tokens, nt, e.d_ncommit, e.d_out);
    enqueue_commit(e);
    if (e.df) enqueue_inject(e);
    launch(k_next_anchor, dim3(1), dim3(1), 0, e.stream, e.d_out, e.d_anchor);
}

// ---------------------------------------------------------------------------------------------
// C API

FAIE2_API const char * faie2_error() { return g_err.c_str(); }

FAIE2_API void faie2_free() {
    delete g_eng;
    g_eng = nullptr;
}

FAIE2_API int faie2_load(const char * path, int max_ctx) {
    faie2_free();
    auto t0 = std::chrono::steady_clock::now();
    Engine * e = new Engine();
    g_eng = e;
    e->max_ctx = max_ctx;
    if (gguf_open(e->gguf, path)) return -1;

    auto num = [&](const char * k) { auto it = e->gguf.nums.find(k); return it == e->gguf.nums.end() ? -1.0 : it->second; };
    if ((int) num("qwen35.embedding_length") != D || (int) num("qwen35.feed_forward_length") != FF ||
        (int) num("qwen35.attention.head_count") != NH || (int) num("qwen35.attention.head_count_kv") != NKV ||
        (int) num("qwen35.attention.key_length") != HD || (int) num("qwen35.rope.dimension_count") != NROT ||
        (int) num("qwen35.ssm.state_size") != DS || (int) num("qwen35.ssm.group_count") != NKH ||
        (int) num("qwen35.ssm.time_step_rank") != NVH || (int) num("qwen35.ssm.inner_size") != DINNER ||
        (int) num("qwen35.ssm.conv_kernel") != CONV_K || (int) num("qwen35.full_attention_interval") != 4 ||
        (int) num("qwen35.block_count") < NL || std::fabs(num("qwen35.rope.freq_base") - ROPE_BASE) > 1.0 ||
        std::fabs(num("qwen35.attention.layer_norm_rms_epsilon") - EPS) > 1e-9) {
        FAIL("GGUF hyperparameters do not match the Qwen3.8-27B constants compiled into FAIE2");
    }

    CK(cudaStreamCreateWithFlags(&e->stream, cudaStreamNonBlocking));
    CK(cudaEventCreate(&e->ev0));
    CK(cudaEventCreate(&e->ev1));
    {
        int dev = 0;
        CK(cudaGetDevice(&dev));
        CK(cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, dev));
    }

    for (int il = 0; il < NL; ++il) {
        LayerW & w = e->L[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        w.attn = is_attn_layer(il);
        if (load_f32(*e, p + "attn_norm.weight", D, &w.attn_norm)) return -1;
        if (load_f32(*e, p + "post_attention_norm.weight", D, &w.post_norm)) return -1;
        if (!w.attn) {
            if (load_q6_group(*e, {p + "attn_qkv.weight", p + "attn_gate.weight", p + "ssm_beta.weight", p + "ssm_alpha.weight"},
                              D, {CONVCH, DINNER, NVH, NVH}, w.in_proj)) return -1;
            if (load_q6_group(*e, {p + "ssm_out.weight"}, DINNER, {D}, w.out_proj)) return -1;
            if (load_f32(*e, p + "ssm_conv1d.weight", (size_t) CONV_K * CONVCH, &w.conv_w)) return -1;
            if (load_f32(*e, p + "ssm_dt.bias", NVH, &w.dt_bias)) return -1;
            if (load_f32(*e, p + "ssm_a", NVH, &w.ssm_a)) return -1;
            if (load_f32(*e, p + "ssm_norm.weight", DS, &w.ssm_norm)) return -1;
        } else {
            if (load_q6_group(*e, {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"},
                              D, {NH * HD * 2, NKV * HD, NKV * HD}, w.in_proj)) return -1;
            if (load_q6_group(*e, {p + "attn_output.weight"}, NH * HD, {D}, w.out_proj)) return -1;
            if (load_f32(*e, p + "attn_q_norm.weight", HD, &w.q_norm)) return -1;
            if (load_f32(*e, p + "attn_k_norm.weight", HD, &w.k_norm)) return -1;
        }
        if (load_q6_group(*e, {p + "ffn_gate.weight", p + "ffn_up.weight"}, D, {FF, FF}, w.gate_up)) return -1;
        if (load_q6_group(*e, {p + "ffn_down.weight"}, FF, {D}, w.ffn_down)) return -1;
    }
    if (load_f32(*e, "output_norm.weight", D, &e->out_norm)) return -1;
    if (load_q6_group(*e, {"output.weight"}, D, {VOCAB}, e->lm_head)) return -1;
    if (load_raw(*e, e->gguf, "token_embd.weight", GGML_TYPE_Q6_K, &e->tok_dev)) return -1;

    if (dalloc(*e, &e->x, MAXP * D) || dalloc(*e, &e->xn, MAXT * D) || dalloc(*e, &e->y, MAXT * D) ||
        dalloc(*e, &e->big1, (size_t) MAXP * FF) || dalloc(*e, &e->big2, (size_t) MAXP * FF) ||
        dalloc(*e, &e->big3, (size_t) MAXT * FF) || dalloc(*e, &e->a_raw, MAXP * NVH) ||
        dalloc(*e, &e->b_raw, MAXP * NVH) || dalloc(*e, &e->g, MAXT * NVH) || dalloc(*e, &e->beta, MAXT * NVH) ||
        dalloc(*e, &e->o, MAXP * DINNER) || dalloc(*e, &e->o2, MAXT * DINNER) || dalloc(*e, &e->kbuf, MAXP * NKV * HD) ||
        dalloc(*e, &e->vbuf, MAXP * NKV * HD) || dalloc(*e, &e->qbuf, MAXP * NH * HD) ||
        dalloc(*e, &e->attn_out, MAXT * NH * HD) || dalloc(*e, &e->logits, (size_t) MAXT * VOCAB) ||
        dalloc(*e, &e->d_pos0, 1) || dalloc(*e, &e->conv_state, (size_t) N_RECR * 3 * CONVCH) ||
        dalloc(*e, &e->ssm_state, (size_t) N_RECR * NVH * DS * DS) ||
        dalloc(*e, &e->Kc, (size_t) N_ATTN * max_ctx * NKV * HD) || dalloc(*e, &e->Vc, (size_t) N_ATTN * max_ctx * NKV * HD) ||
        dalloc(*e, &e->x16, (size_t) MAXP * FF) ||
        dalloc(*e, &e->qkv_hist, (size_t) N_RECR * MAXP * CONVCH) || dalloc(*e, &e->co_hist, (size_t) N_RECR * MAXP * CONVCH) ||
        dalloc(*e, &e->g_hist, (size_t) N_RECR * MAXP * NVH) || dalloc(*e, &e->beta_hist, (size_t) N_RECR * MAXP * NVH) ||
        dalloc(*e, &e->d_tokens, MAXP) || dalloc(*e, &e->d_ncommit, 1) || dalloc(*e, &e->d_out, 1 + 2 * MAXT) ||
        dalloc(*e, &e->am_pv, MAXT * AM_PARTS) || dalloc(*e, &e->am_pi, MAXT * AM_PARTS) || dalloc(*e, &e->d_anchor, 1) ||
        dalloc(*e, &e->sk_partial, SK_PARTIAL_FLOATS) || dalloc(*e, &e->sk_counter, SK_TILES) ||
        dalloc(*e, &e->fa_o, FA_O_FLOATS) || dalloc(*e, &e->fa_ml, FA_ML_FLOATS)) {
        return -1;
    }
    CK(cudaMemset(e->sk_counter, 0, SK_TILES * sizeof(int)));
    if (dalloc(*e, &e->gr_partial, (size_t) 8 * MAXP * GR_MAX_ROWS)) return -1;
    g_gr_partial = e->gr_partial;
    if (gr_set_smem<2>() || gr_set_smem<3>() || gr_set_smem<4>() || gr_set_smem<5>() || gr_set_smem<6>() ||
        gr_set_smem<7>() || gr_set_smem<8>()) return -1;
    CK(cudaDeviceSynchronize());
    CK(cudaFree(e->stage));
    e->stage = nullptr;
    e->stage_bytes = 0;
    CK(cudaMallocHost(&e->h_pos0, sizeof(int)));
    CK(cudaMallocHost(&e->h_tokens, MAXP * sizeof(int)));
    CK(cudaMallocHost(&e->h_out, (1 + 2 * MAXT) * sizeof(int)));
    CK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) attn_smem(*e)));
    CK(cudaFuncSetAttribute(k_fa_split<HD, NH / NKV, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) FaSmem<HD>::BYTES));
    CK(cudaFuncSetAttribute(k_fa_split<DF_HD, DF_NH / DF_NKV, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) FaSmem<DF_HD>::BYTES));
    CK(cudaMemset(e->conv_state, 0, (size_t) N_RECR * 3 * CONVCH * sizeof(float)));
    CK(cudaMemset(e->ssm_state, 0, (size_t) N_RECR * NVH * DS * DS * sizeof(float)));
    CK(cudaDeviceSynchronize());
    e->n_past = 0;
    e->last_host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

FAIE2_API int faie2_reset() {
    Engine * e = g_eng;
    if (!e) FAIL("not loaded");
    CK(cudaMemsetAsync(e->conv_state, 0, (size_t) N_RECR * 3 * CONVCH * sizeof(float), e->stream));
    CK(cudaMemsetAsync(e->ssm_state, 0, (size_t) N_RECR * NVH * DS * DS * sizeof(float), e->stream));
    CK(cudaStreamSynchronize(e->stream));
    e->n_past = 0;
    return 0;
}

FAIE2_API int faie2_set_layer_limit(int n) {
    if (!g_eng) FAIL("not loaded");
    Engine & e = *g_eng;
    e.layer_limit = n < 0 || n > NL ? NL : n;
    e.n_recr_run = 0;
    for (int il = 0; il < e.layer_limit; ++il) e.n_recr_run += !is_attn_layer(il);
    e.drop_graphs();
    return 0;
}

FAIE2_API int faie2_use_graphs(int on) {
    if (!g_eng) FAIL("not loaded");
    g_eng->use_graphs = on != 0;
    return 0;
}

// host side of a block: tokens and start position into pinned staging (embedding happens on the GPU)
static int stage_block(Engine & e, const int * tokens, int nt) {
    if (nt < 1 || nt > MAXP) FAIL("nt must be 1..64");
    if (e.n_past + nt > e.max_ctx) FAIL("context full");
    for (int t = 0; t < nt; ++t) {
        const int id = tokens[t];
        if (id < 0 || id >= VOCAB) FAIL("token id out of range");
        e.h_tokens[t] = id;
    }
    *e.h_pos0 = e.n_past;
    CK(cudaEventRecord(e.ev0, e.stream));
    CK(cudaMemcpyAsync(e.d_pos0, e.h_pos0, sizeof(int), cudaMemcpyHostToDevice, e.stream));
    CK(cudaMemcpyAsync(e.d_tokens, e.h_tokens, nt * sizeof(int), cudaMemcpyHostToDevice, e.stream));
    return 0;
}

template <typename F> static int run_maybe_graph(Engine & e, cudaGraphExec_t & ge, F && enqueue) {
    if (e.use_graphs && e.layer_limit == NL) {
        if (!ge) {
            cudaGraph_t gr;
            CK(cudaStreamBeginCapture(e.stream, cudaStreamCaptureModeThreadLocal));
            enqueue();
            CK(cudaStreamEndCapture(e.stream, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            CK(cudaGraphDestroy(gr));
        }
        CK(cudaGraphLaunch(ge, e.stream));
    } else {
        enqueue();
    }
    if (g_launch_err != cudaSuccess) {
        g_err = std::string("kernel launch -> ") + cudaGetErrorString(g_launch_err);
        g_launch_err = cudaSuccess;
        return -1;
    }
    CK(cudaGetLastError());
    return 0;
}

// programmatic dependent launch: 0 off, 1 all kernels, 2 GEMVs only, 3 all but attention (default, measured best)
FAIE2_API int faie2_set_pdl(int mode) {
    if (mode < 0 || mode > 3) FAIL("pdl mode must be 0..3");
    g_pdl = mode;
    if (g_eng) g_eng->drop_graphs();
    return 0;
}

// attention kernels: 1 (per token and head) or 2 (split-KV tensor cores, default); graphs are rebuilt
FAIE2_API int faie2_set_attn_version(int v) {
    if (v != 1 && v != 2) FAIL("attn version must be 1 or 2");
    g_attn_version = v;
    if (g_eng) g_eng->drop_graphs();
    return 0;
}

// prefill Q6_K GEMM: 1 = residue CTAs with shared activations (default), 0 = k_q6k_mma_n; graphs are rebuilt
FAIE2_API int faie2_set_gemm_r(int on) {
    g_gemm_r = on != 0;
    if (g_eng) g_eng->drop_graphs();
    return 0;
}

// engine GEMV kernel: 3 (one 8-warp CTA per tile, default), 4 (split-K) or 5 (stream-K); graphs are rebuilt
FAIE2_API int faie2_set_gemv_version(int v) {
    if (v < 3 || v > 6) FAIL("gemv version must be 3..6");
    g_gemv_version = v;
    if (g_eng) g_eng->drop_graphs();
    return 0;
}

// Run nt tokens at positions n_past .. n_past+nt-1 and commit all of them (prefill / forced tokens).
// nt <= 8 with logits (copies nt x VOCAB floats out), or nt <= 64 without (prefill: one weight read per 8 groups).
FAIE2_API int faie2_forward(const int * tokens, int nt, float * logits) {
    Engine * e = g_eng;
    if (!e) FAIL("not loaded");
    auto t0 = std::chrono::steady_clock::now();
    if (stage_block(*e, tokens, nt)) return -1;
    const bool want = logits != nullptr;
    if (want && nt > MAXT) FAIL("logits are available for nt <= 8 only");
    if (nt > MAXT && g_attn_version != 2) FAIL("prefill blocks > 8 need the split-KV attention (attn version 2)");
    if (run_maybe_graph(*e, e->graph[nt][want ? 1 : 0], [&]() {
            enqueue_forward(*e, nt, want);
            launch(k_set_int, dim3(1), dim3(1), 0, e->stream, e->d_ncommit, nt);
            enqueue_commit(*e);
            if (e->df) enqueue_inject(*e, (nt + 7) / 8 * 8);
        })) return -1;
    CK(cudaEventRecord(e->ev1, e->stream));
    if (want) CK(cudaMemcpyAsync(logits, e->logits, (size_t) nt * VOCAB * sizeof(float), cudaMemcpyDeviceToHost, e->stream));
    CK(cudaStreamSynchronize(e->stream));
    CK(cudaEventElapsedTime(&e->last_ms, e->ev0, e->ev1));
    e->n_past += nt;
    e->last_host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

// Speculative step on block tokens = [t0, d1..d(nt-1)]: t0 is the last emitted token (not yet fed), d are drafts.
// Greedy-accepts the longest draft prefix matching the target's argmax, commits t0 + accepted drafts, and
// returns *n_emit = k+1 tokens in emit: d1..dk and the target's own next token. nt = 1 is plain greedy decoding.
// argmax (optional, nt ints) receives the target argmax of every row; logits (optional) nt x VOCAB.
FAIE2_API int faie2_verify(const int * tokens, int nt, int * emit, int * n_emit, int * argmax, float * logits) {
    Engine * e = g_eng;
    if (!e) FAIL("not loaded");
    auto t0 = std::chrono::steady_clock::now();
    if (nt < 1 || nt > MAXT) FAIL("verify blocks are 1..8 tokens");
    if (stage_block(*e, tokens, nt)) return -1;
    if (run_maybe_graph(*e, e->graph_verify[nt], [&]() { enqueue_verify(*e, nt); })) return -1;
    CK(cudaEventRecord(e->ev1, e->stream));
    CK(cudaMemcpyAsync(e->h_out, e->d_out, (1 + 2 * MAXT) * sizeof(int), cudaMemcpyDeviceToHost, e->stream));
    if (logits) CK(cudaMemcpyAsync(logits, e->logits, (size_t) nt * VOCAB * sizeof(float), cudaMemcpyDeviceToHost, e->stream));
    CK(cudaStreamSynchronize(e->stream));
    CK(cudaEventElapsedTime(&e->last_ms, e->ev0, e->ev1));
    const int n = e->h_out[0];
    if (n < 1 || n > nt) FAIL("accept produced an invalid count");
    *n_emit = n;
    for (int i = 0; i < n; ++i) emit[i] = e->h_out[1 + i];
    if (argmax) for (int i = 0; i < nt; ++i) argmax[i] = e->h_out[1 + MAXT + i];
    e->n_past += n;
    e->last_host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

// ---- DFlash2 drafter ----

FAIE2_API int faie2_load_drafter(const char * path) {
    Engine * e = g_eng;
    if (!e) FAIL("load the target first");
    if (e->df) FAIL("drafter already loaded");
    auto t0 = std::chrono::steady_clock::now();
    Drafter * d = new Drafter();
    if (gguf_open(d->gguf, path)) { delete d; return -1; }
    auto num = [&](const char * k) { auto it = d->gguf.nums.find(k); return it == d->gguf.nums.end() ? -1.0 : it->second; };
    auto tl = d->gguf.iarrays.find("dflash.target_layers");
    bool layers_ok = tl != d->gguf.iarrays.end() && tl->second.size() == DF_NFEAT;
    for (int k = 0; layers_ok && k < DF_NFEAT; ++k) layers_ok = tl->second[k] == DF_TARGET_LAYERS[k];
    if (!layers_ok || (int) num("dflash.block_count") != DF_L || (int) num("dflash.embedding_length") != D ||
        (int) num("dflash.feed_forward_length") != FF || (int) num("dflash.attention.head_count") != DF_NH ||
        (int) num("dflash.attention.head_count_kv") != DF_NKV || (int) num("dflash.attention.key_length") != DF_HD ||
        (int) num("dflash.block_size") != 8 || (int) num("dflash.conv_kernel_size") != 2 ||
        (int) num("dflash.conv_group_size") != 16 || (int) num("dflash.selector_rank") != DF_SEL_R ||
        (int) num("dflash.selector_top_k") != DF_TOPK || (int) num("dflash.attention.sliding_window") != DF_SWA ||
        std::fabs(num("dflash.rope.freq_base") - ROPE_BASE) > 1.0) {
        delete d;
        FAIL("drafter GGUF does not match the DFlash2 constants compiled into FAIE2");
    }
    e->df = d;   // owned by the engine from here; its device memory is in e->allocs
    Gguf & gg = d->gguf;
    for (int l = 0; l < DF_L; ++l) {
        DLayerW & w = d->L[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        if (load_f32_from(*e, gg, p + "attn_norm.weight", D, &w.attn_norm) || load_f32_from(*e, gg, p + "ffn_norm.weight", D, &w.ffn_norm) ||
            load_f32_from(*e, gg, p + "attn_q_norm.weight", DF_HD, &w.q_norm) || load_f32_from(*e, gg, p + "attn_k_norm.weight", DF_HD, &w.k_norm) ||
            load_f32_from(*e, gg, p + "attn_conv_base", (size_t) 4 * D, &w.attn_base) ||
            load_f32_from(*e, gg, p + "ffn_conv_base", (size_t) 4 * D, &w.ffn_base) ||
            load_q8_group(*e, gg, {p + "attn_q.weight"}, D, {DF_QD}, w.wq) ||
            load_q8_group(*e, gg, {p + "attn_k.weight", p + "attn_v.weight"}, D, {DF_KD, DF_KD}, w.wkv) ||
            load_q8_group(*e, gg, {p + "attn_output.weight"}, DF_QD, {D}, w.wo) ||
            load_q8_group(*e, gg, {p + "attn_conv_proj.weight"}, D, {DF_PROJ}, w.attn_proj) ||
            load_q8_group(*e, gg, {p + "ffn_conv_proj.weight"}, D, {DF_PROJ}, w.ffn_proj) ||
            load_q8_group(*e, gg, {p + "ffn_gate.weight", p + "ffn_up.weight"}, D, {FF, FF}, w.gate_up) ||
            load_q8_group(*e, gg, {p + "ffn_down.weight"}, FF, {D}, w.down)) {
            return -1;
        }
    }
    if (load_q8_group(*e, gg, {"fc.weight"}, DF_FEAT, {D}, d->fc) ||
        load_q8_group(*e, gg, {"selector_hidden.weight"}, D, {DF_SEL_R}, d->sel_hidden) ||
        load_f32_from(*e, gg, "enc.output_norm.weight", D, &d->enc_norm) || load_f32_from(*e, gg, "output_norm.weight", D, &d->out_norm) ||
        load_raw(*e, gg, "selector_predecessor.weight", GGML_TYPE_Q8_0, &d->sel_prev) ||
        load_raw(*e, gg, "selector_successor.weight", GGML_TYPE_Q8_0, &d->sel_next)) {
        return -1;
    }
    if (dalloc(*e, &d->DK, (size_t) DF_L * e->max_ctx * DF_KD) || dalloc(*e, &d->DV, (size_t) DF_L * e->max_ctx * DF_KD) ||
        dalloc(*e, &d->feat16, (size_t) MAXP * DF_FEAT) || dalloc(*e, &d->x, 8 * D) || dalloc(*e, &d->h, 8 * D) ||
        dalloc(*e, &d->tmp, 8 * D) || dalloc(*e, &d->dyn, 8 * DF_PROJ) || dalloc(*e, &d->q, 8 * DF_QD) ||
        dalloc(*e, &d->qr, 8 * DF_QD) || dalloc(*e, &d->k, MAXP * DF_KD) || dalloc(*e, &d->v, MAXP * DF_KD) ||
        dalloc(*e, &d->g, (size_t) 8 * FF) || dalloc(*e, &d->u, (size_t) 8 * FF) || dalloc(*e, &d->genc, MAXP * D) ||
        dalloc(*e, &d->hid, 8 * DF_SEL_R) || dalloc(*e, &d->unary, 8 * DF_TOPK) || dalloc(*e, &d->score, 8 * DF_TOPK * DF_TOPK) ||
        dalloc(*e, &d->tk_pv, 8 * TOPK_PARTS * DF_TOPK) || dalloc(*e, &d->cand, 8 * DF_TOPK) ||
        dalloc(*e, &d->tk_pi, 8 * TOPK_PARTS * DF_TOPK) || dalloc(*e, &d->tok, 8)) {
        return -1;
    }
    CK(cudaMemset(d->feat16, 0, (size_t) MAXP * DF_FEAT * sizeof(__half)));
    CK(cudaMemset(d->score, 0, 8 * DF_TOPK * DF_TOPK * sizeof(float)));
    CK(cudaDeviceSynchronize());
    CK(cudaFree(e->stage));
    e->stage = nullptr;
    e->stage_bytes = 0;
    e->drop_graphs();   // forward graphs now also capture features and inject
    e->last_host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

// the next token to feed (e.g. the last prompt token after prefilling the rest)
FAIE2_API int faie2_spec_begin(int anchor) {
    Engine * e = g_eng;
    if (!e) FAIL("not loaded");
    if (anchor < 0 || anchor >= VOCAB) FAIL("anchor out of range");
    CK(cudaMemcpy(e->d_anchor, &anchor, sizeof(int), cudaMemcpyHostToDevice));
    return 0;
}

// One speculative step entirely on the GPU, one CUDA graph: draft [anchor, d1..d7] -> verify -> greedy accept
// -> commit -> drafter KV inject -> next anchor. Emits n_emit (1..8) tokens; drafts (optional, 7 ints).
FAIE2_API int faie2_spec_step(int * emit, int * n_emit, int * drafts) {
    Engine * e = g_eng;
    if (!e || !e->df) FAIL("load the target and the drafter first");
    if (e->n_past + 8 > e->max_ctx) FAIL("context full");
    auto t0 = std::chrono::steady_clock::now();
    *e->h_pos0 = e->n_past;
    CK(cudaEventRecord(e->ev0, e->stream));
    CK(cudaMemcpyAsync(e->d_pos0, e->h_pos0, sizeof(int), cudaMemcpyHostToDevice, e->stream));
    if (run_maybe_graph(*e, e->graph_spec, [&]() {
            enqueue_draft(*e);
            enqueue_verify(*e, 8);
        })) return -1;
    CK(cudaEventRecord(e->ev1, e->stream));
    CK(cudaMemcpyAsync(e->h_out, e->d_out, (1 + 2 * MAXT) * sizeof(int), cudaMemcpyDeviceToHost, e->stream));
    if (drafts) CK(cudaMemcpyAsync(e->h_tokens, e->d_tokens, 8 * sizeof(int), cudaMemcpyDeviceToHost, e->stream));
    CK(cudaStreamSynchronize(e->stream));
    CK(cudaEventElapsedTime(&e->last_ms, e->ev0, e->ev1));
    const int n = e->h_out[0];
    if (n < 1 || n > 8) FAIL("accept produced an invalid count");
    *n_emit = n;
    for (int i = 0; i < n; ++i) emit[i] = e->h_out[1 + i];
    if (drafts) for (int i = 0; i < 7; ++i) drafts[i] = e->h_tokens[1 + i];
    e->n_past += n;
    e->last_host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

// test hook: run only the drafter at the current state. block = [anchor, d1..d7]; cand 8x16; score 8x16x16
FAIE2_API int faie2_df_draft(int * block, int * cand, float * score) {
    Engine * e = g_eng;
    if (!e || !e->df) FAIL("load the target and the drafter first");
    *e->h_pos0 = e->n_past;
    CK(cudaMemcpyAsync(e->d_pos0, e->h_pos0, sizeof(int), cudaMemcpyHostToDevice, e->stream));
    enqueue_draft(*e);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(block, e->d_tokens, 8 * sizeof(int), cudaMemcpyDeviceToHost, e->stream));
    if (cand) CK(cudaMemcpyAsync(cand, e->df->cand, 8 * DF_TOPK * sizeof(int), cudaMemcpyDeviceToHost, e->stream));
    if (score) CK(cudaMemcpyAsync(score, e->df->score, 8 * DF_TOPK * DF_TOPK * sizeof(float), cudaMemcpyDeviceToHost, e->stream));
    CK(cudaStreamSynchronize(e->stream));
    return 0;
}

// test hook: the fp16 target features captured by the last forward, 8 x DF_FEAT
FAIE2_API int faie2_df_features(uint16_t * out) {
    Engine * e = g_eng;
    if (!e || !e->df) FAIL("load the target and the drafter first");
    CK(cudaMemcpy(out, e->df->feat16, (size_t) 8 * DF_FEAT * sizeof(__half), cudaMemcpyDeviceToHost));
    return 0;
}

// ---- recurrent-state snapshots (prefix cache) ----
// Attention K/V (target and drafter) are stored by position, so entries below a position stay valid when the engine
// moves back to it. Only the recurrent state (conv + DeltaNet S) summarizes the past, so a snapshot is that state
// plus n_past. Restoring one and prefilling the rest of a prompt gives exactly the state a full prefill would.
FAIE2_API int faie2_snapshot_alloc(int nslots) {
    Engine * e = g_eng;
    if (!e) FAIL("not loaded");
    if (nslots < 0 || nslots > 64) FAIL("nslots must be 0..64");
    while ((int) e->snap_conv.size() < nslots) {
        float * c = nullptr, * s = nullptr;
        if (dalloc(*e, &c, (size_t) N_RECR * 3 * CONVCH) || dalloc(*e, &s, (size_t) N_RECR * NVH * DS * DS)) return -1;
        e->snap_conv.push_back(c);
        e->snap_ssm.push_back(s);
        e->snap_npast.push_back(-1);
    }
    return 0;
}

FAIE2_API int faie2_snapshot_save(int slot) {
    Engine * e = g_eng;
    if (!e || slot < 0 || slot >= (int) e->snap_conv.size()) FAIL("bad snapshot slot");
    CK(cudaMemcpyAsync(e->snap_conv[slot], e->conv_state, (size_t) N_RECR * 3 * CONVCH * sizeof(float), cudaMemcpyDeviceToDevice, e->stream));
    CK(cudaMemcpyAsync(e->snap_ssm[slot], e->ssm_state, (size_t) N_RECR * NVH * DS * DS * sizeof(float), cudaMemcpyDeviceToDevice, e->stream));
    CK(cudaStreamSynchronize(e->stream));
    e->snap_npast[slot] = e->n_past;
    return 0;
}

FAIE2_API int faie2_snapshot_load(int slot) {
    Engine * e = g_eng;
    if (!e || slot < 0 || slot >= (int) e->snap_conv.size() || e->snap_npast[slot] < 0) FAIL("bad snapshot slot");
    CK(cudaMemcpyAsync(e->conv_state, e->snap_conv[slot], (size_t) N_RECR * 3 * CONVCH * sizeof(float), cudaMemcpyDeviceToDevice, e->stream));
    CK(cudaMemcpyAsync(e->ssm_state, e->snap_ssm[slot], (size_t) N_RECR * NVH * DS * DS * sizeof(float), cudaMemcpyDeviceToDevice, e->stream));
    CK(cudaStreamSynchronize(e->stream));
    e->n_past = e->snap_npast[slot];
    return 0;
}

FAIE2_API int faie2_max_ctx() { return g_eng ? g_eng->max_ctx : -1; }

FAIE2_API float faie2_last_gpu_ms() { return g_eng ? g_eng->last_ms : -1.0f; }
FAIE2_API double faie2_last_host_ms() { return g_eng ? g_eng->last_host_ms : -1.0; }
FAIE2_API int faie2_n_past() { return g_eng ? g_eng->n_past : -1; }

// residual stream after the last forward (nt x D), for layer-by-layer debugging
FAIE2_API int faie2_get_hidden(float * out, int nt) {
    Engine * e = g_eng;
    if (!e) FAIL("not loaded");
    CK(cudaMemcpy(out, e->x, (size_t) nt * D * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

// unit test: y (nt x rows) = W (Q8_0 rows x cols) * x (nt x cols) through the tensor-core Q8_0 GEMV
// (version 3: k_q8_mma, version 4: k_q8_mma4 with the engine's split choice)
FAIE2_API int faie2_test_q8_gemv_v(const uint8_t * Wh, int rows, int cols, const float * xh, int nt, float * yh, float * ms,
                                   int version) {
    if (cols % 256 || rows % 16 || nt < 1 || nt > MAXT) FAIL("bad shape");
    if (version == 4 && (cols / 256) % GEMV_NW) FAIL("v4 needs a block count divisible by 4");
    const size_t wb = (size_t) rows * (cols / 32) * Q80_BYTES;
    uint8_t * dW = nullptr, * dR = nullptr;
    float * dx = nullptr, * dy = nullptr, * dpart = nullptr;
    int * dcnt = nullptr;
    __half * dx16 = nullptr;
    CK(cudaMalloc(&dW, wb));
    CK(cudaMalloc(&dR, wb));
    CK(cudaMalloc(&dx, (size_t) nt * cols * 4));
    CK(cudaMalloc(&dx16, (size_t) 8 * cols * sizeof(__half)));
    CK(cudaMalloc(&dy, (size_t) nt * rows * 4));
    CK(cudaMalloc(&dpart, std::max((size_t) rows / 16 * (cols / 256) * 128, (size_t) 2 * g_nsm * GEMV5_WARPS * 256) * sizeof(float)));
    CK(cudaMalloc(&dcnt, (size_t) rows / 16 * sizeof(int)));
    CK(cudaMemset(dcnt, 0, (size_t) rows / 16 * sizeof(int)));
    CK(cudaMemcpy(dW, Wh, wb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, xh, (size_t) nt * cols * 4, cudaMemcpyHostToDevice));
    const size_t nthr = (size_t) rows / 16 * (cols / 256) * 32;
    k_q8_repack_mma<<<(unsigned) ((nthr + 255) / 256), 256>>>(dW, dR, rows, cols / 256);
    SegList sl{};
    sl.n = 1;
    sl.row_end[0] = rows;
    sl.y[0] = dy;
    auto run = [&]() {
        k_to_half8<<<(8 * cols + 255) / 256, 256>>>(dx, dx16, cols, nt);
        if (version == 6) launch(k_gemv6<true>, dim3(2 * g_nsm), dim3(G6_WARPS * 32), 0, (cudaStream_t) 0, (const uint8_t *) dR, rows / 16, cols, (const __half *) dx16, sl, nt, 0, SplitK{dpart, dcnt});
        else if (version == 5) launch(k_gemv5<true>, dim3(2 * g_nsm), dim3(GEMV5_WARPS * 32), 0, (cudaStream_t) 0, (const uint8_t *) dR, rows / 16, cols, (const __half *) dx16, sl, nt, 0, SplitK{dpart, dcnt});
        else if (version == 4) gemv4(SplitK{dpart, dcnt}, true, Q6{dR, rows, cols}, dx16, sl, nt, false, 0);
        else k_q8_mma<8><<<rows / 16, 256>>>(dR, rows, cols, dx16, sl, nt, 0);
    };
    run();
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(yh, dy, (size_t) nt * rows * 4, cudaMemcpyDeviceToHost));   // first run's output
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) run();
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float t = 0;
    cudaEventElapsedTime(&t, a, b);
    if (ms) *ms = t / 20;
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(dW);
    cudaFree(dR);
    cudaFree(dx);
    cudaFree(dx16);
    cudaFree(dy);
    cudaFree(dpart);
    cudaFree(dcnt);
    return 0;
}

FAIE2_API int faie2_test_q8_gemv(const uint8_t * Wh, int rows, int cols, const float * xh, int nt, float * yh, float * ms) {
    return faie2_test_q8_gemv_v(Wh, rows, cols, xh, nt, yh, ms, 3);
}

// unit test: y (nt x rows) = W (Q6_K rows x cols) * x (nt x cols), all host memory. version 1, 2, 3 or 4.
FAIE2_API int faie2_test_q6k_gemv(const uint8_t * Wh, int rows, int cols, const float * xh, int nt, float * yh, float * ms,
                                  int version) {
    if (cols % 256 || nt < 1 || nt > MAXT) FAIL("bad shape");
    if (version == 4 && (cols / 256) % GEMV_NW) FAIL("v4 needs a block count divisible by 4");
    const size_t wb = (size_t) rows * (cols / 256) * Q6K_BYTES;
    uint8_t * dW = nullptr, * dR = nullptr;
    float * dx = nullptr, * dy = nullptr, * dpart = nullptr;
    int * dcnt = nullptr;
    CK(cudaMalloc(&dW, wb));
    CK(cudaMalloc(&dR, wb));
    CK(cudaMalloc(&dx, (size_t) nt * cols * 4));
    CK(cudaMalloc(&dy, (size_t) nt * rows * 4));
    CK(cudaMemcpy(dW, Wh, wb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, xh, (size_t) nt * cols * 4, cudaMemcpyHostToDevice));
    const size_t nblk = (size_t) rows * (cols / 256);
    __half * dx16 = nullptr;
    if (version >= 3) {
        if (rows % 16) FAIL("rows not a multiple of 16");
        k_q6k_repack_mma<<<(unsigned) ((nblk / 16 * 32 + 255) / 256), 256>>>(dW, dR, rows, cols / 256);
        CK(cudaMalloc(&dx16, (size_t) 8 * cols * sizeof(__half)));
        CK(cudaMalloc(&dpart, std::max((size_t) rows / 16 * (cols / 256) * 128, (size_t) 2 * g_nsm * GEMV5_WARPS * 256) * sizeof(float)));
        CK(cudaMalloc(&dcnt, (size_t) rows / 16 * sizeof(int)));
        CK(cudaMemset(dcnt, 0, (size_t) rows / 16 * sizeof(int)));
    } else {
        k_q6k_repack<<<(unsigned) ((nblk + 255) / 256), 256>>>(dW, dR, rows, cols / 256);
    }
    Q6 q{version == 1 ? dW : dR, rows, cols};
    SegList sl{};
    sl.n = 1;
    sl.row_end[0] = rows;
    sl.y[0] = dy;
    auto run = [&]() {
        if (version == 6) {
            k_to_half8<<<(8 * cols + 255) / 256, 256>>>(dx, dx16, cols, nt);
            launch(k_gemv6<false>, dim3(2 * g_nsm), dim3(G6_WARPS * 32), 0, (cudaStream_t) 0, (const uint8_t *) dR, rows / 16, cols, (const __half *) dx16, sl, nt, 0, SplitK{dpart, dcnt});
        } else if (version == 5) {
            k_to_half8<<<(8 * cols + 255) / 256, 256>>>(dx, dx16, cols, nt);
            launch(k_gemv5<false>, dim3(2 * g_nsm), dim3(GEMV5_WARPS * 32), 0, (cudaStream_t) 0, (const uint8_t *) dR, rows / 16, cols, (const __half *) dx16, sl, nt, 0, SplitK{dpart, dcnt});
        } else if (version == 4) {
            k_to_half8<<<(8 * cols + 255) / 256, 256>>>(dx, dx16, cols, nt);
            gemv4(SplitK{dpart, dcnt}, false, q, dx16, sl, nt, false, 0);
        } else if (version == 3) {
            k_to_half8<<<(8 * cols + 255) / 256, 256>>>(dx, dx16, cols, nt);
            gemv_mma(q, dx16, dy, nt, 0);
        } else if (version == 2) {
            gemv_v<true>(q, dx, dy, nt, 0);
        } else {
            gemv_v<false>(q, dx, dy, nt, 0);
        }
    };
    run();
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(yh, dy, (size_t) nt * rows * 4, cudaMemcpyDeviceToHost));   // first run's output
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) run();
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float t = 0;
    cudaEventElapsedTime(&t, a, b);
    if (ms) *ms = t / 20;
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(dW);
    cudaFree(dR);
    if (dx16) cudaFree(dx16);
    if (dpart) cudaFree(dpart);
    if (dcnt) cudaFree(dcnt);
    cudaFree(dx);
    cudaFree(dy);
    return 0;
}
