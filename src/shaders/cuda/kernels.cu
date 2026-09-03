// ZINC CUDA kernels — ports of the Vulkan compute shaders. NVRTC-compiled at
// runtime for the running device's arch. Authored to the ZINC dispatch ABI:
// bound buffers come first (device pointers), then one by-value push struct.
//
// This is the start of the CUDA kernel library that forward_cuda.zig will
// orchestrate. Kernels here are correctness-first ports (clean parallelization);
// the fused/tiled perf variants come later (M3). Each kernel is validated
// numerically against an independent CPU reference (see src/cuda/kernels_test.c).

// Tensor-core (wmma) support for the fp16 prefill GEMMs (Effort 24 cycle 11).
// NVRTC resolves <mma.h> (and its cuda_fp16.h) from the -I/usr/local/cuda/include
// path passed in cuda_shim.c. Pulls in `half` / `__float2half`. Only used by the
// gemm_*_tc kernels below; the rest of the library is unaffected.
#ifdef ZINC_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
namespace wmma = rocwmma;

// RDNA4 executes HIP kernels as wave32, matching the CUDA kernels' warp-level
// algorithms. HIP's *_sync masks are 64-bit on AMD, so use the maskless wave32
// forms and retain the CUDA call sites unchanged.
#define __shfl_down_sync(mask, value, offset) __shfl_down(value, offset)
#define __shfl_sync(mask, value, lane) __shfl(value, lane)
#define __shfl_xor_sync(mask, value, lane_mask) __shfl_xor(value, lane_mask)

// HIP does not expose CUDA's __dp4a spelling. RDNA4 provides the equivalent
// signed packed 4xint8 dot product through the sudot4 builtin.
__device__ __forceinline__ int zinc_hip_dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}
#define __dp4a(a, b, c) zinc_hip_dp4a((a), (b), (c))
#else
#include <mma.h>
using namespace nvcuda;
#endif

// ---- shared device helpers --------------------------------------------------

// IEEE half -> float, no <cuda_fp16.h> dependency (keeps NVRTC self-contained).
__device__ __forceinline__ float zinc_half_to_float(unsigned short h) {
#ifdef ZINC_ROCM
    // HIP exposes a native bit-cast plus hardware half conversion. The portable
    // fallback below expands to a sizeable integer normalization sequence, which
    // is especially costly in the decode matvecs where every lane converts scale
    // metadata for every superblock.
    return __half2float(__ushort_as_half(h));
#else
    unsigned sign = (unsigned)(h >> 15) & 1u;
    unsigned exp = (unsigned)(h >> 10) & 0x1Fu;
    unsigned mant = (unsigned)h & 0x3FFu;
    unsigned f;
    if (exp == 0u) {
        if (mant == 0u) {
            f = sign << 31;
        } else {
            // subnormal: normalize
            int e = 1;
            while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = (sign << 31) | ((unsigned)(127 - 15 + e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    return __int_as_float((int)f);
#endif
}

__device__ __forceinline__ float zinc_float4_dot(float4 a, float4 b) {
    return (a.x * b.x + a.y * b.y) + (a.z * b.z + a.w * b.w);
}

// Float -> IEEE half bit pattern (no cuda_fp16.h dependency).
__device__ __forceinline__ unsigned short zinc_float_to_half(float x) {
#ifdef ZINC_ROCM
    return __half_as_ushort(__float2half_rn(x));
#else
    unsigned ux = __float_as_int(x);
    unsigned sign = (ux >> 16) & 0x8000u;
    unsigned abs = ux & 0x7FFFFFFFu;
    // Exponent + mantissa extraction
    unsigned exp = (abs >> 23) & 0xFFu;
    unsigned mant = abs & 0x7FFFFFu;
    if (exp >= 142u) {
        // Overflow or inf/nan → inf (or nan if mant!=0)
        return (unsigned short)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    } else if (exp >= 113u) {
        // Normal range: exp bias 127→15
        unsigned new_exp = exp - 112u;
        unsigned h = sign | (new_exp << 10) | (mant >> 13);
        // Round-to-nearest-even
        unsigned lsb = (mant >> 12) & 1u;
        unsigned rem = mant & 0xFFFu;
        if (rem > 0x800u || (rem == 0x800u && lsb)) {
            h++;
            // Handle mantissa overflow into exponent
        }
        return (unsigned short)h;
    } else if (exp >= 103u) {
        // Subnormal: shift mantissa with implicit 1
        unsigned shift = 113u - exp;
        unsigned mant_full = (mant | 0x800000u) >> shift;
        unsigned h = sign | (mant_full >> 13);
        // Round-to-nearest-even
        unsigned lsb = (mant_full >> 12) & 1u;
        unsigned rem = mant_full & 0xFFFu;
        if (rem > 0x800u || (rem == 0x800u && lsb)) h++;
        return (unsigned short)h;
    } else {
        // Underflow → zero
        return (unsigned short)sign;
    }
#endif
}

// GGML Q4_K 6-bit scale/min unpack (j in 0..7), canonical the reference implementation form.
__device__ __forceinline__ void zinc_q4k_scale_min(int j, const unsigned char* q,
                                                    unsigned char* d, unsigned char* m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float zinc_warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Warp reduce with broadcast — result valid in ALL lanes (no __syncthreads)
__device__ __forceinline__ float zinc_warp_reduce_sum_all(float v) {
    v = zinc_warp_reduce_sum(v);
    return __shfl_sync(0xffffffffu, v, 0);
}

// Block reduce — valid result in thread 0. blockDim must be a multiple of 32.
__device__ __forceinline__ float zinc_block_reduce_sum(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = zinc_warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? sh[lane] : 0.0f;
    if (wid == 0) v = zinc_warp_reduce_sum(v);
    return v;
}

// Reduce two independent values with one shared-memory rendezvous. Results are
// valid in thread 0, matching zinc_block_reduce_sum's contract. The per-value
// shuffle/add order is unchanged; only the redundant block barrier is removed.
__device__ __forceinline__ void zinc_block_reduce_sum_pair(float& a, float& b) {
    __shared__ float sha[32];
    __shared__ float shb[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    a = zinc_warp_reduce_sum(a);
    b = zinc_warp_reduce_sum(b);
    if (lane == 0) {
        sha[wid] = a;
        shb[wid] = b;
    }
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    a = (threadIdx.x < nwarps) ? sha[lane] : 0.0f;
    b = (threadIdx.x < nwarps) ? shb[lane] : 0.0f;
    if (wid == 0) {
        a = zinc_warp_reduce_sum(a);
        b = zinc_warp_reduce_sum(b);
    }
}

__device__ __forceinline__ float zinc_warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float zinc_block_reduce_max(float v) {
    __shared__ float shm[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = zinc_warp_reduce_max(v);
    if (lane == 0) shm[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? shm[lane] : -3.4e38f;
    if (wid == 0) v = zinc_warp_reduce_max(v);
    return v;
}

// Block sum reduction with result broadcast to ALL threads (blockDim mult of 32).
__device__ __forceinline__ float zinc_block_reduce_sum_all(float v) {
    __shared__ float shs[32];
    __shared__ float bc;
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = zinc_warp_reduce_sum(v);
    if (lane == 0) shs[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? shs[lane] : 0.0f;
    if (wid == 0) v = zinc_warp_reduce_sum(v);
    if (threadIdx.x == 0) bc = v;
    __syncthreads();
    float r = bc;
    __syncthreads();
    return r;
}

// ---- rms_norm (port of rms_norm_mul.comp) -----------------------------------
// y = weight * (x / sqrt(mean(x^2) + eps)). One block per token.
struct RmsPush { unsigned N; float eps; };
struct RmsQ8Push { unsigned N; float eps; unsigned T; };

extern "C" __global__ void rms_norm(const float* x, const float* w, float* y, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* yt = y + (size_t)token * pc.N;

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        yt[i] = w[i] * (xt[i] * rinv);
    }
}

// RMS norm + Q8_1 activation packing. The hot ROCm dense path used
// to write the normalized f32 row, then launch quantize_act_q8_0 to read it back
// and pack it. A wider RMS reduction feeds the same equation while each packed
// Q8 group keeps the quantizer's wave32 reduction order. Both views leave in
// one launch.
// Grid: one block per token (decode uses one; prefill launches all prompt rows).
extern "C" __global__ void rms_norm_quant_q8_0(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ y, unsigned char* __restrict__ out, RmsQ8Push pc)
{
    const unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* yt = y + (size_t)token * pc.N;

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        const float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0u) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    const float rinv = rms_inv_sh;

    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nwarps = blockDim.x >> 5;
    const unsigned groups = (pc.N + 31u) >> 5;
    for (unsigned c = warp; c < groups; c += nwarps) {
        const unsigned idx = c * 32u + lane;
        const float val = idx < pc.N ? w[idx] * (xt[idx] * rinv) : 0.0f;
        if (idx < pc.N) yt[idx] = val;

        float av = fabsf(val);
        float sv = val;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
            sv += __shfl_xor_sync(0xffffffffu, sv, o);
        }
        const float d = av / 127.0f;
        const float scale = 127.0f / fmaxf(av, 1e-5f);
        const int q = max(-127, min(127, __float2int_rn(val * scale)));

#ifdef ZINC_ROCM
        // MMQ layout for T=1: four Q8_1 groups share one 144-byte tile.
        const unsigned h = c >> 2;
        const unsigned g = c & 3u;
        unsigned char* out_base = out + ((size_t)h * pc.T + token) * 144u;
        if (lane == 0u && idx < pc.N) {
            const unsigned short dh = zinc_float_to_half(d);
            const unsigned short sh = zinc_float_to_half(sv);
            out_base[g * 4u] = (unsigned char)(dh & 0xffu);
            out_base[g * 4u + 1u] = (unsigned char)(dh >> 8);
            out_base[g * 4u + 2u] = (unsigned char)(sh & 0xffu);
            out_base[g * 4u + 3u] = (unsigned char)(sh >> 8);
        }
        out_base[16u + g * 32u + lane] = (unsigned char)q;
#else
        unsigned char* out_base = out + (size_t)token * groups * 36u + (size_t)c * 36u;
        if (lane == 0u && idx < pc.N) {
            const unsigned short dh = zinc_float_to_half(d);
            const unsigned short sh = zinc_float_to_half(sv);
            out_base[0] = (unsigned char)(dh & 0xffu);
            out_base[1] = (unsigned char)(dh >> 8);
            out_base[2] = (unsigned char)(sh & 0xffu);
            out_base[3] = (unsigned char)(sh >> 8);
        }
        out_base[4u + lane] = (unsigned char)q;
#endif
    }
}

// ---- rms_norm_residual ------------------------------------------------------
// Fused gemma post-norm + residual add: hidden[i] += w[i] * (x[i] / rms(x)).
// Collapses the (rms_norm -> scale_accumulate) pair used after attention and
// the FFN in the gemma decode path (the residual scale is always 1.0 there)
// into one launch, also dropping a full n_embd write+read round-trip of the
// normalized vector. `hidden` is the read-write residual accumulator. One
// block per token (decode: token=1).
extern "C" __global__ void rms_norm_residual(const float* x, const float* w, float* hidden, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        ht[i] += w[i] * (xt[i] * rinv);
    }
}

// Muse decode chains each post-norm residual directly into the following
// pre-norm + Q8 pack. Both operations own the same one-token residual row, so a
// block barrier is sufficient between them and removes a launch plus a complete
// hidden-row reload at every attention/FFN boundary.
struct RmsResidualQ8Push { unsigned N; float post_eps; float pre_eps; unsigned T; };
extern "C" __global__ void rms_norm_residual_norm_quant_q8_0(
    const float* __restrict__ x, const float* __restrict__ w_post,
    float* __restrict__ hidden, const float* __restrict__ w_pre,
    float* __restrict__ pre_out, unsigned char* __restrict__ out,
    RmsResidualQ8Push pc)
{
    const unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;
    float* pt = pre_out + (size_t)token * pc.N;

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        const float value = xt[i];
        ss += value * value;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float post_inv, pre_inv;
    if (threadIdx.x == 0u) post_inv = rsqrtf(ss / (float)pc.N + pc.post_eps);
    __syncthreads();
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x)
        ht[i] += w_post[i] * (xt[i] * post_inv);
    __syncthreads();

    float ss2 = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        const float value = ht[i];
        ss2 += value * value;
    }
    ss2 = zinc_block_reduce_sum(ss2);
    if (threadIdx.x == 0u) pre_inv = rsqrtf(ss2 / (float)pc.N + pc.pre_eps);
    __syncthreads();

    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned nwarps = blockDim.x >> 5;
    const unsigned groups = (pc.N + 31u) >> 5;
    for (unsigned c = warp; c < groups; c += nwarps) {
        const unsigned idx = c * 32u + lane;
        const float value = idx < pc.N ? w_pre[idx] * (ht[idx] * pre_inv) : 0.0f;
        if (idx < pc.N) pt[idx] = value;
        float av = fabsf(value);
        float sv = value;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, offset));
            sv += __shfl_xor_sync(0xffffffffu, sv, offset);
        }
        const float d = av / 127.0f;
        const float scale = 127.0f / fmaxf(av, 1e-5f);
        const int q = max(-127, min(127, __float2int_rn(value * scale)));
#ifdef ZINC_ROCM
        const unsigned h = c >> 2;
        const unsigned g = c & 3u;
        unsigned char* out_base = out + ((size_t)h * pc.T + token) * 144u;
        if (lane == 0u && idx < pc.N) {
            const unsigned short dh = zinc_float_to_half(d);
            const unsigned short sh = zinc_float_to_half(sv);
            out_base[g * 4u] = (unsigned char)(dh & 0xffu);
            out_base[g * 4u + 1u] = (unsigned char)(dh >> 8);
            out_base[g * 4u + 2u] = (unsigned char)(sh & 0xffu);
            out_base[g * 4u + 3u] = (unsigned char)(sh >> 8);
        }
        out_base[16u + g * 32u + lane] = (unsigned char)q;
#else
        unsigned char* out_base = out + (size_t)token * groups * 36u + (size_t)c * 36u;
        if (lane == 0u && idx < pc.N) {
            const unsigned short dh = zinc_float_to_half(d);
            const unsigned short sh = zinc_float_to_half(sv);
            out_base[0] = (unsigned char)(dh & 0xffu);
            out_base[1] = (unsigned char)(dh >> 8);
            out_base[2] = (unsigned char)(sh & 0xffu);
            out_base[3] = (unsigned char)(sh >> 8);
        }
        out_base[4u + lane] = (unsigned char)q;
#endif
    }
}

// ---- rms_norm_residual_scale ------------------------------------------------
// rms_norm_residual that also applies the gemma per-layer output scale s[0] to
// the whole residual stream: hidden[i] = s[0] * (hidden[i] + w[i]*x[i]/rms(x)).
// Folds the standalone scalar_mul (blk.N.layer_output_scale.weight) into the
// post-ffn norm+residual on the dense gemma path, removing one tiny launch + one
// command submission per layer. Bit-identical to the two-kernel path: the
// residual add is the same FMA producing the same f32 value scalar_mul would
// have re-loaded, then a plain multiply by s[0].
extern "C" __global__ void rms_norm_residual_scale(const float* x, const float* w, float* hidden, const float* s, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    float scale = s[0];

    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        ht[i] = (ht[i] + w[i] * (xt[i] * rinv)) * scale;
    }
}

// ---- rms_norm_residual_norm / rms_norm_residual_scale_norm ------------------
// Fuse a block's INPUT rms_norm into the PRECEDING block's output norm+residual.
// On the dense gemma path each layer runs four single-block n_embd reductions:
//   pre-attn rms_norm -> ... -> post-attn rms_norm_residual (writes hidden) ->
//   pre-ffn  rms_norm -> ... -> post-ffn  rms_norm_residual_scale (writes hidden)
// A post-norm-residual's `hidden` is exactly the input the very next pre-norm
// reads, and both are ONE-block (grid {1,1,1}) reductions over the same n_embd
// vector — so the next pre-norm can run in the SAME launch, right after the
// residual add (a __syncthreads makes the just-written `hidden` visible to the
// phase-2 reduction). This removes one tiny launch per norm boundary:
//   post-attn-residual + pre-ffn-norm  (within the layer)
//   post-ffn-residual  + pre-attn-norm (across the layer boundary, into the
//                                        NEXT layer's `attn_norm.weight`)
// = ~2 launches/layer on the dense gemma-31b decode path (~119/token over 60
// layers; only layer 0's pre-attn norm and the last layer's post-ffn stay
// standalone). BIT-IDENTICAL to the two-kernel path: phase 1 is the exact
// rms_norm_residual[_scale] arithmetic (same FMA, same reduction), phase 2 is
// the exact rms_norm arithmetic re-reading `hidden` from global — the same
// values the standalone pre-norm would have read. The intervening __syncthreads
// barriers also make reusing zinc_block_reduce_sum's shared scratch race-free.
extern "C" __global__ void rms_norm_residual_norm(
    const float* x, const float* w_post, float* hidden,
    const float* w_pre, float* pre_out, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;
    float* pt = pre_out + (size_t)token * pc.N;

    // phase 1: post-norm + residual (identical to rms_norm_residual)
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        ht[i] += w_post[i] * (xt[i] * rinv);
    }
    __syncthreads(); // hidden fully written + visible before phase-2 reduction

    // phase 2: pre-norm of the updated residual (identical to rms_norm)
    float ss2 = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = ht[i];
        ss2 += v * v;
    }
    ss2 = zinc_block_reduce_sum(ss2);
    __shared__ float rms_inv_sh2;
    if (threadIdx.x == 0) rms_inv_sh2 = rsqrtf(ss2 / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv2 = rms_inv_sh2;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        pt[i] = w_pre[i] * (ht[i] * rinv2);
    }
}

extern "C" __global__ void rms_norm_residual_scale_norm(
    const float* x, const float* w_post, float* hidden, const float* s,
    const float* w_pre, float* pre_out, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;
    float* pt = pre_out + (size_t)token * pc.N;

    // phase 1: post-norm + residual + per-layer output scale (== rms_norm_residual_scale)
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    float scale = s[0];
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        ht[i] = (ht[i] + w_post[i] * (xt[i] * rinv)) * scale;
    }
    __syncthreads(); // hidden fully written + visible before phase-2 reduction

    // phase 2: pre-norm of the updated residual (identical to rms_norm)
    float ss2 = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = ht[i];
        ss2 += v * v;
    }
    ss2 = zinc_block_reduce_sum(ss2);
    __shared__ float rms_inv_sh2;
    if (threadIdx.x == 0) rms_inv_sh2 = rsqrtf(ss2 / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv2 = rms_inv_sh2;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        pt[i] = w_pre[i] * (ht[i] * rinv2);
    }
}

// ---- rms_norm_residual_triple (gemma MoE: post-attn norm+residual + 3 pre-norms) -
// The MoE analogue of rms_norm_residual_norm. On the gemma-26b MoE decode path the
// attention block ends with rms_norm_residual (post-attn norm + residual → hidden,
// one launch) and the MoE block then opens with rms_norm_triple (the 3 pre-norms off
// the just-updated hidden, another launch) across a command boundary. Both are
// single-block (grid {1,1,1}) n_embd reductions over the SAME hidden vector, so the
// triple can run in the SAME launch right after the residual add (a __syncthreads
// makes the just-written hidden visible to the phase-2 reduction). Removes one tiny
// launch + the hidden store/reload round-trip per MoE layer (~30 launches/token on
// gemma-26b). BIT-IDENTICAL to the two-kernel path: phase 1 is the exact
// rms_norm_residual arithmetic (same FMA + reduction), phase 2 is the exact
// rms_norm_triple arithmetic re-reading hidden from global — the same values the
// standalone triple would have read. The intervening __syncthreads also make reusing
// zinc_block_reduce_sum's shared scratch race-free. Block-count PRESERVED (both
// originals were single-block — the C8/C11/C17 win-class).
extern "C" __global__ void rms_norm_residual_triple(
    const float* x, const float* w_post, float* hidden,
    const float* w1, const float* w3,
    float* y1, float* y2, float* y3, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;

    // phase 1: post-attn norm + residual (identical to rms_norm_residual)
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        ht[i] += w_post[i] * (xt[i] * rinv);
    }
    __syncthreads(); // hidden fully written + visible before phase-2 reduction

    // phase 2: 3 pre-norms off the updated residual (identical to rms_norm_triple)
    float ss2 = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = ht[i];
        ss2 += v * v;
    }
    ss2 = zinc_block_reduce_sum(ss2);
    __shared__ float rms_inv_sh2;
    if (threadIdx.x == 0) rms_inv_sh2 = rsqrtf(ss2 / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv2 = rms_inv_sh2;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float xr = ht[i] * rinv2;
        y1[(size_t)token * pc.N + i] = w1[i] * xr;
        y2[(size_t)token * pc.N + i] = xr;
        y3[(size_t)token * pc.N + i] = w3[i] * xr;
    }
}

// ---- rms_norm_rope ----------------------------------------------------------
// Fused gemma per-head Q/K norm + RoPE: collapses the (per-head rms_norm ->
// rope) pair into ONE launch, dropping a full head_dim write+read round-trip of
// the normalized head. Bit-identical to the two-kernel path: same rms (weight
// shared across heads, applied per head_dim), then NEOX partial rope from the
// layer's inv_freq table with attn_scale=1.0 (the gemma q/k rope path always
// uses freq_base_bits==0 / attn_scale_bits==0). One block per head; the
// normalized head is staged in dynamic shared memory (head_dim floats) so the
// rope pair-reads see the post-norm values regardless of x/y aliasing.
// `dst_offset` lets the K head write its result straight into the KV cache at
// position*kv_dim (Q passes 0 and writes head-contiguously into its own buffer),
// folding away the K half of the separate kv_cache_write launch.
struct RmsRopePush { unsigned head_dim; float eps; unsigned rope_dim; unsigned position; unsigned dst_offset; };

extern "C" __global__ void rms_norm_rope(const float* x, const float* w, const float* inv_freq, float* y, RmsRopePush pc) {
    unsigned head = blockIdx.x;
    const float* xt = x + (size_t)head * pc.head_dim;
    float* yt = y + (size_t)pc.dst_offset + (size_t)head * pc.head_dim;
    extern __shared__ float sh[]; // pc.head_dim normalized values

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.head_dim; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.head_dim + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < pc.head_dim; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    unsigned half_rot = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < half_rot; i += blockDim.x) {
        float xi = sh[i];
        float xih = sh[i + half_rot];
        float theta = (float)pc.position * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i] = xi * ct - xih * st;
        yt[i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < pc.head_dim; i += blockDim.x)
        yt[i] = sh[i];
}

// ---- dmmv_q4k (port of dmmv_q4k.comp) ---------------------------------------
// y[row] = sum_k dequant(W[row][k]) * x[k], W in Q4_K. One block per output row.
// Offsets are in BYTES (matching the Vulkan push constants); acc_mode 1 => y +=.
struct DmmvPush { unsigned M, K, a_offset, x_offset, y_offset, acc_mode; };

extern "C" __global__ void dmmv_q4k(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;                       // blocks per row (K / 256)
    const unsigned* arow = a_u32 + (pc.a_offset >> 2) + (size_t)row * bpr * 36u;
    const float* xrow = x + (pc.x_offset >> 2);

    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned b = e >> 8;                          // which 256-elem block
        unsigned within = e & 255u;                  // 0..255 in block
        const unsigned* blk = arow + (size_t)b * 36u;
        unsigned d_dmin = blk[0];
        float d = zinc_half_to_float((unsigned short)(d_dmin & 0xFFFFu));
        float dmin = zinc_half_to_float((unsigned short)(d_dmin >> 16));
        const unsigned char* scales = (const unsigned char*)(blk + 1);   // 12 bytes
        const unsigned char* qs = (const unsigned char*)(blk + 4);       // 128 bytes

        unsigned chunk = within >> 6;                // 0..3 (64-elem chunk)
        unsigned wc = within & 63u;
        unsigned half = wc >> 5;                      // 0 low nibble, 1 high nibble
        unsigned l = wc & 31u;                        // 0..31
        unsigned char sc, mn;
        zinc_q4k_scale_min((int)(chunk * 2u + half), scales, &sc, &mn);
        unsigned char qb = qs[chunk * 32u + l];
        unsigned nib = (half == 0u) ? (qb & 0xFu) : (unsigned)(qb >> 4);
        float wv = d * (float)sc * (float)nib - dmin * (float)mn;
        sum += wv * xrow[e];
    }

    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- swiglu (port of swiglu.comp) -------------------------------------------
// y = silu(gate) * up, silu(x) = x / (1 + exp(-x)). One element per thread.
struct SwigluPush { unsigned N; };

extern "C" __global__ void swiglu(const float* gate, const float* up, float* y, SwigluPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    y[idx] = silu * up[idx];
}

// ---- scale_accumulate (port of scale_accumulate.comp) -----------------------
// a[i] += scale * b[i]  (residual add). a is read-write.
struct ScaleAccPush { unsigned N; float scale; };

extern "C" __global__ void scale_accumulate(float* a, const float* b, ScaleAccPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    a[idx] += pc.scale * b[idx];
}

// ---- moe_combine_tail (gemma4-MoE decode combine, 3 launches fused to 1) ----
// Fuses the per-token MoE combine tail: hidden += post_ffw_norm(shared + moe).
// Replaces scale_accumulate(shared += moe) + rms_norm(shared, w) + scale_accumulate
// (hidden += shared) — three tiny n_embd-sized launches in a strict dependency
// chain (all bubbles exposed). BYTE-IDENTITY: t = shared[i]+moe[i] is recomputed
// in both passes (never written back), so ss, rinv and w*(t*rinv) match the
// separate kernels' f32 values exactly; the only removed work is the intermediate
// f32 store/reload (an identity). One block (grid {1,1,1}), like rms_norm.
extern "C" __global__ void moe_combine_tail(float* hidden, const float* shared,
                                            const float* moe, const float* w, RmsPush pc) {
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float t = shared[i] + moe[i];
        ss += t * t;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float t = shared[i] + moe[i];
        hidden[i] += w[i] * (t * rinv);
    }
}

// ---- moe_norm_combine_tail (gemma4-MoE decode: post_ffw_norm_2 + combine fused)
// Fuses the routed-experts post_ffw_norm_2 (rms_norm(moe, w_pn2)) with the combine
// tail (moe_combine_tail) — two ADJACENT single-block rms-style launches across a
// command boundary. The fused kernel reads `moe` RAW (the weighted-acc, NOT pre-
// normed): phase 1 norms it (== rms_norm(moe, w_pn2)) producing rinv1; phase 2
// forms t = shared + w_pn2*(moe*rinv1) and reduces ss2 (== moe_combine_tail's
// reduction over the post_ffw_norm_2 output); phase 3 writes hidden += w_post*
// (t*rinv2). BYTE-IDENTITY: the normed-moe value w_pn2[i]*(moe[i]*rinv1) is
// materialized in f32 shared memory, preserving the standalone kernel's store/load
// rounding boundary. The second norm and hidden update retain their original
// operation order. Removes one launch + the global moe_out_buf round trip. One
// block (grid {1,1,1}),
// block-count PRESERVED (both originals were single-block). Intervening
// __syncthreads make the zinc_block_reduce_sum scratch reuse race-free.
extern "C" __global__ void moe_norm_combine_tail(float* hidden, const float* shared,
                                                 const float* moe, const float* w_pn2,
                                                 const float* w_post, RmsPush pc) {
    // Materialize the first norm exactly once at f32 precision. The standalone
    // path writes this value to moe_out_buf before the combine kernel reads it;
    // shared memory preserves that rounding boundary without the global-memory
    // round trip or a second launch.
    extern __shared__ float normed_moe[];
    // phase 1: post_ffw_norm_2 over the raw weighted-acc moe (== rms_norm(moe,w_pn2))
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = moe[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv1 = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x)
        normed_moe[i] = w_pn2[i] * (moe[i] * rinv1);
    __syncthreads();

    // phase 2: t = shared + post_ffw_norm_2(moe); reduce ss2 (== moe_combine_tail)
    float ss2 = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float t = shared[i] + normed_moe[i];
        ss2 += t * t;
    }
    ss2 = zinc_block_reduce_sum(ss2);
    __shared__ float rms_inv_sh2;
    if (threadIdx.x == 0) rms_inv_sh2 = rsqrtf(ss2 / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv2 = rms_inv_sh2;

    // phase 3: hidden += post_ffw_norm(t)
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float t = shared[i] + normed_moe[i];
        hidden[i] += w_post[i] * (t * rinv2);
    }
}

// ---- sigmoid_scale_acc (port of sigmoid_scale_acc.comp) ---------------------
// a[i] += sigmoid(c[0]) * b[i]  (MoE shared-expert sigmoid gating). a read-write.
struct SigmoidAccPush { unsigned N; };

extern "C" __global__ void sigmoid_scale_acc(float* a, const float* b, const float* c, SigmoidAccPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    float gate = 1.0f / (1.0f + expf(-c[0]));
    a[idx] += gate * b[idx];
}

// ---- dmmv_f32 (port of dmmv_f32.comp) ---------------------------------------
// y[row] = sum_k W[row][k] * x[k], W f32. One block per output row. (Router,
// SSM alpha/beta projections.) Reuses DmmvPush; offsets in BYTES.
extern "C" __global__ void dmmv_f32(const float* w, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const float* wrow = w + (pc.a_offset >> 2) + (size_t)row * pc.K;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned k = threadIdx.x; k < pc.K; k += blockDim.x) sum += wrow[k] * xrow[k];
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// SSM alpha and beta are two [dt_rank,K] f32 matrices over the same normalized
// activation. Sharing the x load and launch is particularly valuable for their
// tiny 48-row grids, which otherwise under-fill the device twice per SSM layer.
extern "C" __global__ void dmmv_f32_dual(
    const float* __restrict__ wa, const float* __restrict__ wb,
    const float* __restrict__ x, float* __restrict__ ya,
    float* __restrict__ yb, DmmvPush pc)
{
    const unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const float* arow = wa + (size_t)row * pc.K;
    const float* brow = wb + (size_t)row * pc.K;
    float sa = 0.0f, sb = 0.0f;
    for (unsigned k = threadIdx.x; k < pc.K; k += blockDim.x) {
        const float xv = x[k];
        sa += arow[k] * xv;
        sb += brow[k] * xv;
    }
    sa = zinc_block_reduce_sum(sa);
    __syncthreads();
    sb = zinc_block_reduce_sum(sb);
    if (threadIdx.x == 0u) {
        ya[row] = sa;
        yb[row] = sb;
    }
}

// ---- dmmv_q8_0 (port of dmmv_q8_0.comp) -------------------------------------
// y[row] = sum_k dequant(W[row][k]) * x[k], W in Q8_0 (34-byte blocks: f16 d +
// 32 int8). dequant: d * qs[i]. (SSM in/out proj, shared-expert weights.)
extern "C" __global__ void dmmv_q8_0(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 5;                          // blocks per row (K / 32)
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 34u;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned blk = e >> 5;                          // which 32-elem block
        unsigned i = e & 31u;                           // 0..31 in block
        const unsigned char* blkp = arow + (size_t)blk * 34u;
        unsigned d_bits = (unsigned)blkp[0] | ((unsigned)blkp[1] << 8);
        float d = zinc_half_to_float((unsigned short)d_bits);
        signed char q = (signed char)blkp[2u + i];
        sum += (d * (float)q) * xrow[e];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- dmmv_q5_1 (gemma4 MoE ffn_down_exps) -----------------------------------
// y[row] = sum_k dequant(W[row][k]) * x[k], W in Q5_1 (24-byte blocks of 32:
// f16 d, f16 m, 4-byte qh high-bits, 16-byte qs nibbles). 5-bit q = nibble |
// (qh_bit<<4); value = d*q + m. (UD-Q4_K_M ships ffn_down_exps as Q5_1.)
extern "C" __global__ void dmmv_q5_1(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 5;                          // blocks per row (K / 32)
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 24u;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned blk = e >> 5;                          // which 32-elem block
        unsigned i = e & 31u;                           // 0..31 in block
        const unsigned char* blkp = arow + (size_t)blk * 24u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blkp[0] | ((unsigned)blkp[1] << 8)));
        float m = zinc_half_to_float((unsigned short)((unsigned)blkp[2] | ((unsigned)blkp[3] << 8)));
        unsigned qh = (unsigned)blkp[4] | ((unsigned)blkp[5] << 8) | ((unsigned)blkp[6] << 16) | ((unsigned)blkp[7] << 24);
        const unsigned char* qs = blkp + 8;
        unsigned nib = (i < 16u) ? (unsigned)(qs[i] & 0xFu) : (unsigned)(qs[i - 16u] >> 4);
        unsigned bit = (qh >> i) & 1u;
        unsigned q5 = nib | (bit << 4);
        sum += (d * (float)q5 + m) * xrow[e];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- dmmv_q5k (port of dmmv_q5k.comp) ---------------------------------------
// Q5_K block (256 elems, 176 bytes): [0..3] d+dmin(f16); [4..15] 6-bit scales;
// [16..47] qh (1 high bit/elem); [48..175] qs (4-bit low). 5-bit q = nibble +
// (qh_bit<<4); value = d*scale*q - dmin*min (same get_scale_min_k4 as Q4_K).
extern "C" __global__ void dmmv_q5k(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 176u;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned b = e >> 8, within = e & 255u;
        const unsigned char* blk = arow + (size_t)b * 176u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
        float dmin = zinc_half_to_float((unsigned short)((unsigned)blk[2] | ((unsigned)blk[3] << 8)));
        const unsigned char* scales = blk + 4;     // 12 bytes
        const unsigned char* qh = blk + 16;        // 32 bytes
        const unsigned char* qs = blk + 48;        // 128 bytes
        unsigned chunk = within >> 6;              // 0..3
        unsigned half = (within & 63u) >> 5;       // 0..1
        unsigned l = within & 31u;                 // 0..31
        unsigned char ql = qs[chunk * 32u + l];
        unsigned nib = (half == 0u) ? (ql & 0xFu) : (unsigned)(ql >> 4);
        unsigned bit = (qh[l] >> (2u * chunk + half)) & 1u;
        unsigned q5 = nib + (bit ? 16u : 0u);
        unsigned char sc, mn;
        zinc_q4k_scale_min((int)(chunk * 2u + half), scales, &sc, &mn);
        sum += (d * (float)sc * (float)q5 - dmin * (float)mn) * xrow[e];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- dmmv_q6k (port of dmmv_q6k.comp) ---------------------------------------
// Q6_K block (256 elems, 210 bytes): [0..127] ql (low 4 bits); [128..191] qh
// (high 2 bits); [192..207] scales (16 int8); [208..209] d(f16). 6-bit q =
// (ql_nibble | qh_bits<<4); value = d * int8_scale * (q - 32).
extern "C" __global__ void dmmv_q6k(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 210u;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned b = e >> 8, within = e & 255u;
        const unsigned char* blk = arow + (size_t)b * 210u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blk[208] | ((unsigned)blk[209] << 8)));
        unsigned half = within >> 7;               // 0..1 (128-elem half)
        unsigned wh = within & 127u;
        unsigned l = wh & 31u;                     // 0..31
        unsigned group = wh >> 5;                  // 0..3 (q1..q4)
        const unsigned char* ql = blk + (size_t)half * 64u;
        const unsigned char* qh = blk + 128u + (size_t)half * 32u;
        const signed char* sc = (const signed char*)(blk + 192u + (size_t)half * 8u);
        unsigned is = l >> 4;                       // 0 or 1
        unsigned qhb = qh[l];
        unsigned q; unsigned sci;
        if (group == 0u) { q = (ql[l] & 0xFu) | (((qhb >> 0) & 3u) << 4); sci = is + 0u; }
        else if (group == 1u) { q = (ql[l + 32u] & 0xFu) | (((qhb >> 2) & 3u) << 4); sci = is + 2u; }
        else if (group == 2u) { q = (ql[l] >> 4) | (((qhb >> 4) & 3u) << 4); sci = is + 4u; }
        else { q = (ql[l + 32u] >> 4) | (((qhb >> 6) & 3u) << 4); sci = is + 6u; }
        sum += (d * (float)sc[sci] * ((float)q - 32.0f)) * xrow[e];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- softmax_topk (port of softmax_topk.comp) -------------------------------
// MoE router: pick top-k experts from logits, output [ids(k) | renorm-softmax
// weights(k)] (weights as float-bits). Single block of 64 threads, shared-mem
// winner select (no ballot). out[0..k-1]=ids, out[k..2k-1]=weight bits.
struct TopkPush { unsigned n_experts; unsigned k; };

extern "C" __global__ void softmax_topk(const float* logits, unsigned* out, TopkPush pc) {
    __shared__ float s_logits[256];
    __shared__ float s_val[64];
    __shared__ unsigned s_idx[64];
    const float NEG_INF = __int_as_float(0xff800000);
    unsigned tid = threadIdx.x;
    for (unsigned i = tid; i < pc.n_experts; i += 64) s_logits[i] = logits[i];
    __syncthreads();
    for (unsigned ki = 0; ki < pc.k; ki++) {
        float best = NEG_INF; unsigned bidx = 0;
        for (unsigned i = tid; i < pc.n_experts; i += 64)
            if (s_logits[i] > best) { best = s_logits[i]; bidx = i; }
        s_val[tid] = best; s_idx[tid] = bidx;
        __syncthreads();
        if (tid == 0) {
            float gb = NEG_INF; unsigned gi = 0;
            for (unsigned t = 0; t < 64; t++)
                if (s_val[t] > gb) { gb = s_val[t]; gi = s_idx[t]; }
            out[ki] = gi;
            out[pc.k + ki] = __float_as_uint(gb);
            s_logits[gi] = NEG_INF;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float maxl = NEG_INF;
        for (unsigned i = 0; i < pc.k; i++) maxl = fmaxf(maxl, __uint_as_float(out[pc.k + i]));
        float wsum = 0.0f;
        for (unsigned i = 0; i < pc.k; i++) {
            float w = expf(__uint_as_float(out[pc.k + i]) - maxl);
            out[pc.k + i] = __float_as_uint(w); wsum += w;
        }
        float inv = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
        for (unsigned i = 0; i < pc.k; i++)
            out[pc.k + i] = __float_as_uint(__uint_as_float(out[pc.k + i]) * inv);
    }
}

// ---- softmax_topk_batched (gemma4-MoE prefill router over all T tokens) ------
// Verbatim twin of softmax_topk, batched over queries: block t (blockIdx.x) reads
// its own logits row [t*n_experts ..] and writes its own out row [t*2k ..]. The
// per-block winner-select / renorm-softmax math is byte-for-byte softmax_topk's,
// so per-token routing is identical to the per-token path — just one launch over
// all T tokens instead of T launches. Used by `routerBatched`.
extern "C" __global__ void softmax_topk_batched(const float* logits, unsigned* out, TopkPush pc) {
    __shared__ float s_logits[256];
    __shared__ float s_val[64];
    __shared__ unsigned s_idx[64];
    const float NEG_INF = __int_as_float(0xff800000);
    unsigned tid = threadIdx.x;
    const float* lt = logits + (size_t)blockIdx.x * pc.n_experts;
    unsigned* ot = out + (size_t)blockIdx.x * 2u * pc.k;
    for (unsigned i = tid; i < pc.n_experts; i += 64) s_logits[i] = lt[i];
    __syncthreads();
    for (unsigned ki = 0; ki < pc.k; ki++) {
        float best = NEG_INF; unsigned bidx = 0;
        for (unsigned i = tid; i < pc.n_experts; i += 64)
            if (s_logits[i] > best) { best = s_logits[i]; bidx = i; }
        s_val[tid] = best; s_idx[tid] = bidx;
        __syncthreads();
        if (tid == 0) {
            float gb = NEG_INF; unsigned gi = 0;
            for (unsigned t = 0; t < 64; t++)
                if (s_val[t] > gb) { gb = s_val[t]; gi = s_idx[t]; }
            ot[ki] = gi;
            ot[pc.k + ki] = __float_as_uint(gb);
            s_logits[gi] = NEG_INF;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float maxl = NEG_INF;
        for (unsigned i = 0; i < pc.k; i++) maxl = fmaxf(maxl, __uint_as_float(ot[pc.k + i]));
        float wsum = 0.0f;
        for (unsigned i = 0; i < pc.k; i++) {
            float w = expf(__uint_as_float(ot[pc.k + i]) - maxl);
            ot[pc.k + i] = __float_as_uint(w); wsum += w;
        }
        float inv = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
        for (unsigned i = 0; i < pc.k; i++)
            ot[pc.k + i] = __float_as_uint(__uint_as_float(ot[pc.k + i]) * inv);
    }
}

// ---- rope (port of rope_fused.comp) -----------------------------------------
// RoPE with partial rotation (IMRoPE): rotate first rope_dim dims/head in pairs,
// copy the rest. freq from inv_freq buffer when freq_base_bits==0, else computed.
// One block per head.
struct RopePush { unsigned stride, rope_dim, n_heads, position, freq_base_bits, attn_scale_bits; };

extern "C" __global__ void rope(const float* x, float* y, const float* inv_freq, RopePush pc) {
    unsigned tid = threadIdx.x;
    unsigned head = blockIdx.x;
    unsigned base = head * pc.stride;
    unsigned half_rot = pc.rope_dim >> 1;
    float freq_base = __uint_as_float(pc.freq_base_bits);
    float attn_scale = pc.attn_scale_bits != 0u ? __uint_as_float(pc.attn_scale_bits) : 1.0f;
    bool use_buf = (pc.freq_base_bits == 0u);
    for (unsigned i = tid; i < half_rot; i += blockDim.x) {
        float xi = x[base + i];
        float xih = x[base + i + half_rot];
        float freq_i = use_buf ? inv_freq[i]
                               : (1.0f / powf(freq_base, (float)(2u * i) / (float)pc.rope_dim));
        float theta = (float)pc.position * freq_i;
        float ct = cosf(theta) * attn_scale;
        float st = sinf(theta) * attn_scale;
        y[base + i] = xi * ct - xih * st;
        y[base + i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + tid; i < pc.stride; i += blockDim.x)
        y[base + i] = x[base + i];
}

// ---- argmax (port of argmax.comp) -------------------------------------------
// Greedy sample: token_id = argmax_i logits[i] (lowest index wins ties).
// Single-block reduction (the .comp's two-phase split is a perf detail for huge N).
struct ArgmaxPush { unsigned N; };

extern "C" __global__ void argmax(const float* logits, unsigned* token_id, ArgmaxPush pc) {
    __shared__ float s_val[32];
    __shared__ unsigned s_idx[32];
    unsigned tid = threadIdx.x;
    float best = -3.4e38f; unsigned bidx = 0u;
    for (unsigned i = tid; i < pc.N; i += blockDim.x) {
        float v = logits[i];
        if (v > best) { best = v; bidx = i; }
    }
    unsigned lane = tid & 31u, wid = tid >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, best, o);
        unsigned oi = __shfl_down_sync(0xffffffffu, bidx, o);
        if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
    }
    if (lane == 0u) { s_val[wid] = best; s_idx[wid] = bidx; }
    __syncthreads();
    if (tid == 0u) {
        float gb = s_val[0]; unsigned gi = s_idx[0];
        unsigned nwarps = (blockDim.x + 31u) >> 5;
        for (unsigned w = 1u; w < nwarps; w++)
            if (s_val[w] > gb || (s_val[w] == gb && s_idx[w] < gi)) { gb = s_val[w]; gi = s_idx[w]; }
        *token_id = gi;
    }
}

struct ArgmaxV2Push { unsigned N, partials; };

extern "C" __global__ void argmax_partials(
    const float* __restrict__ logits, unsigned char* __restrict__ scratch,
    ArgmaxV2Push pc)
{
    __shared__ float s_val[32];
    __shared__ unsigned s_idx[32];
    const unsigned tid = threadIdx.x;
    const unsigned stride = pc.partials * blockDim.x;
    float best = -3.4e38f;
    unsigned bidx = 0u;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < pc.N; i += stride) {
        const float v = logits[i];
        if (v > best || (v == best && i < bidx)) {
            best = v;
            bidx = i;
        }
    }
    const unsigned lane = tid & 31u, wid = tid >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best, o);
        const unsigned oi = __shfl_down_sync(0xffffffffu, bidx, o);
        if (ov > best || (ov == best && oi < bidx)) {
            best = ov;
            bidx = oi;
        }
    }
    if (lane == 0u) {
        s_val[wid] = best;
        s_idx[wid] = bidx;
    }
    __syncthreads();
    if (tid == 0u) {
        float gb = s_val[0];
        unsigned gi = s_idx[0];
        for (unsigned w = 1u; w < 8u; w++) {
            if (s_val[w] > gb || (s_val[w] == gb && s_idx[w] < gi)) {
                gb = s_val[w];
                gi = s_idx[w];
            }
        }
        float* values = (float*)scratch;
        unsigned* indices = (unsigned*)(values + pc.partials);
        values[blockIdx.x] = gb;
        indices[blockIdx.x] = gi;
    }
}

extern "C" __global__ void argmax_finalize(
    const unsigned char* __restrict__ scratch, unsigned* __restrict__ token_id,
    ArgmaxV2Push pc)
{
    __shared__ float s_val[4];
    __shared__ unsigned s_idx[4];
    const float* values = (const float*)scratch;
    const unsigned* indices = (const unsigned*)(values + pc.partials);
    const unsigned tid = threadIdx.x;
    float best = tid < pc.partials ? values[tid] : -3.4e38f;
    unsigned bidx = tid < pc.partials ? indices[tid] : 0u;
    const unsigned lane = tid & 31u, wid = tid >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best, o);
        const unsigned oi = __shfl_down_sync(0xffffffffu, bidx, o);
        if (ov > best || (ov == best && oi < bidx)) {
            best = ov;
            bidx = oi;
        }
    }
    if (lane == 0u) {
        s_val[wid] = best;
        s_idx[wid] = bidx;
    }
    __syncthreads();
    if (tid == 0u) {
        float gb = s_val[0];
        unsigned gi = s_idx[0];
        for (unsigned w = 1u; w < 4u; w++) {
            if (s_val[w] > gb || (s_val[w] == gb && s_idx[w] < gi)) {
                gb = s_val[w];
                gi = s_idx[w];
            }
        }
        *token_id = gi;
    }
}

// ---- embed_lookup_q4k (Effort 25 cycle 5: GPU-side embedding dequant) -------
// Dequantize one Q4_K row of token_embd.weight (the row for token `tok[0]`) into
// out[0..K], replacing the per-token CPU dequant + full-row H2D with a GPU
// dispatch reading the token id from a tiny device buffer (so it captures as the
// decode graph's first node). Bit-identical math to the CPU `dequantRow` Q4_K
// path: 144-byte superblocks, d/dmin f16, 12 scale bytes, 128 quant bytes;
// per 256-block output order [g0_lo32, g0_hi32, g1_lo32, ...], value =
// (d*sc) * nibble - (dmin*m), scale sub-index = 2*group + half (low/high nibble).
// Grid: one block per superblock (K/256). Block: 256 threads, one output each.
struct EmbedPush { unsigned K; unsigned vocab; };

extern "C" __global__ void embed_lookup_q4k(const unsigned char* W,
                                            const unsigned* tok, float* out,
                                            EmbedPush pc) {
    unsigned t = tok[0];
    unsigned vmax = pc.vocab ? pc.vocab - 1u : 0u;
    if (t > vmax) t = vmax;
    unsigned nsb = pc.K >> 8;          // superblocks per row (K / 256)
    unsigned sb = blockIdx.x;
    if (sb >= nsb) return;
    const unsigned char* blk = W + ((size_t)t * nsb + sb) * 144u;
    float d = zinc_half_to_float(*(const unsigned short*)(blk + 0));
    float dmin = zinc_half_to_float(*(const unsigned short*)(blk + 2));
    const unsigned char* scales = blk + 4;
    const unsigned char* qs = blk + 16;
    unsigned idx = threadIdx.x;        // 0..255 output element within the superblock
    if (idx >= 256u) return;
    unsigned g = idx >> 6;             // group 0..3 (each 64 outputs)
    unsigned h = (idx >> 5) & 1u;      // 0 = low nibble half, 1 = high nibble half
    unsigned l = idx & 31u;
    unsigned char sc, m;
    zinc_q4k_scale_min((int)(2u * g + h), scales, &sc, &m);
    unsigned char qb = qs[g * 32u + l];
    unsigned nib = (h == 0u) ? (qb & 0xFu) : (unsigned)(qb >> 4);
    out[sb * 256u + idx] = (d * (float)sc) * (float)nib - (dmin * (float)m);
}

// ---- moe_weighted_acc (port of moe_weighted_acc.comp) -----------------------
// a[i] += sum_j weight_j * b[j*src_stride + i]. Weights from the softmax_topk
// routing buffer (routing[n_used .. 2*n_used-1], as float bits).
struct MoeAccPush { unsigned N, n_used, src_stride; };

extern "C" __global__ void moe_weighted_acc(float* a, const float* b, const unsigned* routing, MoeAccPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.N) return;
    float sum = 0.0f;
    for (unsigned j = 0; j < pc.n_used; j++) {
        float w = __uint_as_float(routing[pc.n_used + j]);
        sum += w * b[(size_t)j * pc.src_stride + i];
    }
    a[i] += sum;
}

// ---- moe_weighted_acc_scaled (gemma4 batched MoE, GPU-side down scale) -------
// Same weighted combine as moe_weighted_acc, but folds the per-expert down scale
// (ffn_down_exps.scale[id]) into the weight GPU-side: w_j = weight_j * escale[id_j].
// id_j = routing[j], weight_j = routing[n_used+j]. This removes the per-layer host
// readback gemma previously used to fold the scale into the router weights, so the
// whole batched MoE block can run async on the stream (GPU stays at boost).
extern "C" __global__ void moe_weighted_acc_scaled(float* a, const float* b, const unsigned* routing, const float* escale, MoeAccPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.N) return;
    float sum = 0.0f;
    for (unsigned j = 0; j < pc.n_used; j++) {
        float w = __uint_as_float(routing[pc.n_used + j]) * escale[routing[j]];
        sum += w * b[(size_t)j * pc.src_stride + i];
    }
    a[i] += sum;
}

// ---- moe_weighted_acc_scaled_batched (gemma4-MoE prefill combine, all T) -----
// Token-batched twin of moe_weighted_acc_scaled (Effort 24 cycle 9): one launch
// (grid.y = T) does the routed-expert weighted combine for every prompt token.
// Block (blockIdx.x → output channel i, blockIdx.y → token t) reads token t's own
// accumulator slice a[t*a_tok_stride], down slice b[t*b_tok_stride], and routing
// row routing[t*routing_stride] — so the per-(t,i) math is byte-for-byte the
// single-token kernel's (same j-loop, same FMA order, same GPU-side down scale).
struct MoeAccBatchPush { unsigned N, n_used, src_stride, a_tok_stride, b_tok_stride, routing_stride; };
extern "C" __global__ void moe_weighted_acc_scaled_batched(float* a, const float* b, const unsigned* routing, const float* escale, MoeAccBatchPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.N) return;
    unsigned t = blockIdx.y;
    float* at = a + (size_t)t * pc.a_tok_stride;
    const float* bt = b + (size_t)t * pc.b_tok_stride;
    const unsigned* rt = routing + (size_t)t * pc.routing_stride;
    float sum = 0.0f;
    for (unsigned j = 0; j < pc.n_used; j++) {
        float w = __uint_as_float(rt[pc.n_used + j]) * escale[rt[j]];
        sum += w * bt[(size_t)j * pc.src_stride + i];
    }
    at[i] += sum;
}

// ---- ssm_conv1d (port of ssm_conv1d.comp) -----------------------------------
// Depthwise causal 1D conv (d_conv taps) + SiLU, via a circular state buffer.
// One thread per channel. Updates `state` in place (writes current_input into
// the state_offset slot). conv_kernel is f16 or f32 per kernel_is_f16.
struct ConvPush { unsigned conv_channels, d_conv, kernel_is_f16, state_offset; };

extern "C" __global__ void ssm_conv1d(const float* current_input, const unsigned char* conv_kernel,
                                      float* state, float* out_data, ConvPush pc) {
    unsigned ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= pc.conv_channels) return;
    unsigned d_conv_1 = pc.d_conv - 1u;
    float ci = current_input[ch];
    float sum = 0.0f;
    for (unsigned ki = 0; ki < pc.d_conv; ki++) {
        unsigned k_idx = ch * pc.d_conv + ki;
        float kw = (pc.kernel_is_f16 != 0u)
                       ? zinc_half_to_float(((const unsigned short*)conv_kernel)[k_idx])
                       : ((const float*)conv_kernel)[k_idx];
        float sv;
        if (ki < d_conv_1) {
            unsigned slot = pc.state_offset + ki;
            if (slot >= d_conv_1) slot -= d_conv_1;
            sv = state[(size_t)slot * pc.conv_channels + ch];
        } else {
            sv = ci;
        }
        sum += kw * sv;
    }
    out_data[ch] = sum / (1.0f + expf(-sum));            // SiLU
    state[(size_t)pc.state_offset * pc.conv_channels + ch] = ci;
}

// Decode fusion for the Qwen gated-delta-net path. The scan preparation needs
// every convolution result before it can normalize Q/K, but a single workgroup
// can produce one complete Q/K group, perform the identical reductions, and
// activate that group's alpha/beta scalars. The same grid also covers V. This
// preserves the standalone math while removing one launch and a Q/K round trip
// for every SSM layer.
struct ConvPreparePush {
    unsigned conv_channels, d_conv, kernel_is_f16, state_offset;
    unsigned dt_rank, d_inner, d_state, n_group;
    unsigned ssm_a_is_f16, dt_bias_is_f16;
};

__device__ __forceinline__ float zinc_ssm_conv_silu(
    unsigned ch, const float* current_input, const unsigned char* conv_kernel,
    float* state, ConvPreparePush pc)
{
    const unsigned d_conv_1 = pc.d_conv - 1u;
    const float ci = current_input[ch];
    float sum = 0.0f;
    #pragma unroll
    for (unsigned ki = 0u; ki < pc.d_conv; ++ki) {
        const unsigned k_idx = ch * pc.d_conv + ki;
        const float kw = pc.kernel_is_f16 != 0u
            ? zinc_half_to_float(((const unsigned short*)conv_kernel)[k_idx])
            : ((const float*)conv_kernel)[k_idx];
        float sv = ci;
        if (ki < d_conv_1) {
            unsigned slot = pc.state_offset + ki;
            if (slot >= d_conv_1) slot -= d_conv_1;
            sv = state[(size_t)slot * pc.conv_channels + ch];
        }
        sum += kw * sv;
    }
    state[(size_t)pc.state_offset * pc.conv_channels + ch] = ci;
    return sum / (1.0f + expf(-sum));
}

extern "C" __global__ void ssm_conv1d_prepare(
    const float* __restrict__ current_input,
    const unsigned char* __restrict__ conv_kernel,
    float* __restrict__ state, float* __restrict__ out_data,
    const unsigned char* __restrict__ dt_bias,
    float* __restrict__ alpha, float* __restrict__ beta,
    const unsigned char* __restrict__ ssm_a, ConvPreparePush pc)
{
    const unsigned group = blockIdx.x;
    const unsigned lane = threadIdx.x;
    const unsigned qk_dim = pc.d_state * pc.n_group;

    if (group < pc.n_group) {
        const unsigned q_ch = group * pc.d_state + lane;
        const unsigned k_ch = qk_dim + group * pc.d_state + lane;
        const float q_val = lane < pc.d_state
            ? zinc_ssm_conv_silu(q_ch, current_input, conv_kernel, state, pc)
            : 0.0f;
        const float k_val = lane < pc.d_state
            ? zinc_ssm_conv_silu(k_ch, current_input, conv_kernel, state, pc)
            : 0.0f;
        const float sumq = zinc_block_reduce_sum_all(q_val * q_val);
        const float sumk = zinc_block_reduce_sum_all(k_val * k_val);
        const float q_rinv = rsqrtf(fmaxf(sumq, 1e-12f)) / sqrtf((float)pc.d_state);
        const float k_rinv = rsqrtf(fmaxf(sumk, 1e-12f));
        if (lane < pc.d_state) {
            out_data[q_ch] = q_val * q_rinv;
            out_data[k_ch] = k_val * k_rinv;
        }

        if (lane == 0u) {
            for (unsigned h = group; h < pc.dt_rank; h += pc.n_group) {
                const float dt_bias_val = pc.dt_bias_is_f16 != 0u
                    ? zinc_half_to_float(((const unsigned short*)dt_bias)[h])
                    : ((const float*)dt_bias)[h];
                const float ssm_a_val = pc.ssm_a_is_f16 != 0u
                    ? zinc_half_to_float(((const unsigned short*)ssm_a)[h])
                    : ((const float*)ssm_a)[h];
                const float sp = logf(1.0f + expf(alpha[h] + dt_bias_val));
                alpha[h] = expf(sp * ssm_a_val);
                beta[h] = 1.0f / (1.0f + expf(-beta[h]));
            }
        }
    }

    const unsigned v = group * blockDim.x + lane;
    if (v < pc.d_inner) {
        const unsigned v_ch = 2u * qk_dim + v;
        out_data[v_ch] = zinc_ssm_conv_silu(v_ch, current_input, conv_kernel, state, pc);
    }
}

// ---- ssm_gated_norm (port of ssm_gated_norm.comp) ---------------------------
// Per head: out = (o / rms(o)) * norm_weight * silu(z). One block per head.
struct GatedNormPush { unsigned d_inner, dt_rank, head_v_dim, d_state, norm_per_head; };

extern "C" __global__ void ssm_gated_norm(const float* o, const float* z, const float* norm_weight,
                                          float* out, GatedNormPush pc) {
    unsigned h = blockIdx.x;
    unsigned base = h * pc.head_v_dim;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.head_v_dim; i += blockDim.x) { float v = o[base + i]; ss += v * v; }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.head_v_dim + 1e-6f);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < pc.head_v_dim; i += blockDim.x) {
        float nv = o[base + i] * rinv;
        unsigned norm_idx = (pc.norm_per_head != 0u) ? (base + i) : (i % pc.d_state);
        nv *= norm_weight[norm_idx];
        float zv = z[base + i];
        out[base + i] = nv * (zv / (1.0f + expf(-zv)));
    }
}

// Decode-only fused twin for a Q8-consuming SSM output projection. Qwen's
// 128-value SSM heads map exactly to four Q8_1 groups, so the normalization and
// gate result can be reduced and packed in the same block without a temporary
// f32 write followed by a separate quantizer launch.
extern "C" __global__ void ssm_gated_norm_quant_q8_0(
    const float* __restrict__ o, const float* __restrict__ z,
    const float* __restrict__ norm_weight, unsigned char* __restrict__ out,
    GatedNormPush pc)
{
    const unsigned h = blockIdx.x;
    const unsigned i = threadIdx.x;
    const unsigned base = h * pc.head_v_dim;
    float ss = 0.0f;
    if (i < pc.head_v_dim) {
        const float v = o[base + i];
        ss = v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (i == 0u) rms_inv_sh = rsqrtf(ss / (float)pc.head_v_dim + 1e-6f);
    __syncthreads();

    float val = 0.0f;
    if (i < pc.head_v_dim) {
        const unsigned norm_idx = pc.norm_per_head != 0u ? base + i : i % pc.d_state;
        float nv = o[base + i] * rms_inv_sh;
        nv *= norm_weight[norm_idx];
        const float zv = z[base + i];
        val = nv * (zv / (1.0f + expf(-zv)));
    }

    float av = fabsf(val), sv = val;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, off));
        sv += __shfl_xor_sync(0xffffffffu, sv, off);
    }
    const float d = av / 127.0f;
    const float scale = 127.0f / fmaxf(av, 1e-5f);
    const int q = max(-127, min(127, __float2int_rn(val * scale)));
    const unsigned lane = i & 31u;
    const unsigned c = (base + i) >> 5;
#ifdef ZINC_ROCM
    const unsigned tile = c >> 2;
    const unsigned group = c & 3u;
    unsigned char* out_base = out + (size_t)tile * 144u;
    if (lane == 0u) {
        const unsigned short dh = zinc_float_to_half(d);
        const unsigned short sh = zinc_float_to_half(sv);
        out_base[group * 4u] = (unsigned char)(dh & 0xffu);
        out_base[group * 4u + 1u] = (unsigned char)(dh >> 8);
        out_base[group * 4u + 2u] = (unsigned char)(sh & 0xffu);
        out_base[group * 4u + 3u] = (unsigned char)(sh >> 8);
    }
    out_base[16u + group * 32u + lane] = (unsigned char)q;
#else
    unsigned char* out_base = out + (size_t)c * 36u;
    if (lane == 0u) {
        const unsigned short dh = zinc_float_to_half(d);
        const unsigned short sh = zinc_float_to_half(sv);
        out_base[0] = (unsigned char)(dh & 0xffu);
        out_base[1] = (unsigned char)(dh >> 8);
        out_base[2] = (unsigned char)(sh & 0xffu);
        out_base[3] = (unsigned char)(sh >> 8);
    }
    out_base[4u + lane] = (unsigned char)q;
#endif
}

// ---- Effort 26 T0: BATCHED qwen prefill kernels ----------------------------
// These collapse the per-token prefill launches into one launch over all T
// tokens (token-major buffers). Each is a bit-identical twin of the single-token
// kernel above — same math, same circular-state evolution — so batched prefill
// equals per-token prefill output (the GEMMs ride the token-correctness gate).

// y[i] += x[i] over N elements. Used to fold a batched projection (cuBLAS / the
// matvec fallback both write a FRESH output) into the residual stream.
struct AddPush { unsigned N; };
extern "C" __global__ void add_inplace(float* y, const float* x, AddPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pc.N) y[i] += x[i];
}

// Batched ssm_conv1d: one block-row per channel, loop t = 0..n_tok internally
// so the circular conv-state advances exactly as the per-token launches did.
// input/out are token-major [n_tok, conv_channels]; state is the shared circular
// window. Collapses n_tok launches → 1 (recurrence preserved = bit-identical).
struct ConvBatchPush { unsigned conv_channels, d_conv, kernel_is_f16, n_tok, state_offset; };
extern "C" __global__ void ssm_conv1d_batched(const float* input, const unsigned char* conv_kernel,
                                              float* state, float* out_data, ConvBatchPush pc) {
    unsigned ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= pc.conv_channels) return;
    unsigned d_conv_1 = pc.d_conv - 1u;
    unsigned off = pc.state_offset;
    for (unsigned t = 0; t < pc.n_tok; t++) {
        float ci = input[(size_t)t * pc.conv_channels + ch];
        float sum = 0.0f;
        for (unsigned ki = 0; ki < pc.d_conv; ki++) {
            unsigned k_idx = ch * pc.d_conv + ki;
            float kw = (pc.kernel_is_f16 != 0u)
                           ? zinc_half_to_float(((const unsigned short*)conv_kernel)[k_idx])
                           : ((const float*)conv_kernel)[k_idx];
            float sv;
            if (ki < d_conv_1) {
                unsigned slot = off + ki;
                if (slot >= d_conv_1) slot -= d_conv_1;
                sv = state[(size_t)slot * pc.conv_channels + ch];
            } else {
                sv = ci;
            }
            sum += kw * sv;
        }
        out_data[(size_t)t * pc.conv_channels + ch] = sum / (1.0f + expf(-sum)); // SiLU
        state[(size_t)off * pc.conv_channels + ch] = ci;
        off += 1u;
        if (off >= d_conv_1) off -= d_conv_1;
    }
}

// Batched ssm_gated_norm: grid (dt_rank heads, n_tok). Stateless per (head,token),
// so a plain grid.y over tokens. Token-major o/z/out [n_tok, d_inner]; the
// norm-weight index is the WITHIN-token feature index (no token offset).
struct GatedNormBatchPush { unsigned d_inner, dt_rank, head_v_dim, d_state, norm_per_head, n_tok; };
extern "C" __global__ void ssm_gated_norm_batched(const float* o, const float* z, const float* norm_weight,
                                                  float* out, GatedNormBatchPush pc) {
    unsigned h = blockIdx.x;
    unsigned t = blockIdx.y;
    unsigned base = (size_t)t * pc.d_inner + h * pc.head_v_dim;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.head_v_dim; i += blockDim.x) { float v = o[base + i]; ss += v * v; }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.head_v_dim + 1e-6f);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < pc.head_v_dim; i += blockDim.x) {
        float nv = o[base + i] * rinv;
        unsigned feat = h * pc.head_v_dim + i;
        unsigned norm_idx = (pc.norm_per_head != 0u) ? feat : (i % pc.d_state);
        nv *= norm_weight[norm_idx];
        float zv = z[base + i];
        out[base + i] = nv * (zv / (1.0f + expf(-zv)));
    }
}

// ---- kv_cache_write (port of kv_cache_write.comp) ---------------------------
// Append K and V vectors into their caches at dst_offset (= physical_token*kv_dim).
struct KvWritePush { unsigned kv_dim, dst_offset; };

extern "C" __global__ void kv_cache_write(const float* k_src, float* k_dst, const float* v_src, float* v_dst, KvWritePush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pc.kv_dim) {
        k_dst[pc.dst_offset + i] = k_src[i];
        v_dst[pc.dst_offset + i] = v_src[i];
    }
}

// ---- naive_attention (decode: softmax(QK^T)V, GQA + attention sinks) --------
// Correctness-first decode attention (the flash/paged version is M3). One block
// per query head, contiguous KV cache [seq_len, n_kv_heads, head_dim]. Dynamic
// shared holds seq_len scores. Sink: sinks[sink_offset+head] (NaN = no sink).
struct AttnPush { unsigned head_dim, n_heads, n_kv_heads, seq_len, attn_scale_bits, sink_offset; };

extern "C" __global__ void naive_attention(const float* q, const float* k, const float* v,
                                           const float* sinks, float* out, AttnPush pc) {
    extern __shared__ float s_scores[];           // size = seq_len floats (dynamic)
    __shared__ float s_m, s_rescale, s_inv;
    unsigned head = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned hd = pc.head_dim;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const float* qh = q + (size_t)head * hd;
    float scale = pc.attn_scale_bits != 0u ? __uint_as_float(pc.attn_scale_bits) : rsqrtf((float)hd);

    // Pass 1: scores = scale * (q . k_i), track max.
    float lmax = -3.4e38f;
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x) {
        const float* ki = k + ((size_t)i * pc.n_kv_heads + kv_head) * hd;
        float dot = 0.0f;
        if ((hd & 3u) == 0u) {
            const float4* q4 = (const float4*)qh;
            const float4* k4 = (const float4*)ki;
            const unsigned nd4 = hd >> 2;
            float acc4[8] = {};
            unsigned d4 = 0u;
            for (; d4 + 8u <= nd4; d4 += 8u) {
                #pragma unroll
                for (unsigned j = 0u; j < 8u; ++j)
                    acc4[j] += zinc_float4_dot(q4[d4 + j], k4[d4 + j]);
            }
            dot = ((acc4[0] + acc4[4]) + (acc4[2] + acc4[6]))
                + ((acc4[1] + acc4[5]) + (acc4[3] + acc4[7]));
            for (; d4 < nd4; ++d4) dot += zinc_float4_dot(q4[d4], k4[d4]);
        } else {
            for (unsigned d = 0; d < hd; ++d) dot += qh[d] * ki[d];
        }
        float score = dot * scale;
        s_scores[i] = score;
        lmax = fmaxf(lmax, score);
    }
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0) s_m = lmax;
    __syncthreads();
    float m = s_m;

    // Pass 2: e_i = exp(score_i - m), sum.
    float lsum = 0.0f;
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x) {
        float e = expf(s_scores[i] - m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) {
        float sum = lsum, rescale = 1.0f, final_sum = lsum;
        float sink_val = sinks[pc.sink_offset + head];
        if (sink_val == sink_val) {   // sink present (NaN == NaN is false)
            float sink_max = fmaxf(m, sink_val);
            rescale = (sum > 0.0f) ? expf(m - sink_max) : 0.0f;
            final_sum = sum * rescale + expf(sink_val - sink_max);
        }
        s_rescale = rescale;
        s_inv = (final_sum > 0.0f) ? 1.0f / final_sum : 0.0f;
    }
    __syncthreads();
    float rescale = s_rescale, inv = s_inv;

    // Pass 3: out[d] = (sum_i e_i * V[i,d]) * rescale * inv.
    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
        const float* vh = v + (size_t)kv_head * hd + d;
        const size_t kv_stride = (size_t)pc.n_kv_heads * hd;
        unsigned i = 0u;
        for (; i + 8u <= pc.seq_len; i += 8u) {
            acc0 += s_scores[i] * vh[(size_t)i * kv_stride];
            acc1 += s_scores[i + 1u] * vh[(size_t)(i + 1u) * kv_stride];
            acc2 += s_scores[i + 2u] * vh[(size_t)(i + 2u) * kv_stride];
            acc3 += s_scores[i + 3u] * vh[(size_t)(i + 3u) * kv_stride];
            acc4 += s_scores[i + 4u] * vh[(size_t)(i + 4u) * kv_stride];
            acc5 += s_scores[i + 5u] * vh[(size_t)(i + 5u) * kv_stride];
            acc6 += s_scores[i + 6u] * vh[(size_t)(i + 6u) * kv_stride];
            acc7 += s_scores[i + 7u] * vh[(size_t)(i + 7u) * kv_stride];
        }
        float acc = ((acc0 + acc4) + (acc2 + acc6))
            + ((acc1 + acc5) + (acc3 + acc7));
        for (; i < pc.seq_len; ++i)
            acc += s_scores[i] * vh[(size_t)i * kv_stride];
        out[(size_t)head * hd + d] = acc * rescale * inv;
    }
}

// ---- ssm_delta_net (port of ssm_delta_net.comp) -----------------------------
// Gated delta-net autoregressive selective scan. One block per (head,row), one
// state column per thread (blockDim == head_v_dim, must be a multiple of 32).
// State [dt_rank][head_v_dim][head_v_dim] carries across the n_tok token loop:
// per token — L2-norm Q/K per group, gate=exp(softplus(alpha+dt_bias)*ssm_a),
// beta=sigmoid; decay state*=gate; sk=<state,k>; d=beta*(v-sk); state+=k*d;
// readout o=<state,q>. Unfused scalar form (perf/coopmat fusion is M3).
struct DeltaNetPush {
    unsigned d_inner, dt_rank, head_v_dim, d_state, n_group;
    unsigned ssm_a_is_f16, dt_bias_is_f16, has_dt_bias, has_ssm_a;
    unsigned n_tok, conv_stride_tok, ab_stride_tok, y_stride_tok;
    unsigned preprocessed;
};

extern "C" __global__ void ssm_delta_net(
    const float* conv_out, const unsigned char* dt_bias, const float* alpha,
    const float* beta, const unsigned char* ssm_a, float* state, float* out_data,
    DeltaNetPush pc) {
    // State stored transposed: state[h][col][row] — col-major for coalesced
    // warp access in ssm_delta_net_warp. Block kernel adapts by swapping
    // row/col roles: blockIdx.y = col, threadIdx.x = row.
    unsigned h = blockIdx.x;
    unsigned col = blockIdx.y;
    unsigned row = threadIdx.x;                  // 0..head_v_dim-1
    if (h >= pc.dt_rank || col >= pc.head_v_dim) return;
    unsigned hv = pc.head_v_dim;
    unsigned qk_dim = pc.d_state * pc.n_group;
    unsigned k_len = (hv < pc.d_state) ? hv : pc.d_state;
    size_t col_base = ((size_t)h * hv + col) * hv;
    float rs = state[col_base + row];            // state[h][col][row]

    float dt_bias_val = 0.0f;
    if (pc.has_dt_bias != 0u)
        dt_bias_val = pc.dt_bias_is_f16 ? zinc_half_to_float(((const unsigned short*)dt_bias)[h])
                                        : ((const float*)dt_bias)[h];
    float ssm_a_val = 0.0f;
    if (pc.has_ssm_a != 0u)
        ssm_a_val = pc.ssm_a_is_f16 ? zinc_half_to_float(((const unsigned short*)ssm_a)[h])
                                    : ((const float*)ssm_a)[h];
    unsigned k_hi = (pc.n_group == pc.dt_rank) ? h : (h % pc.n_group);

    __shared__ float s_g, s_b;

    for (unsigned t = 0; t < pc.n_tok; t++) {
        unsigned conv_base = t * pc.conv_stride_tok;
        unsigned q_off = conv_base + k_hi * pc.d_state;
        unsigned k_off = conv_base + qk_dim + k_hi * pc.d_state;
        unsigned v_off = conv_base + 2u * qk_dim + h * hv;

        float sq = (row < k_len) ? conv_out[q_off + row] : 0.0f;
        float skv = (row < k_len) ? conv_out[k_off + row] : 0.0f;
        if (pc.preprocessed == 0u) {
            float sumq = zinc_block_reduce_sum_all(sq * sq);
            float sumk = zinc_block_reduce_sum_all(skv * skv);
            sq *= rsqrtf(fmaxf(sumq, 1e-12f)) / sqrtf((float)pc.d_state);
            skv *= rsqrtf(fmaxf(sumk, 1e-12f));
        }

        float g, b;
        if (pc.preprocessed != 0u) {
            const size_t ab = (size_t)t * pc.ab_stride_tok + h;
            g = alpha[ab];
            b = beta[ab];
        } else {
            if (row == 0) {
                float a = alpha[t * pc.ab_stride_tok + h] + dt_bias_val;
                float sp = logf(1.0f + expf(a));               // softplus
                float gate_val = (pc.has_ssm_a != 0u) ? (sp * ssm_a_val) : (-sp);
                s_g = expf(gate_val);
                s_b = 1.0f / (1.0f + expf(-beta[t * pc.ab_stride_tok + h]));
            }
            __syncthreads();
            g = s_g;
            b = s_b;
        }

        float v_val = conv_out[v_off + col];
        rs *= g;                                            // decay
        float sk = zinc_block_reduce_sum_all((row < k_len) ? rs * skv : 0.0f);
        float d = b * (v_val - sk);
        if (row < k_len) rs += skv * d;                     // rank-1 update
        float o = zinc_block_reduce_sum_all((row < k_len) ? rs * sq : 0.0f);  // readout
        if (row == 0) out_data[t * pc.y_stride_tok + h * hv + col] = o;
        __syncthreads();
    }
    state[col_base + row] = rs;                             // write final state
}

// ---- ssm_delta_net_warp (warp-level scan — no __syncthreads in hot loop) -----
// Gated delta-net approach: each WARP owns one column,
// state elements spread across lanes (rows_per_lane = head_v_dim/warp_size).
// ALL reductions use __shfl_down_sync (warp-level) — ZERO __syncthreads per
// token iteration. Eliminates the 2048 barriers that made the old kernel 60%
// of prefill time.
//
// KNOWN BUG (discovered 2026-06-30): This kernel computes S^T @ k (column-wise
// dot product) while the block kernel (ssm_delta_net) computes S @ k (row-wise
// dot product). These produce DIFFERENT states for n_tok > 1:
//
//   Block kernel: sk[row] = sum_col S[row][col] * k[col] → d[row], S += d ⊗ k
//   Warp kernel:  sk[col] = sum_row S[row][col] * k[row] → d[col], S += k ⊗ d
//
// For T=1 with S_0=0 they're identical (sk=0 → d=beta*v → same rank-1 update).
// For T>1 they diverge, compounding across layers. The 35B MoE model amplifies
// this divergence into garbage output.
//
// FIX: Block kernel is used for all T>1 prefill (see forward_cuda.zig).
// This warp kernel is ONLY safe for T=1 decode.
//
// Grid: (dt_rank, ceilDiv(head_v_dim, N_WARPS))
// Block: (32, N_WARPS) = 128 threads
// Each warp (threadIdx.y) owns column = blockIdx.y * N_WARPS + threadIdx.y.
struct DeltaNetWarpPush {
    unsigned dt_rank, head_v_dim, d_state, n_group;
    unsigned ssm_a_is_f16, dt_bias_is_f16, has_dt_bias, has_ssm_a;
    unsigned n_tok, conv_stride_tok, ab_stride_tok, y_stride_tok;
};

#define DN_WARP_SIZE 32
#define DN_N_WARPS 4
#define DN_MAX_ROWS_PER_LANE 4  // head_v_dim(128) / warp_size(32) = 4

// Normalize each Q/K group and activate each head's gate/beta exactly once per
// token. The recurrent scan previously repeated this invariant work for every
// state row (128 times per head on Qwen 3.8 27B). Q/K are rewritten in place;
// alpha and beta become the activated scalar values consumed by the scan.
struct DeltaNetPreparePush {
    unsigned dt_rank, d_state, n_group;
    unsigned ssm_a_is_f16, dt_bias_is_f16, has_dt_bias, has_ssm_a;
    unsigned n_tok, conv_stride_tok, ab_stride_tok;
};

extern "C" __global__ __launch_bounds__(128, 8) void ssm_delta_net_prepare(
    float* conv_out, const unsigned char* dt_bias, float* alpha, float* beta,
    const unsigned char* ssm_a, DeltaNetPreparePush pc)
{
    const unsigned group = blockIdx.x;
    const unsigned t = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (t >= pc.n_tok) return;

    if (group < pc.n_group) {
        const unsigned qk_dim = pc.d_state * pc.n_group;
        const unsigned conv_base = t * pc.conv_stride_tok;
        const unsigned q_off = conv_base + group * pc.d_state;
        const unsigned k_off = conv_base + qk_dim + group * pc.d_state;
        const float q_val = (lane < pc.d_state) ? conv_out[q_off + lane] : 0.0f;
        const float k_val = (lane < pc.d_state) ? conv_out[k_off + lane] : 0.0f;
        const float sumq = zinc_block_reduce_sum_all(q_val * q_val);
        const float sumk = zinc_block_reduce_sum_all(k_val * k_val);
        const float q_rinv = rsqrtf(fmaxf(sumq, 1e-12f)) / sqrtf((float)pc.d_state);
        const float k_rinv = rsqrtf(fmaxf(sumk, 1e-12f));
        if (lane < pc.d_state) {
            conv_out[q_off + lane] = q_val * q_rinv;
            conv_out[k_off + lane] = k_val * k_rinv;
        }
    }

    if (lane == 0) {
        for (unsigned h = group; h < pc.dt_rank; h += pc.n_group) {
            float dt_bias_val = 0.0f;
            if (pc.has_dt_bias != 0u)
                dt_bias_val = pc.dt_bias_is_f16 ? zinc_half_to_float(((const unsigned short*)dt_bias)[h])
                                                : ((const float*)dt_bias)[h];
            float ssm_a_val = 0.0f;
            if (pc.has_ssm_a != 0u)
                ssm_a_val = pc.ssm_a_is_f16 ? zinc_half_to_float(((const unsigned short*)ssm_a)[h])
                                            : ((const float*)ssm_a)[h];
            const size_t ab = (size_t)t * pc.ab_stride_tok + h;
            const float sp = logf(1.0f + expf(alpha[ab] + dt_bias_val));
            alpha[ab] = (pc.has_ssm_a != 0u) ? expf(sp * ssm_a_val) : expf(-sp);
            beta[ab] = 1.0f / (1.0f + expf(-beta[ab]));
        }
    }
}

// State-compatible wave32 scan. Each warp owns one state column, as in the
// block kernel, but each lane carries rows lane+{0,32,64,96}. Four independent
// wave reductions reproduce the block kernel's four first-stage partials; the
// lane-0 combine uses the same (r0+r2)+(r1+r3) order as its second stage.
struct DeltaNetColWarpPush {
    unsigned dt_rank, head_v_dim, d_state, n_group;
    unsigned n_tok, conv_stride_tok, ab_stride_tok, y_stride_tok, fast_reduce;
};

__device__ __forceinline__ float zinc_warp_reduce_128_exact(float v0, float v1, float v2, float v3) {
    const float r0 = zinc_warp_reduce_sum(v0);
    const float r1 = zinc_warp_reduce_sum(v1);
    const float r2 = zinc_warp_reduce_sum(v2);
    const float r3 = zinc_warp_reduce_sum(v3);
    float total = 0.0f;
    if (threadIdx.x == 0) total = (r0 + r2) + (r1 + r3);
    return __shfl_sync(0xffffffffu, total, 0);
}

extern "C" __global__ __launch_bounds__(128, 8) void ssm_delta_net_col_warp(
    const float* conv_out, const float* gate, const float* beta,
    float* state, float* out_data, DeltaNetColWarpPush pc)
{
    const unsigned h = blockIdx.x;
    const unsigned col = blockIdx.y * 4u + threadIdx.y;
    const unsigned lane = threadIdx.x;
    if (h >= pc.dt_rank || col >= pc.head_v_dim) return;

    const unsigned hv = pc.head_v_dim;
    const unsigned qk_dim = pc.d_state * pc.n_group;
    const unsigned group = (pc.n_group == pc.dt_rank) ? h : (h % pc.n_group);
    float s0 = state[((size_t)h * hv + col) * hv + lane];
    float s1 = (lane + 32u < hv) ? state[((size_t)h * hv + col) * hv + lane + 32u] : 0.0f;
    float s2 = (lane + 64u < hv) ? state[((size_t)h * hv + col) * hv + lane + 64u] : 0.0f;
    float s3 = (lane + 96u < hv) ? state[((size_t)h * hv + col) * hv + lane + 96u] : 0.0f;

    for (unsigned t = 0; t < pc.n_tok; t++) {
        const unsigned conv_base = t * pc.conv_stride_tok;
        const unsigned q_off = conv_base + group * pc.d_state;
        const unsigned k_off = conv_base + qk_dim + group * pc.d_state;
        const unsigned v_off = conv_base + 2u * qk_dim + h * hv;
        const float q0 = conv_out[q_off + lane];
        const float k0 = conv_out[k_off + lane];
        const float q1 = (lane + 32u < hv) ? conv_out[q_off + lane + 32u] : 0.0f;
        const float k1 = (lane + 32u < hv) ? conv_out[k_off + lane + 32u] : 0.0f;
        const float q2 = (lane + 64u < hv) ? conv_out[q_off + lane + 64u] : 0.0f;
        const float k2 = (lane + 64u < hv) ? conv_out[k_off + lane + 64u] : 0.0f;
        const float q3 = (lane + 96u < hv) ? conv_out[q_off + lane + 96u] : 0.0f;
        const float k3 = (lane + 96u < hv) ? conv_out[k_off + lane + 96u] : 0.0f;
        const size_t ab = (size_t)t * pc.ab_stride_tok + h;
        const float g = gate[ab];
        const float b = beta[ab];

        s0 *= g;
        s1 *= g;
        s2 *= g;
        s3 *= g;
        const float sk = pc.fast_reduce
            ? zinc_warp_reduce_sum_all(((s0 * k0 + s1 * k1) + s2 * k2) + s3 * k3)
            : zinc_warp_reduce_128_exact(s0 * k0, s1 * k1, s2 * k2, s3 * k3);
        const float d = b * (conv_out[v_off + col] - sk);
        s0 += k0 * d;
        s1 += k1 * d;
        s2 += k2 * d;
        s3 += k3 * d;
        const float o = pc.fast_reduce
            ? zinc_warp_reduce_sum_all(((s0 * q0 + s1 * q1) + s2 * q2) + s3 * q3)
            : zinc_warp_reduce_128_exact(s0 * q0, s1 * q1, s2 * q2, s3 * q3);
        if (lane == 0) out_data[t * pc.y_stride_tok + h * hv + col] = o;
    }

    state[((size_t)h * hv + col) * hv + lane] = s0;
    if (lane + 32u < hv) state[((size_t)h * hv + col) * hv + lane + 32u] = s1;
    if (lane + 64u < hv) state[((size_t)h * hv + col) * hv + lane + 64u] = s2;
    if (lane + 96u < hv) state[((size_t)h * hv + col) * hv + lane + 96u] = s3;
}

extern "C" __global__ __launch_bounds__(128, 8) void ssm_delta_net_warp(
    const float* conv_out, const unsigned char* dt_bias, const float* alpha,
    const float* beta, const unsigned char* ssm_a, float* state, float* out_data,
    DeltaNetWarpPush pc)
{
    const unsigned h = blockIdx.x;
    const unsigned row = blockIdx.y * DN_N_WARPS + threadIdx.y;
    const unsigned lane = threadIdx.x;
    if (h >= pc.dt_rank || row >= pc.head_v_dim) return;

    const unsigned hv = pc.head_v_dim;
    const unsigned qk_dim = pc.d_state * pc.n_group;
    const unsigned k_len = (hv < pc.d_state) ? hv : pc.d_state;
    const unsigned k_hi = (pc.n_group == pc.dt_rank) ? h : (h % pc.n_group);
    const unsigned cols_per_lane = hv / DN_WARP_SIZE;  // 128/32 = 4

    // State[h][row][col] — each lane owns cols lane, lane+32, lane+64, lane+96
    // FIXED: warp owns ROW (not column) → computes S@k (row-wise) like block kernel
    float s_shard[DN_MAX_ROWS_PER_LANE];
    #pragma unroll
    for (int r = 0; r < cols_per_lane; r++) {
        unsigned col_idx = r * DN_WARP_SIZE + lane;
        s_shard[r] = state[((size_t)h * hv + row) * hv + col_idx];
    }

    // Precompute dt_bias and ssm_a (same for all tokens — per-head constants)
    float dt_bias_val = 0.0f;
    if (pc.has_dt_bias != 0u)
        dt_bias_val = pc.dt_bias_is_f16 ? zinc_half_to_float(((const unsigned short*)dt_bias)[h])
                                        : ((const float*)dt_bias)[h];
    float ssm_a_val = 0.0f;
    if (pc.has_ssm_a != 0u)
        ssm_a_val = pc.ssm_a_is_f16 ? zinc_half_to_float(((const unsigned short*)ssm_a)[h])
                                    : ((const float*)ssm_a)[h];

    const float inv_sqrt_d_state = rsqrtf((float)pc.d_state);

    // Precompute gate[t] and beta[t] for ALL tokens (parallelizable, no state dep).
    // This removes expf/logf from the sequential scan loop, letting the warp
    // scheduler overlap them with the state-dependent compute.
    // Each thread precomputes T/n_threads values, stored in shared memory.
    extern __shared__ float smem_gate_beta[];  // [2 * n_tok]
    float* sh_gate = smem_gate_beta;
    float* sh_beta = smem_gate_beta + pc.n_tok;
    // Use FULL thread index (threadIdx.y * 32 + threadIdx.x = 0..127), not just
    // lane (threadIdx.x = 0..31). Otherwise warps 1-3 duplicate warp 0's work
    // and tokens 32+ are never initialized → garbage gate/beta.
    const unsigned full_tid = threadIdx.y * DN_WARP_SIZE + threadIdx.x;
    for (unsigned t = full_tid; t < pc.n_tok; t += DN_WARP_SIZE * DN_N_WARPS) {
        float a = alpha[t * pc.ab_stride_tok + h] + dt_bias_val;
        float sp = logf(1.0f + expf(a));
        sh_gate[t] = (pc.has_ssm_a != 0u) ? expf(sp * ssm_a_val) : expf(-sp);
        sh_beta[t] = 1.0f / (1.0f + expf(-beta[t * pc.ab_stride_tok + h]));
    }
    __syncthreads();  // one-time sync before the scan loop starts

    for (unsigned t = 0; t < pc.n_tok; t++) {
        const unsigned conv_base = t * pc.conv_stride_tok;
        const unsigned q_off = conv_base + k_hi * pc.d_state;
        const unsigned k_off = conv_base + qk_dim + k_hi * pc.d_state;
        const unsigned v_off = conv_base + 2u * qk_dim + h * hv;

        // Load Q, K for this token — indexed by COLUMN (lane's cols)
        float q_reg[DN_MAX_ROWS_PER_LANE], k_reg[DN_MAX_ROWS_PER_LANE];
        float sumq = 0.0f, sumk = 0.0f;
        #pragma unroll
        for (int r = 0; r < cols_per_lane; r++) {
            unsigned col_idx = r * DN_WARP_SIZE + lane;
            if (col_idx < k_len) {
                q_reg[r] = conv_out[q_off + col_idx];
                k_reg[r] = conv_out[k_off + col_idx];
                sumq += q_reg[r] * q_reg[r];
                sumk += k_reg[r] * k_reg[r];
            } else {
                q_reg[r] = 0.0f;
                k_reg[r] = 0.0f;
            }
        }

        sumq = zinc_warp_reduce_sum_all(sumq);
        sumk = zinc_warp_reduce_sum_all(sumk);
        float q_rinv = rsqrtf(fmaxf(sumq, 1e-12f)) * inv_sqrt_d_state;
        float k_rinv = rsqrtf(fmaxf(sumk, 1e-12f));
        #pragma unroll
        for (int r = 0; r < cols_per_lane; r++) {
            q_reg[r] *= q_rinv;
            k_reg[r] *= k_rinv;
        }

        // Gate and beta: read from shared memory (precomputed above)
        float gate = sh_gate[t];
        float b = sh_beta[t];

        // Load V for this row (indexed by ROW, not column)
        float v_val = conv_out[v_off + row];

        // State update: sk = (S @ k_norm)[row] — warp reduce across COLUMNS
        float sk_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < cols_per_lane; r++) {
            s_shard[r] *= gate;
            sk_partial += s_shard[r] * k_reg[r];
        }
        float sk = zinc_warp_reduce_sum_all(sk_partial);
        float d = b * (v_val - sk);

        // Readout: o = (S @ q_norm)[row] — warp reduce across COLUMNS
        float o_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < cols_per_lane; r++) {
            s_shard[r] += k_reg[r] * d;  // rank-1 update
            o_partial += s_shard[r] * q_reg[r];
        }
        float o = zinc_warp_reduce_sum_all(o_partial);

        // Write output (one value per warp — lane 0 writes)
        if (lane == 0)
            out_data[t * pc.y_stride_tok + h * hv + row] = o;
    }

    // Write final state
    #pragma unroll
    for (int r = 0; r < cols_per_lane; r++) {
        unsigned col_idx = r * DN_WARP_SIZE + lane;
        state[((size_t)h * hv + row) * hv + col_idx] = s_shard[r];
    }
}

// ---- ssm_conv1d_seq (Effort 28 4c-2b: batched DECODE per-seq conv1d) ---------
// Batched twin of ssm_conv1d for DECODE: B sequences in ONE launch, each row b
// reading/writing its OWN slot conv ring at per-row state_offset = positions[b] %
// (d_conv-1). grid = (ceilDiv(conv_channels,64), B). Input/output token-major
// [B, conv_channels]; `state` the slot conv buffer [n_slots*conv_state_len], row
// b at base slots[b]*conv_state_len. Per-channel math copied verbatim from
// ssm_conv1d (state_offset derived from positions[b] instead of a push field).
struct ConvSeqPush { unsigned conv_channels, d_conv, kernel_is_f16, conv_state_len; };
extern "C" __global__ void ssm_conv1d_seq(const float* current_input, const unsigned char* conv_kernel,
                                          float* state, float* out_data,
                                          const unsigned* positions, const unsigned* slots, ConvSeqPush pc) {
    unsigned ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= pc.conv_channels) return;
    unsigned b = blockIdx.y;
    unsigned d_conv_1 = pc.d_conv - 1u;
    unsigned state_offset = positions[b] % d_conv_1;
    const float* in_row = current_input + (size_t)b * pc.conv_channels;
    float* out_row = out_data + (size_t)b * pc.conv_channels;
    float* st = state + (size_t)slots[b] * pc.conv_state_len;
    float ci = in_row[ch];
    float sum = 0.0f;
    for (unsigned ki = 0; ki < pc.d_conv; ki++) {
        unsigned k_idx = ch * pc.d_conv + ki;
        float kw = (pc.kernel_is_f16 != 0u)
                       ? zinc_half_to_float(((const unsigned short*)conv_kernel)[k_idx])
                       : ((const float*)conv_kernel)[k_idx];
        float sv;
        if (ki < d_conv_1) {
            unsigned slot = state_offset + ki;
            if (slot >= d_conv_1) slot -= d_conv_1;
            sv = st[(size_t)slot * pc.conv_channels + ch];
        } else {
            sv = ci;
        }
        sum += kw * sv;
    }
    out_row[ch] = sum / (1.0f + expf(-sum));            // SiLU
    st[(size_t)state_offset * pc.conv_channels + ch] = ci;
}

// ---- ssm_gated_norm_seq (Effort 28 4c-2b: batched DECODE gated norm) ---------
// Batched twin of ssm_gated_norm: grid = (dt_rank, B); block per (head h, row b).
// o/z/out token-major [B, d_inner], row b at b*d_inner; norm_weight is the shared
// layer weight (per-head index uses head_base only, NOT the row offset). Per-head
// math copied verbatim from ssm_gated_norm.
struct GatedNormSeqPush { unsigned d_inner, dt_rank, head_v_dim, d_state, norm_per_head; };
extern "C" __global__ void ssm_gated_norm_seq(const float* o, const float* z, const float* norm_weight,
                                              float* out, GatedNormSeqPush pc) {
    unsigned h = blockIdx.x;
    unsigned b = blockIdx.y;
    unsigned head_base = h * pc.head_v_dim;
    unsigned base = b * pc.d_inner + head_base;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.head_v_dim; i += blockDim.x) { float v = o[base + i]; ss += v * v; }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.head_v_dim + 1e-6f);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < pc.head_v_dim; i += blockDim.x) {
        float nv = o[base + i] * rinv;
        unsigned norm_idx = (pc.norm_per_head != 0u) ? (head_base + i) : (i % pc.d_state);
        nv *= norm_weight[norm_idx];
        float zv = z[base + i];
        out[base + i] = nv * (zv / (1.0f + expf(-zv)));
    }
}

// ---- ssm_delta_net_seq (Effort 28 4c-2b: batched DECODE delta-net scan) ------
// Batched twin of ssm_delta_net for single-token DECODE: grid = (dt_rank,
// head_v_dim, B); block per (head h, row, b). Each row b reads/writes its OWN
// slot recurrent state (base slots[b]*ssm_state_len) and its OWN token-major
// slices of conv_out/alpha/beta/out_data (b*{conv,ab,y}_stride_tok). One token
// per row (no n_tok loop). Per-(h,row,b) math copied verbatim from ssm_delta_net.
struct DeltaNetSeqPush {
    unsigned d_inner, dt_rank, head_v_dim, d_state, n_group;
    unsigned ssm_a_is_f16, dt_bias_is_f16, has_dt_bias, has_ssm_a;
    unsigned conv_stride_tok, ab_stride_tok, y_stride_tok, ssm_state_len;
};
extern "C" __global__ void ssm_delta_net_seq(
    const float* conv_out, const unsigned char* dt_bias, const float* alpha,
    const float* beta, const unsigned char* ssm_a, float* state, float* out_data,
    const unsigned* slots, DeltaNetSeqPush pc) {
    // Transposed state: state[slot][h][col][row] — col-major for coalesced
    // warp access in prefill kernels. blockIdx.y = col, threadIdx.x = row.
    unsigned h = blockIdx.x;
    unsigned col = blockIdx.y;
    unsigned b = blockIdx.z;
    unsigned row = threadIdx.x;                  // 0..head_v_dim-1
    if (h >= pc.dt_rank || col >= pc.head_v_dim) return;
    unsigned hv = pc.head_v_dim;
    unsigned qk_dim = pc.d_state * pc.n_group;
    unsigned k_len = (hv < pc.d_state) ? hv : pc.d_state;
    size_t state_base = (size_t)slots[b] * pc.ssm_state_len;
    size_t col_base = state_base + ((size_t)h * hv + col) * hv;
    float rs = state[col_base + row];            // state[slot][h][col][row]

    float dt_bias_val = 0.0f;
    if (pc.has_dt_bias != 0u)
        dt_bias_val = pc.dt_bias_is_f16 ? zinc_half_to_float(((const unsigned short*)dt_bias)[h])
                                        : ((const float*)dt_bias)[h];
    float ssm_a_val = 0.0f;
    if (pc.has_ssm_a != 0u)
        ssm_a_val = pc.ssm_a_is_f16 ? zinc_half_to_float(((const unsigned short*)ssm_a)[h])
                                    : ((const float*)ssm_a)[h];
    unsigned k_hi = (pc.n_group == pc.dt_rank) ? h : (h % pc.n_group);

    __shared__ float s_g, s_b;

    // Single decode token for row b: bases are b * per-token stride.
    unsigned conv_base = b * pc.conv_stride_tok;
    unsigned q_off = conv_base + k_hi * pc.d_state;
    unsigned k_off = conv_base + qk_dim + k_hi * pc.d_state;
    unsigned v_off = conv_base + 2u * qk_dim + h * hv;

    // L2-normalize Q/K per group (sum-sq reduced across rows), scale Q.
    float qi = (row < k_len) ? conv_out[q_off + row] : 0.0f;
    float ki = (row < k_len) ? conv_out[k_off + row] : 0.0f;
    float sumq = zinc_block_reduce_sum_all(qi * qi);
    float sumk = zinc_block_reduce_sum_all(ki * ki);
    float sq = qi * (rsqrtf(fmaxf(sumq, 1e-12f)) / sqrtf((float)pc.d_state));
    float skv = ki * rsqrtf(fmaxf(sumk, 1e-12f));

    if (row == 0) {
        float a = alpha[b * pc.ab_stride_tok + h] + dt_bias_val;
        float sp = logf(1.0f + expf(a));               // softplus
        float gate_val = (pc.has_ssm_a != 0u) ? (sp * ssm_a_val) : (-sp);
        s_g = expf(gate_val);
        s_b = 1.0f / (1.0f + expf(-beta[b * pc.ab_stride_tok + h]));
    }
    __syncthreads();
    float g = s_g, bcoef = s_b;

    float v_val = conv_out[v_off + col];
    rs *= g;                                            // decay
    float sk = zinc_block_reduce_sum_all((row < k_len) ? rs * skv : 0.0f);
    float delta = bcoef * (v_val - sk);
    if (row < k_len) rs += skv * delta;                 // rank-1 update
    float o = zinc_block_reduce_sum_all((row < k_len) ? rs * sq : 0.0f);  // readout
    if (row == 0) out_data[b * pc.y_stride_tok + h * hv + col] = o;
    state[col_base + row] = rs;                          // write final state
}

// ---- ssm_delta_net_chunked (parallel intra-chunk delta-net) ------------------
// v0: correct but suboptimal. Caches normalized K in shared memory,
// builds Gram matrix, forward substitution, output, state update.
// Same thread org as ssm_delta_net_warp: warp=column, lane=rows.

struct DeltaNetChunkedPush {
    unsigned dt_rank, head_v_dim, d_state, n_group;
    unsigned ssm_a_is_f16, dt_bias_is_f16, has_dt_bias, has_ssm_a;
    unsigned n_tok, conv_stride_tok, ab_stride_tok, y_stride_tok;
};

extern "C" __global__ __launch_bounds__(128, 4) void ssm_delta_net_chunked(
    const float* conv_out, const unsigned char* dt_bias,
    const float* alpha, const float* beta,
    const unsigned char* ssm_a, float* state, float* out_data,
    DeltaNetChunkedPush pc)
{
    const unsigned h = blockIdx.x;
    const unsigned col = blockIdx.y * 4 + threadIdx.y;
    const unsigned lane = threadIdx.x;
    if (h >= pc.dt_rank || col >= pc.head_v_dim) return;

    const unsigned hv = pc.head_v_dim;
    const unsigned T = pc.n_tok;
    const unsigned qk_dim = pc.d_state * pc.n_group;
    const unsigned k_len = (hv < pc.d_state) ? hv : pc.d_state;
    const unsigned k_hi = (pc.n_group == pc.dt_rank) ? h : (h % pc.n_group);
    const unsigned CS = 64;

    float dt_bias_val = pc.has_dt_bias ?
        (pc.dt_bias_is_f16 ? zinc_half_to_float(((const unsigned short*)dt_bias)[h])
                           : ((const float*)dt_bias)[h]) : 0.0f;
    float ssm_a_val = pc.has_ssm_a ?
        (pc.ssm_a_is_f16 ? zinc_half_to_float(((const unsigned short*)ssm_a)[h])
                         : ((const float*)ssm_a)[h]) : 0.0f;

    // Shared memory: k_norm[CS*hv](32KB) + M[CS*CS](16KB) + g_cs[CS] + beta[CS]
    extern __shared__ float smem[];
    float* k_norm  = smem;                       // [CS * hv]
    float* M_mat   = k_norm + CS * hv;           // [CS * CS]
    float* g_cs_sh = M_mat + CS * CS;            // [CS]
    float* beta_sh = g_cs_sh + CS;               // [CS]

    // State: 4 rows per lane (registers)
    float s_shard[4];
    #pragma unroll
    for (int r = 0; r < 4; r++)
        s_shard[r] = state[((size_t)h * hv + (r * 32 + lane)) * hv + col];

    const unsigned n_chunks = (T + CS - 1) / CS;

    for (unsigned chunk = 0; chunk < n_chunks; chunk++) {
        const unsigned t0 = chunk * CS;
        const unsigned cs = (t0 + CS <= T) ? CS : (T - t0);

        // --- Gate cumsum + beta (warp 0, lane 0) ---
        if (threadIdx.y == 0 && lane == 0) {
            float gs = 0.0f;
            for (unsigned t = 0; t < cs; t++) {
                float a = alpha[(t0+t)*pc.ab_stride_tok + h] + dt_bias_val;
                float sp = logf(1.0f + expf(a));
                gs += (pc.has_ssm_a ? sp * ssm_a_val : -sp);
                g_cs_sh[t] = gs;
                beta_sh[t] = 1.0f / (1.0f + expf(-beta[(t0+t)*pc.ab_stride_tok + h]));
            }
        }
        __syncthreads();

        // --- Load + L2-normalize K vectors into shared memory ---
        // 4 warps load 4 K vectors at a time (each warp = 1 vector, 32 lanes × 4 elements)
        for (unsigned t_base = 0; t_base < cs; t_base += 4) {
            unsigned t = t_base + threadIdx.y;
            if (t < cs) {
                float kr[4]; float ss = 0.0f;
                unsigned base = (t0+t)*pc.conv_stride_tok + qk_dim + k_hi*pc.d_state;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    unsigned d = r * 32 + lane;
                    kr[r] = (d < k_len) ? conv_out[base + d] : 0.0f;
                    ss += kr[r] * kr[r];
                }
                ss = zinc_warp_reduce_sum_all(ss);
                float rinv = rsqrtf(fmaxf(ss, 1e-12f));
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    unsigned d = r * 32 + lane;
                    k_norm[t * hv + d] = kr[r] * rinv;
                }
            }
        }
        __syncthreads();

        // --- Build Gram matrix M[i][j] = beta[i]*exp(g_cs[i]-g_cs[j])*<ki,kj> ---
        // Each warp computes rows of M. Dot products from shared k_norm.
        for (unsigned i_base = 0; i_base < cs; i_base += 4) {
            unsigned i = i_base + threadIdx.y;
            if (i < cs && i > 0) {
                float ki[4];
                #pragma unroll
                for (int r = 0; r < 4; r++)
                    ki[r] = k_norm[i * hv + r * 32 + lane];
                float gcs_i = g_cs_sh[i];  // g_cs[t] — includes gate[t]
                for (unsigned j = 0; j < i; j++) {
                    float dp = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < 4; r++)
                        dp += ki[r] * k_norm[j * hv + r * 32 + lane];
                    dp = zinc_warp_reduce_sum_all(dp);
                    if (lane == 0)
                        M_mat[i * CS + j] = beta_sh[i] * expf(gcs_i - g_cs_sh[j]) * dp;
                }
            }
        }
        __syncthreads();

        // --- Forward substitution: d[t] = a[t] - sum_{s<t} M[t][s]*d[s] ---
        float d_vals[64];
        for (unsigned t = 0; t < cs; t++) {
            // sk = <state, k_t> (warp reduce)
            float skp = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; r++)
                skp += s_shard[r] * k_norm[t * hv + r * 32 + lane];
            float sk = zinc_warp_reduce_sum_all(skp);

            // G(t,0) = exp(g_cs[t]) — includes gate[t], matching original kernel
            // where rs *= gate happens BEFORE sk = dot(rs, k)
            float g_state = expf(g_cs_sh[t]);
            unsigned v_off = (t0+t)*pc.conv_stride_tok + 2u*qk_dim + h*hv + col;
            float a_t = beta_sh[t] * (conv_out[v_off] - g_state * sk);

            float d_t = a_t;
            for (unsigned s = 0; s < t; s++)
                d_t -= M_mat[t * CS + s] * d_vals[s];
            d_vals[t] = d_t;  // broadcast: same across lanes in warp
        }

        // --- Output: o[t] = G(t,0)*sq + sum_{s<=t} G(s+1,t)*<ks,qt>*d[s] ---
        // Note: S^t = P(0,t)*S_init + sum_{s=0}^{t} P(s+1,t)*k[s]@d[s]^T
        // o[t] = P(0,t)*<S_init,qt> + sum_{s=0}^{t} P(s+1,t)*<ks,qt>*d[s]
        // P(0,t) = exp(g_cs[t]), P(s+1,t) = exp(g_cs[t]-g_cs[s])
        float inv_sq = rsqrtf((float)pc.d_state);
        for (unsigned t = 0; t < cs; t++) {
            // Load + normalize Q
            float qr[4]; float qss = 0.0f;
            unsigned qbase = (t0+t)*pc.conv_stride_tok + k_hi*pc.d_state;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                unsigned d = r * 32 + lane;
                qr[r] = (d < k_len) ? conv_out[qbase + d] : 0.0f;
                qss += qr[r] * qr[r];
            }
            qss = zinc_warp_reduce_sum_all(qss);
            float qrinv = rsqrtf(fmaxf(qss, 1e-12f)) * inv_sq;
            #pragma unroll
            for (int r = 0; r < 4; r++) qr[r] *= qrinv;

            // sq = <state, q_norm>
            float sqp = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; r++) sqp += s_shard[r] * qr[r];
            float sq = zinc_warp_reduce_sum_all(sqp);

            float g_st = expf(g_cs_sh[t]);
            float o = g_st * sq;

            for (unsigned s = 0; s <= t; s++) {
                float kqp = 0.0f;
                #pragma unroll
                for (int r = 0; r < 4; r++)
                    kqp += k_norm[s * hv + r * 32 + lane] * qr[r];
                float kq = zinc_warp_reduce_sum_all(kqp);
                o += expf(g_cs_sh[t] - g_cs_sh[s]) * kq * d_vals[s];
            }
            if (lane == 0)
                out_data[(t0+t)*pc.y_stride_tok + h*hv + col] = o;
        }

        // --- State update: S = G_total*S + sum_s G_last*k_s*d[s] ---
        {
            float g_total = expf(g_cs_sh[cs - 1]);
            // Apply total gate decay first
            #pragma unroll
            for (int r = 0; r < 4; r++) s_shard[r] *= g_total;
            // Accumulate rank-1 updates
            for (unsigned s = 0; s < cs; s++) {
                float g_last = expf(g_cs_sh[cs - 1] - g_cs_sh[s]);
                float kd = g_last * d_vals[s];
                #pragma unroll
                for (int r = 0; r < 4; r++)
                    s_shard[r] += k_norm[s * hv + r * 32 + lane] * kd;
            }
        }
        __syncthreads();
    }

    // Write final state
    #pragma unroll
    for (int r = 0; r < 4; r++)
        state[((size_t)h * hv + (r * 32 + lane)) * hv + col] = s_shard[r];
}

// ---- dmmv_q4k_fast (perf research, 5090) — port of tuned Vulkan dmmv_q4k -----
// 16 threads per Q4_K superblock: header read once/thread (not 256x), qs read
// once total, x via float4. Block-reduce over the block = one output row.
// The per-row compute is factored into a __device__ helper so the fused dual
// kernel (dmmv_q4k_fast_dual) shares exactly this arithmetic path (bit-exact).
__device__ __forceinline__ float zinc_dmmv_q4k_fast_sum(const unsigned* a_u32, unsigned a_base, const float4* xv, unsigned bpr) {
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 36u;
        unsigned dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
        float dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
        unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
        float f0 = d * (float)(s0 & 0x3Fu), b0 = dm * (float)(s1 & 0x3Fu);
        float f1 = d * (float)((s0 >> 8) & 0x3Fu), b1 = dm * (float)((s1 >> 8) & 0x3Fu);
        float f2 = d * (float)((s2 & 0xFu) | ((s0 & 0xC0u) >> 2)), b2 = dm * (float)(((s2 & 0xF0u) >> 4) | ((s1 & 0xC0u) >> 2));
        float f3 = d * (float)(((s2 >> 8) & 0xFu) | (((s0 >> 8) & 0xC0u) >> 2)), b3 = dm * (float)((((s2 >> 8) & 0xF0u) >> 4) | (((s1 >> 8) & 0xC0u) >> 2));
        sum += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
        sum += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
        sum += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
        sum += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
    }
    return zinc_block_reduce_sum(sum);
}

extern "C" __global__ void dmmv_q4k_fast(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    unsigned tok = blockIdx.y;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = (pc.a_offset >> 2) + row * bpr * 36u;
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2) + (size_t)tok * pc.K);
    float sum = zinc_dmmv_q4k_fast_sum(a_u32, a_base, xv, bpr);
    if (threadIdx.x == 0) { unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row; if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum; }
}

// ---- dmmv_q4k_gate_up_swiglu — fused dense-decode FFN input ---------------
// Gate and up use the same normalized activation and have identical Q4_K
// shapes. One block computes both rows, sharing the four activation vectors,
// then applies SwiGLU in thread 0. This removes two dependent launches and the
// gate/up intermediate writes without changing either matvec reduction order.
__device__ __forceinline__ float zinc_q4k_fast_block_dot(
    const unsigned* a, unsigned blk, unsigned q_off, unsigned shift,
    float4 by0, float4 by1, float4 by2, float4 by3)
{
    const unsigned dd = a[blk];
    const float d = zinc_half_to_float((unsigned short)(dd & 0xffffu));
    const float dm = zinc_half_to_float((unsigned short)(dd >> 16));
    const unsigned sc0 = a[blk + 1u], sc1 = a[blk + 2u], sc2 = a[blk + 3u];
    const unsigned qs0 = a[blk + 4u + (q_off >> 2)];
    const unsigned qs1 = a[blk + 4u + (q_off >> 2) + 16u];
    const unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
    const float f0 = d * (float)(s0 & 0x3fu), b0 = dm * (float)(s1 & 0x3fu);
    const float f1 = d * (float)((s0 >> 8) & 0x3fu), b1 = dm * (float)((s1 >> 8) & 0x3fu);
    const float f2 = d * (float)((s2 & 0xfu) | ((s0 & 0xc0u) >> 2));
    const float b2 = dm * (float)(((s2 & 0xf0u) >> 4) | ((s1 & 0xc0u) >> 2));
    const float f3 = d * (float)(((s2 >> 8) & 0xfu) | (((s0 >> 8) & 0xc0u) >> 2));
    const float b3 = dm * (float)((((s2 >> 8) & 0xf0u) >> 4) | (((s1 >> 8) & 0xc0u) >> 2));
    float sum = 0.0f;
    sum += (f0*(float)(qs0&0xfu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xfu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xfu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xfu)-b0)*by0.w;
    sum += (f1*(float)((qs0>>4)&0xfu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xfu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xfu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xfu)-b1)*by1.w;
    sum += (f2*(float)(qs1&0xfu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xfu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xfu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xfu)-b2)*by2.w;
    sum += (f3*(float)((qs1>>4)&0xfu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xfu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xfu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xfu)-b3)*by3.w;
    return sum;
}

extern "C" __global__ void dmmv_q4k_gate_up_swiglu(
    const unsigned* __restrict__ gate,
    const unsigned* __restrict__ up,
    const float* __restrict__ x,
    float* __restrict__ y, DmmvPush pc)
{
    const unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    const unsigned il = itid >> 2, ir = itid & 3u;
    const unsigned v_im = il >> 1, v_in = il & 1u;
    const unsigned l0 = 4u * (2u * ir + v_in);
    const unsigned q_off = 32u * v_im + l0;
    const unsigned y_loc = 64u * v_im + l0;
    const unsigned shift = v_im * 16u;
    const unsigned ngrp = blockDim.x >> 4;
    const unsigned bpr = pc.K >> 8;
    const unsigned row_base = row * bpr * 36u;
    const float4* xv = (const float4*)x;
    float sum_gate = 0.0f, sum_up = 0.0f;

    for (unsigned sb = grp; sb < bpr; sb += ngrp) {
        const unsigned bidx = (sb * 256u + y_loc) >> 2;
        const unsigned bidx2 = (sb * 256u + y_loc + 128u) >> 2;
        const float4 by0 = xv[bidx], by1 = xv[bidx + 8u];
        const float4 by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        const unsigned blk = row_base + sb * 36u;
        sum_gate += zinc_q4k_fast_block_dot(gate, blk, q_off, shift, by0, by1, by2, by3);
        sum_up += zinc_q4k_fast_block_dot(up, blk, q_off, shift, by0, by1, by2, by3);
    }

    if (pc.acc_mode != 0u) {
        zinc_block_reduce_sum_pair(sum_gate, sum_up);
    } else {
        sum_gate = zinc_block_reduce_sum(sum_gate);
        __syncthreads();
        sum_up = zinc_block_reduce_sum(sum_up);
    }
    if (tid == 0u) {
        const float silu = sum_gate / (1.0f + expf(-sum_gate));
        y[row] = silu * sum_up;
    }
}

// Q8_1-activation twin of the fused dense FFN input. The 16-thread Q4_K
// superblock mapping is identical to dmmv_q4k_gate_up_swiglu, but four packed
// int8 dots replace each group of 16 float FMAs. The quantizer's exact f16 sum
// supplies the asymmetric Q4_K minimum correction once per 32-value group.
#ifdef ZINC_ROCM
struct ExpertsQ8Push {
    unsigned M, K, T, slice, up_base, n_used, routing_stride, dst_tok_stride, fuse_activation;
};

__device__ __forceinline__ float zinc_q4k_q8_block_dot_meta(
    const unsigned* a, unsigned blk, unsigned q_off, unsigned shift,
    unsigned dd, unsigned sc0, unsigned sc1, unsigned sc2,
    int q0, int q1, int q2, int q3,
    float2 ds0, float2 ds1, float2 ds2, float2 ds3, bool min_lane)
{
    const float d = zinc_half_to_float((unsigned short)(dd & 0xffffu));
    const float dm = zinc_half_to_float((unsigned short)(dd >> 16));
    const unsigned qs0 = a[blk + 4u + (q_off >> 2)];
    const unsigned qs1 = a[blk + 4u + (q_off >> 2) + 16u];
    const unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
    const float f0 = d * (float)(s0 & 0x3fu), b0 = dm * (float)(s1 & 0x3fu);
    const float f1 = d * (float)((s0 >> 8) & 0x3fu), b1 = dm * (float)((s1 >> 8) & 0x3fu);
    const float f2 = d * (float)((s2 & 0xfu) | ((s0 & 0xc0u) >> 2));
    const float b2 = dm * (float)(((s2 & 0xf0u) >> 4) | ((s1 & 0xc0u) >> 2));
    const float f3 = d * (float)(((s2 >> 8) & 0xfu) | (((s0 >> 8) & 0xc0u) >> 2));
    const float b3 = dm * (float)((((s2 >> 8) & 0xf0u) >> 4) | (((s1 >> 8) & 0xc0u) >> 2));
    float sum = f0 * ds0.x * (float)__dp4a((int)(qs0 & 0x0f0f0f0fu), q0, 0);
    sum += f1 * ds1.x * (float)__dp4a((int)((qs0 >> 4) & 0x0f0f0f0fu), q1, 0);
    sum += f2 * ds2.x * (float)__dp4a((int)(qs1 & 0x0f0f0f0fu), q2, 0);
    sum += f3 * ds3.x * (float)__dp4a((int)((qs1 >> 4) & 0x0f0f0f0fu), q3, 0);
    if (min_lane) sum -= b0 * ds0.y + b1 * ds1.y + b2 * ds2.y + b3 * ds3.y;
    return sum;
}

__device__ __forceinline__ float zinc_q4k_q8_block_dot(
    const unsigned* a, unsigned blk, unsigned q_off, unsigned shift,
    int q0, int q1, int q2, int q3,
    float2 ds0, float2 ds1, float2 ds2, float2 ds3, bool min_lane)
{
    return zinc_q4k_q8_block_dot_meta(a, blk, q_off, shift,
        a[blk], a[blk + 1u], a[blk + 2u], a[blk + 3u],
        q0, q1, q2, q3, ds0, ds1, ds2, ds3, min_lane);
}

__device__ __forceinline__ void zinc_q4k_q8_scale_factors(
    uint4 meta, unsigned shift,
    float& f0, float& f1, float& f2, float& f3,
    float& b0, float& b1, float& b2, float& b3)
{
    const float d = zinc_half_to_float((unsigned short)(meta.x & 0xffffu));
    const float dm = zinc_half_to_float((unsigned short)(meta.x >> 16));
    const unsigned s0 = meta.y >> shift;
    const unsigned s1 = meta.z >> shift;
    const unsigned s2 = meta.w >> shift;
    f0 = d * (float)(s0 & 0x3fu);
    b0 = dm * (float)(s1 & 0x3fu);
    f1 = d * (float)((s0 >> 8) & 0x3fu);
    b1 = dm * (float)((s1 >> 8) & 0x3fu);
    f2 = d * (float)((s2 & 0xfu) | ((s0 & 0xc0u) >> 2));
    b2 = dm * (float)(((s2 & 0xf0u) >> 4) | ((s1 & 0xc0u) >> 2));
    f3 = d * (float)(((s2 >> 8) & 0xfu) | (((s0 >> 8) & 0xc0u) >> 2));
    b3 = dm * (float)((((s2 >> 8) & 0xf0u) >> 4) | (((s1 >> 8) & 0xc0u) >> 2));
}

__device__ __forceinline__ float zinc_q4k_q8_block_dot_scaled(
    unsigned qs0, unsigned qs1,
    int q0, int q1, int q2, int q3,
    float2 ds0, float2 ds1, float2 ds2, float2 ds3, bool min_lane,
    float f0, float f1, float f2, float f3,
    float b0, float b1, float b2, float b3)
{
    float sum = f0 * ds0.x * (float)__dp4a((int)(qs0 & 0x0f0f0f0fu), q0, 0);
    sum += f1 * ds1.x * (float)__dp4a((int)((qs0 >> 4) & 0x0f0f0f0fu), q1, 0);
    sum += f2 * ds2.x * (float)__dp4a((int)(qs1 & 0x0f0f0f0fu), q2, 0);
    sum += f3 * ds3.x * (float)__dp4a((int)((qs1 >> 4) & 0x0f0f0f0fu), q3, 0);
    if (min_lane) sum -= b0 * ds0.y + b1 * ds1.y + b2 * ds2.y + b3 * ds3.y;
    return sum;
}

extern "C" __global__ void dmmv_q4k_q8_fast(
    const unsigned* __restrict__ a,
    const unsigned char* __restrict__ xq,
    float* __restrict__ y, DmmvPush pc)
{
    const unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    const unsigned il = itid >> 2, ir = itid & 3u;
    const unsigned v_im = il >> 1, v_in = il & 1u;
    const unsigned l0 = 4u * (2u * ir + v_in);
    const unsigned q_off = 32u * v_im + l0;
    const unsigned shift = v_im * 16u;
    const unsigned ngrp = blockDim.x >> 4;
    const unsigned bpr = pc.K >> 8;
    const unsigned row_base = (pc.a_offset >> 2) + row * bpr * 36u;
    float sum = 0.0f;

    for (unsigned sb = grp; sb < bpr; sb += ngrp) {
        const unsigned char* h0 = xq + (size_t)(2u * sb) * 144u;
        const unsigned char* h1 = h0 + 144u;
        const unsigned g = 2u * v_im;
        const int q0 = *(const int*)(h0 + 16u + g * 32u + l0);
        const int q1 = *(const int*)(h0 + 16u + (g + 1u) * 32u + l0);
        const int q2 = *(const int*)(h1 + 16u + g * 32u + l0);
        const int q3 = *(const int*)(h1 + 16u + (g + 1u) * 32u + l0);
        const float2 ds0 = __half22float2(*(const half2*)(h0 + g * 4u));
        const float2 ds1 = __half22float2(*(const half2*)(h0 + (g + 1u) * 4u));
        const float2 ds2 = __half22float2(*(const half2*)(h1 + g * 4u));
        const float2 ds3 = __half22float2(*(const half2*)(h1 + (g + 1u) * 4u));
        const unsigned blk = row_base + sb * 36u;
        sum += zinc_q4k_q8_block_dot(a, blk, q_off, shift, q0, q1, q2, q3, ds0, ds1, ds2, ds3, l0 == 0u);
    }

    sum = zinc_block_reduce_sum(sum);
    if (tid == 0u) {
        const unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum;
        else y[yi] = sum;
    }
}

__device__ __forceinline__ float zinc_q5k_q8_block_dot(
    const unsigned* a, unsigned blk, unsigned q_off, unsigned l0, unsigned shift,
    int q0, int q1, int q2, int q3,
    float2 ds0, float2 ds1, float2 ds2, float2 ds3, bool min_lane)
{
    const unsigned dd = a[blk];
    const float d = zinc_half_to_float((unsigned short)(dd & 0xffffu));
    const float dm = zinc_half_to_float((unsigned short)(dd >> 16));
    const unsigned sc0 = a[blk + 1u], sc1 = a[blk + 2u], sc2 = a[blk + 3u];
    const unsigned qh = a[blk + 4u + (l0 >> 2)];
    const unsigned qs0 = a[blk + 12u + (q_off >> 2)];
    const unsigned qs1 = a[blk + 12u + (q_off >> 2) + 16u];
    const unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
    const float f0 = d * (float)(s0 & 0x3fu), b0 = dm * (float)(s1 & 0x3fu);
    const float f1 = d * (float)((s0 >> 8) & 0x3fu), b1 = dm * (float)((s1 >> 8) & 0x3fu);
    const float f2 = d * (float)((s2 & 0xfu) | ((s0 & 0xc0u) >> 2));
    const float b2 = dm * (float)(((s2 & 0xf0u) >> 4) | ((s1 & 0xc0u) >> 2));
    const float f3 = d * (float)(((s2 >> 8) & 0xfu) | (((s0 >> 8) & 0xc0u) >> 2));
    const float b3 = dm * (float)((((s2 >> 8) & 0xf0u) >> 4) | (((s1 >> 8) & 0xc0u) >> 2));
    const unsigned hb = 2u * (shift >> 4);
    const int v0 = (int)((qs0 & 0x0f0f0f0fu) | (((qh >> hb) & 0x01010101u) << 4));
    const int v1 = (int)(((qs0 >> 4) & 0x0f0f0f0fu) | (((qh >> (hb + 1u)) & 0x01010101u) << 4));
    const int v2 = (int)((qs1 & 0x0f0f0f0fu) | (((qh >> (hb + 4u)) & 0x01010101u) << 4));
    const int v3 = (int)(((qs1 >> 4) & 0x0f0f0f0fu) | (((qh >> (hb + 5u)) & 0x01010101u) << 4));
    float sum = f0 * ds0.x * (float)__dp4a(v0, q0, 0);
    sum += f1 * ds1.x * (float)__dp4a(v1, q1, 0);
    sum += f2 * ds2.x * (float)__dp4a(v2, q2, 0);
    sum += f3 * ds3.x * (float)__dp4a(v3, q3, 0);
    if (min_lane) sum -= b0 * ds0.y + b1 * ds1.y + b2 * ds2.y + b3 * ds3.y;
    return sum;
}

extern "C" __global__ void dmmv_q5k_q8_fast(
    const unsigned* __restrict__ a,
    const unsigned char* __restrict__ xq,
    float* __restrict__ y, DmmvPush pc)
{
    const unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    const unsigned il = itid >> 2, ir = itid & 3u;
    const unsigned v_im = il >> 1, v_in = il & 1u;
    const unsigned l0 = 4u * (2u * ir + v_in);
    const unsigned q_off = 32u * v_im + l0;
    const unsigned shift = v_im * 16u;
    const unsigned ngrp = blockDim.x >> 4;
    const unsigned bpr = pc.K >> 8;
    const unsigned row_base = (pc.a_offset >> 2) + row * bpr * 44u;
    float sum = 0.0f;

    for (unsigned sb = grp; sb < bpr; sb += ngrp) {
        const unsigned char* h0 = xq + (size_t)(2u * sb) * 144u;
        const unsigned char* h1 = h0 + 144u;
        const unsigned g = 2u * v_im;
        const int q0 = *(const int*)(h0 + 16u + g * 32u + l0);
        const int q1 = *(const int*)(h0 + 16u + (g + 1u) * 32u + l0);
        const int q2 = *(const int*)(h1 + 16u + g * 32u + l0);
        const int q3 = *(const int*)(h1 + 16u + (g + 1u) * 32u + l0);
        const float2 ds0 = __half22float2(*(const half2*)(h0 + g * 4u));
        const float2 ds1 = __half22float2(*(const half2*)(h0 + (g + 1u) * 4u));
        const float2 ds2 = __half22float2(*(const half2*)(h1 + g * 4u));
        const float2 ds3 = __half22float2(*(const half2*)(h1 + (g + 1u) * 4u));
        const unsigned blk = row_base + sb * 44u;
        sum += zinc_q5k_q8_block_dot(a, blk, q_off, l0, shift, q0, q1, q2, q3, ds0, ds1, ds2, ds3, l0 == 0u);
    }

    sum = zinc_block_reduce_sum(sum);
    if (tid == 0u) {
        const unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum;
        else y[yi] = sum;
    }
}

extern "C" __global__ void dmmv_q4k_gate_up_swiglu_q8(
    const unsigned* __restrict__ gate,
    const unsigned* __restrict__ up,
    const unsigned char* __restrict__ xq,
    float* __restrict__ y, DmmvPush pc)
{
    const unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    const unsigned il = itid >> 2, ir = itid & 3u;
    const unsigned v_im = il >> 1, v_in = il & 1u;
    const unsigned l0 = 4u * (2u * ir + v_in);
    const unsigned q_off = 32u * v_im + l0;
    const unsigned shift = v_im * 16u;
    const unsigned ngrp = blockDim.x >> 4;
    const unsigned bpr = pc.K >> 8;
    const unsigned row_base = row * bpr * 36u;
    float sum_gate = 0.0f, sum_up = 0.0f;

    #pragma unroll 5
    for (unsigned sb = grp; sb < bpr; sb += ngrp) {
        const unsigned char* h0 = xq + (size_t)(2u * sb) * 144u;
        const unsigned char* h1 = h0 + 144u;
        const unsigned g = 2u * v_im;
        const int q0 = *(const int*)(h0 + 16u + g * 32u + l0);
        const int q1 = *(const int*)(h0 + 16u + (g + 1u) * 32u + l0);
        const int q2 = *(const int*)(h1 + 16u + g * 32u + l0);
        const int q3 = *(const int*)(h1 + 16u + (g + 1u) * 32u + l0);
        const float2 ds0 = __half22float2(*(const half2*)(h0 + g * 4u));
        const float2 ds1 = __half22float2(*(const half2*)(h0 + (g + 1u) * 4u));
        const float2 ds2 = __half22float2(*(const half2*)(h1 + g * 4u));
        const float2 ds3 = __half22float2(*(const half2*)(h1 + (g + 1u) * 4u));
        const bool min_lane = l0 == 0u;
        const unsigned blk = row_base + sb * 36u;
        const unsigned q_word = 4u + (q_off >> 2);
        const unsigned gate_qs0 = gate[blk + q_word];
        const unsigned gate_qs1 = gate[blk + q_word + 16u];
        const unsigned up_qs0 = up[blk + q_word];
        const unsigned up_qs1 = up[blk + q_word + 16u];
        uint4 scale_meta = {};
        const bool scale_lane = (itid & 3u) == 0u;
        const unsigned* scale_matrix = (itid & 4u) != 0u ? up : gate;
        if (scale_lane) scale_meta = *(const uint4*)(scale_matrix + blk);
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
        if (scale_lane)
            zinc_q4k_q8_scale_factors(scale_meta, shift, f0, f1, f2, f3, b0, b1, b2, b3);
        const unsigned gate_scale_src = (grp & 1u) * 16u + v_im * 8u;
        const float gf0 = __shfl_sync(0xffffffffu, f0, gate_scale_src);
        const float gf1 = __shfl_sync(0xffffffffu, f1, gate_scale_src);
        const float gf2 = __shfl_sync(0xffffffffu, f2, gate_scale_src);
        const float gf3 = __shfl_sync(0xffffffffu, f3, gate_scale_src);
        const float gb0 = __shfl_sync(0xffffffffu, b0, gate_scale_src);
        const float gb1 = __shfl_sync(0xffffffffu, b1, gate_scale_src);
        const float gb2 = __shfl_sync(0xffffffffu, b2, gate_scale_src);
        const float gb3 = __shfl_sync(0xffffffffu, b3, gate_scale_src);
        sum_gate += zinc_q4k_q8_block_dot_scaled(gate_qs0, gate_qs1,
            q0, q1, q2, q3, ds0, ds1, ds2, ds3, min_lane,
            gf0, gf1, gf2, gf3, gb0, gb1, gb2, gb3);
        const unsigned up_scale_src = gate_scale_src + 4u;
        const float uf0 = __shfl_sync(0xffffffffu, f0, up_scale_src);
        const float uf1 = __shfl_sync(0xffffffffu, f1, up_scale_src);
        const float uf2 = __shfl_sync(0xffffffffu, f2, up_scale_src);
        const float uf3 = __shfl_sync(0xffffffffu, f3, up_scale_src);
        const float ub0 = __shfl_sync(0xffffffffu, b0, up_scale_src);
        const float ub1 = __shfl_sync(0xffffffffu, b1, up_scale_src);
        const float ub2 = __shfl_sync(0xffffffffu, b2, up_scale_src);
        const float ub3 = __shfl_sync(0xffffffffu, b3, up_scale_src);
        sum_up += zinc_q4k_q8_block_dot_scaled(up_qs0, up_qs1,
            q0, q1, q2, q3, ds0, ds1, ds2, ds3, min_lane,
            uf0, uf1, uf2, uf3, ub0, ub1, ub2, ub3);
    }

    zinc_block_reduce_sum_pair(sum_gate, sum_up);
    if (tid == 0u) {
        if (pc.acc_mode != 0u) {
            const float k = 0.7978845608028654f;
            const float gelu = 0.5f * sum_gate *
                (1.0f + tanhf(k * (sum_gate + 0.044715f * sum_gate * sum_gate * sum_gate)));
            y[row] = gelu * sum_up;
        } else {
            const float silu = sum_gate / (1.0f + expf(-sum_gate));
            y[row] = silu * sum_up;
        }
    }
}

// Exact-route MoE twin of the Q8 gate/up matvec. Unlike padded grouped WMMA,
// grid.y contains only real routed rows; gate and up share activation loads.
extern "C" __global__ void dmmv_q4k_experts_grouped_q8_dual(
    const unsigned* __restrict__ a,
    const unsigned char* __restrict__ xq,
    float* __restrict__ gate,
    float* __restrict__ up,
    const unsigned* __restrict__ expert_ids,
    const unsigned* __restrict__ order,
    ExpertsQ8Push pc)
{
    const unsigned packed = order[blockIdx.y];
    const unsigned tok = packed >> 16;
    const unsigned slot = packed & 0xffffu;
    const unsigned expert = expert_ids[(size_t)tok * pc.routing_stride + slot];
    const unsigned tid = threadIdx.x;
    const unsigned row_group = tid >> 4;
    const unsigned itid = tid & 15u;
    const unsigned row = blockIdx.x * 16u + row_group;
    if (row >= pc.M) return;
    const unsigned il = itid >> 2, ir = itid & 3u;
    const unsigned v_im = il >> 1, v_in = il & 1u;
    const unsigned l0 = 4u * (2u * ir + v_in);
    const unsigned q_off = 32u * v_im + l0;
    const unsigned shift = v_im * 16u;
    const unsigned bpr = pc.K >> 8;
    const unsigned expert_base = (expert * pc.slice) >> 2;
    const unsigned gate_base = expert_base + row * bpr * 36u;
    const unsigned up_base = expert_base + (pc.up_base >> 2) + row * bpr * 36u;
    float sum_gate = 0.0f, sum_up = 0.0f;

    for (unsigned sb = 0u; sb < bpr; ++sb) {
        const unsigned g = 2u * v_im;
        const unsigned char* h0 = xq + ((size_t)(2u * sb) * pc.T + tok) * 144u;
        const unsigned char* h1 = xq + ((size_t)(2u * sb + 1u) * pc.T + tok) * 144u;
        const int q0 = *(const int*)(h0 + 16u + g * 32u + l0);
        const int q1 = *(const int*)(h0 + 16u + (g + 1u) * 32u + l0);
        const int q2 = *(const int*)(h1 + 16u + g * 32u + l0);
        const int q3 = *(const int*)(h1 + 16u + (g + 1u) * 32u + l0);
        const float2 ds0 = __half22float2(*(const half2*)(h0 + g * 4u));
        const float2 ds1 = __half22float2(*(const half2*)(h0 + (g + 1u) * 4u));
        const float2 ds2 = __half22float2(*(const half2*)(h1 + g * 4u));
        const float2 ds3 = __half22float2(*(const half2*)(h1 + (g + 1u) * 4u));
        const bool min_lane = l0 == 0u;
        sum_gate += zinc_q4k_q8_block_dot(a, gate_base + sb * 36u, q_off, shift,
            q0, q1, q2, q3, ds0, ds1, ds2, ds3, min_lane);
        sum_up += zinc_q4k_q8_block_dot(a, up_base + sb * 36u, q_off, shift,
            q0, q1, q2, q3, ds0, ds1, ds2, ds3, min_lane);
    }

    #pragma unroll
    for (unsigned off = 8u; off > 0u; off >>= 1) {
        sum_gate += __shfl_down_sync(0xffffffffu, sum_gate, off);
        sum_up += __shfl_down_sync(0xffffffffu, sum_up, off);
    }
    if (itid == 0u) {
        const size_t yi = (size_t)tok * pc.dst_tok_stride + (size_t)slot * pc.M + row;
        if (pc.fuse_activation != 0u) {
            const float k = 0.7978845608028654f;
            const float gelu = 0.5f * sum_gate *
                (1.0f + tanhf(k * (sum_gate + 0.044715f * sum_gate * sum_gate * sum_gate)));
            gate[yi] = gelu * sum_up;
        } else {
            gate[yi] = sum_gate;
            up[yi] = sum_up;
        }
    }
}

// Exact-route affine Q5_1 down matvec against routed Q8_1 activations. Eight
// threads consume one 32-value quant block via packed dp4a; no padded columns.
template <unsigned RM>
__device__ __forceinline__ void zinc_dmmv_q5_1_experts_grouped_q8(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ xq,
    float* __restrict__ y,
    const unsigned* __restrict__ expert_ids,
    const unsigned* __restrict__ order,
    ExpertsQ8Push pc)
{
    const unsigned packed = order[blockIdx.y];
    const unsigned tok = packed >> 16;
    const unsigned slot = packed & 0xffffu;
    const unsigned expert = expert_ids[(size_t)tok * pc.routing_stride + slot];
    const unsigned tid = threadIdx.x;
    const unsigned row_group = tid >> 3;
    const unsigned lane = tid & 7u;
    const unsigned row = blockIdx.x * RM + row_group;
    if (row >= pc.M) return;
    const unsigned bpr = pc.K >> 5;
    const unsigned src_row = tok * pc.n_used + slot;
    const unsigned char* const arow = a + (size_t)expert * pc.slice +
        (size_t)row * bpr * 24u;
    float sum = 0.0f;

    for (unsigned b = 0u; b < bpr; ++b) {
        const unsigned char* blk = arow + (size_t)b * 24u;
        const float2 dm = __half22float2(*(const half2*)blk);
        const unsigned qh = *(const unsigned*)(blk + 4u);
        const unsigned char* qs = blk + 8u;
        unsigned wq = 0u;
        #pragma unroll
        for (unsigned j = 0u; j < 4u; ++j) {
            const unsigned k = lane * 4u + j;
            const unsigned ql = k < 16u ? (unsigned)(qs[k] & 0xfu) : (unsigned)(qs[k - 16u] >> 4);
            const unsigned q = ql | (((qh >> k) & 1u) << 4);
            wq |= q << (j * 8u);
        }
        const unsigned h = b >> 2;
        const unsigned g = b & 3u;
        const unsigned char* xb = xq + ((size_t)h * pc.T + src_row) * 144u;
        const int xqi = *(const int*)(xb + 16u + g * 32u + lane * 4u);
        const float2 ds = __half22float2(*(const half2*)(xb + g * 4u));
        sum += dm.x * ds.x * (float)__dp4a((int)wq, xqi, 0);
        if (lane == 0u) sum += dm.y * ds.y;
    }

    #pragma unroll
    for (unsigned off = 4u; off > 0u; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0u) {
        y[(size_t)tok * pc.dst_tok_stride + (size_t)slot * pc.M + row] = sum;
    }
}

extern "C" __global__ void dmmv_q5_1_experts_grouped_q8(
    const unsigned char* a, const unsigned char* xq, float* y,
    const unsigned* expert_ids, const unsigned* order, ExpertsQ8Push pc) {
    zinc_dmmv_q5_1_experts_grouped_q8<32u>(a, xq, y, expert_ids, order, pc);
}

extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_m8(
    const unsigned char* a, const unsigned char* xq, float* y,
    const unsigned* expert_ids, const unsigned* order, ExpertsQ8Push pc) {
    zinc_dmmv_q5_1_experts_grouped_q8<8u>(a, xq, y, expert_ids, order, pc);
}

extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_m16(
    const unsigned char* a, const unsigned char* xq, float* y,
    const unsigned* expert_ids, const unsigned* order, ExpertsQ8Push pc) {
    zinc_dmmv_q5_1_experts_grouped_q8<16u>(a, xq, y, expert_ids, order, pc);
}

extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_m64(
    const unsigned char* a, const unsigned char* xq, float* y,
    const unsigned* expert_ids, const unsigned* order, ExpertsQ8Push pc) {
    zinc_dmmv_q5_1_experts_grouped_q8<64u>(a, xq, y, expert_ids, order, pc);
}

// Four-route Q5_1 down tile. Sorting routes into four-wide expert buckets lets
// every weight load feed up to four routed activations while limiting short-
// prompt padding to at most three rows per expert.
template <unsigned RT>
__device__ __forceinline__ void zinc_dmmv_q5_1_experts_grouped_q8_tiled(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ xq,
    float* __restrict__ y,
    const unsigned* __restrict__ order,
    const unsigned* __restrict__ tile_expert,
    ExpertsQ8Push pc)
{
    const unsigned INV = 0xffffffffu;
    const unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return;
    const unsigned tid = threadIdx.x;
    const unsigned row_group = tid >> 3;
    const unsigned lane = tid & 7u;
    const unsigned row = blockIdx.x * 32u + row_group;
    if (row >= pc.M) return;
    const unsigned bpr = pc.K >> 5;
    const unsigned char* const arow = a + (size_t)expert * pc.slice +
        (size_t)row * bpr * 24u;
    const unsigned t0 = blockIdx.y * RT;
    unsigned packed[RT];
    float sum[RT] = {};
    #pragma unroll
    for (unsigned r = 0u; r < RT; ++r) packed[r] = order[t0 + r];

    for (unsigned b = 0u; b < bpr; ++b) {
        const unsigned char* blk = arow + (size_t)b * 24u;
        const float2 dm = __half22float2(*(const half2*)blk);
        const unsigned qh = *(const unsigned*)(blk + 4u);
        const unsigned char* qs = blk + 8u;
        unsigned wq = 0u;
        #pragma unroll
        for (unsigned j = 0u; j < 4u; ++j) {
            const unsigned k = lane * 4u + j;
            const unsigned ql = k < 16u ? (unsigned)(qs[k] & 0xfu) : (unsigned)(qs[k - 16u] >> 4);
            const unsigned q = ql | (((qh >> k) & 1u) << 4);
            wq |= q << (j * 8u);
        }
        const unsigned h = b >> 2;
        const unsigned g = b & 3u;
        #pragma unroll
        for (unsigned r = 0u; r < RT; ++r) {
            if (packed[r] != INV) {
                const unsigned tok = packed[r] >> 16;
                const unsigned slot = packed[r] & 0xffffu;
                const unsigned src_row = tok * pc.n_used + slot;
                const unsigned char* xb = xq + ((size_t)h * pc.T + src_row) * 144u;
                const int xqi = *(const int*)(xb + 16u + g * 32u + lane * 4u);
                const float2 ds = __half22float2(*(const half2*)(xb + g * 4u));
                sum[r] += dm.x * ds.x * (float)__dp4a((int)wq, xqi, 0);
                if (lane == 0u) sum[r] += dm.y * ds.y;
            }
        }
    }

    #pragma unroll
    for (unsigned off = 4u; off > 0u; off >>= 1) {
        #pragma unroll
        for (unsigned r = 0u; r < RT; ++r)
            sum[r] += __shfl_down_sync(0xffffffffu, sum[r], off);
    }
    if (lane == 0u) {
        #pragma unroll
        for (unsigned r = 0u; r < RT; ++r) {
            if (packed[r] != INV) {
                const unsigned tok = packed[r] >> 16;
                const unsigned slot = packed[r] & 0xffffu;
                y[(size_t)tok * pc.dst_tok_stride + (size_t)slot * pc.M + row] = sum[r];
            }
        }
    }
}

extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_t4(
    const unsigned char* a, const unsigned char* xq, float* y,
    const unsigned* order, const unsigned* tile_expert, ExpertsQ8Push pc) {
    zinc_dmmv_q5_1_experts_grouped_q8_tiled<4u>(a, xq, y, order, tile_expert, pc);
}

extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_t8(
    const unsigned char* a, const unsigned char* xq, float* y,
    const unsigned* order, const unsigned* tile_expert, ExpertsQ8Push pc) {
    zinc_dmmv_q5_1_experts_grouped_q8_tiled<8u>(a, xq, y, order, tile_expert, pc);
}

extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_t16(
    const unsigned char* a, const unsigned char* xq, float* y,
    const unsigned* order, const unsigned* tile_expert, ExpertsQ8Push pc) {
    zinc_dmmv_q5_1_experts_grouped_q8_tiled<16u>(a, xq, y, order, tile_expert, pc);
}

// Q8_0 weight x Q8_1 activation decode matvec. Eight lanes consume one
// 32-value block with packed dot products; the remaining thread groups split
// the row's blocks and reduce once at the end.
extern "C" __global__ void dmmv_q8_0_q8_fast(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ xq,
    float* __restrict__ y, DmmvPush pc)
{
    const unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 7u;
    const unsigned grp = tid >> 3;
    const unsigned ngrp = blockDim.x >> 3;
    const unsigned bpr = pc.K >> 5;
    const unsigned char* const arow = a + pc.a_offset + (size_t)row * bpr * 34u;
    float sum = 0.0f;
    for (unsigned b = grp; b < bpr; b += ngrp) {
        const unsigned char* wb = arow + (size_t)b * 34u;
        const float dw = __half2float(*(const half*)wb);
        const int wq = *(const int*)(wb + 2u + lane * 4u);
        const unsigned h = b >> 2;
        const unsigned g = b & 3u;
        const unsigned char* xb = xq + (size_t)h * 144u;
        const float dx = __half2float(*(const half*)(xb + g * 4u));
        const int xqi = *(const int*)(xb + 16u + g * 32u + lane * 4u);
        sum += dw * dx * (float)__dp4a(wq, xqi, 0);
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0u) y[(pc.y_offset >> 2) + row] = sum;
}
#else
extern "C" __global__ void dmmv_q4k_q8_fast() {}
extern "C" __global__ void dmmv_q5k_q8_fast() {}
extern "C" __global__ void dmmv_q4k_gate_up_swiglu_q8() {}
extern "C" __global__ void dmmv_q4k_experts_grouped_q8_dual() {}
extern "C" __global__ void dmmv_q5_1_experts_grouped_q8() {}
extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_m8() {}
extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_m16() {}
extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_m64() {}
extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_t4() {}
extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_t8() {}
extern "C" __global__ void dmmv_q5_1_experts_grouped_q8_t16() {}
extern "C" __global__ void dmmv_q8_0_q8_fast() {}
#endif

// ---- dmmv_q4k_fast_dual — fuse two same-input Q4_K matvecs into ONE launch ----
// Both weights (a0,a1) share input x and inner dim K; outputs go to y0,y1. Grid
// is M0+M1 blocks: block bx<M0 computes row bx of a0→y0, else row bx-M0 of a1→y1.
// Used for the gemma FFN gate/up pair and the attention Q/K pair (both Q4_K, same
// norm input) to remove one kernel-launch boundary per layer. Each block's work
// is bit-identical to the standalone dmmv_q4k_fast with zero offsets (no acc).
struct Dmmv2Push { unsigned M0, M1, K, pair_reduce; };
struct Dmmv3Push { unsigned M0, M1, M2, K; };
extern "C" __global__ void dmmv_q4k_fast_dual(const unsigned* a0, const unsigned* a1, const float* x, float* y0, float* y1, Dmmv2Push pc) {
    unsigned bx = blockIdx.x;
    if (bx >= pc.M0 + pc.M1) return;
    unsigned bpr = pc.K >> 8;
    const unsigned* a; float* y; unsigned row;
    if (bx < pc.M0) { a = a0; y = y0; row = bx; }
    else { a = a1; y = y1; row = bx - pc.M0; }
    unsigned a_base = row * bpr * 36u;
    const float4* xv = (const float4*)x;
    float sum = zinc_dmmv_q4k_fast_sum(a, a_base, xv, bpr);
    if (threadIdx.x == 0) y[row] = sum;
}

// True same-row pair: M0 must be >= M1. Paired rows share activation loads and
// block scheduling; the M0 tail computes only the first projection.
extern "C" __global__ void dmmv_q4k_pair(
    const unsigned* __restrict__ a0, const unsigned* __restrict__ a1,
    const float* __restrict__ x, float* __restrict__ y0,
    float* __restrict__ y1, Dmmv2Push pc)
{
    const unsigned row = blockIdx.x;
    if (row >= pc.M0) return;
    const bool paired = row < pc.M1;
    const unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    const unsigned il = itid >> 2, ir = itid & 3u;
    const unsigned v_im = il >> 1, v_in = il & 1u;
    const unsigned l0 = 4u * (2u * ir + v_in);
    const unsigned q_off = 32u * v_im + l0;
    const unsigned y_loc = 64u * v_im + l0;
    const unsigned shift = v_im * 16u;
    const unsigned ngrp = blockDim.x >> 4;
    const unsigned bpr = pc.K >> 8;
    const unsigned row_base = row * bpr * 36u;
    const float4* xv = (const float4*)x;
    float sum0 = 0.0f, sum1 = 0.0f;

    for (unsigned sb = grp; sb < bpr; sb += ngrp) {
        const unsigned bidx = (sb * 256u + y_loc) >> 2;
        const unsigned bidx2 = (sb * 256u + y_loc + 128u) >> 2;
        const float4 by0 = xv[bidx], by1 = xv[bidx + 8u];
        const float4 by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        const unsigned blk = row_base + sb * 36u;
        sum0 += zinc_q4k_fast_block_dot(a0, blk, q_off, shift, by0, by1, by2, by3);
        if (paired)
            sum1 += zinc_q4k_fast_block_dot(a1, blk, q_off, shift, by0, by1, by2, by3);
    }

    if (paired && pc.pair_reduce != 0u) {
        zinc_block_reduce_sum_pair(sum0, sum1);
    } else {
        sum0 = zinc_block_reduce_sum(sum0);
        if (paired) {
            __syncthreads();
            sum1 = zinc_block_reduce_sum(sum1);
        }
    }
    if (tid == 0u) {
        y0[row] = sum0;
        if (paired) y1[row] = sum1;
    }
}

#ifdef ZINC_ROCM
extern "C" __global__ void dmmv_q4k_pair_q8(
    const unsigned* __restrict__ a0, const unsigned* __restrict__ a1,
    const unsigned char* __restrict__ xq, float* __restrict__ y0,
    float* __restrict__ y1, Dmmv2Push pc)
{
    const unsigned row = blockIdx.x;
    if (row >= pc.M0) return;
    const bool paired = row < pc.M1;
    const unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    const unsigned il = itid >> 2, ir = itid & 3u;
    const unsigned v_im = il >> 1, v_in = il & 1u;
    const unsigned l0 = 4u * (2u * ir + v_in);
    const unsigned q_off = 32u * v_im + l0;
    const unsigned shift = v_im * 16u;
    const unsigned ngrp = blockDim.x >> 4;
    const unsigned bpr = pc.K >> 8;
    const unsigned row_base = row * bpr * 36u;
    float sum0 = 0.0f, sum1 = 0.0f;

    for (unsigned sb = grp; sb < bpr; sb += ngrp) {
        const unsigned char* h0 = xq + (size_t)(2u * sb) * 144u;
        const unsigned char* h1 = h0 + 144u;
        const unsigned g = 2u * v_im;
        const int q0 = *(const int*)(h0 + 16u + g * 32u + l0);
        const int q1 = *(const int*)(h0 + 16u + (g + 1u) * 32u + l0);
        const int q2 = *(const int*)(h1 + 16u + g * 32u + l0);
        const int q3 = *(const int*)(h1 + 16u + (g + 1u) * 32u + l0);
        const float2 ds0 = __half22float2(*(const half2*)(h0 + g * 4u));
        const float2 ds1 = __half22float2(*(const half2*)(h0 + (g + 1u) * 4u));
        const float2 ds2 = __half22float2(*(const half2*)(h1 + g * 4u));
        const float2 ds3 = __half22float2(*(const half2*)(h1 + (g + 1u) * 4u));
        const unsigned blk = row_base + sb * 36u;
        const bool min_lane = l0 == 0u;
        sum0 += zinc_q4k_q8_block_dot(a0, blk, q_off, shift, q0, q1, q2, q3, ds0, ds1, ds2, ds3, min_lane);
        if (paired)
            sum1 += zinc_q4k_q8_block_dot(a1, blk, q_off, shift, q0, q1, q2, q3, ds0, ds1, ds2, ds3, min_lane);
    }

    if (paired && pc.pair_reduce != 0u) {
        zinc_block_reduce_sum_pair(sum0, sum1);
    } else {
        sum0 = zinc_block_reduce_sum(sum0);
        if (paired) {
            __syncthreads();
            sum1 = zinc_block_reduce_sum(sum1);
        }
    }
    if (tid == 0u) {
        y0[row] = sum0;
        if (paired) y1[row] = sum1;
    }
}
#else
extern "C" __global__ void dmmv_q4k_pair_q8() {}
#endif

// ---- dmmv_q6k_fast (perf research) — port of tuned Vulkan dmmv_q6k -----------
extern "C" __global__ void dmmv_q6k_fast(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    unsigned tok = blockIdx.y;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 210u;
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2) + (size_t)tok * pc.K);
    unsigned tid = threadIdx.x, itid = tid & 15u, ix = tid >> 4;
    unsigned half_id = itid >> 3, local_id = itid & 7u, e_start = local_id * 4u, is = e_start >> 4;
    unsigned xvib = (half_id * 128u + e_start) >> 2, ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned b = ix; b < bpr; b += ngrp) {
        const unsigned char* bb = arow + (size_t)b * 210u;
        float d = zinc_half_to_float((unsigned short)((unsigned)bb[208] | ((unsigned)bb[209] << 8)));
        const unsigned char* ql = bb + half_id * 64u;
        const unsigned char* qh = bb + 128u + half_id * 32u;
        const signed char* sc = (const signed char*)(bb + 192u + half_id * 8u);
        float ds0 = d * (float)sc[is], ds2 = d * (float)sc[is + 2], ds4 = d * (float)sc[is + 4], ds6 = d * (float)sc[is + 6];
        unsigned xb = (b * 256u) / 4u + xvib;
        float4 bx0 = xv[xb], bx32 = xv[xb + 8u], bx64 = xv[xb + 16u], bx96 = xv[xb + 24u];
        #pragma unroll
        for (unsigned li = 0; li < 4u; li++) {
            unsigned l = e_start + li, qllo = ql[l], qlhi = ql[l + 32u], qhv = qh[l];
            float q1 = (float)((qllo & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32.0f;
            float q2 = (float)((qlhi & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32.0f;
            float q3 = (float)((qllo >> 4) | (((qhv >> 4) & 3u) << 4)) - 32.0f;
            float q4 = (float)((qlhi >> 4) | (((qhv >> 6) & 3u) << 4)) - 32.0f;
            sum += ds0*q1*(&bx0.x)[li] + ds2*q2*(&bx32.x)[li] + ds4*q3*(&bx64.x)[li] + ds6*q4*(&bx96.x)[li];
        }
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) { unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row; if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum; }
}

// Packed Q6_K x Q8_1 decode matvec. Each 16-thread group keeps the proven
// superblock/row mapping from dmmv_q6k_fast, but consumes four packed activation
// words and issues four int8 dot products instead of sixteen scalar float FMAs.
#ifdef ZINC_ROCM
__device__ __forceinline__ unsigned zinc_q6_decode_signed4(unsigned q) {
    q ^= 0x20202020u;
    const unsigned sign = q & 0x20202020u;
    return q | (sign << 1) | (sign << 2);
}

extern "C" __global__ void dmmv_q6k_q8_fast(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ xq,
    float* __restrict__ y, DmmvPush pc)
{
    const unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 210u;
    const unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    const unsigned half_id = itid >> 3;
    const unsigned local_id = itid & 7u;
    const unsigned e_start = local_id * 4u;
    const unsigned is = e_start >> 4;
    const unsigned ngrp = blockDim.x >> 4;
    float sum = 0.0f;

    for (unsigned sb = grp; sb < bpr; sb += ngrp) {
        const unsigned char* bb = arow + (size_t)sb * 210u;
        const float d = zinc_half_to_float((unsigned short)((unsigned)bb[208] | ((unsigned)bb[209] << 8)));
        const unsigned ql0 = *(const unsigned*)(bb + half_id * 64u + e_start);
        const unsigned ql1 = *(const unsigned*)(bb + half_id * 64u + 32u + e_start);
        const unsigned qh = *(const unsigned*)(bb + 128u + half_id * 32u + e_start);
        const int v0 = (int)zinc_q6_decode_signed4((ql0 & 0x0f0f0f0fu) | ((qh << 4) & 0x30303030u));
        const int v1 = (int)zinc_q6_decode_signed4((ql1 & 0x0f0f0f0fu) | ((qh << 2) & 0x30303030u));
        const int v2 = (int)zinc_q6_decode_signed4(((ql0 >> 4) & 0x0f0f0f0fu) | (qh & 0x30303030u));
        const int v3 = (int)zinc_q6_decode_signed4(((ql1 >> 4) & 0x0f0f0f0fu) | ((qh >> 2) & 0x30303030u));
        const signed char* sc = (const signed char*)(bb + 192u + half_id * 8u);

        const unsigned char* xh = xq + (size_t)(2u * sb + half_id) * 144u;
        const int q0 = *(const int*)(xh + 16u + 0u * 32u + e_start);
        const int q1 = *(const int*)(xh + 16u + 1u * 32u + e_start);
        const int q2 = *(const int*)(xh + 16u + 2u * 32u + e_start);
        const int q3 = *(const int*)(xh + 16u + 3u * 32u + e_start);
        const float d0 = __half2float(*(const half*)(xh + 0u * 4u));
        const float d1 = __half2float(*(const half*)(xh + 1u * 4u));
        const float d2 = __half2float(*(const half*)(xh + 2u * 4u));
        const float d3 = __half2float(*(const half*)(xh + 3u * 4u));

        sum += d * (float)sc[is] * d0 * (float)__dp4a(v0, q0, 0);
        sum += d * (float)sc[is + 2u] * d1 * (float)__dp4a(v1, q1, 0);
        sum += d * (float)sc[is + 4u] * d2 * (float)__dp4a(v2, q2, 0);
        sum += d * (float)sc[is + 6u] * d3 * (float)__dp4a(v3, q3, 0);
    }

    // The dense FFN down projection is dispatched as one native wave. Avoid
    // the generic block reducer's shared-memory rendezvous and second shuffle
    // tree in that case; the larger LM-head dispatch still takes the old path.
    sum = blockDim.x == 32u ? zinc_warp_reduce_sum(sum) : zinc_block_reduce_sum(sum);
    if (tid == 0u) {
        const unsigned yi = pc.y_offset >> 2;
        if (pc.acc_mode != 0u) y[yi + row] += sum;
        else y[yi + row] = sum;
    }
}
#else
extern "C" __global__ void dmmv_q6k_q8_fast() {}
#endif

// ---- dmmv_q5k_fast (perf research) — q4k_fast + Q5_K qh high-bit promote -----
extern "C" __global__ void dmmv_q5k_fast(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x; if (row >= pc.M) return;
    unsigned tok = blockIdx.y;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = (pc.a_offset >> 2) + row * bpr * 44u;       // 176 bytes/block
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2) + (size_t)tok * pc.K);
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u, ngrp = blockDim.x >> 4;
    unsigned sba = 2u*v_im, sbb = 2u*v_im+1u, sbc = 2u*v_im+4u, sbd = 2u*v_im+5u;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 44u, dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF)), dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk+1u], sc1 = a_u32[blk+2u], sc2 = a_u32[blk+3u];
        unsigned qh = a_u32[blk + 4u + (l0 >> 2)];
        unsigned qs0 = a_u32[blk + 12u + (q_off >> 2)], qs1 = a_u32[blk + 12u + (q_off >> 2) + 16u];
        unsigned bidx = (i*256u + y_loc) >> 2, bidx2 = (i*256u + y_loc + 128u) >> 2;
        float4 by0=xv[bidx], by1=xv[bidx+8u], by2=xv[bidx2], by3=xv[bidx2+8u];
        unsigned s0=sc0>>shift, s1=sc1>>shift, s2=sc2>>shift;
        float f0=d*(float)(s0&0x3Fu), b0=dm*(float)(s1&0x3Fu);
        float f1=d*(float)((s0>>8)&0x3Fu), b1=dm*(float)((s1>>8)&0x3Fu);
        float f2=d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2=dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
        float f3=d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3=dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
        #define Q5(q,sh,sb,j) ((float)(((q)>>(sh))&0xFu) + 16.0f*(float)(((qh)>>((sb)+(j)*8u))&1u))
        sum += (f0*Q5(qs0,0,sba,0)-b0)*by0.x + (f0*Q5(qs0,8,sba,1)-b0)*by0.y + (f0*Q5(qs0,16,sba,2)-b0)*by0.z + (f0*Q5(qs0,24,sba,3)-b0)*by0.w;
        sum += (f1*Q5(qs0,4,sbb,0)-b1)*by1.x + (f1*Q5(qs0,12,sbb,1)-b1)*by1.y + (f1*Q5(qs0,20,sbb,2)-b1)*by1.z + (f1*Q5(qs0,28,sbb,3)-b1)*by1.w;
        sum += (f2*Q5(qs1,0,sbc,0)-b2)*by2.x + (f2*Q5(qs1,8,sbc,1)-b2)*by2.y + (f2*Q5(qs1,16,sbc,2)-b2)*by2.z + (f2*Q5(qs1,24,sbc,3)-b2)*by2.w;
        sum += (f3*Q5(qs1,4,sbd,0)-b3)*by3.x + (f3*Q5(qs1,12,sbd,1)-b3)*by3.y + (f3*Q5(qs1,20,sbd,2)-b3)*by3.z + (f3*Q5(qs1,28,sbd,3)-b3)*by3.w;
        #undef Q5
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) { unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row; if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum; }
}
// ---- batched MoE expert matvecs ---------------------------------------------
// One launch over ALL n_used experts: block `g` handles (expert e=g/M, row=g%M).
// The chosen expert id is read GPU-side from `expert_ids` (router_out_buf), so
// there is NO host readback — the whole MoE block runs async and the GPU stays
// busy enough to leave its idle clock. Per-expert weight slice = id*slice; x is
// shared across experts for gate/up (x_stride=0) or per-expert for down
// (x_stride=K, i.e. swiglu[e*K..]). Output is slot-major: y[e*M + row].
struct ExpertsPush { unsigned M, K, slice, x_stride, n_used, base, fuse_activation; };

extern "C" __global__ void dmmv_q4k_experts(const unsigned* a_u32, const float* x, float* y, const unsigned* expert_ids, ExpertsPush pc) {
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = ((expert_ids[e] * pc.slice + pc.base) >> 2) + row * bpr * 36u;
    const float4* xv = (const float4*)(x + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 36u;
        unsigned dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
        float dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
        unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
        float f0 = d * (float)(s0 & 0x3Fu), b0 = dm * (float)(s1 & 0x3Fu);
        float f1 = d * (float)((s0 >> 8) & 0x3Fu), b1 = dm * (float)((s1 >> 8) & 0x3Fu);
        float f2 = d * (float)((s2 & 0xFu) | ((s0 & 0xC0u) >> 2)), b2 = dm * (float)(((s2 & 0xF0u) >> 4) | ((s1 & 0xC0u) >> 2));
        float f3 = d * (float)(((s2 >> 8) & 0xFu) | (((s0 >> 8) & 0xC0u) >> 2)), b3 = dm * (float)((((s2 >> 8) & 0xF0u) >> 4) | (((s1 >> 8) & 0xC0u) >> 2));
        sum += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
        sum += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
        sum += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
        sum += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) y[(size_t)e * pc.M + row] = sum;
}

// ---- dmmv_q4k_experts_dual — fuse the routed gate + up matvecs (gemma MoE) ----
// The fused gate_up expert tensor stores gate (rows base 0) and up (rows base
// pc.base = gu_half) contiguously per expert; both are Q4_K and share the SAME
// input x (the pre-ffn norm, x_stride=0). This kernel computes BOTH for each
// (expert e, row) in ONE launch, reading x ONCE per superblock iteration (shared
// between the gate and up accumulators) instead of once per separate launch.
// Removes one kernel launch per MoE layer with no per-token overhead (vs the
// reverted C4 graph, whose per-token re-instantiate cost cancelled the bubble it
// removed). Each accumulator's dequant arithmetic is byte-for-byte that of
// dmmv_q4k_experts → bit-identical to the two-launch path. Output slot-major:
// y_gate[e*M+row], y_up[e*M+row].
extern "C" __global__ void dmmv_q4k_experts_dual(const unsigned* a_u32, const float* x, float* y_gate, float* y_up, const unsigned* expert_ids, ExpertsPush pc) {
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 8;
    unsigned base_e = (expert_ids[e] * pc.slice) >> 2;
    unsigned a_gate = base_e + row * bpr * 36u;
    unsigned a_up = base_e + (pc.base >> 2) + row * bpr * 36u;
    const float4* xv = (const float4*)(x + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    float sg = 0.0f, su = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        // gate accumulator (weights at a_gate)
        {
            unsigned blk = a_gate + i * 36u;
            unsigned dd = a_u32[blk];
            float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
            float dm = zinc_half_to_float((unsigned short)(dd >> 16));
            unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
            unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
            unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
            float f0 = d * (float)(s0 & 0x3Fu), b0 = dm * (float)(s1 & 0x3Fu);
            float f1 = d * (float)((s0 >> 8) & 0x3Fu), b1 = dm * (float)((s1 >> 8) & 0x3Fu);
            float f2 = d * (float)((s2 & 0xFu) | ((s0 & 0xC0u) >> 2)), b2 = dm * (float)(((s2 & 0xF0u) >> 4) | ((s1 & 0xC0u) >> 2));
            float f3 = d * (float)(((s2 >> 8) & 0xFu) | (((s0 >> 8) & 0xC0u) >> 2)), b3 = dm * (float)((((s2 >> 8) & 0xF0u) >> 4) | (((s1 >> 8) & 0xC0u) >> 2));
            sg += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
            sg += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
            sg += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
            sg += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
        }
        // up accumulator (weights at a_up)
        {
            unsigned blk = a_up + i * 36u;
            unsigned dd = a_u32[blk];
            float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
            float dm = zinc_half_to_float((unsigned short)(dd >> 16));
            unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
            unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
            unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
            float f0 = d * (float)(s0 & 0x3Fu), b0 = dm * (float)(s1 & 0x3Fu);
            float f1 = d * (float)((s0 >> 8) & 0x3Fu), b1 = dm * (float)((s1 >> 8) & 0x3Fu);
            float f2 = d * (float)((s2 & 0xFu) | ((s0 & 0xC0u) >> 2)), b2 = dm * (float)(((s2 & 0xF0u) >> 4) | ((s1 & 0xC0u) >> 2));
            float f3 = d * (float)(((s2 >> 8) & 0xFu) | (((s0 >> 8) & 0xC0u) >> 2)), b3 = dm * (float)((((s2 >> 8) & 0xF0u) >> 4) | (((s1 >> 8) & 0xC0u) >> 2));
            su += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
            su += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
            su += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
            su += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
        }
    }
    sg = zinc_block_reduce_sum(sg);
    __syncthreads(); // sh[] reuse between the two reductions
    su = zinc_block_reduce_sum(su);
    if (tid == 0) {
        const unsigned o = (size_t)e * pc.M + row;
        if (pc.fuse_activation != 0u) {
            const float k = 0.7978845608028654f;
            const float gelu = 0.5f * sg *
                (1.0f + tanhf(k * (sg + 0.044715f * sg * sg * sg)));
            y_gate[o] = gelu * su;
        } else {
            y_gate[o] = sg;
            y_up[o] = su;
        }
    }
}

extern "C" __global__ void dmmv_q5k_experts(const unsigned* a_u32, const float* x, float* y, const unsigned* expert_ids, ExpertsPush pc) {
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = (expert_ids[e] * pc.slice >> 2) + row * bpr * 44u;
    const float4* xv = (const float4*)(x + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u, ngrp = blockDim.x >> 4;
    unsigned sba = 2u*v_im, sbb = 2u*v_im+1u, sbc = 2u*v_im+4u, sbd = 2u*v_im+5u;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 44u, dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF)), dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk+1u], sc1 = a_u32[blk+2u], sc2 = a_u32[blk+3u];
        unsigned qh = a_u32[blk + 4u + (l0 >> 2)];
        unsigned qs0 = a_u32[blk + 12u + (q_off >> 2)], qs1 = a_u32[blk + 12u + (q_off >> 2) + 16u];
        unsigned bidx = (i*256u + y_loc) >> 2, bidx2 = (i*256u + y_loc + 128u) >> 2;
        float4 by0=xv[bidx], by1=xv[bidx+8u], by2=xv[bidx2], by3=xv[bidx2+8u];
        unsigned s0=sc0>>shift, s1=sc1>>shift, s2=sc2>>shift;
        float f0=d*(float)(s0&0x3Fu), b0=dm*(float)(s1&0x3Fu);
        float f1=d*(float)((s0>>8)&0x3Fu), b1=dm*(float)((s1>>8)&0x3Fu);
        float f2=d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2=dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
        float f3=d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3=dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
        #define Q5E(q,sh,sb,j) ((float)(((q)>>(sh))&0xFu) + 16.0f*(float)(((qh)>>((sb)+(j)*8u))&1u))
        sum += (f0*Q5E(qs0,0,sba,0)-b0)*by0.x + (f0*Q5E(qs0,8,sba,1)-b0)*by0.y + (f0*Q5E(qs0,16,sba,2)-b0)*by0.z + (f0*Q5E(qs0,24,sba,3)-b0)*by0.w;
        sum += (f1*Q5E(qs0,4,sbb,0)-b1)*by1.x + (f1*Q5E(qs0,12,sbb,1)-b1)*by1.y + (f1*Q5E(qs0,20,sbb,2)-b1)*by1.z + (f1*Q5E(qs0,28,sbb,3)-b1)*by1.w;
        sum += (f2*Q5E(qs1,0,sbc,0)-b2)*by2.x + (f2*Q5E(qs1,8,sbc,1)-b2)*by2.y + (f2*Q5E(qs1,16,sbc,2)-b2)*by2.z + (f2*Q5E(qs1,24,sbc,3)-b2)*by2.w;
        sum += (f3*Q5E(qs1,4,sbd,0)-b3)*by3.x + (f3*Q5E(qs1,12,sbd,1)-b3)*by3.y + (f3*Q5E(qs1,20,sbd,2)-b3)*by3.z + (f3*Q5E(qs1,28,sbd,3)-b3)*by3.w;
        #undef Q5E
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) y[(size_t)e * pc.M + row] = sum;
}

// Effort 28: Q6_K expert matvec — one launch over ALL n_used experts (block g
// handles expert e=g/M, row=g%M). Same per-element dequant + block reduction as
// dmmv_q6k_fast, but the expert id is read GPU-side from `expert_ids`
// (router_out_buf) so the mixed-quant Q6_K-expert layers run on the async
// (batched_experts) path with NO host readback → they ride the launch-collapse
// instead of the per-row host-gather sync. Per-expert weight slice = id*slice +
// base (bytes); x shared for gate/up (x_stride=0) or per-expert for down
// (x_stride=K). Output slot-major y[e*M + row]. Row b == dmmv_q6k_fast on x[b].
extern "C" __global__ void dmmv_q6k_experts(const unsigned char* a, const float* x, float* y, const unsigned* expert_ids, ExpertsPush pc) {
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + ((size_t)expert_ids[e] * pc.slice + pc.base) + (size_t)row * bpr * 210u;
    const float4* xv = (const float4*)(x + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, ix = tid >> 4;
    unsigned half_id = itid >> 3, local_id = itid & 7u, e_start = local_id * 4u, is = e_start >> 4;
    unsigned xvib = (half_id * 128u + e_start) >> 2, ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned b = ix; b < bpr; b += ngrp) {
        const unsigned char* bb = arow + (size_t)b * 210u;
        float d = zinc_half_to_float((unsigned short)((unsigned)bb[208] | ((unsigned)bb[209] << 8)));
        const unsigned char* ql = bb + half_id * 64u;
        const unsigned char* qh = bb + 128u + half_id * 32u;
        const signed char* sc = (const signed char*)(bb + 192u + half_id * 8u);
        float ds0 = d * (float)sc[is], ds2 = d * (float)sc[is + 2], ds4 = d * (float)sc[is + 4], ds6 = d * (float)sc[is + 6];
        unsigned xb = (b * 256u) / 4u + xvib;
        float4 bx0 = xv[xb], bx32 = xv[xb + 8u], bx64 = xv[xb + 16u], bx96 = xv[xb + 24u];
        #pragma unroll
        for (unsigned li = 0; li < 4u; li++) {
            unsigned l = e_start + li, qllo = ql[l], qlhi = ql[l + 32u], qhv = qh[l];
            float q1 = (float)((qllo & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32.0f;
            float q2 = (float)((qlhi & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32.0f;
            float q3 = (float)((qllo >> 4) | (((qhv >> 4) & 3u) << 4)) - 32.0f;
            float q4 = (float)((qlhi >> 4) | (((qhv >> 6) & 3u) << 4)) - 32.0f;
            sum += ds0*q1*(&bx0.x)[li] + ds2*q2*(&bx32.x)[li] + ds4*q3*(&bx64.x)[li] + ds6*q4*(&bx96.x)[li];
        }
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) y[(size_t)e * pc.M + row] = sum;
}

// Batched Q5_1 expert down-proj (gemma-26b MoE): one launch over all n_used
// experts, expert id read GPU-side. Same dequant as dmmv_q5_1; per-expert x is
// swiglu[e*x_stride..]; output slot-major y[e*M + row]. base unused (0).
extern "C" __global__ void dmmv_q5_1_experts(const unsigned char* a, const float* x, float* y, const unsigned* expert_ids, ExpertsPush pc) {
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 5;
    const unsigned char* arow = a + (size_t)expert_ids[e] * pc.slice + pc.base + (size_t)row * bpr * 24u;
    const float* xrow = x + (size_t)e * pc.x_stride;
    float sum = 0.0f;
    for (unsigned el = threadIdx.x; el < pc.K; el += blockDim.x) {
        unsigned blk = el >> 5, i = el & 31u;
        const unsigned char* blkp = arow + (size_t)blk * 24u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blkp[0] | ((unsigned)blkp[1] << 8)));
        float m = zinc_half_to_float((unsigned short)((unsigned)blkp[2] | ((unsigned)blkp[3] << 8)));
        unsigned qh = (unsigned)blkp[4] | ((unsigned)blkp[5] << 8) | ((unsigned)blkp[6] << 16) | ((unsigned)blkp[7] << 24);
        const unsigned char* qs = blkp + 8;
        unsigned nib = (i < 16u) ? (unsigned)(qs[i] & 0xFu) : (unsigned)(qs[i - 16u] >> 4);
        unsigned q5 = nib | (((qh >> i) & 1u) << 4);
        sum += (d * (float)q5 + m) * xrow[el];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) y[(size_t)e * pc.M + row] = sum;
}

// ---- token-batched routed-expert matvecs (gemma-26b MoE prefill) -------------
// Effort 24 cycle 8: process ALL T prompt tokens' routed experts in ONE launch
// (grid.y = T) instead of looping the single-token dmmv_*_experts kernels. Each
// (token t, expert-slot e, row) block reads token t's router row
// (expert_ids[t*routing_stride + e]), token t's input slice (x + t*x_tok_stride,
// per-expert e*x_stride) and writes token t's output slice (y + t*y_tok_stride,
// per-expert e*M). The per-block dequant + zinc_block_reduce_sum is byte-for-byte
// the single-token kernel's, so these are bit-identical to looping the per-token
// dmmv_q4k_experts/dmmv_q5_1_experts over t — only the launch is batched.
struct ExpertsBatchPush { unsigned M, K, slice, x_stride, n_used, base, routing_stride, x_tok_stride, y_tok_stride; };

extern "C" __global__ void dmmv_q4k_experts_batched(const unsigned* a_u32, const float* x, float* y, const unsigned* expert_ids, ExpertsBatchPush pc) {
    unsigned t = blockIdx.y;
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = ((expert_ids[(size_t)t * pc.routing_stride + e] * pc.slice + pc.base) >> 2) + row * bpr * 36u;
    const float4* xv = (const float4*)(x + (size_t)t * pc.x_tok_stride + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 36u;
        unsigned dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
        float dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
        unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
        float f0 = d * (float)(s0 & 0x3Fu), b0 = dm * (float)(s1 & 0x3Fu);
        float f1 = d * (float)((s0 >> 8) & 0x3Fu), b1 = dm * (float)((s1 >> 8) & 0x3Fu);
        float f2 = d * (float)((s2 & 0xFu) | ((s0 & 0xC0u) >> 2)), b2 = dm * (float)(((s2 & 0xF0u) >> 4) | ((s1 & 0xC0u) >> 2));
        float f3 = d * (float)(((s2 >> 8) & 0xFu) | (((s0 >> 8) & 0xC0u) >> 2)), b3 = dm * (float)((((s2 >> 8) & 0xF0u) >> 4) | (((s1 >> 8) & 0xC0u) >> 2));
        sum += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
        sum += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
        sum += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
        sum += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) y[(size_t)t * pc.y_tok_stride + (size_t)e * pc.M + row] = sum;
}

extern "C" __global__ void dmmv_q5_1_experts_batched(const unsigned char* a, const float* x, float* y, const unsigned* expert_ids, ExpertsBatchPush pc) {
    unsigned t = blockIdx.y;
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 5;
    const unsigned char* arow = a + (size_t)expert_ids[(size_t)t * pc.routing_stride + e] * pc.slice + pc.base + (size_t)row * bpr * 24u;
    const float* xrow = x + (size_t)t * pc.x_tok_stride + (size_t)e * pc.x_stride;
    float sum = 0.0f;
    for (unsigned el = threadIdx.x; el < pc.K; el += blockDim.x) {
        unsigned blk = el >> 5, i = el & 31u;
        const unsigned char* blkp = arow + (size_t)blk * 24u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blkp[0] | ((unsigned)blkp[1] << 8)));
        float m = zinc_half_to_float((unsigned short)((unsigned)blkp[2] | ((unsigned)blkp[3] << 8)));
        unsigned qh = (unsigned)blkp[4] | ((unsigned)blkp[5] << 8) | ((unsigned)blkp[6] << 16) | ((unsigned)blkp[7] << 24);
        const unsigned char* qs = blkp + 8;
        unsigned nib = (i < 16u) ? (unsigned)(qs[i] & 0xFu) : (unsigned)(qs[i - 16u] >> 4);
        unsigned q5 = nib | (((qh >> i) & 1u) << 4);
        sum += (d * (float)q5 + m) * xrow[el];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) y[(size_t)t * pc.y_tok_stride + (size_t)e * pc.M + row] = sum;
}

// ---- token-batched qwen2-MoE routed-expert matvecs (qwen36 MoE prefill) -----
// Effort 29 T2: qwen36-35b-a3b prefill looped the WHOLE per-token MoE block over
// all T prompt tokens (router + 8 routed experts + shared expert ≈ 14 launches ×
// T tokens × 40 layers → heavily launch-bound; pp256 ~50 t/s). These are the
// token-batched (grid.y = T) twins of the single-token dmmv_q5k_experts /
// dmmv_q6k_experts / dmmv_f32 + moe_weighted_acc / sigmoid_scale_acc kernels:
// each (token t, expert-slot e, row) block reads token t's router row
// (expert_ids[t*routing_stride + e]), its input slice (x + t*x_tok_stride,
// per-expert e*x_stride) and writes its output slice (y + t*y_tok_stride,
// per-expert e*M). The per-block dequant + zinc_block_reduce_sum is byte-for-byte
// the single-token kernel's, so the result is bit-identical to looping the
// per-token kernels over t — only the launch is batched. ExpertsBatchPush is the
// gemma prefill struct (M,K,slice,x_stride,n_used,base,routing_stride,
// x_tok_stride,y_tok_stride); the qwen router_out_buf packs n_used ids then
// n_used weight-bits per token (routing_stride = 2*n_used), so expert_ids[...+e]
// reads slot e's chosen expert id exactly as the single-token kernel did.

extern "C" __global__ void dmmv_q5k_experts_batched(const unsigned* a_u32, const float* x, float* y, const unsigned* expert_ids, ExpertsBatchPush pc) {
    unsigned t = blockIdx.y;
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = (expert_ids[(size_t)t * pc.routing_stride + e] * pc.slice + pc.base >> 2) + row * bpr * 44u;
    const float4* xv = (const float4*)(x + (size_t)t * pc.x_tok_stride + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u, ngrp = blockDim.x >> 4;
    unsigned sba = 2u*v_im, sbb = 2u*v_im+1u, sbc = 2u*v_im+4u, sbd = 2u*v_im+5u;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 44u, dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF)), dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk+1u], sc1 = a_u32[blk+2u], sc2 = a_u32[blk+3u];
        unsigned qh = a_u32[blk + 4u + (l0 >> 2)];
        unsigned qs0 = a_u32[blk + 12u + (q_off >> 2)], qs1 = a_u32[blk + 12u + (q_off >> 2) + 16u];
        unsigned bidx = (i*256u + y_loc) >> 2, bidx2 = (i*256u + y_loc + 128u) >> 2;
        float4 by0=xv[bidx], by1=xv[bidx+8u], by2=xv[bidx2], by3=xv[bidx2+8u];
        unsigned s0=sc0>>shift, s1=sc1>>shift, s2=sc2>>shift;
        float f0=d*(float)(s0&0x3Fu), b0=dm*(float)(s1&0x3Fu);
        float f1=d*(float)((s0>>8)&0x3Fu), b1=dm*(float)((s1>>8)&0x3Fu);
        float f2=d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2=dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
        float f3=d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3=dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
        #define Q5E(q,sh,sb,j) ((float)(((q)>>(sh))&0xFu) + 16.0f*(float)(((qh)>>((sb)+(j)*8u))&1u))
        sum += (f0*Q5E(qs0,0,sba,0)-b0)*by0.x + (f0*Q5E(qs0,8,sba,1)-b0)*by0.y + (f0*Q5E(qs0,16,sba,2)-b0)*by0.z + (f0*Q5E(qs0,24,sba,3)-b0)*by0.w;
        sum += (f1*Q5E(qs0,4,sbb,0)-b1)*by1.x + (f1*Q5E(qs0,12,sbb,1)-b1)*by1.y + (f1*Q5E(qs0,20,sbb,2)-b1)*by1.z + (f1*Q5E(qs0,28,sbb,3)-b1)*by1.w;
        sum += (f2*Q5E(qs1,0,sbc,0)-b2)*by2.x + (f2*Q5E(qs1,8,sbc,1)-b2)*by2.y + (f2*Q5E(qs1,16,sbc,2)-b2)*by2.z + (f2*Q5E(qs1,24,sbc,3)-b2)*by2.w;
        sum += (f3*Q5E(qs1,4,sbd,0)-b3)*by3.x + (f3*Q5E(qs1,12,sbd,1)-b3)*by3.y + (f3*Q5E(qs1,20,sbd,2)-b3)*by3.z + (f3*Q5E(qs1,28,sbd,3)-b3)*by3.w;
        #undef Q5E
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) y[(size_t)t * pc.y_tok_stride + (size_t)e * pc.M + row] = sum;
}

extern "C" __global__ void dmmv_q6k_experts_batched(const unsigned char* a, const float* x, float* y, const unsigned* expert_ids, ExpertsBatchPush pc) {
    unsigned t = blockIdx.y;
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + ((size_t)expert_ids[(size_t)t * pc.routing_stride + e] * pc.slice + pc.base) + (size_t)row * bpr * 210u;
    const float4* xv = (const float4*)(x + (size_t)t * pc.x_tok_stride + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, ix = tid >> 4;
    unsigned half_id = itid >> 3, local_id = itid & 7u, e_start = local_id * 4u, is = e_start >> 4;
    unsigned xvib = (half_id * 128u + e_start) >> 2, ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned b = ix; b < bpr; b += ngrp) {
        const unsigned char* bb = arow + (size_t)b * 210u;
        float d = zinc_half_to_float((unsigned short)((unsigned)bb[208] | ((unsigned)bb[209] << 8)));
        const unsigned char* ql = bb + half_id * 64u;
        const unsigned char* qh = bb + 128u + half_id * 32u;
        const signed char* sc = (const signed char*)(bb + 192u + half_id * 8u);
        float ds0 = d * (float)sc[is], ds2 = d * (float)sc[is + 2], ds4 = d * (float)sc[is + 4], ds6 = d * (float)sc[is + 6];
        unsigned xb = (b * 256u) / 4u + xvib;
        float4 bx0 = xv[xb], bx32 = xv[xb + 8u], bx64 = xv[xb + 16u], bx96 = xv[xb + 24u];
        #pragma unroll
        for (unsigned li = 0; li < 4u; li++) {
            unsigned l = e_start + li, qllo = ql[l], qlhi = ql[l + 32u], qhv = qh[l];
            float q1 = (float)((qllo & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32.0f;
            float q2 = (float)((qlhi & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32.0f;
            float q3 = (float)((qllo >> 4) | (((qhv >> 4) & 3u) << 4)) - 32.0f;
            float q4 = (float)((qlhi >> 4) | (((qhv >> 6) & 3u) << 4)) - 32.0f;
            sum += ds0*q1*(&bx0.x)[li] + ds2*q2*(&bx32.x)[li] + ds4*q3*(&bx64.x)[li] + ds6*q4*(&bx96.x)[li];
        }
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) y[(size_t)t * pc.y_tok_stride + (size_t)e * pc.M + row] = sum;
}

// Token-batched f32 matvec (qwen36-MoE prefill router logits + shared-expert gate
// scalar): y[t, row] = sum_k W[row,k] * x[t,k]. grid (M rows, T tokens). Byte-for-
// byte the single-token dmmv_f32's reduction (256 threads, zinc_block_reduce_sum).
struct MatvecBatchPush { unsigned M, K, x_tok_stride, y_tok_stride; };
extern "C" __global__ void dmmv_f32_batched(const float* w, const float* x, float* y, MatvecBatchPush pc) {
    unsigned t = blockIdx.y;
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const float* wrow = w + (size_t)row * pc.K;
    const float* xrow = x + (size_t)t * pc.x_tok_stride;
    float sum = 0.0f;
    for (unsigned k = threadIdx.x; k < pc.K; k += blockDim.x) sum += wrow[k] * xrow[k];
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) y[(size_t)t * pc.y_tok_stride + row] = sum;
}

// Token-batched twin of moe_weighted_acc (qwen36-MoE prefill routed combine; no
// per-expert down scale). a[t,i] += sum_j weight_j * b[t, j*src_stride + i].
extern "C" __global__ void moe_weighted_acc_batched(float* a, const float* b, const unsigned* routing, MoeAccBatchPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.N) return;
    unsigned t = blockIdx.y;
    float* at = a + (size_t)t * pc.a_tok_stride;
    const float* bt = b + (size_t)t * pc.b_tok_stride;
    const unsigned* rt = routing + (size_t)t * pc.routing_stride;
    float sum = 0.0f;
    for (unsigned j = 0; j < pc.n_used; j++) {
        float w = __uint_as_float(rt[pc.n_used + j]);
        sum += w * bt[(size_t)j * pc.src_stride + i];
    }
    at[i] += sum;
}

// Token-batched twin of sigmoid_scale_acc (qwen36-MoE prefill shared-expert
// gating): a[t,i] += sigmoid(c[t]) * b[t,i]. c is the per-token gate logit.
extern "C" __global__ void sigmoid_scale_acc_batched(float* a, const float* b, const float* c, MoeAccBatchPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.N) return;
    unsigned t = blockIdx.y;
    float gate = 1.0f / (1.0f + expf(-c[t]));
    a[(size_t)t * pc.a_tok_stride + i] += gate * b[(size_t)t * pc.b_tok_stride + i];
}

// ---- token-GROUPED routed-expert matvecs (gemma-26b MoE prefill) ------------
// Effort 24 cycle 18: the cycle-8 token-batched kernels above launch grid.y =
// token, so blocks reading the SAME expert weight are scattered across grid.y
// (token t's slot e routes to a different expert than token t+1's slot e) — no
// cross-token L2 reuse of the expert weight. These GROUPED kernels instead take
// a precomputed `order[]` (built by build_expert_order below) that lists the
// T*n_used (token,slot) work-items SORTED BY EXPERT ID, so consecutive grid.y
// work-items share the same expert weight → it stays resident in L2 across all
// the tokens routed to it (a real memory-traffic win beyond launch batching).
// BYTE-IDENTITY: the per-block dequant + zinc_block_reduce_sum + the y write
// location (token t's slice, slot e, row) are byte-for-byte the cycle-8 kernel's
// — only WHICH block computes which (token,slot,row) changes, and every output
// element is still computed exactly once, so GEN_IDS are identical regardless of
// the order permutation. order packs (token<<16 | slot); prefill T < 65536.
// Reuses ExpertsBatchPush (n_used field unused here — slot comes from order).
extern "C" __global__ void dmmv_q4k_experts_grouped(const unsigned* a_u32, const float* x, float* y, const unsigned* expert_ids, const unsigned* order, ExpertsBatchPush pc) {
    unsigned packed = order[blockIdx.y];
    unsigned t = packed >> 16, e = packed & 0xFFFFu;
    unsigned row = blockIdx.x;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = ((expert_ids[(size_t)t * pc.routing_stride + e] * pc.slice + pc.base) >> 2) + row * bpr * 36u;
    const float4* xv = (const float4*)(x + (size_t)t * pc.x_tok_stride + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 36u;
        unsigned dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
        float dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
        unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
        float f0 = d * (float)(s0 & 0x3Fu), b0 = dm * (float)(s1 & 0x3Fu);
        float f1 = d * (float)((s0 >> 8) & 0x3Fu), b1 = dm * (float)((s1 >> 8) & 0x3Fu);
        float f2 = d * (float)((s2 & 0xFu) | ((s0 & 0xC0u) >> 2)), b2 = dm * (float)(((s2 & 0xF0u) >> 4) | ((s1 & 0xC0u) >> 2));
        float f3 = d * (float)(((s2 >> 8) & 0xFu) | (((s0 >> 8) & 0xC0u) >> 2)), b3 = dm * (float)((((s2 >> 8) & 0xF0u) >> 4) | (((s1 >> 8) & 0xC0u) >> 2));
        sum += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
        sum += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
        sum += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
        sum += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) y[(size_t)t * pc.y_tok_stride + (size_t)e * pc.M + row] = sum;
}

extern "C" __global__ void dmmv_q5_1_experts_grouped(const unsigned char* a, const float* x, float* y, const unsigned* expert_ids, const unsigned* order, ExpertsBatchPush pc) {
    unsigned packed = order[blockIdx.y];
    unsigned t = packed >> 16, e = packed & 0xFFFFu;
    unsigned row = blockIdx.x;
    unsigned bpr = pc.K >> 5;
    const unsigned char* arow = a + (size_t)expert_ids[(size_t)t * pc.routing_stride + e] * pc.slice + pc.base + (size_t)row * bpr * 24u;
    const float* xrow = x + (size_t)t * pc.x_tok_stride + (size_t)e * pc.x_stride;
    float sum = 0.0f;
    for (unsigned el = threadIdx.x; el < pc.K; el += blockDim.x) {
        unsigned blk = el >> 5, i = el & 31u;
        const unsigned char* blkp = arow + (size_t)blk * 24u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blkp[0] | ((unsigned)blkp[1] << 8)));
        float m = zinc_half_to_float((unsigned short)((unsigned)blkp[2] | ((unsigned)blkp[3] << 8)));
        unsigned qh = (unsigned)blkp[4] | ((unsigned)blkp[5] << 8) | ((unsigned)blkp[6] << 16) | ((unsigned)blkp[7] << 24);
        const unsigned char* qs = blkp + 8;
        unsigned nib = (i < 16u) ? (unsigned)(qs[i] & 0xFu) : (unsigned)(qs[i - 16u] >> 4);
        unsigned q5 = nib | (((qh >> i) & 1u) << 4);
        sum += (d * (float)q5 + m) * xrow[el];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) y[(size_t)t * pc.y_tok_stride + (size_t)e * pc.M + row] = sum;
}

// build_expert_order — single-block counting sort of the T*n_used (token,slot)
// routed-expert work-items by expert id, producing `order[]` (packed token<<16|slot)
// for the grouped kernels above. Phases: zero shared counts → histogram of expert
// ids (shared atomics) → exclusive prefix-sum into a shared cursor → scatter each
// work-item into its expert's contiguous run. Intra-bin order is race-dependent but
// irrelevant: each (token,slot) maps to a distinct, independently-computed output.
// n_experts <= 256 (gemma-26b = 128). Async on the shared stream (reads the router
// table written by routerBatched; the grouped matvecs read order after this).
struct BuildOrderPush { unsigned T, n_used, n_experts, routing_stride; };
extern "C" __global__ void build_expert_order(const unsigned* expert_ids, unsigned* order, BuildOrderPush pc) {
    __shared__ unsigned counts[256];
    __shared__ unsigned cursor[256];
    unsigned tid = threadIdx.x, nthr = blockDim.x;
    unsigned P = pc.T * pc.n_used;
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        atomicAdd(&counts[expert_ids[(size_t)t * pc.routing_stride + slot]], 1u);
    }
    __syncthreads();
    if (tid == 0) { unsigned acc = 0u; for (unsigned e = 0; e < pc.n_experts; e++) { cursor[e] = acc; acc += counts[e]; } }
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        unsigned pos = atomicAdd(&cursor[expert_ids[(size_t)t * pc.routing_stride + slot]], 1u);
        order[pos] = (t << 16) | slot;
    }
}

// ---- T2 (grouped Tensor-core MoE-expert GEMM) primitives ---------------------
// The routed-expert FFN bottleneck is the scalar matvec compute (grouped matvec
// is dead on both GPUs — Effort-26 c7 / 2026-06-22 4090 A/B). T2 puts the gate/up
// experts on the Tensor cores by GATHERING each expert's tokens contiguously and
// running the existing `gemm_q4k_tc` per expert. These three kernels are the glue:
// an offsets-emitting counting sort + a gather + a scatter (the GEMM is reused).

// build_expert_order_off — build_expert_order + an exclusive prefix-sum `offsets`
// [n_experts+1] (offsets[e]..offsets[e+1] = expert e's contiguous run in order[]),
// so the host can launch one gemm_q4k_tc per expert over its token slice. order[]
// format is unchanged ((token<<16)|slot). Does not touch the existing
// build_expert_order used by the grouped matvecs.
extern "C" __global__ void build_expert_order_off(const unsigned* expert_ids, unsigned* order, unsigned* offsets, BuildOrderPush pc) {
    __shared__ unsigned counts[256];
    __shared__ unsigned cursor[256];
    unsigned tid = threadIdx.x, nthr = blockDim.x;
    unsigned P = pc.T * pc.n_used;
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        atomicAdd(&counts[expert_ids[(size_t)t * pc.routing_stride + slot]], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        unsigned acc = 0u;
        for (unsigned e = 0; e < pc.n_experts; e++) { offsets[e] = acc; cursor[e] = acc; acc += counts[e]; }
        offsets[pc.n_experts] = acc; // == P
    }
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        unsigned pos = atomicAdd(&cursor[expert_ids[(size_t)t * pc.routing_stride + slot]], 1u);
        order[pos] = (t << 16) | slot;
    }
}

// gather_by_order — A_grouped[pos*K + k] = src[t*src_tok_stride + k] where
// (t,_) = order[pos]. The gate/up experts all read the same per-token input row
// (matvec x_stride=0), so gathering by token gives the [P,K] A for gemm_q4k_tc.
struct GatherOrderPush { unsigned P, K, src_tok_stride; };
extern "C" __global__ void gather_by_order(const float* src, const unsigned* order, float* dst, GatherOrderPush pc) {
    unsigned pos = blockIdx.y;
    if (pos >= pc.P) return;
    const float* s = src + (size_t)(order[pos] >> 16) * pc.src_tok_stride;
    float* dd = dst + (size_t)pos * pc.K;
    for (unsigned k = blockIdx.x * blockDim.x + threadIdx.x; k < pc.K; k += gridDim.x * blockDim.x) dd[k] = s[k];
}

// scatter_by_order — dst[t*dst_tok_stride + slot*M + row] = Yg[pos*M + row], the
// inverse of gather: writes each grouped GEMM output row back to its (token,slot).
struct ScatterOrderPush { unsigned P, M, dst_tok_stride; };
extern "C" __global__ void scatter_by_order(const float* yg, const unsigned* order, float* dst, ScatterOrderPush pc) {
    unsigned pos = blockIdx.y;
    if (pos >= pc.P) return;
    unsigned packed = order[pos];
    const float* s = yg + (size_t)pos * pc.M;
    float* dd = dst + (size_t)(packed >> 16) * pc.dst_tok_stride + (size_t)(packed & 0xFFFFu) * pc.M;
    for (unsigned m = blockIdx.x * blockDim.x + threadIdx.x; m < pc.M; m += gridDim.x * blockDim.x) dd[m] = s[m];
}

// ---- T2 v1: SINGLE-LAUNCH grouped Tensor-core MoE-expert GEMM ----------------
// v0 (per-expert gemm_q4k_tc + host readback) was token-correct but launch-bound
// (~10k launches + 40 syncs -> -27%). v1 collapses it to ONE launch per gate/up:
// build_expert_order_padded pads each expert's (token,slot) run up to the 64-token
// tile boundary and tags each tile with its expert id; gemm_q4k_experts_grouped_tc
// runs the gemm_q4k_tc dequant+wmma core, picking the weight per TILE's expert and
// gathering A / scattering Y via the padded order — NO per-expert launch, NO host
// readback. fp16 -> token-tolerance gate (not bit-identical to the matvec).
struct BuildOrderPadPush { unsigned T, n_used, n_experts, routing_stride, max_pos; };
extern "C" __global__ void build_expert_order_padded(const unsigned* expert_ids, unsigned* order, unsigned* tile_expert, BuildOrderPadPush pc) {
    __shared__ unsigned counts[256];
    __shared__ unsigned poff[256]; // padded start position per expert
    const unsigned INV = 0xFFFFFFFFu;
    unsigned tid = threadIdx.x, nthr = blockDim.x;
    unsigned P = pc.T * pc.n_used;
    unsigned maxtiles = pc.max_pos >> 6;
    for (unsigned p = tid; p < pc.max_pos; p += nthr) order[p] = INV;
    for (unsigned tl = tid; tl < maxtiles; tl += nthr) tile_expert[tl] = INV;
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        atomicAdd(&counts[expert_ids[(size_t)t * pc.routing_stride + slot]], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        unsigned acc = 0u;
        for (unsigned e = 0; e < pc.n_experts; e++) {
            poff[e] = acc;
            unsigned ntile = (counts[e] + 63u) >> 6; // ceil(count/64)
            for (unsigned k = 0; k < ntile; k++) tile_expert[(acc >> 6) + k] = e;
            acc += ntile * 64u;
        }
    }
    __syncthreads();
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u; // reuse as cursor
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        unsigned E = expert_ids[(size_t)t * pc.routing_stride + slot];
        unsigned within = atomicAdd(&counts[E], 1u);
        order[poff[E] + within] = (t << 16) | slot;
    }
}

// gfx12 short-prompt twin: pad each expert run to 16 routes instead of 64.
// The integer-WMMA expert kernel consumes one 16-column fragment per tile, so
// this avoids executing 64 columns when an expert received only a few tokens.
extern "C" __global__ void build_expert_order_padded_16(const unsigned* expert_ids, unsigned* order, unsigned* tile_expert, BuildOrderPadPush pc) {
    __shared__ unsigned counts[256];
    __shared__ unsigned poff[256];
    const unsigned INV = 0xFFFFFFFFu;
    unsigned tid = threadIdx.x, nthr = blockDim.x;
    unsigned P = pc.T * pc.n_used;
    unsigned maxtiles = pc.max_pos >> 4;
    for (unsigned p = tid; p < pc.max_pos; p += nthr) order[p] = INV;
    for (unsigned tl = tid; tl < maxtiles; tl += nthr) tile_expert[tl] = INV;
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        atomicAdd(&counts[expert_ids[(size_t)t * pc.routing_stride + slot]], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        unsigned acc = 0u;
        for (unsigned e = 0; e < pc.n_experts; e++) {
            poff[e] = acc;
            unsigned ntile = (counts[e] + 15u) >> 4;
            for (unsigned k = 0; k < ntile; k++) tile_expert[(acc >> 4) + k] = e;
            acc += ntile * 16u;
        }
    }
    __syncthreads();
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        unsigned E = expert_ids[(size_t)t * pc.routing_stride + slot];
        unsigned within = atomicAdd(&counts[E], 1u);
        order[poff[E] + within] = (t << 16) | slot;
    }
}

// Four-route padded ordering for the ROCm exact-Q8 Q5_1 down projection.
extern "C" __global__ void build_expert_order_padded_4(const unsigned* expert_ids, unsigned* order, unsigned* tile_expert, BuildOrderPadPush pc) {
    __shared__ unsigned counts[256];
    __shared__ unsigned poff[256];
    const unsigned INV = 0xFFFFFFFFu;
    const unsigned tid = threadIdx.x, nthr = blockDim.x;
    const unsigned P = pc.T * pc.n_used;
    const unsigned maxtiles = pc.max_pos >> 2;
    for (unsigned p = tid; p < pc.max_pos; p += nthr) order[p] = INV;
    for (unsigned tl = tid; tl < maxtiles; tl += nthr) tile_expert[tl] = INV;
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        const unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        atomicAdd(&counts[expert_ids[(size_t)t * pc.routing_stride + slot]], 1u);
    }
    __syncthreads();
    if (tid == 0u) {
        unsigned acc = 0u;
        for (unsigned e = 0u; e < pc.n_experts; ++e) {
            poff[e] = acc;
            const unsigned ntile = (counts[e] + 3u) >> 2;
            for (unsigned k = 0u; k < ntile; ++k) tile_expert[(acc >> 2) + k] = e;
            acc += ntile * 4u;
        }
    }
    __syncthreads();
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        const unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        const unsigned e = expert_ids[(size_t)t * pc.routing_stride + slot];
        const unsigned within = atomicAdd(&counts[e], 1u);
        order[poff[e] + within] = (t << 16) | slot;
    }
}

// Eight-route ordering A/B for the Q5_1 down tile.
extern "C" __global__ void build_expert_order_padded_8(const unsigned* expert_ids, unsigned* order, unsigned* tile_expert, BuildOrderPadPush pc) {
    __shared__ unsigned counts[256];
    __shared__ unsigned poff[256];
    const unsigned INV = 0xFFFFFFFFu;
    const unsigned tid = threadIdx.x, nthr = blockDim.x;
    const unsigned P = pc.T * pc.n_used;
    const unsigned maxtiles = pc.max_pos >> 3;
    for (unsigned p = tid; p < pc.max_pos; p += nthr) order[p] = INV;
    for (unsigned tl = tid; tl < maxtiles; tl += nthr) tile_expert[tl] = INV;
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        const unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        atomicAdd(&counts[expert_ids[(size_t)t * pc.routing_stride + slot]], 1u);
    }
    __syncthreads();
    if (tid == 0u) {
        unsigned acc = 0u;
        for (unsigned e = 0u; e < pc.n_experts; ++e) {
            poff[e] = acc;
            const unsigned ntile = (counts[e] + 7u) >> 3;
            for (unsigned k = 0u; k < ntile; ++k) tile_expert[(acc >> 3) + k] = e;
            acc += ntile * 8u;
        }
    }
    __syncthreads();
    for (unsigned e = tid; e < pc.n_experts; e += nthr) counts[e] = 0u;
    __syncthreads();
    for (unsigned p = tid; p < P; p += nthr) {
        const unsigned t = p / pc.n_used, slot = p - t * pc.n_used;
        const unsigned e = expert_ids[(size_t)t * pc.routing_stride + slot];
        const unsigned within = atomicAdd(&counts[e], 1u);
        order[poff[e] + within] = (t << 16) | slot;
    }
}

#ifndef ZINC_ROCM
struct GroupedTCPush { unsigned M, K, base, gu_full, dst_tok_stride; };
extern "C" __global__ void gemm_q4k_experts_grouped_tc(const unsigned* a_u32, const float* A, const unsigned* order, const unsigned* tile_expert, float* dst, GroupedTCPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u, INV=0xFFFFFFFFu;
    __shared__ half Ws[BM*BK];
    __shared__ half As[BK*BT];
    __shared__ float Cs[BT*BM];
    unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return; // unused padded tile
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;
    unsigned nchunk = pc.K >> 5;
    unsigned tid = threadIdx.x;
    unsigned a0 = (unsigned)(((size_t)expert * pc.gu_full + pc.base) >> 2);
    unsigned warp = tid >> 5, fm = warp >> 2, ft = warp & 3u;

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u;
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned r = idx >> 5, l = idx & 31u;
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned blk = a0 + row * bpr * 36u + sbk * 36u;
                unsigned dd = a_u32[blk];
                float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
                float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
                const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
                const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
                unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
                unsigned char qb = qs[(sb8 >> 1) * 32u + l];
                unsigned nib = (sb8 & 1u) == 0u ? (qb & 0xFu) : (unsigned)(qb >> 4);
                wv = d * (float)sc * (float)nib - dmin * (float)mn;
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned packed = order[t0 + t];
            As[l * BT + t] = (packed != INV) ? __float2half(A[(size_t)(packed >> 16) * pc.K + c * 32u + l]) : __float2half(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[(fm * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fm + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + fm * 16u], c0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fm + 2u) * 16u], c1, BM, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 16; u++) {
        unsigned idx = tid + (unsigned)u * 256u;
        unsigned t = idx >> 6, m = idx & 63u;
        unsigned row = m0 + m;
        if (row < pc.M) {
            unsigned packed = order[t0 + t];
            if (packed != INV) dst[(size_t)(packed >> 16) * pc.dst_tok_stride + (size_t)(packed & 0xFFFFu) * pc.M + row] = Cs[t * BM + m];
        }
    }
}

// ---- T2 v1 (down): SINGLE-LAUNCH grouped Tensor-core Q5_1 MoE-down GEMM -------
// Twin of gemm_q4k_experts_grouped_tc for the routed-expert DOWN projection,
// whose weight is Q5_1 (gemma ffn_down_exps). Reuses the SAME padded order +
// per-tile expert id (b.padded_order / b.tile_expert) the gate/up GEMM built, so
// no extra sort. Two differences from the Q4_K gate/up kernel:
//   (1) weight dequant is Q5_1 — each 32-wide GEMM K-chunk maps to exactly one
//       Q5_1 block (24 bytes: d|m halfs, 32-bit qh, 16 nibble bytes); value =
//       d*(nib | high-bit) + m, matching dmmv_q5_1_experts_batched bit-for-bit.
//   (2) A is the GeGLU output [P, ef] indexed by WORK-ITEM (token*n_used+slot),
//       not by token — so the A-gather row uses both halves of the packed order
//       entry (the gate/up A was the shared per-token moe_norm row, x_stride=0).
// dst scatter matches gate/up (dst_tok_stride = n_used*n_embd, per-slot *M).
// fp16 -> token-tolerance gate, not bit-identical to the matvec.
struct GroupedTCDownPush { unsigned M, K, slice, n_used, dst_tok_stride; };
extern "C" __global__ void gemm_q5_1_experts_grouped_tc(const unsigned char* a, const float* A, const unsigned* order, const unsigned* tile_expert, float* dst, GroupedTCDownPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u, INV=0xFFFFFFFFu;
    __shared__ half Ws[BM*BK];
    __shared__ half As[BK*BT];
    __shared__ float Cs[BT*BM];
    unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return; // unused padded tile
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 5;     // Q5_1 blocks per weight row (== nchunk)
    unsigned nchunk = pc.K >> 5;
    unsigned tid = threadIdx.x;
    const unsigned char* a_e = a + (size_t)expert * pc.slice; // this tile's expert weight
    unsigned warp = tid >> 5, fm = warp >> 2, ft = warp & 3u;

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned r = idx >> 5, l = idx & 31u;
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                const unsigned char* blkp = a_e + (size_t)row * bpr * 24u + (size_t)c * 24u;
                float d = zinc_half_to_float((unsigned short)((unsigned)blkp[0] | ((unsigned)blkp[1] << 8)));
                float m = zinc_half_to_float((unsigned short)((unsigned)blkp[2] | ((unsigned)blkp[3] << 8)));
                unsigned qh = (unsigned)blkp[4] | ((unsigned)blkp[5] << 8) | ((unsigned)blkp[6] << 16) | ((unsigned)blkp[7] << 24);
                const unsigned char* qs = blkp + 8;
                unsigned nib = (l < 16u) ? (unsigned)(qs[l] & 0xFu) : (unsigned)(qs[l - 16u] >> 4);
                unsigned q5 = nib | (((qh >> l) & 1u) << 4);
                wv = d * (float)q5 + m;
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned packed = order[t0 + t];
            // A row = work-item (token*n_used + slot) into the [P, K] GeGLU buffer.
            size_t arow = (size_t)(packed >> 16) * pc.n_used + (size_t)(packed & 0xFFFFu);
            As[l * BT + t] = (packed != INV) ? __float2half(A[arow * pc.K + c * 32u + l]) : __float2half(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[(fm * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fm + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + fm * 16u], c0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fm + 2u) * 16u], c1, BM, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 16; u++) {
        unsigned idx = tid + (unsigned)u * 256u;
        unsigned t = idx >> 6, m = idx & 63u;
        unsigned row = m0 + m;
        if (row < pc.M) {
            unsigned packed = order[t0 + t];
            if (packed != INV) dst[(size_t)(packed >> 16) * pc.dst_tok_stride + (size_t)(packed & 0xFFFFu) * pc.M + row] = Cs[t * BM + m];
        }
    }
}

// ---- T2 qwen MoE-down: grouped TC Q5_K + Q6_K down kernels (harvested; reuse GroupedTCDownPush above) ----
extern "C" __global__ void gemm_q5k_experts_grouped_tc(const unsigned char* a, const float* A, const unsigned* order, const unsigned* tile_expert, float* dst, GroupedTCDownPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u, INV=0xFFFFFFFFu;
    __shared__ half Ws[BM*BK];
    __shared__ half As[BK*BT];
    __shared__ float Cs[BT*BM];
    unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return; // unused padded tile
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;     // Q5_K super-blocks per weight row
    unsigned nchunk = pc.K >> 5;  // 32-chunks per row (== 8 sub-blocks/super-block)
    unsigned tid = threadIdx.x;
    const unsigned char* a_e = a + (size_t)expert * pc.slice; // this tile's expert weight
    unsigned warp = tid >> 5, fm = warp >> 2, ft = warp & 3u;

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u; // super-block, sub-block (0..7)
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned r = idx >> 5, l = idx & 31u;
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                const unsigned char* blk = a_e + (size_t)row * bpr * 176u + (size_t)sbk * 176u;
                float d = zinc_half_to_float((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
                float dmin = zinc_half_to_float((unsigned short)((unsigned)blk[2] | ((unsigned)blk[3] << 8)));
                const unsigned char* scales = blk + 4;  // 12 bytes (6-bit packed)
                const unsigned char* qh = blk + 16;     // 32 bytes (1 high bit/elem)
                const unsigned char* qs = blk + 48;     // 128 bytes (4-bit low)
                unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
                unsigned char ql = qs[(sb8 >> 1) * 32u + l];
                unsigned nib = (sb8 & 1u) == 0u ? (ql & 0xFu) : (unsigned)(ql >> 4);
                unsigned bit = (qh[l] >> sb8) & 1u;
                unsigned q5 = nib + (bit ? 16u : 0u);
                wv = d * (float)sc * (float)q5 - dmin * (float)mn;
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned packed = order[t0 + t];
            // A row = work-item (token*n_used + slot) into the [P, K] SwiGLU buffer.
            size_t arow = (size_t)(packed >> 16) * pc.n_used + (size_t)(packed & 0xFFFFu);
            As[l * BT + t] = (packed != INV) ? __float2half(A[arow * pc.K + c * 32u + l]) : __float2half(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[(fm * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fm + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + fm * 16u], c0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fm + 2u) * 16u], c1, BM, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 16; u++) {
        unsigned idx = tid + (unsigned)u * 256u;
        unsigned t = idx >> 6, m = idx & 63u;
        unsigned row = m0 + m;
        if (row < pc.M) {
            unsigned packed = order[t0 + t];
            if (packed != INV) dst[(size_t)(packed >> 16) * pc.dst_tok_stride + (size_t)(packed & 0xFFFFu) * pc.M + row] = Cs[t * BM + m];
        }
    }
}

// gemm_q6k_experts_grouped_tc — the DOWN twin of gemm_q5k_experts_grouped_tc for
// the 4/40 qwen36-35b-a3b down layers whose weight is Q6_K (the other 36 are
// Q5_K). Same padded order + tile_expert + wmma core + per-work-item A (the
// SwiGLU output indexed (token*n_used + slot)); ONLY the weight dequant differs.
// Q6_K super-block = 210 bytes / 256 elems: ql[0..127] low 4 bits, qh[128..191]
// high 2 bits, scales[192..207] (16 int8), d[208..209] (f16); value =
// d*int8_scale*(q-32), q = (ql_nibble | qh_bits<<4). The 32-element chunk c maps
// to (super-block c>>3, half (c&7)>>2, group (c&7)&3) — dequant copied from
// dmmv_q6k. fp16 → token-tolerance, not bit-identical to the matvec.
extern "C" __global__ void gemm_q6k_experts_grouped_tc(const unsigned char* a, const float* A, const unsigned* order, const unsigned* tile_expert, float* dst, GroupedTCDownPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u, INV=0xFFFFFFFFu;
    __shared__ half Ws[BM*BK];
    __shared__ half As[BK*BT];
    __shared__ float Cs[BT*BM];
    unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return; // unused padded tile
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;     // Q6_K super-blocks per weight row
    unsigned nchunk = pc.K >> 5;  // 32-chunks per row (8 chunks / super-block)
    unsigned tid = threadIdx.x;
    const unsigned char* a_e = a + (size_t)expert * pc.slice; // this tile's expert weight
    unsigned warp = tid >> 5, fm = warp >> 2, ft = warp & 3u;

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3;       // super-block
        unsigned cw = c & 7u;        // chunk within super-block 0..7
        unsigned qhalf = cw >> 2;    // 0..1 (128-elem half; not the `half` type)
        unsigned group = cw & 3u;    // 0..3 (q1..q4)
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned r = idx >> 5, l = idx & 31u;
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                const unsigned char* blk = a_e + (size_t)row * bpr * 210u + (size_t)sbk * 210u;
                float d = zinc_half_to_float((unsigned short)((unsigned)blk[208] | ((unsigned)blk[209] << 8)));
                const unsigned char* ql = blk + (size_t)qhalf * 64u;
                const unsigned char* qh = blk + 128u + (size_t)qhalf * 32u;
                const signed char* sc = (const signed char*)(blk + 192u + (size_t)qhalf * 8u);
                unsigned is = l >> 4;
                unsigned qhb = qh[l];
                unsigned q; unsigned sci;
                if (group == 0u) { q = (ql[l] & 0xFu) | (((qhb >> 0) & 3u) << 4); sci = is + 0u; }
                else if (group == 1u) { q = (ql[l + 32u] & 0xFu) | (((qhb >> 2) & 3u) << 4); sci = is + 2u; }
                else if (group == 2u) { q = (ql[l] >> 4) | (((qhb >> 4) & 3u) << 4); sci = is + 4u; }
                else { q = (ql[l + 32u] >> 4) | (((qhb >> 6) & 3u) << 4); sci = is + 6u; }
                wv = d * (float)sc[sci] * ((float)q - 32.0f);
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned packed = order[t0 + t];
            size_t arow = (size_t)(packed >> 16) * pc.n_used + (size_t)(packed & 0xFFFFu);
            As[l * BT + t] = (packed != INV) ? __float2half(A[arow * pc.K + c * 32u + l]) : __float2half(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[(fm * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fm + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + fm * 16u], c0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fm + 2u) * 16u], c1, BM, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 16; u++) {
        unsigned idx = tid + (unsigned)u * 256u;
        unsigned t = idx >> 6, m = idx & 63u;
        unsigned row = m0 + m;
        if (row < pc.M) {
            unsigned packed = order[t0 + t];
            if (packed != INV) dst[(size_t)(packed >> 16) * pc.dst_tok_stride + (size_t)(packed & 0xFFFFu) * pc.M + row] = Cs[t * BM + m];
        }
    }
}


#endif // !ZINC_ROCM: NVIDIA WMMA grouped-expert kernels

// ---- dmmv_q8_0_fast — whole-block-per-thread, d once, float4 x --------------
extern "C" __global__ void dmmv_q8_0_fast(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x; if (row >= pc.M) return;
    unsigned tok = blockIdx.y;
    unsigned bpr = pc.K >> 5;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 34u;
    unsigned xb0 = (pc.x_offset >> 2) + (size_t)tok * pc.K, tid = threadIdx.x;
    float sum = 0.0f;
    for (unsigned b = tid; b < bpr; b += blockDim.x) {
        const unsigned char* blk = arow + (size_t)b * 34u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
        const signed char* qs = (const signed char*)(blk + 2);
        const float4* xb = (const float4*)(x + xb0 + (size_t)b * 32u);
        #pragma unroll
        for (unsigned j = 0; j < 8u; j++) { float4 xx = xb[j]; sum += d * ((float)qs[j*4]*xx.x + (float)qs[j*4+1]*xx.y + (float)qs[j*4+2]*xx.z + (float)qs[j*4+3]*xx.w); }
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) { unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row; if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum; }
}

// Fuse two or three independent Q8_0 projections that consume the same input.
// The grid is only concatenated; every output row retains dmmv_q8_0_fast's
// thread mapping and accumulation order.
extern "C" __global__ void dmmv_q8_0_fast_multi(
    const unsigned char* a0, const unsigned char* a1, const unsigned char* a2,
    const float* x, float* y0, float* y1, float* y2, Dmmv3Push pc)
{
    const unsigned bx = blockIdx.x;
    if (bx >= pc.M0 + pc.M1 + pc.M2) return;
    const unsigned char* a;
    float* y;
    unsigned row;
    if (bx < pc.M0) {
        a = a0; y = y0; row = bx;
    } else if (bx < pc.M0 + pc.M1) {
        a = a1; y = y1; row = bx - pc.M0;
    } else {
        a = a2; y = y2; row = bx - pc.M0 - pc.M1;
    }
    const unsigned bpr = pc.K >> 5;
    const unsigned char* arow = a + (size_t)row * bpr * 34u;
    const unsigned tid = threadIdx.x;
    float sum = 0.0f;
    for (unsigned b = tid; b < bpr; b += blockDim.x) {
        const unsigned char* blk = arow + (size_t)b * 34u;
        const float d = zinc_half_to_float(*(const unsigned short*)blk);
        const signed char* qs = (const signed char*)(blk + 2u);
        const float4* xb = (const float4*)(x + (size_t)b * 32u);
        #pragma unroll
        for (unsigned j = 0; j < 8u; ++j) {
            const float4 xx = xb[j];
            sum += d * ((float)qs[j * 4u] * xx.x + (float)qs[j * 4u + 1u] * xx.y + (float)qs[j * 4u + 2u] * xx.z + (float)qs[j * 4u + 3u] * xx.w);
        }
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0u) y[row] = sum;
}

// Two adjacent Q8_0 output rows per block. Both rows reuse the same activation
// loads and share one reduction rendezvous; their per-row accumulation order is
// identical to dmmv_q8_0_fast.
extern "C" __global__ void dmmv_q8_0_mrow2(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    const unsigned row0 = blockIdx.x * 2u;
    if (row0 >= pc.M) return;
    const unsigned row1 = row0 + 1u;
    const unsigned bpr = pc.K >> 5;
    const unsigned char* arow0 = a + pc.a_offset + (size_t)row0 * bpr * 34u;
    const unsigned char* arow1 = a + pc.a_offset + (size_t)row1 * bpr * 34u;
    const unsigned xb0 = pc.x_offset >> 2;
    const unsigned tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (unsigned b = tid; b < bpr; b += blockDim.x) {
        const float4* xb = (const float4*)(x + xb0 + (size_t)b * 32u);
        const unsigned char* blk0 = arow0 + (size_t)b * 34u;
        const float d0 = zinc_half_to_float(*(const unsigned short*)blk0);
        const signed char* qs0 = (const signed char*)(blk0 + 2u);
        const bool have1 = row1 < pc.M;
        const unsigned char* blk1 = arow1 + (size_t)b * 34u;
        const float d1 = have1 ? zinc_half_to_float(*(const unsigned short*)blk1) : 0.0f;
        const signed char* qs1 = (const signed char*)(blk1 + 2u);
        #pragma unroll
        for (unsigned j = 0; j < 8u; ++j) {
            const float4 xx = xb[j];
            sum0 += d0 * ((float)qs0[j * 4u] * xx.x + (float)qs0[j * 4u + 1u] * xx.y + (float)qs0[j * 4u + 2u] * xx.z + (float)qs0[j * 4u + 3u] * xx.w);
            if (have1) sum1 += d1 * ((float)qs1[j * 4u] * xx.x + (float)qs1[j * 4u + 1u] * xx.y + (float)qs1[j * 4u + 2u] * xx.z + (float)qs1[j * 4u + 3u] * xx.w);
        }
    }
    zinc_block_reduce_sum_pair(sum0, sum1);
    if (tid == 0u) {
        const unsigned yi = pc.y_offset >> 2;
        if (pc.acc_mode != 0u) {
            y[yi + row0] += sum0;
            if (row1 < pc.M) y[yi + row1] += sum1;
        } else {
            y[yi + row0] = sum0;
            if (row1 < pc.M) y[yi + row1] = sum1;
        }
    }
}

// ---- dmmv_q4k multi-row (perf research, agenda 1: small/mid-M occupancy) -----
// One block computes R output rows, loading each shared x-superblock ONCE and
// reusing it across all R rows -> amortizes x-loads, block launches, and the
// per-thread index math over R rows. Targets M=4096 (proj) / M=12288 (FFN)
// which are launch/occupancy-bound at 15-65% peak in the single-row *_fast.
// Same Q4_K dequant math as dmmv_q4k_fast (validated 1.96e-5 vs naive).
template<int R>
__device__ __forceinline__ void dmmv_q4k_mrow_impl(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row0 = blockIdx.x * (unsigned)R;
    if (row0 >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2));
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    unsigned a0 = (pc.a_offset >> 2);
    float sum[R];
    #pragma unroll
    for (int r = 0; r < R; r++) sum[r] = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            unsigned row = row0 + (unsigned)r;
            if (row >= pc.M) continue;
            unsigned blk = a0 + row * bpr * 36u + i * 36u;
            unsigned dd = a_u32[blk];
            float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
            float dm = zinc_half_to_float((unsigned short)(dd >> 16));
            unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
            unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
            unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
            float f0 = d*(float)(s0&0x3Fu), b0 = dm*(float)(s1&0x3Fu);
            float f1 = d*(float)((s0>>8)&0x3Fu), b1 = dm*(float)((s1>>8)&0x3Fu);
            float f2 = d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2 = dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
            float f3 = d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3 = dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
            float s = 0.0f;
            s += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
            s += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
            s += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
            s += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
            sum[r] += s;
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float t = zinc_block_reduce_sum(sum[r]);
        if (tid == 0) {
            unsigned row = row0 + (unsigned)r;
            if (row < pc.M) { unsigned yi = (pc.y_offset >> 2) + row; if (pc.acc_mode != 0u) y[yi] += t; else y[yi] = t; }
        }
        __syncthreads();
    }
}
extern "C" __global__ void dmmv_q4k_mr2(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_mrow_impl<2>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_mr4(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_mrow_impl<4>(a_u32, x, y, pc); }

// ---- dmmv_q4k_btok (Effort 28) — token-BATCH matvec: read ONE weight row once,
// reuse its dequant across B token x-vectors -> B outputs. The TRANSPOSE of
// dmmv_q4k_mrow (which reads one x across R weight rows). Targets batched DECODE
// at small B (2..8): the 64x64 gemm_q4k_tiled_v2 wastes 56-62/64 row-slots and
// goes COMPUTE-bound on tile padding (head-to-head: B=8 100% util, slow). This
// reads each Q4_K weight row ONCE (the dominant decode traffic) and amortizes the
// dequant across the B tokens -> bandwidth-bound, no tile waste. x token-major
// [B,K], y token-major [B,M]. Same Q4_K dequant arithmetic as dmmv_q4k_fast
// (bit-exact); per-token reduction matches dmmv_q4k_fast so btok at row b ==
// dmmv_q4k_fast on x[b] (the B==1 path) bit-for-bit -> ARGMAX-identical to gemm.
template<int B>
__device__ __forceinline__ void dmmv_q4k_btok_impl(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    unsigned a0 = (pc.a_offset >> 2), xo = (pc.x_offset >> 2);
    float sum[B];
    #pragma unroll
    for (int b = 0; b < B; b++) sum[b] = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        // weight superblock for `row` — read + dequant ONCE, reuse across B tokens.
        unsigned blk = a0 + row * bpr * 36u + i * 36u;
        unsigned dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
        float dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
        unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
        unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
        float f0 = d*(float)(s0&0x3Fu), b0 = dm*(float)(s1&0x3Fu);
        float f1 = d*(float)((s0>>8)&0x3Fu), b1 = dm*(float)((s1>>8)&0x3Fu);
        float f2 = d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2 = dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
        float f3 = d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3 = dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        #pragma unroll
        for (int b = 0; b < B; b++) {
            const float4* xv = (const float4*)(x + (unsigned)b * pc.K + xo);
            float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
            float s = 0.0f;
            s += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
            s += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
            s += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
            s += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
            sum[b] += s;
        }
    }
    #pragma unroll
    for (int b = 0; b < B; b++) {
        float t = zinc_block_reduce_sum(sum[b]);
        if (tid == 0) {
            unsigned yi = (pc.y_offset >> 2) + (unsigned)b * pc.M + row;
            if (pc.acc_mode != 0u) y[yi] += t; else y[yi] = t;
        }
        __syncthreads();
    }
}
extern "C" __global__ void dmmv_q4k_btok2(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<2>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok3(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<3>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok4(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<4>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok5(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<5>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok6(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<6>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok7(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<7>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok8(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<8>(a_u32, x, y, pc); }
// Effort 28: higher-B (9..16) — btok stays bandwidth-bound (one weight read
// amortized over B tokens) up to the Q4_K roofline crossover (~B≈27), so it
// keeps beating the 64×64 tile GEMM's padded compute in the B>8 serving regime.
extern "C" __global__ void dmmv_q4k_btok9(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<9>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok10(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<10>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok11(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<11>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok12(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<12>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok13(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<13>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok14(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<14>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok15(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<15>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok16(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<16>(a_u32, x, y, pc); }
// Effort 28: B=17..24 — extend btok up to (just below) the ~B≈27 Q4_K roofline
// crossover so the B=17..24 serving regime keeps the bandwidth-bound matvec
// instead of falling back to the padded 64×64 tile GEMM.
extern "C" __global__ void dmmv_q4k_btok17(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<17>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok18(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<18>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok19(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<19>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok20(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<20>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok21(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<21>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok22(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<22>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok23(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<23>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok24(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<24>(a_u32, x, y, pc); }
// Effort 28: B=25..27 — pin the upper edge of the ~B≈27 Q4_K roofline crossover
// empirically. `sum[B]` register pressure grows here, so whether btok still beats
// the 64×64 tile GEMM at B=25..27 is decided by a clean in-process BTOK_TIMING A/B.
extern "C" __global__ void dmmv_q4k_btok25(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<25>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok26(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<26>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_btok27(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_btok_impl<27>(a_u32, x, y, pc); }

// ---- dmmv_q6k_btok (Effort 28) — Q6_K token-BATCH matvec ---------------------
// The Q6_K analog of dmmv_q4k_btok: read each Q6_K weight row's superblock + its
// dequant ONCE, reuse across B token x-vectors. Targets gemma-31b's ffn_down
// (Q6_K) decode GEMM, which otherwise hits the tile-padding gemm_q6k_tiled_v2 at
// small B. Same per-element Q6_K dequant as dmmv_q6k_fast; per-token local sum
// then sum[t] += s (the proven dmmv_q4k_btok reduction pattern) → token-identical
// to dmmv_q6k_fast on x[t]. x token-major [B,K], y token-major [B,M].
template<int B>
__device__ __forceinline__ void dmmv_q6k_btok_impl(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 210u;
    unsigned tid = threadIdx.x, itid = tid & 15u, ix = tid >> 4;
    unsigned half_id = itid >> 3, local_id = itid & 7u, e_start = local_id * 4u, is = e_start >> 4;
    unsigned xvib = (half_id * 128u + e_start) >> 2, ngrp = blockDim.x >> 4;
    unsigned xo = (pc.x_offset >> 2);
    float sum[B];
    #pragma unroll
    for (int t = 0; t < B; t++) sum[t] = 0.0f;
    for (unsigned bi = ix; bi < bpr; bi += ngrp) {
        const unsigned char* bb = arow + (size_t)bi * 210u;
        float d = zinc_half_to_float((unsigned short)((unsigned)bb[208] | ((unsigned)bb[209] << 8)));
        const unsigned char* ql = bb + half_id * 64u;
        const unsigned char* qh = bb + 128u + half_id * 32u;
        const signed char* sc = (const signed char*)(bb + 192u + half_id * 8u);
        float ds0 = d * (float)sc[is], ds2 = d * (float)sc[is + 2], ds4 = d * (float)sc[is + 4], ds6 = d * (float)sc[is + 6];
        // dequant the 4 quads ONCE (weight-only) — reused across all B tokens.
        float q1[4], q2[4], q3[4], q4[4];
        #pragma unroll
        for (unsigned li = 0; li < 4u; li++) {
            unsigned l = e_start + li, qllo = ql[l], qlhi = ql[l + 32u], qhv = qh[l];
            q1[li] = (float)((qllo & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32.0f;
            q2[li] = (float)((qlhi & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32.0f;
            q3[li] = (float)((qllo >> 4) | (((qhv >> 4) & 3u) << 4)) - 32.0f;
            q4[li] = (float)((qlhi >> 4) | (((qhv >> 6) & 3u) << 4)) - 32.0f;
        }
        unsigned xb = (bi * 256u) / 4u + xvib;
        #pragma unroll
        for (int t = 0; t < B; t++) {
            const float4* xv = (const float4*)(x + (unsigned)t * pc.K + xo);
            float4 bx0 = xv[xb], bx32 = xv[xb + 8u], bx64 = xv[xb + 16u], bx96 = xv[xb + 24u];
            float s = 0.0f;
            #pragma unroll
            for (unsigned li = 0; li < 4u; li++)
                s += ds0 * q1[li] * (&bx0.x)[li] + ds2 * q2[li] * (&bx32.x)[li] + ds4 * q3[li] * (&bx64.x)[li] + ds6 * q4[li] * (&bx96.x)[li];
            sum[t] += s;
        }
    }
    #pragma unroll
    for (int t = 0; t < B; t++) {
        float r = zinc_block_reduce_sum(sum[t]);
        if (tid == 0) {
            unsigned yi = (pc.y_offset >> 2) + (unsigned)t * pc.M + row;
            if (pc.acc_mode != 0u) y[yi] += r; else y[yi] = r;
        }
        __syncthreads();
    }
}
extern "C" __global__ void dmmv_q6k_btok2(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<2>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok3(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<3>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok4(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<4>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok5(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<5>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok6(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<6>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok7(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<7>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok8(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<8>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok9(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<9>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok10(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<10>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok11(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<11>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok12(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<12>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok13(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<13>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok14(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<14>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok15(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<15>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok16(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<16>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok17(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<17>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok18(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<18>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok19(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<19>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok20(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<20>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok21(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<21>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok22(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<22>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok23(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<23>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok24(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<24>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok25(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<25>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok26(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<26>(a, x, y, pc); }
extern "C" __global__ void dmmv_q6k_btok27(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q6k_btok_impl<27>(a, x, y, pc); }

// ---- dmmv_q5k_btok (Effort 28) — Q5_K token-BATCH matvec ---------------------
// The Q5_K analog: read each Q5_K weight row's superblock + dequant ONCE, reuse
// across B tokens. Same per-element Q5_K dequant (qh 5th-bit promote) as
// dmmv_q5k_fast; per-token local sum then sum[t] += s → token-identical to
// dmmv_q5k_fast on x[t]. x token-major [B,K], y token-major [B,M].
template<int B>
__device__ __forceinline__ void dmmv_q5k_btok_impl(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = (pc.a_offset >> 2) + row * bpr * 44u;
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u, ngrp = blockDim.x >> 4;
    unsigned sba = 2u*v_im, sbb = 2u*v_im+1u, sbc = 2u*v_im+4u, sbd = 2u*v_im+5u;
    unsigned xo = (pc.x_offset >> 2);
    float sum[B];
    #pragma unroll
    for (int t = 0; t < B; t++) sum[t] = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 44u, dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF)), dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk+1u], sc1 = a_u32[blk+2u], sc2 = a_u32[blk+3u];
        unsigned qh = a_u32[blk + 4u + (l0 >> 2)];
        unsigned qs0 = a_u32[blk + 12u + (q_off >> 2)], qs1 = a_u32[blk + 12u + (q_off >> 2) + 16u];
        unsigned s0=sc0>>shift, s1=sc1>>shift, s2=sc2>>shift;
        float f0=d*(float)(s0&0x3Fu), b0=dm*(float)(s1&0x3Fu);
        float f1=d*(float)((s0>>8)&0x3Fu), b1=dm*(float)((s1>>8)&0x3Fu);
        float f2=d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2=dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
        float f3=d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3=dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
        // dequant the 16 nibbles ONCE (weight-only, qh 5th-bit promote) — reused
        // across all B tokens. v0[j] = group-0 quad lane j, etc. (matches the Q5
        // macro in dmmv_q5k_fast: nibble + 16*qh_bit).
        #define Q5B(q,sh,sb,j) ((float)(((q)>>(sh))&0xFu) + 16.0f*(float)(((qh)>>((sb)+(j)*8u))&1u))
        float v0[4] = { Q5B(qs0,0,sba,0), Q5B(qs0,8,sba,1), Q5B(qs0,16,sba,2), Q5B(qs0,24,sba,3) };
        float v1[4] = { Q5B(qs0,4,sbb,0), Q5B(qs0,12,sbb,1), Q5B(qs0,20,sbb,2), Q5B(qs0,28,sbb,3) };
        float v2[4] = { Q5B(qs1,0,sbc,0), Q5B(qs1,8,sbc,1), Q5B(qs1,16,sbc,2), Q5B(qs1,24,sbc,3) };
        float v3[4] = { Q5B(qs1,4,sbd,0), Q5B(qs1,12,sbd,1), Q5B(qs1,20,sbd,2), Q5B(qs1,28,sbd,3) };
        #undef Q5B
        unsigned bidx = (i*256u + y_loc) >> 2, bidx2 = (i*256u + y_loc + 128u) >> 2;
        #pragma unroll
        for (int t = 0; t < B; t++) {
            const float4* xv = (const float4*)(x + (unsigned)t * pc.K + xo);
            float4 by0=xv[bidx], by1=xv[bidx+8u], by2=xv[bidx2], by3=xv[bidx2+8u];
            float s = 0.0f;
            s += (f0*v0[0]-b0)*by0.x + (f0*v0[1]-b0)*by0.y + (f0*v0[2]-b0)*by0.z + (f0*v0[3]-b0)*by0.w;
            s += (f1*v1[0]-b1)*by1.x + (f1*v1[1]-b1)*by1.y + (f1*v1[2]-b1)*by1.z + (f1*v1[3]-b1)*by1.w;
            s += (f2*v2[0]-b2)*by2.x + (f2*v2[1]-b2)*by2.y + (f2*v2[2]-b2)*by2.z + (f2*v2[3]-b2)*by2.w;
            s += (f3*v3[0]-b3)*by3.x + (f3*v3[1]-b3)*by3.y + (f3*v3[2]-b3)*by3.z + (f3*v3[3]-b3)*by3.w;
            sum[t] += s;
        }
    }
    #pragma unroll
    for (int t = 0; t < B; t++) {
        float r = zinc_block_reduce_sum(sum[t]);
        if (tid == 0) {
            unsigned yi = (pc.y_offset >> 2) + (unsigned)t * pc.M + row;
            if (pc.acc_mode != 0u) y[yi] += r; else y[yi] = r;
        }
        __syncthreads();
    }
}
extern "C" __global__ void dmmv_q5k_btok2(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<2>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok3(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<3>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok4(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<4>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok5(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<5>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok6(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<6>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok7(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<7>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok8(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<8>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok9(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<9>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok10(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<10>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok11(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<11>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok12(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<12>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok13(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<13>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok14(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<14>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok15(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<15>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok16(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<16>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok17(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<17>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok18(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<18>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok19(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<19>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok20(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<20>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok21(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<21>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok22(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<22>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok23(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<23>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok24(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<24>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok25(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<25>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok26(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<26>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q5k_btok27(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q5k_btok_impl<27>(a_u32, x, y, pc); }

// ---- dmmv_q8_0_btok (Effort 28) — Q8_0 token-BATCH matvec --------------------
// The Q8_0 analog: read each Q8_0 block (d + 32 int8) ONCE, reuse across B tokens.
// Same per-element arithmetic as dmmv_q8_0_fast; per-token local sum then
// sum[t] += s → token-identical to dmmv_q8_0_fast on x[t]. x/y token-major.
template<int B>
__device__ __forceinline__ void dmmv_q8_0_btok_impl(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 5;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 34u;
    unsigned xb0 = pc.x_offset >> 2, tid = threadIdx.x;
    float sum[B];
    #pragma unroll
    for (int t = 0; t < B; t++) sum[t] = 0.0f;
    for (unsigned bi = tid; bi < bpr; bi += blockDim.x) {
        const unsigned char* blk = arow + (size_t)bi * 34u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
        const signed char* qs = (const signed char*)(blk + 2);
        #pragma unroll
        for (int t = 0; t < B; t++) {
            const float4* xb = (const float4*)(x + (unsigned)t * pc.K + xb0 + (size_t)bi * 32u);
            float s = 0.0f;
            #pragma unroll
            for (unsigned j = 0; j < 8u; j++) { float4 xx = xb[j]; s += d * ((float)qs[j*4]*xx.x + (float)qs[j*4+1]*xx.y + (float)qs[j*4+2]*xx.z + (float)qs[j*4+3]*xx.w); }
            sum[t] += s;
        }
    }
    #pragma unroll
    for (int t = 0; t < B; t++) {
        float r = zinc_block_reduce_sum(sum[t]);
        if (tid == 0) {
            unsigned yi = (pc.y_offset >> 2) + (unsigned)t * pc.M + row;
            if (pc.acc_mode != 0u) y[yi] += r; else y[yi] = r;
        }
        __syncthreads();
    }
}
extern "C" __global__ void dmmv_q8_0_btok2(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<2>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok3(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<3>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok4(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<4>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok5(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<5>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok6(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<6>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok7(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<7>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok8(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<8>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok9(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<9>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok10(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<10>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok11(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<11>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok12(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<12>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok13(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<13>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok14(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<14>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok15(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<15>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok16(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<16>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok17(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<17>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok18(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<18>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok19(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<19>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok20(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<20>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok21(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<21>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok22(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<22>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok23(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<23>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok24(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<24>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok25(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<25>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok26(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<26>(a, x, y, pc); }
extern "C" __global__ void dmmv_q8_0_btok27(const unsigned char* a, const float* x, float* y, DmmvPush pc) { dmmv_q8_0_btok_impl<27>(a, x, y, pc); }

// ---- dmmv_q5k multi-row (perf research, agenda 1: extend mr2 to Q5_K) --------
// One block computes R=2 output rows; each shared x-superblock loaded once and
// reused across both rows. Same Q5_K dequant as dmmv_q5k_fast (qh 5th-bit
// promote). Targets Q5_K mid-M matvecs (ffn_down / attn_v in Q4_K_M mixes),
// stuck ~65% peak at M=12288 in the single-row fast kernel.
template<int R>
__device__ __forceinline__ void dmmv_q5k_mrow_impl(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row0 = blockIdx.x * (unsigned)R;
    if (row0 >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2));
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u, ngrp = blockDim.x >> 4;
    unsigned sba = 2u*v_im, sbb = 2u*v_im+1u, sbc = 2u*v_im+4u, sbd = 2u*v_im+5u;
    unsigned a0 = (pc.a_offset >> 2);
    float sum[R];
    #pragma unroll
    for (int r=0;r<R;r++) sum[r]=0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned bidx = (i*256u + y_loc) >> 2, bidx2 = (i*256u + y_loc + 128u) >> 2;
        float4 by0=xv[bidx], by1=xv[bidx+8u], by2=xv[bidx2], by3=xv[bidx2+8u];
        #pragma unroll
        for (int r=0;r<R;r++){
            unsigned row = row0 + (unsigned)r;
            if (row >= pc.M) continue;
            unsigned blk = a0 + row * bpr * 44u + i * 44u, dd = a_u32[blk];
            float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF)), dm = zinc_half_to_float((unsigned short)(dd >> 16));
            unsigned sc0 = a_u32[blk+1u], sc1 = a_u32[blk+2u], sc2 = a_u32[blk+3u];
            unsigned qh = a_u32[blk + 4u + (l0 >> 2)];
            unsigned qs0 = a_u32[blk + 12u + (q_off >> 2)], qs1 = a_u32[blk + 12u + (q_off >> 2) + 16u];
            unsigned s0=sc0>>shift, s1=sc1>>shift, s2=sc2>>shift;
            float f0=d*(float)(s0&0x3Fu), b0=dm*(float)(s1&0x3Fu);
            float f1=d*(float)((s0>>8)&0x3Fu), b1=dm*(float)((s1>>8)&0x3Fu);
            float f2=d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2=dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
            float f3=d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3=dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
            #define Q5MR(q,sh,sb,j) ((float)(((q)>>(sh))&0xFu) + 16.0f*(float)(((qh)>>((sb)+(j)*8u))&1u))
            float s = 0.0f;
            s += (f0*Q5MR(qs0,0,sba,0)-b0)*by0.x + (f0*Q5MR(qs0,8,sba,1)-b0)*by0.y + (f0*Q5MR(qs0,16,sba,2)-b0)*by0.z + (f0*Q5MR(qs0,24,sba,3)-b0)*by0.w;
            s += (f1*Q5MR(qs0,4,sbb,0)-b1)*by1.x + (f1*Q5MR(qs0,12,sbb,1)-b1)*by1.y + (f1*Q5MR(qs0,20,sbb,2)-b1)*by1.z + (f1*Q5MR(qs0,28,sbb,3)-b1)*by1.w;
            s += (f2*Q5MR(qs1,0,sbc,0)-b2)*by2.x + (f2*Q5MR(qs1,8,sbc,1)-b2)*by2.y + (f2*Q5MR(qs1,16,sbc,2)-b2)*by2.z + (f2*Q5MR(qs1,24,sbc,3)-b2)*by2.w;
            s += (f3*Q5MR(qs1,4,sbd,0)-b3)*by3.x + (f3*Q5MR(qs1,12,sbd,1)-b3)*by3.y + (f3*Q5MR(qs1,20,sbd,2)-b3)*by3.z + (f3*Q5MR(qs1,28,sbd,3)-b3)*by3.w;
            #undef Q5MR
            sum[r] += s;
        }
    }
    #pragma unroll
    for (int r=0;r<R;r++){
        float t = zinc_block_reduce_sum(sum[r]);
        if (tid==0){ unsigned row=row0+(unsigned)r; if(row<pc.M){ unsigned yi=(pc.y_offset>>2)+row; if(pc.acc_mode!=0u) y[yi]+=t; else y[yi]=t; } }
        __syncthreads();
    }
}
extern "C" __global__ void dmmv_q5k_mr2(const unsigned* a_u32, const float* x, float* y, DmmvPush pc){ dmmv_q5k_mrow_impl<2>(a_u32,x,y,pc); }

// ---- gemm_q4k_tiled (perf research, agenda 5: PREFILL) -----------------------
// Y[T,M] = A[T,K] x W[M,K]^T, W = Q4_K, A = f32. Block computes an 8-row x
// 32-token output tile. Per K-superblock (256 elems): dequant the 8 W-rows and
// stage 32 A-rows into shared (transposed: [e][row] so warp-lane reads are
// bank-conflict-free), __syncthreads, then a 256-thread tile-multiply (each
// thread owns one Y[tok,row], mm=warp, tt=lane). W (dequant) reused 32x across
// tokens, A reused 8x across rows -> both operands served from shared memory.
// fp32 accumulate (correctness-first; tensor-core inner product is the next step).
struct GemmPush { unsigned M, K, T, a_offset, x_offset, y_offset, acc_mode, q8_stride; };
extern "C" __global__ void gemm_q4k_tiled(const unsigned* a_u32, const float* A, float* Y, GemmPush pc) {
    __shared__ float Ws[256 * 8];   // Ws[e*8 + mm]  (8 rows)
    __shared__ float As[256 * 32];  // As[e*32 + tt] (32 tokens)
    const unsigned BM = 8u, BT = 32u;
    unsigned m0 = blockIdx.x * BM;
    unsigned t0 = blockIdx.y * BT;
    unsigned bpr = pc.K >> 8;
    unsigned tid = threadIdx.x;                 // 0..255
    unsigned warp = tid >> 5, lane = tid & 31u; // 8 warps x 32 lanes
    unsigned a0 = (pc.a_offset >> 2);
    const float* Abase = A + (pc.x_offset >> 2);
    unsigned mm = warp, tt = lane;              // this thread's output (row mm, token tt)
    float acc = 0.0f;
    for (unsigned sb = 0; sb < bpr; sb++) {
        // --- dequant W: warp `warp` dequants row (m0+warp)'s superblock sb ---
        unsigned row = m0 + warp;
        if (row < pc.M) {
            unsigned blk = a0 + row * bpr * 36u + sb * 36u;
            unsigned dd = a_u32[blk];
            float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
            float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
            const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
            const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned e = lane + (unsigned)j * 32u;
                unsigned chunk = e >> 6, half_ = (e & 63u) >> 5, l = e & 31u;
                unsigned char sc, mn; zinc_q4k_scale_min((int)(chunk * 2u + half_), scales, &sc, &mn);
                unsigned char qb = qs[chunk * 32u + l];
                unsigned nib = (half_ == 0u) ? (qb & 0xFu) : (unsigned)(qb >> 4);
                Ws[e * 8u + warp] = d * (float)sc * (float)nib - dmin * (float)mn;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) Ws[(lane + (unsigned)j * 32u) * 8u + warp] = 0.0f;
        }
        // --- load A tile: As[e*32 + t], 256 threads x 32 = 8192 elems ---
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            unsigned idx = tid + (unsigned)j * 256u;   // 0..8191
            unsigned t = idx >> 8, e = idx & 255u;     // token 0..31, elem 0..255
            unsigned tok = t0 + t;
            As[e * 32u + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + sb * 256u + e] : 0.0f;
        }
        __syncthreads();
        // --- tile multiply: acc += sum_e Ws[mm] * As[tt] ---
        #pragma unroll 8
        for (int e = 0; e < 256; e++) acc += Ws[(unsigned)e * 8u + mm] * As[(unsigned)e * 32u + tt];
        __syncthreads();
    }
    unsigned tok = t0 + tt, row = m0 + mm;
    if (tok < pc.T && row < pc.M) {
        unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
        if (pc.acc_mode != 0u) Y[yi] += acc; else Y[yi] = acc;
    }
}

// ---- gemm_q4k_tiled_v2 — register-blocked prefill GEMM (perf research) -------
// 64-row x 64-token output tile, 256 threads (16x16) each computing a 4x4
// register micro-tile. BK=32 (one Q4_K sub-block) per chunk: dequant the 64
// W-rows' sub-block + stage 64 A-rows into shared, then a register-blocked
// multiply. W reused 64x, A reused 64x; 16 FMA per 8 shared loads (4x the
// arithmetic intensity of v1's 1-output-per-thread inner loop). fp32 accumulate.
extern "C" __global__ void gemm_q4k_tiled_v2(const unsigned* a_u32, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float Ws[BK * BM];   // Ws[kk*64 + r]
    __shared__ float As[BK * BT];   // As[kk*64 + t]
    unsigned m0 = blockIdx.x * BM, t0 = blockIdx.y * BT;
    unsigned bpr = pc.K >> 8;
    unsigned nchunk = pc.K >> 5;    // K/32 sub-blocks
    unsigned tid = threadIdx.x;
    unsigned tx = tid & 15u, ty = tid >> 4;   // 16x16 thread grid
    unsigned a0 = (pc.a_offset >> 2);
    const float* Abase = A + (pc.x_offset >> 2);
    float acc[4][4];
    #pragma unroll
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) acc[i][j]=0.0f;
    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u;   // superblock, sub-block 0..7
        // dequant W tile: BM*BK = 2048 elems / 256 threads = 8 each
        // Optimized: each warp processes one row at a time. Lane 0 reads the
        // Q4_K block header and broadcasts d*scale and dmin*min via __shfl_sync.
        // This eliminates 31 redundant header reads + scale lookups per row.
        unsigned warp_id = tid >> 5;  // 0..7
        unsigned lane = tid & 31u;    // 0..31
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned r = warp_id + (unsigned)u * 8u;  // row (same for all lanes in warp)
            unsigned l = lane;                          // element within sub-block
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned blk = a0 + row * bpr * 36u + sbk * 36u;
                float d_sc, dm_mn;
                if (lane == 0) {
                    unsigned dd = a_u32[blk];
                    float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
                    float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
                    const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
                    unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
                    d_sc = d * (float)sc;
                    dm_mn = dmin * (float)mn;
                }
                d_sc = __shfl_sync(0xFFFFFFFFu, d_sc, 0);
                dm_mn = __shfl_sync(0xFFFFFFFFu, dm_mn, 0);
                const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
                unsigned char qb = qs[(sb8 >> 1) * 32u + l];
                unsigned nib = (sb8 & 1u) == 0u ? (qb & 0xFu) : (unsigned)(qb >> 4);
                wv = d_sc * (float)nib - dm_mn;
            }
            Ws[l * BM + r] = wv;
        }
        // load A tile: BT*BK = 2048 / 256 = 8 each
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned tok = t0 + t;
            As[l * BT + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + c * 32u + l] : 0.0f;
        }
        __syncthreads();
        // register-blocked multiply — vectorized float4 loads from shared memory
        #pragma unroll
        for (unsigned kk = 0; kk < BK; kk++) {
            float4 wr4 = *((const float4*)(Ws + kk * BM + ty*4u));
            float4 ar4 = *((const float4*)(As + kk * BT + tx*4u));
            #pragma unroll
            for (int i=0;i<4;i++)
                #pragma unroll
                for (int j=0;j<4;j++)
                    acc[i][j] += ((&wr4.x)[i]) * ((&ar4.x)[j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i=0;i<4;i++) {
        unsigned row = m0 + ty*4u + (unsigned)i;
        #pragma unroll
        for (int j=0;j<4;j++) {
            unsigned tok = t0 + tx*4u + (unsigned)j;
            if (row < pc.M && tok < pc.T) {
                unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
                if (pc.acc_mode != 0u) Y[yi] += acc[i][j]; else Y[yi] = acc[i][j];
            }
        }
    }
}

// ---- gemm_q8_0_tc_lowsmem — 16KB-shared TC Q8_0 GEMM with float input ------
// Same aliased-shared structure as gemm_q4k_tc_lowsmem but with Q8_0 dequant
// (trivial: d * int8). Used for the 35B's Q8_0 shared-expert weights.
extern "C" __global__ void gemm_q8_0_tc_lowsmem(const unsigned char* a, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float smem[BT*BM];
    half* Ws = (half*)smem;
    half* As = ((half*)smem) + (BM*BK);
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT;
    unsigned bpr=pc.K>>5; // Q8_0: 32 elements per block
    unsigned tid=threadIdx.x;
    unsigned warp=tid>>5, fm=warp>>2, ft=warp&3u;
    unsigned wid=tid>>5, lane=tid&31u;
    const float* Abase=A+(pc.x_offset>>2);

    wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
    wmma::fill_fragment(c0,0.0f); wmma::fill_fragment(c1,0.0f);

    for (unsigned c=0u;c<bpr;c++){
        // Dequant Q8_0 W block: 32 elements per block, 34 bytes each (2B scale + 32B int8)
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned r=wid+(unsigned)u*8u, l=lane;
            unsigned row=m0+r, k=c*32u+l;
            float wv=0.0f;
            if(row<pc.M){
                const unsigned char* blkp = a + (size_t)row * bpr * 34u + (size_t)c * 34u;
                float d = zinc_half_to_float((unsigned short)((unsigned)blkp[0] | ((unsigned)blkp[1] << 8)));
                wv = d * (float)((signed char)blkp[2u + l]);
            }
            Ws[r*BK+l]=__float2half(wv);
        }
        // Float→half A load
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u;
            unsigned t=idx>>5,l=idx&31u,tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?__float2half(Abase[(size_t)tok*pc.K+c*32u+l]):(half)0;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned ks=0;ks<2;ks++){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f,a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f,&Ws[(fm*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(a1f,&Ws[((fm+2u)*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(bf,&As[(ks*16u)*BT+ft*16u],BT);
            wmma::mma_sync(c0,a0f,bf,c0);
            wmma::mma_sync(c1,a1f,bf,c1);
        }
        __syncthreads();
    }
    float* Cs=smem;
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+fm*16u],c0,BM,wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+(fm+2u)*16u],c1,BM,wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for(int u=0;u<16;u++){
        unsigned idx=tid+(unsigned)u*256u;
        unsigned r=idx%BM,t=idx/BM;
        unsigned row=m0+r, tok=t0+t;
        if(row<pc.M&&tok<pc.T){
            unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row;
            if(pc.acc_mode!=0u) Y[yi]+=Cs[t*BM+r]; else Y[yi]=Cs[t*BM+r];
        }
    }
}

// ---- gemm_q6k_tc_lowsmem — 16KB-shared TC Q6_K GEMM with float input ------
extern "C" __global__ void gemm_q6k_tc_lowsmem(const unsigned char* a, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float smem[BT*BM];
    half* Ws = (half*)smem;
    half* As = ((half*)smem) + (BM*BK);
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT;
    unsigned bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x;
    unsigned warp=tid>>5, fm=warp>>2, ft=warp&3u;
    unsigned wid=tid>>5, lane=tid&31u;
    const float* Abase=A+(pc.x_offset>>2);
    wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
    wmma::fill_fragment(c0,0.0f); wmma::fill_fragment(c1,0.0f);
    for (unsigned c=0u;c<nchunk;c++){
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u, r=idx>>5, l=idx&31u, row=m0+r;
            float wv=0.0f;
            if(row<pc.M){
                unsigned e=c*32u+l, within=e&255u, sb=e>>8;
                const unsigned char* blk=a+pc.a_offset+(size_t)row*bpr*210u+(size_t)sb*210u;
                float d=zinc_half_to_float((unsigned short)((unsigned)blk[208]|((unsigned)blk[209]<<8)));
                unsigned half_=within>>7, wh=within&127u, ll=wh&31u, group=wh>>5;
                const unsigned char* ql=blk+(size_t)half_*64u;
                const unsigned char* qh=blk+128u+(size_t)half_*32u;
                const signed char* sc=(const signed char*)(blk+192u+(size_t)half_*8u);
                unsigned is=ll>>4, qhb=qh[ll], q, sci;
                if(group==0u){ q=(ql[ll]&0xFu)|(((qhb>>0)&3u)<<4); sci=is+0u; }
                else if(group==1u){ q=(ql[ll+32u]&0xFu)|(((qhb>>2)&3u)<<4); sci=is+2u; }
                else if(group==2u){ q=(ql[ll]>>4)|(((qhb>>4)&3u)<<4); sci=is+4u; }
                else { q=(ql[ll+32u]>>4)|(((qhb>>6)&3u)<<4); sci=is+6u; }
                wv=d*(float)sc[sci]*((float)q-32.0f);
            }
            Ws[r*BK+l]=__float2half(wv);
        }
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u, t=idx>>5,l=idx&31u,tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?__float2half(Abase[(size_t)tok*pc.K+c*32u+l]):(half)0;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned ks=0;ks<2;ks++){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f,a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f,&Ws[(fm*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(a1f,&Ws[((fm+2u)*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(bf,&As[(ks*16u)*BT+ft*16u],BT);
            wmma::mma_sync(c0,a0f,bf,c0); wmma::mma_sync(c1,a1f,bf,c1);
        }
        __syncthreads();
    }
    float* Cs=smem;
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+fm*16u],c0,BM,wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+(fm+2u)*16u],c1,BM,wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for(int u=0;u<16;u++){
        unsigned idx=tid+(unsigned)u*256u, r=idx%BM,t=idx/BM, row=m0+r, tok=t0+t;
        if(row<pc.M&&tok<pc.T){
            unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row;
            if(pc.acc_mode!=0u) Y[yi]+=Cs[t*BM+r]; else Y[yi]=Cs[t*BM+r];
        }
    }
}

// ---- gemm_q5k_tc_lowsmem — 16KB-shared TC Q5_K GEMM with float input ------
extern "C" __global__ void gemm_q5k_tc_lowsmem(const unsigned char* a, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float smem[BT*BM];
    half* Ws = (half*)smem;
    half* As = ((half*)smem) + (BM*BK);
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT;
    unsigned bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x;
    unsigned warp=tid>>5, fm=warp>>2, ft=warp&3u;
    unsigned wid=tid>>5, lane=tid&31u;
    const float* Abase=A+(pc.x_offset>>2);
    wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
    wmma::fill_fragment(c0,0.0f); wmma::fill_fragment(c1,0.0f);
    for (unsigned c=0u;c<nchunk;c++){
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u, r=idx>>5, l=idx&31u, row=m0+r;
            float wv=0.0f;
            if(row<pc.M){
                unsigned e=c*32u+l, within=e&255u, sb=e>>8;
                const unsigned char* blk=a+pc.a_offset+(size_t)row*bpr*176u+(size_t)sb*176u;
                float d=zinc_half_to_float((unsigned short)((unsigned)blk[0]|((unsigned)blk[1]<<8)));
                float dmin=zinc_half_to_float((unsigned short)((unsigned)blk[2]|((unsigned)blk[3]<<8)));
                const unsigned char* scales=blk+4u; const unsigned char* qh=blk+16u; const unsigned char* qs=blk+48u;
                unsigned chunk=within>>6, half_=(within&63u)>>5, ll=within&31u;
                unsigned char qb=qs[chunk*32u+ll]; unsigned nib=half_==0u?(qb&0xFu):(unsigned)(qb>>4);
                unsigned bit=(qh[ll]>>(2u*chunk+half_))&1u; unsigned q5=nib+(bit?16u:0u);
                unsigned char sc,mn; zinc_q4k_scale_min((int)(chunk*2u+half_),scales,&sc,&mn);
                wv=d*(float)sc*(float)q5 - dmin*(float)mn;
            }
            Ws[r*BK+l]=__float2half(wv);
        }
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u, t=idx>>5,l=idx&31u,tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?__float2half(Abase[(size_t)tok*pc.K+c*32u+l]):(half)0;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned ks=0;ks<2;ks++){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f,a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f,&Ws[(fm*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(a1f,&Ws[((fm+2u)*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(bf,&As[(ks*16u)*BT+ft*16u],BT);
            wmma::mma_sync(c0,a0f,bf,c0); wmma::mma_sync(c1,a1f,bf,c1);
        }
        __syncthreads();
    }
    float* Cs=smem;
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+fm*16u],c0,BM,wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+(fm+2u)*16u],c1,BM,wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for(int u=0;u<16;u++){
        unsigned idx=tid+(unsigned)u*256u, r=idx%BM,t=idx/BM, row=m0+r, tok=t0+t;
        if(row<pc.M&&tok<pc.T){
            unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row;
            if(pc.acc_mode!=0u) Y[yi]+=Cs[t*BM+r]; else Y[yi]=Cs[t*BM+r];
        }
    }
}

// ---- quantize_act_q8_0 — pre-quantize float activations to Q8_1 format -----
// Grid: (ceilDiv(K, 256), T). Block: 256 threads. Each block quantizes 256 elements
// (8 blocks of 32 elements each). Output is [f16 d, f16 sum, 32xi8] so the
// asymmetric Q4_K minimum correction does not have to re-sum activations in
// every matrix multiplication.
struct QuantActPush { unsigned K, T; };
extern "C" __global__ void quantize_act_q8_0(const float* __restrict__ act,
    unsigned char* __restrict__ out, QuantActPush pc) {
    unsigned tok = blockIdx.y;
    unsigned chunk_base = blockIdx.x * 256u;
    unsigned tid = threadIdx.x;
    unsigned k_idx = chunk_base + tid;
    const float* in = act + (size_t)tok * pc.K;
#ifdef ZINC_ROCM
    // MMQ layout: [K/128][T][4 scales + 4*32 quants]. Keeping one 128-value
    // activation tile contiguous lets the WMMA GEMM stage it with vector-like
    // coalesced copies instead of reconstructing a token-major layout.
    const unsigned wblk = tid >> 5;
    const unsigned wlane = tid & 31u;
    const unsigned c = (chunk_base >> 5) + wblk;
    const unsigned h = c >> 2;
    const unsigned g = c & 3u;
    unsigned char* out_base = out + ((size_t)h * pc.T + tok) * 144u;
#else
    unsigned char* out_base = out + (size_t)tok * (pc.K / 32u) * 36u + (chunk_base / 32u) * 36u;

    // Each warp handles 32 elements → 1 Q8_1 block
    const unsigned wblk = tid >> 5;
    const unsigned wlane = tid & 31u;
#endif

    // Read element (zero-pad beyond K)
    float val = (k_idx < pc.K) ? in[k_idx] : 0.0f;

    // Warp-reduce max_abs and the original-value sum.
    float av = fabsf(val), sv = val;
    for (int o = 16; o > 0; o >>= 1) {
        av = fmaxf(av, __shfl_xor_sync(0xFFFFFFFFu, av, o));
        sv += __shfl_xor_sync(0xFFFFFFFFu, sv, o);
    }

    float d = av / 127.0f;
    float scale = 127.0f / fmaxf(av, 1e-5f);
    int q = max(-127, min(127, __float2int_rn(val * scale)));

    // Lane 0 writes the fp16 scale and sum.
    if (wlane == 0 && k_idx < pc.K) {
        unsigned short dh = zinc_float_to_half(d);
        unsigned short sh = zinc_float_to_half(sv);
#ifdef ZINC_ROCM
        out_base[g * 4u]     = (unsigned char)(dh & 0xFF);
        out_base[g * 4u + 1] = (unsigned char)(dh >> 8);
        out_base[g * 4u + 2] = (unsigned char)(sh & 0xFF);
        out_base[g * 4u + 3] = (unsigned char)(sh >> 8);
#else
        out_base[wblk * 36u]     = (unsigned char)(dh & 0xFF);
        out_base[wblk * 36u + 1] = (unsigned char)(dh >> 8);
        out_base[wblk * 36u + 2] = (unsigned char)(sh & 0xFF);
        out_base[wblk * 36u + 3] = (unsigned char)(sh >> 8);
#endif
    }
    // All threads write their int8 value
#ifdef ZINC_ROCM
    out_base[16u + g * 32u + wlane] = (unsigned char)q;
#else
    out_base[wblk * 36u + 4u + wlane] = (unsigned char)q;
#endif
}

// ---- gemm_q4k_wmma_i8 — gfx12 Q4_K x Q8_1 integer-WMMA GEMM ---------------
// One 256-thread block computes a 128-row x 112-token output tile. Each wave
// covers 16 weight rows and seven 16-token columns. Q4_K nibbles
// are expanded into the WMMA input layout once per 256-value superblock; four
// Q8_1 blocks (128 values) are staged at a time. Weight scale/min and activation
// scale/sum pairs remain packed as half2, minimizing shared-memory traffic and
// register pressure while f32 accumulators preserve output quality.
#ifdef ZINC_ROCM
template <bool Q5, unsigned NFRAG>
__device__ __forceinline__ void zinc_gemm_q45k_wmma_i8(
    const unsigned* __restrict__ a_u32, const float* __restrict__ A_unused,
    const unsigned char* __restrict__ A_q8, float* __restrict__ Y, GemmPush pc) {
    (void)A_unused;
    const unsigned BM = 128u, BT = NFRAG * 16u;
    // Four trailing words make the row stride 4 mod 8, avoiding LDS bank
    // conflicts in adjacent WMMA fragment loads.
    const unsigned X_STRIDE = 76u; // 64 q words + 8 half2 pairs + 4 padding words
    const unsigned B_STRIDE = 36u; // 4 packed half2 scale/sum pairs + 32 q words
    const unsigned BLOCK_WORDS = Q5 ? 44u : 36u;
    __shared__ __align__(16) int Xtile[BM * X_STRIDE];
    __shared__ __align__(16) int Btile[((NFRAG * 16u * 36u + 255u) / 256u) * 256u];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wid = tid >> 5;
    const unsigned m0 = blockIdx.x * BM;
    const unsigned token_base = pc.x_offset / (pc.K * sizeof(float));
    const unsigned t0 = token_base + blockIdx.y * BT;
    const unsigned bpr = pc.K >> 8;
    const unsigned nsuper = pc.K >> 8;

    float accum[NFRAG * 8u];
    #pragma unroll
    for (unsigned i = 0; i < NFRAG * 8u; ++i) accum[i] = 0.0f;

    using zinc_i32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using zinc_i32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
    half2* const Xdm = (half2*)(Xtile + 64u);
    half2* const Bds = (half2*)Btile;

    for (unsigned sb = 0; sb < nsuper; ++sb) {
        // Expand one packed Q4_K row per wave at a time. Each lane loads four
        // packed bytes and writes four low plus four high nibbles as byte lanes
        // in two int32 values. The resulting 64 words are directly consumable
        // by gfx12 WMMA without a transpose in the math loop.
        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += 8u) {
            const unsigned r = r0 + wid;
            const unsigned row = m0 + r;
            const unsigned blk = row * bpr * BLOCK_WORDS + sb * BLOCK_WORDS;
            if constexpr (!Q5) {
                const unsigned qs4 = a_u32[blk + 4u + lane];
                const unsigned xi = r * X_STRIDE + 16u * (lane >> 3) + (lane & 7u);
                Xtile[xi] = (int)(qs4 & 0x0F0F0F0Fu);
                Xtile[xi + 8u] = (int)((qs4 >> 4) & 0x0F0F0F0Fu);
            } else {
                const unsigned ql = a_u32[blk + 12u + lane];
                const unsigned qli0 = ql & 0x0F0F0F0Fu;
                const unsigned qli1 = (ql >> 4) & 0x0F0F0F0Fu;
                const unsigned qhi = a_u32[blk + 4u + (lane & 7u)];
                const unsigned qh0 = ((qhi >> (2u * (lane >> 3))) << 4) & 0x10101010u;
                const unsigned qh1 = ((qhi >> (2u * (lane >> 3) + 1u)) << 4) & 0x10101010u;
                const unsigned ky = 2u * lane;
                const unsigned kq0 = ky - (ky & 15u) + (lane & 7u);
                Xtile[r * X_STRIDE + kq0] = (int)(qli0 | qh0);
                Xtile[r * X_STRIDE + kq0 + 8u] = (int)(qli1 | qh1);
            }
        }

        // Two lanes per row unpack all eight 6-bit scale/min pairs. Store
        // (d*scale, -dmin*min) together as half2, matching each 32-value group.
        {
            const unsigned r = wid * 16u + (lane >> 1);
            const unsigned row = m0 + r;
            const unsigned ksc = lane & 1u;
            const unsigned blk = row * bpr * BLOCK_WORDS + sb * BLOCK_WORDS;
            const half2 dm = *(const half2*)(a_u32 + blk);
            const int* scales = (const int*)(a_u32 + blk + 1u);
            const int sc32 = ((scales[ksc + (ksc != 0u)] >> (4u * (ksc & (ksc / 2u)))) & 0x0F0F0F0F) |
                ((scales[ksc / 2u] >> (2u * (ksc & 1u))) & 0x30303030);
            const unsigned mksc = ksc + 2u;
            const int mn32 = ((scales[(mksc & 1u) + (mksc != 0u)] >> (4u * (mksc & (mksc / 2u)))) & 0x0F0F0F0F) |
                ((scales[mksc / 2u] >> (2u * (mksc & 1u))) & 0x30303030);
            const unsigned char* sc8 = (const unsigned char*)&sc32;
            const unsigned char* mn8 = (const unsigned char*)&mn32;
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                Xdm[r * X_STRIDE + 4u * ksc + (unsigned)l] = __hmul2(
                    dm, __floats2half2_rn((float)sc8[l], -(float)mn8[l]));
            }
        }

        // Stage four Q8_1 groups (128 K values) at a time. Xtile remains
        // resident while Btile is reused for both halves of the superblock.
        #pragma unroll
        for (int ah = 0; ah < 2; ++ah) {
            const int* by0 = (const int*)A_q8 +
                ((size_t)(sb * 2u + (unsigned)ah) * pc.T + t0) * B_STRIDE;
            #pragma unroll
            for (unsigned l0 = 0u; l0 < ((NFRAG * 16u * 36u + 255u) / 256u) * 256u; l0 += 256u) {
                const unsigned l = l0 + tid;
                Btile[l] = by0[l];
            }
            __syncthreads();

            const unsigned row_base = wid * 16u;
            // Keep the four K groups rolled. Fully unrolling this loop emits
            // 112 WMMA instructions in the kernel body (versus 28 when rolled)
            // and overflows the gfx12 instruction cache on the large Q4/Q5
            // projections. The token-fragment loop below remains unrolled.
            #pragma unroll 1
            for (int ag = 0; ag < 4; ++ag) {
                const unsigned g = (unsigned)ah * 4u + (unsigned)ag;
                const unsigned wr = row_base + (lane & 15u);
                const unsigned wk = (lane >> 4) * 4u;
                const zinc_i32x2_t* wp = (const zinc_i32x2_t*)(Xtile + wr * X_STRIDE + g * 8u + wk);
                const zinc_i32x2_t av0 = wp[0];
                const zinc_i32x2_t av1 = wp[1];

                #pragma unroll
                for (unsigned jt = 0; jt < NFRAG; ++jt) {
                    const unsigned at = jt * 16u + (lane & 15u);
                    const unsigned ak = (lane >> 4) * 4u;
                    const zinc_i32x2_t* bp = (const zinc_i32x2_t*)(Btile + at * B_STRIDE + 4u + (unsigned)ag * 8u + ak);
                    const zinc_i32x2_t bv0 = bp[0];
                    const zinc_i32x2_t bv1 = bp[1];
                    const float2 dsB = __half22float2(Bds[at * B_STRIDE + (unsigned)ag]);
                    zinc_i32x8_t ci = {};
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, av0, true, bv0, ci, true);
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, av1, true, bv1, ci, true);
                    const int* cv = (const int*)&ci;
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const unsigned ri = row_base + (lane >> 4) * 8u + (unsigned)l;
                        const float2 dmA = __half22float2(Xdm[ri * X_STRIDE + g]);
                        // Keep the scale and minimum terms as separate
                        // accumulations. HIP can contract both statements to
                        // FMAs and dual-issue the scale products on gfx12;
                        // combining them into one expression inhibits that
                        // contraction and costs roughly one VALU multiply per
                        // output element.
                        accum[jt * 8u + (unsigned)l] +=
                            dmA.x * dsB.x * (float)cv[l];
                        accum[jt * 8u + (unsigned)l] +=
                            dmA.y * dsB.y;
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (unsigned frag = 0; frag < NFRAG; ++frag) {
        const unsigned tok = t0 + frag * 16u + (lane & 15u);
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const unsigned row = m0 + wid * 16u + (lane >> 4) * 8u + (unsigned)l;
            if (tok < pc.T) {
                const size_t yi = (size_t)tok * pc.M + row;
                Y[yi] = accum[frag * 8u + (unsigned)l];
            }
        }
    }
}

// Q4_K-only variant that feeds each wave's weight fragment directly from the
// packed matrix. Adjacent low/high 4-bit groups reuse the same global loads;
// only the row scale/min pairs and activation tile occupy LDS.
template <unsigned NFRAG>
__device__ __forceinline__ void zinc_gemm_q4k_wmma_i8_direct(
    const unsigned* __restrict__ a_u32, const float* __restrict__ A_unused,
    const unsigned char* __restrict__ A_q8, float* __restrict__ Y, GemmPush pc) {
    (void)A_unused;
    const unsigned BM = 128u, BT = NFRAG * 16u;
    const unsigned B_STRIDE = 36u;
    __shared__ __align__(16) int Btile[((NFRAG * 16u * 36u + 255u) / 256u) * 256u];
    __shared__ __align__(16) half2 Adm[BM * 8u];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wid = tid >> 5;
    const unsigned m0 = blockIdx.x * BM;
    const unsigned token_base = pc.x_offset / (pc.K * sizeof(float));
    const unsigned t0 = token_base + blockIdx.y * BT;
    const unsigned bpr = pc.K >> 8;
    const unsigned nsuper = pc.K >> 8;
    const unsigned a0 = pc.a_offset >> 2;
    half2* const Bds = (half2*)Btile;

    float accum[NFRAG * 8u];
    #pragma unroll
    for (unsigned i = 0; i < NFRAG * 8u; ++i) accum[i] = 0.0f;

    using zinc_direct_i32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using zinc_direct_i32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    const unsigned wr = wid * 16u + (lane & 15u);
    const unsigned row = m0 + wr;
    const unsigned wk = (lane >> 4) * 4u;

    for (unsigned sb = 0; sb < nsuper; ++sb) {
        // Two lanes per output row expand all eight (scale,-min) pairs. The
        // shared tile remaps them to gfx12 accumulator-row ownership.
        {
            const unsigned sr = wid * 16u + (lane >> 1);
            const unsigned scale_row = m0 + sr;
            const unsigned ksc = lane & 1u;
            if (scale_row < pc.M) {
                const unsigned sblk = a0 + scale_row * bpr * 36u + sb * 36u;
                const half2 dm = *(const half2*)(a_u32 + sblk);
                const int* scale_words = (const int*)(a_u32 + sblk + 1u);
                const int sc32 = ((scale_words[ksc + (ksc != 0u)] >>
                    (4u * (ksc & (ksc / 2u)))) & 0x0F0F0F0F) |
                    ((scale_words[ksc / 2u] >> (2u * (ksc & 1u))) & 0x30303030);
                const unsigned mksc = ksc + 2u;
                const int mn32 = ((scale_words[(mksc & 1u) + (mksc != 0u)] >>
                    (4u * (mksc & (mksc / 2u)))) & 0x0F0F0F0F) |
                    ((scale_words[mksc / 2u] >> (2u * (mksc & 1u))) & 0x30303030);
                const unsigned char* sc8 = (const unsigned char*)&sc32;
                const unsigned char* mn8 = (const unsigned char*)&mn32;
                #pragma unroll
                for (unsigned si = 0u; si < 4u; ++si) {
                    Adm[sr * 8u + ksc * 4u + si] = __hmul2(
                        dm, __floats2half2_rn((float)sc8[si], -(float)mn8[si]));
                }
            } else {
                #pragma unroll
                for (unsigned si = 0u; si < 4u; ++si)
                    Adm[sr * 8u + ksc * 4u + si] = __floats2half2_rn(0.0f, 0.0f);
            }
        }

        #pragma unroll
        for (unsigned ah = 0u; ah < 2u; ++ah) {
            const int* by0 = (const int*)A_q8 +
                ((size_t)(sb * 2u + ah) * pc.T + t0) * B_STRIDE;
            #pragma unroll
            for (unsigned l0 = 0u; l0 < ((NFRAG * 16u * 36u + 255u) / 256u) * 256u; l0 += 256u) {
                const unsigned l = l0 + tid;
                Btile[l] = by0[l];
            }
            __syncthreads();

            // Process low/high nibble groups together so the four packed Q4
            // words are fetched once for both groups.
            #pragma unroll 1
            for (unsigned gp = 0u; gp < 2u; ++gp) {
                const unsigned g0 = ah * 4u + gp * 2u;
                const unsigned g1 = g0 + 1u;
                const unsigned p = g0 * 8u + wk;
                const unsigned src_word = (p >> 4) * 8u + (p & 7u);
                unsigned q0 = 0u, q1 = 0u, q2 = 0u, q3 = 0u;
                if (row < pc.M) {
                    const unsigned blk = a0 + row * bpr * 36u + sb * 36u;
                    q0 = a_u32[blk + 4u + src_word];
                    q1 = a_u32[blk + 5u + src_word];
                    q2 = a_u32[blk + 6u + src_word];
                    q3 = a_u32[blk + 7u + src_word];
                }
                const zinc_direct_i32x2_t av00 = {
                    (int)(q0 & 0x0F0F0F0Fu), (int)(q1 & 0x0F0F0F0Fu)
                };
                const zinc_direct_i32x2_t av01 = {
                    (int)(q2 & 0x0F0F0F0Fu), (int)(q3 & 0x0F0F0F0Fu)
                };
                const zinc_direct_i32x2_t av10 = {
                    (int)((q0 >> 4) & 0x0F0F0F0Fu), (int)((q1 >> 4) & 0x0F0F0F0Fu)
                };
                const zinc_direct_i32x2_t av11 = {
                    (int)((q2 >> 4) & 0x0F0F0F0Fu), (int)((q3 >> 4) & 0x0F0F0F0Fu)
                };

                #pragma unroll
                for (unsigned jt = 0u; jt < NFRAG; ++jt) {
                    const unsigned at = jt * 16u + (lane & 15u);
                    const unsigned ak = (lane >> 4) * 4u;
                    const zinc_direct_i32x2_t* bp0 = (const zinc_direct_i32x2_t*)(
                        Btile + at * B_STRIDE + 4u + (g0 & 3u) * 8u + ak);
                    const zinc_direct_i32x2_t* bp1 = (const zinc_direct_i32x2_t*)(
                        Btile + at * B_STRIDE + 4u + (g1 & 3u) * 8u + ak);
                    zinc_direct_i32x8_t ci0 = {};
                    zinc_direct_i32x8_t ci1 = {};
                    ci0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av00, true, bp0[0], ci0, true);
                    ci0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av01, true, bp0[1], ci0, true);
                    ci1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av10, true, bp1[0], ci1, true);
                    ci1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av11, true, bp1[1], ci1, true);
                    const float2 ds0 = __half22float2(Bds[at * B_STRIDE + (g0 & 3u)]);
                    const float2 ds1 = __half22float2(Bds[at * B_STRIDE + (g1 & 3u)]);
                    const int* cv0 = (const int*)&ci0;
                    const int* cv1 = (const int*)&ci1;
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const unsigned ri = wid * 16u + (lane >> 4) * 8u + (unsigned)l;
                        const float2 dm0 = __half22float2(Adm[ri * 8u + g0]);
                        const float2 dm1 = __half22float2(Adm[ri * 8u + g1]);
                        accum[jt * 8u + (unsigned)l] += dm0.x * ds0.x * (float)cv0[l];
                        accum[jt * 8u + (unsigned)l] += dm0.y * ds0.y;
                        accum[jt * 8u + (unsigned)l] += dm1.x * ds1.x * (float)cv1[l];
                        accum[jt * 8u + (unsigned)l] += dm1.y * ds1.y;
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (unsigned frag = 0u; frag < NFRAG; ++frag) {
        const unsigned tok = t0 + frag * 16u + (lane & 15u);
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const unsigned out_row = m0 + wid * 16u + (lane >> 4) * 8u + (unsigned)l;
            if (tok < pc.T && out_row < pc.M)
                Y[(size_t)tok * pc.M + out_row] = accum[frag * 8u + (unsigned)l];
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_wmma_i8(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<false, 7u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5k_wmma_i8(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<true, 7u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_wmma_i8_t48(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<false, 3u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_wmma_i8_t48_direct(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q4k_wmma_i8_direct<3u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5k_wmma_i8_t48(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<true, 3u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_wmma_i8_t64(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<false, 4u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5k_wmma_i8_t64(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<true, 4u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_wmma_i8_t80(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<false, 5u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5k_wmma_i8_t80(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<true, 5u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_wmma_i8_t16(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<false, 1u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5k_wmma_i8_t16(
    const unsigned* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q45k_wmma_i8<true, 1u>(a, x, xq, y, pc);
}

// Q6_K uses a signed 6-bit value and one signed scale per 16 K values. The
// 16-wide gfx12 WMMA form keeps those scale boundaries exact (one instruction
// per scale), while retaining the same 128-row x 112-token output geometry.
__device__ __forceinline__ unsigned zinc_q6_sub32x4(unsigned q) {
    q ^= 0x20202020u;
    const unsigned sign = q & 0x20202020u;
    return q | (sign << 1) | (sign << 2);
}

template <unsigned NFRAG>
__device__ __forceinline__ void zinc_gemm_q6k_wmma_i8(
    const unsigned char* __restrict__ a, const float* __restrict__ A_unused,
    const unsigned char* __restrict__ A_q8, float* __restrict__ Y, GemmPush pc) {
    (void)A_unused;
    const unsigned BM = 128u, BT = NFRAG * 16u;
    const unsigned X_STRIDE = 76u; // 64 q words + d + 4 scale words + padding
    const unsigned B_STRIDE = 36u;
    __shared__ __align__(16) int Xtile[BM * X_STRIDE];
    __shared__ __align__(16) int Btile[((NFRAG * 16u * 36u + 255u) / 256u) * 256u];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wid = tid >> 5;
    const unsigned m0 = blockIdx.x * BM;
    const unsigned token_base = pc.x_offset / (pc.K * sizeof(float));
    const unsigned t0 = token_base + blockIdx.y * BT;
    const unsigned bpr = pc.K >> 8;
    const unsigned nsuper = pc.K >> 8;
    float* const Xdf = (float*)(Xtile + 64u);
    int* const Xsc = (int*)(Xdf + 1u);
    half2* const Bds = (half2*)Btile;

    float accum[NFRAG * 8u];
    #pragma unroll
    for (unsigned i = 0; i < NFRAG * 8u; ++i) accum[i] = 0.0f;

    using zinc_i32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using zinc_i32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    for (unsigned sb = 0; sb < nsuper; ++sb) {
        // Expand the 4-bit low and 2-bit high planes to signed int8 lanes.
        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += 8u) {
            const unsigned r = r0 + wid;
            const unsigned row = m0 + r;
            const unsigned char* blk = a + ((size_t)row * bpr + sb) * 210u;
            const unsigned short* ql16 = (const unsigned short*)blk;
            const unsigned ql = (unsigned)ql16[2u * lane] |
                ((unsigned)ql16[2u * lane + 1u] << 16);
            const unsigned qh_idx = 8u * (lane >> 4) + (lane & 7u);
            const unsigned short* qh16 = (const unsigned short*)(blk + 128u);
            const unsigned qh = (unsigned)qh16[2u * qh_idx] |
                ((unsigned)qh16[2u * qh_idx + 1u] << 16);
            const unsigned shift = (lane & 8u) >> 2;
            const unsigned qh0 = ((qh >> shift) << 4) & 0x30303030u;
            const unsigned qh1 = (qh >> shift) & 0x30303030u;
            const unsigned kq0 = 2u * lane - (lane & 15u);
            Xtile[r * X_STRIDE + kq0] = (int)zinc_q6_sub32x4((ql & 0x0F0F0F0Fu) | qh0);
            Xtile[r * X_STRIDE + kq0 + 16u] = (int)zinc_q6_sub32x4(((ql >> 4) & 0x0F0F0F0Fu) | qh1);
        }

        // Native half conversion for the superblock multiplier.
        if (tid < BM) {
            const unsigned row = m0 + tid;
            const unsigned char* blk = a + ((size_t)row * bpr + sb) * 210u;
            Xdf[tid * X_STRIDE] = __half2float(*(const half*)(blk + 208u));
        }

        // Four packed int32 values contain the sixteen signed group scales.
        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += 64u) {
            const unsigned r = r0 + wid * 8u + (lane >> 2);
            const unsigned row = m0 + r;
            const unsigned sk = lane & 3u;
            const unsigned char* blk = a + ((size_t)row * bpr + sb) * 210u;
            const unsigned short* sc16 = (const unsigned short*)(blk + 192u);
            Xsc[r * X_STRIDE + sk] = (int)((unsigned)sc16[2u * sk] |
                ((unsigned)sc16[2u * sk + 1u] << 16));
        }

        #pragma unroll
        for (int ah = 0; ah < 2; ++ah) {
            const int* by0 = (const int*)A_q8 +
                ((size_t)(sb * 2u + (unsigned)ah) * pc.T + t0) * B_STRIDE;
            #pragma unroll
            for (unsigned l0 = 0u; l0 < ((NFRAG * 16u * 36u + 255u) / 256u) * 256u; l0 += 256u)
                Btile[l0 + tid] = by0[l0 + tid];
            __syncthreads();

            const unsigned row_base = wid * 16u;
            #pragma unroll
            for (int kg = 0; kg < 8; ++kg) {
                const unsigned g16 = (unsigned)ah * 8u + (unsigned)kg;
                const unsigned ag = (unsigned)kg >> 1;
                const unsigned khalf = (unsigned)kg & 1u;
                const unsigned wr = row_base + (lane & 15u);
                const unsigned wk = (lane >> 4) * 2u;
                const zinc_i32x2_t av = *(const zinc_i32x2_t*)(
                    Xtile + wr * X_STRIDE + g16 * 4u + wk);

                #pragma unroll
                for (unsigned jt = 0; jt < NFRAG; ++jt) {
                    const unsigned at = jt * 16u + (lane & 15u);
                    const unsigned ak = (lane >> 4) * 2u;
                    const zinc_i32x2_t bv = *(const zinc_i32x2_t*)(Btile +
                        at * B_STRIDE + 4u + ag * 8u + khalf * 4u + ak);
                    const float dB = __half2float(__low2half(Bds[at * B_STRIDE + ag]));
                    zinc_i32x8_t ci = {};
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, av, true, bv, ci, false);
                    const int* cv = (const int*)&ci;
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const unsigned ri = row_base + (lane >> 4) * 8u + (unsigned)l;
                        const signed char* scales = (const signed char*)(Xsc + ri * X_STRIDE);
                        accum[jt * 8u + (unsigned)l] +=
                            (float)cv[l] * (float)scales[g16] * Xdf[ri * X_STRIDE] * dB;
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (unsigned frag = 0; frag < NFRAG; ++frag) {
        const unsigned tok = t0 + frag * 16u + (lane & 15u);
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const unsigned row = m0 + wid * 16u + (lane >> 4) * 8u + (unsigned)l;
            if (tok < pc.T) Y[(size_t)tok * pc.M + row] = accum[frag * 8u + (unsigned)l];
        }
    }
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q6k_wmma_i8(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q6k_wmma_i8<7u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q6k_wmma_i8_t48(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q6k_wmma_i8<3u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q6k_wmma_i8_t64(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q6k_wmma_i8<4u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q6k_wmma_i8_t80(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q6k_wmma_i8<5u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q6k_wmma_i8_t16(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q6k_wmma_i8<1u>(a, x, xq, y, pc);
}

// Q8_0 weights are already signed int8. Pair them directly with the Q8_1
// activation tile and keep only the per-32-value d multipliers around WMMA.
template <unsigned NFRAG, unsigned BM = 128u>
__device__ __forceinline__ void zinc_gemm_q8_0_wmma_i8(
    const unsigned char* __restrict__ a, const float* __restrict__ A_unused,
    const unsigned char* __restrict__ A_q8, float* __restrict__ Y, GemmPush pc) {
    (void)A_unused;
    const unsigned BT = NFRAG * 16u;
    const unsigned NWARP = BM / 16u;
    const unsigned X_STRIDE = 36u;
    const unsigned B_STRIDE = 36u;
    __shared__ __align__(16) int Xtile[BM * X_STRIDE];
    __shared__ __align__(16) int Btile[BT * B_STRIDE];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wid = tid >> 5;
    const unsigned m0 = blockIdx.x * BM;
    const unsigned token_base = pc.x_offset / (pc.K * sizeof(float));
    const unsigned t0 = token_base + blockIdx.y * BT;
    const unsigned blocks_per_row = pc.K >> 5;
    const unsigned nsuper = pc.K >> 7;
    const unsigned char* const a0 = a + pc.a_offset;
    float* const Xdf = (float*)(Xtile + 32u);
    half2* const Bds = (half2*)Btile;

    float accum[NFRAG * 8u];
    #pragma unroll
    for (unsigned i = 0; i < NFRAG * 8u; ++i) accum[i] = 0.0f;

    using zinc_q8_i32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using zinc_q8_i32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    for (unsigned sb = 0; sb < nsuper; ++sb) {
        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += NWARP) {
            const unsigned r = r0 + wid;
            const unsigned row = m0 + r;
            if (lane < 8u) {
                #pragma unroll
                for (unsigned g = 0u; g < 4u; ++g) {
                    int q = 0;
                    if (row < pc.M) {
                        const unsigned char* blk = a0 +
                            ((size_t)row * blocks_per_row + sb * 4u + g) * 34u;
                        q = *(const int*)(blk + 2u + lane * 4u);
                    }
                    Xtile[r * X_STRIDE + g * 8u + lane] = q;
                }
            }
        }

        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += NWARP * 8u) {
            const unsigned r = r0 + wid * 8u + (lane >> 2);
            const unsigned row = m0 + r;
            const unsigned g = lane & 3u;
            Xdf[r * X_STRIDE + g] = row < pc.M
                ? __half2float(*(const half*)(a0 + ((size_t)row * blocks_per_row + sb * 4u + g) * 34u))
                : 0.0f;
        }

        #pragma unroll
        for (unsigned l0 = 0u; l0 < BT * B_STRIDE; l0 += blockDim.x) {
            const unsigned l = l0 + tid;
            if (l < BT * B_STRIDE) {
                const unsigned local_t = l / B_STRIDE;
                const unsigned word = l - local_t * B_STRIDE;
                const unsigned tok = t0 + local_t;
                if (tok < pc.T) {
                    const int* src = (const int*)A_q8 + ((size_t)sb * pc.T + tok) * B_STRIDE;
                    Btile[l] = src[word];
                } else {
                    Btile[l] = 0;
                }
            }
        }
        __syncthreads();

        const unsigned row_base = wid * 16u;
        #pragma unroll
        for (unsigned g = 0u; g < 4u; ++g) {
            const unsigned wr = row_base + (lane & 15u);
            const unsigned wk = (lane >> 4) * 2u;
            #pragma unroll
            for (unsigned jt = 0; jt < NFRAG; ++jt) {
                const unsigned at = jt * 16u + (lane & 15u);
                const unsigned ak = (lane >> 4) * 2u;
                zinc_q8_i32x8_t ci = {};
                #pragma unroll
                for (unsigned kh = 0u; kh < 2u; ++kh) {
                    const zinc_q8_i32x2_t av = *(const zinc_q8_i32x2_t*)(
                        Xtile + wr * X_STRIDE + g * 8u + kh * 4u + wk);
                    const zinc_q8_i32x2_t bv = *(const zinc_q8_i32x2_t*)(
                        Btile + at * B_STRIDE + 4u + g * 8u + kh * 4u + ak);
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, av, true, bv, ci, true);
                }
                const float dB = __half2float(__low2half(Bds[at * B_STRIDE + g]));
                const int* cv = (const int*)&ci;
                #pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const unsigned ri = row_base + (lane >> 4) * 8u + (unsigned)l;
                    accum[jt * 8u + (unsigned)l] +=
                        (float)cv[l] * Xdf[ri * X_STRIDE + g] * dB;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned frag = 0; frag < NFRAG; ++frag) {
        const unsigned tok = t0 + frag * 16u + (lane & 15u);
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const unsigned row = m0 + wid * 16u + (lane >> 4) * 8u + (unsigned)l;
            if (tok < pc.T && row < pc.M)
                Y[(size_t)tok * pc.M + row] = accum[frag * 8u + (unsigned)l];
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q8_0_wmma_i8(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q8_0_wmma_i8<7u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q8_0_wmma_i8_t48(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q8_0_wmma_i8<3u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q8_0_wmma_i8_t64(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q8_0_wmma_i8<4u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(128, 4) void gemm_q8_0_wmma_i8_t64_m64(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q8_0_wmma_i8<4u, 64u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(64, 8) void gemm_q8_0_wmma_i8_t64_m32(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q8_0_wmma_i8<4u, 32u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q8_0_wmma_i8_t80(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q8_0_wmma_i8<5u>(a, x, xq, y, pc);
}
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q8_0_wmma_i8_t16(
    const unsigned char* a, const float* x, const unsigned char* xq, float* y, GemmPush pc) {
    zinc_gemm_q8_0_wmma_i8<1u>(a, x, xq, y, pc);
}
#else
extern "C" __global__ void gemm_q4k_wmma_i8() {}
extern "C" __global__ void gemm_q5k_wmma_i8() {}
extern "C" __global__ void gemm_q6k_wmma_i8() {}
extern "C" __global__ void gemm_q4k_wmma_i8_t48() {}
extern "C" __global__ void gemm_q4k_wmma_i8_t48_direct() {}
extern "C" __global__ void gemm_q5k_wmma_i8_t48() {}
extern "C" __global__ void gemm_q6k_wmma_i8_t48() {}
extern "C" __global__ void gemm_q4k_wmma_i8_t64() {}
extern "C" __global__ void gemm_q5k_wmma_i8_t64() {}
extern "C" __global__ void gemm_q6k_wmma_i8_t64() {}
extern "C" __global__ void gemm_q4k_wmma_i8_t80() {}
extern "C" __global__ void gemm_q5k_wmma_i8_t80() {}
extern "C" __global__ void gemm_q6k_wmma_i8_t80() {}
extern "C" __global__ void gemm_q4k_wmma_i8_t16() {}
extern "C" __global__ void gemm_q5k_wmma_i8_t16() {}
extern "C" __global__ void gemm_q6k_wmma_i8_t16() {}
extern "C" __global__ void gemm_q8_0_wmma_i8() {}
extern "C" __global__ void gemm_q8_0_wmma_i8_t48() {}
extern "C" __global__ void gemm_q8_0_wmma_i8_t64() {}
extern "C" __global__ void gemm_q8_0_wmma_i8_t64_m64() {}
extern "C" __global__ void gemm_q8_0_wmma_i8_t64_m32() {}
extern "C" __global__ void gemm_q8_0_wmma_i8_t80() {}
extern "C" __global__ void gemm_q8_0_wmma_i8_t16() {}
#endif

// ---- grouped gfx12 Q4_K/Q5_K x Q8_1 expert GEMM --------------------------
// One block handles a 128-row x 64-route tile for one expert selected by
// build_expert_order_padded. Activations are quantized once per source token;
// the padded order gathers them into LDS and scatters results back to the
// original (token, route-slot) layout. This is the ROCm counterpart of the
// CUDA grouped fp16-WMMA path above.
// route_stride=0 addresses one activation per source token (gate/up). A nonzero
// route_stride addresses one activation per routed (token,slot) row (down).
struct GroupedI8Push { unsigned M, K, T, base, expert_stride, dst_tok_stride, route_stride; };
#ifdef ZINC_ROCM
template <bool Q5, unsigned NFRAG, unsigned BM = 128u>
__device__ __forceinline__ void zinc_gemm_q45k_experts_grouped_i8(
    const unsigned* __restrict__ a_u32,
    const unsigned char* __restrict__ A_q8,
    const unsigned* __restrict__ order,
    const unsigned* __restrict__ tile_expert,
    float* __restrict__ Y,
    GroupedI8Push pc) {
    const unsigned BT = NFRAG * 16u;
    const unsigned NWARP = BM / 16u;
    const unsigned X_STRIDE = 76u;
    const unsigned B_STRIDE = 36u;
    const unsigned BLOCK_WORDS = Q5 ? 44u : 36u;
    const unsigned INV = 0xFFFFFFFFu;
    __shared__ __align__(16) int Xtile[BM * X_STRIDE];
    __shared__ __align__(16) int Btile[BT * B_STRIDE];

    const unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wid = tid >> 5;
    const unsigned m0 = blockIdx.x * BM;
    const unsigned t0 = blockIdx.y * BT;
    const unsigned bpr = pc.K >> 8;
    const unsigned nsuper = pc.K >> 8;
    const unsigned a0 = (unsigned)(((size_t)expert * pc.expert_stride + pc.base) >> 2);

    float accum[NFRAG * 8u];
    #pragma unroll
    for (unsigned i = 0; i < NFRAG * 8u; ++i) accum[i] = 0.0f;

    using zinc_group_i32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using zinc_group_i32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
    half2* const Xdm = (half2*)(Xtile + 64u);
    half2* const Bds = (half2*)Btile;

    for (unsigned sb = 0; sb < nsuper; ++sb) {
        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += NWARP) {
            const unsigned r = r0 + wid;
            const unsigned row = m0 + r;
            const unsigned xi = r * X_STRIDE + 16u * (lane >> 3) + (lane & 7u);
            if (row < pc.M) {
                const unsigned blk = a0 + row * bpr * BLOCK_WORDS + sb * BLOCK_WORDS;
                if constexpr (!Q5) {
                    const unsigned qs4 = a_u32[blk + 4u + lane];
                    Xtile[xi] = (int)(qs4 & 0x0F0F0F0Fu);
                    Xtile[xi + 8u] = (int)((qs4 >> 4) & 0x0F0F0F0Fu);
                } else {
                    const unsigned ql = a_u32[blk + 12u + lane];
                    const unsigned qli0 = ql & 0x0F0F0F0Fu;
                    const unsigned qli1 = (ql >> 4) & 0x0F0F0F0Fu;
                    const unsigned qhi = a_u32[blk + 4u + (lane & 7u)];
                    const unsigned qh0 = ((qhi >> (2u * (lane >> 3))) << 4) & 0x10101010u;
                    const unsigned qh1 = ((qhi >> (2u * (lane >> 3) + 1u)) << 4) & 0x10101010u;
                    const unsigned ky = 2u * lane;
                    const unsigned kq0 = ky - (ky & 15u) + (lane & 7u);
                    Xtile[r * X_STRIDE + kq0] = (int)(qli0 | qh0);
                    Xtile[r * X_STRIDE + kq0 + 8u] = (int)(qli1 | qh1);
                }
            } else {
                Xtile[xi] = 0;
                Xtile[xi + 8u] = 0;
            }
        }

        {
            const unsigned r = wid * 16u + (lane >> 1);
            const unsigned row = m0 + r;
            const unsigned ksc = lane & 1u;
            if (row < pc.M) {
                const unsigned blk = a0 + row * bpr * BLOCK_WORDS + sb * BLOCK_WORDS;
                const half2 dm = *(const half2*)(a_u32 + blk);
                const int* scales = (const int*)(a_u32 + blk + 1u);
                const int sc32 = ((scales[ksc + (ksc != 0u)] >> (4u * (ksc & (ksc / 2u)))) & 0x0F0F0F0F) |
                    ((scales[ksc / 2u] >> (2u * (ksc & 1u))) & 0x30303030);
                const unsigned mksc = ksc + 2u;
                const int mn32 = ((scales[(mksc & 1u) + (mksc != 0u)] >> (4u * (mksc & (mksc / 2u)))) & 0x0F0F0F0F) |
                    ((scales[mksc / 2u] >> (2u * (mksc & 1u))) & 0x30303030);
                const unsigned char* sc8 = (const unsigned char*)&sc32;
                const unsigned char* mn8 = (const unsigned char*)&mn32;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    Xdm[r * X_STRIDE + 4u * ksc + (unsigned)l] = __hmul2(
                        dm, __floats2half2_rn((float)sc8[l], -(float)mn8[l]));
                }
            } else {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    Xdm[r * X_STRIDE + 4u * ksc + (unsigned)l] = __floats2half2_rn(0.0f, 0.0f);
                }
            }
        }

        #pragma unroll
        for (int ah = 0; ah < 2; ++ah) {
            #pragma unroll
            for (unsigned l0 = 0u; l0 < BT * B_STRIDE; l0 += blockDim.x) {
                const unsigned l = l0 + tid;
                if (l < BT * B_STRIDE) {
                    const unsigned local_t = l / B_STRIDE;
                    const unsigned word = l - local_t * B_STRIDE;
                    const unsigned packed = order[t0 + local_t];
                    if (packed != INV) {
                        const unsigned tok = packed >> 16;
                        const unsigned slot = packed & 0xFFFFu;
                        const unsigned src_row = pc.route_stride != 0u ? tok * pc.route_stride + slot : tok;
                        const int* src = (const int*)A_q8 +
                            ((size_t)(sb * 2u + (unsigned)ah) * pc.T + src_row) * B_STRIDE;
                        Btile[l] = src[word];
                    } else {
                        Btile[l] = 0;
                    }
                }
            }
            __syncthreads();

            const unsigned row_base = wid * 16u;
            #pragma unroll 1
            for (int ag = 0; ag < 4; ++ag) {
                const unsigned g = (unsigned)ah * 4u + (unsigned)ag;
                const unsigned wr = row_base + (lane & 15u);
                const unsigned wk = (lane >> 4) * 4u;
                const zinc_group_i32x2_t* wp = (const zinc_group_i32x2_t*)(Xtile + wr * X_STRIDE + g * 8u + wk);
                const zinc_group_i32x2_t av0 = wp[0];
                const zinc_group_i32x2_t av1 = wp[1];

                #pragma unroll
                for (unsigned jt = 0; jt < NFRAG; ++jt) {
                    const unsigned at = jt * 16u + (lane & 15u);
                    const unsigned ak = (lane >> 4) * 4u;
                    const zinc_group_i32x2_t* bp = (const zinc_group_i32x2_t*)(Btile + at * B_STRIDE + 4u + (unsigned)ag * 8u + ak);
                    const zinc_group_i32x2_t bv0 = bp[0];
                    const zinc_group_i32x2_t bv1 = bp[1];
                    const float2 dsB = __half22float2(Bds[at * B_STRIDE + (unsigned)ag]);
                    zinc_group_i32x8_t ci = {};
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av0, true, bv0, ci, true);
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av1, true, bv1, ci, true);
                    const int* cv = (const int*)&ci;
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const unsigned ri = row_base + (lane >> 4) * 8u + (unsigned)l;
                        const float2 dmA = __half22float2(Xdm[ri * X_STRIDE + g]);
                        accum[jt * 8u + (unsigned)l] += dmA.x * dsB.x * (float)cv[l];
                        accum[jt * 8u + (unsigned)l] += dmA.y * dsB.y;
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (unsigned frag = 0; frag < NFRAG; ++frag) {
        const unsigned local_t = frag * 16u + (lane & 15u);
        const unsigned packed = order[t0 + local_t];
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const unsigned row = m0 + wid * 16u + (lane >> 4) * 8u + (unsigned)l;
            if (packed != INV && row < pc.M) {
                const unsigned tok = packed >> 16;
                const unsigned slot = packed & 0xFFFFu;
                Y[(size_t)tok * pc.dst_tok_stride + (size_t)slot * pc.M + row] =
                    accum[frag * 8u + (unsigned)l];
            }
        }
    }
}

// Q4_K short-route variant that keeps the weight fragment in registers instead
// of staging the entire 128 x 76-word tile in LDS. The activation tile remains
// shared because every output-row warp consumes it; row scales stay in a small
// shared tile because WMMA accumulator lanes do not own their input-fragment row.
// On gfx12 this cuts LDS from about 41 KiB to 6.25 KiB.
template <unsigned NFRAG>
__device__ __forceinline__ void zinc_gemm_q4k_experts_grouped_i8_direct(
    const unsigned* __restrict__ a_u32,
    const unsigned char* __restrict__ A_q8,
    const unsigned* __restrict__ order,
    const unsigned* __restrict__ tile_expert,
    float* __restrict__ Y,
    GroupedI8Push pc) {
    constexpr unsigned BM = 128u;
    const unsigned BT = NFRAG * 16u;
    const unsigned B_STRIDE = 36u;
    const unsigned INV = 0xFFFFFFFFu;
    __shared__ __align__(16) int Btile[BT * B_STRIDE];
    __shared__ __align__(16) half2 Adm[BM * 8u];

    const unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wid = tid >> 5;
    const unsigned m0 = blockIdx.x * BM;
    const unsigned t0 = blockIdx.y * BT;
    const unsigned bpr = pc.K >> 8;
    const unsigned nsuper = pc.K >> 8;
    const unsigned a0 = (unsigned)(((size_t)expert * pc.expert_stride + pc.base) >> 2);
    half2* const Bds = (half2*)Btile;

    float accum[NFRAG * 8u];
    #pragma unroll
    for (unsigned i = 0; i < NFRAG * 8u; ++i) accum[i] = 0.0f;

    using zinc_direct_i32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using zinc_direct_i32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    const unsigned wr = wid * 16u + (lane & 15u);
    const unsigned row = m0 + wr;
    const unsigned wk = (lane >> 4) * 4u;

    for (unsigned sb = 0; sb < nsuper; ++sb) {
        const unsigned blk = a0 + row * bpr * 36u + sb * 36u;

        // Two lanes per output row expand the eight (scale,-min) pairs. These
        // are consumed according to accumulator-row ownership after WMMA.
        {
            const unsigned sr = wid * 16u + (lane >> 1);
            const unsigned scale_row = m0 + sr;
            const unsigned ksc = lane & 1u;
            if (scale_row < pc.M) {
                const unsigned sblk = a0 + scale_row * bpr * 36u + sb * 36u;
                const half2 dm = *(const half2*)(a_u32 + sblk);
                const int* scale_words = (const int*)(a_u32 + sblk + 1u);
                const int sc32 = ((scale_words[ksc + (ksc != 0u)] >>
                    (4u * (ksc & (ksc / 2u)))) & 0x0F0F0F0F) |
                    ((scale_words[ksc / 2u] >> (2u * (ksc & 1u))) & 0x30303030);
                const unsigned mksc = ksc + 2u;
                const int mn32 = ((scale_words[(mksc & 1u) + (mksc != 0u)] >>
                    (4u * (mksc & (mksc / 2u)))) & 0x0F0F0F0F) |
                    ((scale_words[mksc / 2u] >> (2u * (mksc & 1u))) & 0x30303030);
                const unsigned char* sc8 = (const unsigned char*)&sc32;
                const unsigned char* mn8 = (const unsigned char*)&mn32;
                #pragma unroll
                for (unsigned si = 0u; si < 4u; ++si) {
                    Adm[sr * 8u + ksc * 4u + si] = __hmul2(
                        dm, __floats2half2_rn((float)sc8[si], -(float)mn8[si]));
                }
            } else {
                #pragma unroll
                for (unsigned si = 0u; si < 4u; ++si)
                    Adm[sr * 8u + ksc * 4u + si] = __floats2half2_rn(0.0f, 0.0f);
            }
        }

        #pragma unroll
        for (int ah = 0; ah < 2; ++ah) {
            #pragma unroll
            for (unsigned l0 = 0u; l0 < BT * B_STRIDE; l0 += blockDim.x) {
                const unsigned l = l0 + tid;
                if (l < BT * B_STRIDE) {
                    const unsigned local_t = l / B_STRIDE;
                    const unsigned word = l - local_t * B_STRIDE;
                    const unsigned packed = order[t0 + local_t];
                    if (packed != INV) {
                        const unsigned tok = packed >> 16;
                        const unsigned slot = packed & 0xFFFFu;
                        const unsigned src_row = pc.route_stride != 0u ? tok * pc.route_stride + slot : tok;
                        const int* src = (const int*)A_q8 +
                            ((size_t)(sb * 2u + (unsigned)ah) * pc.T + src_row) * B_STRIDE;
                        Btile[l] = src[word];
                    } else {
                        Btile[l] = 0;
                    }
                }
            }
            __syncthreads();

            #pragma unroll 1
            for (int ag = 0; ag < 4; ++ag) {
                const unsigned g = (unsigned)ah * 4u + (unsigned)ag;
                const unsigned p = g * 8u + wk;
                const unsigned src_word = (p >> 4) * 8u + (p & 7u);
                const bool high = (p & 8u) != 0u;
                unsigned q0 = 0u, q1 = 0u, q2 = 0u, q3 = 0u;
                if (row < pc.M) {
                    q0 = a_u32[blk + 4u + src_word];
                    q1 = a_u32[blk + 5u + src_word];
                    q2 = a_u32[blk + 6u + src_word];
                    q3 = a_u32[blk + 7u + src_word];
                }
                if (high) {
                    q0 >>= 4; q1 >>= 4; q2 >>= 4; q3 >>= 4;
                }
                const zinc_direct_i32x2_t av0 = {
                    (int)(q0 & 0x0F0F0F0Fu), (int)(q1 & 0x0F0F0F0Fu)
                };
                const zinc_direct_i32x2_t av1 = {
                    (int)(q2 & 0x0F0F0F0Fu), (int)(q3 & 0x0F0F0F0Fu)
                };
                #pragma unroll
                for (unsigned jt = 0; jt < NFRAG; ++jt) {
                    const unsigned at = jt * 16u + (lane & 15u);
                    const unsigned ak = (lane >> 4) * 4u;
                    const zinc_direct_i32x2_t* bp = (const zinc_direct_i32x2_t*)(
                        Btile + at * B_STRIDE + 4u + (unsigned)ag * 8u + ak);
                    const zinc_direct_i32x2_t bv0 = bp[0];
                    const zinc_direct_i32x2_t bv1 = bp[1];
                    const float2 dsB = __half22float2(Bds[at * B_STRIDE + (unsigned)ag]);
                    zinc_direct_i32x8_t ci = {};
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av0, true, bv0, ci, true);
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av1, true, bv1, ci, true);
                    const int* cv = (const int*)&ci;
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const unsigned ri = wid * 16u + (lane >> 4) * 8u + (unsigned)l;
                        const float2 dmA = __half22float2(Adm[ri * 8u + g]);
                        accum[jt * 8u + (unsigned)l] += dmA.x * dsB.x * (float)cv[l];
                        accum[jt * 8u + (unsigned)l] += dmA.y * dsB.y;
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (unsigned frag = 0; frag < NFRAG; ++frag) {
        const unsigned local_t = frag * 16u + (lane & 15u);
        const unsigned packed = order[t0 + local_t];
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const unsigned out_row = m0 + wid * 16u + (lane >> 4) * 8u + (unsigned)l;
            if (packed != INV && out_row < pc.M) {
                const unsigned tok = packed >> 16;
                const unsigned slot = packed & 0xFFFFu;
                Y[(size_t)tok * pc.dst_tok_stride + (size_t)slot * pc.M + out_row] =
                    accum[frag * 8u + (unsigned)l];
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_experts_grouped_i8(
    const unsigned* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q45k_experts_grouped_i8<false, 4u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_experts_grouped_i8_direct(
    const unsigned* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q4k_experts_grouped_i8_direct<4u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_experts_grouped_i8_t16(
    const unsigned* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q45k_experts_grouped_i8<false, 1u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_experts_grouped_i8_t16_direct(
    const unsigned* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q4k_experts_grouped_i8_direct<1u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(128, 4) void gemm_q4k_experts_grouped_i8_t16_m64(
    const unsigned* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q45k_experts_grouped_i8<false, 1u, 64u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(64, 8) void gemm_q4k_experts_grouped_i8_t16_m32(
    const unsigned* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q45k_experts_grouped_i8<false, 1u, 32u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5k_experts_grouped_i8(
    const unsigned* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q45k_experts_grouped_i8<true, 4u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5k_experts_grouped_i8_t16(
    const unsigned* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q45k_experts_grouped_i8<true, 1u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(64, 8) void gemm_q5k_experts_grouped_i8_t16_m32(
    const unsigned* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q45k_experts_grouped_i8<true, 1u, 32u>(a, xq, order, tile_expert, y, pc);
}

template <unsigned NFRAG, unsigned BM = 128u>
__device__ __forceinline__ void zinc_gemm_q6k_experts_grouped_i8(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ A_q8,
    const unsigned* __restrict__ order,
    const unsigned* __restrict__ tile_expert,
    float* __restrict__ Y,
    GroupedI8Push pc) {
    const unsigned BT = NFRAG * 16u;
    const unsigned NWARP = BM / 16u;
    const unsigned X_STRIDE = 76u;
    const unsigned B_STRIDE = 36u;
    const unsigned INV = 0xFFFFFFFFu;
    __shared__ __align__(16) int Xtile[BM * X_STRIDE];
    __shared__ __align__(16) int Btile[BT * B_STRIDE];

    const unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wid = tid >> 5;
    const unsigned m0 = blockIdx.x * BM;
    const unsigned t0 = blockIdx.y * BT;
    const unsigned bpr = pc.K >> 8;
    const unsigned nsuper = pc.K >> 8;
    const unsigned char* const a0 = a + (size_t)expert * pc.expert_stride + pc.base;
    float* const Xdf = (float*)(Xtile + 64u);
    int* const Xsc = (int*)(Xdf + 1u);
    half2* const Bds = (half2*)Btile;

    float accum[NFRAG * 8u];
    #pragma unroll
    for (unsigned i = 0; i < NFRAG * 8u; ++i) accum[i] = 0.0f;

    using zinc_group_q6_i32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using zinc_group_q6_i32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    for (unsigned sb = 0; sb < nsuper; ++sb) {
        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += NWARP) {
            const unsigned r = r0 + wid;
            const unsigned row = m0 + r;
            const unsigned kq0 = 2u * lane - (lane & 15u);
            if (row < pc.M) {
                const unsigned char* blk = a0 + ((size_t)row * bpr + sb) * 210u;
                const unsigned short* ql16 = (const unsigned short*)blk;
                const unsigned ql = (unsigned)ql16[2u * lane] |
                    ((unsigned)ql16[2u * lane + 1u] << 16);
                const unsigned qh_idx = 8u * (lane >> 4) + (lane & 7u);
                const unsigned short* qh16 = (const unsigned short*)(blk + 128u);
                const unsigned qh = (unsigned)qh16[2u * qh_idx] |
                    ((unsigned)qh16[2u * qh_idx + 1u] << 16);
                const unsigned shift = (lane & 8u) >> 2;
                const unsigned qh0 = ((qh >> shift) << 4) & 0x30303030u;
                const unsigned qh1 = (qh >> shift) & 0x30303030u;
                Xtile[r * X_STRIDE + kq0] = (int)zinc_q6_sub32x4((ql & 0x0F0F0F0Fu) | qh0);
                Xtile[r * X_STRIDE + kq0 + 16u] = (int)zinc_q6_sub32x4(((ql >> 4) & 0x0F0F0F0Fu) | qh1);
            } else {
                Xtile[r * X_STRIDE + kq0] = 0;
                Xtile[r * X_STRIDE + kq0 + 16u] = 0;
            }
        }

        if (tid < BM) {
            const unsigned row = m0 + tid;
            Xdf[tid * X_STRIDE] = row < pc.M
                ? __half2float(*(const half*)(a0 + ((size_t)row * bpr + sb) * 210u + 208u))
                : 0.0f;
        }

        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += NWARP * 8u) {
            const unsigned r = r0 + wid * 8u + (lane >> 2);
            const unsigned row = m0 + r;
            const unsigned sk = lane & 3u;
            if (row < pc.M) {
                const unsigned char* blk = a0 + ((size_t)row * bpr + sb) * 210u;
                const unsigned short* sc16 = (const unsigned short*)(blk + 192u);
                Xsc[r * X_STRIDE + sk] = (int)((unsigned)sc16[2u * sk] |
                    ((unsigned)sc16[2u * sk + 1u] << 16));
            } else {
                Xsc[r * X_STRIDE + sk] = 0;
            }
        }

        #pragma unroll
        for (int ah = 0; ah < 2; ++ah) {
            #pragma unroll
            for (unsigned l0 = 0u; l0 < BT * B_STRIDE; l0 += blockDim.x) {
                const unsigned l = l0 + tid;
                if (l < BT * B_STRIDE) {
                    const unsigned local_t = l / B_STRIDE;
                    const unsigned word = l - local_t * B_STRIDE;
                    const unsigned packed = order[t0 + local_t];
                    if (packed != INV) {
                        const unsigned tok = packed >> 16;
                        const unsigned slot = packed & 0xFFFFu;
                        const unsigned src_row = pc.route_stride != 0u ? tok * pc.route_stride + slot : tok;
                        const int* src = (const int*)A_q8 +
                            ((size_t)(sb * 2u + (unsigned)ah) * pc.T + src_row) * B_STRIDE;
                        Btile[l] = src[word];
                    } else {
                        Btile[l] = 0;
                    }
                }
            }
            __syncthreads();

            const unsigned row_base = wid * 16u;
            #pragma unroll
            for (int kg = 0; kg < 8; ++kg) {
                const unsigned g16 = (unsigned)ah * 8u + (unsigned)kg;
                const unsigned ag = (unsigned)kg >> 1;
                const unsigned khalf = (unsigned)kg & 1u;
                const unsigned wr = row_base + (lane & 15u);
                const unsigned wk = (lane >> 4) * 2u;
                const zinc_group_q6_i32x2_t av = *(const zinc_group_q6_i32x2_t*)(
                    Xtile + wr * X_STRIDE + g16 * 4u + wk);

                #pragma unroll
                for (unsigned jt = 0; jt < NFRAG; ++jt) {
                    const unsigned at = jt * 16u + (lane & 15u);
                    const unsigned ak = (lane >> 4) * 2u;
                    const zinc_group_q6_i32x2_t bv = *(const zinc_group_q6_i32x2_t*)(Btile +
                        at * B_STRIDE + 4u + ag * 8u + khalf * 4u + ak);
                    const float dB = __half2float(__low2half(Bds[at * B_STRIDE + ag]));
                    zinc_group_q6_i32x8_t ci = {};
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, av, true, bv, ci, false);
                    const int* cv = (const int*)&ci;
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const unsigned ri = row_base + (lane >> 4) * 8u + (unsigned)l;
                        const signed char* scales = (const signed char*)(Xsc + ri * X_STRIDE);
                        accum[jt * 8u + (unsigned)l] +=
                            (float)cv[l] * (float)scales[g16] * Xdf[ri * X_STRIDE] * dB;
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (unsigned frag = 0; frag < NFRAG; ++frag) {
        const unsigned local_t = frag * 16u + (lane & 15u);
        const unsigned packed = order[t0 + local_t];
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const unsigned row = m0 + wid * 16u + (lane >> 4) * 8u + (unsigned)l;
            if (packed != INV && row < pc.M) {
                const unsigned tok = packed >> 16;
                const unsigned slot = packed & 0xFFFFu;
                Y[(size_t)tok * pc.dst_tok_stride + (size_t)slot * pc.M + row] =
                    accum[frag * 8u + (unsigned)l];
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q6k_experts_grouped_i8(
    const unsigned char* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q6k_experts_grouped_i8<4u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q6k_experts_grouped_i8_t16(
    const unsigned char* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q6k_experts_grouped_i8<1u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(64, 8) void gemm_q6k_experts_grouped_i8_t16_m32(
    const unsigned char* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q6k_experts_grouped_i8<1u, 32u>(a, xq, order, tile_expert, y, pc);
}

// Gemma's routed down weights use Q5_1 (32 values per block, affine d/m).
// Accumulate the unsigned 5-bit integer dot with gfx12 WMMA and apply the
// exact affine correction from Q8_1's stored activation sum once per block.
__device__ __forceinline__ unsigned zinc_q51_pack4(
    const unsigned char* __restrict__ blk, unsigned word) {
    const unsigned qh = *(const unsigned*)(blk + 4u);
    const unsigned char* qs = blk + 8u;
    unsigned packed = 0u;
    #pragma unroll
    for (unsigned j = 0u; j < 4u; ++j) {
        const unsigned k = word * 4u + j;
        const unsigned ql = k < 16u ? (unsigned)(qs[k] & 0xFu) : (unsigned)(qs[k - 16u] >> 4);
        packed |= (ql | (((qh >> k) & 1u) << 4)) << (j * 8u);
    }
    return packed;
}

template <unsigned NFRAG>
__device__ __forceinline__ void zinc_gemm_q5_1_experts_grouped_i8(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ A_q8,
    const unsigned* __restrict__ order,
    const unsigned* __restrict__ tile_expert,
    float* __restrict__ Y,
    GroupedI8Push pc) {
    const unsigned BM = 128u, BT = NFRAG * 16u;
    const unsigned X_STRIDE = 40u; // 32 q words + 4 d/m pairs + padding
    const unsigned B_STRIDE = 36u;
    const unsigned INV = 0xFFFFFFFFu;
    __shared__ __align__(16) int Xtile[BM * X_STRIDE];
    __shared__ __align__(16) int Btile[BT * B_STRIDE];

    const unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wid = tid >> 5;
    const unsigned m0 = blockIdx.x * BM;
    const unsigned t0 = blockIdx.y * BT;
    const unsigned blocks_per_row = pc.K >> 5;
    const unsigned nsuper = pc.K >> 7;
    const unsigned char* const a0 = a + (size_t)expert * pc.expert_stride + pc.base;
    half2* const Xdm = (half2*)(Xtile + 32u);
    half2* const Bds = (half2*)Btile;

    float accum[NFRAG * 8u];
    #pragma unroll
    for (unsigned i = 0; i < NFRAG * 8u; ++i) accum[i] = 0.0f;

    using zinc_group_q51_i32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using zinc_group_q51_i32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    for (unsigned sb = 0; sb < nsuper; ++sb) {
        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += 8u) {
            const unsigned r = r0 + wid;
            const unsigned row = m0 + r;
            #pragma unroll
            for (unsigned g = 0u; g < 4u; ++g) {
                if (lane < 8u) {
                    unsigned packed_q = 0u;
                    if (row < pc.M) {
                        const unsigned char* blk = a0 +
                            ((size_t)row * blocks_per_row + sb * 4u + g) * 24u;
                        const unsigned qh = *(const unsigned*)(blk + 4u);
                        const unsigned char* qs = blk + 8u;
                        #pragma unroll
                        for (unsigned j = 0u; j < 4u; ++j) {
                            const unsigned k = lane * 4u + j;
                            const unsigned ql = k < 16u ? (unsigned)(qs[k] & 0xFu) : (unsigned)(qs[k - 16u] >> 4);
                            const unsigned q = ql | (((qh >> k) & 1u) << 4);
                            packed_q |= q << (j * 8u);
                        }
                    }
                    Xtile[r * X_STRIDE + g * 8u + lane] = (int)packed_q;
                }
            }
        }

        // One lane per row copies the four packed (d,m) half2 pairs.
        #pragma unroll
        for (unsigned r0 = 0u; r0 < BM; r0 += 64u) {
            const unsigned r = r0 + wid * 8u + (lane >> 2);
            const unsigned row = m0 + r;
            const unsigned g = lane & 3u;
            if (row < pc.M) {
                const unsigned char* blk = a0 +
                    ((size_t)row * blocks_per_row + sb * 4u + g) * 24u;
                Xdm[r * X_STRIDE + g] = *(const half2*)blk;
            } else {
                Xdm[r * X_STRIDE + g] = __floats2half2_rn(0.0f, 0.0f);
            }
        }

        #pragma unroll
        for (unsigned l0 = 0u; l0 < BT * B_STRIDE; l0 += 256u) {
            const unsigned l = l0 + tid;
            if (l < BT * B_STRIDE) {
                const unsigned local_t = l / B_STRIDE;
                const unsigned word = l - local_t * B_STRIDE;
                const unsigned packed = order[t0 + local_t];
                if (packed != INV) {
                    const unsigned tok = packed >> 16;
                    const unsigned slot = packed & 0xFFFFu;
                    const unsigned src_row = pc.route_stride != 0u ? tok * pc.route_stride + slot : tok;
                    const int* src = (const int*)A_q8 +
                        ((size_t)sb * pc.T + src_row) * B_STRIDE;
                    Btile[l] = src[word];
                } else {
                    Btile[l] = 0;
                }
            }
        }
        __syncthreads();

        const unsigned row_base = wid * 16u;
        #pragma unroll
        for (unsigned g = 0u; g < 4u; ++g) {
            const unsigned wr = row_base + (lane & 15u);
            const unsigned wk = (lane >> 4) * 2u;
            const unsigned at_base = lane & 15u;
            const unsigned ak = (lane >> 4) * 2u;
            #pragma unroll
            for (unsigned jt = 0; jt < NFRAG; ++jt) {
                const unsigned at = jt * 16u + at_base;
                zinc_group_q51_i32x8_t ci = {};
                #pragma unroll
                for (unsigned kh = 0u; kh < 2u; ++kh) {
                    const zinc_group_q51_i32x2_t av = *(const zinc_group_q51_i32x2_t*)(
                        Xtile + wr * X_STRIDE + g * 8u + kh * 4u + wk);
                    const zinc_group_q51_i32x2_t bv = *(const zinc_group_q51_i32x2_t*)(
                        Btile + at * B_STRIDE + 4u + g * 8u + kh * 4u + ak);
                    ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, av, true, bv, ci, true);
                }
                const float2 dsB = __half22float2(Bds[at * B_STRIDE + g]);
                const int* cv = (const int*)&ci;
                #pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const unsigned ri = row_base + (lane >> 4) * 8u + (unsigned)l;
                    const float2 dmA = __half22float2(Xdm[ri * X_STRIDE + g]);
                    accum[jt * 8u + (unsigned)l] += dmA.x * dsB.x * (float)cv[l];
                    accum[jt * 8u + (unsigned)l] += dmA.y * dsB.y;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned frag = 0; frag < NFRAG; ++frag) {
        const unsigned local_t = frag * 16u + (lane & 15u);
        const unsigned packed = order[t0 + local_t];
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const unsigned row = m0 + wid * 16u + (lane >> 4) * 8u + (unsigned)l;
            if (packed != INV && row < pc.M) {
                const unsigned tok = packed >> 16;
                const unsigned slot = packed & 0xFFFFu;
                Y[(size_t)tok * pc.dst_tok_stride + (size_t)slot * pc.M + row] =
                    accum[frag * 8u + (unsigned)l];
            }
        }
    }
}

template <unsigned NFRAG>
__device__ __forceinline__ void zinc_gemm_q5_1_experts_grouped_i8_direct(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ A_q8,
    const unsigned* __restrict__ order,
    const unsigned* __restrict__ tile_expert,
    float* __restrict__ Y,
    GroupedI8Push pc) {
    constexpr unsigned BM = 128u;
    const unsigned BT = NFRAG * 16u;
    const unsigned B_STRIDE = 36u;
    const unsigned INV = 0xFFFFFFFFu;
    __shared__ __align__(16) int Btile[BT * B_STRIDE];
    __shared__ __align__(16) half2 Adm[BM * 4u];

    const unsigned expert = tile_expert[blockIdx.y];
    if (expert == INV) return;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned wid = tid >> 5;
    const unsigned m0 = blockIdx.x * BM;
    const unsigned t0 = blockIdx.y * BT;
    const unsigned blocks_per_row = pc.K >> 5;
    const unsigned nsuper = pc.K >> 7;
    const unsigned char* const a0 = a + (size_t)expert * pc.expert_stride + pc.base;
    half2* const Bds = (half2*)Btile;

    float accum[NFRAG * 8u];
    #pragma unroll
    for (unsigned i = 0u; i < NFRAG * 8u; ++i) accum[i] = 0.0f;

    using zinc_q51_direct_i32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
    using zinc_q51_direct_i32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

    const unsigned wr = wid * 16u + (lane & 15u);
    const unsigned row = m0 + wr;
    const unsigned wk = (lane >> 4) * 2u;

    for (unsigned sb = 0u; sb < nsuper; ++sb) {
        #pragma unroll
        for (unsigned i = tid; i < BM * 4u; i += blockDim.x) {
            const unsigned sr = i >> 2;
            const unsigned g = i & 3u;
            const unsigned scale_row = m0 + sr;
            Adm[i] = scale_row < pc.M
                ? *(const half2*)(a0 + ((size_t)scale_row * blocks_per_row + sb * 4u + g) * 24u)
                : __floats2half2_rn(0.0f, 0.0f);
        }

        #pragma unroll
        for (unsigned l0 = 0u; l0 < BT * B_STRIDE; l0 += blockDim.x) {
            const unsigned l = l0 + tid;
            if (l < BT * B_STRIDE) {
                const unsigned local_t = l / B_STRIDE;
                const unsigned word = l - local_t * B_STRIDE;
                const unsigned packed = order[t0 + local_t];
                if (packed != INV) {
                    const unsigned tok = packed >> 16;
                    const unsigned slot = packed & 0xFFFFu;
                    const unsigned src_row = pc.route_stride != 0u ? tok * pc.route_stride + slot : tok;
                    const int* src = (const int*)A_q8 + ((size_t)sb * pc.T + src_row) * B_STRIDE;
                    Btile[l] = src[word];
                } else {
                    Btile[l] = 0;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (unsigned g = 0u; g < 4u; ++g) {
            zinc_q51_direct_i32x8_t ci_frag[NFRAG];
            #pragma unroll
            for (unsigned jt = 0u; jt < NFRAG; ++jt) ci_frag[jt] = zinc_q51_direct_i32x8_t{};

            #pragma unroll
            for (unsigned kh = 0u; kh < 2u; ++kh) {
                unsigned aq0 = 0u, aq1 = 0u;
                if (row < pc.M) {
                    const unsigned char* blk = a0 +
                        ((size_t)row * blocks_per_row + sb * 4u + g) * 24u;
                    const unsigned qword = kh * 4u + wk;
                    aq0 = zinc_q51_pack4(blk, qword);
                    aq1 = zinc_q51_pack4(blk, qword + 1u);
                }
                const zinc_q51_direct_i32x2_t av = { (int)aq0, (int)aq1 };
                #pragma unroll
                for (unsigned jt = 0u; jt < NFRAG; ++jt) {
                    const unsigned at = jt * 16u + (lane & 15u);
                    const unsigned ak = (lane >> 4) * 2u;
                    const zinc_q51_direct_i32x2_t bv = *(const zinc_q51_direct_i32x2_t*)(
                        Btile + at * B_STRIDE + 4u + g * 8u + kh * 4u + ak);
                    ci_frag[jt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, av, true, bv, ci_frag[jt], true);
                }
            }

            #pragma unroll
            for (unsigned jt = 0u; jt < NFRAG; ++jt) {
                const unsigned at = jt * 16u + (lane & 15u);
                const float2 dsB = __half22float2(Bds[at * B_STRIDE + g]);
                const int* cv = (const int*)&ci_frag[jt];
                #pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const unsigned ri = wid * 16u + (lane >> 4) * 8u + (unsigned)l;
                    const float2 dmA = __half22float2(Adm[ri * 4u + g]);
                    accum[jt * 8u + (unsigned)l] += dmA.x * dsB.x * (float)cv[l];
                    accum[jt * 8u + (unsigned)l] += dmA.y * dsB.y;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned frag = 0u; frag < NFRAG; ++frag) {
        const unsigned packed = order[t0 + frag * 16u + (lane & 15u)];
        #pragma unroll
        for (int l = 0; l < 8; ++l) {
            const unsigned out_row = m0 + wid * 16u + (lane >> 4) * 8u + (unsigned)l;
            if (packed != INV && out_row < pc.M) {
                const unsigned tok = packed >> 16;
                const unsigned slot = packed & 0xFFFFu;
                Y[(size_t)tok * pc.dst_tok_stride + (size_t)slot * pc.M + out_row] =
                    accum[frag * 8u + (unsigned)l];
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5_1_experts_grouped_i8(
    const unsigned char* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q5_1_experts_grouped_i8<4u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5_1_experts_grouped_i8_t16(
    const unsigned char* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q5_1_experts_grouped_i8<1u>(a, xq, order, tile_expert, y, pc);
}

extern "C" __global__ __launch_bounds__(256, 2) void gemm_q5_1_experts_grouped_i8_t16_direct(
    const unsigned char* a, const unsigned char* xq, const unsigned* order,
    const unsigned* tile_expert, float* y, GroupedI8Push pc) {
    zinc_gemm_q5_1_experts_grouped_i8_direct<1u>(a, xq, order, tile_expert, y, pc);
}

#else
extern "C" __global__ void gemm_q4k_experts_grouped_i8() {}
extern "C" __global__ void gemm_q4k_experts_grouped_i8_direct() {}
extern "C" __global__ void gemm_q4k_experts_grouped_i8_t16() {}
extern "C" __global__ void gemm_q4k_experts_grouped_i8_t16_direct() {}
extern "C" __global__ void gemm_q4k_experts_grouped_i8_t16_m64() {}
extern "C" __global__ void gemm_q4k_experts_grouped_i8_t16_m32() {}
extern "C" __global__ void gemm_q5k_experts_grouped_i8() {}
extern "C" __global__ void gemm_q5k_experts_grouped_i8_t16() {}
extern "C" __global__ void gemm_q5k_experts_grouped_i8_t16_m32() {}
extern "C" __global__ void gemm_q6k_experts_grouped_i8() {}
extern "C" __global__ void gemm_q6k_experts_grouped_i8_t16() {}
extern "C" __global__ void gemm_q6k_experts_grouped_i8_t16_m32() {}
extern "C" __global__ void gemm_q5_1_experts_grouped_i8() {}
extern "C" __global__ void gemm_q5_1_experts_grouped_i8_t16() {}
extern "C" __global__ void gemm_q5_1_experts_grouped_i8_t16_direct() {}
#endif

// ---- gemm_q4k_q8_dp4a — Q4_K weights × Q8_0 pre-quantized input, DP4a ------
// Reads pre-quantized Q8_0 activations (from quantize_act_q8_0) instead of
// quantizing float on-the-fly in the K-loop. Saves ~40% per-iteration cost.
struct GemmQ8Push { unsigned M, K, T, a_offset, x_q8_stride; };
extern "C" __global__ __launch_bounds__(256, 4) void gemm_q4k_q8_dp4a(
    const unsigned* __restrict__ a_u32,
    const unsigned char* __restrict__ x_q8,
    float* __restrict__ Y, GemmQ8Push pc)
{
    const unsigned BM=64u, BT=1u;  // One token per block (matvec-style for now)
    unsigned row = blockIdx.x * BM + (threadIdx.x >> 2);  // 64 rows, 4 threads per row
    unsigned t = blockIdx.y;
    unsigned grp = threadIdx.x & 3u;  // 0..3: which group of 4 within BK
    if (row >= pc.M) return;

    unsigned bpr = pc.K >> 8;
    const unsigned char* xq8 = x_q8 + (size_t)t * pc.x_q8_stride;
    float acc = 0.0f;

    // Iterate over K in chunks of 32 (one Q8_0 block per chunk)
    for (unsigned c = 0u; c < (pc.K >> 5); c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u;

        // --- Weight Q4_K ---
        unsigned blk = (pc.a_offset >> 2) + row * bpr * 36u + sbk * 36u;
        unsigned dd = a_u32[blk];
        float d_w = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
        float dm_w = zinc_half_to_float((unsigned short)(dd >> 16));
        const unsigned char* sc = (const unsigned char*)(a_u32 + blk + 1u);
        unsigned char s, mn;
        zinc_q4k_scale_min((int)sb8, sc, &s, &mn);

        // Weight nibbles: 4 packed int8 (from 4-bit nibbles, 0-15)
        const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
        unsigned q_off = (sb8 >> 1) * 32u + grp * 4u;
        unsigned w32 = *((const unsigned*)(qs + q_off));
        int w_pk = (sb8 & 1u) == 0u ? (int)(w32 & 0x0F0F0F0Fu)
                                     : (int)((w32 >> 4) & 0x0F0F0F0Fu);

        // --- Input Q8_0 (pre-quantized) ---
        const unsigned char* ab = xq8 + (size_t)c * 36u;
        float d_a = zinc_half_to_float((unsigned short)((unsigned)ab[0] | ((unsigned)ab[1] << 8)));
        const unsigned char* ap = ab + 4u + grp * 4u;
        int a_pk = (int)ap[0] | ((int)ap[1] << 8) | ((int)ap[2] << 16) | ((int)ap[3] << 24);

        // DP4a: sum of nibble[i] * int8[i]
        int dp = __dp4a(w_pk, a_pk, 0);

        // Accumulate scaled result
        acc += (float)dp * d_w * (float)s * d_a;
    }

    // Reduce 4 partial sums per row via shared memory
    __shared__ float s_row[BM];
    unsigned row_idx = threadIdx.x >> 2;
    unsigned slot = threadIdx.x & 3u;
    // Thread 0 of each group initializes
    if (slot == 0) s_row[row_idx] = 0.0f;
    __syncthreads();
    atomicAdd(&s_row[row_idx], acc);
    __syncthreads();

    if (slot == 0 && t < pc.T) {
        Y[(size_t)t * pc.M + row] = s_row[row_idx];
    }
}

// ---- gemm_q4k_dp4a — DP4a int8 dot product Q4_K GEMM ------------------------
// Key insight: Q4_K nibbles (0-15) are valid unsigned int8 values. Instead of
// dequanting to float (20+ instructions/element), just extract nibbles and pack
// 4 per int32 for __dp4a (1 instruction per 4 multiplies). Input activations
// are quantized to int8 with a fixed scale. Weight scale (d*scale, dmin*min)
// applied post-accumulation.
//
// Math: result = d_sc * sum(nib*in)/scale - dm_mn * sum(in)
//   DP4a computes sum(nib * in_int8) = scale * sum(nib * in)
//   So: result = d_sc * dp4a / scale - dm_mn * sum_input
extern "C" __global__ __launch_bounds__(256, 3) void gemm_q4k_dp4a(const unsigned* a_u32, const float* A,
    const unsigned char* Ab_q8, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ int Ws_pk[BK/4 * BM];       // [8][64] packed nibble int32
    __shared__ int As_pk[BK/4 * BT];       // [8][64] packed int8 input
    __shared__ float W_dsc[BM], W_dmn[BM]; // per-row weight scales
    __shared__ float A_inv[BT];            // per-token max_abs/127 (inverse scale)
    __shared__ float A_sum[BT];            // per-token sum(input)

    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT, bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x, tx=tid&15u, ty=tid>>4;
    unsigned wid=tid>>5, lane=tid&31u;
    unsigned a0=(pc.a_offset>>2);
    const float* Ab=A+(pc.x_offset>>2);
    float acc[4][4]; memset(acc,0,sizeof(acc));

    for (unsigned c=0u; c<nchunk; c++) {
        unsigned sbk=c>>3, sb8=c&7u;
        // W: extract nibbles + store scales (same as before)
        #pragma unroll
        for (int u=0;u<8;u++){
            unsigned r=wid+(unsigned)u*8u, row=m0+r;
            if(row<pc.M){
                unsigned blk=a0+row*bpr*36u+sbk*36u;
                if(lane==0){
                    unsigned dd=a_u32[blk];
                    float d=zinc_half_to_float((unsigned short)(dd&0xFFFFu));
                    float dm=zinc_half_to_float((unsigned short)(dd>>16));
                    const unsigned char* sc=(const unsigned char*)(a_u32+blk+1u);
                    unsigned char s,mn; zinc_q4k_scale_min((int)sb8,sc,&s,&mn);
                    W_dsc[r]=d*(float)s; W_dmn[r]=dm*(float)mn;
                }
                if(lane<8u){
                    const unsigned char* qs=(const unsigned char*)(a_u32+blk+4u);
                    unsigned off=(sb8>>1)*32u+lane*4u;
                    unsigned q32=*((const unsigned*)(qs+off));
                    Ws_pk[lane*BM+r]=(sb8&1u)==0u ? (int)(q32&0x0F0F0F0Fu) : (int)((q32>>4)&0x0F0F0F0Fu);
                }
            } else {
                if(lane<8u) Ws_pk[lane*BM+r]=0;
                if(lane==0){W_dsc[r]=0;W_dmn[r]=0;}
            }
        }
        // A: dynamic per-token quantization with warp-reduce max_abs + sum
        // OR: read pre-quantized Q8_0 if pc.q8_stride > 0
        #pragma unroll
        for (int u=0;u<8;u++){
            unsigned t=wid+(unsigned)u*8u, tok=t0+t;
            if(tok<pc.T){
                if (pc.q8_stride != 0u) {
                    // Pre-quantized Q8_0 path: read scale + packed int8 directly
                    const unsigned char* ab = Ab_q8 + (size_t)tok * pc.q8_stride + (size_t)c * 36u;
                    float d = zinc_half_to_float((unsigned short)((unsigned)ab[0] | ((unsigned)ab[1] << 8)));
                    float sv = 0.0f;
                    if(lane<8u){
                        const unsigned char* p = ab + 4u + lane * 4u;
                        int pk = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16) | ((int)p[3] << 24);
                        As_pk[lane*BT+t] = pk;
                        // Sum of 32 int8 values (each lane sums 4)
                        sv = (float)((signed char)(pk & 0xFF)) + (float)((signed char)((pk >> 8) & 0xFF))
                           + (float)((signed char)((pk >> 16) & 0xFF)) + (float)((signed char)((pk >> 24) & 0xFF));
                    }
                    // Warp-reduce sum across 8 lanes
                    for(int o=4;o>0;o>>=1) sv += __shfl_xor_sync(0xFFFFFFFFu, sv, o);
                    if(lane==0){A_inv[t]=d; A_sum[t]=d*sv;}
                } else {
                    // Dynamic quantization from float input
                    float val=Ab[(size_t)tok*pc.K+c*32u+lane];
                    float av=fabsf(val), sv=val;
                    for(int o=16;o>0;o>>=1){
                        av=fmaxf(av,__shfl_xor_sync(0xFFFFFFFFu,av,o));
                        sv+=__shfl_xor_sync(0xFFFFFFFFu,sv,o);
                    }
                    float scale=127.0f/fmaxf(av,1e-5f);
                    float inv_sc=fmaxf(av,1e-5f)/127.0f;
                    if(lane==0){A_inv[t]=inv_sc; A_sum[t]=sv;}
                    if(lane<8u){
                        float v0=Ab[(size_t)tok*pc.K+c*32u+lane*4u];
                        float v1=Ab[(size_t)tok*pc.K+c*32u+lane*4u+1u];
                        float v2=Ab[(size_t)tok*pc.K+c*32u+lane*4u+2u];
                        float v3=Ab[(size_t)tok*pc.K+c*32u+lane*4u+3u];
                        int pk=0;
                        pk|=(max(-128,min(127,__float2int_rn(v0*scale)))&0xFF);
                        pk|=(max(-128,min(127,__float2int_rn(v1*scale)))&0xFF)<<8;
                        pk|=(max(-128,min(127,__float2int_rn(v2*scale)))&0xFF)<<16;
                        pk|=(max(-128,min(127,__float2int_rn(v3*scale)))&0xFF)<<24;
                        As_pk[lane*BT+t]=pk;
                    }
                }
            } else {
                if(lane<8u) As_pk[lane*BT+t]=0;
                if(lane==0){A_inv[t]=0;A_sum[t]=0;}
            }
        }
        __syncthreads();
        // DP4a multiply
        int ia[4][4]; memset(ia,0,sizeof(ia));
        #pragma unroll
        for(unsigned kk4=0u;kk4<BK/4;kk4++){
            int ar[4];
            #pragma unroll
            for(int j=0;j<4;j++) ar[j]=As_pk[kk4*BT+tx*4u+(unsigned)j];
            #pragma unroll
            for(int i=0;i<4;i++){
                int w=Ws_pk[kk4*BM+ty*4u+(unsigned)i];
                #pragma unroll
                for(int j=0;j<4;j++) ia[i][j]=__dp4a(w,ar[j],ia[i][j]);
            }
        }
        // Apply scales: acc += d_sc * dp4a * inv_sc - dmn * sum
        #pragma unroll
        for(int i=0;i<4;i++){
            float dsc=W_dsc[ty*4u+(unsigned)i], dmn=W_dmn[ty*4u+(unsigned)i];
            #pragma unroll
            for(int j=0;j<4;j++){
                unsigned ti=tx*4u+(unsigned)j;
                acc[i][j]+=dsc*(float)ia[i][j]*A_inv[ti] - dmn*A_sum[ti];
            }
        }
        __syncthreads();
    }
    // Write output
    #pragma unroll
    for(int i=0;i<4;i++){
        unsigned row=m0+ty*4u+(unsigned)i;
        #pragma unroll
        for(int j=0;j<4;j++){
            unsigned tok=t0+tx*4u+(unsigned)j;
            if(row<pc.M&&tok<pc.T){
                unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row;
                if(pc.acc_mode!=0u) Y[yi]+=acc[i][j]; else Y[yi]=acc[i][j];
            }
        }
    }
}
// Mirror of gemm_q4k_tiled_v2 (64x64 tile, 256 threads, 4x4 register micro-tile,
// BK=32) with the Q6_K dequant in the stage-to-shared step. Q6_K block = 210
// bytes/256 elems: ql[0..127] low4, qh[128..191] high2, scales[192..207] int8,
// d[208..209] f16; q = (ql_nibble | (qh_bits<<4)) 6-bit; value = d*sc*(q-32).
// Q6_K is BYTE-addressed -> param `const unsigned char* a`, pc.a_offset is BYTES.
extern "C" __global__ void gemm_q6k_tiled_v2(const unsigned char* a, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float Ws[BK*BM]; __shared__ float As[BK*BT];
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT, bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x, tx=tid&15u, ty=tid>>4;
    const float* Abase=A+(pc.x_offset>>2);
    float acc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j]=0.0f;
    for(unsigned c=0;c<nchunk;c++){
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, r=idx>>5, l=idx&31u, row=m0+r; float wv=0.0f;
            if(row<pc.M){ unsigned e=c*32u+l, within=e&255u, sb=e>>8;
                const unsigned char* blk=a+pc.a_offset+(size_t)row*bpr*210u+(size_t)sb*210u;
                float d=zinc_half_to_float((unsigned short)((unsigned)blk[208]|((unsigned)blk[209]<<8)));
                unsigned half_=within>>7, wh=within&127u, ll=wh&31u, group=wh>>5;
                const unsigned char* ql=blk+(size_t)half_*64u;
                const unsigned char* qh=blk+128u+(size_t)half_*32u;
                const signed char* sc=(const signed char*)(blk+192u+(size_t)half_*8u);
                unsigned is=ll>>4, qhb=qh[ll], q, sci;
                if(group==0u){ q=(ql[ll]&0xFu)|(((qhb>>0)&3u)<<4); sci=is+0u; }
                else if(group==1u){ q=(ql[ll+32u]&0xFu)|(((qhb>>2)&3u)<<4); sci=is+2u; }
                else if(group==2u){ q=(ql[ll]>>4)|(((qhb>>4)&3u)<<4); sci=is+4u; }
                else { q=(ql[ll+32u]>>4)|(((qhb>>6)&3u)<<4); sci=is+6u; }
                wv=d*(float)sc[sci]*((float)q-32.0f); }
            Ws[l*BM+r]=wv; }
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, t=idx>>5, l=idx&31u, tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?Abase[(size_t)tok*pc.K+c*32u+l]:0.0f; }
        __syncthreads();
        #pragma unroll
        for(unsigned kk=0;kk<BK;kk++){ float wr[4],ar[4];
            #pragma unroll
            for(int i=0;i<4;i++) wr[i]=Ws[kk*BM+ty*4u+(unsigned)i];
            #pragma unroll
            for(int j=0;j<4;j++) ar[j]=As[kk*BT+tx*4u+(unsigned)j];
            #pragma unroll
            for(int i=0;i<4;i++)
                #pragma unroll
                for(int j=0;j<4;j++) acc[i][j]+=wr[i]*ar[j]; }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++){ unsigned row=m0+ty*4u+(unsigned)i;
        #pragma unroll
        for(int j=0;j<4;j++){ unsigned tok=t0+tx*4u+(unsigned)j;
            if(row<pc.M&&tok<pc.T){ unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row; if(pc.acc_mode!=0u) Y[yi]+=acc[i][j]; else Y[yi]=acc[i][j]; } } }
}

// ---- gemm_q5k_tiled_v2 — register-blocked prefill GEMM for Q5_K weights ------
// Mirror of gemm_q4k_tiled_v2 with Q5_K dequant in the stage-to-shared step.
// Q5_K block = 176 B/256 elems: d,dmin f16 [0..3]; scales[4..15] (12B); qh[16..47]
// (32B 5th-bit); qs[48..175] (128B). q5 = nib + (qh_bit?16:0); value = d*sc*q5 - dmin*mn.
extern "C" __global__ void gemm_q5k_tiled_v2(const unsigned char* a, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float Ws[BK*BM]; __shared__ float As[BK*BT];
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT, bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x, tx=tid&15u, ty=tid>>4;
    const float* Abase=A+(pc.x_offset>>2);
    float acc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j]=0.0f;
    for(unsigned c=0;c<nchunk;c++){
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, r=idx>>5, l=idx&31u, row=m0+r; float wv=0.0f;
            if(row<pc.M){ unsigned e=c*32u+l, within=e&255u, sb=e>>8;
                const unsigned char* blk=a+pc.a_offset+(size_t)row*bpr*176u+(size_t)sb*176u;
                float d=zinc_half_to_float((unsigned short)((unsigned)blk[0]|((unsigned)blk[1]<<8)));
                float dmin=zinc_half_to_float((unsigned short)((unsigned)blk[2]|((unsigned)blk[3]<<8)));
                const unsigned char* scales=blk+4u; const unsigned char* qh=blk+16u; const unsigned char* qs=blk+48u;
                unsigned chunk=within>>6, half_=(within&63u)>>5, ll=within&31u;
                unsigned char qb=qs[chunk*32u+ll]; unsigned nib=half_==0u?(qb&0xFu):(unsigned)(qb>>4);
                unsigned bit=(qh[ll]>>(2u*chunk+half_))&1u; unsigned q5=nib+(bit?16u:0u);
                unsigned char sc,mn; zinc_q4k_scale_min((int)(chunk*2u+half_),scales,&sc,&mn);
                wv=d*(float)sc*(float)q5 - dmin*(float)mn; }
            Ws[l*BM+r]=wv; }
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, t=idx>>5, l=idx&31u, tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?Abase[(size_t)tok*pc.K+c*32u+l]:0.0f; }
        __syncthreads();
        #pragma unroll
        for(unsigned kk=0;kk<BK;kk++){ float wr[4],ar[4];
            #pragma unroll
            for(int i=0;i<4;i++) wr[i]=Ws[kk*BM+ty*4u+(unsigned)i];
            #pragma unroll
            for(int j=0;j<4;j++) ar[j]=As[kk*BT+tx*4u+(unsigned)j];
            #pragma unroll
            for(int i=0;i<4;i++)
                #pragma unroll
                for(int j=0;j<4;j++) acc[i][j]+=wr[i]*ar[j]; }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++){ unsigned row=m0+ty*4u+(unsigned)i;
        #pragma unroll
        for(int j=0;j<4;j++){ unsigned tok=t0+tx*4u+(unsigned)j;
            if(row<pc.M&&tok<pc.T){ unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row; if(pc.acc_mode!=0u) Y[yi]+=acc[i][j]; else Y[yi]=acc[i][j]; } } }
}

// ---- gemm_q8_0_tiled_v2 — register-blocked prefill GEMM for Q8_0 weights -----
// Mirror of gemm_q4k_tiled_v2 (64x64 tile, 256 threads, 4x4 register micro-tile,
// BK=32) with Q8_0 dequant in the stage-to-shared step. Q8_0 block = 34 B/32
// elems: f16 d [0..1], 32 int8 qs [2..33]; value = d * q (matches dmmv_q8_0's
// d*(float)q so the batched gemma4-MoE shared-expert FFN stays output-identical
// to the per-token path). Byte-addressed -> param `const unsigned char* a`,
// pc.a_offset is BYTES. BK=32 == one Q8_0 block, so chunk c maps to block c.
extern "C" __global__ void gemm_q8_0_tiled_v2(const unsigned char* a, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float Ws[BK*BM]; __shared__ float As[BK*BT];
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT, bpr=pc.K>>5, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x, tx=tid&15u, ty=tid>>4;
    const float* Abase=A+(pc.x_offset>>2);
    float acc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j]=0.0f;
    for(unsigned c=0;c<nchunk;c++){
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, r=idx>>5, l=idx&31u, row=m0+r; float wv=0.0f;
            if(row<pc.M){ const unsigned char* blk=a+pc.a_offset+(size_t)row*bpr*34u+(size_t)c*34u;
                float d=zinc_half_to_float((unsigned short)((unsigned)blk[0]|((unsigned)blk[1]<<8)));
                signed char q=(signed char)blk[2u+l];
                wv=d*(float)q; }
            Ws[l*BM+r]=wv; }
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, t=idx>>5, l=idx&31u, tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?Abase[(size_t)tok*pc.K+c*32u+l]:0.0f; }
        __syncthreads();
        #pragma unroll
        for(unsigned kk=0;kk<BK;kk++){ float wr[4],ar[4];
            #pragma unroll
            for(int i=0;i<4;i++) wr[i]=Ws[kk*BM+ty*4u+(unsigned)i];
            #pragma unroll
            for(int j=0;j<4;j++) ar[j]=As[kk*BT+tx*4u+(unsigned)j];
            #pragma unroll
            for(int i=0;i<4;i++)
                #pragma unroll
                for(int j=0;j<4;j++) acc[i][j]+=wr[i]*ar[j]; }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++){ unsigned row=m0+ty*4u+(unsigned)i;
        #pragma unroll
        for(int j=0;j<4;j++){ unsigned tok=t0+tx*4u+(unsigned)j;
            if(row<pc.M&&tok<pc.T){ unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row; if(pc.acc_mode!=0u) Y[yi]+=acc[i][j]; else Y[yi]=acc[i][j]; } } }
}

// ---- gemm_f32_tiled_v2 — register-blocked prefill GEMM for plain f32 weights --
// Mirror of gemm_q4k_tiled_v2 (64x64 tile, 256 threads, 4x4 register micro-tile,
// BK=32) with NO dequant — the W tile is staged straight from a row-major f32
// weight [M, K] (W[row*K + k]). Same K-tile accumulation as the quant GEMMs, so
// Y[T,M] = A[T,K]·W[M,K]^T is the batched twin of looping dmmv_f32 per token
// (token-correct, not necessarily bit-identical — same class as the quant GEMMs).
// Used by the gemma4-MoE batched router (ffn_gate_inp.weight is f32). pc.a_offset
// is BYTES.
extern "C" __global__ void gemm_f32_tiled_v2(const float* W, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float Ws[BK*BM]; __shared__ float As[BK*BT];
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT, nchunk=(pc.K+31u)>>5;
    unsigned tid=threadIdx.x, tx=tid&15u, ty=tid>>4;
    const float* Wbase=W+(pc.a_offset>>2);
    const float* Abase=A+(pc.x_offset>>2);
    float acc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j]=0.0f;
    for(unsigned c=0;c<nchunk;c++){
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, r=idx>>5, l=idx&31u, row=m0+r, k=c*32u+l;
            Ws[l*BM+r]=(row<pc.M && k<pc.K)?Wbase[(size_t)row*pc.K+k]:0.0f; }
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, t=idx>>5, l=idx&31u, tok=t0+t, k=c*32u+l;
            As[l*BT+t]=(tok<pc.T && k<pc.K)?Abase[(size_t)tok*pc.K+k]:0.0f; }
        __syncthreads();
        #pragma unroll
        for(unsigned kk=0;kk<BK;kk++){ float wr[4],ar[4];
            #pragma unroll
            for(int i=0;i<4;i++) wr[i]=Ws[kk*BM+ty*4u+(unsigned)i];
            #pragma unroll
            for(int j=0;j<4;j++) ar[j]=As[kk*BT+tx*4u+(unsigned)j];
            #pragma unroll
            for(int i=0;i<4;i++)
                #pragma unroll
                for(int j=0;j<4;j++) acc[i][j]+=wr[i]*ar[j]; }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++){ unsigned row=m0+ty*4u+(unsigned)i;
        #pragma unroll
        for(int j=0;j<4;j++){ unsigned tok=t0+tx*4u+(unsigned)j;
            if(row<pc.M&&tok<pc.T){ unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row; if(pc.acc_mode!=0u) Y[yi]+=acc[i][j]; else Y[yi]=acc[i][j]; } } }
}

#ifndef ZINC_ROCM
// ---- gemm_f16_tc — tensor-core GEMM with PRE-DEQUANTED fp16 weights ---------
// Skip Q4_K dequant entirely — weight is already fp16. 10-30x faster than
// gemm_q4k_tc because the dequant (95% of kernel time) is eliminated.
// Requires pre-dequanted weights via ZINC_PRE_DEQUANT cache.
extern "C" __global__ void gemm_f16_tc(const unsigned short* W, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ half Ws[BM*BK];
    __shared__ half As[BK*BT];
    __shared__ float Cs[BT*BM];
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT;
    unsigned nchunk=(pc.K+31u)>>5;
    unsigned tid=threadIdx.x;
    unsigned warp=tid>>5, fm=warp>>2, ft=warp&3u;
    const half* Wbase=(const half*)W + (pc.a_offset>>1);
    const float* Abase=A+(pc.x_offset>>2);

    wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
    wmma::fill_fragment(c0,0.0f); wmma::fill_fragment(c1,0.0f);

    for (unsigned c=0u;c<nchunk;c++){
        // Weight: direct fp16 copy from global (NO DEQUANT)
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u;
            unsigned r=idx>>5,l=idx&31u,row=m0+r,k=c*32u+l;
            Ws[r*BK+l]=(row<pc.M&&k<pc.K)?Wbase[(size_t)row*pc.K+k]:(half)0;
        }
        // Input: f32→fp16
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u;
            unsigned t=idx>>5,l=idx&31u,tok=t0+t,k=c*32u+l;
            As[l*BT+t]=(tok<pc.T&&k<pc.K)?__float2half(Abase[(size_t)tok*pc.K+k]):(half)0;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned ks=0;ks<2;ks++){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f,a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f,&Ws[(fm*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(a1f,&Ws[((fm+2u)*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(bf,&As[(ks*16u)*BT+ft*16u],BT);
            wmma::mma_sync(c0,a0f,bf,c0);
            wmma::mma_sync(c1,a1f,bf,c1);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+fm*16u],c0,BM,wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+(fm+2u)*16u],c1,BM,wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for(int u=0;u<16;u++){
        unsigned idx=tid+(unsigned)u*256u;
        unsigned r=idx%BM,t=idx/BM,row=m0+r,tok=t0+t;
        if(row<pc.M&&tok<pc.T){
            unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row;
            if(pc.acc_mode!=0u) Y[yi]+=Cs[t*BM+r]; else Y[yi]=Cs[t*BM+r];
        }
    }
}

// ---- gemm_q4k_tc — tensor-core (wmma) prefill GEMM for Q4_K weights ----------
// Same Y[T,M] = A[T,K]·W[M,K]^T as gemm_q4k_tiled_v2, but the inner product runs
// on the fp16 tensor cores: the dequant'd W tile and the f32 activations are cast
// to __half in shared memory, then wmma 16x16x16 fragments accumulate in fp32.
// NOT bit-identical to the f32 tiled GEMM (fp16 input rounding) → gated by its own
// token-correctness tolerance gate (validate_catalog), NOT the byte-identical
// GEN_IDS gate. 64x64 output tile, 256 threads = 8 warps, BK=32 (one Q4_K
// sub-block = 2 wmma k-steps). Each warp owns two 16x16 accumulator fragments
// (M-blocks fm and fm+2 at the same T-block ft), so all 4x4 = 16 fragments of the
// tile are covered exactly once. The dequant + A staging mirror gemm_q4k_tiled_v2
// op-for-op (same Q4_K unpack); only the multiply switches to tensor cores.
extern "C" __global__ void gemm_q4k_tc(const unsigned* a_u32, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ half Ws[BM*BK];   // m-major: Ws[r*BK + k]  (matrix_a row-major M×K)
    __shared__ half As[BK*BT];   // k-major: As[k*BT + t]  (matrix_b row-major K×N)
    __shared__ float Cs[BT*BM];  // token-major out tile: Cs[t*BM + m]
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;          // Q4_K superblocks per row (256 elems = 36 u32)
    unsigned nchunk = pc.K >> 5;       // K/32 sub-blocks
    unsigned tid = threadIdx.x;
    unsigned a0 = (pc.a_offset >> 2);
    const float* Abase = A + (pc.x_offset >> 2);
    unsigned warp = tid >> 5;          // 0..7
    unsigned fm = warp >> 2;           // 0..1  (M-block pair base: fm, fm+2)
    unsigned ft = warp & 3u;           // 0..3  (T-block)

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u;
        // dequant W sub-block (64 rows x 32 elems) into Ws (m-major)
        // Optimized: warp-broadcast header reads (same as gemm_q4k_tiled_v2)
        unsigned warp_id = tid >> 5;
        unsigned lane = tid & 31u;
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned r = warp_id + (unsigned)u * 8u;
            unsigned l = lane;
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned blk = a0 + row * bpr * 36u + sbk * 36u;
                float d_sc, dm_mn;
                if (lane == 0) {
                    unsigned dd = a_u32[blk];
                    float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
                    float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
                    const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
                    unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
                    d_sc = d * (float)sc;
                    dm_mn = dmin * (float)mn;
                }
                d_sc = __shfl_sync(0xFFFFFFFFu, d_sc, 0);
                dm_mn = __shfl_sync(0xFFFFFFFFu, dm_mn, 0);
                const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
                unsigned char qb = qs[(sb8 >> 1) * 32u + l];
                unsigned nib = (sb8 & 1u) == 0u ? (qb & 0xFu) : (unsigned)(qb >> 4);
                wv = d_sc * (float)nib - dm_mn;
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        // stage A sub-block (64 tokens x 32 elems) into As (k-major).
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned tok = t0 + t;
            As[l * BT + t] = (tok < pc.T) ? __float2half(Abase[(size_t)tok * pc.K + c * 32u + l]) : __float2half(0.0f);
        }
        __syncthreads();
        // two wmma k-steps over the 32-wide sub-block.
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[(fm * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fm + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
        }
        __syncthreads();
    }
    // store both fragments (out[m][t]) col-major into the token-major Cs tile.
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + fm * 16u], c0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fm + 2u) * 16u], c1, BM, wmma::mem_col_major);
    __syncthreads();
    // guarded copy Cs -> Y[T,M] (16 elems/thread over the 64x64 tile).
    #pragma unroll
    for (int u = 0; u < 16; u++) {
        unsigned idx = tid + (unsigned)u * 256u;   // 0..4095
        unsigned t = idx >> 6, m = idx & 63u;      // token 0..63, row 0..63
        unsigned tok = t0 + t, row = m0 + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * BM + m]; else Y[yi] = Cs[t * BM + m];
        }
    }
}

#endif // !ZINC_ROCM: NVIDIA WMMA dense kernels

// ---- f32_to_f16 — element-wise activation downcast (Effort 24 cycle 12) -------
// y[i] = __float2half(x[i]). Used to pre-convert a GEMM's f32 activation tile to
// fp16 ONCE before gemm_q4k_tc_f16a reads it. The TC GEMM otherwise re-reads the
// f32 activation from global once per output M-block (blockIdx.x) — for a 64x64
// tile that f32 activation traffic is ~7x the Q4_K weight traffic and dominates
// the memory-bound dense GEMM. Pre-converting halves it. Uses the SAME
// __float2half the TC kernel applies in shared, so the staged half bits are
// IDENTICAL → gemm_q4k_tc_f16a's output is byte-for-byte gemm_q4k_tc's.
struct F32ToF16Push { unsigned N; };
extern "C" __global__ void f32_to_f16(const float* x, half* y, F32ToF16Push pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    y[idx] = __float2half(x[idx]);
}

// ---- dequant_q4k_to_f16 (Effort 26 cycle 9: full-weight Q4_K -> fp16) ---------
// Dequant a Q4_K weight W[M,K] (row-major, 256-elem superblocks = 36 u32/row-blk)
// to a dense fp16 buffer Wf16[M,K] (row-major), so the prefill GEMM can run on
// cuBLAS fp16 tensor cores. Same per-element unpack as gemm_q4k_tc's Ws stage
// (d*sc*nib - dmin*mn, then __float2half) → the cuBLAS path's fp16 W bits match
// the hand kernel's staged W bits exactly; only the multiply backend changes.
// One thread per (row,col) element. a_offset is in BYTES (matches GemmPush).
struct DequantQ4KPush { unsigned M, K, a_offset; };
extern "C" __global__ void dequant_q4k_to_f16(const unsigned* a_u32, half* Wf16, DequantQ4KPush pc) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)pc.M * pc.K;
    if (i >= total) return;
    unsigned row = (unsigned)(i / pc.K), k = (unsigned)(i % pc.K);
    unsigned a0 = pc.a_offset >> 2;
    unsigned bpr = pc.K >> 8;          // superblocks per row
    unsigned sb = k >> 8, within = k & 255u, sb8 = within >> 5, l = within & 31u;
    unsigned blk = a0 + row * bpr * 36u + sb * 36u;
    unsigned dd = a_u32[blk];
    float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
    float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
    const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
    const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
    unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
    unsigned char qb = qs[(sb8 >> 1) * 32u + l];
    unsigned nib = (sb8 & 1u) == 0u ? (qb & 0xFu) : (unsigned)(qb >> 4);
    Wf16[i] = __float2half(d * (float)sc * (float)nib - dmin * (float)mn);
}

// ---- dequant_q6k_to_f16 (Effort 26 cycle 10: full-weight Q6_K -> fp16) --------
// Q6_K analog of dequant_q4k_to_f16: dequant a Q6_K weight W[M,K] (row-major,
// 256-elem superblocks = 210 BYTES each) to a dense fp16 buffer Wf16[M,K] so the
// prefill GEMM can run on cuBLAS fp16 tensor cores (the gemma-31b ffn_down is
// Q6_K — ~1/7 of the dense GEMM that still ran on the hand TC kernel). Per-element
// unpack identical to gemm_q6k_tc_f16a / gemm_q6k_tiled_v2 (q = 6-bit, val =
// d*sc*(q-32)), then __float2half → the cuBLAS path's fp16 W bits match the TC
// kernel's staged W bits exactly. Q6_K is BYTE-addressed → `const unsigned char*
// a`, a_offset is BYTES. One thread per (row,col) element.
struct DequantQ6KPush { unsigned M, K, a_offset; };
extern "C" __global__ void dequant_q6k_to_f16(const unsigned char* a, half* Wf16, DequantQ6KPush pc) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)pc.M * pc.K;
    if (i >= total) return;
    unsigned row = (unsigned)(i / pc.K), e = (unsigned)(i % pc.K);
    unsigned bpr = pc.K >> 8;           // Q6_K superblocks per row
    unsigned within = e & 255u, sb = e >> 8;
    const unsigned char* blk = a + pc.a_offset + (size_t)row * bpr * 210u + (size_t)sb * 210u;
    float d = zinc_half_to_float((unsigned short)((unsigned)blk[208] | ((unsigned)blk[209] << 8)));
    unsigned half_ = within >> 7, wh = within & 127u, ll = wh & 31u, group = wh >> 5;
    const unsigned char* ql = blk + (size_t)half_ * 64u;
    const unsigned char* qh = blk + 128u + (size_t)half_ * 32u;
    const signed char* sc = (const signed char*)(blk + 192u + (size_t)half_ * 8u);
    unsigned is = ll >> 4, qhb = qh[ll], q, sci;
    if (group == 0u) { q = (ql[ll] & 0xFu) | (((qhb >> 0) & 3u) << 4); sci = is + 0u; }
    else if (group == 1u) { q = (ql[ll + 32u] & 0xFu) | (((qhb >> 2) & 3u) << 4); sci = is + 2u; }
    else if (group == 2u) { q = (ql[ll] >> 4) | (((qhb >> 4) & 3u) << 4); sci = is + 4u; }
    else { q = (ql[ll + 32u] >> 4) | (((qhb >> 6) & 3u) << 4); sci = is + 6u; }
    Wf16[i] = __float2half(d * (float)sc[sci] * ((float)q - 32.0f));
}

// ---- dequant_q8_0_to_f16 (Effort 29 cycle 5: full-weight Q8_0 -> fp16) --------
// Q8_0 analog of dequant_q4k/q6k_to_f16: dequant a Q8_0 weight W[M,K] (row-major,
// 32-elem blocks = 34 BYTES each: f16 d + 32 int8) to a dense fp16 buffer
// Wf16[M,K] so prefill GEMMs can run on cuBLAS fp16 tensor cores. The qwen36
// MoE's shared-expert gate/up/down are Q8_0 → gemmDispatchPrefill's cuBLAS path
// only fired for Q4_K/Q6_K, so they fell back to per-token matvec (~T launches
// each). Per-element unpack identical to dmmv_q8_0_fast (val = d*(float)qs[e]),
// then __float2half. One thread per (row,col) element. a_offset is BYTES.
struct DequantQ8_0Push { unsigned M, K, a_offset; };
extern "C" __global__ void dequant_q8_0_to_f16(const unsigned char* a, half* Wf16, DequantQ8_0Push pc) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)pc.M * pc.K;
    if (i >= total) return;
    unsigned row = (unsigned)(i / pc.K), k = (unsigned)(i % pc.K);
    unsigned bpr = pc.K >> 5;            // Q8_0 blocks per row
    unsigned bi = k >> 5, within = k & 31u;
    const unsigned char* blk = a + pc.a_offset + (size_t)row * bpr * 34u + (size_t)bi * 34u;
    float d = zinc_half_to_float((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
    const signed char* qs = (const signed char*)(blk + 2);
    Wf16[i] = __float2half(d * (float)qs[within]);
}

// ---- dequant_q5k_to_f16 (Effort 29 cycle 16: full-weight Q5_K -> fp16) --------
// Q5_K analog of dequant_q4k/q6k/q8_0_to_f16: dequant a Q5_K weight W[M,K]
// (row-major, 256-elem superblocks = 176 BYTES each) to a dense fp16 buffer
// Wf16[M,K] so prefill/serving GEMMs run on cuBLAS fp16 tensor cores. The
// dense-path qwen models store attn_qkv (qwen35-9b) and ssm_out (qwen35-9b /
// qwen36-27b) as Q5_K → gemmDispatchPrefill's cuBLAS path only fired for
// Q4_K/Q6_K/Q8_0, so those large dense GEMMs fell back to per-token matvec
// (~T launches each = a launch storm, same shape as the cycle-5 Q8_0 shared
// expert). Per-element unpack identical to dmmv_q5k (5-bit q = nibble +
// (qh_bit<<4); value = d*sc*q5 - dmin*mn, same get_scale_min_k4/zinc_q4k_scale_min
// as Q4_K), then __float2half → the cuBLAS path's fp16 W bits match the matvec's
// effective W exactly. Q5_K is BYTE-addressed → `const unsigned char* a`,
// a_offset is BYTES. One thread per (row,col) element.
struct DequantQ5KPush { unsigned M, K, a_offset; };
extern "C" __global__ void dequant_q5k_to_f16(const unsigned char* a, half* Wf16, DequantQ5KPush pc) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)pc.M * pc.K;
    if (i >= total) return;
    unsigned row = (unsigned)(i / pc.K), k = (unsigned)(i % pc.K);
    unsigned bpr = pc.K >> 8;            // Q5_K superblocks per row
    unsigned within = k & 255u, b = k >> 8;
    const unsigned char* blk = a + pc.a_offset + (size_t)row * bpr * 176u + (size_t)b * 176u;
    float d = zinc_half_to_float((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
    float dmin = zinc_half_to_float((unsigned short)((unsigned)blk[2] | ((unsigned)blk[3] << 8)));
    const unsigned char* scales = blk + 4;     // 12 bytes
    const unsigned char* qh = blk + 16;        // 32 bytes
    const unsigned char* qs = blk + 48;        // 128 bytes
    unsigned chunk = within >> 6;              // 0..3
    unsigned half_ = (within & 63u) >> 5;      // 0..1
    unsigned l = within & 31u;                 // 0..31
    unsigned char ql = qs[chunk * 32u + l];
    unsigned nib = (half_ == 0u) ? (ql & 0xFu) : (unsigned)(ql >> 4);
    unsigned bit = (qh[l] >> (2u * chunk + half_)) & 1u;
    unsigned q5 = nib + (bit ? 16u : 0u);
    unsigned char sc, mn;
    zinc_q4k_scale_min((int)(chunk * 2u + half_), scales, &sc, &mn);
    Wf16[i] = __float2half(d * (float)sc * (float)q5 - dmin * (float)mn);
}

// ---- rms_norm_f16 (Effort 24 cycle 21: emit the fp16 norm DIRECTLY for the TC path) ----
// Byte-for-byte f32_to_f16(rms_norm(x,w)): computes the SAME f32 normalized value
// w[i]*(x[i]*rinv) with the SAME reduction order as rms_norm, then __float2half-stores
// it into a half output. So the fp16-A tensor-core GEMMs (gemm_q4k/q6k_tc_f16a*) read
// the producer's act_f16 directly — the per-GEMM f32->f16 recast launch AND the f32
// b.norm round-trip are dropped on the TC path. One block per token (grid.x = T).
extern "C" __global__ void rms_norm_f16(const float* x, const float* w, half* y, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    half* yt = y + (size_t)token * pc.N;

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        yt[i] = __float2half(w[i] * (xt[i] * rinv));
    }
}

// ---- geglu_f16 (Effort 24 cycle 21: emit the fp16 GeGLU DIRECTLY for the TC down GEMM) ----
// Byte-for-byte f32_to_f16(geglu(gate,up)): same gelu(gate)*up f32 value, __float2half-stored,
// so the ffn_down TC GEMM reads act_f16 with no separate recast launch.
extern "C" __global__ void geglu_f16(const float* gate, const float* up, half* y, SwigluPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    float g = gate[idx];
    float gelu = 0.5f * g * (1.0f + tanhf(k * (g + 0.044715f * g * g * g)));
    y[idx] = __float2half(gelu * up[idx]);
}

#ifndef ZINC_ROCM
// ---- gemm_q4k_tc_f16a — tensor-core Q4_K GEMM with a PRE-CONVERTED fp16 A -----
// Identical to gemm_q4k_tc in every respect (same Q4_K dequant, same wmma 16x16x16
// fragment schedule, same Cs store / guarded copy) EXCEPT the activation A arrives
// already in fp16 (from f32_to_f16) and is staged into shared with no per-load
// __float2half. The staged half bits match gemm_q4k_tc's (which casts the same f32
// with __float2half), so Y is byte-for-byte identical to gemm_q4k_tc — only the
// global activation read traffic is halved (and read once, not once per M-block).
extern "C" __global__ void gemm_q4k_tc_f16a(const unsigned* a_u32, const half* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ half Ws[BM*BK];   // m-major: Ws[r*BK + k]  (matrix_a row-major M×K)
    __shared__ half As[BK*BT];   // k-major: As[k*BT + t]  (matrix_b row-major K×N)
    __shared__ float Cs[BT*BM];  // token-major out tile: Cs[t*BM + m]
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;          // Q4_K superblocks per row (256 elems = 36 u32)
    unsigned nchunk = pc.K >> 5;       // K/32 sub-blocks
    unsigned tid = threadIdx.x;
    unsigned a0 = (pc.a_offset >> 2);
    const half* Abase = A + (pc.x_offset >> 1);   // x_offset in bytes → half elems
    unsigned warp = tid >> 5;          // 0..7
    unsigned fm = warp >> 2;           // 0..1  (M-block pair base: fm, fm+2)
    unsigned ft = warp & 3u;           // 0..3  (T-block)

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u;
        // dequant W sub-block (64 rows x 32 elems) into Ws (m-major) — identical
        // Q4_K unpack to gemm_q4k_tc, then cast to half.
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;   // 0..2047
            unsigned r = idx >> 5, l = idx & 31u;      // row 0..63, elem 0..31
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned blk = a0 + row * bpr * 36u + sbk * 36u;
                unsigned dd = a_u32[blk];
                float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
                float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
                const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
                const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
                unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
                unsigned char qb = qs[(sb8 >> 1) * 32u + l];
                unsigned nib = (sb8 & 1u) == 0u ? (qb & 0xFu) : (unsigned)(qb >> 4);
                wv = d * (float)sc * (float)nib - dmin * (float)mn;
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        // stage A sub-block (64 tokens x 32 elems) into As (k-major) — A is
        // already fp16, so no per-load __float2half (the only change vs gemm_q4k_tc).
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned tok = t0 + t;
            As[l * BT + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + c * 32u + l] : __float2half(0.0f);
        }
        __syncthreads();
        // two wmma k-steps over the 32-wide sub-block.
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[(fm * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fm + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
        }
        __syncthreads();
    }
    // store both fragments (out[m][t]) col-major into the token-major Cs tile.
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + fm * 16u], c0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fm + 2u) * 16u], c1, BM, wmma::mem_col_major);
    __syncthreads();
    // guarded copy Cs -> Y[T,M] (16 elems/thread over the 64x64 tile).
    #pragma unroll
    for (int u = 0; u < 16; u++) {
        unsigned idx = tid + (unsigned)u * 256u;   // 0..4095
        unsigned t = idx >> 6, m = idx & 63u;      // token 0..63, row 0..63
        unsigned tok = t0 + t, row = m0 + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * BM + m]; else Y[yi] = Cs[t * BM + m];
        }
    }
}

// ---- gemm_q6k_tc_f16a — tensor-core Q6_K GEMM with a PRE-CONVERTED fp16 A -------
// Effort 24 cycle 13: the dense gemma-31b carries Q6_K weights (notably ffn_down) as
// well as Q4_K. Cycles 11/12 wired the fp16 tensor cores only for Q4_K (gemmDispatch
// idx 0), so the Q6_K GEMMs still fell back to the f32 register-tiled gemm_q6k_tiled_v2
// even with ZINC_BATCHED_TC on. This kernel extends the proven f16-A TC pattern to
// Q6_K: it is gemm_q4k_tc_f16a in every respect (same wmma 16x16x16 schedule, m-major
// Ws[r*BK+l] half weight tile, k-major fp16 As tile, fp32 accumulate, token-major Cs
// store + guarded copy) EXCEPT the weight sub-block is dequant'd with the Q6_K unpack
// from gemm_q6k_tiled_v2 (210 B/256 elems; q = (ql_nibble | (qh_bits<<4)) 6-bit;
// value = d*sc*(q-32)). Q6_K is BYTE-addressed → `const unsigned char* a`, pc.a_offset
// is BYTES (0 on the batched per-tensor buffer). A arrives fp16 (f32_to_f16, read once).
// NOT bit-identical to the f32 Q6_K path (fp16 rounding) → token-correctness gate, like
// the Q4_K TC kernels; opt-in behind ZINC_BATCHED_TC (toggle off → unchanged f32 path).
extern "C" __global__ void gemm_q6k_tc_f16a(const unsigned char* a, const half* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ half Ws[BM*BK];   // m-major: Ws[r*BK + k]  (matrix_a row-major M×K)
    __shared__ half As[BK*BT];   // k-major: As[k*BT + t]  (matrix_b row-major K×N)
    __shared__ float Cs[BT*BM];  // token-major out tile: Cs[t*BM + m]
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;          // Q6_K superblocks per row (256 elems = 210 bytes)
    unsigned nchunk = pc.K >> 5;       // K/32 sub-blocks
    unsigned tid = threadIdx.x;
    const half* Abase = A + (pc.x_offset >> 1);   // x_offset in bytes → half elems
    unsigned warp = tid >> 5;          // 0..7
    unsigned fm = warp >> 2;           // 0..1  (M-block pair base: fm, fm+2)
    unsigned ft = warp & 3u;           // 0..3  (T-block)

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        // dequant W sub-block (64 rows x 32 elems) into Ws (m-major) — Q6_K unpack
        // identical to gemm_q6k_tiled_v2, then cast to half.
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;   // 0..2047
            unsigned r = idx >> 5, l = idx & 31u;      // row 0..63, elem 0..31
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned e = c*32u + l, within = e & 255u, sb = e >> 8;
                const unsigned char* blk = a + pc.a_offset + (size_t)row*bpr*210u + (size_t)sb*210u;
                float d = zinc_half_to_float((unsigned short)((unsigned)blk[208] | ((unsigned)blk[209]<<8)));
                unsigned half_ = within>>7, wh = within&127u, ll = wh&31u, group = wh>>5;
                const unsigned char* ql = blk + (size_t)half_*64u;
                const unsigned char* qh = blk + 128u + (size_t)half_*32u;
                const signed char* sc = (const signed char*)(blk + 192u + (size_t)half_*8u);
                unsigned is = ll>>4, qhb = qh[ll], q, sci;
                if (group==0u) { q=(ql[ll]&0xFu)|(((qhb>>0)&3u)<<4); sci=is+0u; }
                else if (group==1u) { q=(ql[ll+32u]&0xFu)|(((qhb>>2)&3u)<<4); sci=is+2u; }
                else if (group==2u) { q=(ql[ll]>>4)|(((qhb>>4)&3u)<<4); sci=is+4u; }
                else { q=(ql[ll+32u]>>4)|(((qhb>>6)&3u)<<4); sci=is+6u; }
                wv = d*(float)sc[sci]*((float)q-32.0f);
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        // stage A sub-block (64 tokens x 32 elems) into As (k-major) — A already fp16.
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned tok = t0 + t;
            As[l * BT + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + c * 32u + l] : __float2half(0.0f);
        }
        __syncthreads();
        // two wmma k-steps over the 32-wide sub-block.
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[(fm * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fm + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
        }
        __syncthreads();
    }
    // store both fragments (out[m][t]) col-major into the token-major Cs tile.
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + fm * 16u], c0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fm + 2u) * 16u], c1, BM, wmma::mem_col_major);
    __syncthreads();
    // guarded copy Cs -> Y[T,M] (16 elems/thread over the 64x64 tile).
    #pragma unroll
    for (int u = 0; u < 16; u++) {
        unsigned idx = tid + (unsigned)u * 256u;   // 0..4095
        unsigned t = idx >> 6, m = idx & 63u;      // token 0..63, row 0..63
        unsigned tok = t0 + t, row = m0 + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * BM + m]; else Y[yi] = Cs[t * BM + m];
        }
    }
}

// ---- gemm_q6k_tc_f16a_lowsmem — 8 KB-shared TC Q6_K GEMM (cycle 16) ------------
// gemm_q6k_tc_f16a in every respect (same Q6_K dequant from gemm_q6k_tiled_v2, same
// wmma 16x16x16 fp16 schedule, same fp32 accumulate, pre-converted fp16 A read once)
// EXCEPT it uses only 8 KB of static shared instead of 24 KB — the SAME two-phase-Cs
// occupancy trick cycle 15 applied to the Q4_K TC kernel (gemm_q4k_tc_f16a_lowsmem),
// here extended to the dense gemma-31b ffn_down Q6_K GEMM (idx 2). The 24 KB m64 Q6_K
// kernel's shared is dominated by the 16 KB float Cs output stage (BM*BT*4) → caps
// occupancy at 2 blocks/SM. Here the Cs stage REUSES the (now-dead) Ws+As shared region
// after the K-loop, and the 64x64 output is written to Y in TWO PHASES of 8 fragments
// (c0 = M-tile rows 0..31, c1 = rows 32..63), each phase needing only an 8 KB
// float[BT*32] tile → total static shared = max(Ws+As = 8 KB during K-loop, Cs = 8 KB
// after) = 8 KB → ~3x the m64 occupancy (thread-limited to 8 blocks/SM at 256 thr). The
// wmma math + Q6_K unpack are IDENTICAL to gemm_q6k_tc_f16a → output is byte-for-byte the
// same (each Y element written exactly once across the two phases; the phase split only
// reorders writes, not values; the syncs fence the smem reuse). Q6_K is BYTE-addressed →
// `const unsigned char* a`, pc.a_offset is BYTES. OPT-IN via ZINC_BATCHED_TC_Q6_LOWSMEM —
// perf-neutral (Q6_K is ~1/7 of the dense GEMM, below the boost floor); the proven 24 KB
// m64 kernel (gemm_q6k_tc_f16a) stays the default Q6_K TC path.
extern "C" __global__ void gemm_q6k_tc_f16a_lowsmem(const unsigned char* a, const half* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    // 8 KB shared, aliased: half Ws[BM*BK] (4 KB) + half As[BK*BT] (4 KB) during the
    // K-loop, then float Cs[BT*32] (8 KB) reusing the SAME memory for the output.
    __shared__ float smem[BT*32u];               // 2048 floats = 8192 B
    half* Ws = (half*)smem;                       // bytes [0,4096)   m-major Ws[r*BK + k]
    half* As = ((half*)smem) + (BM*BK);           // bytes [4096,8192) k-major As[k*BT + t]
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;          // Q6_K superblocks per row (256 elems = 210 bytes)
    unsigned nchunk = pc.K >> 5;       // K/32 sub-blocks
    unsigned tid = threadIdx.x;
    const half* Abase = A + (pc.x_offset >> 1);   // x_offset in bytes → half elems
    unsigned warp = tid >> 5;          // 0..7
    unsigned fm = warp >> 2;           // 0..1  (M-block pair base: fm, fm+2)
    unsigned ft = warp & 3u;           // 0..3  (T-block)

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        // dequant W sub-block (64 rows x 32 elems) into Ws (m-major) — Q6_K unpack
        // identical to gemm_q6k_tc_f16a / gemm_q6k_tiled_v2, then cast to half.
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;   // 0..2047
            unsigned r = idx >> 5, l = idx & 31u;      // row 0..63, elem 0..31
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned e = c*32u + l, within = e & 255u, sb = e >> 8;
                const unsigned char* blk = a + pc.a_offset + (size_t)row*bpr*210u + (size_t)sb*210u;
                float d = zinc_half_to_float((unsigned short)((unsigned)blk[208] | ((unsigned)blk[209]<<8)));
                unsigned half_ = within>>7, wh = within&127u, ll = wh&31u, group = wh>>5;
                const unsigned char* ql = blk + (size_t)half_*64u;
                const unsigned char* qh = blk + 128u + (size_t)half_*32u;
                const signed char* sc = (const signed char*)(blk + 192u + (size_t)half_*8u);
                unsigned is = ll>>4, qhb = qh[ll], q, sci;
                if (group==0u) { q=(ql[ll]&0xFu)|(((qhb>>0)&3u)<<4); sci=is+0u; }
                else if (group==1u) { q=(ql[ll+32u]&0xFu)|(((qhb>>2)&3u)<<4); sci=is+2u; }
                else if (group==2u) { q=(ql[ll]>>4)|(((qhb>>4)&3u)<<4); sci=is+4u; }
                else { q=(ql[ll+32u]>>4)|(((qhb>>6)&3u)<<4); sci=is+6u; }
                wv = d*(float)sc[sci]*((float)q-32.0f);
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        // stage A sub-block (64 tokens x 32 elems) into As (k-major) — A already fp16.
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned tok = t0 + t;
            As[l * BT + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + c * 32u + l] : __float2half(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[(fm * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fm + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
        }
        __syncthreads();   // also fences Ws/As reads before the output phases reuse smem
    }
    // Output in two phases reusing the 8 KB tile as float Cs[BT*32] (token-major, m in
    // [0,32) within each half). Phase 1 = c0 (rows 0..31), phase 2 = c1 (rows 32..63).
    // Each Y element written exactly once → byte-identical to gemm_q6k_tc_f16a.
    float* Cs = smem;   // [BT*32] = 2048 floats, m-half token-major: Cs[t*32 + m]
    // ---- phase 1: c0 (fm*16 ∈ {0,16} → rows 0..31) ----
    wmma::store_matrix_sync(&Cs[(ft * 16u) * 32u + fm * 16u], c0, 32u, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        unsigned idx = tid + (unsigned)u * 256u;   // 0..2047
        unsigned t = idx >> 5, m = idx & 31u;      // token 0..63, half-row 0..31
        unsigned tok = t0 + t, row = m0 + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * 32u + m]; else Y[yi] = Cs[t * 32u + m];
        }
    }
    __syncthreads();
    // ---- phase 2: c1 ((fm+2)*16 ∈ {32,48} → rows 32..63) ----
    wmma::store_matrix_sync(&Cs[(ft * 16u) * 32u + fm * 16u], c1, 32u, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        unsigned idx = tid + (unsigned)u * 256u;
        unsigned t = idx >> 5, m = idx & 31u;
        unsigned tok = t0 + t, row = m0 + 32u + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * 32u + m]; else Y[yi] = Cs[t * 32u + m];
        }
    }
}

// ---- gemm_q4k_tc_f16a_m128 — wider 128x64 M-tile TC Q4_K GEMM (cycle 14) -------
// gemm_q4k_tc_f16a in every respect (same Q4_K dequant, same wmma 16x16x16 fp16
// schedule, fp32 accumulate, token-major Cs store + guarded copy, pre-converted
// fp16 A read once) EXCEPT the output M-tile is 128 rows instead of 64, so a block
// covers twice as many output rows per pass. Why: cycle 12 found the f32/fp16 A
// activation re-read is the DOMINANT traffic of this memory-bound GEMM — A is read
// once per output M-block (grid.x = M/BM). Doubling BM to 128 HALVES grid.x → halves
// the dominant A traffic (weight bytes & dequant compute are unchanged: every row is
// still dequant'd grid.y = T/64 times). 256 threads = 8 warps; each warp owns ONE
// 16-token T-block (ft = warp&3) and FOUR 16-row M-blocks (fmbase = warp>>2 selects
// the even {0,2,4,6} or odd {1,3,5,7} M-blocks) → 8 warps x 4 frags = 32 = all
// 8x4 (m,t) 16x16 blocks of the 128x64 tile, covered once. Static shared = 8K Ws +
// 4K As + 32K Cs = 44 KB (< 48 KB). Output byte-for-byte gemm_q4k_tc_f16a's (same
// per-output dequant + wmma math, only the tiling/grid differ — verified identical
// GEN_IDS on a varied prompt). fp16 → token-correct gate, like the other TC kernels.
// NEGATIVE RESULT: the 44 KB static shared caps occupancy at 1 block/SM (vs m64's
// 24 KB → 2 blocks/SM), so the lost latency-hiding outweighs the halved A read on
// this memory-bound GEMM — measured -11.8% on gemma-31b (ABBA x2, 4090). So the TC
// path DEFAULTS to the 64x64 gemm_q4k_tc_f16a; this kernel is kept as a documented
// experiment, opt-in via ZINC_BATCHED_TC_M128.
extern "C" __global__ void gemm_q4k_tc_f16a_m128(const unsigned* a_u32, const half* A, float* Y, GemmPush pc) {
    const unsigned BM=128u, BT=64u, BK=32u;
    __shared__ half Ws[BM*BK];   // m-major: Ws[r*BK + k]  (128 rows x 32)
    __shared__ half As[BK*BT];   // k-major: As[k*BT + t]  (32 x 64 tokens)
    __shared__ float Cs[BT*BM];  // token-major out tile: Cs[t*BM + m]  (64 x 128)
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;          // Q4_K superblocks per row (256 elems = 36 u32)
    unsigned nchunk = pc.K >> 5;       // K/32 sub-blocks
    unsigned tid = threadIdx.x;
    unsigned a0 = (pc.a_offset >> 2);
    const half* Abase = A + (pc.x_offset >> 1);   // x_offset in bytes → half elems
    unsigned warp = tid >> 5;          // 0..7
    unsigned fmbase = warp >> 2;       // 0 → M-blocks {0,2,4,6}, 1 → {1,3,5,7}
    unsigned ft = warp & 3u;           // 0..3  (T-block)

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1, c2, c3;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);
    wmma::fill_fragment(c2, 0.0f);
    wmma::fill_fragment(c3, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u;
        // dequant W sub-block (128 rows x 32 elems) into Ws (m-major) — identical
        // Q4_K unpack to gemm_q4k_tc_f16a, 16 elems/thread for the 128-row tile.
        #pragma unroll
        for (int u = 0; u < 16; u++) {
            unsigned idx = tid + (unsigned)u * 256u;   // 0..4095
            unsigned r = idx >> 5, l = idx & 31u;      // row 0..127, elem 0..31
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned blk = a0 + row * bpr * 36u + sbk * 36u;
                unsigned dd = a_u32[blk];
                float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
                float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
                const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
                const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
                unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
                unsigned char qb = qs[(sb8 >> 1) * 32u + l];
                unsigned nib = (sb8 & 1u) == 0u ? (qb & 0xFu) : (unsigned)(qb >> 4);
                wv = d * (float)sc * (float)nib - dmin * (float)mn;
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        // stage A sub-block (64 tokens x 32 elems) into As (k-major) — A already fp16.
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned tok = t0 + t;
            As[l * BT + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + c * 32u + l] : __float2half(0.0f);
        }
        __syncthreads();
        // two wmma k-steps over the 32-wide sub-block; 4 M-blocks per warp.
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f, a2f, a3f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[((fmbase + 0u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fmbase + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a2f, &Ws[((fmbase + 4u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a3f, &Ws[((fmbase + 6u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
            wmma::mma_sync(c2, a2f, bf, c2);
            wmma::mma_sync(c3, a3f, bf, c3);
        }
        __syncthreads();
    }
    // store the 4 fragments (out[m][t]) col-major into the token-major Cs tile.
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fmbase + 0u) * 16u], c0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fmbase + 2u) * 16u], c1, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fmbase + 4u) * 16u], c2, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft * 16u) * BM + (fmbase + 6u) * 16u], c3, BM, wmma::mem_col_major);
    __syncthreads();
    // guarded copy Cs -> Y[T,M] (32 elems/thread over the 64x128 tile).
    #pragma unroll
    for (int u = 0; u < 32; u++) {
        unsigned idx = tid + (unsigned)u * 256u;   // 0..8191
        unsigned t = idx >> 7, m = idx & 127u;     // token 0..63, row 0..127
        unsigned tok = t0 + t, row = m0 + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * BM + m]; else Y[yi] = Cs[t * BM + m];
        }
    }
}

// ---- gemm_q4k_tc_f16a_lowsmem — 8 KB-shared TC Q4_K GEMM (cycle 15) ------------
// gemm_q4k_tc_f16a in every respect (same Q4_K dequant, same wmma 16x16x16 fp16
// schedule, same fp32 accumulate, pre-converted fp16 A read once) EXCEPT it uses
// only 8 KB of static shared instead of 24 KB. Why: cycle 14's m128 result proved
// this GEMM is OCCUPANCY/latency-bound (44 KB → 1 block/SM lost 11.8%). The proven
// m64 kernel's 24 KB is dominated by the 16 KB float Cs output stage (BM*BT*4) →
// 2 blocks/SM. Here the Cs stage REUSES the (now-dead) Ws+As shared region after
// the K-loop, and the 64x64 output is written to Y in TWO PHASES of 8 fragments
// (c0 = M-tile rows 0..31, c1 = rows 32..63), each phase needing only an 8 KB
// float[BT*32] tile. So total static shared = max(Ws+As = 8 KB during K-loop,
// Cs = 8 KB after) = 8 KB → up to ~6 blocks/SM (thread-limited to 8 at 256 thr),
// ~3x the m64 occupancy → more latency hiding on this memory-bound GEMM. The wmma
// math and the Q4_K unpack are IDENTICAL to gemm_q4k_tc_f16a → output is byte-for-
// byte the same (each Y element is written exactly once across the two phases; the
// phase split only reorders writes, not values). MEASURED +11.6% / +8.9% (two ABBA
// x2 runs, gemma-31b 250-tok, 4090) over the m64 kernel → now the DEFAULT Q4_K TC
// path; ZINC_BATCHED_TC_M64 is the A/B kill-switch back to the 24 KB m64 kernel.
extern "C" __global__ void gemm_q4k_tc_f16a_lowsmem(const unsigned* a_u32, const half* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    // 8 KB shared, aliased: half Ws[BM*BK] (4 KB) + half As[BK*BT] (4 KB) during the
    // K-loop, then float Cs[BT*32] (8 KB) reusing the SAME memory for the output.
    __shared__ float smem[BT*32u];               // 2048 floats = 8192 B
    half* Ws = (half*)smem;                       // bytes [0,4096)   m-major Ws[r*BK + k]
    half* As = ((half*)smem) + (BM*BK);           // bytes [4096,8192) k-major As[k*BT + t]
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;          // Q4_K superblocks per row (256 elems = 36 u32)
    unsigned nchunk = pc.K >> 5;       // K/32 sub-blocks
    unsigned tid = threadIdx.x;
    unsigned a0 = (pc.a_offset >> 2);
    const half* Abase = A + (pc.x_offset >> 1);   // x_offset in bytes → half elems
    unsigned warp = tid >> 5;          // 0..7
    unsigned fm = warp >> 2;           // 0..1  (M-block pair base: fm, fm+2)
    unsigned ft = warp & 3u;           // 0..3  (T-block)

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u;
        // dequant W sub-block (64 rows x 32 elems) into Ws — identical Q4_K unpack.
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;   // 0..2047
            unsigned r = idx >> 5, l = idx & 31u;      // row 0..63, elem 0..31
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned blk = a0 + row * bpr * 36u + sbk * 36u;
                unsigned dd = a_u32[blk];
                float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
                float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
                const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
                const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
                unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
                unsigned char qb = qs[(sb8 >> 1) * 32u + l];
                unsigned nib = (sb8 & 1u) == 0u ? (qb & 0xFu) : (unsigned)(qb >> 4);
                wv = d * (float)sc * (float)nib - dmin * (float)mn;
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        // stage A sub-block (64 tokens x 32 elems) into As (k-major) — A already fp16.
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned tok = t0 + t;
            As[l * BT + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + c * 32u + l] : __float2half(0.0f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[(fm * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fm + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
        }
        __syncthreads();   // also fences Ws/As reads before the output phases reuse smem
    }
    // Output in two phases reusing the 8 KB tile as float Cs[BT*32] (token-major,
    // m in [0,32) within each half). Phase 1 = c0 (M-tile rows 0..31), phase 2 = c1
    // (rows 32..63). Each Y element written exactly once → byte-identical to m64.
    float* Cs = smem;   // [BT*32] = 2048 floats, m-half token-major: Cs[t*32 + m]
    // ---- phase 1: c0 (fm*16 ∈ {0,16} → rows 0..31) ----
    wmma::store_matrix_sync(&Cs[(ft * 16u) * 32u + fm * 16u], c0, 32u, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        unsigned idx = tid + (unsigned)u * 256u;   // 0..2047
        unsigned t = idx >> 5, m = idx & 31u;      // token 0..63, half-row 0..31
        unsigned tok = t0 + t, row = m0 + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * 32u + m]; else Y[yi] = Cs[t * 32u + m];
        }
    }
    __syncthreads();
    // ---- phase 2: c1 ((fm+2)*16 ∈ {32,48} → rows 32..63) ----
    wmma::store_matrix_sync(&Cs[(ft * 16u) * 32u + fm * 16u], c1, 32u, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        unsigned idx = tid + (unsigned)u * 256u;
        unsigned t = idx >> 5, m = idx & 31u;
        unsigned tok = t0 + t, row = m0 + 32u + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * 32u + m]; else Y[yi] = Cs[t * 32u + m];
        }
    }
}

#endif // !ZINC_ROCM: NVIDIA WMMA/inline-PTX kernels before q4k low-memory GEMM

// ---- gemm_q4k_tc_lowsmem — 16KB-shared TC Q4_K GEMM (50% better occupancy) ---
// Aliases Ws[4KB half] + As[4KB half] within a 16KB float smem that doubles as
// Cs[BT*BM] output buffer after the K-loop completes. 3 blocks/SM vs 2 for the
// 24KB gemm_q4k_tc. Output uses stride BM=64 (same as gemm_q4k_tc — no column
// overflow issues that plagued the 8KB two-phase attempt).
extern "C" __global__ void gemm_q4k_tc_lowsmem(const unsigned* a_u32, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    // 16KB shared: Ws+As occupy first 8KB during K-loop; full 16KB reused as Cs for output
    __shared__ float smem[BT*BM];                    // 4096 floats = 16384 B
    half* Ws = (half*)smem;                           // bytes [0, 4096)
    half* As = ((half*)smem) + (BM*BK);               // bytes [4096, 8192)
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT;
    unsigned bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x;
    unsigned warp=tid>>5, fm=warp>>2, ft=warp&3u;
    unsigned a0=(pc.a_offset>>2);
    const float* Abase=A+(pc.x_offset>>2);

    wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
    wmma::fill_fragment(c0,0.0f); wmma::fill_fragment(c1,0.0f);

    for (unsigned c=0u;c<nchunk;c++){
        unsigned sbk=c>>3, sb8=c&7u;
        // Warp-broadcast dequant: lane 0 reads header, broadcasts d*scale + dmin*min
        unsigned wid=tid>>5, lane=tid&31u;
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned r=wid+(unsigned)u*8u, l=lane, row=m0+r;
            float wv=0.0f;
            if(row<pc.M){
                unsigned blk=a0+row*bpr*36u+sbk*36u;
                float d_sc,dm_mn;
                if(lane==0){
                    unsigned dd=a_u32[blk];
                    float d=zinc_half_to_float((unsigned short)(dd&0xFFFFu));
                    float dmin=zinc_half_to_float((unsigned short)(dd>>16));
                    const unsigned char* sc=(const unsigned char*)(a_u32+blk+1u);
                    unsigned char s,mn; zinc_q4k_scale_min((int)sb8,sc,&s,&mn);
                    d_sc=d*(float)s; dm_mn=dmin*(float)mn;
                }
                d_sc=__shfl_sync(0xFFFFFFFFu,d_sc,0);
                dm_mn=__shfl_sync(0xFFFFFFFFu,dm_mn,0);
                const unsigned char* qs=(const unsigned char*)(a_u32+blk+4u);
                unsigned char qb=qs[(sb8>>1)*32u+l];
                unsigned nib=(sb8&1u)==0u?(qb&0xFu):(unsigned)(qb>>4);
                wv=d_sc*(float)nib-dm_mn;
            }
            Ws[r*BK+l]=__float2half(wv);
        }
        // Float→half A load
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u;
            unsigned t=idx>>5,l=idx&31u,tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?__float2half(Abase[(size_t)tok*pc.K+c*32u+l]):(half)0;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned ks=0;ks<2;ks++){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f,a1f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f,&Ws[(fm*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(a1f,&Ws[((fm+2u)*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(bf,&As[(ks*16u)*BT+ft*16u],BT);
            wmma::mma_sync(c0,a0f,bf,c0);
            wmma::mma_sync(c1,a1f,bf,c1);
        }
        __syncthreads();
    }
    // Output: store to Cs (= smem) with stride BM=64, then copy to Y.
    // Same pattern as gemm_q4k_tc — single phase, all 64 rows at once.
    float* Cs = smem;
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+fm*16u], c0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+(fm+2u)*16u], c1, BM, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for(int u=0;u<16;u++){
        unsigned idx=tid+(unsigned)u*256u;
        unsigned r=idx%BM, t=idx/BM;
        unsigned row=m0+r, tok=t0+t;
        if(row<pc.M&&tok<pc.T){
            unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row;
            if(pc.acc_mode!=0u) Y[yi]+=Cs[t*BM+r]; else Y[yi]=Cs[t*BM+r];
        }
    }
}

#ifndef ZINC_ROCM
// ---- Inline PTX mma.sync.m16n8k16 helpers (bypass broken wmma fp32 path) -----
// NVRTC's wmma::mma_sync generates .f32.f32 (fp32) MMA instead of .f32.f16
// (fp16). This macro uses inline PTX to force the fp16 MMA instruction.
// Uses "+f" (read-write) constraints so D and C share registers (accumulate).
// MUST be a macro, not a function — nvcc generates wrong code when float&
// output aliases float input params in __forceinline__ functions.
#define mma_m16n8k16(d0,d1,d2,d3,a0,a1,a2,a3,b0,b1) \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))

// Load A[16,16] fp16 row-major fragment using ldmatrix.x4 (matches mma.sync layout).
// smem layout: Ws[row * stride + col]. Each thread provides one row address.
__device__ __forceinline__ void load_a_frag(
    unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3,
    const half* smem, unsigned base, unsigned stride) {
    const unsigned lane = threadIdx.x & 31u;
    // ldmatrix.x4: 4 matrices of 8×8. Threads 0-7: matrix0 rows, 8-15: matrix1, etc.
    // Matrix layout for [16,16] tile:
    //   mat0 = [0:8, 0:8], mat1 = [0:8, 8:16], mat2 = [8:16, 0:8], mat3 = [8:16, 8:16]
    unsigned row = lane % 8u;
    unsigned mat = lane / 8u;
    unsigned k_row = row + (mat / 2u) * 8u;  // 0-7 or 8-15
    unsigned k_col = (mat % 2u) * 8u;         // 0 or 8
    const half* ptr = smem + base + k_row * stride + k_col;
    unsigned addr = (unsigned)__cvta_generic_to_shared(ptr);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}

// Load B[16,8] fp16 fragment using ldmatrix.x2.trans (handles col-major transposition).
// smem layout: As[k * stride + n]. The .trans flag transposes row-major → col-major
// for the mma.sync .col B operand. Each thread provides one row address.
__device__ __forceinline__ void load_b_frag_ldm(
    unsigned& b0, unsigned& b1,
    const half* smem, unsigned base, unsigned stride) {
    const unsigned lane = threadIdx.x & 31u;
    // ldmatrix.x2.trans: threads 0-15 provide row addresses for 2 8×8 matrices.
    // Matrix 0 = K-rows 0-7, Matrix 1 = K-rows 8-15, both reading N-cols 0-7.
    const half* row_ptr = smem + base + (lane & 15u) * stride;
    unsigned addr = (unsigned)__cvta_generic_to_shared(row_ptr);
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b0), "=r"(b1) : "r"(addr));
}

// ---- gemm_q4k_mma_lowsmem — inline-PTX fp16 MMA Q4_K GEMM (fixes wmma bug) ----
// Same structure as gemm_q4k_tc_lowsmem but uses mma.sync.m16n8k16.f32.f16.f16.f32
// via inline PTX instead of wmma::mma_sync (which generates fp32 MMA on NVRTC).
extern "C" __global__ void gemm_q4k_mma_lowsmem(const unsigned* a_u32, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float smem[BT*BM];
    half* Ws = (half*)smem;
    half* As = ((half*)smem) + (BM*BK);
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT;
    unsigned bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x;
    unsigned warp=tid>>5, fm=warp>>2, ft=warp&3u;
    unsigned a0_base=(pc.a_offset>>2);
    const float* Abase=A+(pc.x_offset>>2);
    const unsigned lane = tid & 31u;

    // Accumulators: 4 mma outputs per warp (2 M-halves × 2 N-halves)
    float cL0a=0,cL1a=0,cL2a=0,cL3a=0; // first M-half (rows fm*16), left N
    float cR0a=0,cR1a=0,cR2a=0,cR3a=0; // first M-half, right N
    float cL0b=0,cL1b=0,cL2b=0,cL3b=0; // second M-half (rows (fm+2)*16), left N
    float cR0b=0,cR1b=0,cR2b=0,cR3b=0; // second M-half, right N

    for (unsigned c=0u;c<nchunk;c++){
        unsigned sbk=c>>3, sb8=c&7u;
        // Warp-broadcast dequant (same as lowsmem)
        unsigned wid=tid>>5, wl=tid&31u;
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned r=wid+(unsigned)u*8u, l=wl, row=m0+r;
            float wv=0.0f;
            if(row<pc.M){
                unsigned blk=a0_base+row*bpr*36u+sbk*36u;
                float d_sc,dm_mn;
                if(l==0){
                    unsigned dd=a_u32[blk];
                    float d=zinc_half_to_float((unsigned short)(dd&0xFFFFu));
                    float dmin=zinc_half_to_float((unsigned short)(dd>>16));
                    const unsigned char* sc=(const unsigned char*)(a_u32+blk+1u);
                    unsigned char s,mn; zinc_q4k_scale_min((int)sb8,sc,&s,&mn);
                    d_sc=d*(float)s; dm_mn=dmin*(float)mn;
                }
                d_sc=__shfl_sync(0xFFFFFFFFu,d_sc,0);
                dm_mn=__shfl_sync(0xFFFFFFFFu,dm_mn,0);
                const unsigned char* qs=(const unsigned char*)(a_u32+blk+4u);
                unsigned char qb=qs[(sb8>>1)*32u+l];
                unsigned nib=(sb8&1u)==0u?(qb&0xFu):(unsigned)(qb>>4);
                wv=d_sc*(float)nib-dm_mn;
            }
            Ws[r*BK+l]=__float2half(wv);
        }
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u;
            unsigned t=idx>>5,l=idx&31u,tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?__float2half(Abase[(size_t)tok*pc.K+c*32u+l]):(half)0;
        }
        __syncthreads();
        // MMA: 2 K-sub-blocks per BK=32 chunk. Each sub-block: 2 A loads + 2 B loads + 4 mma
        #pragma unroll
        for(unsigned ks=0;ks<2;ks++){
            unsigned a0r,a1r,a2r,a3r, b0r,b1r,b2r,b3r;
            load_a_frag(a0r,a1r,a2r,a3r, Ws,(fm*16u)*BK+ks*16u,BK);
            load_a_frag(b0r,b1r,b2r,b3r, Ws,((fm+2u)*16u)*BK+ks*16u,BK);
            unsigned bL0,bL1,bR0,bR1;
            load_b_frag_ldm(bL0,bL1,As,(ks*16u)*BT+ft*16u,BT);
            load_b_frag_ldm(bR0,bR1,As,(ks*16u)*BT+ft*16u+8u,BT);
            mma_m16n8k16(cL0a,cL1a,cL2a,cL3a,a0r,a1r,a2r,a3r,bL0,bL1);
            mma_m16n8k16(cR0a,cR1a,cR2a,cR3a,a0r,a1r,a2r,a3r,bR0,bR1);
            mma_m16n8k16(cL0b,cL1b,cL2b,cL3b,b0r,b1r,b2r,b3r,bL0,bL1);
            mma_m16n8k16(cR0b,cR1b,cR2b,cR3b,b0r,b1r,b2r,b3r,bR0,bR1);
        }
        __syncthreads();
    }
    // Store: each thread holds D[row][col], D[row][col+1], D[row+8][col], D[row+8][col+1]
    // for both M-halves and both N-halves.
    {
        // First M-half (rows fm*16..fm*16+15)
        unsigned row0 = m0 + fm*16u + (lane>>2);
        unsigned row1 = row0 + 8u;
        unsigned col0 = t0 + ft*16u + (lane&3u)*2u;
        unsigned col1 = col0 + 1u;
        if(row0<pc.M&&col0<pc.T) Y[(pc.y_offset>>2)+(size_t)col0*pc.M+row0]=cL0a;
        if(row0<pc.M&&col1<pc.T) Y[(pc.y_offset>>2)+(size_t)col1*pc.M+row0]=cL1a;
        if(row1<pc.M&&col0<pc.T) Y[(pc.y_offset>>2)+(size_t)col0*pc.M+row1]=cL2a;
        if(row1<pc.M&&col1<pc.T) Y[(pc.y_offset>>2)+(size_t)col1*pc.M+row1]=cL3a;
        col0 += 8u; col1 += 8u;
        if(row0<pc.M&&col0<pc.T) Y[(pc.y_offset>>2)+(size_t)col0*pc.M+row0]=cR0a;
        if(row0<pc.M&&col1<pc.T) Y[(pc.y_offset>>2)+(size_t)col1*pc.M+row0]=cR1a;
        if(row1<pc.M&&col0<pc.T) Y[(pc.y_offset>>2)+(size_t)col0*pc.M+row1]=cR2a;
        if(row1<pc.M&&col1<pc.T) Y[(pc.y_offset>>2)+(size_t)col1*pc.M+row1]=cR3a;
        // Second M-half (rows (fm+2)*16..(fm+2)*16+15)
        col0 -= 8u; col1 -= 8u;
        row0 = m0 + (fm+2u)*16u + (lane>>2);
        row1 = row0 + 8u;
        if(row0<pc.M&&col0<pc.T) Y[(pc.y_offset>>2)+(size_t)col0*pc.M+row0]=cL0b;
        if(row0<pc.M&&col1<pc.T) Y[(pc.y_offset>>2)+(size_t)col1*pc.M+row0]=cL1b;
        if(row1<pc.M&&col0<pc.T) Y[(pc.y_offset>>2)+(size_t)col0*pc.M+row1]=cL2b;
        if(row1<pc.M&&col1<pc.T) Y[(pc.y_offset>>2)+(size_t)col1*pc.M+row1]=cL3b;
        col0 += 8u; col1 += 8u;
        if(row0<pc.M&&col0<pc.T) Y[(pc.y_offset>>2)+(size_t)col0*pc.M+row0]=cR0b;
        if(row0<pc.M&&col1<pc.T) Y[(pc.y_offset>>2)+(size_t)col1*pc.M+row0]=cR1b;
        if(row1<pc.M&&col0<pc.T) Y[(pc.y_offset>>2)+(size_t)col0*pc.M+row1]=cR2b;
        if(row1<pc.M&&col1<pc.T) Y[(pc.y_offset>>2)+(size_t)col1*pc.M+row1]=cR3b;
    }
}

// ---- gemm_q4k_gate_up_swiglu_lowsmem — fused gate+up+SwiGLU TC Q4_K GEMM ------
// Reads the activation input ONCE, multiplies by both gate and up Q4_K weight
// matrices, applies SwiGLU (silu(gate)*up) inline. Eliminates 2 intermediate
// buffers and 2 kernel launches vs the unfused path. Same 16KB alias pattern as
// gemm_q4k_tc_lowsmem, but the K-loop uses 12KB (3 half-tiles: gate_Ws, up_Ws, As).
struct GateUpSwigluPush { unsigned M, K, T, gate_a_offset, up_a_offset, x_offset, y_offset; };
extern "C" __global__ __launch_bounds__(256, 2) void gemm_q4k_gate_up_swiglu_lowsmem(
    const unsigned* __restrict__ gate_u32,
    const unsigned* __restrict__ up_u32,
    const float* __restrict__ A,
    float* __restrict__ Y, GateUpSwigluPush pc)
{
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float smem[BT*BM];                    // 16 KB
    half* gate_Ws = (half*)smem;                      // [0, 4 KB)
    half* up_Ws   = ((half*)smem) + (BM*BK);          // [4, 8 KB)
    half* As      = ((half*)smem) + (2u*BM*BK);       // [8, 12 KB)
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT;
    unsigned bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x;
    unsigned warp=tid>>5, fm=warp>>2, ft=warp&3u;
    unsigned wid=tid>>5, lane=tid&31u;
    const float* Abase=A+(pc.x_offset>>2);
    unsigned gate_a0=(pc.gate_a_offset>>2);
    unsigned up_a0  =(pc.up_a_offset>>2);

    wmma::fragment<wmma::accumulator,16,16,16,float> cg0,cg1,cu0,cu1;
    wmma::fill_fragment(cg0,0.0f); wmma::fill_fragment(cg1,0.0f);
    wmma::fill_fragment(cu0,0.0f); wmma::fill_fragment(cu1,0.0f);

    for (unsigned c=0u;c<nchunk;c++){
        unsigned sbk=c>>3, sb8=c&7u;
        // Stage gate weights
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned r=wid+(unsigned)u*8u, l=lane, row=m0+r;
            float gv=0.0f, uv=0.0f;
            if(row<pc.M){
                // Gate
                unsigned gblk=gate_a0+row*bpr*36u+sbk*36u;
                float gd_sc,gdm_mn;
                if(lane==0){
                    unsigned dd=gate_u32[gblk];
                    float d=zinc_half_to_float((unsigned short)(dd&0xFFFFu));
                    float dmin=zinc_half_to_float((unsigned short)(dd>>16));
                    const unsigned char* sc=(const unsigned char*)(gate_u32+gblk+1u);
                    unsigned char s,mn; zinc_q4k_scale_min((int)sb8,sc,&s,&mn);
                    gd_sc=d*(float)s; gdm_mn=dmin*(float)mn;
                }
                gd_sc=__shfl_sync(0xFFFFFFFFu,gd_sc,0);
                gdm_mn=__shfl_sync(0xFFFFFFFFu,gdm_mn,0);
                const unsigned char* gqs=(const unsigned char*)(gate_u32+gblk+4u);
                unsigned gqb=gqs[(sb8>>1)*32u+l];
                unsigned gnib=(sb8&1u)==0u?(gqb&0xFu):(unsigned)(gqb>>4);
                gv=gd_sc*(float)gnib-gdm_mn;
                // Up
                unsigned ublk=up_a0+row*bpr*36u+sbk*36u;
                float ud_sc,udm_mn;
                if(lane==0){
                    unsigned dd=up_u32[ublk];
                    float d=zinc_half_to_float((unsigned short)(dd&0xFFFFu));
                    float dmin=zinc_half_to_float((unsigned short)(dd>>16));
                    const unsigned char* sc=(const unsigned char*)(up_u32+ublk+1u);
                    unsigned char s,mn; zinc_q4k_scale_min((int)sb8,sc,&s,&mn);
                    ud_sc=d*(float)s; udm_mn=dmin*(float)mn;
                }
                ud_sc=__shfl_sync(0xFFFFFFFFu,ud_sc,0);
                udm_mn=__shfl_sync(0xFFFFFFFFu,udm_mn,0);
                const unsigned char* uqs=(const unsigned char*)(up_u32+ublk+4u);
                unsigned uqb=uqs[(sb8>>1)*32u+l];
                unsigned unib=(sb8&1u)==0u?(uqb&0xFu):(unsigned)(uqb>>4);
                uv=ud_sc*(float)unib-udm_mn;
            }
            gate_Ws[r*BK+l]=__float2half(gv);
            up_Ws[r*BK+l]=__float2half(uv);
        }
        // Float->half A load (read input ONCE for both gate and up)
        #pragma unroll
        for(int u=0;u<8;u++){
            unsigned idx=tid+(unsigned)u*256u;
            unsigned t=idx>>5,l=idx&31u,tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?__float2half(Abase[(size_t)tok*pc.K+c*32u+l]):(half)0;
        }
        __syncthreads();
        #pragma unroll
        for(unsigned ks=0;ks<2;ks++){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> ag0,ag1,au0,au1;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(ag0,&gate_Ws[(fm*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(ag1,&gate_Ws[((fm+2u)*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(au0,&up_Ws[(fm*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(au1,&up_Ws[((fm+2u)*16u)*BK+ks*16u],BK);
            wmma::load_matrix_sync(bf,&As[(ks*16u)*BT+ft*16u],BT);
            wmma::mma_sync(cg0,ag0,bf,cg0);
            wmma::mma_sync(cg1,ag1,bf,cg1);
            wmma::mma_sync(cu0,au0,bf,cu0);
            wmma::mma_sync(cu1,au1,bf,cu1);
        }
        __syncthreads();
    }
    // Apply SwiGLU in fragment registers: result = silu(gate) * up
    #pragma unroll
    for(int i=0;i<cg0.num_elements;i++){
        float g=cg0.x[i], u=cu0.x[i];
        cg0.x[i]=g/(1.0f+expf(-g))*u;
    }
    #pragma unroll
    for(int i=0;i<cg1.num_elements;i++){
        float g=cg1.x[i], u=cu1.x[i];
        cg1.x[i]=g/(1.0f+expf(-g))*u;
    }
    // Store to smem, then copy to Y
    float* Cs = smem;
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+fm*16u], cg0, BM, wmma::mem_col_major);
    wmma::store_matrix_sync(&Cs[(ft*16u)*BM+(fm+2u)*16u], cg1, BM, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for(int u=0;u<16;u++){
        unsigned idx=tid+(unsigned)u*256u;
        unsigned r=idx%BM, t=idx/BM;
        unsigned row=m0+r, tok=t0+t;
        if(row<pc.M&&tok<pc.T){
            Y[(pc.y_offset>>2)+(size_t)tok*pc.M+row]=Cs[t*BM+r];
        }
    }
}

// ---- gemm_q4k_tc_f16a_m128_lowsmem — 12 KB-shared wider 128x64 M-tile TC Q4_K GEMM (cycle 17)
// The SYNTHESIS of cycle 14 (wider M-tile) and cycle 15 (low-shared two-phase Cs):
// it is gemm_q4k_tc_f16a_m128 in every wmma respect (same Q4_K dequant, same 16x16x16
// fp16 schedule, fp32 accumulate, BM=128 → grid.x = M/128 so the dominant f16-A read is
// HALVED vs the 64x64 default) EXCEPT the 128x64 output is NOT held in a 32 KB float
// Cs[BT*BM] tile. Instead the Cs stage REUSES the (now-dead) Ws+As shared region after
// the K-loop and the tile is written to Y in FOUR PHASES of two 16-row M-blocks each
// (phase p = rows 32p..32p+31 = even-group frag c_p ∪ odd-group frag c_p), each phase
// needing only an 8 KB float[BT*32] tile. So static shared = max(Ws 8 KB + As 4 KB
// during the K-loop, Cs 8 KB after) = 12 KB — vs the m128 kernel's 44 KB. Cycle 14
// proved m128 was -11.8% ONLY because its 44 KB capped occupancy at 1 block/SM; at 12 KB
// (256 thr) occupancy is thread/register-limited (~6 blocks/SM, the SAME as the m64
// lowsmem default), so the halved A read should now pay off instead of being eaten by
// lost latency hiding. Output is BYTE-FOR-BYTE the m128 / m64 / lowsmem kernels' (same
// per-output dequant + wmma math; the four phases only REORDER writes — each Y element
// is written exactly once; syncs fence the smem reuse). fp16 → token-correct gate, like
// the other TC kernels. Opt-in via ZINC_BATCHED_TC_M128_LOWSMEM until measured.
extern "C" __global__ void gemm_q4k_tc_f16a_m128_lowsmem(const unsigned* a_u32, const half* A, float* Y, GemmPush pc) {
    const unsigned BM=128u, BT=64u, BK=32u;
    // 12 KB shared (3072 floats), aliased: half Ws[BM*BK] (8 KB) + half As[BK*BT] (4 KB)
    // during the K-loop, then float Cs[BT*32] (8 KB) reusing the SAME memory after it.
    __shared__ float smem[BM*BK/2u + BK*BT/2u];   // 3072 floats = 12288 B
    half* Ws = (half*)smem;                        // bytes [0,8192)     m-major Ws[r*BK + k]
    half* As = ((half*)smem) + (BM*BK);            // bytes [8192,12288) k-major As[k*BT + t]
    unsigned m0 = blockIdx.x*BM, t0 = blockIdx.y*BT;
    unsigned bpr = pc.K >> 8;          // Q4_K superblocks per row (256 elems = 36 u32)
    unsigned nchunk = pc.K >> 5;       // K/32 sub-blocks
    unsigned tid = threadIdx.x;
    unsigned a0 = (pc.a_offset >> 2);
    const half* Abase = A + (pc.x_offset >> 1);   // x_offset in bytes → half elems
    unsigned warp = tid >> 5;          // 0..7
    unsigned fmbase = warp >> 2;       // 0 → M-blocks {0,2,4,6}, 1 → {1,3,5,7}
    unsigned ft = warp & 3u;           // 0..3  (T-block)

    wmma::fragment<wmma::accumulator,16,16,16,float> c0, c1, c2, c3;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);
    wmma::fill_fragment(c2, 0.0f);
    wmma::fill_fragment(c3, 0.0f);

    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u;
        // dequant W sub-block (128 rows x 32 elems) into Ws — identical Q4_K unpack.
        #pragma unroll
        for (int u = 0; u < 16; u++) {
            unsigned idx = tid + (unsigned)u * 256u;   // 0..4095
            unsigned r = idx >> 5, l = idx & 31u;      // row 0..127, elem 0..31
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned blk = a0 + row * bpr * 36u + sbk * 36u;
                unsigned dd = a_u32[blk];
                float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
                float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
                const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
                const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
                unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
                unsigned char qb = qs[(sb8 >> 1) * 32u + l];
                unsigned nib = (sb8 & 1u) == 0u ? (qb & 0xFu) : (unsigned)(qb >> 4);
                wv = d * (float)sc * (float)nib - dmin * (float)mn;
            }
            Ws[r * BK + l] = __float2half(wv);
        }
        // stage A sub-block (64 tokens x 32 elems) into As (k-major) — A already fp16.
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned tok = t0 + t;
            As[l * BT + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + c * 32u + l] : __float2half(0.0f);
        }
        __syncthreads();
        // two wmma k-steps over the 32-wide sub-block; 4 M-blocks per warp.
        #pragma unroll
        for (unsigned ks = 0; ks < 2; ks++) {
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a0f, a1f, a2f, a3f;
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
            wmma::load_matrix_sync(a0f, &Ws[((fmbase + 0u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a1f, &Ws[((fmbase + 2u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a2f, &Ws[((fmbase + 4u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(a3f, &Ws[((fmbase + 6u) * 16u) * BK + ks * 16u], BK);
            wmma::load_matrix_sync(bf, &As[(ks * 16u) * BT + ft * 16u], BT);
            wmma::mma_sync(c0, a0f, bf, c0);
            wmma::mma_sync(c1, a1f, bf, c1);
            wmma::mma_sync(c2, a2f, bf, c2);
            wmma::mma_sync(c3, a3f, bf, c3);
        }
        __syncthreads();   // also fences Ws/As reads before the output phases reuse smem
    }
    // Output in FOUR phases reusing the 12 KB region as float Cs[BT*32] (8 KB). Phase p
    // stores fragment c_p of BOTH warp groups: the even group (fmbase=0) → Cs local rows
    // 0..31 of M-block 2p (global rows 32p..32p+15), the odd group (fmbase=1) → Cs local
    // rows 16..31 of M-block 2p+1 (global rows 32p+16..32p+31). Each Y element written
    // exactly once across the four phases → byte-identical to the m128/m64 kernels.
    float* Cs = smem;   // [BT*32] token-major: Cs[t*32 + m], m-half in [0,32)
    // ---- phase 0: c0 (M-blocks 0/1 → global rows 0..31) ----
    wmma::store_matrix_sync(&Cs[(ft * 16u) * 32u + fmbase * 16u], c0, 32u, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        unsigned idx = tid + (unsigned)u * 256u;   // 0..2047
        unsigned t = idx >> 5, m = idx & 31u;      // token 0..63, half-row 0..31
        unsigned tok = t0 + t, row = m0 + 0u + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * 32u + m]; else Y[yi] = Cs[t * 32u + m];
        }
    }
    __syncthreads();
    // ---- phase 1: c1 (M-blocks 2/3 → global rows 32..63) ----
    wmma::store_matrix_sync(&Cs[(ft * 16u) * 32u + fmbase * 16u], c1, 32u, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        unsigned idx = tid + (unsigned)u * 256u;
        unsigned t = idx >> 5, m = idx & 31u;
        unsigned tok = t0 + t, row = m0 + 32u + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * 32u + m]; else Y[yi] = Cs[t * 32u + m];
        }
    }
    __syncthreads();
    // ---- phase 2: c2 (M-blocks 4/5 → global rows 64..95) ----
    wmma::store_matrix_sync(&Cs[(ft * 16u) * 32u + fmbase * 16u], c2, 32u, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        unsigned idx = tid + (unsigned)u * 256u;
        unsigned t = idx >> 5, m = idx & 31u;
        unsigned tok = t0 + t, row = m0 + 64u + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * 32u + m]; else Y[yi] = Cs[t * 32u + m];
        }
    }
    __syncthreads();
    // ---- phase 3: c3 (M-blocks 6/7 → global rows 96..127) ----
    wmma::store_matrix_sync(&Cs[(ft * 16u) * 32u + fmbase * 16u], c3, 32u, wmma::mem_col_major);
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        unsigned idx = tid + (unsigned)u * 256u;
        unsigned t = idx >> 5, m = idx & 31u;
        unsigned tok = t0 + t, row = m0 + 96u + m;
        if (row < pc.M && tok < pc.T) {
            unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
            if (pc.acc_mode != 0u) Y[yi] += Cs[t * 32u + m]; else Y[yi] = Cs[t * 32u + m];
        }
    }
}

#endif // !ZINC_ROCM: NVIDIA WMMA/inline-PTX kernels

#ifdef ZINC_ROCM
// The ROCm correctness milestone runs the scalar/matvec path. Forward state
// still resolves every CUDA-era pipeline at initialization, so keep inert
// symbols for NVIDIA-only kernels until native gfx12 WMMA/MFMA ports replace
// them. Backend defaults guarantee these symbols are never dispatched.
#define ZINC_ROCM_PIPELINE_STUB(name) extern "C" __global__ void name() {}
ZINC_ROCM_PIPELINE_STUB(gemm_q4k_experts_grouped_tc)
ZINC_ROCM_PIPELINE_STUB(gemm_q5_1_experts_grouped_tc)
ZINC_ROCM_PIPELINE_STUB(gemm_q5k_experts_grouped_tc)
ZINC_ROCM_PIPELINE_STUB(gemm_q6k_experts_grouped_tc)
ZINC_ROCM_PIPELINE_STUB(gemm_f16_tc)
ZINC_ROCM_PIPELINE_STUB(gemm_q4k_tc)
ZINC_ROCM_PIPELINE_STUB(gemm_q4k_tc_f16a)
ZINC_ROCM_PIPELINE_STUB(gemm_q6k_tc_f16a)
ZINC_ROCM_PIPELINE_STUB(gemm_q6k_tc_f16a_lowsmem)
ZINC_ROCM_PIPELINE_STUB(gemm_q4k_tc_f16a_m128)
ZINC_ROCM_PIPELINE_STUB(gemm_q4k_tc_f16a_lowsmem)
ZINC_ROCM_PIPELINE_STUB(gemm_q4k_mma_lowsmem)
ZINC_ROCM_PIPELINE_STUB(gemm_q4k_gate_up_swiglu_lowsmem)
ZINC_ROCM_PIPELINE_STUB(gemm_q4k_tc_f16a_m128_lowsmem)
#undef ZINC_ROCM_PIPELINE_STUB
#endif

// ---- sigmoid_mul (qwen35 attention gate) — out[i] = a[i] * sigmoid(gate[i]) ---
// ABI: inputs first, output last (matches swiglu). In-place safe (out may alias a).
struct SigmoidMulPush { unsigned N; };
extern "C" __global__ void sigmoid_mul(const float* a, const float* gate, float* out, SigmoidMulPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    float g = gate[idx];
    out[idx] = a[idx] * (1.0f / (1.0f + expf(-g)));
}

// Attention gate + Q8_1 packing for the single-token ROCm O projection. The
// unfused path first overwrites the f32 attention row with a*sigmoid(gate), then
// reads the row again in quantize_act_q8_0. O only consumes the packed view, so
// compute the gated values in registers and write Q8_1 directly.
extern "C" __global__ void sigmoid_mul_quant_q8_0(
    const float* __restrict__ a, const float* __restrict__ gate,
    unsigned char* __restrict__ out, QuantActPush pc)
{
    const unsigned tok = blockIdx.y;
    const unsigned chunk_base = blockIdx.x * 256u;
    const unsigned tid = threadIdx.x;
    const unsigned k_idx = chunk_base + tid;
    const unsigned wblk = tid >> 5;
    const unsigned wlane = tid & 31u;
    const float* at = a + (size_t)tok * pc.K;
    const float* gt = gate + (size_t)tok * pc.K;
    float val = 0.0f;
    if (k_idx < pc.K) {
        const float g = gt[k_idx];
        val = at[k_idx] * (1.0f / (1.0f + expf(-g)));
    }

    float av = fabsf(val), sv = val;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
        sv += __shfl_xor_sync(0xffffffffu, sv, o);
    }
    const float d = av / 127.0f;
    const float scale = 127.0f / fmaxf(av, 1e-5f);
    const int q = max(-127, min(127, __float2int_rn(val * scale)));

#ifdef ZINC_ROCM
    const unsigned c = (chunk_base >> 5) + wblk;
    const unsigned h = c >> 2;
    const unsigned g = c & 3u;
    unsigned char* out_base = out + ((size_t)h * pc.T + tok) * 144u;
    if (wlane == 0u && k_idx < pc.K) {
        const unsigned short dh = zinc_float_to_half(d);
        const unsigned short sh = zinc_float_to_half(sv);
        out_base[g * 4u] = (unsigned char)(dh & 0xffu);
        out_base[g * 4u + 1u] = (unsigned char)(dh >> 8);
        out_base[g * 4u + 2u] = (unsigned char)(sh & 0xffu);
        out_base[g * 4u + 3u] = (unsigned char)(sh >> 8);
    }
    out_base[16u + g * 32u + wlane] = (unsigned char)q;
#else
    unsigned char* out_base = out + (size_t)tok * (pc.K / 32u) * 36u +
        ((size_t)(chunk_base >> 5) + wblk) * 36u;
    if (wlane == 0u && k_idx < pc.K) {
        const unsigned short dh = zinc_float_to_half(d);
        const unsigned short sh = zinc_float_to_half(sv);
        out_base[0] = (unsigned char)(dh & 0xffu);
        out_base[1] = (unsigned char)(dh >> 8);
        out_base[2] = (unsigned char)(sh & 0xffu);
        out_base[3] = (unsigned char)(sh >> 8);
    }
    out_base[4u + wlane] = (unsigned char)q;
#endif
}

// ===========================================================================
// Gemma 4 kernels (additive — never used by the qwen35/qwen36 path).
// ===========================================================================

// ---- rms_norm_noweight (gemma V normalization) -----------------------------
// y = x / sqrt(mean(x^2) + eps). Pure RMS normalize with NO learnable weight
// (gemma4 applies ggml_rms_norm to V per head before the KV write). One block
// per row of pc.N elements (grid = n_kv_heads, N = head_dim).
extern "C" __global__ void rms_norm_noweight(const float* x, float* y, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* yt = y + (size_t)token * pc.N;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x)
        yt[i] = xt[i] * rinv;
}

// ---- rms_norm_triple (gemma MoE: 3 pre-norms off the SAME hidden → 1 launch) -
// In the gemma-26b MoE decode block, `self.hidden` is rms-normalized THREE times
// per layer over the identical Σhidden² reduction, only the trailing weight
// differs: ffn_norm (shared expert), no-weight (router), pre_ffw_norm_2 (routed
// experts). hidden is unchanged until the combine tail, so all three share one
// reduction. This fuses them: ONE block reduces Σx² once, then emits
//   y1[i] = w1[i]*(x[i]*rinv)   (== rms_norm(x, w1))
//   y2[i] =        x[i]*rinv    (== rms_norm_noweight(x))
//   y3[i] = w3[i]*(x[i]*rinv)   (== rms_norm(x, w3))
// Byte-identical to the three originals (same ss / rsqrtf(ss/N+eps) / left-to-
// right multiply); removes 2 launches AND 2 redundant full hidden reads +
// reductions per layer. Single block (grid {1,1,1}) like the originals — no
// in-flight block count is lost (the C8-class block-count-preserving fusion).
extern "C" __global__ void rms_norm_triple(const float* x, const float* w1, const float* w3,
                                           float* y1, float* y2, float* y3, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float xr = xt[i] * rinv;
        y1[(size_t)token * pc.N + i] = w1[i] * xr;
        y2[(size_t)token * pc.N + i] = xr;
        y3[(size_t)token * pc.N + i] = w3[i] * xr;
    }
}

// ---- rms_norm_kvwrite (gemma: fuse per-head V rms_norm-noweight + V KV write) -
// One block per KV head. Plain-normalizes V over head_dim (no weight, like
// rms_norm_noweight) and writes the result STRAIGHT into the V cache at
// dst_offset (= position*kv_dim), per head at dst_offset + head*head_dim.
// Replaces the rms_norm_noweight(→v_buf) + the V half of kv_cache_write pair on
// the gemma attention path — bit-equivalent (identical normalization, identical
// destination layout), one launch instead of two, no v_buf round-trip.
struct RmsKvWritePush { unsigned head_dim; float eps; unsigned dst_offset; };
extern "C" __global__ void rms_norm_kvwrite(const float* v_src, float* v_dst, RmsKvWritePush pc) {
    unsigned head = blockIdx.x;
    unsigned hd = pc.head_dim;
    const float* vh = v_src + (size_t)head * hd;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = vh[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    size_t base = (size_t)pc.dst_offset + (size_t)head * hd;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        v_dst[base + i] = vh[i] * rinv;
}

// ---- rms_norm_rope_batched (Effort 24: batched-prefill twin of rms_norm_rope) -
// Identical per-(head,token) math to rms_norm_rope, batched over T prompt
// queries: block=(head=blockIdx.x, t=blockIdx.y). Token t sits at sequence
// position base_position+t; src/dst token strides are explicit (q_dim for Q
// in-place, kv_dim for K writing straight into the KV cache). Replaces the T
// per-token rms_norm_rope launches with ONE launch (grid.y=T); per-block
// reduction order is unchanged, so the result is bit-identical.
struct RmsRopeBatchPush { unsigned head_dim; float eps; unsigned rope_dim; unsigned base_position; unsigned src_stride; unsigned dst_stride; };
extern "C" __global__ void rms_norm_rope_batched(const float* x, const float* w, const float* inv_freq, float* y, RmsRopeBatchPush pc) {
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;
    const float* xt = x + (size_t)t * pc.src_stride + (size_t)head * pc.head_dim;
    float* yt = y + (size_t)t * pc.dst_stride + (size_t)head * pc.head_dim;
    extern __shared__ float sh[]; // pc.head_dim normalized values

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.head_dim; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.head_dim + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < pc.head_dim; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    unsigned half_rot = pc.rope_dim >> 1;
    unsigned position = pc.base_position + t;
    for (unsigned i = threadIdx.x; i < half_rot; i += blockDim.x) {
        float xi = sh[i];
        float xih = sh[i + half_rot];
        float theta = (float)position * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i] = xi * ct - xih * st;
        yt[i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < pc.head_dim; i += blockDim.x)
        yt[i] = sh[i];
}

// ---- rms_norm_kvwrite_batched (Effort 24: batched twin of rms_norm_kvwrite) ---
// Batched over T prompt tokens: block=(head=blockIdx.x, t=blockIdx.y). Per-token
// V source/dest strides are explicit (kv_dim both ways: token-major [T,kv_dim]
// source, position-major [seq,n_kv_head,head_dim] dest). One launch (grid.y=T)
// replaces the T per-token rms_norm_kvwrite launches; bit-identical math.
struct RmsKvWriteBatchPush { unsigned head_dim; float eps; unsigned src_stride; unsigned dst_stride; };
extern "C" __global__ void rms_norm_kvwrite_batched(const float* v_src, float* v_dst, RmsKvWriteBatchPush pc) {
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;
    unsigned hd = pc.head_dim;
    const float* vh = v_src + (size_t)t * pc.src_stride + (size_t)head * hd;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = vh[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    size_t base = (size_t)t * pc.dst_stride + (size_t)head * hd;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        v_dst[base + i] = vh[i] * rinv;
}

// ---- rms_norm_rope_qkv (gemma: fuse the per-head V/Q/K norm launches) -------
// Collapses the THREE per-head norm launches on the gemma attention path into
// ONE: the V plain-normalize+KV-write (rms_norm_kvwrite), the Q norm+rope, and
// the K norm+rope (both rms_norm_rope). Grid is n_head + 2*n_kv_head blocks:
//   block <  n_head                       -> Q head: weighted norm + rope -> q_out (offset 0)
//   block <  n_head + n_kv_head           -> K head: weighted norm + rope -> k_out at kv_offset
//   else                                  -> V head: plain norm (no weight, no rope) -> v_out at kv_offset
// Each branch's arithmetic is COPIED verbatim from the standalone kernels
// (Q/K from rms_norm_rope, V from rms_norm_kvwrite) so the fused result is
// bit-identical. No cross-block hazard: K writes kv_k (never k_in), V writes
// kv_v (never v_in), Q writes q_out in-place per head — no block reads a buffer
// another block writes. Removes 2 tiny launch boundaries/layer on the gemma
// attention path (dense + MoE).
struct RmsRopeQkvPush {
    unsigned head_dim; float eps; unsigned rope_dim; unsigned position;
    unsigned n_head; unsigned n_kv_head; unsigned kv_offset;
};
extern "C" __global__ void rms_norm_rope_qkv(
    const float* q_in, const float* k_in, const float* v_in,
    const float* wq, const float* wk, const float* inv_freq,
    float* q_out, float* k_out, float* v_out, RmsRopeQkvPush pc)
{
    unsigned bx = blockIdx.x;
    unsigned hd = pc.head_dim;
    extern __shared__ float sh[]; // hd normalized values (Q/K rope staging)

    const float* xt;
    float* yt;
    const float* w;
    bool do_rope;
    if (bx < pc.n_head) {
        unsigned head = bx;
        xt = q_in + (size_t)head * hd;
        yt = q_out + (size_t)head * hd;                 // dst_offset 0
        w = wq; do_rope = true;
    } else if (bx < pc.n_head + pc.n_kv_head) {
        unsigned head = bx - pc.n_head;
        xt = k_in + (size_t)head * hd;
        yt = k_out + (size_t)pc.kv_offset + (size_t)head * hd;
        w = wk; do_rope = true;
    } else {
        unsigned head = bx - pc.n_head - pc.n_kv_head;
        xt = v_in + (size_t)head * hd;
        yt = v_out + (size_t)pc.kv_offset + (size_t)head * hd;
        w = nullptr; do_rope = false;
    }

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    if (!do_rope) { // V: plain normalize, no weight (matches rms_norm_kvwrite)
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
            yt[i] = xt[i] * rinv;
        return;
    }

    // Q/K: weighted norm into shared, then NEOX partial rope (matches rms_norm_rope)
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    unsigned half_rot = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < half_rot; i += blockDim.x) {
        float xi = sh[i];
        float xih = sh[i + half_rot];
        float theta = (float)pc.position * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i] = xi * ct - xih * st;
        yt[i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < hd; i += blockDim.x)
        yt[i] = sh[i];
}

// ---- rms_norm_rope_qkv_seq (Effort 28 1c: batched DECODE per-seq norm/rope) --
// Batched twin of rms_norm_rope_qkv: collapses the per-row decode loop into ONE
// launch over B independent sequences. grid = (n_head + 2*n_kv_head, B). Row b is
// sequence b at its OWN position positions[b] writing its OWN KV slot slots[b].
//   q_in/k_in/v_in are token-major [B, q_dim] / [B, kv_dim] (this step's activations)
//   q_out is token-major [B, q_dim] (in place); k_out/v_out are the slot KV buffers
//   [n_slots*slot_ctx, kv_dim] — row b writes at (slot*slot_ctx + pos)*kv_dim.
// Per-(head,b) block arithmetic is COPIED verbatim from rms_norm_rope_qkv (with
// pc.position -> positions[b]), so it is bit-identical to the per-row launches.
// No cross-block hazard: K writes slot KV, V writes slot KV, Q writes q_out per
// (b,head) — disjoint regions; q_in==q_out is consumed into shared before write.
struct RmsRopeQkvSeqPush {
    unsigned head_dim; float eps; unsigned rope_dim;
    unsigned n_head; unsigned n_kv_head; unsigned slot_ctx;
};
extern "C" __global__ void rms_norm_rope_qkv_seq(
    const float* q_in, const float* k_in, const float* v_in,
    const float* wq, const float* wk, const float* inv_freq,
    float* q_out, float* k_out, float* v_out,
    const unsigned* positions, const unsigned* slots, RmsRopeQkvSeqPush pc)
{
    unsigned bx = blockIdx.x;
    unsigned b  = blockIdx.y;
    unsigned hd = pc.head_dim;
    unsigned q_dim  = pc.n_head * hd;
    unsigned kv_dim = pc.n_kv_head * hd;
    unsigned pos  = positions[b];
    unsigned slot = slots[b];
    size_t kv_base = ((size_t)slot * pc.slot_ctx + pos) * kv_dim;  // slot KV write pos
    extern __shared__ float sh[]; // hd normalized values (Q/K rope staging)

    const float* xt;
    float* yt;
    const float* w;
    bool do_rope;
    if (bx < pc.n_head) {
        unsigned head = bx;
        xt = q_in + (size_t)b * q_dim + (size_t)head * hd;
        yt = q_out + (size_t)b * q_dim + (size_t)head * hd;
        w = wq; do_rope = true;
    } else if (bx < pc.n_head + pc.n_kv_head) {
        unsigned head = bx - pc.n_head;
        xt = k_in + (size_t)b * kv_dim + (size_t)head * hd;
        yt = k_out + kv_base + (size_t)head * hd;
        w = wk; do_rope = true;
    } else {
        unsigned head = bx - pc.n_head - pc.n_kv_head;
        xt = v_in + (size_t)b * kv_dim + (size_t)head * hd;
        yt = v_out + kv_base + (size_t)head * hd;
        w = nullptr; do_rope = false;
    }

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    if (!do_rope) { // V: plain normalize, no weight (matches rms_norm_kvwrite)
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
            yt[i] = xt[i] * rinv;
        return;
    }

    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    unsigned half_rot = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < half_rot; i += blockDim.x) {
        float xi = sh[i];
        float xih = sh[i + half_rot];
        float theta = (float)pos * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i] = xi * ct - xih * st;
        yt[i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < hd; i += blockDim.x)
        yt[i] = sh[i];
}

// ---- Muse Glimmer QKV normalization / positional encoding -----------------
// Muse differs from Gemma in three important details: V is written to the KV
// cache without unit normalization, global-attention layers use NoPE, and every
// layer still applies learned per-head Q/K RMSNorm. Keep separate kernels so the
// established Gemma path and ABI remain unchanged.
struct MuseQkvPush {
    unsigned head_dim; float eps; unsigned rope_dim; unsigned position;
    unsigned n_head; unsigned n_kv_head; unsigned kv_offset; unsigned use_rope;
};
extern "C" __global__ void muse_norm_qkv(
    const float* q_in, const float* k_in, const float* v_in,
    const float* wq, const float* wk, const float* inv_freq,
    float* q_out, float* k_out, float* v_out, MuseQkvPush pc)
{
    unsigned bx = blockIdx.x;
    unsigned hd = pc.head_dim;
    extern __shared__ float sh[];

    const float* xt;
    float* yt;
    const float* w;
    if (bx < pc.n_head) {
        unsigned head = bx;
        xt = q_in + (size_t)head * hd;
        yt = q_out + (size_t)head * hd;
        w = wq;
    } else if (bx < pc.n_head + pc.n_kv_head) {
        unsigned head = bx - pc.n_head;
        xt = k_in + (size_t)head * hd;
        yt = k_out + (size_t)pc.kv_offset + (size_t)head * hd;
        w = wk;
    } else {
        unsigned head = bx - pc.n_head - pc.n_kv_head;
        xt = v_in + (size_t)head * hd;
        yt = v_out + (size_t)pc.kv_offset + (size_t)head * hd;
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) yt[i] = xt[i];
        return;
    }

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    if (pc.use_rope == 0u) {
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) yt[i] = sh[i];
        return;
    }
    // Muse/Llama-4 uses normal RoPE: rotate consecutive pairs (2i, 2i+1),
    // unlike Gemma's NEOX half-split layout.
    unsigned pairs = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < pairs; i += blockDim.x) {
        unsigned i0 = i << 1;
        unsigned i1 = i0 + 1;
        float x0 = sh[i0];
        float x1 = sh[i1];
        float theta = (float)pc.position * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i0] = x0 * ct - x1 * st;
        yt[i1] = x0 * st + x1 * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < hd; i += blockDim.x) yt[i] = sh[i];
}

struct MuseQkvBatchPush {
    unsigned head_dim; float eps; unsigned rope_dim; unsigned base_position;
    unsigned n_head; unsigned n_kv_head; unsigned use_rope;
};
extern "C" __global__ void muse_norm_qkv_batched(
    const float* q_in, const float* k_in, const float* v_in,
    const float* wq, const float* wk, const float* inv_freq,
    float* q_out, float* k_out, float* v_out, MuseQkvBatchPush pc)
{
    unsigned bx = blockIdx.x;
    unsigned t = blockIdx.y;
    unsigned hd = pc.head_dim;
    unsigned q_dim = pc.n_head * hd;
    unsigned kv_dim = pc.n_kv_head * hd;
    extern __shared__ float sh[];

    const float* xt;
    float* yt;
    const float* w;
    if (bx < pc.n_head) {
        unsigned head = bx;
        xt = q_in + (size_t)t * q_dim + (size_t)head * hd;
        yt = q_out + (size_t)t * q_dim + (size_t)head * hd;
        w = wq;
    } else if (bx < pc.n_head + pc.n_kv_head) {
        unsigned head = bx - pc.n_head;
        xt = k_in + (size_t)t * kv_dim + (size_t)head * hd;
        yt = k_out + (size_t)t * kv_dim + (size_t)head * hd;
        w = wk;
    } else {
        unsigned head = bx - pc.n_head - pc.n_kv_head;
        xt = v_in + (size_t)t * kv_dim + (size_t)head * hd;
        yt = v_out + (size_t)t * kv_dim + (size_t)head * hd;
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) yt[i] = xt[i];
        return;
    }

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    if (pc.use_rope == 0u) {
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) yt[i] = sh[i];
        return;
    }
    unsigned position = pc.base_position + t;
    unsigned pairs = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < pairs; i += blockDim.x) {
        unsigned i0 = i << 1;
        unsigned i1 = i0 + 1;
        float x0 = sh[i0];
        float x1 = sh[i1];
        float theta = (float)position * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i0] = x0 * ct - x1 * st;
        yt[i1] = x0 * st + x1 * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < hd; i += blockDim.x) yt[i] = sh[i];
}

struct MuseQkvSeqPush {
    unsigned head_dim; float eps; unsigned rope_dim;
    unsigned n_head; unsigned n_kv_head; unsigned slot_ctx; unsigned use_rope;
    unsigned write_f16;
};
extern "C" __global__ void muse_norm_qkv_seq(
    const float* q_in, const float* k_in, const float* v_in,
    const float* wq, const float* wk, const float* inv_freq,
    float* q_out, float* k_out, float* v_out,
    half* q_f16, half* k_f16, half* v_f16,
    const unsigned* positions, const unsigned* slots, MuseQkvSeqPush pc)
{
    unsigned bx = blockIdx.x;
    unsigned b = blockIdx.y;
    unsigned hd = pc.head_dim;
    unsigned q_dim = pc.n_head * hd;
    unsigned kv_dim = pc.n_kv_head * hd;
    unsigned pos = positions[b];
    unsigned slot = slots[b];
    size_t kv_base = ((size_t)slot * pc.slot_ctx + pos) * kv_dim;
    extern __shared__ float sh[];

    const float* xt;
    float* yt;
    half* yt16;
    const float* w;
    if (bx < pc.n_head) {
        unsigned head = bx;
        xt = q_in + (size_t)b * q_dim + (size_t)head * hd;
        yt = q_out + (size_t)b * q_dim + (size_t)head * hd;
        yt16 = q_f16 + (size_t)b * q_dim + (size_t)head * hd;
        w = wq;
    } else if (bx < pc.n_head + pc.n_kv_head) {
        unsigned head = bx - pc.n_head;
        xt = k_in + (size_t)b * kv_dim + (size_t)head * hd;
        yt = k_out + kv_base + (size_t)head * hd;
        yt16 = k_f16 + kv_base + (size_t)head * hd;
        w = wk;
    } else {
        unsigned head = bx - pc.n_head - pc.n_kv_head;
        xt = v_in + (size_t)b * kv_dim + (size_t)head * hd;
        yt = v_out + kv_base + (size_t)head * hd;
        yt16 = v_f16 + kv_base + (size_t)head * hd;
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
            const float value = xt[i];
            yt[i] = value;
            if (pc.write_f16 != 0u) yt16[i] = __float2half(value);
        }
        return;
    }

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    if (pc.use_rope == 0u) {
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
            const float value = sh[i];
            yt[i] = value;
            if (pc.write_f16 != 0u) yt16[i] = __float2half(value);
        }
        return;
    }
    unsigned pairs = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < pairs; i += blockDim.x) {
        unsigned i0 = i << 1;
        unsigned i1 = i0 + 1;
        float x0 = sh[i0];
        float x1 = sh[i1];
        float theta = (float)pos * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        const float y0 = x0 * ct - x1 * st;
        const float y1 = x0 * st + x1 * ct;
        yt[i0] = y0;
        yt[i1] = y1;
        if (pc.write_f16 != 0u) {
            yt16[i0] = __float2half(y0);
            yt16[i1] = __float2half(y1);
        }
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < hd; i += blockDim.x) {
        const float value = sh[i];
        yt[i] = value;
        if (pc.write_f16 != 0u) yt16[i] = __float2half(value);
    }
}

// Seed the persistent half KV mirror after batched prefill. Decode appends one
// row at a time directly in muse_norm_qkv_seq, so this O(prompt) conversion is
// paid once per request rather than once per generated token.
struct KvF16Push { unsigned N, offset; };
extern "C" __global__ void muse_kv_f32_to_f16(
    const float* k, const float* v, half* k16, half* v16, KvF16Push pc)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    const size_t at = (size_t)pc.offset + idx;
    k16[at] = __float2half(k[at]);
    v16[at] = __float2half(v[at]);
}

// ---- geglu (gemma FFN activation: gelu(gate) * up) -------------------------
// Matches ggml LLM_FFN_GELU (tanh approximation). gemma norm weights already
// carry the +1 offset (baked at GGUF conversion), so the surrounding norms use
// the standard rms_norm kernel.
extern "C" __global__ void geglu(const float* gate, const float* up, float* y, SwigluPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    float g = gate[idx];
    float gelu = 0.5f * g * (1.0f + tanhf(k * (g + 0.044715f * g * g * g)));
    y[idx] = gelu * up[idx];
}

// ---- scalar_mul (gemma per-layer output scale) ----------------------------
// a[i] *= s[0]. s is a device [1] buffer (blk.N.layer_output_scale.weight).
struct ScalarMulPush { unsigned N; };
extern "C" __global__ void scalar_mul(float* a, const float* s, ScalarMulPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    a[idx] *= s[0];
}

// ---- mul_vec_scaled (gemma4 MoE router pre-scale) --------------------------
// a[i] = a[i] * b[i] * scale. The gemma4 MoE router computes its logits from a
// plain-RMS-normed residual scaled by 1/sqrt(n_embd) and a per-channel weight
// (ffn_gate_inp.scale) before the gate projection.
struct MulVecPush { unsigned N; float scale; };
extern "C" __global__ void mul_vec_scaled(float* a, const float* b, MulVecPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    a[idx] = a[idx] * b[idx] * pc.scale;
}

// ---- mul_vec_scaled_batched (gemma4-MoE prefill router pre-scale, all T) ----
// a[t*row + i] = a[t*row + i] * b[i] * scale over all T token rows. The per-channel
// weight b (ffn_gate_inp.scale, length `row`=n_embd) is broadcast across tokens;
// element math is byte-for-byte mul_vec_scaled's (a*b*scale). Used by routerBatched.
struct MulVecBatchPush { unsigned row; unsigned total; float scale; };
extern "C" __global__ void mul_vec_scaled_batched(float* a, const float* b, MulVecBatchPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.total) return;
    a[idx] = a[idx] * b[idx % pc.row] * pc.scale;
}

// ---- zero_vec --------------------------------------------------------------
// a[i] = 0. Clears the MoE combine accumulator before moe_weighted_acc (which
// is a += kernel).
struct ZeroPush { unsigned N; };
extern "C" __global__ void zero_vec(float* a, ZeroPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    a[idx] = 0.0f;
}

// ---- gemma_attention (decode: softmax(scale*QK^T)V, GQA, sliding window) ----
// One block per query head. Causal + optional sliding-window mask: with window
// W>0 only the last W keys are attended (gemma4 SWA layers, W=1024). window==0
// is full attention. No attention sink. kq_scale is gemma's f_attention_scale
// (1.0); scale_bits==0 falls back to 1/sqrt(head_dim). KV cache layout matches
// kv_cache_write: contiguous [seq_len, n_kv_heads, head_dim].
struct GemmaAttnPush { unsigned head_dim, n_heads, n_kv_heads, seq_len, scale_bits, window; };
extern "C" __global__ void gemma_attention(const float* q, const float* k, const float* v,
                                           float* out, GemmaAttnPush pc) {
    extern __shared__ float s_scores[];           // size = seq_len floats (dynamic)
    __shared__ float s_m, s_inv;
    unsigned head = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned hd = pc.head_dim;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const float* qh = q + (size_t)head * hd;
    float scale = pc.scale_bits != 0u ? __uint_as_float(pc.scale_bits) : rsqrtf((float)hd);

    // sliding-window start: decode query is the last position (seq_len-1).
    unsigned start = 0;
    if (pc.window != 0u && pc.seq_len > pc.window) start = pc.seq_len - pc.window;

    // Pass 1: scores = scale * (q . k_i), track max.
    float lmax = -3.4e38f;
    for (unsigned i = start + tid; i < pc.seq_len; i += blockDim.x) {
        const float* ki = k + ((size_t)i * pc.n_kv_heads + kv_head) * hd;
        float dot = 0.0f;
        for (unsigned d = 0; d < hd; d++) dot += qh[d] * ki[d];
        float score = dot * scale;
        s_scores[i] = score;
        lmax = fmaxf(lmax, score);
    }
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0) s_m = lmax;
    __syncthreads();
    float m = s_m;

    // Pass 2: e_i = exp(score_i - m), sum.
    float lsum = 0.0f;
    for (unsigned i = start + tid; i < pc.seq_len; i += blockDim.x) {
        float e = expf(s_scores[i] - m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) s_inv = (lsum > 0.0f) ? 1.0f / lsum : 0.0f;
    __syncthreads();
    float inv = s_inv;

    // Pass 3: out[d] = (sum_i e_i * V[i,d]) * inv.
    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = start; i < pc.seq_len; i++)
            acc += s_scores[i] * v[((size_t)i * pc.n_kv_heads + kv_head) * hd + d];
        out[(size_t)head * hd + d] = acc * inv;
    }
}

extern "C" __global__ void gemma_attention_v2_decode(
    const float* q, const float* k, const float* v, float* out,
    GemmaAttnPush pc) {
    extern __shared__ float s_scores[];
    __shared__ float s_q[512];
    __shared__ float s_m, s_inv;
    const unsigned head = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;
    const unsigned hd = pc.head_dim;
    const unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const size_t kv_dim = (size_t)pc.n_kv_heads * hd;
    const float* qh = q + (size_t)head * hd;
    const float scale = pc.scale_bits != 0u ? __uint_as_float(pc.scale_bits) : rsqrtf((float)hd);

    unsigned start = 0u;
    if (pc.window != 0u && pc.seq_len > pc.window) start = pc.seq_len - pc.window;
    for (unsigned d = tid; d < hd; d += blockDim.x) s_q[d] = qh[d];
    __syncthreads();

    for (unsigned i = start + warp; i < pc.seq_len; i += 8u) {
        const float* ki = k + (size_t)i * kv_dim + (size_t)kv_head * hd;
        float dot = 0.0f;
        for (unsigned d = lane; d < hd; d += 32u) dot += s_q[d] * ki[d];
        dot = zinc_warp_reduce_sum(dot);
        if (lane == 0u) s_scores[i] = dot * scale;
    }
    __syncthreads();

    float lmax = -3.4e38f;
    for (unsigned i = start + tid; i < pc.seq_len; i += blockDim.x)
        lmax = fmaxf(lmax, s_scores[i]);
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0u) s_m = lmax;
    __syncthreads();

    float lsum = 0.0f;
    for (unsigned i = start + tid; i < pc.seq_len; i += blockDim.x) {
        const float e = expf(s_scores[i] - s_m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0u) s_inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
    __syncthreads();

    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = start; i < pc.seq_len; ++i)
            acc += s_scores[i] * v[(size_t)i * kv_dim + (size_t)kv_head * hd + d];
        out[(size_t)head * hd + d] = acc * s_inv;
    }
}

// Exact two-query-per-KV-head decode. Gemma 4 26B maps two adjacent query
// heads to each KV head; computing that pair in one workgroup halves K/V cache
// traffic while retaining each query head's serial dot and softmax reductions.
extern "C" __global__ void gemma_attention_gqa2(
    const float* q, const float* k, const float* v, float* out,
    GemmaAttnPush pc) {
    extern __shared__ float scores[];
    float* scores0 = scores;
    float* scores1 = scores + pc.seq_len;
    __shared__ float s_m0, s_m1, s_inv0, s_inv1;
    const unsigned kv_head = blockIdx.x;
    const unsigned head0 = kv_head * 2u;
    const unsigned head1 = head0 + 1u;
    const unsigned tid = threadIdx.x;
    const unsigned hd = pc.head_dim;
    const size_t kv_dim = (size_t)pc.n_kv_heads * hd;
    const float* q0 = q + (size_t)head0 * hd;
    const float* q1 = q + (size_t)head1 * hd;
    const float scale = pc.scale_bits != 0u ? __uint_as_float(pc.scale_bits) : rsqrtf((float)hd);

    unsigned start = 0u;
    if (pc.window != 0u && pc.seq_len > pc.window) start = pc.seq_len - pc.window;

    float lmax0 = -3.4e38f;
    float lmax1 = -3.4e38f;
    for (unsigned i = start + tid; i < pc.seq_len; i += blockDim.x) {
        const float* ki = k + (size_t)i * kv_dim + (size_t)kv_head * hd;
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        for (unsigned d = 0u; d < hd; ++d) {
            const float kval = ki[d];
            dot0 += q0[d] * kval;
            dot1 += q1[d] * kval;
        }
        const float score0 = dot0 * scale;
        const float score1 = dot1 * scale;
        scores0[i] = score0;
        scores1[i] = score1;
        lmax0 = fmaxf(lmax0, score0);
        lmax1 = fmaxf(lmax1, score1);
    }
    lmax0 = zinc_block_reduce_max(lmax0);
    lmax1 = zinc_block_reduce_max(lmax1);
    if (tid == 0u) {
        s_m0 = lmax0;
        s_m1 = lmax1;
    }
    __syncthreads();

    float lsum0 = 0.0f;
    float lsum1 = 0.0f;
    for (unsigned i = start + tid; i < pc.seq_len; i += blockDim.x) {
        const float e0 = expf(scores0[i] - s_m0);
        const float e1 = expf(scores1[i] - s_m1);
        scores0[i] = e0;
        scores1[i] = e1;
        lsum0 += e0;
        lsum1 += e1;
    }
    lsum0 = zinc_block_reduce_sum(lsum0);
    lsum1 = zinc_block_reduce_sum(lsum1);
    if (tid == 0u) {
        s_inv0 = lsum0 > 0.0f ? 1.0f / lsum0 : 0.0f;
        s_inv1 = lsum1 > 0.0f ? 1.0f / lsum1 : 0.0f;
    }
    __syncthreads();

    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (unsigned i = start; i < pc.seq_len; ++i) {
            const float vv = v[(size_t)i * kv_dim + (size_t)kv_head * hd + d];
            acc0 += scores0[i] * vv;
            acc1 += scores1[i] * vv;
        }
        out[(size_t)head0 * hd + d] = acc0 * s_inv0;
        out[(size_t)head1 * hd + d] = acc1 * s_inv1;
    }
}

// ---- gemma_attention_batched (Effort 24: batched prefill, gemma SWA/full) ----
// Batched twin of gemma_attention: prefill runs all T prompt queries at once.
// block (head=blockIdx.x, t=blockIdx.y) computes attention for query position t,
// head h, causally masked to keys [0..t] with the SAME optional sliding-window
// mask as gemma_attention (window>0 → last `window` keys). Q is token-major
// [T, n_heads, head_dim] (post norm+RoPE in b.q); K/V are the prompt region of the
// KV cache [T, n_kv_heads, head_dim] (positions 0..T-1, written per token). out is
// token-major [T, n_heads, head_dim]. No sink (gemma). Math/reduction order are
// byte-for-byte identical to gemma_attention with seq_len=t+1 → bit-identical out.
struct GemmaAttnBatchPush { unsigned head_dim, n_heads, n_kv_heads, T, scale_bits, window; };
extern "C" __global__ void gemma_attention_batched(const float* q, const float* k, const float* v,
                                                   float* out, GemmaAttnBatchPush pc) {
    extern __shared__ float s_scores[];           // size = T floats (max causal length)
    __shared__ float s_m, s_inv;
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;                       // query position
    if (t >= pc.T) return;
    unsigned seq_len = t + 1u;                     // causal: query t attends keys [0..t]
    unsigned tid = threadIdx.x;
    unsigned hd = pc.head_dim;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const float* qh = q + ((size_t)t * pc.n_heads + head) * hd;   // query t, head h
    float scale = pc.scale_bits != 0u ? __uint_as_float(pc.scale_bits) : rsqrtf((float)hd);

    // sliding-window start: identical to gemma_attention for this query's seq_len.
    unsigned start = 0;
    if (pc.window != 0u && seq_len > pc.window) start = seq_len - pc.window;

    // Pass 1: scores = scale * (q . k_i), track max.
    float lmax = -3.4e38f;
    for (unsigned i = start + tid; i < seq_len; i += blockDim.x) {
        const float* ki = k + ((size_t)i * pc.n_kv_heads + kv_head) * hd;
        float dot = 0.0f;
        for (unsigned d = 0; d < hd; d++) dot += qh[d] * ki[d];
        float score = dot * scale;
        s_scores[i] = score;
        lmax = fmaxf(lmax, score);
    }
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0) s_m = lmax;
    __syncthreads();
    float m = s_m;

    // Pass 2: e_i = exp(score_i - m), sum.
    float lsum = 0.0f;
    for (unsigned i = start + tid; i < seq_len; i += blockDim.x) {
        float e = expf(s_scores[i] - m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) s_inv = (lsum > 0.0f) ? 1.0f / lsum : 0.0f;
    __syncthreads();
    float inv = s_inv;

    // Pass 3: out[t,head,d] = (sum_i e_i * V[i,d]) * inv.
    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = start; i < seq_len; i++)
            acc += s_scores[i] * v[((size_t)i * pc.n_kv_heads + kv_head) * hd + d];
        out[((size_t)t * pc.n_heads + head) * hd + d] = acc * inv;
    }
}

// ---- gemma_attention_batched_v2 — coalesced prefill attention ----------------
// Same 3-pass softmax math as gemma_attention_batched, but pass 1 (scores) is
// WARP-PER-KEY: the 32 lanes of a warp cooperate on ONE key's head_dim dot
// product, reading CONTIGUOUS ki[lane], ki[lane+32], ... (one coalesced 128B
// transaction) instead of the naive thread-per-key layout (256 threads reading
// 256 keys strided by kv_stride => 32 separate transactions per element). Query
// row staged once in shared. NOT bit-identical (warp-tree reduction reorders the
// FMA sum vs the naive serial dot) — token-tolerance gate. Pass 2/3 unchanged.
// Dynamic shared = (T + head_dim) floats. Opt-in ZINC_ATTN_V2.
extern "C" __global__ void gemma_attention_batched_v2(const float* q, const float* k, const float* v,
                                                      float* out, GemmaAttnBatchPush pc) {
    extern __shared__ float smem[];
    float* s_scores = smem;               // [T]
    float* qsh = smem + pc.T;             // [head_dim]
    __shared__ float s_m, s_inv;
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;
    if (t >= pc.T) return;
    unsigned seq_len = t + 1u;
    unsigned tid = threadIdx.x, hd = pc.head_dim, nthr = blockDim.x;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const float* qh = q + ((size_t)t * pc.n_heads + head) * hd;
    float scale = pc.scale_bits != 0u ? __uint_as_float(pc.scale_bits) : rsqrtf((float)hd);
    unsigned start = 0;
    if (pc.window != 0u && seq_len > pc.window) start = seq_len - pc.window;

    for (unsigned d = tid; d < hd; d += nthr) qsh[d] = qh[d];
    __syncthreads();

    // Pass 1: warp-per-key coalesced dot -> s_scores[i].
    unsigned warp = tid >> 5, lane = tid & 31u, nwarp = nthr >> 5;
    for (unsigned i = start + warp; i < seq_len; i += nwarp) {
        const float* ki = k + ((size_t)i * pc.n_kv_heads + kv_head) * hd;
        float p = 0.0f;
        for (unsigned d = lane; d < hd; d += 32u) p += qsh[d] * ki[d];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
        if (lane == 0u) s_scores[i] = p * scale;
    }
    __syncthreads();

    // Pass 1b: block max.
    float lmax = -3.4e38f;
    for (unsigned i = start + tid; i < seq_len; i += nthr) lmax = fmaxf(lmax, s_scores[i]);
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0) s_m = lmax;
    __syncthreads();
    float mval = s_m;

    // Pass 2: exp + sum.
    float lsum = 0.0f;
    for (unsigned i = start + tid; i < seq_len; i += nthr) {
        float e = expf(s_scores[i] - mval);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) s_inv = (lsum > 0.0f) ? 1.0f / lsum : 0.0f;
    __syncthreads();
    float inv = s_inv;

    // Pass 3: out[t,head,d] = (sum_i e_i * V[i,d]) * inv  (coalesced, unchanged).
    for (unsigned d = tid; d < hd; d += nthr) {
        float acc = 0.0f;
        for (unsigned i = start; i < seq_len; i++)
            acc += s_scores[i] * v[((size_t)i * pc.n_kv_heads + kv_head) * hd + d];
        out[((size_t)t * pc.n_heads + head) * hd + d] = acc * inv;
    }
}

// ---- gemma_attention_batched_seq (Effort 28 1c: batched request DECODE) ------
// Batched twin of gemma_attention for DECODE: B independent sequences, each at
// its OWN position over its OWN KV slot, in ONE launch. block (head=blockIdx.x,
// b=blockIdx.y) computes attention for sequence b's single decode query, head h,
// causally masked to its slot's keys [0..positions[b]] with the same optional
// sliding-window mask. Q is token-major [B, n_heads, head_dim] (b.q, post
// norm+RoPE). K/V are the slot KV buffers [n_slots*slot_ctx, n_kv_heads, head_dim];
// sequence b reads slot slots[b] at base (slot*slot_ctx)*kv_dim. out is token-major
// [B, n_heads, head_dim]. Math/reduction order are byte-for-byte identical to
// gemma_attention against an aliased slot with seq_len=positions[b]+1 → the
// batched output equals the per-row looped form bit-for-bit.
struct GemmaAttnSlotPush { unsigned head_dim, n_heads, n_kv_heads, slot_ctx, scale_bits, window; };
extern "C" __global__ void gemma_attention_batched_seq(
    const float* q, const float* k, const float* v, float* out,
    const unsigned* positions, const unsigned* slots, GemmaAttnSlotPush pc) {
    extern __shared__ float s_scores[];           // size = max(seq_len) floats
    __shared__ float s_m, s_inv;
    unsigned head = blockIdx.x;
    unsigned b = blockIdx.y;
    unsigned pos = positions[b];
    unsigned slot = slots[b];
    unsigned seq_len = pos + 1u;                   // causal: attend keys [0..pos]
    unsigned tid = threadIdx.x;
    unsigned hd = pc.head_dim;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    size_t kv_dim = (size_t)pc.n_kv_heads * hd;
    size_t slot_off = (size_t)slot * pc.slot_ctx * kv_dim;
    const float* qh = q + ((size_t)b * pc.n_heads + head) * hd;   // seq b, head h
    const float* kbase = k + slot_off;
    const float* vbase = v + slot_off;
    float scale = pc.scale_bits != 0u ? __uint_as_float(pc.scale_bits) : rsqrtf((float)hd);

    unsigned start = 0;
    if (pc.window != 0u && seq_len > pc.window) start = seq_len - pc.window;

    // Pass 1: scores = scale * (q . k_i), track max.
    float lmax = -3.4e38f;
    for (unsigned i = start + tid; i < seq_len; i += blockDim.x) {
        const float* ki = kbase + (size_t)i * kv_dim + (size_t)kv_head * hd;
        float dot = 0.0f;
        for (unsigned d = 0; d < hd; d++) dot += qh[d] * ki[d];
        float score = dot * scale;
        s_scores[i] = score;
        lmax = fmaxf(lmax, score);
    }
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0) s_m = lmax;
    __syncthreads();
    float m = s_m;

    // Pass 2: e_i = exp(score_i - m), sum.
    float lsum = 0.0f;
    for (unsigned i = start + tid; i < seq_len; i += blockDim.x) {
        float e = expf(s_scores[i] - m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) s_inv = (lsum > 0.0f) ? 1.0f / lsum : 0.0f;
    __syncthreads();
    float inv = s_inv;

    // Pass 3: out[b,head,d] = (sum_i e_i * V[i,d]) * inv.
    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = start; i < seq_len; i++)
            acc += s_scores[i] * vbase[(size_t)i * kv_dim + (size_t)kv_head * hd + d];
        out[((size_t)b * pc.n_heads + head) * hd + d] = acc * inv;
    }
}

// Coalesced decode attention. Eight wave32 warps cooperate on eight keys at a
// time, so head_dim=512 K rows are read in contiguous cache lines instead of
// the naive thread-per-key kernel's widely strided loads. The softmax and V
// accumulation retain the existing block reduction and output order.
extern "C" __global__ void gemma_attention_batched_seq_v2(
    const float* q, const float* k, const float* v, float* out,
    const unsigned* positions, const unsigned* slots, GemmaAttnSlotPush pc) {
    extern __shared__ float s_scores[];
    __shared__ float s_q[512];
    __shared__ float s_m, s_inv;
    const unsigned head = blockIdx.x;
    const unsigned b = blockIdx.y;
    const unsigned pos = positions[b];
    const unsigned slot = slots[b];
    const unsigned seq_len = pos + 1u;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5;
    const unsigned hd = pc.head_dim;
    const unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const size_t kv_dim = (size_t)pc.n_kv_heads * hd;
    const size_t slot_off = (size_t)slot * pc.slot_ctx * kv_dim;
    const float* qh = q + ((size_t)b * pc.n_heads + head) * hd;
    const float* kbase = k + slot_off;
    const float* vbase = v + slot_off;
    const float scale = pc.scale_bits != 0u ? __uint_as_float(pc.scale_bits) : rsqrtf((float)hd);

    unsigned start = 0u;
    if (pc.window != 0u && seq_len > pc.window) start = seq_len - pc.window;

    for (unsigned d = tid; d < hd; d += blockDim.x) s_q[d] = qh[d];
    __syncthreads();

    for (unsigned i = start + warp; i < seq_len; i += 8u) {
        const float* ki = kbase + (size_t)i * kv_dim + (size_t)kv_head * hd;
        float dot = 0.0f;
        for (unsigned d = lane; d < hd; d += 32u) dot += s_q[d] * ki[d];
        dot = zinc_warp_reduce_sum(dot);
        if (lane == 0u) s_scores[i] = dot * scale;
    }
    __syncthreads();

    float lmax = -3.4e38f;
    for (unsigned i = start + tid; i < seq_len; i += blockDim.x)
        lmax = fmaxf(lmax, s_scores[i]);
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0u) s_m = lmax;
    __syncthreads();

    float lsum = 0.0f;
    for (unsigned i = start + tid; i < seq_len; i += blockDim.x) {
        const float e = expf(s_scores[i] - s_m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0u) s_inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
    __syncthreads();

    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = start; i < seq_len; ++i)
            acc += s_scores[i] * vbase[(size_t)i * kv_dim + (size_t)kv_head * hd + d];
        out[((size_t)b * pc.n_heads + head) * hd + d] = acc * s_inv;
    }
}

// Softmax for the grouped-BLAS Muse decode path. hipBLAS writes one contiguous
// score row per query head; normalize and downcast those rows for the fp16
// probability GEMM. The score scale is applied here so both GEMMs stay plain.
struct MuseAttnSoftmaxPush { unsigned seq_len, stride, n_heads, scale_bits; };
extern "C" __global__ void muse_attention_softmax_inplace(float* scores, half* probs, MuseAttnSoftmaxPush pc) {
    const unsigned head = blockIdx.x;
    if (head >= pc.n_heads) return;
    const unsigned tid = threadIdx.x;
    float* row = scores + (size_t)head * pc.stride;
    const float scale = __uint_as_float(pc.scale_bits);

    float lmax = -3.4e38f;
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x)
        lmax = fmaxf(lmax, row[i] * scale);
    lmax = zinc_block_reduce_max(lmax);
    __shared__ float s_max, s_inv;
    if (tid == 0u) s_max = lmax;
    __syncthreads();

    float lsum = 0.0f;
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x) {
        const float e = expf(row[i] * scale - s_max);
        row[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0u) s_inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
    __syncthreads();
    half* out = probs + (size_t)head * pc.stride;
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x)
        out[i] = __float2half(row[i] * s_inv);
}

// fp32 twin for Gemma decode. Keeping this separate avoids an unused fp16
// probability write on Gemma and preserves Muse's lean fp16 attention path.
extern "C" __global__ void gemma_attention_softmax_f32_inplace(float* scores, MuseAttnSoftmaxPush pc) {
    const unsigned head = blockIdx.x;
    if (head >= pc.n_heads) return;
    const unsigned tid = threadIdx.x;
    float* row = scores + (size_t)head * pc.stride;
    const float scale = __uint_as_float(pc.scale_bits);

    float lmax = -3.4e38f;
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x)
        lmax = fmaxf(lmax, row[i] * scale);
    lmax = zinc_block_reduce_max(lmax);
    __shared__ float s_max, s_inv;
    if (tid == 0u) s_max = lmax;
    __syncthreads();

    float lsum = 0.0f;
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x) {
        const float e = expf(row[i] * scale - s_max);
        row[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0u) s_inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
    __syncthreads();
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x)
        row[i] *= s_inv;
}

// Muse single-client decode specialization: fold the post-attention sigmoid
// gate and Q8_1 packing into attention's output pass. The following O projection
// consumes only Q8_1, so the intermediate f32 attention row and a full extra
// kernel launch are unnecessary. Grid/block/shared sizing matches
// gemma_attention_batched_seq; q8 uses quantize_act_q8_0's ROCm MMQ layout.
extern "C" __global__ void muse_attention_batched_seq_q8(
    const float* q, const float* k, const float* v, const float* gate,
    unsigned char* q8, const unsigned* positions, const unsigned* slots,
    GemmaAttnSlotPush pc)
{
    extern __shared__ float s_scores[];
    __shared__ float s_m, s_inv;
    const unsigned head = blockIdx.x;
    const unsigned b = blockIdx.y;
    const unsigned pos = positions[b];
    const unsigned slot = slots[b];
    const unsigned seq_len = pos + 1u;
    const unsigned tid = threadIdx.x;
    const unsigned hd = pc.head_dim;
    const unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const size_t kv_dim = (size_t)pc.n_kv_heads * hd;
    const size_t slot_off = (size_t)slot * pc.slot_ctx * kv_dim;
    const float* qh = q + ((size_t)b * pc.n_heads + head) * hd;
    const float* kbase = k + slot_off;
    const float* vbase = v + slot_off;
    const float scale = pc.scale_bits != 0u ? __uint_as_float(pc.scale_bits) : rsqrtf((float)hd);
    unsigned start = 0u;
    if (pc.window != 0u && seq_len > pc.window) start = seq_len - pc.window;

    float lmax = -3.4e38f;
    for (unsigned i = start + tid; i < seq_len; i += blockDim.x) {
        const float* ki = kbase + (size_t)i * kv_dim + (size_t)kv_head * hd;
        float dot = 0.0f;
        for (unsigned di = 0; di < hd; di++) dot += qh[di] * ki[di];
        const float score = dot * scale;
        s_scores[i] = score;
        lmax = fmaxf(lmax, score);
    }
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0u) s_m = lmax;
    __syncthreads();

    float lsum = 0.0f;
    for (unsigned i = start + tid; i < seq_len; i += blockDim.x) {
        const float e = expf(s_scores[i] - s_m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0u) s_inv = lsum > 0.0f ? 1.0f / lsum : 0.0f;
    __syncthreads();
    if (tid >= hd) return;

    float acc = 0.0f;
    for (unsigned i = start; i < seq_len; i++)
        acc += s_scores[i] * vbase[(size_t)i * kv_dim + (size_t)kv_head * hd + tid];
    const size_t out_idx = ((size_t)b * pc.n_heads + head) * hd + tid;
    const float gval = gate[out_idx];
    const float val = (acc * s_inv) * (1.0f / (1.0f + expf(-gval)));

    float av = fabsf(val), sv = val;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
        sv += __shfl_xor_sync(0xffffffffu, sv, o);
    }
    const float d = av / 127.0f;
    const float qscale = 127.0f / fmaxf(av, 1e-5f);
    const int qv = max(-127, min(127, __float2int_rn(val * qscale)));
    const unsigned lane = tid & 31u;
    const unsigned c = (head * hd + tid) >> 5;
    const unsigned tile = c >> 2;
    const unsigned group = c & 3u;
    unsigned char* out = q8 + ((size_t)tile * gridDim.y + b) * 144u;
    if (lane == 0u) {
        const unsigned short dh = zinc_float_to_half(d);
        const unsigned short sh = zinc_float_to_half(sv);
        out[group * 4u] = (unsigned char)(dh & 0xffu);
        out[group * 4u + 1u] = (unsigned char)(dh >> 8);
        out[group * 4u + 2u] = (unsigned char)(sh & 0xffu);
        out[group * 4u + 3u] = (unsigned char)(sh >> 8);
    }
    out[16u + group * 32u + lane] = (unsigned char)qv;
}

// ---- deinterleave_qgate (qwen35 packed Q+gate projection) ----
// wq outputs [2*head_dim] per head, laid out as [Q(head_dim) | gate(head_dim)]
// interleaved across heads: [Q0,g0,Q1,g1,...]. Split into contiguous q_out and
// gate_out (each [n_head*head_dim]). One thread per (head,dim) element.
struct DeintPush { unsigned head_dim, n_head; };
extern "C" __global__ void deinterleave_qgate(const float* qfull, float* q_out, float* gate_out, DeintPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = pc.n_head * pc.head_dim;
    if (i >= total) return;
    unsigned h = i / pc.head_dim;
    unsigned d = i % pc.head_dim;
    unsigned src = h * 2u * pc.head_dim + d;
    q_out[i]    = qfull[src];
    gate_out[i] = qfull[src + pc.head_dim];
}

// ---- qwen_norm_rope_qkv (single-sequence DECODE attn front-end) ------------
// Fuses deinterleave_qgate + per-head Q/K RMS norm + Q/K RoPE + KV writes.
// The Q projection is packed [Q(hd)|gate(hd)] per head.  One block owns one
// Q, K, or V head, so all writes are disjoint and the norm reduction order
// matches the standalone rms_norm kernel (one 256-thread block per head).
struct QwenQkvPush {
    unsigned head_dim; float eps; unsigned rope_dim;
    unsigned n_head; unsigned n_kv_head; unsigned position; unsigned kv_offset;
};
extern "C" __global__ void qwen_norm_rope_qkv(
    const float* qfull, const float* k_in, const float* v_in,
    const float* wq, const float* wk, const float* inv_freq,
    float* q_out, float* gate_out, float* k_out, float* v_out,
    QwenQkvPush pc)
{
    unsigned bx = blockIdx.x;
    unsigned hd = pc.head_dim;
    extern __shared__ float sh[]; // hd normalized values (Q/K rope staging)

    const float* xt;
    float* yt;
    const float* w;
    if (bx < pc.n_head) {
        unsigned head = bx;
        xt = qfull + (size_t)head * 2u * hd;
        yt = q_out + (size_t)head * hd;
        w = wq;
        float* gt = gate_out + (size_t)head * hd;
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
            gt[i] = xt[hd + i];
    } else if (bx < pc.n_head + pc.n_kv_head) {
        unsigned head = bx - pc.n_head;
        xt = k_in + (size_t)head * hd;
        yt = k_out + (size_t)pc.kv_offset + (size_t)head * hd;
        w = wk;
    } else {
        unsigned head = bx - pc.n_head - pc.n_kv_head;
        const float* vt = v_in + (size_t)head * hd;
        float* vo = v_out + (size_t)pc.kv_offset + (size_t)head * hd;
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
            vo[i] = vt[i];
        return;
    }

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    unsigned half_rot = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < half_rot; i += blockDim.x) {
        float xi = sh[i];
        float xih = sh[i + half_rot];
        float theta = (float)pc.position * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i] = xi * ct - xih * st;
        yt[i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < hd; i += blockDim.x)
        yt[i] = sh[i];
}

// ---- qwen_norm_rope_qkv_seq (Effort 28 4c-2: batched DECODE attn front-end) --
// Batched fused twin of the qwen per-row decode attention front-end
// (deinterleave_qgate + per-head q/k RMS-norm + RoPE + slot KV write + gate
// extract): collapses the per-row decode loop into ONE launch over B sequences.
// grid = (n_head + 2*n_kv_head, B). Row b is sequence b at its OWN position
// positions[b] writing its OWN KV slot slots[b].
//   qfull     token-major [B, 2*q_dim], q+gate INTERLEAVED per head (the
//             deinterleave_qgate source layout: [Q(hd)|gate(hd)] per head)
//   k_in/v_in token-major [B, kv_dim] (this step's K/V projections)
//   wq/wk     per-head q/k norm weights ([head_dim]); inv_freq the RoPE buffer
//   q_out     token-major [B, q_dim] (normed+roped Q)
//   gate_out  token-major [B, q_dim] (RAW gate half, consumed post-attn by sigmoid_mul)
//   k_out/v_out the slot KV buffers [n_slots*slot_ctx, kv_dim]; row b writes at
//             (slot*slot_ctx + pos)*kv_dim + head*head_dim. V is NOT normalized (qwen).
// Per-(head,b) arithmetic is COPIED from deinterleave_qgate + rms_norm(N=head_dim)
// + rope + kv_cache_write (with pc.position -> positions[b]) so it is bit-identical
// to the per-row launches. No cross-block hazard: each (b,head) writes disjoint
// regions (Q/gate per (b,head); K/V into the slot at this step's position).
struct QwenQkvSeqPush {
    unsigned head_dim; float eps; unsigned rope_dim;
    unsigned n_head; unsigned n_kv_head; unsigned slot_ctx;
};
extern "C" __global__ void qwen_norm_rope_qkv_seq(
    const float* qfull, const float* k_in, const float* v_in,
    const float* wq, const float* wk, const float* inv_freq,
    float* q_out, float* gate_out, float* k_out, float* v_out,
    const unsigned* positions, const unsigned* slots, QwenQkvSeqPush pc)
{
    unsigned bx = blockIdx.x;
    unsigned b  = blockIdx.y;
    unsigned hd = pc.head_dim;
    unsigned q_dim  = pc.n_head * hd;
    unsigned kv_dim = pc.n_kv_head * hd;
    unsigned pos  = positions[b];
    unsigned slot = slots[b];
    size_t kv_base = ((size_t)slot * pc.slot_ctx + pos) * kv_dim;  // slot KV write pos
    extern __shared__ float sh[]; // hd normalized values (Q/K rope staging)

    const float* xt;
    float* yt;
    const float* w;
    if (bx < pc.n_head) {
        unsigned head = bx;
        // Q is interleaved with gate: [Q(hd) | gate(hd)] per head in qfull.
        xt = qfull + (size_t)b * 2u * q_dim + (size_t)head * 2u * hd;
        yt = q_out + (size_t)b * q_dim + (size_t)head * hd;
        w  = wq;
        // Extract the RAW gate half (no norm) — matches deinterleave_qgate's gate_out.
        float* gt = gate_out + (size_t)b * q_dim + (size_t)head * hd;
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
            gt[i] = xt[hd + i];
    } else if (bx < pc.n_head + pc.n_kv_head) {
        unsigned head = bx - pc.n_head;
        xt = k_in + (size_t)b * kv_dim + (size_t)head * hd;
        yt = k_out + kv_base + (size_t)head * hd;
        w  = wk;
    } else {
        // V: NOT normalized for qwen — raw projection written straight to slot KV.
        unsigned head = bx - pc.n_head - pc.n_kv_head;
        const float* vt = v_in + (size_t)b * kv_dim + (size_t)head * hd;
        float* vo = v_out + kv_base + (size_t)head * hd;
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
            vo[i] = vt[i];
        return;
    }

    // RMS-norm over head_dim (matches rms_norm with N=head_dim, per head).
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    // RoPE [0..rope_dim) (attn_scale 1, inv_freq buffer), copy tail [rope_dim..hd).
    unsigned half_rot = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < half_rot; i += blockDim.x) {
        float xi = sh[i];
        float xih = sh[i + half_rot];
        float theta = (float)pos * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i] = xi * ct - xih * st;
        yt[i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < hd; i += blockDim.x)
        yt[i] = sh[i];
}

// ---- naive_attention_batched_seq (Effort 28 4c-2: batched request DECODE) ----
// Batched twin of naive_attention for DECODE: B independent sequences, each at
// its OWN position over its OWN KV slot, in ONE launch. block (head=blockIdx.x,
// b=blockIdx.y) computes sequence b's single decode query for head h, causally
// over its slot's keys [0..positions[b]], with the attention-sink rescale.
// Q is token-major [B, n_heads, head_dim] (b.q, post norm+RoPE); K/V are the slot
// KV buffers [n_slots*slot_ctx, n_kv_heads, head_dim] (seq b reads slot slots[b]
// at base (slot*slot_ctx)*kv_dim); out token-major [B, n_heads, head_dim]. Sink:
// sinks[sink_offset+head] (NaN = none). Math/reduction order are byte-for-byte
// naive_attention against an aliased slot with seq_len=positions[b]+1 → the
// batched output equals the per-row looped form bit-for-bit.
struct AttnSlotPush { unsigned head_dim, n_heads, n_kv_heads, slot_ctx, attn_scale_bits, sink_offset; };
extern "C" __global__ void naive_attention_batched_seq(
    const float* q, const float* k, const float* v, const float* sinks, float* out,
    const unsigned* positions, const unsigned* slots, AttnSlotPush pc) {
    extern __shared__ float s_scores[];           // size = max(seq_len) floats
    __shared__ float s_m, s_rescale, s_inv;
    unsigned head = blockIdx.x;
    unsigned b = blockIdx.y;
    unsigned pos = positions[b];
    unsigned slot = slots[b];
    unsigned seq_len = pos + 1u;
    unsigned tid = threadIdx.x;
    unsigned hd = pc.head_dim;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    size_t kv_dim = (size_t)pc.n_kv_heads * hd;
    size_t slot_off = (size_t)slot * pc.slot_ctx * kv_dim;
    const float* qh = q + ((size_t)b * pc.n_heads + head) * hd;
    const float* kbase = k + slot_off;
    const float* vbase = v + slot_off;
    float scale = pc.attn_scale_bits != 0u ? __uint_as_float(pc.attn_scale_bits) : rsqrtf((float)hd);

    // Pass 1: scores = scale * (q . k_i), track max.
    float lmax = -3.4e38f;
    for (unsigned i = tid; i < seq_len; i += blockDim.x) {
        const float* ki = kbase + (size_t)i * kv_dim + (size_t)kv_head * hd;
        float dot = 0.0f;
        for (unsigned d = 0; d < hd; d++) dot += qh[d] * ki[d];
        float score = dot * scale;
        s_scores[i] = score;
        lmax = fmaxf(lmax, score);
    }
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0) s_m = lmax;
    __syncthreads();
    float m = s_m;

    // Pass 2: e_i = exp(score_i - m), sum.
    float lsum = 0.0f;
    for (unsigned i = tid; i < seq_len; i += blockDim.x) {
        float e = expf(s_scores[i] - m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) {
        float sum = lsum, rescale = 1.0f, final_sum = lsum;
        float sink_val = sinks[pc.sink_offset + head];
        if (sink_val == sink_val) {   // sink present (NaN == NaN is false)
            float sink_max = fmaxf(m, sink_val);
            rescale = (sum > 0.0f) ? expf(m - sink_max) : 0.0f;
            final_sum = sum * rescale + expf(sink_val - sink_max);
        }
        s_rescale = rescale;
        s_inv = (final_sum > 0.0f) ? 1.0f / final_sum : 0.0f;
    }
    __syncthreads();
    float rescale = s_rescale, inv = s_inv;

    // Pass 3: out[b,head,d] = (sum_i e_i * V[i,d]) * rescale * inv.
    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = 0; i < seq_len; i++)
            acc += s_scores[i] * vbase[(size_t)i * kv_dim + (size_t)kv_head * hd + d];
        out[((size_t)b * pc.n_heads + head) * hd + d] = acc * rescale * inv;
    }
}

// ---- attention_causal_batched (Effort 24: batched prefill attention) --------
// Prefill processes T prompt tokens at once. naive_attention is single-query
// (grid=n_heads, one query over [0..seq_len)). This batches all T queries: block
// (head=blockIdx.x, t=blockIdx.y) computes attention for query position t, head h,
// CAUSALLY masked to keys [0..t]. Same 3-pass softmax(QK^T)V + GQA + sink logic.
// Q/K/V are [T, n_kv_heads-or-n_heads, head_dim]; out is [T, n_heads, head_dim].
struct AttnBatchPush { unsigned head_dim, n_heads, n_kv_heads, T, attn_scale_bits, sink_offset; };

extern "C" __global__ void attention_causal_batched(const float* q, const float* k, const float* v,
                                                    const float* sinks, float* out, AttnBatchPush pc) {
    extern __shared__ float s_scores[];          // size = T floats (max causal length)
    __shared__ float s_m, s_rescale, s_inv;
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;                      // query position
    if (t >= pc.T) return;
    unsigned seq_len = t + 1u;                    // causal: query t attends keys [0..t]
    unsigned tid = threadIdx.x;
    unsigned hd = pc.head_dim;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const float* qh = q + ((size_t)t * pc.n_heads + head) * hd;   // query t, head h
    float scale = pc.attn_scale_bits != 0u ? __uint_as_float(pc.attn_scale_bits) : rsqrtf((float)hd);

    // Pass 1: scores = scale*(q.k_i), track max.
    float lmax = -3.4e38f;
    for (unsigned i = tid; i < seq_len; i += blockDim.x) {
        const float* ki = k + ((size_t)i * pc.n_kv_heads + kv_head) * hd;
        float dot = 0.0f;
        for (unsigned d = 0; d < hd; d++) dot += qh[d] * ki[d];
        float score = dot * scale;
        s_scores[i] = score;
        lmax = fmaxf(lmax, score);
    }
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0) s_m = lmax;
    __syncthreads();
    float m = s_m;

    // Pass 2: e_i = exp(score_i - m), sum.
    float lsum = 0.0f;
    for (unsigned i = tid; i < seq_len; i += blockDim.x) {
        float e = expf(s_scores[i] - m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) {
        float sum = lsum, rescale = 1.0f, final_sum = lsum;
        float sink_val = sinks[pc.sink_offset + head];
        if (sink_val == sink_val) {
            float sink_max = fmaxf(m, sink_val);
            rescale = (sum > 0.0f) ? expf(m - sink_max) : 0.0f;
            final_sum = sum * rescale + expf(sink_val - sink_max);
        }
        s_rescale = rescale;
        s_inv = (final_sum > 0.0f) ? 1.0f / final_sum : 0.0f;
    }
    __syncthreads();
    float rescale = s_rescale, inv = s_inv;

    // Pass 3: out[t,head,d] = (sum_i e_i V[i,d]) * rescale * inv.
    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = 0; i < seq_len; i++)
            acc += s_scores[i] * v[((size_t)i * pc.n_kv_heads + kv_head) * hd + d];
        out[((size_t)t * pc.n_heads + head) * hd + d] = acc * rescale * inv;
    }
}

// ---- attention_causal_batched_v2 — coalesced qwen prefill attention ----------
// Coalesced twin of attention_causal_batched: pass 1 is WARP-PER-KEY (32 lanes
// cooperate on one key's head_dim dot, contiguous ki reads + warp-shuffle reduce)
// instead of thread-per-key strided reads. Query row staged in shared. Pass 2
// (attention sinks / rescale) and pass 3 are byte-for-byte the naive kernel's.
// NOT bit-identical (pass-1 reduction reorder) -> token-tolerance. Opt via
// ZINC_ATTN_V2. Dynamic shared = (T + head_dim) floats.
extern "C" __global__ void attention_causal_batched_v2(const float* q, const float* k, const float* v,
                                                       const float* sinks, float* out, AttnBatchPush pc) {
    extern __shared__ float smem2[];
    float* s_scores = smem2;              // [T]
    float* qsh = smem2 + pc.T;            // [head_dim]
    __shared__ float s_m, s_rescale, s_inv;
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;
    if (t >= pc.T) return;
    unsigned seq_len = t + 1u;
    unsigned tid = threadIdx.x, hd = pc.head_dim, nthr = blockDim.x;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const float* qh = q + ((size_t)t * pc.n_heads + head) * hd;
    float scale = pc.attn_scale_bits != 0u ? __uint_as_float(pc.attn_scale_bits) : rsqrtf((float)hd);

    for (unsigned d = tid; d < hd; d += nthr) qsh[d] = qh[d];
    __syncthreads();

    // Pass 1: warp-per-key coalesced dot -> s_scores[i].
    unsigned warp = tid >> 5, lane = tid & 31u, nwarp = nthr >> 5;
    for (unsigned i = warp; i < seq_len; i += nwarp) {
        const float* ki = k + ((size_t)i * pc.n_kv_heads + kv_head) * hd;
        float p = 0.0f;
        for (unsigned d = lane; d < hd; d += 32u) p += qsh[d] * ki[d];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
        if (lane == 0u) s_scores[i] = p * scale;
    }
    __syncthreads();

    // Pass 1b: block max.
    float lmax = -3.4e38f;
    for (unsigned i = tid; i < seq_len; i += nthr) lmax = fmaxf(lmax, s_scores[i]);
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0) s_m = lmax;
    __syncthreads();
    float m = s_m;

    // Pass 2: exp + sum, with attention sinks (byte-for-byte the naive kernel).
    float lsum = 0.0f;
    for (unsigned i = tid; i < seq_len; i += nthr) {
        float e = expf(s_scores[i] - m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) {
        float sum = lsum, rescale = 1.0f, final_sum = lsum;
        float sink_val = sinks[pc.sink_offset + head];
        if (sink_val == sink_val) {
            float sink_max = fmaxf(m, sink_val);
            rescale = (sum > 0.0f) ? expf(m - sink_max) : 0.0f;
            final_sum = sum * rescale + expf(sink_val - sink_max);
        }
        s_rescale = rescale;
        s_inv = (final_sum > 0.0f) ? 1.0f / final_sum : 0.0f;
    }
    __syncthreads();
    float rescale = s_rescale, inv = s_inv;

    // Pass 3: out[t,head,d] = (sum_i e_i V[i,d]) * rescale * inv (unchanged).
    for (unsigned d = tid; d < hd; d += nthr) {
        float acc = 0.0f;
        for (unsigned i = 0; i < seq_len; i++)
            acc += s_scores[i] * v[((size_t)i * pc.n_kv_heads + kv_head) * hd + d];
        out[((size_t)t * pc.n_heads + head) * hd + d] = acc * rescale * inv;
    }
}

// ---- deinterleave_qgate_batched (Effort 26 T0: batched-prefill twin) ---------
// Batched twin of deinterleave_qgate over T prompt tokens. qfull is token-major
// [T, 2*q_dim] (each token's row is [Q0,g0,Q1,g1,...] interleaved per head);
// split into contiguous token-major q_out / gate_out (each [T, q_dim]). Block
// (chunk=blockIdx.x, token=blockIdx.y); identical per-element math to the
// single-token kernel.
struct DeintBatchPush { unsigned head_dim, n_head, T; };
extern "C" __global__ void deinterleave_qgate_batched(const float* qfull, float* q_out, float* gate_out, DeintBatchPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned t = blockIdx.y;
    unsigned total = pc.n_head * pc.head_dim;        // = q_dim
    if (i >= total || t >= pc.T) return;
    unsigned h = i / pc.head_dim;
    unsigned d = i % pc.head_dim;
    unsigned src = t * 2u * total + h * 2u * pc.head_dim + d;
    q_out[(size_t)t * total + i]    = qfull[src];
    gate_out[(size_t)t * total + i] = qfull[src + pc.head_dim];
}

// ---- rope_batched (Effort 26 T0: batched-prefill twin of rope) --------------
// Batched twin of `rope` over T prompt tokens. x/y token-major [T, n_heads*stride];
// block (head=blockIdx.x, token=blockIdx.y) rotates head `head` of token `t` at
// sequence position base_position+t. Identical per-(head,position) math to `rope`.
struct RopeBatchPush { unsigned stride, rope_dim, n_heads, base_position, freq_base_bits, attn_scale_bits; };
extern "C" __global__ void rope_batched(const float* x, float* y, const float* inv_freq, RopeBatchPush pc) {
    unsigned tid = threadIdx.x;
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;
    unsigned base = ((size_t)t * pc.n_heads + head) * pc.stride;
    unsigned position = pc.base_position + t;
    unsigned half_rot = pc.rope_dim >> 1;
    float freq_base = __uint_as_float(pc.freq_base_bits);
    float attn_scale = pc.attn_scale_bits != 0u ? __uint_as_float(pc.attn_scale_bits) : 1.0f;
    bool use_buf = (pc.freq_base_bits == 0u);
    for (unsigned i = tid; i < half_rot; i += blockDim.x) {
        float xi = x[base + i];
        float xih = x[base + i + half_rot];
        float freq_i = use_buf ? inv_freq[i]
                               : (1.0f / powf(freq_base, (float)(2u * i) / (float)pc.rope_dim));
        float theta = (float)position * freq_i;
        float ct = cosf(theta) * attn_scale;
        float st = sinf(theta) * attn_scale;
        y[base + i] = xi * ct - xih * st;
        y[base + i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + tid; i < pc.stride; i += blockDim.x)
        y[base + i] = x[base + i];
}

// ---- kv_cache_write_batched (Effort 26 T0: batched-prefill twin) ------------
// Batched twin of kv_cache_write over T prompt tokens. k_src/v_src token-major
// [T, kv_dim]; token t writes into the KV cache at physical position dst_base+t
// (= (dst_base+t)*kv_dim). Block (chunk=blockIdx.x, token=blockIdx.y).
struct KvWriteBatchPush { unsigned kv_dim, dst_base, T; };
extern "C" __global__ void kv_cache_write_batched(const float* k_src, float* k_dst, const float* v_src, float* v_dst, KvWriteBatchPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned t = blockIdx.y;
    if (i >= pc.kv_dim || t >= pc.T) return;
    size_t dst = (size_t)(pc.dst_base + t) * pc.kv_dim + i;
    size_t src = (size_t)t * pc.kv_dim + i;
    k_dst[dst] = k_src[src];
    v_dst[dst] = v_src[src];
}
