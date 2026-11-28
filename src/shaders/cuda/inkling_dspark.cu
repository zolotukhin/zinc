// Greedy Inkling 2 DSpark draft, driven by four hidden-state taps from the target.
namespace inkling_dspark {

struct Feature { unsigned width, stride, index; };
extern "C" __global__ void dspark_capture(const float* source, float* features, Feature p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i >= p.width) return;
    const size_t row = size_t(blockIdx.y) * p.width;
    features[size_t(p.index) * p.stride + row + i] = source[row + i];
}

struct FeatureMix { unsigned width, stride, first; float eps; };
extern "C" __global__ void dspark_feature_mix(
        const float* features, const float* weights, const float* norm, float* out, FeatureMix p) {
    features += size_t(p.first + blockIdx.x) * p.width;
    out += size_t(blockIdx.x) * p.width;
    float sum = 0;
    for (unsigned i = threadIdx.x; i < p.width; i += 64) {
        float x = features[i] * weights[0];
        for (unsigned tap = 1; tap < 4; ++tap) x = fmaf(features[tap * p.stride + i], weights[tap], x);
        out[i] = x;
        sum = fmaf(x, x, sum);
    }
    const float inverse = rsqrtf(zinc_block_reduce_sum_all(sum) / p.width + p.eps);
    for (unsigned i = threadIdx.x; i < p.width; i += 64) out[i] = out[i] * inverse * norm[i];
}

struct DraftEmbed { unsigned width, kind; float eps; };
extern "C" __global__ void dspark_embed(
        const unsigned char* weights, const float* norm, const float* masks,
        const unsigned* ids, float* output, DraftEmbed p) {
    const unsigned row = blockIdx.x;
    output += size_t(row) * p.width;
    if (row != 0) {
        masks += size_t(row - 1) * p.width;
        for (unsigned i = threadIdx.x; i < p.width; i += 64) output[i] = masks[i];
        return;
    }
    weights += size_t(ids[0]) * inkling::row_bytes(p.kind, p.width);
    float sum = 0;
    for (unsigned i = threadIdx.x; i < p.width; i += 64) {
        float x = inkling::read_weight(weights, p.kind, i);
        output[i] = x;
        sum = fmaf(x, x, sum);
    }
    const float inverse = rsqrtf(zinc_block_reduce_sum_all(sum) / p.width + p.eps);
    for (unsigned i = threadIdx.x; i < p.width; i += 64) output[i] = output[i] * inverse * norm[i];
}

struct Matmul { unsigned rows, cols, count; };
extern "C" __global__ void dspark_q8_wmma(
        const unsigned char* weights, const inkling::Q8Block* activations, float* output, Matmul p) {
    const unsigned wave = threadIdx.x / 32;
    const unsigned lane = threadIdx.x & 31;
    const unsigned half = lane / 16;
    const unsigned row_base = blockIdx.x * 32 + wave * 16;
    const unsigned weight_row = row_base + (lane & 15);
    const unsigned column = blockIdx.y * 16 + (lane & 15);
    const bool valid_column = column < p.count;
    const unsigned blocks = p.cols / 32;
    const auto* wr = weights + size_t(min(weight_row, p.rows - 1)) * blocks * 34;
    const auto* src = activations + size_t(valid_column ? column : 0) * blocks;

    using I32x2 = __attribute__((__vector_size__(8))) int;
    using I32x8 = __attribute__((__vector_size__(32))) int;
    float accum[8] = {};
    for (unsigned block = 0; block < blocks; ++block) {
        const unsigned char* row = wr + block * 34;
        const float d = weight_row < p.rows ? inkling::half_at(row) : 0.0f;
        const auto* left = row + 2 + half * 8;
        const I32x2 wlo = {int(inkling::inkling_load_q4_word(left)), int(inkling::inkling_load_q4_word(left + 4))};
        const I32x2 whi = {int(inkling::inkling_load_q4_word(left + 16)), int(inkling::inkling_load_q4_word(left + 20))};
        const auto* x = src + block;
        const I32x2 xlo = {x->words[half * 2], x->words[half * 2 + 1]};
        const I32x2 xhi = {x->words[half * 2 + 4], x->words[half * 2 + 5]};
        I32x8 dot = {};
        dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, wlo, true, xlo, dot, true);
        dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, whi, true, xhi, dot, true);
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i)
            accum[i] = fmaf(__shfl(d, half * 8 + i, 32) * x->scale, float(dot[i]), accum[i]);
    }
    if (!valid_column) return;
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned row = row_base + half * 8 + i;
        if (row < p.rows) output[size_t(column) * p.rows + row] = accum[i];
    }
}

struct Rope { unsigned heads, dim, position; float eps; };
extern "C" __global__ void dspark_qk_rope(float* x, const float* norm, const float* frequencies, Rope p) {
    const unsigned i = threadIdx.x;
    x += (size_t(blockIdx.y) * p.heads + blockIdx.x) * p.dim;
    const unsigned h = p.dim / 2;
    float a = x[i], b = x[i + h];
    const float inverse = rsqrtf(zinc_block_reduce_sum_all(fmaf(a, a, b * b)) / p.dim + p.eps);
    a *= inverse * norm[i];
    b *= inverse * norm[i + h];
    float sn, cs;
    sincosf(frequencies[i] * float(p.position + blockIdx.y), &sn, &cs);
    x[i] = fmaf(-b, sn, a * cs);
    x[i + h] = fmaf(a, sn, b * cs);
}

struct Kv { unsigned width, position, capacity; };
extern "C" __global__ void dspark_store_kv(
        const float* k, const float* v, __half* keys, __half* values, Kv p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i >= p.width) return;
    const size_t source = size_t(blockIdx.y) * p.width + i;
    const size_t dest = size_t((p.position + blockIdx.y) % p.capacity) * p.width + i;
    keys[dest] = __float2half_rn(k[source]);
    values[dest] = __float2half_rn(v[source]);
}

constexpr unsigned tile_size = 128;
struct Attention {
    unsigned dim, heads, kv_heads, position, capacity, first, rows, tiles;
};

__device__ __forceinline__ const __half* kv_row(
        const __half* prefix, const __half* block, unsigned pos, unsigned head, Attention p) {
    const __half* base = pos < p.position ? prefix + size_t(pos % p.capacity) * p.kv_heads * p.dim
                                          : block + size_t(pos - p.position) * p.kv_heads * p.dim;
    return base + head * p.dim;
}

extern "C" __global__ void dspark_attention(
        const float* q, const __half* keys, const __half* values,
        const __half* block_keys, const __half* block_values,
        float* partial, Attention p) {
    const unsigned kv_head = blockIdx.x / (p.heads / p.kv_heads);
    const unsigned first = p.first + blockIdx.z * tile_size;
    const unsigned len = min(tile_size, p.position + p.rows - first);
    const size_t row = size_t(blockIdx.y) * p.heads + blockIdx.x;
    q += row * p.dim;
    partial += (row * p.tiles + blockIdx.z) * (p.dim + 2);

    const unsigned lane = threadIdx.x & 15;
    __shared__ float scores[tile_size];
    for (unsigned i = threadIdx.x / 16; i < len; i += 4) {
        const __half* k = kv_row(keys, block_keys, first + i, kv_head, p);
        float dot = 0;
        for (unsigned d = lane; d < p.dim; d += 16) dot = fmaf(q[d], __half2float(k[d]), dot);
        for (unsigned step = 8; step; step >>= 1) dot += __shfl_down(dot, step, 16);
        if (lane == 0) scores[i] = dot * rsqrtf(float(p.dim));
    }
    __syncthreads();
    float maximum = inkling::negative_infinity;
    for (unsigned i = threadIdx.x; i < len; i += 64) maximum = fmaxf(maximum, scores[i]);
    const float max_row = zinc_block_reduce_max(maximum);
    __shared__ float max_score;
    if (threadIdx.x == 0) max_score = max_row;
    __syncthreads();
    float sum = 0;
    for (unsigned i = threadIdx.x; i < len; i += 64) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }
    const float mass = zinc_block_reduce_sum(sum);
    __syncthreads();
    for (unsigned d = threadIdx.x; d < p.dim; d += 64) {
        float x = 0;
        for (unsigned i = 0; i < len; ++i) {
            const __half* v = kv_row(values, block_values, first + i, kv_head, p);
            x = fmaf(scores[i], __half2float(v[d]), x);
        }
        partial[d] = x;
    }
    if (threadIdx.x == 0) {
        partial[p.dim] = max_score;
        partial[p.dim + 1] = mass;
    }
}

extern "C" __global__ void dspark_attention_merge(const float* partial, float* out, Attention p) {
    const size_t row = size_t(blockIdx.y) * p.heads + blockIdx.x;
    const unsigned stride = p.dim + 2;
    partial += row * p.tiles * stride;
    out += row * p.dim;
    __shared__ float max_score, inverse;
    if (threadIdx.x == 0) {
        float high = partial[p.dim];
        for (unsigned i = 1; i < p.tiles; ++i) high = fmaxf(high, partial[i * stride + p.dim]);
        float mass = 0;
        for (unsigned i = 0; i < p.tiles; ++i)
            mass = fmaf(expf(partial[i * stride + p.dim] - high), partial[i * stride + p.dim + 1], mass);
        max_score = high;
        inverse = 1.0f / mass;
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < p.dim; d += 64) {
        float value = 0;
        for (unsigned i = 0; i < p.tiles; ++i) {
            const float* tile = partial + i * stride;
            value = fmaf(expf(tile[p.dim] - max_score), tile[d], value);
        }
        out[d] = value * inverse;
    }
}

struct Count { unsigned n; };
extern "C" __global__ void dspark_add(float* x, const float* y, Count p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i < p.n) x[i] += y[i];
}

extern "C" __global__ void dspark_swiglu(float* gate, const float* up, Count p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i < p.n) gate[i] = gate[i] * inkling::sigmoid(gate[i]) * up[i];
}

struct Markov { unsigned width, rank, slot; };
extern "C" __global__ void dspark_joint(
        const float* state, const unsigned short* w1, const float* hidden, const float* context,
        const unsigned* tokens, float* out, Markov p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i < p.rank) out[i] = p.slot == 1 ? 0.0f : state[i];
    else if (i < 2 * p.rank) {
        const unsigned tok = tokens[p.slot - 1];
        out[i] = __uint_as_float(unsigned(w1[size_t(tok) * p.rank + i - p.rank]) << 16);
    } else if (i < 2 * p.rank + p.width) {
        out[i] = hidden[size_t(p.slot) * p.width + i - 2 * p.rank];
    } else if (i < 2 * (p.rank + p.width)) {
        out[i] = context[i - 2 * p.rank - p.width];
    }
}

extern "C" __global__ void dspark_markov_update(
        const float* activation, const float* bias, float* state, float* markov, Markov p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i >= p.rank) return;
    const float gate = inkling::sigmoid(activation[i] + bias[i]);
    const float candidate = tanhf(activation[i + p.rank] + bias[i + p.rank]);
    const float old = p.slot == 1 ? 0.0f : state[i];
    state[i] = fmaf(gate, old - candidate, candidate);
    markov[i] = tanhf(activation[i + 2 * p.rank] + bias[i + 2 * p.rank]);
}

struct Head { unsigned vocab, rank, slot; };
extern "C" __global__ void dspark_markov_logits(
        const unsigned short* weights, const float* bias, const float* markov,
        const float* base, float* logits, Head p) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 16;
    if (row >= p.vocab) return;
    const unsigned lane = threadIdx.x & 15;
    weights += size_t(row) * p.rank;
    float4 accum = {};
    for (unsigned i = lane * 4; i < p.rank; i += 64) {
        const float4 w = make_float4(__uint_as_float(unsigned(weights[i]) << 16),
                                    __uint_as_float(unsigned(weights[i + 1]) << 16),
                                    __uint_as_float(unsigned(weights[i + 2]) << 16),
                                    __uint_as_float(unsigned(weights[i + 3]) << 16));
        inkling::accumulate4(w, reinterpret_cast<const float4*>(markov)[i / 4], accum);
    }
    float sum = inkling::sum4(accum);
    for (unsigned i = 8; i; i >>= 1) sum += __shfl_down(sum, i, 16);
    if (lane == 0) logits[row] = base[size_t(p.slot) * p.vocab + row] + (sum + bias[row]);
}

struct Pick { unsigned partials, slot; };
extern "C" __global__ void dspark_pick(const uint2* partial, unsigned* tokens, Pick p) {
    const unsigned t = threadIdx.x;
    float score = inkling::negative_infinity;
    unsigned id = 0xffffffffu;
    for (unsigned i = t; i < p.partials; i += 64) {
        const float x = __uint_as_float(partial[i].x);
        const unsigned j = partial[i].y;
        if (x > score || (x == score && j < id)) { score = x; id = j; }
    }
    __shared__ float scores[64];
    __shared__ unsigned ids[64];
    scores[t] = score; ids[t] = id;
    __syncthreads();
    for (unsigned i = 32; i; i >>= 1) {
        if (t < i && (scores[t + i] > scores[t] || (scores[t + i] == scores[t] && ids[t + i] < ids[t]))) {
            scores[t] = scores[t + i];
            ids[t] = ids[t + i];
        }
        __syncthreads();
    }
    if (t == 0) tokens[p.slot] = ids[0];
}
} // namespace inkling_dspark
