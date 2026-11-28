// Inkling 2 Mini text decoder on ROCm. This source shares the accelerator
// module with kernels.cu, including its half conversions and reductions.

namespace inkling {

constexpr float negative_infinity = -__builtin_inff();

__device__ __forceinline__ float half_at(const unsigned char* data) {
    return zinc_half_to_float(static_cast<unsigned short>(data[0] | (data[1] << 8)));
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

enum Weight : unsigned { F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, BF16 = 30 };

__device__ __forceinline__ unsigned row_bytes(unsigned type, unsigned cols) {
    switch (type) {
        case F32: return cols * 4;
        case F16: case BF16: return cols * 2;
        case Q4_0: return cols / 32 * 18;
        case Q8_0: return cols / 32 * 34;
        case Q4_K: return cols / 256 * 144;
        case Q5_K: return cols / 256 * 176;
        case Q6_K: return cols / 256 * 210;
        default: return 0;
    }
}

__device__ __forceinline__ float read_weight(const unsigned char* row, unsigned type, unsigned i) {
    switch (type) {
        case F32: return reinterpret_cast<const float*>(row)[i];
        case F16: return half_at(row + 2 * i);
        case BF16: return __uint_as_float(unsigned(row[2 * i] | (row[2 * i + 1] << 8)) << 16);
        case Q4_0: {
            const auto* block = row + (i / 32) * 18;
            const unsigned packed = block[2 + (i & 15)];
            return half_at(block) * (int((packed >> ((i & 16) ? 4 : 0)) & 15) - 8);
        }
        case Q8_0: {
            const auto* block = row + (i / 32) * 34;
            return half_at(block) * reinterpret_cast<const signed char*>(block)[2 + (i & 31)];
        }
        case Q4_K: case Q5_K: {
            const bool five = type == Q5_K;
            const auto* block = row + (i / 256) * (five ? 176 : 144);
            const unsigned sub = (i & 255) / 32;
            const unsigned lane = i & 31;
            const auto* scale = block + 4;
            const unsigned sc = sub < 4 ? scale[sub] & 63 :
                (scale[sub + 4] & 15) | ((scale[sub - 4] >> 6) << 4);
            const unsigned mn = sub < 4 ? scale[sub + 4] & 63 :
                (scale[sub + 4] >> 4) | ((scale[sub] >> 6) << 4);
            const unsigned packed = block[(five ? 48 : 16) + (sub / 2) * 32 + lane];
            const unsigned q = ((packed >> ((sub & 1) * 4)) & 15) |
                (five ? ((unsigned(block[16 + lane]) >> sub) & 1) << 4 : 0);
            return half_at(block) * sc * q - half_at(block + 2) * mn;
        }
        case Q6_K: {
            const auto* block = row + (i / 256) * 210;
            const unsigned half = (i & 255) / 128;
            const unsigned part = (i & 127) / 32;
            const unsigned lane = i & 31;
            const unsigned lo = block[half * 64 + (part & 1) * 32 + lane];
            const unsigned hi = block[128 + half * 32 + lane];
            const int q = int(((lo >> ((part / 2) * 4)) & 15) | (((hi >> (part * 2)) & 3) << 4)) - 32;
            const auto scale = reinterpret_cast<const signed char*>(block)[192 + half * 8 + part * 2 + lane / 16];
            return half_at(block + 208) * scale * q;
        }
        default: return 0.0f;
    }
}

struct Matrix { unsigned rows, cols, type, routes, per_route, blocked; };

template<unsigned Tokens, bool Grouped = false>
__device__ void matvec(const unsigned char* w, const float* x, const unsigned* ids, float* y, Matrix p,
                       const unsigned* order = nullptr, const unsigned* groups = nullptr) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 16;
    const unsigned lane = threadIdx.x & 15;
    const unsigned item = blockIdx.y * Tokens;
    const unsigned expert = Grouped ? groups[blockIdx.y] : p.routes ? ids[item] : 0;
    if (row >= p.rows || expert == 0xffffffffu) return;
    w += (size_t(expert) * p.rows + row) * row_bytes(p.type, p.cols);

    const float* inputs[Tokens];
    unsigned outputs[Tokens];
    #pragma unroll
    for (unsigned j = 0; j < Tokens; ++j) {
        if constexpr (Grouped) {
            const unsigned packed = order[item + j];
            const unsigned token = packed >> 16;
            const unsigned slot = packed & 0xffffu;
            const unsigned route = token * p.routes + slot;
            outputs[j] = packed == 0xffffffffu ? 0xffffffffu : route;
            const unsigned source = packed == 0xffffffffu ? 0 : p.per_route ? route : token;
            inputs[j] = x + size_t(source) * p.cols;
        } else {
            const unsigned source = p.routes && !p.per_route ? item / p.routes : item + j;
            inputs[j] = x + size_t(source) * p.cols;
            outputs[j] = item + j;
        }
    }
    float values[Tokens] = {};
    if (p.type == Q4_0) {
        // A lane loads a byte once to decode both packed half-blocks.
        for (unsigned b = 0; b < p.cols / 32; ++b) {
            const auto* wb = w + b * 18;
            const unsigned packed = wb[2 + lane];
            const float scale = half_at(wb);
            const float lo = scale * (int(packed & 15) - 8);
            const float hi = scale * (int(packed >> 4) - 8);
            #pragma unroll
            for (unsigned j = 0; j < Tokens; ++j) {
                const float* xb = inputs[j] + b * 32;
                values[j] = fmaf(lo, xb[lane], values[j]);
                values[j] = fmaf(hi, xb[lane + 16], values[j]);
            }
        }
    } else {
        for (unsigned i = lane; i < p.cols; i += 16) {
            const float weight = read_weight(w, p.type, i);
            #pragma unroll
            for (unsigned j = 0; j < Tokens; ++j) {
                values[j] = fmaf(weight, inputs[j][i], values[j]);
            }
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < Tokens; ++j) {
        for (unsigned step = 8; step; step >>= 1) values[j] += __shfl_down(values[j], step, 16);
        if (lane == 0 && outputs[j] != 0xffffffffu) y[size_t(outputs[j]) * p.rows + row] = values[j];
    }
}

extern "C" __global__ void inkling_matvec(
        const unsigned char* w, const float* x, const unsigned* ids, float* y, Matrix p) {
    matvec<1>(w, x, ids, y, p);
}

extern "C" __global__ void inkling_matvec4(
        const unsigned char* w, const float* x, const unsigned* ids, float* y, Matrix p) {
    matvec<4>(w, x, ids, y, p);
}

extern "C" __global__ void inkling_experts4(
        const unsigned char* w, const float* x, const unsigned* order, const unsigned* groups,
        float* y, Matrix p) {
    matvec<4, true>(w, x, nullptr, y, p, order, groups);
}

__device__ __forceinline__ void accumulate4(float4 a, float4 b, float4& sum) {
    sum.x = fmaf(a.x, b.x, sum.x);
    sum.y = fmaf(a.y, b.y, sum.y);
    sum.z = fmaf(a.z, b.z, sum.z);
    sum.w = fmaf(a.w, b.w, sum.w);
}

__device__ __forceinline__ float sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }

// The router and loop controllers project to a small number of rows. Give
// each row a wave and load aligned vectors to expose more memory parallelism.
extern "C" __global__ void inkling_f32_decode(
        const float4* w, const float4* x, float* y, Matrix p) {
    const unsigned row = blockIdx.x * 2 + threadIdx.x / 32;
    if (row >= p.rows) return;
    const unsigned lane = threadIdx.x & 31;
    const unsigned cols = p.cols / 4;
    w += size_t(row) * cols;
    x += size_t(blockIdx.y) * cols;
    y += size_t(blockIdx.y) * p.rows;
    float4 dot = {};
    for (unsigned i = lane; i < cols; i += 32) accumulate4(w[i], x[i], dot);
    float result = sum4(dot);
    for (unsigned step = 16; step; step >>= 1) result += __shfl_down(result, step, 32);
    if (lane == 0) y[row] = result;
}

// Packed activations for Q4_0 decode. Each group of 32 channels has its own
// scale, so dot4 accumulation preserves the weight block's scale boundaries.
struct Q8Block { float scale; int words[8]; };
struct Quantize { unsigned cols; };
extern "C" __global__ void inkling_quantize(const float* x, Q8Block* y, Quantize p) {
    const unsigned chunk = blockIdx.x * 2 + threadIdx.x / 32;
    if (chunk >= p.cols / 32) return;
    const unsigned lane = threadIdx.x & 31;
    const float value = x[size_t(blockIdx.y) * p.cols + chunk * 32 + lane];
    float maximum = fabsf(value);
    for (unsigned step = 16; step; step >>= 1) maximum = fmaxf(maximum, __shfl_xor(maximum, step, 32));
    auto* output = y + size_t(blockIdx.y) * (p.cols / 32) + chunk;
    if (lane == 0) output->scale = maximum / 127.0f;
    reinterpret_cast<signed char*>(output->words)[lane] =
        static_cast<signed char>(maximum > 0 ? __float2int_rn(value * (127.0f / maximum)) : 0);
}

__device__ __forceinline__ int inkling_signed_nibbles(unsigned packed) {
    const unsigned x = (packed & 0x0f0f0f0fu) ^ 0x08080808u;
    return static_cast<int>(x | (x & 0x08080808u) * 30u);
}

__device__ __forceinline__ unsigned inkling_load_q4_word(const unsigned char* address) {
    const auto* pair = reinterpret_cast<const unsigned short*>(address);
    return unsigned(pair[0]) | (unsigned(pair[1]) << 16);
}

struct Repack { unsigned blocks; };
// Reorder four complete rows in place. Each tile consists of 16 aligned
// quant blocks followed by their 16 half scales. No model-sized copy is needed.
extern "C" __global__ void inkling_repack_q4(unsigned short* weights, Repack p) {
    extern __shared__ unsigned short original[];
    const unsigned blocks_per_group = 4 * p.blocks;
    weights += size_t(blockIdx.x) * blocks_per_group * 9;
    for (unsigned i = threadIdx.x; i < blocks_per_group * 9; i += 64) original[i] = weights[i];
    __syncthreads();
    for (unsigned i = threadIdx.x; i < blocks_per_group; i += 64) {
        const unsigned block = i % p.blocks;
        const unsigned row = i / p.blocks;
        const unsigned slot = row * 4 + block % 4;
        unsigned short* tile = weights + (block / 4) * (16 * 9);
        tile[128 + slot] = original[i * 9];
        #pragma unroll
        for (unsigned word = 0; word < 8; ++word) tile[slot * 8 + word] = original[i * 9 + 1 + word];
    }
}

__device__ __forceinline__ const unsigned char* blocked_q4_tile(
        const unsigned char* weights, size_t row, unsigned blocks, unsigned block) {
    return weights + ((row / 4) * blocks + (block & ~3u)) * (4 * 18);
}

extern "C" __global__ void inkling_q4_0_q8(
        const unsigned char* weights, const Q8Block* x, const unsigned* ids, float* y, Matrix p) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 16;
    if (row >= p.rows) return;
    const unsigned lane = threadIdx.x & 15;
    const unsigned word = lane & 3;
    const unsigned route = blockIdx.y;
    const unsigned expert = p.routes ? ids[route] : 0;
    const unsigned input = p.routes && !p.per_route ? route / p.routes : route;
    const unsigned blocks = p.cols / 32;
    const size_t weight_row = size_t(expert) * p.rows + row;
    const auto* weight_data = weights + weight_row * blocks * 18;
    x += size_t(input) * blocks;
    float sum = 0;
    for (unsigned i = lane / 4; i < blocks; i += 4) {
        unsigned packed;
        float scale;
        if (p.blocked) {
            const auto* tile = blocked_q4_tile(weights, weight_row, blocks, i);
            const unsigned slot = (row & 3) * 4 + (i & 3);
            packed = reinterpret_cast<const unsigned*>(tile)[slot * 4 + word];
            scale = zinc_half_to_float(reinterpret_cast<const unsigned short*>(tile + 256)[slot]);
        } else {
            const auto* block = weight_data + i * 18;
            // Q4_0 quants have 16-bit alignment in GGUF.
            packed = inkling_load_q4_word(block + 2 + word * 4);
            scale = half_at(block);
        }
        const int lo = inkling_signed_nibbles(packed);
        const int hi = inkling_signed_nibbles(packed >> 4);
        const int dot = __dp4a(hi, x[i].words[4 + word], __dp4a(lo, x[i].words[word], 0));
        sum = fmaf(scale * x[i].scale, float(dot), sum);
    }
    for (unsigned step = 8; step; step >>= 1) sum += __shfl_down(sum, step, 16);
    if (lane == 0) y[size_t(route) * p.rows + row] = sum;
}

struct MatrixTile { unsigned rows, cols, count, routes, per_route, blocked; };

// Map four draft rows to one weight row. Sparse rows are grouped by expert;
// padding is skipped so weights and their scales are loaded just once.
extern "C" __global__ void inkling_q4_0_q8_shared4(
        const unsigned char* weights, const Q8Block* x, const unsigned* order,
        const unsigned* groups, float* output, MatrixTile p) {
    constexpr unsigned missing = 0xffffffffu;
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 16;
    const unsigned expert = p.routes ? groups[blockIdx.y] : 0;
    if (row >= p.rows || expert == missing) return;
    const unsigned lane = threadIdx.x & 15;
    const unsigned word = lane & 3;
    const unsigned blocks = p.cols / 32;
    const size_t weight_row = size_t(expert) * p.rows + row;
    const auto* weight_data = weights + weight_row * blocks * 18;
    const Q8Block* activations[4];
    unsigned offsets[4];
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned slot = blockIdx.y * 4 + j;
        const unsigned packed = p.routes ? order[slot] : slot;
        const unsigned token = p.routes ? packed >> 16 : slot;
        const bool valid = p.routes ? packed != missing : slot < p.count;
        const unsigned route = p.routes ? token * p.routes + (packed & 0xffffu) : token;
        offsets[j] = valid ? route : missing;
        activations[j] = x + size_t(valid ? (p.per_route ? route : token) : 0) * blocks;
    }
    float sums[4] = {};
    for (unsigned i = lane / 4; i < blocks; i += 4) {
        unsigned packed;
        float scale;
        if (p.blocked) {
            const auto* tile = blocked_q4_tile(weights, weight_row, blocks, i);
            const unsigned slot = (row & 3) * 4 + (i & 3);
            packed = reinterpret_cast<const unsigned*>(tile)[slot * 4 + word];
            scale = zinc_half_to_float(reinterpret_cast<const unsigned short*>(tile + 256)[slot]);
        } else {
            const auto* block = weight_data + i * 18;
            packed = inkling_load_q4_word(block + 2 + word * 4);
            scale = half_at(block);
        }
        const int lo = inkling_signed_nibbles(packed);
        const int hi = inkling_signed_nibbles(packed >> 4);
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            if (offsets[j] == missing) continue;
            const auto* block = activations[j] + i;
            const int dot = __dp4a(hi, block->words[4 + word], __dp4a(lo, block->words[word], 0));
            sums[j] = fmaf(scale * block->scale, float(dot), sums[j]);
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        if (offsets[j] == missing) continue;
        for (unsigned step = 8; step; step >>= 1) sums[j] += __shfl_down(sums[j], step, 16);
        if (lane == 0) output[size_t(offsets[j]) * p.rows + row] = sums[j];
    }
}

// Each wave computes 16 weight rows by 16 activation rows, sharing an
// expert's weights across its routes during prefill. Dequantize each 32-wide
// integer dot with the corresponding weight and activation scales.
extern "C" __global__ void inkling_q4_0_wmma(
        const unsigned char* weights, const Q8Block* activations, const unsigned* order,
        const unsigned* groups, float* output, MatrixTile p) {
    constexpr unsigned missing = 0xffffffffu;
    const unsigned expert = p.routes ? groups[blockIdx.y] : 0;
    if (expert == missing) return;
    const unsigned wave = threadIdx.x / 32;
    const unsigned lane = threadIdx.x & 31;
    const unsigned half = lane / 16;
    const unsigned row_base = blockIdx.x * 32 + wave * 16;
    const unsigned weight_row = row_base + (lane & 15);
    const unsigned column = blockIdx.y * 16 + (lane & 15);
    const unsigned packed_route = p.routes ? order[column] : 0;
    const bool valid_column = p.routes ? packed_route != missing : column < p.count;
    const unsigned token = p.routes ? packed_route >> 16 : column;
    const unsigned route = p.routes ? token * p.routes + (packed_route & 0xffff) : column;
    const unsigned source = valid_column ? (p.per_route ? route : token) : 0;
    const unsigned blocks = p.cols / 32;
    const size_t row_index = size_t(expert) * p.rows + min(weight_row, p.rows - 1);
    const auto* wr = weights + row_index * blocks * 18;
    const auto* src = activations + size_t(source) * blocks;

    using I32x2 = __attribute__((__vector_size__(8))) int;
    using I32x8 = __attribute__((__vector_size__(32))) int;
    float accum[8] = {};
    for (unsigned block = 0; block < blocks; ++block) {
        float weight_scale;
        unsigned bits0, bits1;
        if (p.blocked) {
            const auto* tile = blocked_q4_tile(weights, row_index, blocks, block);
            const unsigned slot = (row_index & 3) * 4 + (block & 3);
            const auto* words = reinterpret_cast<const unsigned*>(tile) + slot * 4 + half * 2;
            bits0 = words[0];
            bits1 = words[1];
            weight_scale = zinc_half_to_float(reinterpret_cast<const unsigned short*>(tile + 256)[slot]);
        } else {
            const unsigned char* row = wr + block * 18;
            bits0 = inkling_load_q4_word(row + 2 + half * 8);
            bits1 = inkling_load_q4_word(row + 6 + half * 8);
            weight_scale = half_at(row);
        }
        if (weight_row >= p.rows) weight_scale = 0.0f;
        const I32x2 lo = {inkling_signed_nibbles(bits0), inkling_signed_nibbles(bits1)};
        const I32x2 hi = {inkling_signed_nibbles(bits0 >> 4), inkling_signed_nibbles(bits1 >> 4)};
        const Q8Block* x = src + block;
        const I32x2 xlo = {x->words[half * 2], x->words[half * 2 + 1]};
        const I32x2 xhi = {x->words[half * 2 + 4], x->words[half * 2 + 5]};
        I32x8 dot = {};
        dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, lo, true, xlo, dot, true);
        dot = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, hi, true, xhi, dot, true);
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const float d = __shfl(weight_scale, half * 8 + i, 32);
            accum[i] = fmaf(d * x->scale, float(dot[i]), accum[i]);
        }
    }
    if (!valid_column) return;
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned row = row_base + half * 8 + i;
        if (row < p.rows) output[size_t(route) * p.rows + row] = accum[i];
    }
}

struct Embed { unsigned hidden, type; };
extern "C" __global__ void inkling_embed(
        const unsigned char* embeddings, const unsigned* ids, float* out, Embed p) {
    const unsigned channel = blockIdx.x * 64 + threadIdx.x;
    if (channel >= p.hidden) return;
    const auto* row = embeddings + size_t(ids[1 + blockIdx.y]) * row_bytes(p.type, p.hidden);
    out[size_t(blockIdx.y) * p.hidden + channel] = read_weight(row, p.type, channel);
}

struct Norm { unsigned width, offset; float eps; unsigned weighted; };

// Inkling 2 Mini's normalizations share one native width. Keep the input
// vectors in registers through the reduction to avoid re-reading the row.
constexpr unsigned mini_width = 2304;
__device__ __forceinline__ void norm_mini(const float* x, const float* weight, float* y, float eps) {
    constexpr unsigned vectors = mini_width / 256;
    const auto* in4 = reinterpret_cast<const float4*>(x);
    const auto* w4 = reinterpret_cast<const float4*>(weight);
    auto* out4 = reinterpret_cast<float4*>(y);
    float4 values[vectors];
    float4 sum = {};
    #pragma unroll
    for (unsigned j = 0; j < vectors; ++j) {
        values[j] = in4[threadIdx.x + j * 64];
        accumulate4(values[j], values[j], sum);
    }
    const float inv = rsqrtf(zinc_block_reduce_sum_all(sum4(sum)) / mini_width + eps);
    #pragma unroll
    for (unsigned j = 0; j < vectors; ++j) {
        const unsigned i = threadIdx.x + j * 64;
        const float4 value = values[j];
        const float4 scale = weight ? w4[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        out4[i] = make_float4(value.x * inv * scale.x, value.y * inv * scale.y,
                              value.z * inv * scale.z, value.w * inv * scale.w);
    }
}

extern "C" __global__ void inkling_norm(const float* x, const float* weight, float* y, Norm p) {
    x += size_t(blockIdx.x) * p.width;
    y += size_t(blockIdx.x + p.offset) * p.width;
    if (p.width == mini_width) {
        norm_mini(x, p.weighted ? weight : nullptr, y, p.eps);
        return;
    }
    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < p.width; i += 64) sum = fmaf(x[i], x[i], sum);
    const float inv = rsqrtf(zinc_block_reduce_sum_all(sum) / p.width + p.eps);
    for (unsigned i = threadIdx.x; i < p.width; i += 64) {
        y[i] = (x[i] * inv) * (p.weighted ? weight[i] : 1.0f);
    }
}

struct Mix { unsigned hidden, capacity, first_token, blocks; unsigned block_ids[16]; };

extern "C" __global__ void inkling_mix(const float* current, const float* saved, const float* query, float* out, Mix p) {
    const unsigned token = p.first_token + blockIdx.x;
    current += size_t(token) * p.hidden;
    query += size_t(blockIdx.y) * p.hidden;
    out += (size_t(blockIdx.y) * p.capacity + blockIdx.x) * p.hidden;

    // Both waves reduce independent residual-state scores. Only the current
    // partial block needs normalization, and one barrier publishes the scores.
    const unsigned lane = threadIdx.x & 31;
    const unsigned wave = threadIdx.x / 32;
    const float scale = rsqrtf(float(p.hidden));
    __shared__ float weights[17];
    __shared__ float inv;
    for (unsigned b = wave; b <= p.blocks; b += 2) {
        const bool partial = b == p.blocks;
        const float* state = partial ? current : saved + (size_t(p.block_ids[b]) * p.capacity + token) * p.hidden;
        float dot = 0.0f, norm = 0.0f;
        if (p.hidden % 4 == 0) {
            const auto* query4 = reinterpret_cast<const float4*>(query);
            const auto* state4 = reinterpret_cast<const float4*>(state);
            float4 d = {}, n = {};
            for (unsigned i = lane; i < p.hidden / 4; i += 32) {
                const float4 value = state4[i];
                accumulate4(value, query4[i], d);
                if (partial) accumulate4(value, value, n);
            }
            dot = sum4(d);
            norm = sum4(n);
        } else {
            for (unsigned i = lane; i < p.hidden; i += 32) {
                dot = fmaf(state[i], query[i], dot);
                if (partial) norm = fmaf(state[i], state[i], norm);
            }
        }
        for (unsigned step = 16; step; step >>= 1) {
            dot += __shfl_down(dot, step, 32);
            if (partial) norm += __shfl_down(norm, step, 32);
        }
        if (lane == 0) {
            if (partial) inv = rsqrtf(norm / p.hidden + 1e-6f);
            weights[b] = (partial ? dot * inv : dot) * scale;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = weights[0];
        for (unsigned b = 1; b <= p.blocks; ++b) maximum = fmaxf(maximum, weights[b]);
        float sum = 0.0f;
        for (unsigned b = 0; b <= p.blocks; ++b) {
            weights[b] = expf(weights[b] - maximum);
            sum += weights[b];
        }
        for (unsigned b = 0; b <= p.blocks; ++b) weights[b] /= sum;
    }
    __syncthreads();
    if (p.hidden % 4 == 0) {
        const unsigned width = p.hidden / 4;
        const auto* current4 = reinterpret_cast<const float4*>(current);
        auto* output4 = reinterpret_cast<float4*>(out);
        for (unsigned i = threadIdx.x; i < width; i += 64) {
            const float4 v = current4[i];
            float4 result = make_float4(v.x * inv * weights[p.blocks], v.y * inv * weights[p.blocks],
                                        v.z * inv * weights[p.blocks], v.w * inv * weights[p.blocks]);
            for (unsigned b = 0; b < p.blocks; ++b) {
                const auto* bank = reinterpret_cast<const float4*>(saved) +
                    (size_t(p.block_ids[b]) * p.capacity + token) * width;
                const float w = weights[b];
                accumulate4(bank[i], make_float4(w, w, w, w), result);
            }
            output4[i] = result;
        }
        return;
    }
    for (unsigned i = threadIdx.x; i < p.hidden; i += 64) {
        float val = current[i] * inv * weights[p.blocks];
        for (unsigned b = 0; b < p.blocks; ++b) {
            val = fmaf(saved[(size_t(p.block_ids[b]) * p.capacity + token) * p.hidden + i], weights[b], val);
        }
        out[i] = val;
    }
}

struct PairNorm { unsigned hidden; };
extern "C" __global__ void inkling_pair_norm(const float* input, const float* message, float* pair, PairNorm p) {
    const size_t offset = size_t(blockIdx.x) * p.hidden;
    const float* x = (blockIdx.y ? message : input) + offset;
    float* y = pair + offset * 2 + blockIdx.y * p.hidden;
    if (p.hidden == mini_width) {
        norm_mini(x, nullptr, y, 1e-6f);
        return;
    }
    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < p.hidden; i += 64) sum = fmaf(x[i], x[i], sum);
    const float inv = rsqrtf(zinc_block_reduce_sum_all(sum) / p.hidden + 1e-6f);
    for (unsigned i = threadIdx.x; i < p.hidden; i += 64) y[i] = x[i] * inv;
}

struct Count { unsigned n; };
extern "C" __global__ void inkling_control(
        float* input, const float* message, const float* gain, const float* forget, Count p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i < p.n) input[i] = fmaf(tanhf(gain[i]), message[i], (1.0f - tanhf(forget[i])) * input[i]);
}

struct ControlRows { unsigned width, rank; };
extern "C" __global__ void inkling_control_rows(
        float* input, const float* message, const float* low_rank, const float* gain,
        const float* forget, ControlRows p) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 16;
    if (row >= p.width) return;
    const unsigned lane = threadIdx.x & 15;
    const size_t offset = size_t(row) * p.rank;
    input += size_t(blockIdx.y) * p.width;
    message += size_t(blockIdx.y) * p.width;
    low_rank += size_t(blockIdx.y) * p.rank;
    float g = 0.0f, f = 0.0f;
    for (unsigned i = lane; i < p.rank; i += 16) {
        g = fmaf(gain[offset + i], low_rank[i], g);
        f = fmaf(forget[offset + i], low_rank[i], f);
    }
    for (unsigned step = 8; step; step >>= 1) {
        g += __shfl_down(g, step, 16);
        f += __shfl_down(f, step, 16);
    }
    if (lane == 0) input[row] = fmaf(tanhf(g), message[row], (1.0f - tanhf(f)) * input[row]);
}

extern "C" __global__ void inkling_copy(const float* input, float* out, Count p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i < p.n) out[i] = input[i];
}

struct Convolution {
    unsigned channels, taps, count, history_offset, dynamic_head_dim, residual, trace_offset, trace_stride;
    float scale;
};

extern "C" __global__ void inkling_conv(
        const float* raw, const float* taps, const float* dynamic_taps, float* history,
        const float* residual, float* output, const unsigned* step, float* trace, Convolution p) {
    const unsigned c = blockIdx.x * 64 + threadIdx.x;
    if (c >= p.channels) return;
    history += (size_t(p.history_offset) + c) * (p.taps - 1);
    const unsigned dynamic_stride = p.dynamic_head_dim ? (p.channels / p.dynamic_head_dim) * p.taps : 0;
    const unsigned dynamic_channel = p.dynamic_head_dim ? (c / p.dynamic_head_dim) * p.taps : 0;
    const float* kernel = taps + c * p.taps;
    const unsigned position = step[0];
    if (p.trace_stride) {
        trace += size_t(p.trace_offset) + c;
        for (unsigned i = 0; i + 1 < p.taps; ++i) {
            const bool valid = position + i + 1 >= p.taps;
            trace[size_t(i) * p.trace_stride] = valid ? history[i] : 0.0f;
        }
    }
    for (int t = 0; t < int(p.count); ++t) {
        const size_t idx = size_t(t) * p.channels + c;
        if (p.trace_stride) trace[size_t(t + p.taps - 1) * p.trace_stride] = raw[idx];
        float out = raw[idx];
        for (unsigned lag = 0; lag < p.taps && lag <= position + t; ++lag) {
            const int src = t - int(lag);
            const float x = src >= 0 ? raw[size_t(src) * p.channels + c] : history[int(p.taps - 1) + src];
            const float tap = kernel[p.taps - 1 - lag] +
                (dynamic_stride ? dynamic_taps[size_t(t) * dynamic_stride + dynamic_channel + lag] : 0.0f);
            out = fmaf(x, tap, out);
        }
        output[idx] = out * p.scale + (p.residual ? residual[idx] : 0.0f);
    }

    // Shift towards the oldest entry. When the chunk is shorter than the
    // history, the retained entries have not yet been overwritten.
    for (unsigned i = 0; i + 1 < p.taps; ++i) {
        const int src = int(p.count + i) - int(p.taps - 1);
        if (src >= 0) history[i] = raw[size_t(src) * p.channels + c];
        else if (int(position) + src >= 0) history[i] = history[p.count + i];
        else history[i] = 0.0f;
    }
}

struct RestoreConv { unsigned channels, history_rows, offset, committed; };
extern "C" __global__ void inkling_restore_conv(float* history, const float* trace, RestoreConv p) {
    const unsigned c = blockIdx.x * 64 + threadIdx.x;
    if (c >= p.channels) return;
    trace += size_t(p.offset) + c;
    for (unsigned i = 0; i < p.history_rows; ++i)
        history[size_t(c) * p.history_rows + i] = trace[size_t(p.committed + i) * p.channels];
}

struct Clip { unsigned dim, heads, scale_floor; float limit, alpha; };
__device__ __forceinline__ void clip_projection(
        float* x, const float* weights, unsigned dim, float limit, float tau) {
    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += 64) {
        const float value = x[i] * weights[i];
        sum = fmaf(value, value, sum);
    }
    const float norm = sqrtf(zinc_block_reduce_sum_all(sum));
    const float scale = tau * limit / fmaxf(norm, limit);
    for (unsigned i = threadIdx.x; i < dim; i += 64) x[i] = x[i] * weights[i] * scale;
}

__device__ __forceinline__ float clip_temperature(unsigned position, unsigned floor, float alpha) {
    return alpha == 0 ? 1.0f : 1.0f + alpha * logf(fmaxf(1.0f, float(position + 1) / floor));
}

extern "C" __global__ void inkling_clip(float* x, const float* weights, const unsigned* step, Clip p) {
    x += (size_t(blockIdx.y) * p.heads + blockIdx.x) * p.dim;
    weights += blockIdx.x * p.dim;
    clip_projection(x, weights, p.dim, p.limit, clip_temperature(step[0] + blockIdx.y, p.scale_floor, p.alpha));
}

struct KvWrite {
    unsigned dim, heads, capacity;
    float clip, eps;
};
extern "C" __global__ void inkling_kv(
        const float* k, const float* v, __half* key_cache, __half* val_cache,
        const unsigned* step, KvWrite p) {
    const unsigned width = p.dim * p.heads;
    const size_t input = size_t(blockIdx.y) * width + blockIdx.x * p.dim;
    const size_t output = size_t((step[0] + blockIdx.y) % p.capacity) * width + blockIdx.x * p.dim;
    float sum_k = 0.0f, sum_v = 0.0f;
    for (unsigned i = threadIdx.x; i < p.dim; i += 64) {
        sum_k = fmaf(k[input + i], k[input + i], sum_k);
        sum_v = fmaf(v[input + i], v[input + i], sum_v);
    }
    const float k_scale = p.clip / fmaxf(p.clip, sqrtf(zinc_block_reduce_sum_all(sum_k)));
    const float v_scale = rsqrtf(zinc_block_reduce_sum_all(sum_v) / p.dim + p.eps);
    for (unsigned i = threadIdx.x; i < p.dim; i += 64) {
        key_cache[output + i] = __float2half_rn(k[input + i] * k_scale);
        val_cache[output + i] = __float2half_rn(v[input + i] * v_scale);
    }
}

__device__ __forceinline__ float convolve_one(
        float raw, const float* kernel, const float* dynamic, float* history,
        unsigned taps, unsigned position) {
    float out = fmaf(raw, kernel[taps - 1] + (dynamic ? dynamic[0] : 0.0f), raw);
    for (unsigned lag = 1; lag < taps && lag <= position; ++lag) {
        const float tap = kernel[taps - 1 - lag] + (dynamic ? dynamic[lag] : 0.0f);
        out = fmaf(history[taps - 1 - lag], tap, out);
    }
    for (unsigned i = 0; i + 2 < taps; ++i) history[i] = position + i + 2 >= taps ? history[i + 1] : 0.0f;
    history[taps - 2] = raw;
    return out;
}

struct DecodeKv {
    unsigned dim, heads, kv_heads, rel_dim, taps, scale_floor, capacity, dynamic;
    float q_clip, rel_clip, alpha, eps, conv_scale;
};

// Query clipping and key/value convolution are independent after projection.
// Publish both in one decode launch, normalizing K and V in the cache writer.
extern "C" __global__ void inkling_qkv_decode(
        float* q, float* relative, const float* q_weight, const float* rel_weight,
        const float* k, const float* v, const float* k_kernel, const float* v_kernel,
        const float* k_dynamic, const float* v_dynamic, float* history,
        __half* keys, __half* values, const unsigned* step, DecodeKv p) {
    const unsigned position = step[0];
    const unsigned block = blockIdx.x;
    if (block < p.heads * 2) {
        const float tau = clip_temperature(position, p.scale_floor, p.alpha);
        if (block < p.heads) {
            clip_projection(q + block * p.dim, q_weight + block * p.dim, p.dim, p.q_clip, tau);
        } else {
            const unsigned head = block - p.heads;
            clip_projection(relative + head * p.rel_dim, rel_weight + head * p.rel_dim, p.rel_dim, p.rel_clip, tau);
        }
        return;
    }

    const unsigned head = block - p.heads * 2;
    const unsigned input = head * p.dim;
    const unsigned width = p.kv_heads * p.dim;
    const size_t output = size_t(position % p.capacity) * width + input;
    float data_k[2] = {}, data_v[2] = {};
    float sum_k = 0.0f, sum_v = 0.0f;
    for (unsigned i = threadIdx.x; i < p.dim; i += 64) {
        const unsigned channel = input + i;
        const float* kd = p.dynamic ? k_dynamic + head * p.taps : nullptr;
        const float* vd = p.dynamic ? v_dynamic + head * p.taps : nullptr;
        const float cv_k = convolve_one(k[channel], k_kernel + channel * p.taps, kd,
            history + size_t(channel) * (p.taps - 1), p.taps, position) * p.conv_scale;
        const float cv_v = convolve_one(v[channel], v_kernel + channel * p.taps, vd,
            history + size_t(width + channel) * (p.taps - 1), p.taps, position) * p.conv_scale;
        data_k[i / 64] = cv_k;
        data_v[i / 64] = cv_v;
        sum_k = fmaf(cv_k, cv_k, sum_k);
        sum_v = fmaf(cv_v, cv_v, sum_v);
    }
    const float k_scale = p.q_clip / fmaxf(p.q_clip, sqrtf(zinc_block_reduce_sum_all(sum_k)));
    const float v_scale = rsqrtf(zinc_block_reduce_sum_all(sum_v) / p.dim + p.eps);
    for (unsigned i = threadIdx.x; i < p.dim; i += 64) {
        keys[output + i] = __float2half_rn(data_k[i / 64] * k_scale);
        values[output + i] = __float2half_rn(data_v[i / 64] * v_scale);
    }
}

constexpr unsigned attention_tile = 128;
struct Attention {
    unsigned dim, heads, kv_heads, capacity, window, rel_dim, rel_extent, tiles;
};

extern "C" __global__ void inkling_attention(
        const float* q, const float* rel, const float* bias, const __half* keys, const __half* values,
        float* partial, const unsigned* step, const float* sink, const float* gate, float* output, Attention p) {
    const unsigned head = blockIdx.x;
    const unsigned query_pos = step[0] + blockIdx.y;
    const unsigned history = p.window ? min(p.window, query_pos + 1) : query_pos + 1;
    const unsigned first = query_pos + 1 - history + blockIdx.z * attention_tile;
    const unsigned len = first <= query_pos ? min(attention_tile, query_pos + 1 - first) : 0;
    const unsigned kv_head = head / (p.heads / p.kv_heads);
    const unsigned kv_width = p.dim * p.kv_heads;
    const size_t query_idx = size_t(blockIdx.y) * p.heads + head;
    const unsigned lane = threadIdx.x & 15;
    q += query_idx * p.dim;
    rel += query_idx * p.rel_dim;
    partial += (query_idx * p.tiles + blockIdx.z) * (p.dim + 2);
    if (!len) {
        if (threadIdx.x == 0) { partial[p.dim] = negative_infinity; partial[p.dim + 1] = 0.0f; }
        return;
    }

    __shared__ float scores[attention_tile];
    const float scale = rsqrtf(float(p.dim));
    for (unsigned i = threadIdx.x / 16; i < len; i += 4) {
        const unsigned pos = first + i;
        const unsigned distance = query_pos - pos;
        const __half* key = keys + size_t(pos % p.capacity) * kv_width + kv_head * p.dim;
        float dot = 0.0f, offset = 0.0f;
        for (unsigned d = lane; d < p.dim; d += 16) dot = fmaf(q[d], __half2float(key[d]), dot);
        if (distance < p.rel_extent) {
            const float* tap = bias + size_t(distance) * p.rel_dim;
            for (unsigned d = lane; d < p.rel_dim; d += 16) offset = fmaf(rel[d], tap[d], offset);
        }
        float score = dot * scale + offset;
        for (unsigned step = 8; step; step >>= 1) score += __shfl_down(score, step, 16);
        if (lane == 0) scores[i] = score;
    }
    __syncthreads();
    float maximum = negative_infinity;
    for (unsigned i = threadIdx.x; i < len; i += 64) maximum = fmaxf(maximum, scores[i]);
    const float max0 = zinc_block_reduce_max(maximum);
    __shared__ float max_score;
    if (threadIdx.x == 0) max_score = max0;
    __syncthreads();
    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < len; i += 64) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }
    const float mass = zinc_block_reduce_sum(sum);
    __shared__ float tile_weight, inv_mass;
    if (p.tiles == 1 && threadIdx.x == 0) {
        const float sink_score = sink[query_idx];
        const float highest = fmaxf(max_score, sink_score);
        tile_weight = expf(max_score - highest);
        inv_mass = 1.0f / fmaf(tile_weight, mass, expf(sink_score - highest));
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < p.dim; d += 64) {
        float out = 0.0f;
        for (unsigned i = 0; i < len; ++i) {
            const size_t row = size_t((first + i) % p.capacity) * kv_width + kv_head * p.dim;
            out = fmaf(scores[i], __half2float(values[row + d]), out);
        }
        if (p.tiles == 1) {
            const size_t idx = query_idx * p.dim + d;
            output[idx] = out * tile_weight * inv_mass * sigmoid(gate[idx]);
        } else {
            partial[d] = out;
        }
    }
    if (p.tiles != 1 && threadIdx.x == 0) {
        partial[p.dim] = max_score;
        partial[p.dim + 1] = mass;
    }
}

extern "C" __global__ void inkling_attention_merge(
        const float* partial, const float* sink, const float* gate, float* out, Attention p) {
    const size_t query_idx = size_t(blockIdx.y) * p.heads + blockIdx.x;
    partial += query_idx * p.tiles * (p.dim + 2);
    unsigned stride = p.dim + 2;
    __shared__ float maximum, inv_mass;
    if (threadIdx.x == 0) {
        float highest = sink[query_idx];
        for (unsigned i = 0; i < p.tiles; ++i) highest = fmaxf(highest, partial[i * stride + p.dim]);
        maximum = highest;
        float denom = expf(sink[query_idx] - highest);
        for (unsigned i = 0; i < p.tiles; ++i) {
            denom = fmaf(expf(partial[i * stride + p.dim] - highest), partial[i * stride + p.dim + 1], denom);
        }
        inv_mass = 1.0f / denom;
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < p.dim; d += 64) {
        float weighted = 0.0f;
        for (unsigned i = 0; i < p.tiles; ++i) {
            const float* tile = partial + i * stride;
            if (tile[p.dim + 1] > 0) weighted = fmaf(expf(tile[p.dim] - maximum), tile[d], weighted);
        }
        const size_t idx = query_idx * p.dim + d;
        out[idx] = weighted * inv_mass * sigmoid(gate[idx]);
    }
}

struct Route { unsigned experts, active, biased; float scale; };
extern "C" __global__ void inkling_route(
        const float* logits, const float* bias, unsigned* ids, float* weights, Route p) {
    const unsigned lane = threadIdx.x;
    if (lane >= 32) return;
    logits += size_t(blockIdx.x) * p.experts;
    ids += size_t(blockIdx.x) * p.active;
    weights += size_t(blockIdx.x) * p.active;
    float probs[8], scores[8];
    const unsigned rounds = (p.experts + 31) / 32;
    for (unsigned j = 0; j < rounds; ++j) {
        const unsigned e = j * 32 + lane;
        probs[j] = e < p.experts ? sigmoid(logits[e]) : 0.0f;
        scores[j] = e < p.experts ? probs[j] + (p.biased ? bias[e] : 0.0f) : negative_infinity;
    }
    float sum = 1e-20f;
    for (unsigned k = 0; k < p.active; ++k) {
        unsigned id = lane;
        float best = scores[0];
        for (unsigned j = 1; j < rounds; ++j) {
            if (scores[j] > best) { best = scores[j]; id = j * 32 + lane; }
        }
        for (unsigned step = 16; step; step >>= 1) {
            const float other = __shfl_xor(best, step, 32);
            const unsigned other_id = __shfl_xor(id, step, 32);
            if (other > best || (other == best && other_id < id)) { best = other; id = other_id; }
        }
        const unsigned j = id / 32;
        const unsigned winner = id & 31;
        const float probability = __shfl(probs[j], winner, 32);
        if (lane == winner) scores[j] = negative_infinity;
        if (lane == 0) {
            ids[k] = id;
            weights[k] = probability;
            sum += probability;
        }
    }
    if (lane == 0) {
        const float scale = (p.scale == 0 ? 1.0f : p.scale) / sum;
        for (unsigned k = 0; k < p.active; ++k) weights[k] *= scale;
    }
}

struct Situ { unsigned width; float beta, linear_beta; };
extern "C" __global__ void inkling_situ(const float* gate_up, float* out, Situ p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i >= p.width) return;
    const size_t base = size_t(blockIdx.y) * p.width;
    const float gate = gate_up[2 * base + i];
    const float up = gate_up[2 * base + p.width + i];
    out[base + i] = (p.beta * tanhf(gate / p.beta) * sigmoid(gate)) *
                    (p.linear_beta * tanhf(up / p.linear_beta));
}

struct Combine { unsigned hidden, active; };
__device__ __forceinline__ float combine_experts(const float* values, const float* weights, unsigned i, Combine p) {
    float sum = 0.0f;
    for (unsigned k = 0; k < p.active; ++k) sum = fmaf(values[k * p.hidden + i], weights[k], sum);
    return sum;
}

extern "C" __global__ void inkling_combine(const float* values, const float* weights, float* out, Combine p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i >= p.hidden) return;
    const size_t route = size_t(blockIdx.y) * p.active;
    out[size_t(blockIdx.y) * p.hidden + i] = combine_experts(values + route * p.hidden, weights + route, i, p);
}

struct CombineConv { Combine combine; unsigned taps, offset; float scale; };
extern "C" __global__ void inkling_combine_conv(
        const float* values, const float* weights, const float* kernel, float* history,
        const float* residual, float* out, const unsigned* step, CombineConv p) {
    const unsigned i = blockIdx.x * 64 + threadIdx.x;
    if (i >= p.combine.hidden) return;
    const float raw = combine_experts(values, weights, i, p.combine);
    const float conv = convolve_one(raw, kernel + i * p.taps, nullptr,
        history + size_t(p.offset + i) * (p.taps - 1), p.taps, step[0]);
    out[i] = conv * p.scale + residual[i];
}

struct Argmax { unsigned vocab, partials; float cap; };
extern "C" __global__ void inkling_argmax_partial(const float* logits, uint2* partial, Argmax p) {
    logits += size_t(blockIdx.y) * p.vocab;
    partial += size_t(blockIdx.y) * p.partials;
    float best = negative_infinity;
    unsigned id = 0;
    for (unsigned i = blockIdx.x * 64 + threadIdx.x; i < p.vocab; i += p.partials * 64) {
        const float value = p.cap > 0 ? p.cap * tanhf(logits[i] / p.cap) : logits[i];
        if (value > best) { best = value; id = i; }
    }
    __shared__ float scores[64];
    __shared__ unsigned ids[64];
    scores[threadIdx.x] = best;
    ids[threadIdx.x] = id;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned i = 1; i < 64; ++i) {
            if (scores[i] > best || (scores[i] == best && ids[i] < id)) { best = scores[i]; id = ids[i]; }
        }
        partial[blockIdx.x] = make_uint2(__float_as_uint(best), id);
    }
}

extern "C" __global__ void inkling_argmax(const uint2* partial, unsigned* out, Argmax p) {
    if (threadIdx.x != 0) return;
    partial += size_t(blockIdx.y) * p.partials;
    out += blockIdx.y;
    float best = negative_infinity;
    unsigned id = 0;
    for (unsigned i = 0; i < p.partials; ++i) {
        const float value = __uint_as_float(partial[i].x);
        const unsigned token = partial[i].y;
        if (value > best || (value == best && token < id)) { best = value; id = token; }
    }
    *out = id;
}

} // namespace inkling

// Share the head's weight reads without changing the scalar decoder's
// accumulation order between Q6_K superblocks.
#define ZINC_INKLING_VERIFY_HEAD(n) \
    extern "C" __global__ void inkling_verify_q6_##n( \
            const unsigned char* weights, const float* x, float* y, DmmvPush p) { \
        dmmv_q6k_btok_impl<n, true>(weights, x, y, p); \
    }
ZINC_INKLING_VERIFY_HEAD(2)
ZINC_INKLING_VERIFY_HEAD(3)
ZINC_INKLING_VERIFY_HEAD(4)
ZINC_INKLING_VERIFY_HEAD(5)
ZINC_INKLING_VERIFY_HEAD(6)
ZINC_INKLING_VERIFY_HEAD(7)
ZINC_INKLING_VERIFY_HEAD(8)
#undef ZINC_INKLING_VERIFY_HEAD
