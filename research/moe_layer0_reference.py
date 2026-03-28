#!/usr/bin/env python3
"""
ZINC reference: CPU computation of full layer 0 MoE FFN pass.

Reads a Qwen3.5-35B-A3B GGUF file directly, dequantizes all needed tensors,
and computes the complete layer 0 MoE FFN pass for the BOS token (token ID 1).

Pipeline:
  1. BOS embedding (Q8_0 dequant of token_embd.weight row 1)
  2. attn_norm (RMS norm with blk.0.attn_norm.weight)
  3. Skip attention (pass hidden through)
  4. post_attention_norm (RMS norm with blk.0.post_attention_norm.weight)
  5. Router: ffn_gate_inp.weight (F32) x ffn_norm -> 256 logits
  6. Softmax + top-8 expert selection
  7. For each top-8 expert: gate/up (Q4_K) -> SwiGLU -> down (Q5_K) -> weighted accum
  8. Shared expert: gate/up/down (Q8_0) + sigmoid gate (F32) -> accum
  9. Residual add: hidden = BOS_embed + moe_output + shared_expert_output

No numpy -- uses only struct and math from stdlib.

Usage: python3 moe_layer0_reference.py [/path/to/model.gguf]
"""

import struct
import math
import sys
import time

# ============================================================================
# GGUF parser (no mmap -- we seek and read only what we need)
# ============================================================================

GGUF_MAGIC = 0x46554747

# Metadata value type tags
GGUF_TYPE_UINT8    = 0
GGUF_TYPE_INT8     = 1
GGUF_TYPE_UINT16   = 2
GGUF_TYPE_INT16    = 3
GGUF_TYPE_UINT32   = 4
GGUF_TYPE_INT32    = 5
GGUF_TYPE_FLOAT32  = 6
GGUF_TYPE_BOOL     = 7
GGUF_TYPE_STRING   = 8
GGUF_TYPE_ARRAY    = 9
GGUF_TYPE_UINT64   = 10
GGUF_TYPE_INT64    = 11
GGUF_TYPE_FLOAT64  = 12

# GGML tensor types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q8_0 = 8

GGML_BLOCK_SIZE = {
    GGML_TYPE_F32: 1, GGML_TYPE_F16: 1,
    GGML_TYPE_Q4_K: 256, GGML_TYPE_Q5_K: 256, GGML_TYPE_Q8_0: 32,
}
GGML_BYTES_PER_BLOCK = {
    GGML_TYPE_F32: 4, GGML_TYPE_F16: 2,
    GGML_TYPE_Q4_K: 144, GGML_TYPE_Q5_K: 176, GGML_TYPE_Q8_0: 34,
}
GGML_TYPE_NAME = {
    GGML_TYPE_F32: "F32", GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_K: "Q4_K", GGML_TYPE_Q5_K: "Q5_K", GGML_TYPE_Q8_0: "Q8_0",
}


def f16_to_f32(bits):
    """Convert IEEE 754 half-precision bits (u16) to Python float."""
    sign = (bits >> 15) & 1
    exp  = (bits >> 10) & 0x1F
    frac = bits & 0x3FF
    if exp == 0:
        # Subnormal or zero
        val = (frac / 1024.0) * (2.0 ** -14)
    elif exp == 31:
        # Inf or NaN
        val = float('inf') if frac == 0 else float('nan')
    else:
        val = (1.0 + frac / 1024.0) * (2.0 ** (exp - 15))
    return -val if sign else val


class GGUFReader:
    """Sequential binary reader for GGUF parsing."""

    def __init__(self, f):
        self.f = f

    def read_u8(self):
        return struct.unpack('<B', self.f.read(1))[0]

    def read_i8(self):
        return struct.unpack('<b', self.f.read(1))[0]

    def read_u16(self):
        return struct.unpack('<H', self.f.read(2))[0]

    def read_i16(self):
        return struct.unpack('<h', self.f.read(2))[0]

    def read_u32(self):
        return struct.unpack('<I', self.f.read(4))[0]

    def read_i32(self):
        return struct.unpack('<i', self.f.read(4))[0]

    def read_u64(self):
        return struct.unpack('<Q', self.f.read(8))[0]

    def read_i64(self):
        return struct.unpack('<q', self.f.read(8))[0]

    def read_f32(self):
        return struct.unpack('<f', self.f.read(4))[0]

    def read_f64(self):
        return struct.unpack('<d', self.f.read(8))[0]

    def read_string(self):
        length = self.read_u64()
        return self.f.read(length).decode('utf-8', errors='replace')

    def read_typed_value(self, vtype):
        if vtype == GGUF_TYPE_UINT8:    return self.read_u8()
        if vtype == GGUF_TYPE_INT8:     return self.read_i8()
        if vtype == GGUF_TYPE_UINT16:   return self.read_u16()
        if vtype == GGUF_TYPE_INT16:    return self.read_i16()
        if vtype == GGUF_TYPE_UINT32:   return self.read_u32()
        if vtype == GGUF_TYPE_INT32:    return self.read_i32()
        if vtype == GGUF_TYPE_FLOAT32:  return self.read_f32()
        if vtype == GGUF_TYPE_BOOL:     return self.read_u8() != 0
        if vtype == GGUF_TYPE_STRING:   return self.read_string()
        if vtype == GGUF_TYPE_UINT64:   return self.read_u64()
        if vtype == GGUF_TYPE_INT64:    return self.read_i64()
        if vtype == GGUF_TYPE_FLOAT64:  return self.read_f64()
        raise ValueError(f"Unknown GGUF type {vtype}")

    def read_metadata_value(self):
        vtype = self.read_u32()
        if vtype == GGUF_TYPE_ARRAY:
            elem_type = self.read_u32()
            count = self.read_u64()
            return [self.read_typed_value(elem_type) for _ in range(count)]
        return self.read_typed_value(vtype)

    def tell(self):
        return self.f.tell()


class TensorInfo:
    """Tensor descriptor from GGUF header."""
    __slots__ = ('name', 'n_dims', 'dims', 'type_id', 'offset')

    def __init__(self, name, n_dims, dims, type_id, offset):
        self.name = name
        self.n_dims = n_dims
        self.dims = dims          # list of dims, dims[0] = innermost
        self.type_id = type_id
        self.offset = offset      # offset from start of tensor data section

    def num_elements(self):
        n = 1
        for d in self.dims[:self.n_dims]:
            n *= d
        return n

    def size_bytes(self):
        n = self.num_elements()
        bs = GGML_BLOCK_SIZE.get(self.type_id, 1)
        bpb = GGML_BYTES_PER_BLOCK.get(self.type_id, 4)
        blocks = (n + bs - 1) // bs
        return blocks * bpb


def parse_gguf_header(filepath):
    """Parse GGUF header, return (metadata_dict, tensor_list, tensor_data_offset, file_handle)."""
    f = open(filepath, 'rb')
    r = GGUFReader(f)

    magic = r.read_u32()
    assert magic == GGUF_MAGIC, f"Bad GGUF magic: 0x{magic:08x}"

    version = r.read_u32()
    tensor_count = r.read_u64()
    metadata_count = r.read_u64()
    print(f"GGUF v{version}: {tensor_count} tensors, {metadata_count} metadata entries")

    # Parse metadata
    metadata = {}
    for _ in range(metadata_count):
        key = r.read_string()
        val = r.read_metadata_value()
        metadata[key] = val

    # Parse tensor descriptors
    tensors = {}
    for _ in range(tensor_count):
        name = r.read_string()
        n_dims = r.read_u32()
        dims = []
        for _ in range(n_dims):
            dims.append(r.read_u64())
        # Pad to 4
        while len(dims) < 4:
            dims.append(1)
        type_id = r.read_u32()
        offset = r.read_u64()
        tensors[name] = TensorInfo(name, n_dims, dims, type_id, offset)

    # Alignment
    alignment = metadata.get('general.alignment', 32)
    if isinstance(alignment, list):
        alignment = 32
    pos = r.tell()
    tensor_data_offset = (pos + alignment - 1) & ~(alignment - 1)

    return metadata, tensors, tensor_data_offset, f


# ============================================================================
# Dequantization routines (no numpy)
# ============================================================================

def read_tensor_raw(f, tensor_data_offset, tinfo, byte_offset=0, byte_length=None):
    """Read raw bytes for a tensor (or a slice of it) from the GGUF file."""
    abs_offset = tensor_data_offset + tinfo.offset + byte_offset
    if byte_length is None:
        byte_length = tinfo.size_bytes() - byte_offset
    f.seek(abs_offset)
    return f.read(byte_length)


def dequant_q8_0_row(raw, row, cols):
    """Dequantize one row of Q8_0 data -> list of floats."""
    block_size = 32
    bpb = 34
    bpr = cols // block_size
    row_off = row * bpr * bpb
    out = []
    for b in range(bpr):
        bo = row_off + b * bpb
        d_bits = raw[bo] | (raw[bo + 1] << 8)
        d = f16_to_f32(d_bits)
        for j in range(block_size):
            q = raw[bo + 2 + j]
            if q >= 128:
                q -= 256  # signed int8
            out.append(q * d)
    return out


def get_scale_min_k4(j, scales):
    """Extract 6-bit scale and min from Q4_K/Q5_K packed scale array."""
    if j < 4:
        return (scales[j] & 63, scales[j + 4] & 63)
    else:
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m  = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4)
        return (sc, m)


def dequant_q4k_row(raw, row, cols):
    """Dequantize one row of Q4_K data -> list of floats.

    Q4_K block: d(f16,2) dmin(f16,2) scales(12) qs(128) = 144 bytes / 256 elems
    Element ordering: 4 groups of 64. Each group: 32 from low nibble (sub-block 2g),
    then 32 from high nibble (sub-block 2g+1). This is CONTIGUOUS, NOT interleaved.
    """
    bpb = 144
    bpr = cols // 256
    row_off = row * bpr * bpb
    out = []
    for bi in range(bpr):
        bb = row_off + bi * bpb
        d_bits = raw[bb] | (raw[bb + 1] << 8)
        d = f16_to_f32(d_bits)
        dm_bits = raw[bb + 2] | (raw[bb + 3] << 8)
        dmin = f16_to_f32(dm_bits)

        scales = raw[bb + 4 : bb + 16]
        qs = raw[bb + 16 : bb + 144]

        is_idx = 0
        qo = 0
        for _ in range(4):
            sc0, m0 = get_scale_min_k4(is_idx, scales)
            d1 = d * sc0
            m1 = dmin * m0
            sc1, m1_ = get_scale_min_k4(is_idx + 1, scales)
            d2 = d * sc1
            m2 = dmin * m1_

            # 32 elements from low nibble (sub-block 2g)
            for l in range(32):
                out.append(d1 * (qs[qo + l] & 0xF) - m1)
            # 32 elements from high nibble (sub-block 2g+1)
            for l in range(32):
                out.append(d2 * (qs[qo + l] >> 4) - m2)

            qo += 32
            is_idx += 2

    return out


def dequant_q5k_row(raw, row, cols):
    """Dequantize one row of Q5_K data -> list of floats.

    Q5_K block: d(f16,2) dmin(f16,2) scales(12) qh(32) qs(128) = 176 bytes / 256 elems
    Element ordering is INTERLEAVED: y[2l] from low nibble, y[2l+1] from high nibble.
    """
    bpb = 176
    bpr = cols // 256
    row_off = row * bpr * bpb
    out = []
    for bi in range(bpr):
        bb = row_off + bi * bpb
        d_bits = raw[bb] | (raw[bb + 1] << 8)
        d = f16_to_f32(d_bits)
        dm_bits = raw[bb + 2] | (raw[bb + 3] << 8)
        dmin = f16_to_f32(dm_bits)

        scales = raw[bb + 4 : bb + 16]
        qh = raw[bb + 16 : bb + 48]
        qs = raw[bb + 48 : bb + 176]

        is_idx = 0
        for j in range(4):
            sc0, m0 = get_scale_min_k4(is_idx, scales)
            d1 = d * sc0
            m1 = dmin * m0
            sc1, m1_ = get_scale_min_k4(is_idx + 1, scales)
            d2 = d * sc1
            m2 = dmin * m1_

            for l in range(32):
                ql_lo = qs[j * 32 + l] & 0xF
                ql_hi = qs[j * 32 + l] >> 4
                hb_lo = (qh[l] >> (j * 2)) & 1
                hb_hi = (qh[l] >> (j * 2 + 1)) & 1
                # Interleaved: element 2l from low nibble, element 2l+1 from high nibble
                out.append(d1 * (ql_lo | (hb_lo << 4)) - m1)
                out.append(d2 * (ql_hi | (hb_hi << 4)) - m2)

            is_idx += 2

    return out


def dequant_f32_row(raw, row, cols):
    """Read one row of F32 data -> list of floats."""
    row_bytes = cols * 4
    offset = row * row_bytes
    return list(struct.unpack(f'<{cols}f', raw[offset:offset + row_bytes]))


def dequant_f16_row(raw, row, cols):
    """Read one row of F16 data -> list of floats."""
    row_bytes = cols * 2
    offset = row * row_bytes
    out = []
    for i in range(cols):
        bits = raw[offset + i*2] | (raw[offset + i*2 + 1] << 8)
        out.append(f16_to_f32(bits))
    return out


# ============================================================================
# Math helpers
# ============================================================================

def dot(a, b):
    """Dot product of two lists."""
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def rms_norm(x, weight, eps=1e-6):
    """RMS normalization: out[i] = weight[i] * x[i] / sqrt(mean(x^2) + eps)."""
    n = len(x)
    sum_sq = 0.0
    for v in x:
        sum_sq += v * v
    rms_inv = 1.0 / math.sqrt(sum_sq / n + eps)
    return [weight[i] * x[i] * rms_inv for i in range(n)]


def silu(x):
    """SiLU activation: x * sigmoid(x)."""
    # Clamp to avoid overflow in exp
    if x > 80.0:
        return x
    if x < -80.0:
        return 0.0
    return x / (1.0 + math.exp(-x))


def sigmoid(x):
    """Sigmoid function."""
    if x > 80.0:
        return 1.0
    if x < -80.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def softmax(logits):
    """Softmax over a list of floats."""
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def top_k_softmax(logits, k):
    """Softmax over ALL logits, then pick top-k. Return (ids, weights) with renormalized weights."""
    probs = softmax(logits)
    # Top-k selection
    used = set()
    ids = []
    weights = []
    for _ in range(k):
        best_idx = -1
        best_val = -1.0
        for i in range(len(probs)):
            if i not in used and probs[i] > best_val:
                best_val = probs[i]
                best_idx = i
        ids.append(best_idx)
        weights.append(best_val)
        used.add(best_idx)
    # Renormalize
    wsum = sum(weights)
    if wsum > 0:
        weights = [w / wsum for w in weights]
    return ids, weights


# ============================================================================
# DMMV: dequantized matrix-vector multiply
# ============================================================================

def dmmv_q4k(raw_data, x_vec, M, K, a_byte_offset=0):
    """Q4_K DMMV: raw_data[a_byte_offset:] contains M rows of K elements in Q4_K format.
    Returns list of M output values.
    """
    bpb = 144
    bpr = K // 256
    out = [0.0] * M
    for row in range(M):
        row_off = a_byte_offset + row * bpr * bpb
        acc = 0.0
        xi = 0  # index into x_vec
        for blk in range(bpr):
            bb = row_off + blk * bpb
            d_bits = raw_data[bb] | (raw_data[bb + 1] << 8)
            d = f16_to_f32(d_bits)
            dm_bits = raw_data[bb + 2] | (raw_data[bb + 3] << 8)
            dmin = f16_to_f32(dm_bits)
            scales = raw_data[bb + 4 : bb + 16]
            qs = raw_data[bb + 16 : bb + 144]

            is_idx = 0
            qo = 0
            for _ in range(4):
                sc0, m0 = get_scale_min_k4(is_idx, scales)
                factor_lo = d * sc0
                bias_lo = dmin * m0
                sc1, m1 = get_scale_min_k4(is_idx + 1, scales)
                factor_hi = d * sc1
                bias_hi = dmin * m1

                # Low nibble: 32 elements for sub-block 2g
                for e in range(32):
                    q_lo = qs[qo + e] & 0xF
                    acc += (factor_lo * q_lo - bias_lo) * x_vec[xi + is_idx * 32 + e]
                # High nibble: 32 elements for sub-block 2g+1
                for e in range(32):
                    q_hi = qs[qo + e] >> 4
                    acc += (factor_hi * q_hi - bias_hi) * x_vec[xi + (is_idx + 1) * 32 + e]

                qo += 32
                is_idx += 2

            xi += 256
        out[row] = acc
    return out


def dmmv_q5k(raw_data, x_vec, M, K, a_byte_offset=0):
    """Q5_K DMMV: raw_data[a_byte_offset:] contains M rows of K elements in Q5_K format.
    Q5_K is INTERLEAVED: x[2l] from low nibble, x[2l+1] from high nibble.
    Returns list of M output values.
    """
    bpb = 176
    bpr = K // 256
    out = [0.0] * M
    for row in range(M):
        row_off = a_byte_offset + row * bpr * bpb
        acc = 0.0
        xi = 0  # element offset into x_vec
        for blk in range(bpr):
            bb = row_off + blk * bpb
            d_bits = raw_data[bb] | (raw_data[bb + 1] << 8)
            d = f16_to_f32(d_bits)
            dm_bits = raw_data[bb + 2] | (raw_data[bb + 3] << 8)
            dmin = f16_to_f32(dm_bits)
            scales = raw_data[bb + 4 : bb + 16]
            qh = raw_data[bb + 16 : bb + 48]
            qs = raw_data[bb + 48 : bb + 176]

            is_idx = 0
            for g in range(4):
                sb_lo = g * 2
                sb_hi = g * 2 + 1
                sc_lo, m_lo = get_scale_min_k4(sb_lo, scales)
                sc_hi, m_hi = get_scale_min_k4(sb_hi, scales)
                factor_lo = d * sc_lo
                bias_lo = dmin * m_lo
                factor_hi = d * sc_hi
                bias_hi = dmin * m_hi

                qs_base = g * 32
                x_grp = xi
                for e in range(32):
                    byte_val = qs[qs_base + e]
                    q_lo = byte_val & 0xF
                    q_hi = byte_val >> 4
                    qh_val = qh[e]
                    hb_lo = (qh_val >> (g * 2)) & 1
                    hb_hi = (qh_val >> (g * 2 + 1)) & 1
                    v_lo = q_lo | (hb_lo << 4)
                    v_hi = q_hi | (hb_hi << 4)
                    # Interleaved: element 2e from low nibble, element 2e+1 from high nibble
                    acc += (factor_lo * v_lo - bias_lo) * x_vec[x_grp + 2 * e]
                    acc += (factor_hi * v_hi - bias_hi) * x_vec[x_grp + 2 * e + 1]

                xi += 64

            # xi has been advanced by 256 (4 groups * 64 each)
        out[row] = acc
    return out


def dmmv_q8_0(raw_data, x_vec, M, K, a_byte_offset=0):
    """Q8_0 DMMV: standard layout. Returns list of M output values."""
    bpb = 34
    bpr = K // 32
    out = [0.0] * M
    for row in range(M):
        row_off = a_byte_offset + row * bpr * bpb
        acc = 0.0
        for blk in range(bpr):
            bo = row_off + blk * bpb
            d_bits = raw_data[bo] | (raw_data[bo + 1] << 8)
            d = f16_to_f32(d_bits)
            x_base = blk * 32
            block_sum = 0.0
            for e in range(32):
                q = raw_data[bo + 2 + e]
                if q >= 128:
                    q -= 256
                block_sum += q * x_vec[x_base + e]
            acc += d * block_sum
        out[row] = acc
    return out


def dmmv_f32(raw_data, x_vec, M, K, a_byte_offset=0):
    """F32 DMMV: standard row-major layout. Returns list of M output values."""
    out = [0.0] * M
    for row in range(M):
        row_bytes = a_byte_offset + row * K * 4
        acc = 0.0
        for j in range(K):
            w = struct.unpack_from('<f', raw_data, row_bytes + j * 4)[0]
            acc += w * x_vec[j]
        out[row] = acc
    return out


# ============================================================================
# Expert slice byte computation (matching ZINC's expertSliceBytes)
# ============================================================================

def expert_slice_bytes(type_id, rows, cols):
    """Compute byte size of one expert slice in a stacked weight tensor."""
    bs = GGML_BLOCK_SIZE.get(type_id, 1)
    bpb = GGML_BYTES_PER_BLOCK.get(type_id, 4)
    if bs == 0 or bpb == 0:
        return rows * cols * 4
    blocks_per_row = cols // bs
    return rows * blocks_per_row * bpb


# ============================================================================
# Main computation
# ============================================================================

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    print(f"Loading GGUF: {model_path}")

    t0 = time.time()
    metadata, tensors, tensor_data_offset, f = parse_gguf_header(model_path)
    t1 = time.time()
    print(f"Header parsed in {t1 - t0:.2f}s, tensor_data_offset = {tensor_data_offset}")

    # Print key model metadata
    arch_prefix = metadata.get('general.architecture', 'llama')
    hidden_dim = metadata.get(f'{arch_prefix}.embedding_length', 0)
    expert_inter_dim = metadata.get(f'{arch_prefix}.expert_feed_forward_length', 0)
    shared_inter_dim = metadata.get(f'{arch_prefix}.feed_forward_length', 0)
    n_experts = metadata.get(f'{arch_prefix}.expert_count', 0)
    n_experts_used = metadata.get(f'{arch_prefix}.expert_used_count', 0)
    vocab_size = metadata.get(f'{arch_prefix}.vocab_size', 0)
    rms_eps = 1e-6  # ZINC uses 1e-6

    print(f"Architecture: {arch_prefix}")
    print(f"hidden_dim={hidden_dim}, expert_inter_dim={expert_inter_dim}, "
          f"shared_inter_dim={shared_inter_dim}")
    print(f"n_experts={n_experts}, n_experts_used={n_experts_used}, vocab_size={vocab_size}")

    # Validate expected dimensions
    inter_dim = expert_inter_dim if expert_inter_dim > 0 else (shared_inter_dim if shared_inter_dim > 0 else hidden_dim * 4)
    shexp_inter_dim = shared_inter_dim if shared_inter_dim > 0 else inter_dim

    # List the tensors we need
    needed_tensors = [
        'token_embd.weight',
        'blk.0.attn_norm.weight',
        'blk.0.post_attention_norm.weight',
        'blk.0.ffn_gate_inp.weight',
        'blk.0.ffn_gate_exps.weight',
        'blk.0.ffn_up_exps.weight',
        'blk.0.ffn_down_exps.weight',
        'blk.0.ffn_gate_shexp.weight',
        'blk.0.ffn_up_shexp.weight',
        'blk.0.ffn_down_shexp.weight',
        'blk.0.ffn_gate_inp_shexp.weight',
    ]
    for tn in needed_tensors:
        if tn in tensors:
            ti = tensors[tn]
            tname = GGML_TYPE_NAME.get(ti.type_id, f"type_{ti.type_id}")
            print(f"  {tn}: dims={ti.dims[:ti.n_dims]} type={tname} "
                  f"offset={ti.offset} size={ti.size_bytes()}")
        else:
            print(f"  {tn}: NOT FOUND")

    # -----------------------------------------------------------------------
    # Step 1: BOS embedding (token ID 1, Q8_0 dequant)
    # -----------------------------------------------------------------------
    print("\n=== Step 1: BOS embedding (token 1) ===")
    t_start = time.time()

    embd_info = tensors['token_embd.weight']
    embd_K = embd_info.dims[0]  # hidden_dim (innermost)
    assert embd_K == hidden_dim, f"Embedding dim mismatch: {embd_K} vs {hidden_dim}"

    # Read only the rows we need (token 1 = BOS)
    token_id = 1
    if embd_info.type_id == GGML_TYPE_Q8_0:
        bpr = embd_K // 32
        row_bytes = bpr * 34
        raw = read_tensor_raw(f, tensor_data_offset, embd_info,
                              byte_offset=token_id * row_bytes,
                              byte_length=row_bytes)
        bos_embed = dequant_q8_0_row(raw, 0, embd_K)
    elif embd_info.type_id == GGML_TYPE_F32:
        row_bytes = embd_K * 4
        raw = read_tensor_raw(f, tensor_data_offset, embd_info,
                              byte_offset=token_id * row_bytes,
                              byte_length=row_bytes)
        bos_embed = dequant_f32_row(raw, 0, embd_K)
    elif embd_info.type_id == GGML_TYPE_F16:
        row_bytes = embd_K * 2
        raw = read_tensor_raw(f, tensor_data_offset, embd_info,
                              byte_offset=token_id * row_bytes,
                              byte_length=row_bytes)
        bos_embed = dequant_f16_row(raw, 0, embd_K)
    else:
        raise ValueError(f"Unsupported embedding type: {embd_info.type_id}")

    print(f"  BOS embed[0..8]: {bos_embed[:8]}")
    print(f"  BOS embed norm: {math.sqrt(sum(v*v for v in bos_embed)):.6f}")
    print(f"  Time: {time.time() - t_start:.3f}s")

    # -----------------------------------------------------------------------
    # Step 2: attn_norm (RMS norm with blk.0.attn_norm.weight)
    # -----------------------------------------------------------------------
    print("\n=== Step 2: attn_norm ===")
    t_start = time.time()

    attn_norm_info = tensors['blk.0.attn_norm.weight']
    raw = read_tensor_raw(f, tensor_data_offset, attn_norm_info)
    if attn_norm_info.type_id == GGML_TYPE_F32:
        attn_norm_w = dequant_f32_row(raw, 0, hidden_dim)
    elif attn_norm_info.type_id == GGML_TYPE_F16:
        attn_norm_w = dequant_f16_row(raw, 0, hidden_dim)
    else:
        raise ValueError(f"Unsupported attn_norm type: {attn_norm_info.type_id}")

    # hidden state after attn_norm
    hidden_normed_attn = rms_norm(bos_embed, attn_norm_w, rms_eps)
    print(f"  attn_normed[0..8]: {hidden_normed_attn[:8]}")
    print(f"  Time: {time.time() - t_start:.3f}s")

    # -----------------------------------------------------------------------
    # Step 3: Skip attention (hidden state = BOS embedding, unchanged)
    # -----------------------------------------------------------------------
    print("\n=== Step 3: Skip attention (pass-through) ===")
    hidden = list(bos_embed)  # copy, hidden = BOS embedding
    print(f"  hidden[0..8] (= BOS embed): {hidden[:8]}")

    # -----------------------------------------------------------------------
    # Step 4: post_attention_norm (RMS norm with blk.0.post_attention_norm.weight)
    # -----------------------------------------------------------------------
    print("\n=== Step 4: post_attention_norm ===")
    t_start = time.time()

    pan_info = tensors['blk.0.post_attention_norm.weight']
    raw = read_tensor_raw(f, tensor_data_offset, pan_info)
    if pan_info.type_id == GGML_TYPE_F32:
        pan_w = dequant_f32_row(raw, 0, hidden_dim)
    elif pan_info.type_id == GGML_TYPE_F16:
        pan_w = dequant_f16_row(raw, 0, hidden_dim)
    else:
        raise ValueError(f"Unsupported post_attention_norm type: {pan_info.type_id}")

    ffn_norm = rms_norm(hidden, pan_w, rms_eps)
    print(f"  ffn_norm[0..8]: {ffn_norm[:8]}")
    print(f"  ffn_norm norm: {math.sqrt(sum(v*v for v in ffn_norm)):.6f}")
    print(f"  Time: {time.time() - t_start:.3f}s")

    # -----------------------------------------------------------------------
    # Step 5: Router DMMV (F32)
    # -----------------------------------------------------------------------
    print("\n=== Step 5: Router (ffn_gate_inp.weight, F32) ===")
    t_start = time.time()

    router_info = tensors['blk.0.ffn_gate_inp.weight']
    # Router dims: [hidden_dim, n_experts] in GGUF = K=hidden_dim, M=n_experts
    router_K = router_info.dims[0]
    router_M = router_info.dims[1]
    print(f"  Router shape: M={router_M} x K={router_K} (type={GGML_TYPE_NAME.get(router_info.type_id, '?')})")
    assert router_K == hidden_dim
    assert router_M == n_experts

    raw = read_tensor_raw(f, tensor_data_offset, router_info)
    router_logits = dmmv_f32(raw, ffn_norm, router_M, router_K)

    print(f"  Router logits[0..8]: {router_logits[:8]}")
    print(f"  Router logits min={min(router_logits):.6f} max={max(router_logits):.6f}")
    print(f"  Time: {time.time() - t_start:.3f}s")

    # -----------------------------------------------------------------------
    # Step 6: Softmax + top-k expert selection
    # -----------------------------------------------------------------------
    print("\n=== Step 6: Top-{} expert selection ===".format(n_experts_used))
    t_start = time.time()

    expert_ids, expert_weights = top_k_softmax(router_logits, n_experts_used)
    print(f"  Expert IDs: {expert_ids}")
    print(f"  Expert weights: {expert_weights}")
    print(f"  Weight sum: {sum(expert_weights):.6f}")
    print(f"  Time: {time.time() - t_start:.3f}s")

    # -----------------------------------------------------------------------
    # Step 7: Expert FFN (gate/up Q4_K, down Q5_K)
    # -----------------------------------------------------------------------
    print("\n=== Step 7: Expert FFN pass ===")
    t_start = time.time()

    gate_exps_info = tensors['blk.0.ffn_gate_exps.weight']
    up_exps_info = tensors['blk.0.ffn_up_exps.weight']
    down_exps_info = tensors['blk.0.ffn_down_exps.weight']

    # Expert dims: [K, inter_dim, n_experts]
    expert_K = gate_exps_info.dims[0]   # hidden_dim
    expert_M = gate_exps_info.dims[1]   # inter_dim (per expert output rows)
    expert_N = gate_exps_info.dims[2]   # n_experts
    print(f"  Gate/Up exps shape per expert: M={expert_M} x K={expert_K} "
          f"(type={GGML_TYPE_NAME.get(gate_exps_info.type_id, '?')})")
    print(f"  Down exps shape per expert: M={down_exps_info.dims[1]} x K={down_exps_info.dims[0]} "
          f"(type={GGML_TYPE_NAME.get(down_exps_info.type_id, '?')})")

    assert expert_K == hidden_dim
    assert expert_N == n_experts

    gate_quant = gate_exps_info.type_id
    down_quant = down_exps_info.type_id
    down_K = down_exps_info.dims[0]   # inter_dim
    down_M = down_exps_info.dims[1]   # hidden_dim

    # Expert slice sizes (byte offsets per expert)
    gate_expert_bytes = expert_slice_bytes(gate_quant, expert_M, expert_K)
    down_expert_bytes = expert_slice_bytes(down_quant, down_M, down_K)
    # up has same shape as gate
    up_expert_bytes = gate_expert_bytes

    print(f"  gate_expert_bytes={gate_expert_bytes}, down_expert_bytes={down_expert_bytes}")

    # Read entire expert tensors (they're stacked)
    gate_raw = read_tensor_raw(f, tensor_data_offset, gate_exps_info)
    up_raw = read_tensor_raw(f, tensor_data_offset, up_exps_info)
    down_raw = read_tensor_raw(f, tensor_data_offset, down_exps_info)

    # MoE output accumulator
    moe_out = [0.0] * hidden_dim

    # Select DMMV functions based on quant type
    if gate_quant == GGML_TYPE_Q4_K:
        gate_dmmv = dmmv_q4k
        up_dmmv = dmmv_q4k
    elif gate_quant == GGML_TYPE_Q8_0:
        gate_dmmv = dmmv_q8_0
        up_dmmv = dmmv_q8_0
    else:
        raise ValueError(f"Unsupported gate/up quant: {gate_quant}")

    if down_quant == GGML_TYPE_Q5_K:
        down_dmmv = dmmv_q5k
    elif down_quant == GGML_TYPE_Q4_K:
        down_dmmv = dmmv_q4k
    elif down_quant == GGML_TYPE_Q8_0:
        down_dmmv = dmmv_q8_0
    else:
        raise ValueError(f"Unsupported down quant: {down_quant}")

    for ei in range(n_experts_used):
        eid = expert_ids[ei]
        weight = expert_weights[ei]
        gate_offset = eid * gate_expert_bytes
        up_offset = eid * up_expert_bytes
        down_offset = eid * down_expert_bytes

        t_exp = time.time()

        # Gate DMMV
        gate_out = gate_dmmv(gate_raw, ffn_norm, expert_M, expert_K, gate_offset)
        # Up DMMV
        up_out = up_dmmv(up_raw, ffn_norm, expert_M, expert_K, up_offset)

        # SwiGLU: silu(gate) * up
        swiglu_out = [silu(gate_out[i]) * up_out[i] for i in range(expert_M)]

        # Down DMMV
        down_out = down_dmmv(down_raw, swiglu_out, down_M, down_K, down_offset)

        # Weighted accumulate
        for i in range(hidden_dim):
            moe_out[i] += weight * down_out[i]

        print(f"  Expert {eid} (weight={weight:.6f}): "
              f"gate[0..4]={gate_out[:4]}, up[0..4]={up_out[:4]}, "
              f"down[0..4]={down_out[:4]} ({time.time() - t_exp:.2f}s)")

    print(f"  MoE out[0..8]: {moe_out[:8]}")
    print(f"  Expert FFN time: {time.time() - t_start:.2f}s")

    # -----------------------------------------------------------------------
    # Step 8: Shared expert FFN
    # -----------------------------------------------------------------------
    print("\n=== Step 8: Shared expert FFN ===")
    t_start = time.time()

    shexp_out = [0.0] * hidden_dim

    gate_shexp_name = 'blk.0.ffn_gate_shexp.weight'
    up_shexp_name = 'blk.0.ffn_up_shexp.weight'
    down_shexp_name = 'blk.0.ffn_down_shexp.weight'
    shexp_gate_name = 'blk.0.ffn_gate_inp_shexp.weight'

    has_shared = (gate_shexp_name in tensors and up_shexp_name in tensors
                  and down_shexp_name in tensors)

    if has_shared:
        gate_sh_info = tensors[gate_shexp_name]
        up_sh_info = tensors[up_shexp_name]
        down_sh_info = tensors[down_shexp_name]

        sh_K = gate_sh_info.dims[0]   # hidden_dim
        sh_M = gate_sh_info.dims[1]   # shexp_inter_dim
        print(f"  Shared expert shape: M={sh_M} x K={sh_K}")
        print(f"  Gate type={GGML_TYPE_NAME.get(gate_sh_info.type_id, '?')}, "
              f"Down type={GGML_TYPE_NAME.get(down_sh_info.type_id, '?')}")

        # Read shared expert tensors
        gate_sh_raw = read_tensor_raw(f, tensor_data_offset, gate_sh_info)
        up_sh_raw = read_tensor_raw(f, tensor_data_offset, up_sh_info)
        down_sh_raw = read_tensor_raw(f, tensor_data_offset, down_sh_info)

        # Select DMMV based on quant type
        def select_dmmv(type_id):
            if type_id == GGML_TYPE_Q8_0: return dmmv_q8_0
            if type_id == GGML_TYPE_Q4_K: return dmmv_q4k
            if type_id == GGML_TYPE_Q5_K: return dmmv_q5k
            if type_id == GGML_TYPE_F32:  return dmmv_f32
            raise ValueError(f"Unsupported shared expert quant: {type_id}")

        # Gate DMMV
        gate_sh_out = select_dmmv(gate_sh_info.type_id)(gate_sh_raw, ffn_norm, sh_M, sh_K)
        # Up DMMV
        up_sh_out = select_dmmv(up_sh_info.type_id)(up_sh_raw, ffn_norm, sh_M, sh_K)

        # SwiGLU
        swiglu_sh = [silu(gate_sh_out[i]) * up_sh_out[i] for i in range(sh_M)]

        # Down DMMV
        down_sh_K = down_sh_info.dims[0]   # shexp_inter_dim
        down_sh_M = down_sh_info.dims[1]   # hidden_dim
        shexp_out = select_dmmv(down_sh_info.type_id)(down_sh_raw, swiglu_sh, down_sh_M, down_sh_K)

        print(f"  Shared gate[0..4]: {gate_sh_out[:4]}")
        print(f"  Shared up[0..4]: {up_sh_out[:4]}")
        print(f"  Shared down[0..4]: {shexp_out[:4]}")

        # Shared expert gate (sigmoid scaling)
        shexp_weight = 1.0
        if shexp_gate_name in tensors:
            sg_info = tensors[shexp_gate_name]
            sg_raw = read_tensor_raw(f, tensor_data_offset, sg_info)
            # This is a 1D weight vector [hidden_dim], compute dot with ffn_norm -> scalar
            # Then sigmoid
            sg_K = sg_info.dims[0]
            print(f"  Shared expert gate: dims={sg_info.dims[:sg_info.n_dims]} "
                  f"type={GGML_TYPE_NAME.get(sg_info.type_id, '?')}")

            if sg_info.n_dims == 1:
                # 1D: dot product with ffn_norm -> scalar
                sg_logit = dmmv_f32(sg_raw, ffn_norm, 1, sg_K)[0] if sg_info.type_id == GGML_TYPE_F32 else 0.0
            elif sg_info.n_dims == 2:
                # 2D [hidden_dim, 1]: single row
                sg_logit = dmmv_f32(sg_raw, ffn_norm, 1, sg_K)[0] if sg_info.type_id == GGML_TYPE_F32 else 0.0
            else:
                sg_logit = 0.0

            shexp_weight = sigmoid(sg_logit)
            print(f"  Shared expert gate logit: {sg_logit:.6f}, sigmoid: {shexp_weight:.6f}")

        # Scale shared expert output
        for i in range(hidden_dim):
            shexp_out[i] *= shexp_weight

        print(f"  Scaled shared expert out[0..8]: {shexp_out[:8]}")
    else:
        print("  No shared expert tensors found")

    print(f"  Shared expert time: {time.time() - t_start:.2f}s")

    # -----------------------------------------------------------------------
    # Step 9: Residual add
    # -----------------------------------------------------------------------
    print("\n=== Step 9: Residual add ===")
    # hidden = BOS_embed + moe_output + shared_expert_output
    # In ZINC: moe_out_buf accumulates both routed + shared experts,
    # then hidden_buf += moe_out_buf. So final = hidden + moe_out + shexp_out
    final_hidden = [hidden[i] + moe_out[i] + shexp_out[i] for i in range(hidden_dim)]

    print(f"  hidden (BOS)[0..8]:    {hidden[:8]}")
    print(f"  moe_out[0..8]:         {moe_out[:8]}")
    print(f"  shexp_out[0..8]:       {shexp_out[:8]}")
    print(f"  final_hidden[0..8]:    {final_hidden[:8]}")
    print(f"  final_hidden[2040..]:  {final_hidden[2040:]}")
    print(f"  final_hidden norm:     {math.sqrt(sum(v*v for v in final_hidden)):.6f}")

    # Also print as comma-separated for easy pasting
    print(f"\n  === FINAL HIDDEN STATE (first 16 values) ===")
    for i in range(min(16, hidden_dim)):
        print(f"  [{i:4d}] = {final_hidden[i]:.8f}")

    print(f"\n  === FINAL HIDDEN STATE (last 8 values) ===")
    for i in range(max(0, hidden_dim - 8), hidden_dim):
        print(f"  [{i:4d}] = {final_hidden[i]:.8f}")

    f.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
