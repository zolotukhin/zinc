#!/usr/bin/env python3
"""CPU reference for layer 0 MoE FFN — compare against ZINC GPU output."""
import struct, math, sys

def read_gguf(path):
    f = open(path, 'rb')
    f.read(4+4+8+8)  # magic, ver, n_tensors, n_meta
    # Skip metadata
    for _ in range(52):
        klen = struct.unpack('<Q', f.read(8))[0]; f.read(klen)
        vtype = struct.unpack('<I', f.read(4))[0]
        if vtype == 8: sl = struct.unpack('<Q', f.read(8))[0]; f.read(sl)
        elif vtype in (0,1): f.read(1)
        elif vtype in (2,3): f.read(2)
        elif vtype in (4,5,6): f.read(4)
        elif vtype == 7: f.read(1)
        elif vtype == 9:
            etype = struct.unpack('<I', f.read(4))[0]; elen = struct.unpack('<Q', f.read(8))[0]
            for _ in range(elen):
                if etype == 8: sl = struct.unpack('<Q', f.read(8))[0]; f.read(sl)
                elif etype in (4,5,6): f.read(4)
                elif etype in (0,1): f.read(1)
                elif etype in (2,3): f.read(2)
                elif etype in (10,11,12): f.read(8)
        elif vtype in (10,11,12): f.read(8)
    offsets = {}
    for i in range(733):
        nlen = struct.unpack('<Q', f.read(8))[0]; name = f.read(nlen).decode()
        ndims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        offsets[name] = (dims, ttype, offset)
    aligned = (f.tell() + 31) & ~31
    return f, offsets, aligned

def read_f32(f, base, offset, n):
    f.seek(base + offset)
    return list(struct.unpack('<%df' % n, f.read(n * 4)))

def dequant_q8_0_row(f, base, offset, row, cols):
    bpr = cols // 32
    f.seek(base + offset + row * bpr * 34)
    data = f.read(bpr * 34)
    vals = []
    for b in range(bpr):
        bo = b * 34
        sb = struct.unpack_from('<e', data, bo)[0]
        for j in range(32):
            v = data[bo + 2 + j]
            if v > 127: v -= 256
            vals.append(float(v) * float(sb))
    return vals

def get_scale_min_k4(j, scales):
    if j < 4:
        return scales[j] & 63, scales[j+4] & 63
    else:
        sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
        m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
        return sc, m

def dequant_q4k_row(data, row, cols):
    """Dequantize one row of Q4_K data. Consecutive sub-block pairing."""
    bpr = cols // 256
    vals = []
    for bi in range(bpr):
        bb = row * bpr * 144 + bi * 144
        d = struct.unpack_from('<e', data, bb)[0]
        dmin = struct.unpack_from('<e', data, bb + 2)[0]
        scales = data[bb+4:bb+16]
        qs = data[bb+16:bb+144]
        qo = 0
        for sp in range(4):
            sb_lo = sp * 2
            sb_hi = sp * 2 + 1
            sc_lo, m_lo = get_scale_min_k4(sb_lo, scales)
            sc_hi, m_hi = get_scale_min_k4(sb_hi, scales)
            d_lo = d * sc_lo; bias_lo = dmin * m_lo
            d_hi = d * sc_hi; bias_hi = dmin * m_hi
            for e in range(32):
                vals.append(d_lo * (qs[qo+e] & 0xF) - bias_lo)
            for e in range(32):
                vals.append(d_hi * (qs[qo+e] >> 4) - bias_hi)
            qo += 32
    return vals

def dequant_q5k_row(data, row, cols):
    """Dequantize one row of Q5_K data."""
    bpr = cols // 256
    vals = []
    for bi in range(bpr):
        bb = row * bpr * 176 + bi * 176
        d = struct.unpack_from('<e', data, bb)[0]
        dmin = struct.unpack_from('<e', data, bb + 2)[0]
        scales = data[bb+4:bb+16]
        qh_data = data[bb+16:bb+48]
        qs = data[bb+48:bb+176]
        for g in range(4):
            sb_lo = g * 2
            sb_hi = g * 2 + 1
            sc_lo, m_lo = get_scale_min_k4(sb_lo, scales)
            sc_hi, m_hi = get_scale_min_k4(sb_hi, scales)
            d_lo = d * sc_lo; bias_lo = dmin * m_lo
            d_hi = d * sc_hi; bias_hi = dmin * m_hi
            qs_base = g * 32
            for e in range(32):
                q_lo = qs[qs_base + e] & 0xF
                q_hi = qs[qs_base + e] >> 4
                qh_val = qh_data[e]
                hb_lo = (qh_val >> sb_lo) & 1
                hb_hi = (qh_val >> sb_hi) & 1
                vals.append(d_lo * (q_lo | (hb_lo << 4)) - bias_lo)
                vals.append(d_hi * (q_hi | (hb_hi << 4)) - bias_hi)
    return vals

def rms_norm(x, w, eps=1e-6):
    n = len(x)
    sq = sum(v*v for v in x)
    rms = math.sqrt(sq/n + eps)
    return [w[i] * x[i] / rms for i in range(n)]

def silu(x):
    return x / (1 + math.exp(-x)) if abs(x) < 80 else (x if x > 0 else 0)

def matvec_f32(w, x, M, K):
    """F32 matrix-vector multiply: w[M*K] × x[K] → y[M]"""
    y = []
    for r in range(M):
        dot = sum(w[r*K + j] * x[j] for j in range(K))
        y.append(dot)
    return y

def matvec_q4k(f, base, offset, expert_offset, x, M, K):
    """Q4_K matrix-vector multiply with expert byte offset."""
    f.seek(base + offset + expert_offset)
    data = f.read(M * (K // 256) * 144)
    y = []
    for r in range(M):
        row = dequant_q4k_row(data, r, K)
        dot = sum(row[j] * x[j] for j in range(K))
        y.append(dot)
    return y

def matvec_q5k(f, base, offset, expert_offset, x, M, K):
    """Q5_K matrix-vector multiply with expert byte offset."""
    f.seek(base + offset + expert_offset)
    data = f.read(M * (K // 256) * 176)
    y = []
    for r in range(M):
        row = dequant_q5k_row(data, r, K)
        dot = sum(row[j] * x[j] for j in range(K))
        y.append(dot)
    return y

def matvec_q8_0(f, base, offset, x, M, K):
    """Q8_0 matrix-vector multiply."""
    y = []
    for r in range(M):
        row = dequant_q8_0_row(f, base, offset, r, K)
        dot = sum(row[j] * x[j] for j in range(K))
        y.append(dot)
    return y

# --- Main ---
path = sys.argv[1] if len(sys.argv) > 1 else '/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf'
f, offsets, aligned = read_gguf(path)

HIDDEN = 2048
INTER = 512  # per-expert intermediate dim
N_EXPERTS = 256
N_USED = 8

# 1. BOS embedding (token 1)
_, _, off = offsets['token_embd.weight']
bos = dequant_q8_0_row(f, aligned, off, 1, HIDDEN)
print('BOS[0..4]:', [round(v, 8) for v in bos[:4]])

# 2. Layer 0 attn_norm (to get norm_buf)
_, _, off = offsets['blk.0.attn_norm.weight']
attn_norm_w = read_f32(f, aligned, off, HIDDEN)
norm_buf = rms_norm(bos, attn_norm_w)
print('attn_norm[0..4]:', [round(v, 6) for v in norm_buf[:4]])

# 3. Skip SSM (hidden stays as BOS)
hidden = list(bos)

# 4. Post-attention norm → ffn_norm_buf
_, _, off = offsets['blk.0.post_attention_norm.weight']
post_norm_w = read_f32(f, aligned, off, HIDDEN)
ffn_input = rms_norm(hidden, post_norm_w)
print('ffn_input[0..4]:', [round(v, 6) for v in ffn_input[:4]])

# 5. Router: F32 matvec
_, _, off = offsets['blk.0.ffn_gate_inp.weight']
router_w = read_f32(f, aligned, off, HIDDEN * N_EXPERTS)
router_logits = matvec_f32(router_w, ffn_input, N_EXPERTS, HIDDEN)
print('router[0..4]:', [round(v, 4) for v in router_logits[:4]])

# 6. Softmax + top-8
mx = max(router_logits)
exp_rl = [math.exp(x - mx) for x in router_logits]
s = sum(exp_rl)
probs = [e/s for e in exp_rl]
indexed = sorted(enumerate(probs), key=lambda x: -x[1])
top8 = indexed[:8]
top_sum = sum(w for _, w in top8)
top8_normed = [(idx, w/top_sum) for idx, w in top8]
print('top8:', [(idx, round(w, 4)) for idx, w in top8_normed])

# 7. Expert dispatch
_, _, gate_off = offsets['blk.0.ffn_gate_exps.weight']
_, _, up_off = offsets['blk.0.ffn_up_exps.weight']
dims_d, _, down_off = offsets['blk.0.ffn_down_exps.weight']

gate_bytes_per_expert = INTER * (HIDDEN // 256) * 144  # Q4_K
down_bytes_per_expert = HIDDEN * (INTER // 256) * 176  # Q5_K

moe_out = [0.0] * HIDDEN
for expert_id, weight in top8_normed:
    expert_gate_off = expert_id * gate_bytes_per_expert
    expert_up_off = expert_id * gate_bytes_per_expert  # same shape
    expert_down_off = expert_id * down_bytes_per_expert

    gate_out = matvec_q4k(f, aligned, gate_off, expert_gate_off, ffn_input, INTER, HIDDEN)
    up_out = matvec_q4k(f, aligned, up_off, expert_up_off, ffn_input, INTER, HIDDEN)

    # SwiGLU: silu(gate) * up
    swiglu = [silu(gate_out[i]) * up_out[i] for i in range(INTER)]

    down_out = matvec_q5k(f, aligned, down_off, expert_down_off, swiglu, HIDDEN, INTER)

    for i in range(HIDDEN):
        moe_out[i] += weight * down_out[i]

    if expert_id == top8_normed[0][0]:
        print('expert %d gate[0..4]:' % expert_id, [round(v, 6) for v in gate_out[:4]])
        print('expert %d down[0..4]:' % expert_id, [round(v, 6) for v in down_out[:4]])

print('moe_out[0..4]:', [round(v, 6) for v in moe_out[:4]])

# 8. Shared expert
_, _, sg_off = offsets['blk.0.ffn_gate_shexp.weight']
_, _, su_off = offsets['blk.0.ffn_up_shexp.weight']
_, _, sd_off = offsets['blk.0.ffn_down_shexp.weight']

shexp_gate = matvec_q8_0(f, aligned, sg_off, ffn_input, INTER, HIDDEN)
shexp_up = matvec_q8_0(f, aligned, su_off, ffn_input, INTER, HIDDEN)
shexp_swiglu = [silu(shexp_gate[i]) * shexp_up[i] for i in range(INTER)]
shexp_down = matvec_q8_0(f, aligned, sd_off, shexp_swiglu, HIDDEN, INTER)

# Shared expert gate sigmoid
_, _, sgate_off = offsets['blk.0.ffn_gate_inp_shexp.weight']
sgate_w = read_f32(f, aligned, sgate_off, HIDDEN)
sgate_logit = sum(sgate_w[j] * ffn_input[j] for j in range(HIDDEN))
sgate_sigmoid = 1.0 / (1.0 + math.exp(-sgate_logit))
print('shared_expert gate_sigmoid:', round(sgate_sigmoid, 6))
shexp_out = [sgate_sigmoid * shexp_down[i] for i in range(HIDDEN)]
print('shared_expert out[0..4]:', [round(v, 6) for v in shexp_out[:4]])

# 9. Residual: hidden = bos + moe_out + shared_expert_out
for i in range(HIDDEN):
    hidden[i] = bos[i] + moe_out[i] + shexp_out[i]

print()
print('=== FINAL hidden after layer 0 ===')
print('hidden[0..8]:', [round(v, 8) for v in hidden[:8]])
# Compare against ZINC GPU: L0_HIDDEN[0..8]
