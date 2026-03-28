#!/usr/bin/env python3
"""CPU reference for layer 0 SSM (delta-net) — compare against ZINC GPU output."""
import struct, math, sys

def read_gguf(path):
    f = open(path, 'rb')
    f.read(4+4+8+8)
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

def matvec_q8_0(f, base, offset, x, M, K):
    y = []
    for r in range(M):
        row = dequant_q8_0_row(f, base, offset, r, K)
        dot = sum(row[j] * x[j] for j in range(K))
        y.append(dot)
    return y

def rms_norm(x, w, eps=1e-6):
    n = len(x)
    sq = sum(v*v for v in x)
    rms = math.sqrt(sq/n + eps)
    return [w[i] * x[i] / rms for i in range(n)]

def silu(x):
    if abs(x) > 80: return x if x > 0 else 0.0
    return x / (1 + math.exp(-x))

def sigmoid(x):
    if x > 80: return 1.0
    if x < -80: return 0.0
    return 1.0 / (1.0 + math.exp(-x))

def softplus(x):
    if x > 80: return x
    return math.log(1.0 + math.exp(x))

def l2_norm(v, eps=1e-12):
    sq = sum(x*x for x in v)
    n = math.sqrt(sq + eps)
    return [x/n for x in v]

# --- Main ---
path = sys.argv[1] if len(sys.argv) > 1 else '/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf'
f, offsets, aligned = read_gguf(path)

HIDDEN = 2048
D_INNER = 4096
D_CONV = 4
D_STATE = 128  # head_k_dim
N_GROUP = 16   # num_k_heads
DT_RANK = 32   # num_v_heads
HEAD_V_DIM = D_INNER // DT_RANK  # 128
CONV_CHANNELS = D_INNER + 2 * N_GROUP * D_STATE  # 8192
QK_DIM = D_STATE * N_GROUP  # 2048

# 1. BOS embedding
_, _, off = offsets['token_embd.weight']
bos = dequant_q8_0_row(f, aligned, off, 1, HIDDEN)

# 2. attn_norm → norm_buf
_, _, off = offsets['blk.0.attn_norm.weight']
attn_norm_w = read_f32(f, aligned, off, HIDDEN)
norm_buf = rms_norm(bos, attn_norm_w)
print('attn_norm[0..4]:', [round(v, 6) for v in norm_buf[:4]])

# 3. SSM projections (wqkv, attn_gate, ssm_alpha, ssm_beta)
print('\n=== SSM PROJECTIONS ===')

# wqkv: norm_buf → conv_channels (8192)
_, _, wqkv_off = offsets['blk.0.attn_qkv.weight']
qkv_proj = matvec_q8_0(f, aligned, wqkv_off, norm_buf, CONV_CHANNELS, HIDDEN)
print('qkv_proj[0..4]:', [round(v, 6) for v in qkv_proj[:4]])
print('qkv_proj L2:', round(math.sqrt(sum(v*v for v in qkv_proj)), 4))

# attn_gate (z): norm_buf → d_inner (4096)
_, _, gate_off = offsets['blk.0.attn_gate.weight']
z_proj = matvec_q8_0(f, aligned, gate_off, norm_buf, D_INNER, HIDDEN)
print('z_proj[0..4]:', [round(v, 6) for v in z_proj[:4]])

# ssm_alpha: norm_buf → dt_rank (32)
_, _, alpha_off = offsets['blk.0.ssm_alpha.weight']
alpha_proj = matvec_q8_0(f, aligned, alpha_off, norm_buf, DT_RANK, HIDDEN)
print('alpha_proj[0..4]:', [round(v, 6) for v in alpha_proj[:4]])

# ssm_beta: norm_buf → dt_rank (32)
_, _, beta_off = offsets['blk.0.ssm_beta.weight']
beta_proj = matvec_q8_0(f, aligned, beta_off, norm_buf, DT_RANK, HIDDEN)
print('beta_proj[0..4]:', [round(v, 6) for v in beta_proj[:4]])

# 4. Conv1d: for first token, state is zero, only current input matters
# conv1d kernel: [d_conv=4, conv_channels=8192], stored as F32
print('\n=== CONV1D ===')
_, _, conv_off = offsets['blk.0.ssm_conv1d.weight']
conv_w = read_f32(f, aligned, conv_off, D_CONV * CONV_CHANNELS)

# First token: conv state is all zeros, so only the last kernel tap applies
# conv_out[ch] = sum over k of (kernel[k, ch] * input_at_position[k])
# With zero history, only k=D_CONV-1 (the current position) contributes
conv_out = []
for ch in range(CONV_CHANNELS):
    val = conv_w[(D_CONV - 1) * CONV_CHANNELS + ch] * qkv_proj[ch]
    # SiLU activation
    conv_out.append(silu(val))

print('conv_out[0..4]:', [round(v, 8) for v in conv_out[:4]])
print('conv_out L2:', round(math.sqrt(sum(v*v for v in conv_out)), 4))

# Wait — ZINC stores conv state before current input, so the convolution
# uses state[0..d_conv-2] + current_input for all d_conv taps.
# For first token with zero state:
# conv_out[ch] = kernel[0]*0 + kernel[1]*0 + kernel[2]*0 + kernel[3]*qkv[ch]
# That's kernel tap 3 (last) applied to the current input.
# BUT: the kernel layout in GGUF might be [ch * d_conv + k] not [k * channels + ch]
# Let me check both:
conv_out_alt = []
for ch in range(min(CONV_CHANNELS, 8)):
    val_v1 = conv_w[(D_CONV - 1) * CONV_CHANNELS + ch] * qkv_proj[ch]  # [k*channels + ch]
    val_v2 = conv_w[ch * D_CONV + (D_CONV - 1)] * qkv_proj[ch]          # [ch*d_conv + k]
    print('  ch=%d: layout1=%.8f layout2=%.8f qkv=%.6f k_last1=%.6f k_last2=%.6f' % (
        ch, silu(val_v1), silu(val_v2), qkv_proj[ch],
        conv_w[(D_CONV - 1) * CONV_CHANNELS + ch],
        conv_w[ch * D_CONV + (D_CONV - 1)]))

# 5. Split Q/K/V from conv output
# llama.cpp: [Q(qk_dim=2048), K(qk_dim=2048), V(d_inner=4096)]
q_ssm = conv_out[0:QK_DIM]
k_ssm = conv_out[QK_DIM:2*QK_DIM]
v_ssm = conv_out[2*QK_DIM:2*QK_DIM+D_INNER]
print('\nq_ssm[0..4]:', [round(v, 6) for v in q_ssm[:4]])
print('k_ssm[0..4]:', [round(v, 6) for v in k_ssm[:4]])
print('v_ssm[0..4]:', [round(v, 6) for v in v_ssm[:4]])

# 6. L2 normalize Q and K per-head
for h in range(N_GROUP):
    q_ssm[h*D_STATE:(h+1)*D_STATE] = l2_norm(q_ssm[h*D_STATE:(h+1)*D_STATE])
    k_ssm[h*D_STATE:(h+1)*D_STATE] = l2_norm(k_ssm[h*D_STATE:(h+1)*D_STATE])

# 7. Q scaling
q_scale = 1.0 / math.sqrt(D_STATE)
q_ssm = [v * q_scale for v in q_ssm]

# 8. Gate and beta computation
_, _, dt_bias_off = offsets['blk.0.ssm_dt.bias']
dt_bias = read_f32(f, aligned, dt_bias_off, DT_RANK)

_, _, ssm_a_off = offsets['blk.0.ssm_a']
ssm_a = read_f32(f, aligned, ssm_a_off, DT_RANK)

gate = []
beta_vals = []
for i in range(DT_RANK):
    a = alpha_proj[i] + dt_bias[i]
    sp = softplus(a)
    gate.append(sp * ssm_a[i])
    beta_vals.append(sigmoid(beta_proj[i]))

print('\ngate[0..4]:', [round(v, 6) for v in gate[:4]])
print('beta[0..4]:', [round(v, 6) for v in beta_vals[:4]])

# 9. Delta-net with zero initial state
# State is [HEAD_V_DIM, HEAD_V_DIM] per head (square since D_STATE == HEAD_V_DIM)
ssm_state = [0.0] * (DT_RANK * HEAD_V_DIM * HEAD_V_DIM)
ssm_output = [0.0] * D_INNER

for h in range(DT_RANK):
    s_base = h * HEAD_V_DIM * HEAD_V_DIM
    g_val = math.exp(gate[h])
    b_val = beta_vals[h]

    # K head mapping
    k_hi = h % N_GROUP if N_GROUP != DT_RANK else h
    k_head = k_ssm[k_hi * D_STATE:(k_hi + 1) * D_STATE]
    v_head = v_ssm[h * HEAD_V_DIM:(h + 1) * HEAD_V_DIM]

    # Decay (state is zero, so this is no-op)
    # sk = s @ k (all zero)
    # d = beta * (v - sk) = beta * v
    # s += outer(k, d) = outer(k, beta*v)
    for row in range(HEAD_V_DIM):
        sk = 0  # state is zero
        d_val = b_val * (v_head[row] - sk)
        for col in range(min(HEAD_V_DIM, len(k_head))):
            ssm_state[s_base + col * HEAD_V_DIM + row] += k_head[col] * d_val

    # Read: o = s @ q
    q_hi = h % N_GROUP if N_GROUP != DT_RANK else h
    q_head = q_ssm[q_hi * D_STATE:(q_hi + 1) * D_STATE]
    for row in range(HEAD_V_DIM):
        val = 0
        for col in range(min(HEAD_V_DIM, len(q_head))):
            val += ssm_state[s_base + row * HEAD_V_DIM + col] * q_head[col]
        ssm_output[h * HEAD_V_DIM + row] = val

print('\nssm_output[0..4]:', [round(v, 8) for v in ssm_output[:4]])
print('ssm_output L2:', round(math.sqrt(sum(v*v for v in ssm_output)), 6))

# 10. Gated norm: RMS_norm(ssm_output) * SiLU(z)
_, _, snorm_off = offsets['blk.0.ssm_norm.weight']
ssm_norm_w = read_f32(f, aligned, snorm_off, D_STATE)

for h in range(DT_RANK):
    o_sl = ssm_output[h*HEAD_V_DIM:(h+1)*HEAD_V_DIM]
    z_sl = z_proj[h*HEAD_V_DIM:(h+1)*HEAD_V_DIM]
    sq = sum(v*v for v in o_sl)
    rms = math.sqrt(sq / HEAD_V_DIM + 1e-6)
    for i in range(HEAD_V_DIM):
        nv = o_sl[i] / rms
        nv *= ssm_norm_w[i % D_STATE]
        zv = z_sl[i]
        ssm_output[h*HEAD_V_DIM + i] = nv * silu(zv)

print('gated_norm_output[0..4]:', [round(v, 8) for v in ssm_output[:4]])

# 11. ssm_out projection: d_inner → hidden_dim
_, _, ssm_out_off = offsets['blk.0.ssm_out.weight']
ssm_proj = matvec_q8_0(f, aligned, ssm_out_off, ssm_output, HIDDEN, D_INNER)
print('ssm_proj[0..4]:', [round(v, 8) for v in ssm_proj[:4]])
print('ssm_proj L2:', round(math.sqrt(sum(v*v for v in ssm_proj)), 6))

# 12. Residual: hidden = bos + ssm_proj
hidden_after_ssm = [bos[i] + ssm_proj[i] for i in range(HIDDEN)]
print('\nhidden_after_ssm[0..4]:', [round(v, 8) for v in hidden_after_ssm[:4]])
# This is what hidden_buf should be before post_attention_norm
