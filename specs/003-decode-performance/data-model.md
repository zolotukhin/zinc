# Data Model: Decode Performance Optimization

**Feature**: 003-decode-performance
**Date**: 2026-03-28

## GPU Buffer Entities

### SSM Conv State (NEW — persistent)

Per-layer GPU buffer holding the sliding window of previous inputs for 1D convolution.

| Attribute | Value |
|-----------|-------|
| Scope | Per-layer (40 buffers) |
| Shape | (d_conv−1) × conv_channels = 3 × 8192 = 24,576 floats |
| Size per layer | 96 KB |
| Total | 40 × 96 KB = 3.75 MB |
| Memory type | Device-local |
| Lifetime | Initialized to zero at engine init, persists across all tokens |
| Access pattern | Read+write per SSM layer per token (shift left, write newest) |

### SSM Recurrent State (NEW — persistent)

Per-layer GPU buffer holding the delta-net state matrices.

| Attribute | Value |
|-----------|-------|
| Scope | Per-layer (40 buffers, but only 30 SSM layers use them) |
| Shape | num_heads × head_v_dim × head_v_dim = 32 × 128 × 128 = 524,288 floats |
| Size per layer | 2 MB |
| Total | 40 × 2 MB = 80 MB (allocate for all layers; SSM layers use, attention layers ignore) |
| Memory type | Device-local |
| Lifetime | Initialized to zero at engine init, persists across all tokens |
| Access pattern | Read+write per SSM layer per token (decay + outer product + accumulate) |

### Router Output Buffer (NEW)

GPU buffer holding the output of the softmax+top-k shader for MoE routing.

| Attribute | Value |
|-----------|-------|
| Scope | Single shared buffer (reused per layer) |
| Shape | expert_ids[k] (u32) + expert_weights[k] (f32) = 8 + 8 = 16 values |
| Size | 64 bytes |
| Memory type | Host-visible (for expert dispatch loop to read IDs on CPU) OR device-local with staging |
| Lifetime | Written per MoE layer, consumed immediately by expert dispatch |
| Access pattern | Write by softmax_topk shader, read by CPU for dispatch loop (until expert dispatch is also GPU-driven) |

**Note on Router Output**: Initially the expert dispatch loop still runs on CPU (iterating over selected experts), so expert_ids must be readable by CPU. The router output buffer should be host-visible or have a staging copy. This is a temporary compromise — full GPU-side expert dispatch is a future optimization.

### Existing Buffers (modified usage)

| Buffer | Current Usage | Modified Usage |
|--------|--------------|----------------|
| `router_logits_buf` | Router DMMV output → readback to CPU | Router DMMV output → stays on GPU, consumed by softmax_topk shader |
| `logits_staging` | SSM projection readback to CPU | No longer used for SSM (projections stay on GPU) |
| `router_staging` | Router logits readback + gate scalar readback | Only used for router output buffer staging (expert_ids + weights) |
| `ssm_hidden_staging` | CPU SSM output upload to GPU | No longer needed (SSM output stays on GPU) |
| `ssm_conv_states` (CPU) | Per-layer CPU-side conv state arrays | REPLACED by GPU SSM Conv State buffers |
| `ssm_states` (CPU) | Per-layer CPU-side recurrent state arrays | REPLACED by GPU SSM Recurrent State buffers |

## State Transitions

### Token Decode Lifecycle (post-optimization)

```
1. embedToken(token_id)                    [CPU → embed_staging → hidden_buf]
2. BEGIN single command buffer
3. For each of 40 layers:
   ├─ RMS norm                             [GPU: hidden_buf → norm_buf]
   ├─ IF attention layer:
   │   └─ Q/K/V DMMV → RoPE → KV cache → flash attn → gate → O-proj → residual
   ├─ ELSE SSM layer:
   │   └─ projections → conv1d_silu → delta_net → gated_norm → ssm_out → residual
   ├─ Post-attention norm                  [GPU: hidden_buf → ffn_norm_buf]
   └─ MoE:
       ├─ Router DMMV → softmax_topk       [GPU: all on-device]
       ├─ SUBMIT + WAIT (read expert_ids)  [only remaining CPU readback]
       ├─ For each of 8 experts:
       │   └─ gate+up DMMV → SwiGLU → down DMMV → weighted accumulate
       ├─ Shared expert: gate+up → SwiGLU → down → sigmoid_scale_acc
       └─ FFN residual
4. Final norm → LM head DMMV → logits readback
5. END command buffer, SUBMIT + WAIT
6. CPU argmax sampling
```

Post-optimization submit count: ~42 (1 per MoE layer for expert ID readback + 1 final).
With future GPU-side expert dispatch: 1 submit per token.
