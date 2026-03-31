# Qwen3.5 Debug Notes

Last updated: 2026-03-29

## Scope

This note captures the current debugging state for the hybrid Qwen3.5 family in ZINC.

Models exercised during this pass:

- `Qwen3.5-2B-Q4_K_M.gguf`
- `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

Primary test prompt:

- `The capital of France is`

Primary remote node:

- RDNA4 node at `root@71.212.158.115:2223`

## Current Status

- Local `zig build test --summary all` passes.
- The small `qwen35` model still does not generate correct text.
- The larger `qwen35moe` model is also not currently correct from source builds on the RDNA4 node.
- Do not treat `/root/zinc` on the node as a clean baseline. On 2026-03-29 it was on commit `2bb74e5` but had a large dirty patch stack in inference files.

Observed first-token states on 2026-03-29:

- Clean temp build from current git `HEAD` before the packed-attention fix: 35B first token `195229`
- Current packed-attention-aligned tree in `/tmp/zinc-qwen35-clean`: 2B first token `228`
- Current packed-attention-aligned tree in `/tmp/zinc-qwen35-clean`: 35B first token `264`
- Dirty `/root/zinc` on the node after rebuild: 35B first token `264`

The implication is simple: if 35B worked on the node at some earlier point, that was a different remote state than the current `/root/zinc` tree.

## Verified-Correct Pieces

These are not the main blockers anymore:

- Flash attention math is self-consistent on the current tensors.
  - `ATTN_SELFTEST` at `seq_len=1` matched the current V slice.
  - `ATTN_REFTEST` at `seq_len=5` matched a naive CPU attention reference to about `1e-6`.
- Multiple DMMV paths were checked against CPU reference rows and matched closely enough:
  - `wqkv`
  - `ffn_gate`
  - `ffn_up`
  - `attn_q`
  - `attn_output`
  - `ffn_down`
- The original 2B SSM blow-up is fixed.
  - Layer-0 `delta_out` no longer explodes to huge values over decode.
- Qwen3.5 norm weights do not look zero-centered.
  - Example from 2B layer 3:
    - `attn_norm[0..4]=[1.118652,1.147461,0.930176,1.166992]`
    - `q_norm[0..4]=[1.419922,1.316406,1.343750,1.455078]`
    - `k_norm[0..4]=[1.451172,1.341797,1.609375,1.464844]`

That means the remaining mismatch is more likely in hybrid linear-attention sequencing/math than in the plain full-attention or DMMV pieces.

## Architecture References

### Full Attention

The strongest current reference is vLLM's `Qwen3NextAttention`:

- Source:
  - `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next.py`

Relevant behavior:

- `qkv_proj` can emit Q plus an attention-output gate.
- The packed Q path is per-head contiguous halves, not element-interleaved.
- `q_norm` and `k_norm` are applied before RoPE.
- `sigmoid(gate)` is applied to the attention output, before `o_proj`.

The current ZINC attention path was updated to match that behavior more closely:

- packed Q/gate is split as per-head `[Q(head_dim), gate(head_dim)]`
- gate is applied after flash attention, not on Q before attention

This moved outputs in the right direction structurally, but did not finish the fix.

### Linear Attention / Gated Delta Net

The most important current reference is vLLM's `GatedDeltaNetAttention`:

- Source:
  - `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/mamba/gdn_linear_attn.py`

For the `Qwen3.5` path (`gqa_interleaved_layout=False`), the reference does this:

1. Split `in_proj_qkvz` into `[q, k, v, z]`
2. Split `in_proj_ba` into `[b, a]`
3. Run causal conv on the mixed `qkv`
4. Compute
   - `g = -exp(A_log) * softplus(a + dt_bias)`
   - `beta = sigmoid(b)`
5. Run the gated delta-net recurrent core on `q, k, v, g, beta`
6. Apply `RMSNormGated(core_attn_out, z)`
7. Run the output projection

This is the main remaining area to validate against ZINC's CPU/GPU SSM path.

## Current ZINC Mapping Assumptions

The current code assumes roughly:

- `attn_qkv.weight` -> fused linear-attention `qkv`
- `attn_gate.weight` -> `z`
- `ssm_alpha.weight` -> one half of `ba`
- `ssm_beta.weight` -> the other half of `ba`
- `ssm_dt.bias` -> `dt_bias`
- `ssm_a` -> some representation of `A`
- `ssm_norm.weight` -> shared gated RMS norm weight

The highest-risk unresolved detail is the exact meaning of `ssm_a`.

The reference wants:

- `g = -exp(A_log) * softplus(a + dt_bias)`

The current CPU path effectively does:

- `gate_arr = softplus(alpha + dt_bias) * ssm_a`
  or `-softplus(...)` when `ssm_a` is absent

If GGUF stores `A_log` rather than `-exp(A_log)`, this is still wrong by an exponential.

## Useful Diagnostics Captured

2B SSM diagnostics from 2026-03-29:

```text
SSM tensor types: conv1d=f32 dt_bias=f32 ssm_a=f32 n_group=16 dt_rank=16 d_state=128 head_v=128
SSM gate L0 pos=0: alpha0=2.890070 dt_bias0=3.703125 ssm_a0=-0.772740 gate_log=[-5.095884,-0.000000] decay=[0.006122,1.000000] beta=[0.357056,0.926990]
ssm_norm.weight: type=f32 n_dims=1 dims=[128,1] elems=128 d_state=128 d_inner=2048 head_v=128 per_head=false
```

Notes:

- `ssm_a` is present and `f32`
- `ssm_norm.weight` is shared over `d_state`, not per-`d_inner`
- decay is no longer numerically blowing up

## Next Checks

- Verify whether GGUF `ssm_a` is already `-exp(A_log)` or whether it still stores `A_log`
- Compare ZINC's `ssm_alpha/ssm_beta` mapping against the reference `b/a` ordering
- Compare one full linear-attention layer end-to-end against a reference implementation, not just internal self-consistency
- Keep using the larger 35B model as a regression guard, but do not treat its current node checkout as a known-good baseline

## Repro Commands

Sync and build on the RDNA4 node:

```bash
rsync -az --delete \
  --exclude '.git' \
  --exclude '.zig-cache' \
  --exclude 'zig-out' \
  --exclude 'node_modules' \
  --exclude '.DS_Store' \
  --exclude 'site' \
  --exclude 'research/turboquant-pytorch-master' \
  -e "ssh -p 2223" . root@71.212.158.115:/tmp/zinc-qwen35-clean/

ssh -p 2223 root@71.212.158.115 'cd /tmp/zinc-qwen35-clean && zig build'
```

2B first-token smoke:

```bash
ssh -p 2223 root@71.212.158.115 \
  'cd /tmp/zinc-qwen35-clean && \
   timeout 30 env ZINC_DEBUG=1 RADV_PERFTEST=coop_matrix \
   ./zig-out/bin/zinc --profile \
   -m /root/models/Qwen3.5-2B-Q4_K_M.gguf \
   --prompt "The capital of France is" 2>&1 | \
   rg --line-buffered "decode\\[0\\]|TOP5\\[0\\]|Output text:"'
```

35B first-token smoke:

```bash
ssh -p 2223 root@71.212.158.115 \
  'cd /tmp/zinc-qwen35-clean && \
   timeout 30 env ZINC_DEBUG=1 RADV_PERFTEST=coop_matrix \
   ./zig-out/bin/zinc --profile \
   -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
   --prompt "The capital of France is" 2>&1 | \
   rg --line-buffered "decode\\[0\\]|TOP5\\[0\\]|Output text:"'
```
