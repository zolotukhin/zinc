# Quickstart: Decode Performance Optimization

**Feature**: 003-decode-performance
**Date**: 2026-03-28

## Prerequisites

- AMD Radeon AI PRO R9700 (RDNA4) or similar RDNA3/4 GPU
- Linux with RADV driver (Mesa 25.0.7+), `amdgpu.ras_enable=0`
- Qwen3.5-35B-A3B-UD-Q4_K_XL GGUF model files
- ZINC built from source: `zig build -Doptimize=ReleaseFast`

## Baseline (before optimization)

Record the current performance for comparison:

```bash
# SSH to test node
ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST

# Run baseline
./zig-out/bin/zinc \
  --model /path/to/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt "The capital of France is" \
  --max-tokens 32

# Expected output: ~4 tok/s, coherent English
# Save the first 32 generated tokens as reference
```

## Validation After Each Step

### Step 0: Profiling

```bash
# Run with profiling enabled
./zig-out/bin/zinc \
  --model /path/to/model.gguf \
  --prompt "The capital of France is" \
  --max-tokens 32 \
  --profile

# Expected: per-layer timing breakdown showing CPU SSM vs GPU dispatch vs submit overhead
```

### Step 1: GPU-Side SSM

```bash
# After implementing GPU SSM shaders
zig build -Doptimize=ReleaseFast && \
./zig-out/bin/zinc \
  --model /path/to/model.gguf \
  --prompt "The capital of France is" \
  --max-tokens 32

# Verify: identical first 32 tokens as baseline
# Expected: measurable speedup (CPU SSM bottleneck removed)
```

### Step 2: GPU-Side Router

```bash
# After implementing softmax_topk shader
# Same command as above
# Verify: identical first 32 tokens as baseline
# Expected: further speedup (40 fewer submits)
```

### Step 3: GPU-Side Shared Expert Gate

```bash
# Same command as above
# Verify: identical first 32 tokens as baseline
# Expected: further speedup (40 fewer submits)
```

### Step 4: Command Buffer Batching

```bash
# Same command as above
# Verify: identical first 32 tokens as baseline
# Expected: approaching 27+ tok/s (memory-bandwidth floor)
```

### Final Validation

```bash
# Full 256-token generation
./zig-out/bin/zinc \
  --model /path/to/model.gguf \
  --prompt "The capital of France is" \
  --max-tokens 256

# Success criteria:
# - ≥27 tok/s (milestone 1)
# - ≥107 tok/s (target)
# - Coherent English output
# - No VRAM overflow
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Garbage output after GPU SSM | Numerical divergence in delta-net | Compare per-layer SSM output vs CPU reference |
| Wrong experts selected | softmax_topk shader bug | Compare expert_ids vs CPU topKSoftmax |
| Descriptor pool exhaustion | Too many layers in one cmd buffer | Add pool-full detection and mid-batch submit |
| No speedup after batching | Bottleneck is elsewhere | Check profiling output for actual time split |
| VRAM OOM at init | SSM state buffers too large | Check total: model (21 GB) + KV cache + SSM state (84 MB) vs 32 GB |
