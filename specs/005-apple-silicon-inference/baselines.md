# Baselines: Apple Silicon Inference

**Hardware**: Mac Studio M4 Max, 64 GB unified memory, 40-core GPU
**Model**: Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf (20.7 GB, MoE 35B/3B active)

## llama.cpp Metal (build afe65aa)

Measured with `llama-bench`:

| Metric | Value |
|--------|-------|
| Prefill (pp512) | **1421 tok/s** |
| Decode (tg128) | **72.93 tok/s** |
| Backend | MTL,BLAS |
| Threads | 12 |
| GPU family | MTLGPUFamilyApple9 (M4) |
| Unified memory | true |
| simdgroup matrix mul | true |
| bfloat support | true |

## mlx-lm (TBD)

TODO: Install and benchmark `mlx-lm` with `mlx-community/Qwen3.5-35B-A3B-4bit`.

## ZINC Metal (current)

| Metric | Value | Notes |
|--------|-------|-------|
| Model load | <2s | Zero-copy mmap |
| Decode (CPU LM head only) | ~5s/token | 248K × 2048 CPU matmul, no GPU layers |
| GPU dispatch | not yet | Shaders cross-compiled, dispatch pending |

## Targets

| Metric | llama.cpp | ZINC Target | Improvement |
|--------|-----------|-------------|-------------|
| Single decode | 72.93 tok/s | ≥80 tok/s | +10% |
| Prefill (512) | 1421 tok/s | ≥350 tok/s | (lower target initially) |
| 5-req aggregate | N/A | ≥200 tok/s | llama.cpp has no batching |
