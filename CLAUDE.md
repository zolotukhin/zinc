<!-- All project instructions live in AGENTS.md — this file exists so Claude Code picks them up automatically. -->
See [AGENTS.md](AGENTS.md)

## Active Technologies
- Zig 0.15.2+ (host language and build system)
- GLSL 4.60 (Vulkan compute shaders) compiled to SPIR-V via system glslc
- Vulkan 1.3 (AMD RDNA3/RDNA4 GPU inference)
- MSL / Metal Shading Language (Apple Silicon GPU inference)
- Objective-C thin shim for Metal.framework bridge
- GGUF model files (mmap'd, zero-copy on both backends)

## Supported Model Architectures
- **Qwen3 / Qwen3.5** — standard dense + MoE (Q4_K, Q5_K, Q8_0)
- **Gemma 3** — with Gemma-specific post-attention norm and GeGLU activation
- **OpenAI GPT-OSS** — MoE with OAI SwiGLU, MXFP4 expert weights, Q5_0 attention weights, attention sinks, ISWA, YaRN RoPE

## Quantization Formats
Q4_K, Q5_K, Q6_K, Q8_0, Q5_0, MXFP4 (type 39), F16, F32
