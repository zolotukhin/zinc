<!-- All project instructions live in AGENTS.md — this file exists so Claude Code picks them up automatically. -->
See [AGENTS.md](AGENTS.md)

## Active Technologies
- Zig 0.14-dev (host), GLSL 4.60 (shaders) compiled to SPIR-V via system glslc (shaderc 2023.8) + Vulkan 1.3 (RADV driver, Mesa 25.0.7), VK_KHR_cooperative_matrix (003-decode-performance)
- GGUF model files, memory-mapped with DMA to GPU VRAM (003-decode-performance)
- Zig 0.14-dev + Zig std.net, std.json, existing ZINC inference engine (004-openai-api-server)
- Zig 0.15.2+ (host), MSL / Metal Shading Language (GPU shaders), Objective-C (thin shim) + Metal.framework, Foundation.framework, SPIRV-Cross (build tool), Xcode Command Line Tools (005-apple-silicon-inference)
- GGUF model files (mmap'd, zero-copy via `newBufferWithBytesNoCopy`) (005-apple-silicon-inference)

## Recent Changes
- 003-decode-performance: Added Zig 0.14-dev (host), GLSL 4.60 (shaders) compiled to SPIR-V via system glslc (shaderc 2023.8) + Vulkan 1.3 (RADV driver, Mesa 25.0.7), VK_KHR_cooperative_matrix
