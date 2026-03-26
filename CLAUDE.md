# zinc Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-25

## Active Technologies

- Zig 0.15.2+ (host code) + GLSL (compute shaders compiled to SPIR-V) + Vulkan 1.3 API (direct C ABI calls), system glslc (shaderc 2023.8) (001-zinc-inference-engine)

## Project Structure

```text
src/
tests/
```

## Commands

# Add commands for Zig 0.15.2+ (host code) + GLSL (compute shaders compiled to SPIR-V)

## Code Style

Zig 0.15.2+ (host code) + GLSL (compute shaders compiled to SPIR-V): Follow standard conventions

## Recent Changes

- 001-zinc-inference-engine: Added Zig 0.15.2+ (host code) + GLSL (compute shaders compiled to SPIR-V) + Vulkan 1.3 API (direct C ABI calls), system glslc (shaderc 2023.8)

<!-- MANUAL ADDITIONS START -->

## Remote Test Node (RDNA4)

A remote test node with AMD Radeon AI PRO R9700 (RDNA4, 32GB, 576 GB/s) is available for building, running, and benchmarking ZINC.

Access via environment variables (set in `.env`, gitignored):
- `ZINC_HOST` — hostname/IP
- `ZINC_USER` — SSH username
- `ZINC_PORT` — SSH port

Connect: `ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST`

Use this node for:
- Compiling and running ZINC with `zig build`
- Running inference benchmarks on real RDNA4 hardware
- Validating GPU kernel correctness against llama.cpp reference
- Measuring bandwidth utilization, dispatch overhead, and tok/s

<!-- MANUAL ADDITIONS END -->
