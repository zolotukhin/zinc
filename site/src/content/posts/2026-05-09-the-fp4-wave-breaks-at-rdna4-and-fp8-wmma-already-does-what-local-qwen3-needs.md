---
title: "The FP4 wave breaks at RDNA4 and FP8 WMMA already does what local Qwen3 needs"
seoTitle: "FP4 vs FP8 for RDNA4 Inference"
date: "2026-05-09"
tags:
  - zinc
  - rdna4
  - amd
  - quantization
  - fp4
  - nvfp4
  - mxfp4
  - fp8
  - wmma
  - qwen3
  - llama-cpp
  - llm-inference
keywords:
  - NVFP4 RDNA4 inference
  - MXFP4 gfx1201 fallback
  - FP4 WMMA AMD Radeon AI PRO R9700
  - llama.cpp NVFP4 PR 22196
  - GGML_TYPE_NVFP4 type 40
  - FP8 WMMA E4M3FN local Qwen3
  - AITER gfx1201 arch table patch
  - ROCm 7.2.1 silent FP32 fallback
  - OCP MXFP4 microscaling spec
  - v_wmma_f32_16x16x16 instruction set
  - bandwidth bound vs compute bound RDNA4
  - Qwen3 30B FP8 vLLM speedup
excerpt: "FP4 weight formats landed across the GGUF ecosystem in the last two weeks: NVFP4 in mainline llama.cpp, MXFP4 in ik_llama.cpp, with Blackwell-native FP4 tensor cores flipped on in build b8967. The bandwidth math is right on every card. The compute math only works where the tensor cores speak FP4. The RDNA4 ISA on the Radeon AI PRO R9700 lists v_wmma_f32_16x16x16 in fp16, bf16, fp8_e4m3, int8, and iu4 forms but no fp4 form, so an NVFP4 or MXFP4 weight on gfx1201 dequantizes to FP16 before the matmul and lands at the 191 TFLOPS dense FP16 ceiling. The format that earns its place on this card is FP8 E4M3FN, where 383 TFLOPS dense WMMA already exists in hardware and the only thing standing in the way is a small AITER patch and a handful of tuned kernel configs that route gfx1201 through the MI350X Triton path. Skip the FP4 wave on RDNA4. Ship FP8 weights with the patch."
seoDescription: "FP4 vs FP8 on AMD RDNA4 for local LLM inference: why Radeon AI PRO R9700 lacks FP4 WMMA but accelerates FP8 E4M3."
---

Quick answer: FP4 saves memory on RDNA4, but it does not map to a native FP4 WMMA instruction on gfx1201. FP8 E4M3 is the format that has the useful hardware path on Radeon AI PRO R9700, so local Qwen inference should prioritize FP8 kernels before chasing FP4 compute speedups.

NVFP4 has been the headline of the GGUF release cycle for two weeks. The type ID merged into mainline llama.cpp as `GGML_TYPE_NVFP4 = 40` through a sequence of pull requests in late March and April, and [build b8967 on April 29 flipped on the Blackwell-native FP4 tensor-core path on `sm_120`](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/), measuring +43% to +68% prefill on a 27B Qwen NVFP4 checkpoint against the previous build on an RTX 5090. The companion MXFP4 path in ik_llama.cpp has been live since the [Nov 2025 gguf-py constants merge](https://github.com/ikawrakow/ik_llama.cpp/pull/1007), with the OCP open-standard kernels filling in over the months since. The marketing read is that FP4 is now general-purpose for local inference. It is not. Not on RDNA4.

The bandwidth half of the FP4 argument is real on every card. A 27B dense Qwen checkpoint that sits at 17 GB in Q4_K_M lands closer to 14 GB in NVFP4 because the per-block scale shrinks from llama.cpp's 6-byte FP16 super-block-plus-block scheme to NVFP4's 1-byte FP8 scale, and the block size shrinks from 256 to 16. On a 32 GB Radeon AI PRO R9700 that 3 GB of headroom is the difference between running a 64k-context conversation entirely on the card and spilling KV pages to host memory at the upper end of the context window. If the only thing the user cares about is footprint, FP4 is a real win on RDNA4.

The compute half does not transfer. The [RDNA4 ISA on gfx1201](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) exposes its 128 AI accelerators through a small family of `v_wmma_f32_16x16x16` instructions parameterized by element type. The supported element types are `f16`, `bf16`, `fp8_e4m3` (with `fp8_e5m2` allowed as the second operand), `int8`, and `iu4`. There is no `fp4` form. The MXFP4 and NVFP4 weight formats both store 4-bit floats with a per-block scale, but the matmul itself has to land on an instruction that exists, and on this card the only instruction below 8-bit precision is the integer `iu4` path. NVFP4 on gfx1201 dequantizes the weight tensor to FP16 in a fused prelude shader and then runs `v_wmma_f32_16x16x16_f16`, which is the same kernel a plain FP16 matmul would use, capped at the same [191 TFLOPS dense matrix peak that AMD publishes for the R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html).

This post is the structural reason the FP4 wave breaks at RDNA4, what the WMMA throughput table actually looks like by precision, and why the right local-inference move on a 32 GB Radeon today is FP8 E4M3FN with a two-line patch into AITER's architecture table, not chasing an FP4 GGUF.

## What the two FP4 formats actually are

NVFP4 and MXFP4 share their per-element bit layout. Both store 4-bit floats with a sign bit, two exponent bits, and one mantissa bit, the IEEE-shaped E2M1 element. They diverge in how the per-block scale is carried.

MXFP4 follows the [OCP Microscaling Formats v1.0 specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) and uses a single 8-bit shared exponent, the E8M0 format, per block of 32 elements. One scale per 32 weights buys 4.25 bits per stored weight after metadata. The standard has signatures from AMD, Arm, Intel, Microsoft, Nvidia, Qualcomm, and Meta, which is the part that keeps it open, and AMD has shipped MI350X with native MXFP4 matrix-engine support.

NVFP4 is Nvidia's variant. The block size is 16 instead of 32 and the per-block scale is an FP8 E4M3 value rather than a power-of-two exponent, with a per-tensor FP32 scale layered on top. Smaller blocks plus a finer-grained scale buy back accuracy in cases where a single E8M0 exponent across 32 elements clips the dynamic range, and the second-level scale gives the calibrator another knob. The cost is that the matmul kernel has to consume two scales per dot product, which is exactly the kind of thing the Blackwell FP4 tensor core was designed to absorb without slowing down.

Both formats are real. Both are open. Neither is a llama.cpp-internal convention; both arrive with conversion scripts that target compressed-tensors checkpoints and produce GGUFs the existing infrastructure can load. NVFP4 in mainline llama.cpp landed across [PR 20644 (CUDA dp4a kernel)](https://github.com/ggml-org/llama.cpp/pull/20644), [PR 21074 (generic MMQ)](https://github.com/ggml-org/llama.cpp/pull/21074), [PR 21455 and PR 21539 (Vulkan)](https://github.com/ggml-org/llama.cpp/pull/21455), and [PR 22196 (Blackwell-native dispatch)](https://github.com/ggml-org/llama.cpp/pull/22196). The Vulkan kernels are the part that matters for RDNA4, and they exist, and they load NVFP4 GGUFs without crashing. The throughput is the question.

## What the gfx1201 WMMA table actually offers

The R9700's 128 AI accelerators expose a fixed instruction menu, and the menu does not include FP4. The AMD-published peak matrix throughputs by element type, dense and with structured sparsity, fall out of that menu directly:

| Element type | Dense | Structured sparsity |
| --- | ---: | ---: |
| FP16 | 191 TFLOPS | 383 TFLOPS |
| BF16 | 191 TFLOPS | 383 TFLOPS |
| FP8 (E4M3, E5M2) | 383 TFLOPS | 766 TFLOPS |
| INT8 | 383 TOPS | 766 TOPS |
| INT4 (iu4) | 766 TOPS | 1531 TOPS |
| FP4 | not present | not present |

The FP8 row is the one that carries the local-inference weight on this card. It is twice the throughput of FP16 because the hardware can pack two FP8 dot products into one FP16 lane, and it is the same E4M3FN format that MI350X uses, so the kernel work that vLLM and SGLang have done for the data-center side is reusable on gfx1201 with two trivial code changes. The INT4 row would be the highest-throughput line if anyone shipped a stack that targeted it, but the calibration story for unsigned 4-bit integer matmul is closer to QAT than to the post-training quantizers the GGUF ecosystem has standardized on, so it sits idle in practice.

The FP4 row is missing, and there is no near-term path to add it. AMD's CDNA-side roadmap puts FP4 matrix support on MI350X-class hardware, where the OCP microscaling formats are part of the matrix engine. RDNA4 is a separate ISA branch, and the silicon shipped without an FP4 element type in the WMMA decoder. A future RDNA generation can add it. The R9700 cannot.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-09-rdna4-wmma-precision-throughput.svg" alt="Eight horizontal bars stacked vertically, one per quantization or precision format, comparing dense WMMA matmul throughput on the AMD Radeon AI PRO R9700 (gfx1201, RDNA4). Each row shows the format name and bytes per stored weight on the left, a horizontal bar in the center scaled to the dense WMMA TFLOPS or TOPS, and a hardware-path annotation on the right. The native-WMMA rows for FP16, BF16, FP8 E4M3FN, INT8, and INT4 are filled green and reach 191, 191, 383, 383, and 766 respectively, with annotations naming the v_wmma instruction each lands on. The Q4_K_M, NVFP4, and MXFP4 rows are filled red with a hatched overlay and clipped to 191 TFLOPS effective, annotated with dequantize-to-FP16 paths and the absence of any v_wmma_f32_*_fp4 instruction on gfx1201. A vertical orange dashed reference line at 191 TFLOPS marks the FP16 matrix ceiling, with a label that ends every fallback bar at the same horizontal position. A footer notes that ROCm 7.2.1 silently dequantizes FP8 to FP32 on gfx1201 without an AITER patch and that bandwidth saved by FP4 is real on every card while compute saved is only real where the tensor cores speak FP4." loading="lazy" />
  <figcaption>The chart is the argument in shape. Bandwidth saved by FP4 is real on every card. Compute saved is only real where the tensor cores speak FP4, and on gfx1201 they do not.</figcaption>
</figure>

The reader should notice three things in the chart. First, the green bars span an order of magnitude on this card, from 191 TFLOPS at FP16 to 766 TOPS at INT4, which is a real spread that the right format can actually capture. Second, the FP8 bar sits at exactly the right place: twice the FP16 ceiling, native instruction, no dequant prelude, same format as MI350X. Third, the three red bars — Q4_K_M, NVFP4, MXFP4 — all clip at the FP16 line because the kernel has to dequantize before the matmul. The bytes-per-weight column on the left says they save bandwidth. The throughput column says they do not save compute on this hardware.

## Why FP8 is what RDNA4 actually accelerates

The FP8 path on gfx1201 has been working in vLLM since November, but only behind a patch that has not been upstreamed. The default execution path in vLLM for FP8 weights on RDNA4 routes through `torch_channelwise_w8a8_scaled_mm`, which calls `torch._scaled_mm` with `out_dtype=torch.float32`, which dequantizes both the weights and the activations to FP32 before the matmul. The 128 AI accelerators sit idle for the entire forward pass and the FP8 quantization buys a memory footprint reduction with no throughput uplift.

The fix is small. Two lines in `vllm/platforms/rocm.py` add `gfx1201` to the architecture set the FP8 Triton kernel path checks against, a runtime patch into AITER's `_ARCH_TO_DEVICE` dictionary maps `gfx1201` to `MI350X` so the existing MI350X Triton kernel finds a target, and a handful of tuned kernel-config JSON files cover the matrix sizes a Qwen3 layer actually issues. [The community working notes from `Rob-P-Smith` on the vLLM forums and the same author's follow-up issue](https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900) measured the resulting deltas on FP8-quantized Qwen3 models on the R9700: roughly 160 to 200 decode tok/s on Qwen3-0.6B and 52 to 85 on Qwen3-30B-2507, with prefill nearly doubling at prompt lengths up to 10,000 tokens before memory pressure tapered the gain. The author later partially attributed the uplift to CUDA-graph improvements that arrived in the same nightly rather than the architecture patch alone, but the underlying claim that the gfx1201 path needs explicit kernel configs to dispatch through native WMMA instead of falling back to a slower lane is reproduced in the kernel-config JSONs and confirmed independently by [ROCm TransformerEngine issue 520 on the silent FP32 fallback](https://github.com/ROCm/TransformerEngine/issues/520).

The mechanism is unsexy. RDNA4 uses the same FP8 E4M3FN element type as MI350X, not the FNUZ variant CDNA3 used. Triton's RDNA backend already lowers `tl.dot` on FP8 operands to `v_wmma_f32_16x16x16_fp8_fp8` when the architecture matches, so the obstruction is platform detection plus a tile-size table. Adding `gfx1201` to the right list and dropping the right config files in place flips on the kernel.

This has not been fully upstreamed because AITER's C and assembly kernels do not work on RDNA4 — they target CDNA layouts — so the integration pattern needs to disable AITER's native path while keeping its Triton-level dispatch. The shape of that change is captured in [open vLLM issue 28649 requesting upstream of the gfx1201 FP8 path](https://github.com/vllm-project/vllm/issues/28649). Until it lands cleanly in mainline, every local FP8 deployment on the R9700 is one configuration accident away from the FP32 fallback.

The same story plays out on the llama.cpp side in slower motion. The Vulkan FP8 path has been stable since the spring, but the K-quant fallback for unsupported element types still routes Q4_K_M through a dequant-to-FP16 prelude rather than into `v_wmma_i32_16x16x16_iu4`. That is what the chart's red Q4_K_M bar represents. Nothing in the GGUF format prevents an `iu4`-native fast path; the kernel just has not been written for the Vulkan backend on RDNA4 yet, and the demand from the FP4 wave has pulled engineering attention away from finishing it.

## What the right local quant target on RDNA4 looks like today

The Pareto frontier on this card is short. For 27B dense Qwen weights at 32 GB total VRAM with a long-context KV cache competing for the remaining memory, FP8 with the AITER patch wins on throughput at a 14 GB weight footprint. Q4_K_M wins on weight footprint at 11 GB but pays the dequant tax at runtime and clips to the FP16 ceiling. NVFP4 buys roughly 0.5 GB more headroom than Q4_K_M while paying the same dequant tax with a more complicated kernel prelude. MXFP4 trades the FP8 scale for an E8M0 exponent and lands in the same place. None of the FP4 paths cross the 191 TFLOPS ceiling on this card, and FP8 already sits well above it.

For MoE models like Qwen3.6-35B-A3B and Qwen3-Next-80B-A3B, the calculus shifts because the active weight footprint is much smaller than the resident footprint. There the bandwidth side of FP4 starts to matter more, since the bytes that move per token are a function of the resident expert pool plus the active path, and the resident pool dominates total VRAM. But the throughput math still says the active matmul lands on the FP16 ceiling, so the wall-time benefit of FP4 weights for the active path is nil on RDNA4, and the bandwidth win is captured equally well by Q4_K_M with the same kernel.

The honest local-inference plan for the next month on a 32 GB R9700 has three steps that have nothing to do with FP4. First, apply the gfx1201 AITER patch to vLLM (or wait for it to merge upstream) and re-quantize the dense Qwen3 targets to FP8 E4M3FN. The 383 TFLOPS line is the largest unclaimed throughput on this card by a comfortable margin, and the calibration story for FP8 is mature. Second, keep the KV cache at INT8 for long context, where the bandwidth-bound decode loop benefits from the smaller per-step read; the FP8 KV path inherits the same hardware acceleration as the weights. Third, watch the `iu4` Vulkan kernel work in llama.cpp, because that is the path that will eventually let Q4_K_M cross 191 TFLOPS on RDNA4 — and when it lands, the gap between Q4_K_M and any FP4 weight format on this card will widen, not close.

The general lesson is narrow and specific. New quantization formats do not arrive with new arithmetic units. They arrive with new arrangements of the bits a weight tensor stores, and whether the kernel that consumes those bits hits a fast path or a fallback path is a property of the hardware that has already shipped. RDNA4 shipped without FP4 in its WMMA decoder, and no GGUF release will change that. What it shipped with — FP8 E4M3FN matrix at 383 TFLOPS, INT4 at 766 TOPS, an AITER architecture table that is missing two lines — is what the local stack should be wringing out before the next quantization fashion lands.
