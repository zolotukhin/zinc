---
title: "int8 WMMA doubles RDNA4's matrix rate, and Q4_K's block scales take half of it back"
seoTitle: "RDNA4 int8 WMMA: MMQ Prefill and the Q4_K Descale Tax"
date: "2026-07-12"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - r9700
  - vulkan
  - prefill
  - gemm
  - wmma
  - int8
  - mmq
  - q4-k
  - quantization
  - local-llm
  - llm-inference
keywords:
  - RDNA4 int8 WMMA
  - MMQ quantized GEMM Vulkan
  - Q4_K block scales descale cost
  - wmma i32 16x16x16 iu8 gfx12
  - wide-K WMMA 128-bit loads
  - RX 9070 XT 389 TOPS INT8
  - Q8_1 activation quantization
  - RDNA3 int8 same rate as fp16
  - llama.cpp WMMA-MMQ RDNA4
  - local LLM prefill AMD
excerpt: "RDNA4's matrix core runs int8 at 2048 ops per clock per CU against 1024 FLOP for fp16, so ZINC's fused prefill GEMM should get a free 2x by keeping Q4_K weights in integers all the way into the matmul. It gets 1.2x. The missing half is the descale: a block-scaled quant format forces an int32-to-float drain every 32 weights, and on RDNA4 that drain issues down the same pipe as the WMMA it is supposed to be feeding."
seoDescription: "Why converting ZINC's RDNA4 prefill GEMM to int8 WMMA delivers 1.2x instead of the 2x the matrix rate promises: Q4_K's per-32-weight scales force an int32 accumulator drain onto the vector pipe that shares an issue port with WMMA. Folding the 6-bit sub-scales into an integer multiply-add and fusing two 16x16x16 iu8 instructions into a K=32 op recovers most of it. Qwen3.5-9B prefill on the RX 9070 XT, 665 to 815 tok/s."
faqs:
  - question: "Is int8 WMMA actually faster than fp16 WMMA on RDNA4?"
    answer: "On the matrix core, yes, by exactly 2x. AMD's own numbers put the RDNA4 CU at 1024 FP16 FLOP per clock and 2048 INT8 ops per clock, which is why the RX 9070 XT is specified at 195 TFLOPS FP16 matrix and 389 TOPS INT8. But a quantized GEMM is not only a matmul. Q4_K carries a scale every 32 weights, so the int32 accumulator has to be drained and converted to float 32 times more often than a plain int8 GEMM would need, and that drain runs on the vector pipe. Measured on ZINC's fused Qwen3.5-9B prefill GEMM, the naive int8 port returned 1.2x, not 2x."
  - question: "Why did the same change lose on RDNA3?"
    answer: "Because RDNA3's matrix core runs INT8 at 512 ops per clock per CU, the same rate as FP16. There is no arithmetic prize for keeping the weights in integers, so the extra descale work is pure loss. That is the whole reason quantized matmul on RDNA3 went through the DP4a vector path rather than WMMA, and it is why the RDNA4 doubling of the int8 rate, not the fp8 support everybody talks about, is the interesting number on the spec sheet for local inference."
  - question: "What is the integer sub-scale trick?"
    answer: "A Q4_K super-block holds 256 weights as eight groups of 32, and each group's scale is a 6-bit integer that gets multiplied by a single fp16 value shared across the super-block. Instead of converting each group's int32 dot product to float and scaling it there, ZINC multiplies the int32 sum by the 6-bit integer scale and accumulates in int32 across all eight groups. The worst case is about 31 million, which fits in int32 with room left. One float convert and one fused multiply-add per 256 weights instead of per 32, which cuts the descale from 32 cycles per tile per K block to about 20."
  - question: "How much prefill did the int8 GEMM buy end to end?"
    answer: "Qwen3.5-9B on the RX 9070 XT went from about 665 tok/s to about 815 tok/s, a 1.23x end-to-end gain from a GEMM that got roughly 1.6x. The gap to llama.cpp's 973 tok/s pp512 is now about 1.19x. Decode did not move and was not expected to: at batch one the GEMM is a matrix-vector product that never approaches the matrix core's rate, and the bottleneck is weight bandwidth."
draft: false
---

The RX 9070 XT is specified at [195 TFLOPS of FP16 matrix throughput and 389 TOPS of INT8](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html). ZINC's prefill GEMM reads Q4_K weights, unpacks them to fp16 in registers, and multiplies on the fp16 half of that number. The weights arrive as integers and leave as integers. Nothing in the model requires them to become floats in between, and the card will multiply integers twice as fast.

So the port looked like free money. Take the fused dequant GEMM from [four days ago](https://zolotukhin.ai/blog/2026-07-08-the-dequant-scratch-round-trip-is-zincs-last-rdna4-prefill-tax/), unpack Q4_K into int8 instead of fp16, quantize the activations to Q8_1, swap `wmma_f32_16x16x16_f16` for `wmma_i32_16x16x16_iu8`, and collect the doubling.

It returned 1.2x. Prefill on Qwen3.5-9B went from 665 tok/s to about 700, which is nowhere near the 2x the spec sheet promises. The missing half is not in the matmul. It is in the sixteen bytes of scale metadata that a Q4_K super-block carries around, and in a fact from [yesterday's post](https://zolotukhin.ai/blog/2026-07-11-softmax-steals-rdna4-wmma-issue-slots-flash-attention/) that keeps turning out to be the whole story on this architecture: on RDNA4, WMMA and ordinary vector instructions issue down the same pipe.

## The int8 rate is the number worth caring about on RDNA4

AMD's matrix core guide publishes the rates per compute unit per clock, and the interesting row is not fp16.

| FLOP or OP per clock per CU | RX 6950 XT (RDNA2) | RX 7900 XTX (RDNA3) | RX 9070 XT (RDNA4) |
| --- | ---: | ---: | ---: |
| FP16 | 256 | 512 | 1024 |
| BF16 | n/a | 512 | 1024 |
| INT8 | 512 | 512 | **2048** |

Those numbers come straight from [AMD's RDNA4 matrix core article](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/). RDNA3 ran int8 at the same 512 ops per clock as fp16, which means a quantized matmul on a 7900 XTX has no arithmetic prize to win. Keeping the weights in integers there buys you nothing on the matrix core and costs you the bookkeeping, which is exactly why quantized matmul on RDNA3 lived on the DP4a vector path instead. RDNA4 is the first AMD consumer part where the integer GEMM has a rate advantage worth chasing, and it is a 4x jump in the int8 line while fp16 only doubled.

That is the setup. The rest of this post is about why a 2x hardware rate turned into a 1.2x kernel, and what it took to get most of the rest.

## Q4_K makes you stop and change money every 32 weights

A [Q4_K](https://github.com/ggerganov/llama.cpp/pull/1684) super-block is 256 weights. Each group of 32 has its own 6-bit scale and 6-bit minimum, and those integers are themselves scaled by two fp16 values shared across the whole super-block. The dequantized weight is `d * s_j * q - dmin * m_j`, where `q` is the 4-bit payload, `s_j` and `m_j` are the group's integers, and `d` and `dmin` are the super-block floats.

An int8 matmul can multiply `q` directly. What it cannot do is carry `s_j` inside the accumulator, because the scale changes every 32 elements along K. So the natural kernel accumulates an int32 dot product over 32 weights, converts it to float, multiplies by `d * s_j` and by the activation block's scale, and adds it into an fp32 accumulator. Then it starts a fresh int32 sum for the next group.

That drain is one `v_cvt_f32_i32` and one `v_fma_f32` per output element per 32-element K block. It sounds tiny. Put it next to the matmul it is attached to and it is not.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-12-rdna4-int8-wmma-descale-timeline.svg" alt="A four-lane horizontal timeline showing how one 32-element K block of a 16 by 64 output tile spends cycles on one RDNA4 compute unit. Each lane is a stacked bar of blue weight-unpack work, amber WMMA matmul work, and pink block-descale work, and the bar length is time because matrix and vector instructions share an issue pipe. The first lane, today's fp16 WMMA path, is 32 cycles of unpack plus 64 cycles of matmul, 96 cycles total, marked one times. The second lane, int8 WMMA with an fp32 descale every 32 weights, is 16 cycles of unpack, 32 of matmul and 32 of descale, 80 cycles total, 1.20 times, with a note that the matmul halved and the descale ate the win. The third lane folds the 6-bit Q4_K scales into an int32 multiply-add and converts to float once per 256-weight super-block: 16 plus 32 plus 20, 68 cycles, 1.41 times. The fourth lane adds wide-K WMMA and packed unpacking: 8 plus 32 plus 20, 60 cycles, 1.61 times. An inset table gives AMD's matrix rates per clock per compute unit: FP16 512 on RDNA3 versus 1024 on RDNA4, INT8 512 on RDNA3 versus 2048 on RDNA4, with a note that on RDNA3 an int8 matmul buys nothing because the matrix core runs int8 at the same rate as fp16." loading="lazy" />
  <figcaption>Cycle budget for one 16x64 output tile against one 32-element K block, on one CU. The int8 matmul is half the length of the fp16 one, and the descale it drags along is exactly as long as the matmul it sits next to. First-order model at RX 9070 XT rates: 1024 fp16 FLOP, 2048 int8 ops, and 64 vector lane-ops per clock per CU.</figcaption>
</figure>

Read the second lane and the disappointment explains itself. Halving the matmul saved 32 cycles. The descale added 32 back. The only reason the kernel came out ahead at all is that unpacking Q4_K to int8 is cheaper than unpacking it to fp16: you skip the convert and the scale-and-subtract, which is two vector ops per weight element you no longer issue.

The ratio is fixed and unpleasant. Per output element, the int8 matmul over a 32-element K block is 64 ops at 2048 per clock, so 0.031 cycles. The descale is 2 vector ops at 64 lane-ops per clock, so 0.031 cycles. They are exactly equal. The vector pipe is 32 times slower than the matrix pipe on this chip, and a block-scaled format hands it one instruction for every 32 the matrix core runs.

## Paying the scale in integers instead of floats

The fix is to notice that `s_j` is an integer. It is six bits. The int32 dot product of a 32-element group is at most `32 * 15 * 127`, about 61,000. Multiply that by a scale of up to 63 and you get 3.8 million. Sum eight of those across the super-block and the worst case is about 31 million, comfortably inside int32.

So the accumulator does not have to leave the integer domain at the group boundary. Multiply the group's int32 sum by `s_j` with an integer multiply-add, keep accumulating in int32 across all eight groups of the super-block, and convert to float exactly once per 256 weights, applying `d` and the activation scale there. The drain drops from two vector ops per 32 weights to one, plus an eighth of the old pair.

The `dmin * m_j` correction is the other half of the format and it does not need a per-element pass either. That term is the block minimum times the sum of the activations in the block, and Q8_1 already carries that sum: it stores a scale and a block sum per 32 activations, which is why [llama.cpp's Vulkan MMQ shader](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp) binds its B operand as `block_q8_1_x4_packed128` rather than raw int8. The correction collapses into a small dot product between the row's eight `m_j` values and the column's eight block sums, computed once per super-block per output element rather than once per group.

Two more instruction-count trims come straight from AMD's own guide. RDNA4's WMMA fragment layout gives each lane 8 elements of A and B, which saturates the 128-bit load path at fp16 but only fills 64 bits at int8. AMD's fix is to [fuse two `wmma_i32_16x16x16_iu8` instructions into an effective K=32 operation](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/) so the loads go back to 128 bits wide, and they show it is bit-identical to the unfused version. llama.cpp took the same technique into its [WMMA-MMQ kernels for RDNA4](https://github.com/ggml-org/llama.cpp/commit/0543f928a3ae576e6e16d3bbf02c0bf9fddba688). The unpack itself then goes packed, two 4-bit nibbles per instruction, which halves the blue segment in the diagram.

| Per output element, one 32-weight K block | matmul | unpack | descale | total | vs fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 WMMA (before) | 0.063 | 0.031 | — | 0.094 | 1.00x |
| int8 WMMA, fp32 descale per group | 0.031 | 0.016 | 0.031 | 0.078 | 1.20x |
| int8 WMMA, integer sub-scale accumulation | 0.031 | 0.016 | 0.020 | 0.066 | 1.42x |
| + wide-K WMMA, packed unpack | 0.031 | 0.008 | 0.020 | 0.058 | 1.61x |

Cycles per output element, 16x64 tile, one CU. The model assumes the tile is resident and ignores DRAM, which the [fused dequant work](https://zolotukhin.ai/blog/2026-07-08-the-dequant-scratch-round-trip-is-zincs-last-rdna4-prefill-tax/) already took off the critical path. What it tells you is that the matmul column shrinks by half and then stops mattering: after the port, less than a third of the kernel's cycles are spent in the matrix core, and every further gain has to come from deleting vector instructions.

## What it measured

On Qwen3.5-9B, Q4_K_M weights, RX 9070 XT, Mesa 25.2.8:

| Qwen3.5-9B prefill, RX 9070 XT | prefill | attention share | gap to llama.cpp |
| --- | ---: | ---: | ---: |
| fp16 WMMA GEMM (yesterday) | ~665 tok/s | ~35% | 1.46x |
| int8 WMMA, fp32 descale per group | ~700 tok/s | ~37% | 1.39x |
| + integer sub-scale accumulation | ~775 tok/s | ~41% | 1.26x |
| + wide-K WMMA, packed unpack | ~815 tok/s | ~43% | 1.19x |
| llama.cpp `pp512` reference | 973 tok/s | — | — |

The quantized projections are roughly 48 percent of prefill wall clock, so a 1.61x GEMM turns into a 1.23x post. That is the arithmetic of Amdahl and there is nothing to complain about in it. What it does is push attention up to 43 percent of the profile, because the flash attention kernel gets none of this. Its operands are activations and a KV cache, not block-scaled weights, so there is no integer format to exploit and no descale to optimize away. Attention stays on fp16 WMMA at 1024 FLOP per clock while every other matmul in the model now runs at 2048 ops per clock, and the profile is going to keep tilting that way.

Accuracy moved less than the noise in the measurement. Wikitext-2 perplexity on Qwen3.5-9B Q4_K_M went from 8.41 on the fp16-activation path to 8.43 with Q8_1 activations, which is a 0.24 percent regression and consistent with what the Q8_1 path has always cost in llama.cpp. The int32 accumulation itself is exact. The only new error in the pipeline is the round-to-int8 on the activations, and post-RMSNorm activations at block size 32 quantize cleanly.

Decode did not move, at 39.6 tok/s, and this is worth stating because it is the part people get wrong about MMQ. At batch one the GEMM is a matrix-vector product with an arithmetic intensity of about two FLOP per weight byte. It never gets within an order of magnitude of the matrix core's rate, it is limited by how fast the weights arrive from VRAM, and doubling a compute rate that was never the constraint changes nothing.

## The shape of the lesson

The spec sheet says 389 TOPS. The kernel got 1.6x on a path where the arithmetic said 2x, and the difference is entirely accounted for by twenty-odd vector instructions per tile that exist only because the weight format changes its scale every 32 elements.

That is the same finding as the [softmax post](https://zolotukhin.ai/blog/2026-07-11-softmax-steals-rdna4-wmma-issue-slots-flash-attention/) and the same finding as the [VGPR post](https://zolotukhin.ai/blog/2026-07-09-vgpr-pressure-caps-fused-rdna4-prefill-gemm-at-nine-waves/), wearing a third costume. RDNA4 gives you an enormous matrix rate through a single issue port shared with a comparatively tiny vector unit. Every non-matmul instruction in an inner loop is not overhead in the usual sense of a few percent. It is a direct subtraction from the matrix throughput, at a 32-to-1 exchange rate, and the quantization format you chose for disk-space reasons in 2023 is now writing instructions into that loop on your behalf.

The practical conclusion for anyone porting a quantized inference kernel to RDNA4 is narrow and worth stating plainly. Switch the matmul to int8, because the rate is really there. Then go find every float that touches the accumulator and ask whether it could have been an integer, because that is where the other half of the 2x is hiding.

The gap to llama.cpp is 1.19x. Attention is 43 percent of prefill and running at half the matrix rate of everything around it, which makes the next question obvious: what would it take to put the QK matmul on int8 as well.
