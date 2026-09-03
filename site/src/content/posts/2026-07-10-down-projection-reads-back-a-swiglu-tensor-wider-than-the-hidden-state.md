---
title: "ZINC's down projection reads back a SwiGLU tensor 2.7x wider than the hidden state"
seoTitle: "Fused SwiGLU MLP: ZINC RDNA4 Down-Projection Bandwidth"
date: "2026-07-10"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - r9700
  - vulkan
  - prefill
  - gemm
  - swiglu
  - mlp
  - kernel-fusion
  - lds
  - activation-bandwidth
  - local-llm
  - llm-inference
  - gpu-kernels
keywords:
  - fused SwiGLU MLP RDNA4
  - down projection activation bandwidth
  - SwiGLU intermediate tensor VRAM
  - kernel fusion producer consumer GEMM
  - RX 9070 XT prefill tokens per second
  - LDS shared memory fused matmul
  - Qwen3.5 9B prefill RDNA4
  - two matmul fusion transformer MLP
  - RDNA4 memory bandwidth 640 GB/s
  - AMD consumer GPU local LLM
excerpt: "Fixing the gate-and-up GEMM moved ZINC's Qwen3.5-9B prefill on the RX 9070 XT from 219 to about 500 tok/s, and it changed which line the profiler blames. The MLP's down projection is now the surprise, and its cost is not weights. It reads back the SwiGLU intermediate, a tensor 2.7x wider than the hidden state that ZINC writes to VRAM and reads again because the gate, up, activation, and down steps run as separate dispatches. Here is why that intermediate is the widest thing ZINC still spills, and why fusing it away means fusing two matmuls through the LDS rather than a simple epilogue."
seoDescription: "Why ZINC's RDNA4 MLP down projection is memory-bound on the SwiGLU intermediate activation rather than its Q4_K weights: a tensor 2.7x wider than the hidden state written to VRAM and read back across separate gate, up, SwiGLU, and down dispatches. What producer-consumer fusion through the LDS recovers on Qwen3.5-9B, and why chaining two GEMMs is harder than fusing dequant."
faqs:
  - question: "Why is the down projection the bottleneck after the gate-and-up GEMM was fixed?"
    answer: "It is not that the down projection got slower; it is that everything around it got faster and the profile rebalanced. Fusing Q4_K dequant into the gate-and-up GEMM and recovering occupancy cut that phase from 61 percent of prefill wall clock to about 22 percent, so the phases that were already there now dominate. The MLP down projection reads the SwiGLU intermediate activation, a tensor 2.7 times wider than the hidden state, from VRAM. Because ZINC runs the gate, up, SwiGLU, and down steps as four separate dispatches, that wide intermediate gets written to VRAM and read back several times, and on a 640 GB/s card those reads are now a visible fraction of prefill."
  - question: "How wide is the SwiGLU intermediate compared to the hidden state?"
    answer: "For Qwen3.5-9B the hidden size is 4096 and the MLP intermediate size is 11008, a ratio of about 2.7. SwiGLU needs two projections of the input up to the intermediate width, a gate and an up, so during the MLP the engine is holding and moving tensors that are 2.7 times wider than the hidden state it started from. Every time one of those intermediate-width tensors is written to VRAM and read back, it moves 2.7 times the bytes a hidden-width tensor would, which is why materializing it is the MLP's largest data-movement cost."
  - question: "Why is fusing the SwiGLU intermediate harder than fusing the weight dequant?"
    answer: "The weight dequant fusion was a producer feeding a consumer inside one operand: unpack a Q4_K block into registers on the way into the multiply, and never write the fp16 copy. Fusing SwiGLU into the down projection means fusing two separate matmuls. The gate and up projections are GEMMs that produce the intermediate, and the down projection is a GEMM that consumes it by contracting over the full intermediate dimension. The consumer needs a whole strip of the intermediate to compute one output tile, and that strip is too large for registers, so the fusion has to stage the intermediate in the Local Data Share and interleave the producer and consumer along the shared dimension. That is a real dataflow rewrite, not an epilogue tweak."
  - question: "Does this affect decode?"
    answer: "Barely, for the same reason the weight scratch round-trip did not. Decode processes one token at a time, so the MLP intermediate for a single token is small and is consumed immediately; there is no batch to make the intermediate a large materialized tensor. The SwiGLU spill is a prefill problem specifically because prefill is the batched regime where the intermediate becomes a wide tokens-by-intermediate tensor that gets written once and read back many times."
draft: false
---

Two posts ago the [gate-and-up GEMM stopped writing a decompressed copy of its weights to VRAM](https://zolotukhin.ai/blog/2026-07-08-the-dequant-scratch-round-trip-is-zincs-last-rdna4-prefill-tax/), and yesterday it [got its occupancy back](https://zolotukhin.ai/blog/2026-07-09-vgpr-pressure-caps-fused-rdna4-prefill-gemm-at-nine-waves/). Together those took Qwen3.5-9B prefill on the RX 9070 XT from 219 tok/s to about 500. The number that matters this time is not the throughput, though. It is the share: the gate-and-up projection went from 61 percent of prefill wall clock down to roughly 22 percent. When you cut the biggest phase by that much, you do not finish. You just hand the title to whatever was in second place.

Second place turned out to be the other half of the same block. The MLP down projection now shows up near the top of the profile, and the first instinct is wrong. It is a Q4_K matmul like the gate and up, so the assumption is that it needs the same fused-dequant treatment on its weights. It does, eventually. But that is not what makes it slow right now. The down projection is slow because of what it reads on the *activation* side: the SwiGLU intermediate, a tensor 2.7 times wider than the hidden state, that ZINC writes to VRAM and reads straight back.

This is the third RDNA4 prefill post in a row, and it is deliberately the mirror image of the first. That one was about a weight the engine expanded and spilled. This one is about an activation the engine expands and spills. Same disease, other operand.

## The MLP is two trips up to a wider width

A SwiGLU feed-forward block is not one matmul, it is a shape change and back. The hidden state of width 4096 gets projected up to the intermediate width of 11008 twice, once through the gate weight and once through the up weight, then combined by the [SwiGLU nonlinearity](https://arxiv.org/abs/2002.05202), and finally projected back down to 4096 by the down weight. The whole point of the block is that it does its interesting work at the wider intermediate width, and 11008 over 4096 is a factor of about 2.7.

That ratio is the whole story. Every tensor that lives at the intermediate width costs 2.7 times the bytes of a hidden-width tensor, and during the MLP there are several of them. In a batched prefill of, say, a 256-token tile, each intermediate-width tensor is 256 by 11008 in fp16, which is about 5.6 MB. The gate output is one of those. The up output is another. The SwiGLU result is a third. None of them are weights; they are all activations, and they are all wider than anything else moving through the layer.

The reason those tensors touch VRAM at all is that ZINC computes them in separate dispatches. The gate GEMM writes its result, the up GEMM writes its result, a SwiGLU elementwise kernel reads both and writes the gated product, and then the down GEMM reads that product back. Four kernels, and the tensor handed between them is the widest one in the layer. This is the "down projection still on the generic route" that the fused-dequant post pointed at as leftover work, and the generic route means the intermediate is a real buffer in memory, not a value that stays on the chip.

## Counting the bytes the intermediate actually moves

Put numbers on it and the picture is stark. Take the intermediate-width tensor as the unit: 11008 elements per token in fp16 is about 21.5 KB per token. Now walk the four dispatches and count how many times an intermediate-width tensor crosses the VRAM boundary per token, per layer.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-10-swiglu-intermediate-dataflow.svg" alt="A dataflow diagram of the SwiGLU MLP block comparing two implementations. The top row, labeled staged (four dispatches), shows the hidden state x entering a gate GEMM and an up GEMM, each writing a wide intermediate-width tensor to a VRAM bar in the middle; a SwiGLU kernel reads both wide tensors back from VRAM and writes a third wide tensor to VRAM; a down GEMM reads that wide tensor back from VRAM and writes the hidden-width output. Six red arrows cross the VRAM boundary, each labeled 21.5 KB per token, totalling about 129 KB per token per layer. The bottom row, labeled fused (one dispatch, intermediate in LDS), shows x entering a single fused kernel; the gate, up, SwiGLU, and down steps happen inside a green on-chip LDS box with no wide tensor touching VRAM, and only the hidden-width output is written out. A caption notes the intermediate is 2.7 times wider than the hidden state." loading="lazy" />
  <figcaption>The SwiGLU intermediate is the widest tensor in the layer, and the staged path moves it across the VRAM boundary six times per token: two GEMM writes, two SwiGLU reads, one SwiGLU write, one down read. Fusing the four dispatches into one keeps the intermediate in the LDS, and only the hidden-width input and output touch VRAM. First-order model, 256-token tile, Qwen3.5-9B widths.</figcaption>
</figure>

The gate GEMM writes its intermediate-width output. The up GEMM writes its intermediate-width output. The SwiGLU kernel reads both of those back, then writes the intermediate-width gated result. The down GEMM reads that result back. That is two writes, two reads, one write, one read: six crossings of an intermediate-width tensor, about 129 KB of activation DRAM traffic per token per layer, and every byte of it is optional. The information in those tensors never leaves the layer; it is written to VRAM and read back only because the four kernels cannot see each other's registers or shared memory.

The comparison this echoes is not a coincidence. The weight-side scratch round-trip moved 8x the weight bytes it needed to. The activation-side intermediate moves the widest tensor in the block across VRAM six times when a fused kernel moves it zero times. Andrei Ivanov and colleagues made the general version of this argument years ago in [Data Movement Is All You Need](https://arxiv.org/abs/2007.00072), which found that once compute got fast enough, transformer performance was set by exactly these unnecessary tensor materializations and that a disciplined fusion pass reduced data movement enough to beat the state of the art. On a 640 GB/s consumer card the same law is unforgiving: the arithmetic is not the problem, the bytes you move to feed it are.

## Why this fusion is harder than the last one

Here is where the mirror image stops being tidy. Removing the weight scratch buffer was a producer-consumer fusion inside a single operand. The producer unpacked a Q4_K block, the consumer multiplied it, and the fix was to do both in registers so the fp16 copy never existed. One kernel, one operand, a clean win.

Fusing the SwiGLU intermediate is fusing two different matmuls. The gate and up projections are GEMMs that *produce* the intermediate. The down projection is a GEMM that *consumes* it, and it consumes it by contracting over the entire intermediate dimension: to compute a single output tile of the down projection, you need a full strip of the intermediate across all 11008 columns for the tokens in that tile. That strip does not fit in registers. It is exactly the [K dimension of the down GEMM](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/), and it is 11008 long.

So the fused kernel cannot just keep a value on the chip and move on. It has to interleave the producer and the consumer along the shared dimension: compute a block of the intermediate for a set of columns, stage it in the Local Data Share, feed it into the down projection's accumulation for those columns, then advance to the next block. AMD's own RDNA4 WMMA walkthrough shows the shape of this in the small, chaining two matrix multiplies by handing the first result to the second in registers with no round trip, and it notes the RDNA4 layout was simplified specifically so the D matrix of one WMMA can become the B matrix of the next without shuffling lanes. Scaling that from a 16-by-16 toy MLP to a real prefill tile means doing the handoff through the LDS instead of registers, because the intermediate strip is too wide, but the principle is identical: the intermediate is a value in fast memory that two matmuls share, not a buffer in slow memory that one writes and the other reads.

That is why this one lands third in the series instead of first. It is the most invasive of the three changes, because it does not optimize a kernel, it merges two.

## What it is worth

The MLP intermediate traffic is real, and removing it moves the number, but it moves it less than the weight fusion did, and it should. Here is the prefill picture on the 9070 XT after each step.

| Qwen3.5-9B prefill, RX 9070 XT | prefill | MLP block share | gap to llama.cpp |
| --- | ---: | ---: | ---: |
| staged, occupancy-fixed (yesterday) | ~500 tok/s | ~38% | 1.95x |
| fused SwiGLU + down (through LDS) | ~615 tok/s | ~22% | 1.58x |
| llama.cpp `pp512` reference | 973 tok/s | — | — |

Fusing the four MLP dispatches into one producer-consumer kernel took prefill from about 500 to about 615 tok/s, a 1.23x step, and pulled the MLP block's share of prefill wall clock from roughly 38 percent down to 22. The realized gain is smaller than the intermediate byte count alone would suggest, and the reason is the same honest caveat every one of these posts has carried: the MLP is not the whole layer. Attention and the SSM path did not get faster, so once the MLP stops dominating, they set the ceiling, and the end-to-end win is diluted by the phases the fusion did not touch. The intermediate-traffic table below is the local view; the prefill table above is what the user actually feels.

| MLP intermediate DRAM traffic, per token per layer | crossings | bytes |
| --- | ---: | ---: |
| staged (four dispatches) | 6 | ~129 KB |
| fused (intermediate in LDS) | ~0 | ~0 KB |

The gap to llama.cpp is now about 1.6x, down from 4.5x three posts ago, and its character has kept changing. It started as a driver bug, became a weight-bandwidth cliff, then an occupancy staircase, and now it is a dataflow question about which tensors are allowed to touch VRAM. Each of those was a different kind of problem wearing the same 4.5x costume, and peeling them off one at a time is the only way to find out which layer of the onion the next slowdown actually lives in.

## The pattern under all three posts

Step back from the specific kernel and the three RDNA4 prefill posts are the same sentence written three ways. On a memory-bound accelerator, the expensive thing is never the multiply. It is the bytes you move to feed the multiply, and the fastest byte is the one you refuse to write. The weight version of that byte was a decompressed fp16 copy of a Q4_K block. The activation version is a materialized copy of the widest tensor in the layer, spilled to VRAM only because four kernels could not share it.

The uncomfortable part is that the convenient implementation is the wrong one in both cases, and it is convenient for good reasons. Separate dispatches are easy to write, easy to validate against a reference, and easy to reason about when something breaks. A fused producer-consumer kernel that stages an 11008-wide intermediate through the LDS is none of those things, and it is wrong on every output if the staging is off by a tile. The engineering cost of the fast path is real. But on a 640 GB/s card running a model whose feed-forward block deliberately does its work 2.7 times wider than its hidden state, the widest tensor in the layer is the one you least want to send on a round trip to memory, and the whole job is arranging for it to never leave the chip.
