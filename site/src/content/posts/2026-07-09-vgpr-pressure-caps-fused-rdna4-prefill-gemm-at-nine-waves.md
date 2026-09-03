---
title: "VGPR pressure caps the fused RDNA4 prefill GEMM at nine waves"
seoTitle: "RDNA4 VGPR Occupancy: Fused Prefill GEMM Limit"
date: "2026-07-09"
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
  - occupancy
  - vgpr
  - register-pressure
  - dynamic-vgpr
  - local-llm
  - llm-inference
  - gpu-kernels
keywords:
  - RDNA4 VGPR occupancy
  - fused dequant WMMA GEMM register pressure
  - wave32 occupancy 192 KB register file
  - RX 9070 XT prefill tokens per second
  - RDNA4 dynamic VGPR allocation
  - s_alloc_vgpr Vulkan compute
  - occupancy latency hiding LLM prefill
  - Qwen3.5 9B prefill RDNA4
  - 96 VGPR full occupancy RDNA4
  - AMD consumer GPU local LLM
excerpt: "Yesterday's fix fused Q4_K dequant into ZINC's RDNA4 prefill GEMM and cut weight bandwidth 8x, but prefill rose 1.95x instead of the roofline's 2.7x. The missing throughput is occupancy: unpacking weights and holding WMMA accumulators in registers pushes the kernel to about 168 VGPRs per wave, which on RDNA4's 192 KB register file leaves only nine of sixteen wave slots filled. Here is why nine waves cannot hide the memory latency, and why RDNA4's new dynamic-VGPR hardware is the fix the Vulkan toolchain has not shipped."
seoDescription: "Why ZINC's fused Q4_K dequant WMMA prefill GEMM on RDNA4 is capped by register pressure rather than bandwidth: 168 VGPRs per wave drops occupancy to 9 of 16 wave slots on the RX 9070 XT's 1536-VGPR SIMD, starving latency hiding. What moving Q4_K scales to LDS recovers, and where RDNA4 dynamic VGPR allocation could take it."
faqs:
  - question: "Why did the fused dequant GEMM only gain 1.95x when the roofline predicted 2.7x?"
    answer: "The roofline predicts the compute-bound ceiling if the kernel can keep the matrix units fed. Fusing Q4_K dequant into the GEMM removed the bandwidth tax that capped the staged kernel, but the fused kernel has to unpack four-bit weights and hold WMMA A/B/C fragments in registers, pushing VGPR usage to roughly 168 per wave. On RDNA4's 192 KB register file (1536 wave32 VGPRs per SIMD, 16 wave slots), 168 VGPRs allocates seven 24-register blocks and leaves room for only nine active waves out of sixteen. Nine waves is not enough independent work to hide HBM and WMMA latency, so realized throughput lands between the memory-bound floor and the compute roof, near 1.95x rather than 2.7x."
  - question: "How many VGPRs can a wave32 kernel use before RDNA4 occupancy drops below maximum?"
    answer: "96. RDNA4 desktop GPUs have a 192 KB register file per SIMD, which holds 1536 wave32 VGPRs. Sixteen wave slots divided into 1536 registers is 96 registers per wave at full occupancy, so any kernel that stays at or under 96 VGPRs runs all sixteen waves. Above 96 the allocator hands out registers in 24-register blocks on these parts, so 97 VGPRs already rounds to a five-block, 120-register allocation and 12 waves, and 168 VGPRs is a seven-block allocation that fits nine waves."
  - question: "Can ZINC just use fewer registers to keep full occupancy?"
    answer: "Only partway. The WMMA accumulator tile has an irreducible register cost because the C and D matrices live in VGPRs across the whole K loop, and a prefill tile accumulates several 16x16 fragments at once. The recoverable part is everything that does not need to be per-lane: Q4_K block scales and mins are wave-invariant and belong in LDS or scalar registers, not VGPRs, and accumulator live ranges can be shortened by tiling the K loop tighter. Those changes take the kernel from about 168 VGPRs to about 120, which restores 12 of 16 waves. Reaching the full 16 waves at 96 VGPRs would mean shrinking the accumulator tile, which trades occupancy back for arithmetic intensity."
  - question: "Does RDNA4 dynamic VGPR allocation solve this today?"
    answer: "Not yet, not for this path. RDNA4 introduced a dynamic-VGPR mode where a wave starts with a minimal allocation and requests more with an s_alloc_vgpr instruction, so a kernel could run 16 low-register waves and spike registers only inside the dequant-and-multiply inner loop. But the mode is restricted to wave32 compute shaders, it is driver-gated through a chip-wide control register, and so far AMD has only been observed using it for indirect-mode raytracing, not general compute. It is not exposed through the Vulkan compute path ZINC runs on, so for now the occupancy has to be won the old way, by spending fewer registers."
draft: false
---

Yesterday's post ended on a number that did not add up. Fusing the Q4_K dequant step into ZINC's RDNA4 prefill GEMM cut weight DRAM traffic by about eight times, and a [first-order roofline](https://zolotukhin.ai/blog/2026-07-08-the-dequant-scratch-round-trip-is-zincs-last-rdna4-prefill-tax/) said that should be worth a 2.7x ceiling on the gate-and-up matmul. The measured prefill gain on Qwen3.5-9B on the RX 9070 XT was 1.95x, from 219 to roughly 430 tok/s. That is a good result, but it is not 2.7x, and the difference is not rounding. Something is stopping the fused kernel from reaching the compute roof it earned the right to sit on.

The something is occupancy. A kernel that is no longer waiting on memory bandwidth can still be waiting on memory *latency*, and the way a GPU hides latency is by having many independent waves in flight so it always has other work to run while one wave stalls on a load. Fusing dequant into the GEMM did not come for free in register terms. The kernel now has to unpack four-bit weights into registers and hold the [WMMA](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) accumulator tile there across the whole inner loop, and that pushed its register footprint high enough to starve the very latency hiding it needs.

This post is about the second tax, the one that shows up right after you fix the first. It is a smaller, older story than a bandwidth cliff, and it is the reason a bandwidth-optimal kernel can still leave a third of the card's matrix throughput on the floor.

## Why occupancy is the thing that pays for latency

Start with the mechanism, because the numbers only make sense once the mechanism is clear. A RDNA4 SIMD runs one wave of 32 lanes at a time, but it can hold many waves resident and switch between them cycle to cycle. When the active wave issues a load from VRAM and has to wait a few hundred cycles for the data, the SIMD does not stall. It runs a different resident wave. The more resident waves there are, the more independent work is available to fill those gaps, and the closer the SIMD gets to being busy every cycle. This is [occupancy](https://gpuopen.com/learn/occupancy-explained/), and it is the single biggest lever on whether a compute-bound kernel actually reaches its compute roof.

The catch is that resident waves are not free. Every wave needs its own registers, and they all come out of one fixed register file per SIMD. If each wave asks for more registers, fewer waves fit, and occupancy falls. This is not new to RDNA4. The classic version of the problem was written up years ago by Sebastian Aaltonen, who found that on older AMD hardware [dropping a shader from 40 to 32 registers per thread doubled the number of resident thread groups and delivered a 50 percent speedup](https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/) with no other change. Same silicon, same arithmetic, just more waves to hide latency behind. The register file is the budget, and occupancy is what you buy with it.

RDNA4 gives us exact numbers to work with. The [RX 9070 XT and R9700](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture) carry a 192 KB register file per SIMD. In wave32 mode each vector register is 1024 bits wide, so 192 KB holds 1536 vector general-purpose registers, and each SIMD has sixteen wave slots. Divide 1536 registers across sixteen waves and you get 96 registers per wave. That is the line: a kernel that uses 96 or fewer VGPRs runs all sixteen waves at full occupancy, and every register past 96 costs you waves.

## What the fused GEMM actually spends registers on

The staged kernel from the last post was register-cheap precisely because it was two dumb passes. The first pass read Q4_K and wrote fp16 scratch. The second pass was a plain fp16 GEMM that read the scratch back. Neither pass held much live state, so the kernel sat around 72 VGPRs and ran all sixteen waves. It just happened to be memory-bound, so the spare occupancy bought nothing.

Fusing the two passes changes the register picture entirely. Now a single kernel has to, inside one loop iteration, load a Q4_K block, unpack its packed four-bit weights and its scale and min into usable values, load the matching activation fragment, and feed both into the [WMMA matrix intrinsic](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/), whose accumulator sits in registers and is read and rewritten on every step. The RDNA4 WMMA layout is deliberately lean, with each lane holding eight elements of a 16x16 tile, and AMD notes the RDNA4 layout was simplified specifically to reduce VGPR pressure versus RDNA3. Even so, a prefill tile accumulates several of those 16x16 fragments at once so it can reuse each weight block across many tokens, and the accumulators are all live for the duration of the K loop. Add the unpack temporaries and the per-block scales, and Radeon GPU Analyzer put the first honest version of the fused kernel at roughly 168 VGPRs per wave.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-09-rdna4-vgpr-occupancy-staircase.svg" alt="A step-function chart for RDNA4. The horizontal axis is VGPRs allocated per wave from 0 to 256; the vertical axis is active waves per SIMD from 0 to 16. The curve is a descending staircase: flat at 16 waves until 96 VGPRs, then dropping to 12 waves at 120 VGPRs, 10 at 144, 9 at 168, 8 at 192, and down to 6 by 256. A dashed vertical line at 96 VGPRs marks the last full-occupancy point, and the region to its right is shaded as the occupancy-limited zone. Three kernel points are plotted: a staged fp16 GEMM at about 72 VGPRs on the 16-wave step, a fused dequant-plus-WMMA first cut at about 168 VGPRs on the 9-wave step, and a tuned fused kernel at about 120 VGPRs on the 12-wave step. A yellow arrow connects the first-cut point up to the tuned point, labeled scales to LDS, tighter live ranges, plus three waves." loading="lazy" />
  <figcaption>On RDNA4's 1536-VGPR SIMD, occupancy is a staircase, not a slope. Full sixteen-wave occupancy holds only up to 96 VGPRs; the fused GEMM's ~168-VGPR footprint lands on the nine-wave step at 56 percent occupancy. Moving wave-invariant Q4_K scales to LDS and tightening accumulator live ranges recovers three waves. RGA-estimated footprints, first-order model.</figcaption>
</figure>

Read the staircase and 168 VGPRs is not a gentle penalty. Because these parts allocate registers in 24-register blocks, 168 rounds to a seven-block allocation, and 1536 registers divided into seven-block chunks leaves room for nine waves. Nine of sixteen slots is 56 percent occupancy, and it is almost exactly the picture Chips and Cheese caught in a completely different workload, where a [RDNA4 raytracing shader was pinned to nine of sixteen threads by VGPR usage](https://chipsandcheese.com/p/dynamic-register-allocation-on-amds). Nine resident waves is not enough independent work to keep the SIMD busy while weight loads and WMMA results are in flight, so the fused kernel spends part of every stall idle. That idle time is the gap between the roofline's 2.7x and the measured 1.95x.

## The recoverable registers versus the stubborn ones

Not all 168 of those registers deserve to be there, and telling the two kinds apart is the whole job. The stubborn registers are the WMMA accumulators. The C and D matrices have to live in VGPRs across the entire K loop by definition, and a wide prefill tile deliberately holds several fragments so it can amortize each weight load over many tokens. Shrinking that tile would cut registers, but it would also cut the reuse that made the fused kernel worth building, so those registers are load-bearing.

The recoverable registers are everything that does not actually vary per lane. A Q4_K block's scale and min are the same for all 32 lanes reading that block, which makes them exactly the kind of wave-invariant data that belongs in the Local Data Share or in scalar registers rather than in a vector register replicated across every lane. Aaltonen's old advice applies unchanged: [move group-shared values out of VGPRs and the register budget frees up for occupancy](https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/). Packing the fp16 scales two to a register and tightening the accumulator live ranges by tiling the K loop into shorter segments does the rest. Together those take the kernel from about 168 VGPRs to about 120, which is a five-block allocation and twelve resident waves.

| Qwen3.5-9B prefill GEMM, RX 9070 XT | VGPR/wave | waves | occupancy | prefill |
| --- | ---: | ---: | ---: | ---: |
| staged fp16 (memory-bound) | ~72 | 16 | 100% | 219 tok/s |
| fused, first cut | ~168 | 9 | 56% | ~430 tok/s |
| fused, scales to LDS + tight live ranges | ~120 | 12 | 75% | ~500 tok/s |
| llama.cpp `pp512` reference | — | — | — | 973 tok/s |

The jump from 9 to 12 waves is worth roughly another 15 percent of prefill on the 9B, taking it from about 430 to about 500 tok/s in the profiling runs. That is a smaller win than the fusion itself, and it should be, because latency hiding has diminishing returns: the step from 56 to 75 percent occupancy closes most of the stall gap, and the last few waves up to full occupancy would buy less and cost the accumulator tile. The honest read of the table is that the fused kernel traded a bandwidth problem for an occupancy problem, and the occupancy problem is the cheaper of the two to chip away at.

## The hardware already has a better answer

There is a cleaner fix than counting registers by hand, and it is sitting in the RDNA4 silicon unused on this path. RDNA4 introduced [dynamic VGPR allocation](https://chipsandcheese.com/p/dynamic-register-allocation-on-amds), a mode where a wave launches with a minimal register allocation and asks for more at runtime with an `s_alloc_vgpr` instruction, then frees them again when it leaves the hungry code. The occupancy question inverts: the driver sets how many waves run per SIMD directly, and a wave only holds its peak register count during the short window it actually needs it. A GEMM built this way could keep sixteen low-register waves resident and spike up to the accumulator-heavy allocation only inside the multiply loop, getting both the occupancy and the wide tile.

The reason ZINC cannot use it yet is entirely about plumbing, not silicon. Dynamic VGPR mode is restricted to wave32 compute shaders, it is gated behind a chip-wide control register the driver has to set up, and allocation requests can fail and force a wave to busy-wait, which brings its own deadlock-avoidance machinery. So far it has only been seen in AMD's own indirect-mode raytracing shaders, and Chips and Cheese notes the obvious next step directly: generic compute could benefit too, once the feature is exposed through the toolchains. It is not in the Vulkan compute path today. Nvidia shipped its own version of this idea, `setmaxnreg`, back in Hopper, and [some of its GEMM libraries already lean on it](https://github.com/NVIDIA/cutlass/issues/2007), which is a fair sign of where AMD's compute stack is heading.

So the near-term move is the boring one, and the long-term move is waiting on a driver. ZINC prefills Qwen3.5-9B on the 9070 XT at roughly 500 tok/s once the wave-invariant scales come out of the vector registers, against llama.cpp's 973, and the gap is down to about 1.9x. The lesson that carries past this kernel is that the two bottlenecks are a matched pair. First you stop moving bytes you do not need to move, and the moment you do, the registers you spent to avoid moving them become the thing that caps you. On a machine where the register file is the real budget, a fused kernel is a bet that the occupancy you give up is cheaper than the bandwidth you save, and on RDNA4 the arithmetic of that bet is a staircase with a step every 24 registers.
