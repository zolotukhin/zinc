---
title: "The fastest Q5_K lm-head kernel was already in the binary, gated off by one constant"
seoTitle: "A K<=4096 Threadgroup Cache Gate Kept ZINC's Fast Metal Q5_K lm-head Kernel Unused"
date: "2026-09-01"
tags:
  - zinc
  - metal
  - apple-silicon
  - m4-max
  - lm-head
  - q5-k
  - kernel-selection
  - dmmv
  - llm-inference
  - local-llm
keywords:
  - Metal Q5_K matvec lm-head kernel
  - dmmv_q5k_native threadgroup input cache
  - lm-head decode bandwidth Apple Silicon
  - SPIRV-Cross shader byte-at-a-time weight reads
  - kernel selection gate shape predicate
  - Muse Glimmer 30B Metal decode
  - M4 Max 546 GB/s local inference
  - threadgroup memory 16 KiB simdgroup per row
excerpt: "A commit on August 17 made Muse Glimmer decode about six percent faster on an M4 Max without adding a kernel. The faster Q5_K matvec was already compiled, already loaded, and never dispatched, because its threadgroup input cache was declared for K<=4096 and Muse's lm-head is K=6656. The lm-head is 925 MB of the roughly 15 GiB the decoder reads per token, and it was running on a translated shader that reads weights a byte at a time. The interesting part is not the fix, it is why the kernel was marked intentionally unused: a correct measurement about dispatch fusion got generalized into a wrong decision about a kernel that was never about dispatch count."
seoDescription: "ZINC's hand-written Metal Q5_K matvec (dmmv_q5k_native) shipped compiled and loaded but never dispatched, because a threadgroup input cache sized for K<=4096 excluded Muse Glimmer's K=6656 lm-head. Raising the cache to 2048 half4 (16 KiB) and routing Q5_K matvecs with K<=8192 and K%256==0 to the native kernel took decode from 23 to 24.4 tok/s on an M4 Max with byte-identical output. This post walks the arithmetic of where the 2.5 ms per token went, why the earlier dispatch-fusion measurement was right but its generalization to the lm-head was wrong, and what a kernel-selection audit should look for."
faqs:
  - question: "Why was the faster Metal Q5_K kernel never used?"
    answer: "It was excluded by its own shape predicate. The hand-written dmmv_q5k_native kernel stages the input vector in threadgroup memory before the matvec, and that staging array was declared to hold K/4 half4 entries with K capped at 4096, which is 8 KiB of threadgroup memory. Muse Glimmer's lm-head has K=6656, so every dispatch fell through to the legacy SPIRV-Cross dmmv_q5k shader. The kernel compiled, loaded at engine startup, and was never selected. Raising the array to 2048 half4 (16 KiB) and gating on K<=8192 with K%256==0 routed the lm-head to it."
  - question: "How much did routing the lm-head to the native kernel actually gain?"
    answer: "Decode went from 23 to 24.4 tok/s on an M4 Max with a 40-core GPU running Muse Glimmer 30B at Q4_K_M, about six percent, with byte-identical output against the previous path. In wall-clock terms that is 43.5 ms per token down to 41.0 ms, so about 2.5 ms recovered. The lm-head reads about 925 MB per token, which at the decode step's measured effective bandwidth of roughly 378 GB/s should cost about 2.4 ms, so the saving is consistent with the old kernel having run that read at roughly half the bandwidth the rest of the step sustains."
  - question: "Why is the lm-head such a large read on this model?"
    answer: "Muse Glimmer's output projection is 202,048 rows by 6,656 columns, which is 1.34 billion parameters. At Q5_K, which packs 256 weights into a 176-byte block for 5.5 bits per weight, that tensor is about 925 MB on disk and in VRAM. The decoder reads all of it for every token it emits, because producing one logit per vocabulary entry touches every row. On a model whose full per-token weight traffic is roughly 15 GiB, the lm-head alone is close to six percent of the bytes and is the single largest tensor in the file."
  - question: "What is the general lesson for a multi-backend inference engine?"
    answer: "Kernel-selection predicates are written against the model in front of you and then silently exclude the next one. The gate here was not wrong when it was written; every model in the catalog at that time had a hidden size at or below 4096. It became wrong when a model with K=6656 arrived, and nothing in the build reported that a compiled, loaded pipeline had a dispatch count of zero. The cheap defense is instrumentation: log per-pipeline dispatch counts on a decode run and treat any loaded-but-never-dispatched kernel as a bug until proven otherwise."
draft: false
---

A commit on August 17 made Muse Glimmer decode about six percent faster on an M4 Max. It did not add a kernel and it did not change a single line of arithmetic. The faster kernel was already compiled into the binary and already loaded at engine startup. It had simply never been dispatched, because an array inside it was declared with 1024 entries instead of 2048.

Muse Glimmer's lm-head is a Q5_K matrix of 202,048 rows by 6,656 columns. That is 1.34 billion parameters in one tensor, about 925 MB of weights, and the decoder reads all of it for every token it emits. On a model whose total per-token weight traffic is roughly 15 GiB, that one matrix is close to six percent of the bytes and the largest single tensor in the file. It was running on an auto-translated shader that reads those weights one byte at a time.

The fix took two lines. What is worth writing down is not the fix but the reason the kernel was sitting there unused, because it was not an oversight. Someone had looked at that kernel, decided it would not help, and written the decision down.

## What the gate was actually protecting

ZINC's Metal backend carries two Q5_K matvec kernels. The legacy one, `dmmv_q5k`, is a shader that came through [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross), the Khronos tool that translates SPIR-V into Metal Shading Language so a Vulkan-first engine can reuse its shaders on Apple Silicon. Translation preserves semantics, not access patterns. The result assigns one thread per output row, reads the packed quantized weights byte at a time, and gives every thread in the workgroup its own trip to device memory for the same input vector.

The other kernel, `dmmv_q5k_native`, is hand-written Metal in the style of [llama.cpp's own Metal kernels](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal). It assigns a simdgroup per row so 32 lanes cooperate on one dot product, reads weights in wide coalesced loads instead of scalar byte fetches, and stages the input vector once in threadgroup memory so the whole workgroup shares it. The comment in the shader is blunt about what it replaces: no threadgroup input caching means 64 times redundant device reads per workgroup.

The staging is where the gate lived. Threadgroup memory is a small fixed budget per threadgroup, 32 KiB on the Apple GPU families this backend targets, per the [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), so the cache has to be declared at compile time with a size that fits. It was declared to hold K/4 `half4` entries with K capped at 4096, which is 8 KiB. Every model in the catalog when that line was written had a hidden size at or below 4096, so the cap cost nothing.

Muse Glimmer's hidden size is 6,656. Every Q5_K matvec on that model failed the predicate and fell through to the translated shader. The pipeline still compiled. It still loaded. It just never ran.

```zig
.q5_k => blk: {
    if (K <= 8192 and K % 256 == 0 and self.dmmv_q5k_native_pipe.handle != null and
        self.dmmv_q5k_native_pipe.max_threads_per_threadgroup >= 256)
    {
        break :blk .{ .pipe = &self.dmmv_q5k_native_pipe, .rows_per_wg = 8, .block_size = 256 };
    }
    break :blk .{ .pipe = &self.dmmv_q5k_pipe, .rows_per_wg = 64, .block_size = 64 };
},
```

The change was to grow the cache to 2048 `half4`, which is 16 KiB and half the threadgroup budget, and to raise the predicate to K at or below 8192 with K divisible by 256. Output stayed byte-identical against the previous path, which is the correctness gate every math-preserving cycle in this effort has to clear.

## The measurement was right, the generalization was not

The reason the native kernel was left unused is written into the effort log, and it is a better story than a missed TODO.

An earlier cycle on this model had tested dispatch fusion directly. Admitting Muse to the fused QKV kernels collapsed Q, K and V into one dispatch across all 52 layers, path evidence confirmed the fused route was taken, output was byte-identical, and wall-clock decode did not move at all: 22.5 tok/s before, 22.5 tok/s after, on a slightly earlier baseline than the 23.0 this post starts from. The conclusion was correct and well supported. Decode on this model is bandwidth-bound, Metal's concurrent encoder already hides the small QKV and norm dispatches behind the bandwidth-bound FFN matmuls, and cutting dispatch count is not the lever.

Then the log extended that conclusion one step too far. It recorded that the same logic applies to fused gate and up projections, and to the Q5_K lm-head, and noted that `dmmv_q5k_native` is loaded but intentionally unused. That is where it went wrong. The native kernel is not a fusion. It does not reduce dispatch count by one. It changes how a fixed number of bytes gets read: coalesced instead of scalar, shared instead of redundant, 32 lanes per row instead of one. A finding about dispatch count has nothing to say about it.

This is the failure mode worth naming, because it is not laziness and it does not look like a bug in review. A real measurement produced a real conclusion, the conclusion was written in general terms, and the general form quietly covered a case the experiment never touched.

## Where the 2.5 milliseconds went

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-09-01-metal-q5k-lm-head-kernel-gate-decode-waterfall.svg" alt="A two-panel chart drawn on a warm cream paper background, titled 'A two-line routing change, in milliseconds and gigabytes per second', with a subtitle naming Muse Glimmer 30B at Q4_K_M on the ZINC Metal backend and an M4 Max 40-core GPU, and noting the Q5_K lm-head is 202,048 by 6,656, about 925 MB per token. The upper panel, titled 'One decode token on an M4 Max, before and after routing the lm-head', shows two horizontal stacked bars against a time axis marked from 0 to 45 milliseconds per token. The upper bar is labelled 'before, SPIRV-Cross dmmv_q5k' and runs to 43.5 milliseconds: a long deep slate segment labelled 'rest of the decode step, 38.6 ms' followed by a hatched terracotta segment annotated to the right as '4.9 ms lm-head, 23.0 tok/s'. The lower bar is labelled 'after, dmmv_q5k_native' and runs to 41.0 milliseconds: the same 38.6 millisecond slate segment followed by a much shorter solid terracotta segment annotated '2.4 ms lm-head, 24.4 tok/s'. A terracotta bracket under the two bar ends spans the gap between them and is labelled '2.5 ms per token recovered'. The lower panel, titled 'Effective read bandwidth on the same machine', is a horizontal bar chart with four bars on a common baseline: a short hatched terracotta bar at 190 GB/s labelled 'old lm-head kernel, inferred from the 2.5 ms saving'; a deep slate bar at 378 GB/s labelled 'ZINC decode step, measured, about 15 GiB/token'; a teal bar at 470 GB/s labelled 'llama.cpp reference, measured on this box'; and a longest bar drawn only as a pale dashed outline at 546 GB/s labelled 'M4 Max peak, Apple published spec'. A footer separates measured from derived quantities, stating that the two decode rates, the 925 MB lm-head and the 378 and 470 GB/s step bandwidths were measured and the output was byte-identical across the change, while the hatched 4.9 ms, 2.4 ms and 190 GB/s figures were computed from the tok/s delta and the tensor size." loading="lazy" />
  <figcaption>Muse Glimmer 30B at Q4_K_M on an M4 Max with a 40-core GPU. The measured quantities are the two decode rates, the 925 MB lm-head, and the 378 GB/s effective bandwidth of the decode step. The 4.9 ms and 2.4 ms lm-head slices and the 190 GB/s bar are derived from those, not separately instrumented, so treat them as arithmetic that is consistent with the result rather than as a kernel-level profile.</figcaption>
</figure>

The arithmetic is worth stepping through because it is the check that the story holds together. Decode went from 23.0 to 24.4 tok/s, which is 43.5 ms per token down to 41.0 ms, a saving of 2.5 ms. The lm-head is 925 MB. A clean profile of this decode path measured the whole step reading roughly 15 GiB per token at about 378 GB/s effective, so 925 MB at that rate should take about 2.4 ms.

Put those together and the old kernel was spending roughly 4.9 ms on a read that costs 2.4 ms at the bandwidth the rest of the step already sustains. That is about half the throughput, which is exactly the shape you expect from scalar byte-at-a-time fetches and 64 redundant copies of the input vector. The saving and the tensor size agree to within the precision of a tok/s number, which is the most reassuring thing about the result: nothing here required a new idea, only running the read at the speed the rest of the model already runs at.

For scale, the same machine peaks at [546 GB/s of unified memory bandwidth](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/), and the llama.cpp reference build measured on this box sustains about 470 GB/s on the same model. ZINC's 378 GB/s is still short of both. Six percent narrows that gap without closing it.

## What this does not fix

Decode on Muse Glimmer is 24.4 tok/s against a reference build recorded in the same effort log at 29.3 tok/s on the same hardware. The lm-head was never the whole gap. The effort log's own profile puts 66 to 78 percent of decode traffic in the dense FFN, gate and up at Q4_K and down split between Q4_K and Q6_K, and those kernels already run at the reference-style rate of roughly 440 GB/s. The remaining gap is per-layer non-matmul latency plus the extra bytes Q6_K costs, and neither of those has a two-line fix.

The gate also moved rather than disappeared. K at or below 8192 covers every hidden size in the catalog today, the same way K at or below 4096 covered every hidden size in the catalog when it was written. A model with a hidden size of 12,288 would fall through to the translated shader exactly as Muse did, and nothing in the build would say so. Sixteen KiB is already half the threadgroup budget, so the next raise is not free; it starts trading occupancy for cache size.

## The audit this deserves

The generalizable part is not about Metal or Q5_K. Any engine that selects kernels by shape predicate accumulates gates written against the model that was in front of someone at the time, and those gates decay silently as the catalog grows. The same pattern shows up on the Vulkan side, where [specialization constants let one shader family expand into several pipelines](/blog/2026-04-23-vulkan-specialization-constants-unlock-rdna4-dmmv-variants/) and each pipeline carries its own admission condition. More paths mean more ways for a model to fall off the fast one without anything looking broken.

What makes this class of bug expensive is that it is invisible from both ends. The build succeeds. The output is correct. The benchmark is slower than it should be, but there is no reference for what it should be, so the number looks like the model's cost rather than the router's mistake. It stayed hidden here through a full multi-hour optimization effort in which someone specifically looked at the kernel and reasoned about it.

The defense is cheap and we did not have it: count dispatches per pipeline on a decode run, and treat any pipeline that loaded successfully and dispatched zero times as a defect until someone explains why it exists. That check would have flagged `dmmv_q5k_native` on the first Muse run, before the effort log ever had a chance to write down that it was intentionally unused. It also generalizes past this one bug, because the shape of the failure, a compiled kernel with no callers, is the same whatever the backend or the quantization format.

The earlier post on [what a 151k lm-head costs on RDNA4 decode](/blog/2026-05-16-what-qwen3-151k-lmhead-costs-on-rdna4-decode/) ended by arguing that the output projection is the cleanest specialization target in the decode loop, because its shape is fixed by the model and never changes during a session. That is still true. What this commit adds is the less flattering corollary: a fixed shape is also the easiest thing to write a predicate against, and a predicate written against one fixed shape is how you exclude the next one.
