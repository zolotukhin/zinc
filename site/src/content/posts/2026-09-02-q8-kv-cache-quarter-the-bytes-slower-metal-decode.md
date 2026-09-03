---
title: "Q8 KV cache: a quarter of the bytes, seventeen percent slower decode on Metal"
seoTitle: "A Q8_0 KV Cache Reads 3.8x Fewer Bytes And Still Costs ZINC 17% Of Metal Decode"
date: "2026-09-02"
tags:
  - zinc
  - metal
  - apple-silicon
  - m4-max
  - kv-cache
  - quantization
  - flash-attention
  - llm-inference
  - local-llm
keywords:
  - Q8_0 KV cache Metal decode
  - KV cache quantization slower decode
  - flash attention per-position dequant
  - block scale dependent load Metal
  - Muse Glimmer 30B M4 Max decode
  - ZINC_METAL_KV_Q8 default
  - llama.cpp cache-type-k f16 default
  - unified memory KV cache capacity
excerpt: "In July this blog argued a Q8 KV cache is close to free on an RDNA4 card. On August 17 we turned it off by default for Muse Glimmer on Apple Silicon, because it was costing 15 to 22 percent of decode. Both results are correct. Q8_0 stores a KV element in 1.06 bytes instead of 4, so the quantized path reads 3.8 times fewer bytes and still loses, which means the flash-attention kernel at ordinary context lengths is not bandwidth-bound at all. KV quantization is a capacity decision that a decode kernel charges you for in time."
seoDescription: "ZINC's Metal backend defaulted Muse Glimmer 30B to a Q8_0 KV cache and measured decode at 18.4 tok/s against 22.3 tok/s on the unquantized cache at 257 tokens of context on an M4 Max. The Q8 path reads 3.8 times fewer bytes, so the loss is not bandwidth: it is the per-position dequant inside the batched flash-attention kernel, where every float4 of key or value needs a second dependent load of the block scale plus four integer-to-float conversions. On a 64 GB machine holding a 16 GB model, capacity was never the binding constraint, so the trade bought nothing. This post walks the byte arithmetic, quotes both shaders, and argues that a KV cache format is a per-backend decision about which resource is actually scarce."
faqs:
  - question: "Why would a smaller KV cache make decode slower?"
    answer: "Because the flash-attention kernel that reads it is not bandwidth-bound at ordinary context lengths. A Q8_0 block is 34 bytes for 32 values, so a four-element vector costs a 4-byte packed load plus a second, dependent 2-byte load of the block scale, then four integer-to-float conversions and four multiplies. The unquantized path is one 16-byte vector load and nothing else. At 257 tokens of context the entire KV cache is on the order of 27 MB against roughly 15 GiB of weights read per token, so the bytes saved are noise and the extra instructions are not."
  - question: "How much did the Q8 KV cache actually cost on Apple Silicon?"
    answer: "Between 15 and 22 percent of decode across the runs. The clearest case measured 22.3 tok/s on the unquantized cache against 18.4 tok/s on Q8_0, for Muse Glimmer 30B at Q4_K_M on an M4 Max with 257 tokens of context. The cost scales with context because the dequant is per position, so a longer prompt means more positions paying it."
  - question: "Does that mean KV cache quantization is a bad idea?"
    answer: "No, it means it is a capacity optimization rather than a bandwidth optimization, and it pays only where capacity binds. On a 16 GB RX 9070 XT running eight agents, an FP16 KV cache runs the card out of memory near 9k tokens per agent and Q8 roughly doubles that, which is the difference between a swarm that fits and one that does not. On a 64 GB M4 Max holding a 16 GB model, nothing binds, so the same knob is pure loss. ZINC now defaults Muse to the unquantized cache and keeps Q8 behind ZINC_METAL_KV_Q8=1 for memory-constrained setups."
  - question: "Is an f32 KV cache the right default on Metal?"
    answer: "It is the right default among the two formats that exist in this backend today, and it is still one format too wide. llama.cpp defaults both halves of the cache to f16, which is half the bytes of ZINC's unquantized path and carries no dequant instructions at all, since a half-to-float conversion is a load-time widening rather than a block-scale lookup. That is the option ZINC does not implement yet, and it is the one that should win: f16 keeps the fast load shape and halves the 14 GB the f32 cache reserves at maximum context."
draft: false
---

In July this blog worked through an eight-agent swarm on a 16 GB RX 9070 XT and concluded that [a Q8 KV cache is close to free](/blog/2026-07-31-q8-kv-cache-pushes-an-rdna4-swarm-crossover-from-5k-to-10k-tokens/). Halve the bytes, push the crossover where private KV reads overtake the shared weight stream from 5k tokens to 10k, roughly double the context the card can hold. It was a storage-format change with no kernel work and it paid on both axes.

On August 17 we turned the same optimization off by default on Apple Silicon. Muse Glimmer 30B at Q4_K_M decodes at 22.3 tok/s on the unquantized cache and 18.4 tok/s on Q8_0, measured back to back on an M4 Max at 257 tokens of context. Across the runs the quantized cache was 15 to 22 percent slower.

The part worth writing down is not that the number went the other way on a different backend. It is why. Q8_0 packs a KV element into 1.06 bytes against the unquantized path's 4, so the slower path reads 3.8 times less data. A cache that reads a quarter of the bytes and loses seventeen percent of decode is telling you something specific: the kernel doing the reading was never bandwidth-bound to begin with.

## The bytes are not the story, which is the story

ZINC reserves the cache for the model's 131,072-token maximum context, which comes to about 14 GB unquantized and 3.7 GB at Q8_0. Both figures back out to the same geometry, 256 key values and 256 value values per layer per token across the 52 layers, so the two formats really are describing the same tensor. Those are the numbers that make KV quantization look obviously correct, and on a card where they bind, they are.

Now put them next to the workload. At 257 tokens of context the resident cache is 257 of those 131,072 positions, roughly 27 MB. The decode step reads about 15 GiB of weights per token at an effective 378 GB/s, a figure from the same per-kernel profile that produced [yesterday's lm-head result](/blog/2026-09-01-the-fast-q5k-lm-head-kernel-was-gated-off-by-one-constant/). Even allowing for grouped-query attention re-reading each key and value group once per query head, the KV traffic is a rounding error against the weight stream.

So the 9.5 milliseconds per token that Q8 costs, which is 54.3 ms against 44.8 ms once you invert the two rates, cannot be bytes. There are not enough bytes involved for a 3.8 times reduction in them to move anything, in either direction. Whatever is happening is happening per element, inside the kernel, and it scales with the number of positions rather than with the number of gigabytes.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-09-02-q8-kv-block-dependent-scale-load-metal.svg" alt="A diagram on a pale cool-grey background titled 'The smaller KV cache costs more to read', subtitled with Muse Glimmer 30B at Q4_K_M on ZINC's Metal backend, M4 Max, 257 tokens of context, and noting it shows one 32-value slice of the key or value cache as the batched flash-attention kernel consumes it. The left column is headed 'Bytes on the wire, drawn to scale' and shows two horizontal byte strips for the same 32 elements. The upper strip, in deep teal and labelled 'unquantized f32, 128 bytes per 32 values', is a long bar divided by faint rules into eight equal 16-byte segments, captioned '8 x float4, 16 bytes each. One aligned vector load per float4.' Below it a much shorter plum strip labelled 'Q8_0, 34 bytes per 32 values' runs about a quarter the width, made of a solid 2-byte scale block at its left edge followed by a lighter run of int8 cells, captioned '2-byte block scale, then 32 x int8. Two loads per float4, the second dependent.' A dashed bracket spans the gap between the two strip ends and is labelled '3.8x fewer bytes', with a note that both strips are drawn at one pixel per 0.32 bytes so the widths are the real ratio. Beneath them sits a pale blue callout box headed 'Why the byte saving cannot be the story', reading that at 257 tokens the whole cache is roughly 27 MB against about 15 GiB of weights read per decode token, so the bytes saved are noise. The right column is headed 'Work per float4, exploded' and holds two rounded call-out panels. The teal 'f32 path' panel contains a single node reading 'one 16-byte vector load' with a short arrow into a node reading 'dot()'. The taller plum 'Q8_0 path' panel is a vertical chain: '4-byte packed char4 load', then '2-byte block scale load', then '4 int to float, 4 multiplies', then 'dot()'. A dashed curved arrow runs from the block scale node back up to the packed load node, annotated 'dependent, re-read for all 8 vectors in the block'. Across the bottom is a band headed 'The two axes disagree' with two paired-bar groups. The left group, 'Decode rate, 257 tokens of context', shows a teal bar at 22.3 tok/s for f32 above a shorter plum bar at 18.4 tok/s labelled 'Q8_0, 17% slower'. The right group, 'KV cache reserved at the 131,072-token maximum', shows a long teal bar at 14 GB for f32 above a much shorter plum bar at 3.7 GB labelled 'Q8_0, 10.3 GB smaller'. A footer states that both decode rates were measured back to back on the same prompt, that both cache sizes were measured, and that the byte strips and load counts are read off the Q8_0 block layout in ggml of 34 bytes per 32 values." loading="lazy" />
  <figcaption>The same 32 key or value elements, in both cache formats. The strips on the left are drawn to scale, so the Q8_0 cache really is about a quarter the width. The exploded view on the right is what the flash-attention kernel has to execute to consume each four-element vector, and it is where the seventeen percent goes. The two bars at the bottom are the trade in its honest form: Q8 wins the capacity axis by 10.3 GB and loses the speed axis by 3.9 tok/s, and which of those you care about is a property of the machine, not of the format.</figcaption>
</figure>

What the figure is meant to show is that the two panels are measuring different things and only one of them was ever in dispute. Nobody doubted the byte strip. The exploded view on the right is the part that was assumed rather than measured, and it is where the decision went wrong.

## Two loads where there was one

ZINC's Metal backend carries the batched flash-attention kernel in two versions. The unquantized one reads a key vector like this, once per four elements of head dimension:

```metal
const float4 kv = *(device const float4*)(k_cache + kv_base + (i << 2));
score += dot(qv, kv);
```

One aligned 16-byte load, one dot product. The Q8_0 version has to unpack a [ggml Q8_0 block](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h), which is a 2-byte half-precision scale followed by 32 signed bytes, 34 bytes for 32 values:

```metal
inline float4 loadQ8_0Vec4(device const uchar* base, uint vec4_idx) {
    const uint block_idx = vec4_idx >> 3u;          // 8 vec4s per 32-element block
    device const uchar* block = base + block_idx * 34u;
    const float scale = float(as_type<half>(*(device const ushort*)(block)));
    device const packed_char4* quants = (device const packed_char4*)(block + 2u);
    const char4 q = char4(quants[vec4_in_block]);
    return float4(float(q[0]), float(q[1]), float(q[2]), float(q[3])) * scale;
}
```

Count what changed. One vector load became a 4-byte packed load plus a separate 2-byte scalar load of the block header, and the second load is dependent: the address arithmetic for the quants comes from the same block pointer, and the multiply at the end cannot retire until the scale arrives. Then four integer-to-float conversions and four multiplies that the unquantized path does not perform at all.

The scale load is also redundant eight times over. There are eight four-element vectors in a 32-value block and each one re-reads the same 2-byte header. That read almost certainly hits cache, but it is still an instruction, an address computation, and a dependency edge in a loop that runs once per key position per head per layer, 52 layers deep.

This is the same failure shape as [the lm-head kernel gate](/blog/2026-09-01-the-fast-q5k-lm-head-kernel-was-gated-off-by-one-constant/) from yesterday, viewed from the other side. There, the kernel that read more bytes per instruction sat unused and cost us six percent. Here, the format that reads fewer bytes per element was switched on and cost us seventeen. In both cases the thing that mattered was the shape of the load, not the size of the tensor.

## Where the crossover actually is

| Property | Unquantized f32 cache | Q8_0 cache |
| --- | ---: | ---: |
| Bytes per KV element | 4 | 1.06 |
| Cache reserved at 131,072 context | 14 GB | 3.7 GB |
| Device loads per four elements | 1 | 2, one dependent |
| Conversions per four elements | 0 | 4 |
| Decode at 257 tokens, M4 Max | 22.3 tok/s | 18.4 tok/s |

The table is the whole argument in five rows. Q8 wins everything about storage and loses everything about execution, and on this machine execution is the only side that is scarce. The M4 Max under test has 64 GB of unified memory holding a 16 GB model. Even the full 14 GB reservation at maximum context fits, and a realistic session is nowhere near maximum context.

Compare that to the RDNA4 case where this series first reached for Q8. Eight agents on a 16 GB card, 5.5 GB of that spent on resident weights, leaving about 10 GB of KV budget for all eight. An FP16 cache exhausts it near 9k tokens per agent. There, the quantized cache is not an optimization, it is the only configuration that runs.

The honest generalization is that KV quantization trades instructions for capacity at a fixed rate, and the exchange rate is set by the kernel while the value of each side is set by the machine. Both of this blog's earlier KV posts held that view implicitly. The July post measured a card where capacity was the binding constraint. The August post on [where KV quantization stops being free](/blog/2026-08-14-kv-cache-quantization-stops-being-free-below-8-bits/) measured where the quality side breaks down below 8 bits. Neither of them checked what the dequant costs in a kernel that is not bandwidth-limited, and that is the term that dominated here.

## The default we shipped is still one format too wide

The fix was five lines in `defaultKvCacheQ8Enabled`: return false for the Muse Glimmer architecture, leave `ZINC_METAL_KV_Q8=1` as the opt-in for memory-constrained setups. Output stayed correct on the standing gates, and speculative decode stayed byte-identical against the per-token path.

It is not the right answer, only the better of two available ones. ZINC's unquantized Metal cache is f32. The reference build we measure against, llama.cpp, defaults both halves of the cache to f16, which its [server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) lists as the default for `--cache-type-k` and `--cache-type-v` alike. Half precision is the format that should actually win this argument, because a half-to-float widening happens in the load itself rather than through a block-scale lookup. It keeps the single-load shape, drops no dependency edge into the inner loop, and still cuts the 14 GB reservation in half. We do not implement it on this path, so we defaulted to the wide, exact, fast thing instead of the narrow, exact, fast thing.

There is also a context length we have not measured. Attention cost grows with position count while the weight stream does not, which is the entire premise of [Flash-Decoding](https://pytorch.org/blog/flash-decoding/) and the reason the analytical inference models in [Pope et al.](https://arxiv.org/abs/2211.05102) treat the KV term separately from the parameter term. Far enough out, KV bytes stop being noise. What is unclear is whether Q8 ever crosses back over on this kernel, because the dequant cost is per position and therefore scales with context on exactly the same slope as the byte saving. If both terms grow linearly, the ratio does not move and Q8 stays slower forever, just with a larger absolute gap. That is a prediction from the shape of the code, not a measurement, and it is the next thing to run.

## What to take from a knob that reversed sign

The generalizable claim is narrow and I want to keep it narrow. It is not that KV quantization is bad, and it is not that measurements do not transfer between backends. It is that a compression format is two decisions wearing one name. One decision is how many bytes the data occupies, which is a property of the format. The other is how much work it takes to consume a byte, which is a property of the kernel that consumes it. Marketing a format by the first number and inheriting the second by accident is how you ship a default that costs seventeen percent.

The cheap defense is to make the second number visible. We knew the cache sizes to the gigabyte before we enabled Q8 and had never once measured the two attention kernels against each other on the same prompt. That comparison takes one run. It is now the gate on any storage-format change in this engine, alongside the byte-identical correctness check that every math-preserving cycle already has to clear.
