---
title: "Attention's two matmuls want different number formats on RDNA4"
seoTitle: "int8 Flash Attention on RDNA4: QK Wants Integers, PV Wants FP8"
date: "2026-07-13"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - vulkan
  - flash-attention
  - wmma
  - int8
  - fp8
  - quantization
  - kv-cache
  - local-llm
  - llm-inference
keywords:
  - int8 flash attention RDNA4
  - quantized attention QK PV
  - SageAttention K smoothing
  - fp8 e4m3 softmax probabilities
  - RDNA4 WMMA int8 attention
  - int8 KV cache RDNA4 decode
  - flash attention row max quantization scale
  - Q8_1 activation quantization Vulkan
  - local LLM long context accuracy
  - llama.cpp prefill AMD gap
excerpt: "Yesterday's int8 port doubled the matrix rate for every projection in Qwen3.5-9B and left attention untouched at fp16, which now makes attention 43 percent of ZINC's RDNA4 prefill. Quantizing it splits cleanly in two: QK-transpose takes int8 with one exact correction, and PV takes int8 and quietly loses 4 percent of long-context perplexity, because after a softmax the operand is an exponential and int8 has no exponent."
seoDescription: "Quantizing flash attention on RDNA4: QK-transpose runs on int8 WMMA once K's channel outliers are removed by a mean subtraction that softmax cancels exactly, while the PV matmul needs fp8 rather than int8 because 38 percent of a 32k attention row's mass falls below int8's smallest representable value. Qwen3.5-9B prefill 815 to 962 tok/s on the RX 9070 XT, and an int8 KV cache that the same kernel gets for free."
faqs:
  - question: "Can flash attention run on int8 WMMA?"
    answer: "Half of it can. The QK-transpose matmul takes int8 cleanly once K's channel outliers are handled, and on RDNA4 that doubles its rate from 1024 FP16 FLOP to 2048 INT8 ops per clock per CU. The PV matmul is the problem. Its left operand is the post-softmax probability matrix, whose values are exponentials spread over four or five decades, and int8's uniform grid has 127 steps between zero and one. Measured on Qwen3.5-9B at a 32k context, 38 percent of an attention row's probability mass sits below int8's smallest representable value and rounds to zero."
  - question: "Why does subtracting K's channel mean not change the attention output?"
    answer: "Because softmax is invariant to a constant added to every score in a row. Subtracting a per-channel mean vector from K shifts each score by the dot product of that query with the mean, which is the same constant for every key in the row, so the probabilities are unchanged. The trick is from SageAttention. It is exact rather than approximate, it costs one reduction over the K block, and it drops K's worst channel from 44 times the median magnitude to about 4 times, which returns roughly five bits of the int8 grid to the typical channel."
  - question: "Does quantizing attention help decode?"
    answer: "Not through the matrix core. At batch one the attention matmuls are matrix-vector products bound by how fast the KV cache streams out of VRAM, and doubling a compute rate that was never the constraint changes nothing. What helps is that the int8 K the QK kernel wants is also an int8 K the cache can store. Halving the KV cache halves the bytes attention reads, and on Qwen3.5-9B at 32k that moved decode from about 21 to about 27 tok/s on the RX 9070 XT."
  - question: "Is fp8 attention accurate enough for local models?"
    answer: "On the PV matmul, yes, and this is what FlashAttention-3 and SageAttention2 both do. e4m3 has three mantissa bits and a four-bit exponent, so with a static scale of 448 it represents probabilities down to about 4e-6 with constant relative error, which is what an exponential distribution needs. In ZINC, int8 QK plus fp8 PV cost 0.3 percent perplexity at 4k and 0.8 percent at 32k, against 1.1 and 4.5 percent for int8 PV."
draft: false
---

Yesterday's int8 port took every quantized projection in Qwen3.5-9B onto RDNA4's integer matrix core and left one matmul behind. Attention still runs on fp16 WMMA at half the rate of everything around it, which is why it went from 35 percent of ZINC's prefill profile to [43 percent](https://zolotukhin.ai/blog/2026-07-12-int8-wmma-doubles-rdna4-matrix-rate-q4-k-block-scales-take-half-back/) without anyone touching it. The obvious next move is to quantize attention too.

It is not one move. Flash attention has two matmuls, and they want different number formats.

The QK-transpose product takes int8 without much argument, once you deal with a nuisance in K that has an exact fix. The PV product takes int8 and then quietly destroys long-context accuracy, because its left operand is the output of a softmax. A softmax produces exponentials. int8 is a uniform grid with 127 steps between zero and one, and at a 32k context, 38 percent of a Qwen3.5-9B attention row's probability mass falls below the smallest number that grid can represent.

## The scale P needs is already sitting in a register

Start with the part that looks hard and is not. Quantizing an activation tensor normally means a calibration pass: find the max, derive a scale, and pay for a reduction you did not previously need. In an inner loop on RDNA4, where WMMA and vector instructions [share an issue port](https://zolotukhin.ai/blog/2026-07-11-softmax-steals-rdna4-wmma-issue-slots-flash-attention/), an extra reduction is not a rounding error in the cost model. It is the cost model.

Flash attention hands you the reduction for free. The kernel already tracks a running row maximum `m` and already computes `exp(s - m)` for every score, which means every entry of P is bounded above by 1 by construction. There is no max to find. The scale is a compile-time constant, 127, and the conversion folds into the multiply the softmax was going to issue anyway.

So the PV matmul is the one that should be free, and the QK matmul is the one that should be annoying, because Q and K are live tensors with no bound anyone gave you.

That intuition survives about ten minutes of measurement.

## K's outliers, and the subtraction softmax does not notice

K is not badly behaved on average. It is badly behaved in five channels out of 128.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-13-attention-p-distribution-int8-vs-fp8-grid.svg" alt="A two-panel diagram. The left panel is a bar chart of where a Qwen3.5-9B attention row keeps its probability mass, by magnitude decade, for a 4k prompt in teal and a 32k prompt in violet. At 4k the mass concentrates in the top two decades, 33 percent between 0.01 and 0.1 and 42 percent between 0.1 and 1. At 32k the mass spreads down the tail: 9 percent below 1e-4, 18 percent between 1e-4 and 1e-3, 27 percent between 1e-3 and 1e-2, and only 18 percent above 0.1. A dashed line marks int8's round-to-zero threshold at 0.0039, with a shaded region to its left labelled as 12 percent of the row mass at 4k and 38 percent at 32k. Below the chart, two tick rows show what each 8-bit grid can represent on the same logarithmic axis: the int8 row with a static scale of 127 has ticks only in the rightmost decade and nothing at all in the left three decades, while the fp8 e4m3 row with a static scale of 448 has evenly spaced ticks across the entire axis, eight steps per binade, down to about 4e-6. The right panel plots the per-channel absolute max of K relative to the median channel, on a log axis. Raw K has most channels clustered between 0.4 and 3.6 times the median with five outlier channels at 12, 18.5, 26, 37 and 44 times. After subtracting the per-channel mean, every channel falls between 0.5 and 4.1 times the median. A note says a per-tensor int8 scale set by a 44x channel leaves the typical channel only 3 of the 127 available levels, and that softmax ignores a per-row constant, so the subtraction is exact and free." loading="lazy" />
  <figcaption>Left: the probability mass of an attention row against what each 8-bit grid can hold. Right: K's channel magnitudes before and after the mean subtraction. Both measured on Qwen3.5-9B, layer 20, RX 9070 XT.</figcaption>
</figure>

The right panel is the whole problem with int8 QK-transpose. A symmetric int8 scale has to cover the largest magnitude in whatever the scale is shared across, and if one channel is 44 times the median, the typical channel lands on three of the 127 available levels. That is not an 8-bit matmul. It is a 2-bit matmul with extra steps, and the perplexity says so: a naive per-tensor int8 QK on Qwen3.5-9B moves Wikitext-2 from 8.43 to 9.12, an 8 percent regression that no amount of prefill speed pays for.

The fix comes from [SageAttention](https://arxiv.org/abs/2410.02367), and it is one of those results that seems like a trick until you check the algebra and find it is an identity. Subtract the per-channel mean of K from K. Every score in a row then changes by the dot product of that row's query with the mean vector, which is the same constant for every key in the row. Softmax is invariant to a constant shift within a row. The output does not change. Not approximately, exactly.

What does change is the grid. K's worst channel drops from 44 times the median to 4.1 times, the typical channel goes from 3 levels to 31, and the perplexity regression collapses from 8 percent to 0.2 percent. The mean is one reduction over the K block, computed once per block and reused across every query tile that visits it, so it amortizes into nothing during prefill. During decode, K arrives one token at a time and ZINC keeps a running channel mean in the cache header, which is a small compromise: the mean lags by one token and the shift stops being exactly zero. In practice it is below the noise floor of the fp16 accumulator.

## Then the easy matmul turns out to be the hard one

With QK on int8 and PV still on fp16, the attention kernel gets 1.24x and prefill goes from 815 to 888 tok/s. Then you quantize P, which as established costs nothing to scale, and the kernel gets another 1.3x, and the model starts getting worse at exactly the thing you built long context for.

Look back at the left panel of the figure. At a 4k prompt, an attention row is fairly concentrated: three quarters of its mass sits above 0.01, comfortably inside int8's range. At 32k, the same row has to spread the same total mass of 1.0 across eight times as many keys, and it does. Nine percent of the mass ends up below 1e-4. int8 with a static scale of 127 has a step size of 0.0079 and a round-to-nearest threshold of 0.0039, so every one of those entries becomes zero. Not small. Zero.

Individually that is fine. Collectively, 38 percent of the row's mass is being deleted and renormalized away, and the tokens it belonged to are exactly the diffuse, low-confidence background that long-context retrieval depends on.

| Qwen3.5-9B, RX 9070 XT | attention kernel | prefill | ppl 4k | ppl 32k |
| --- | ---: | ---: | ---: | ---: |
| fp16 WMMA (yesterday) | 1.00x | 815 tok/s | 8.43 | 7.95 |
| int8 QK, no K smoothing | 1.24x | 888 tok/s | 9.12 | 8.71 |
| int8 QK, K mean subtracted | 1.24x | 888 tok/s | 8.45 | 7.98 |
| + int8 PV | 1.61x | 968 tok/s | 8.52 | 8.31 |
| + fp8 PV instead | 1.58x | 962 tok/s | 8.46 | 8.01 |

Perplexity is measured on Wikitext-2 at the stated window. The int8 PV row is the one to read twice: it costs 1.1 percent at 4k, which looks survivable, and 4.5 percent at 32k, which is not. The damage scales with context length because the number of entries that round to zero scales with context length, and a benchmark run at 4k will never show it to you.

## fp8 is not faster, it is shaped correctly

The fix is not more bits. It is the same eight bits arranged differently.

e4m3 spends four of them on an exponent and three on a mantissa. Its steps are logarithmic, so it holds a constant relative error across every decade rather than a constant absolute error across one. With a static scale of 448, the smallest probability it can represent is about 4e-6, a thousand times below int8's floor, and every entry that int8 was flushing to zero survives with three bits of precision. The bottom row of the figure shows the two grids on the same axis, and the argument is visual: int8's ticks all pile up in the last decade, where the mass is not.

This is why [FlashAttention-3](https://arxiv.org/abs/2407.08608) uses FP8 rather than INT8 for its low-precision path on Hopper, and why [SageAttention2](https://arxiv.org/abs/2411.10958) splits the formats the same way ZINC now does, quantizing Q and K to integers and P and V to FP8. Two independent groups arrived at the same asymmetry, and it is not a hardware accident. It follows from the softmax.

RDNA4 can do this. The matrix core exposes fp8 WMMA variants, and AMD's [WMMA guide](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/) treats FP8 and INT8 as the same wide-load class when fusing two 16x16x16 instructions into an effective K=32. AMD's published [matrix rate table](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) only quotes FP16, BF16 and I8, so I cannot cite a number for the FP8 rate, but the two kernels measured within 2 percent of each other on the same tile, which is consistent with parity. That is the honest version: the spec sheet does not say, and the stopwatch says they are the same.

## What actually changed for decode

None of the above touches decode, and it is worth being blunt about why. At batch one, both attention matmuls are matrix-vector products. They are limited by how fast the KV cache streams out of VRAM, not by the rate of the matrix core, and doubling a rate that was never the constraint buys nothing. Decode stayed at 39.6 tok/s at short context, exactly as it did after yesterday's int8 GEMM.

What decode gets is a side effect. The int8 K the QK kernel wants is the same int8 K the cache can store. Once the kernel consumes integers, the cache holds integers, and Qwen3.5-9B's 147 KB per token of fp16 KV becomes 74 KB. Against roughly 5.5 GB of Q4_K_M weights, that is nothing at a 2k context and a great deal at 32k, where the cache was 4.8 GB and is now 2.4. Decode at 32k went from about 21 tok/s to about 27, a 1.3x that comes entirely from bytes not read, and it lands right where the [KV-versus-weights crossover post](https://zolotukhin.ai/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode/) said it would.

## Where the series lands

Prefill on Qwen3.5-9B is 962 tok/s. llama.cpp's `pp512` reference on the same card and the same weights is 973. After six posts of chasing it, the gap is inside run-to-run variance, and I am going to stop calling it a gap.

The lesson that survives all six is narrower than "quantize things." Every one of these posts found the same shape: RDNA4 gives you an enormous matrix rate through an issue port it shares with a small vector unit, so the format your operands are in decides how many vector instructions you are forced to issue in the inner loop, and that decides your throughput. Q4_K put a descale in the loop. The softmax puts a rescale in the loop. And P, the one tensor whose scale was free, turned out to be the one whose grid was wrong.

The next question is whether Q and K want to go further. SageAttention2 puts them in INT4, and RDNA4's WMMA layout does have an INT4 path, where each lane's 8 elements are only a 32-bit load and the wide-K fusion matters twice as much. Whether Q and K survive four bits with per-thread scales is an accuracy question, and after today I am inclined to ask it of the grid before I ask it of the hardware.
