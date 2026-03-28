---
title: "What broke first in local LLM inference on AMD RDNA4"
date: "2026-03-27"
tags:
  - zinc
  - llm-inference
  - rdna4
  - vulkan
  - qwen3-5
  - flash-attention
  - page-attention
  - vllm
  - llama-cpp
keywords:
  - local LLM inference
  - AMD GPU inference
  - RDNA4 LLM
  - Vulkan inference
  - flash attention
  - KV cache
  - mixture of experts
  - state space model
  - RoPE
  - GGUF
  - Qwen3.5-35B-A3B
  - llama.cpp alternative AMD
  - GPT-2 byte-level BPE
excerpt: "Early ZINC failures in local LLM inference on AMD RDNA4: Vulkan bugs in flash attention, KV cache, RoPE, MoE, SSM, and tokenization."
---

The first version of ZINC, our local LLM inference engine for AMD RDNA4 GPUs, did not fail in one impressive way. It failed in ten smaller, more humiliating ways. The forward pass skipped all 40 transformer layers. The tokenizer turned spaces into the wrong token. Flash attention read the K cache as a page table and hung the GPU. One dispatch bug quietly zeroed 97% of the vocabulary logits.

I think these early local LLM inference failures are worth writing down because people tend to talk about inference engines as if the hard part starts with optimization. It does not. The hard part starts earlier, when you are still proving that the computation you think you are running is actually the computation the model needs.

That was the real beginning of [ZINC](/zinc). It is the same project I introduced in [why we're building ZINC for local LLM inference on AMD GPUs](/blog/2026-03-25-why-we-are-building-zinc), but this was the less glamorous part of that story. Before any serious throughput work, before any clean benchmark story, the project had to survive a long sequence of correctness failures on the path to a working forward pass for Qwen3.5-35B-A3B on AMD RDNA4. The exact GGUF we were running was [Unsloth's `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/blob/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf). By the time these fixes landed, ZINC was still only around 0.8 tok/s against a llama.cpp baseline of 107 tok/s on the same node. But for the first time, it was running something recognizably close to the right model.

## The first mistake was thinking the model was simpler than it was

The most basic bug was also the most important one. The initial `decodeStep` was barely a forward pass. It embedded the token, ran a final RMS norm, projected to the LM head, and stopped. None of the 40 transformer layers ran at all, which meant every token was effectively being classified from an almost empty computation graph.

That bug would have been easier if the model had at least matched our mental model. It did not. We started by treating Qwen3.5-35B-A3B as a straightforward mixture-of-experts, or MoE, transformer. The GGUF tensor names quickly proved otherwise. The model includes attention, MoE, and a structured state space model, or SSM, in the same network. In practice, that meant 30 of the 40 layers were not standard attention layers at all. They were recurrent delta-net style SSM layers, with only every fourth layer using full attention.

That changed the implementation plan immediately. We needed two execution paths, not one. Full attention layers needed Q/K/V projections, rotary position encoding, KV cache writes, flash attention, and gated output. SSM layers needed a different path with GPU projections feeding CPU-side recurrent updates and persistent state. The project only really started once the engine was honest about that architectural split.

What matters in this sequence is that the failures were dependent. We could not reason about throughput until the model architecture was right. We could not trust output quality until dispatch geometry was right. We could not trust decode at all until the KV cache and flash attention path stopped corrupting memory.

## Where the local LLM inference pipeline was actually failing

To make the debugging surface more concrete, this is where the failures sat in the Vulkan inference path. The tokenizer bug was upstream. The rotary position encoding, or RoPE, bug sat inside the attention math. The key-value cache, or KV cache, and flash attention bugs only appeared once decode started. The MoE and SSM bugs lived deeper in the 40-layer stack and were easy to misread as vague model-quality problems instead of plain wrong execution.

This is also why the early work did not look much like a typical vLLM or TensorRT-LLM optimization story. Before scheduling or batching mattered, the bare Vulkan inference path had to become correct from prompt tokenization all the way to logits.

## The bugs got smaller and more dangerous

Once all 40 layers were finally running, the failure mode changed. ZINC started producing varied-looking output, which is a dangerous phase because it looks more alive than it really is. The text was still incoherent, and the real clue came from the logits: 97% of them were exactly zero.

That turned out to be one of the cleanest bugs in the whole project. The Q8_0 decode matmul-vector, or DMMV, shader processes two output rows per workgroup, but the dispatch logic was written as if each workgroup handled 64 rows. On the LM head, that mistake meant we were launching enough workgroups to compute only 7,760 of 248,320 vocabulary rows. The model was not making a bad choice across the whole vocabulary. It was only seeing about 3% of it.

```zig
const workgroups_x = switch (quant_type) {
    .q8_0, .f16 => (M + 1) / 2,
    else => (M + 63) / 64,
};
```

This was a good example of why low-level inference bugs are so unpleasant. The shader itself was not obviously crashing. The engine was not obviously dead. The output just looked vaguely wrong until the row count made the problem undeniable.

The flash attention hang was worse because it was louder. Decode would reach a full attention layer, the GPU would stall, and the process would die under a timeout. The root cause was embarrassingly concrete: the shader expected a page table buffer at one binding, but the forward pass was binding the K-cache buffer there instead. The shader read float values as page IDs, computed garbage offsets, and walked straight into out-of-bounds memory.

The fix was simple once the problem was visible. We created an identity page table so `page_ids[i] = i` and used a flat `page_size=1` mapping until real paged KV handling was ready. But getting to that point took the usual Vulkan debugging tax. When a GPU hang shows up at the wrong abstraction level, you spend a lot of time asking whether the bug is in the shader, the descriptor bindings, the barrier ordering, or the model logic. In this case it was the bindings.

## A lot of the hardest bugs were quiet correctness bugs

Some of the most damaging problems were easy to underestimate because they did not immediately crash anything.

The tokenizer was one of them. `"The capital of France is"` should have become five tokens. Instead it became nine, because the tokenizer was splitting raw UTF-8 characters and never applying the GPT-2 byte-to-unicode mapping. In a GPT-2 byte pair encoding, or BPE, tokenizer, the space character is not just a space. It gets remapped into the byte-level representation that makes the merge table work. Without that step, the model can still emit text-like fragments, but the token boundaries are wrong from the very beginning.

Rotary position encoding was another quiet bug. Qwen3.5 uses Interleaved Multi-section RoPE, which means only part of each head gets rotated. We were rotating the full head dimension, which quietly corrupted most of the Q and K vectors. The head dimension itself was also wrong. We inferred `2048 / 16 = 128`, but the GGUF metadata said the actual key length was 256. That one mistake infected buffer sizing, RoPE parameters, and flash attention math all at once.

Then there was the MoE shared expert. Qwen3.5 does not route tokens only through top-k experts. It also has a shared expert path that every token passes through. We initially skipped it. That is the kind of omission that does not necessarily produce an obvious crash or an obvious all-zero tensor. It just makes the model systematically worse in a way that feels hard to pin down until you compare the tensor set carefully.

The same pattern showed up in prefill. Our first prefill shortcut only ran the last prompt token through the final norm and LM head. That meant the KV cache was still empty and the SSM state was still zero when decode began. The first generated token had almost no context, even though the prompt looked like it had been "processed." Replacing that shortcut with a real token-by-token pass through all 40 layers was one of those changes that sounds obvious in hindsight and completely necessary once you see the bug.

## Most of the failures came from three places

What made the early ZINC debugging loop feel chaotic is that the bugs were not all the same kind of bug.

The first bucket was model interpretation. We were reading the architecture wrong, skipping the shared expert, inferring the wrong head dimension, and rotating more of the head than Qwen3.5 actually rotates. Those are the bugs that make the engine look alive while the math underneath is still wrong.

The second bucket was decode state. The KV cache was empty when it should have been populated. The flash attention path was wired to the wrong buffer. Prefill was pretending to do work it had not really done. Those are the bugs that only show up once generation begins, which makes them much harder to isolate than a clean compile failure.

The third bucket was trust. The tokenizer looked plausible until we compared tokens. The logits looked varied until we counted how many were exactly zero. The baseline looked stable until Mesa changed underneath it. Those are the bugs that force you to stop trusting vibes and start measuring everything.

## Reproducibility became part of the engine

One of the more frustrating lessons from this stage had nothing to do with Zig or shaders. The llama.cpp baseline moved from roughly 110 tok/s down to 89 tok/s between runs, and for a while it looked like we had some mysterious instability in the benchmark environment.

The answer was simpler and more annoying. Ubuntu had auto-updated Mesa from 25.0.7 to 25.2.8, and RADV performance on RDNA4 cooperative matrix workloads dropped by about 14%. That is not just a benchmark inconvenience. It changes how every optimization result gets interpreted. If the baseline is drifting under your feet, you stop learning from your measurements.

That is why driver pinning ended up in the same category as tokenizer correctness and descriptor bindings. Reproducibility is not administrative overhead on a project like this. It is part of the engine. If the target environment moves too much, the development loop gets noisier and the wrong ideas survive longer than they should. I wrote the hardware and driver side of that setup down in the [home AI rig post](/blog/2026-03-26-building-a-local-ai-rig), and the lower-level environment details live in the [RDNA4 tuning notes](/zinc/docs/rdna4-tuning).

## What these early failures changed

The biggest thing these initial challenges changed was the way I think about progress on ZINC. At the beginning, it was tempting to frame the project as a straight line from "Vulkan kernels on RDNA4" to "fast local inference on AMD." The real path was much messier. First I had to make the engine honest. Then I had to make the model interpretation honest. Only after that did throughput optimization start to mean anything.

That is why I do not find early low token-per-second numbers especially discouraging. A correct 0.8 tok/s engine teaches more than a fast wrong one. Once the tokenizer is correct, the layer stack is real, the KV cache is populated, the page table binding is sane, and the driver baseline is pinned, the next stage becomes much more concrete. Move more of the SSM path onto the GPU. Reduce per-token submission overhead. Collapse the huge number of Vulkan submissions into something the hardware can actually digest.

Those are hard problems, but they are the right problems. These were not ROCm packaging issues or CUDA portability issues. They were local LLM inference correctness problems in a raw Vulkan engine on AMD RDNA4. The initial phase of ZINC was mostly about getting past the fake ones.

## Related reading behind these bugs

If you want the bigger story around this project, the first two posts are still the right place to start: [why we're building ZINC for local LLM inference on AMD GPUs](/blog/2026-03-25-why-we-are-building-zinc) explains the thesis, and [building a local AI rig: from trading workstation to home AI server](/blog/2026-03-26-building-a-local-ai-rig) shows the RDNA4 node the engine runs on.

The underlying algorithms and systems ideas also have a clear paper trail:

- [FlashAttention](https://arxiv.org/abs/2205.14135), for the attention kernel side of the story.
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180), for why KV cache layout and page indexing matter so much in decode.
- [RoFormer](https://arxiv.org/abs/2104.09864), for the original RoPE formulation behind the partial-rotation bug.
- [OpenAI GPT-2 encoder implementation](https://github.com/openai/gpt-2/blob/master/src/encoder.py), for the byte-to-unicode remapping our tokenizer originally missed.
- [Mamba](https://arxiv.org/abs/2312.00752), for broader state space model context on the SSM side of the network.

These links are not one-to-one explanations of our exact bugs, but they describe the algorithmic terrain those bugs came from. That is part of why the early ZINC work felt so uneven. Every wrong assumption lived at the boundary between model architecture, tokenization, memory layout, and GPU execution.
