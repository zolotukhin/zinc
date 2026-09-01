---
title: "Qwen 3.8 27B Shipped Yesterday. Here's What ZINC's Day-One Support Actually Took."
seoTitle: "Day-One Qwen 3.8 27B Support on ZINC: Same Architecture, Old Bug Included"
date: "2026-08-15"
tags:
  - zinc
  - qwen
  - qwen3-8
  - model-bringup
  - rdna4
  - apple-silicon
  - metal
  - rocm
  - benchmarks
  - llm-inference
keywords:
  - Qwen 3.8 27B GGUF
  - Qwen 3.8 27B local inference
  - Qwen 3.8 27B benchmark
  - ZINC model bringup
  - Qwen 3.8 27B Apple Silicon
  - Qwen 3.8 27B RDNA4
  - Qwen 3.8 27B ROCm
  - Metal decode stall
  - local LLM day one support
excerpt: "Qwen 3.8 27B landed on Hugging Face on August 14 and ran through ZINC the next day. Two weeks later, the tuned ROCm path beats llama.cpp ROCm across all four published workloads. This is what architecture reuse bought us, what the new fused RMS-norm and Q8 path changed, and why Metal decode still needs work."
seoDescription: "How ZINC added day-one support for Qwen 3.8 27B by reusing the existing qwen35 architecture kernels with zero kernel changes, plus real RDNA4 and Apple Silicon Metal benchmark numbers against llama.cpp, including a decode stall that survived two model generations."
faqs:
  - question: "Why did Qwen 3.8 27B support ship so fast on ZINC?"
    answer: "Because it didn't require new kernel work. Qwen 3.8 27B keeps the same text architecture and tensor dimensions as the prior dense-hybrid 27B checkpoint, which ZINC already supported. The catalog entry, the GGUF download URL, and the VRAM budget are new; the CUDA and Metal kernels underneath are not, because ZINC's kernel-selection logic keys off tensor shape (embedding dimension, expert count, and so on), not off a model name string. A model that keeps the same shape as something already tuned inherits that tuning automatically."
  - question: "Is ZINC faster than llama.cpp on Qwen 3.8 27B?"
    answer: "On the current ROCm suite, yes: ZINC beats llama.cpp ROCm in all four published workloads. The quick-chat result is 135% of llama.cpp prefill, 117% of its decode speed, and 122% overall. Results remain backend-specific: the day-one Vulkan snapshot also led overall, while Metal prefill leads and Metal decode still trails."
  - question: "What's still unresolved for Qwen 3.8 27B on ZINC?"
    answer: "The Metal decode stall for this model's dense-hybrid architecture, first documented on the prior dense-hybrid checkpoint in June, is still present on 3.8: 15.55 tok/s against llama.cpp's 23.38 tok/s. It reproduced almost exactly, which is useful evidence that it's a shape-level bug in the Metal decode path rather than something specific to one checkpoint's weights. CUDA and Intel Arc results for this model are also still outstanding — the catalog currently lists only AMD RDNA4 32GB and Apple Silicon as validated profiles."
draft: false
---

Qwen 3.8 27B went up on Hugging Face on August 14. This post is being written on August 15. In between, it got a ZINC catalog entry, a verified download, a fresh AMD RDNA4 benchmark run, and a fresh Apple Silicon benchmark run — and on RDNA4, it's already ahead of llama.cpp on decode.

> **Update, August 31:** ROCm is now a first-class ZINC backend, and the 27B path has moved again. A fused RMS-norm plus Q8 activation-packing kernel improved decode by about 2.9% in the same-binary A/B test. In the final five-run, same-GGUF comparison, ZINC beat llama.cpp ROCm across all four workloads. The original day-one Vulkan and Metal measurements remain below as a snapshot of bring-up.

| ROCm workload | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode | Overall |
|---|---:|---:|---:|---:|---:|
| Quick Chat | **380.69** | 281.44 | **28.68** | 24.47 | **121.83%** |
| Coding Review | **592.79** | 465.13 | **28.30** | 24.31 | **117.71%** |
| Incident Context | **659.32** | 596.21 | **28.15** | 24.37 | **116.07%** |
| Long Coding Draft | **422.58** | 330.21 | **28.36** | 24.29 | **118.27%** |

These are median tokens per second from two discarded warmups and five measured runs on the same Radeon AI PRO R9700, with both engines loading the same Q4_K_M file. Output previews passed all four coherence checks. The [benchmark page](/zinc/benchmarks/) keeps ROCm separate from Vulkan so results from different driver stacks are not blended together.

That turnaround is the whole story, and it's worth being honest about why it happened: not because bring-up got faster, but because for this specific model, there was barely any bring-up to do.

## The trick: it's not a new shape

Every time ZINC picks up a new model family, the real cost isn't the catalog entry — it's the kernels. Dense versus mixture-of-experts, attention versus hybrid state-space blocks, embedding dimension, quantization mix per tensor: all of it has to be measured and tuned per architecture, which is why the effort logs in this repo run to dozens of multi-hour sessions per backend.

Qwen 3.8 27B skipped all of that, because it isn't a new shape. It has the same `qwen35` dense-hybrid text architecture and dimensions as the prior 27B checkpoint, with retrained weights. ZINC's kernel-selection code doesn't ask which model release is running — it asks "what's the embedding dimension, how many experts, what quant format per tensor." A 27B dense model with the same shape as one ZINC already knows how to run inherits every tuned path automatically: the CUDA decode graph, the fused gate+up+SwiGLU kernels, the Q5/Q6 prefill paths — all of it, unmodified.

So day-one bring-up was, literally: add one catalog entry.

```zig
.{
    .id = "qwen38-27b-q4k-m",
    .display_name = "Qwen3.8 27B Dense Q4_K_M",
    .release_date = "2026-08-14",
    .family = "qwen3.8",
    .file_name = "Qwen3.8-27B-Q4_K_M.gguf",
    .download_url = "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/...",
    .sha256 = "7e78da5d...",
    .size_bytes = 17_106_775_008,
    // Qwen3.8-27B retains the established qwen35 text architecture and dimensions.
    .required_vram_bytes = 20 * 1024 * 1024 * 1024,
    ...
},
```

No new shader. No new env-var-gated kernel path. The legacy shape-specific dense-27B tuning knobs sitting in the CUDA forward pass are still named after the bring-up that discovered them, because renaming them wouldn't change what they do — they gate on tensor shape, not on a release-name string. They just quietly started applying to 3.8 too.

## What day one actually measured

<figure class="diagram-card diagram-wide">

| Metric | ZINC | llama.cpp | ZINC as % of llama.cpp |
|---|---:|---:|---:|
| Prefill (tok/s) | **241.6** | 198.0 | **122%** |
| Decode (tok/s) | **32.2** | 30.9 | **104%** |
| Quick-chat end-to-end (tok/s) | 48.2 | 49.0 | 98% |
| Phase-combined overall | — | — | **109%** |

  <figcaption>AMD RDNA4 (Radeon AI PRO R9700), Qwen 3.8 27B Dense Q4_K_M, ZINC vs llama.cpp, measured August 15, 2026. ZINC leads on prefill, decode, and overall; the one disclosed exception is quick-chat end-to-end throughput at 98%, a scenario dominated by short-generation overhead rather than steady-state decode.</figcaption>
</figure>

RDNA4 was a clean bring-up in the sense that mattered: no new losses, one honest near-miss. Apple Silicon told a stranger story.

<figure class="diagram-card diagram-wide">

| | June 13 — Prior 27B checkpoint | August 15 — Qwen 3.8 27B | Change |
|---|---:|---:|---:|
| Prefill (ZINC) | 15.9 tok/s | **116.0 tok/s** | **7.3x** |
| Prefill vs llama.cpp | 15% | **111%** | stall fixed |
| Decode (ZINC) | 15.4 tok/s | 15.6 tok/s | flat |
| Decode vs llama.cpp | 70% | 66% | still stalled |

  <figcaption>Mac Studio (M4 Max, 40-core GPU, 64 GB unified), same qwen35 dense-hybrid architecture, two model generations apart. Prefill went from a near-total stall to a solid win because Metal shipped batched prefill in the months between these two runs. Decode reproduced the same stall almost exactly, on different weights, two months apart — strong evidence it's a bug in ZINC's Metal decode path for this architecture, not something specific to one checkpoint.</figcaption>
</figure>

That second table is the actual finding of this bring-up. It isn't "Qwen 3.8 27B is fast" or "Qwen 3.8 27B is slow" — it's that a single model shape can carry a fixed bug and an unfixed bug across a full model generation, at the same time, and the only way to know which is which is to measure the new checkpoint instead of assuming the old numbers still apply.

The prefill fix wasn't specific to this model at all — it's the batched-prefill work that shipped for Metal generally after June, and Qwen 3.8 27B simply walked into it for free, the same way it walked into the CUDA tuning for free. The decode stall is the opposite kind of inheritance: whatever causes it lives in the shape-specific decode path for this architecture on Metal, and neither the June bring-up nor the August one has fixed it. It survived a full weights swap untouched, down to the second decimal.

## What isn't done yet

The catalog currently lists Qwen 3.8 27B as validated on two profiles: `amd-rdna4-32gb` and Apple Silicon. That's deliberately not the full backend list. CUDA (RTX 5090) and Intel Arc haven't run this model yet — the last CUDA and Intel catalog sweeps predate the 3.8 release, and there's no fresh row for it on either target. Those are next, not done.

The Metal decode stall is also next, not done. Reproducing it cleanly on a second model generation is actually useful: it rules out "bad checkpoint" as the explanation and points squarely at ZINC's own decode path for this architecture shape on Metal. That's a better bug report than the June version had, even if the bug itself is unchanged.

## The takeaway

Fast bring-up wasn't a process improvement. It was architecture reuse paying off exactly the way it's supposed to: build the kernels once, tune them once, and every model that shares the shape gets the tuning for free, on release day if you're paying attention. The cost of that leverage is that "for free" cuts both ways — you also get the shape's unfixed bugs for free, and the only way to find out which is which is to actually run the new weights instead of trusting the last model's numbers.

Qwen 3.8 27B is running on ZINC today, ahead of llama.cpp on the backend that matters most for it so far, with one open regression that's now better understood than it was in June. That's what day one looks like when the hard work already happened for a different checkpoint.
