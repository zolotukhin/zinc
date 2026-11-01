---
title: 'Bringing ZINC to Intel Arc Pro B70: the trouble with 64 threads'
seoTitle: 'Intel Arc Pro B70 & B65 LLM Inference: ZINC’s Xe2 Work'
date: '2026-11-01'
tags:
  - zinc
  - intel-arc
  - intel-arc-pro-b70
  - intel-arc-pro-b65
  - intel-arc-pro-b60
  - intel-arc-pro-b50
  - intel-arc-b580
  - intel-arc-b570
  - vulkan
  - gemma
keywords:
  - Intel Arc Pro B70 LLM inference
  - Intel Arc Pro B65 local AI
  - Intel Arc Pro B60 GGUF
  - Intel Arc Pro B50 local LLM
  - Intel Arc B580 LLM inference
  - Intel Arc B570 Vulkan
  - Intel Battlemage Xe2 Linux
  - ZINC Intel Arc support
  - Gemma 4 Intel Arc
  - Qwen Intel Arc Vulkan
  - Vulkan subgroup reduction
excerpt: 'Five Intel release-model checks passed. Gemma 26B did not. Following that failure exposed a subgroup reduction bug, an expert-count shortcut and the difference between a working preview and a qualified release.'
seoDescription: 'Inside ZINC’s Intel Arc work: Pro B70/B65, B60/B50 and B580/B570 specs, Vulkan subgroup bugs, Gemma MoE routing fixes and Linux release validation.'
faqs:
  - question: 'What is ZINC’s Intel Arc Pro B70 support status?'
    answer: 'ZINC has a Linux Vulkan path tested on an Arc Pro B70/Xe2 system. The recorded d611994a preview passed five model rows but failed Gemma 26B. A separate corrected-shader control passed all 48 Gemma reference choices; a fresh archive using the new defaults still needs qualification.'
  - question: 'Were Arc Pro B65/B60/B50 and Arc B580/B570 all tested?'
    answer: 'This investigation used one B70/Xe2 system. Those other B-series cards are relevant hardware targets with different memory capacities, but the article does not claim individual ZINC release qualification or measured performance for them.'
  - question: 'Does ZINC use Vulkan or oneAPI for Intel Arc?'
    answer: 'The Intel path discussed here uses Vulkan on Linux with Mesa’s Intel driver. It is separate from a oneAPI, SYCL or OpenVINO backend, and these results do not establish Windows support.'
  - question: 'Why does Intel Gemma MoE use conservative prefill defaults?'
    answer: 'Corrected routing, the model’s full expert count and serial execution with F32 activations matched all three frozen Gemma continuations. Grouped prefill still exceeded the unchanged reference policy, so it remains an experimental override.'
draft: false
---

The sixth model in our Intel release matrix was the most useful one.

Five model rows had passed in our [Intel preview matrix](/zinc/validation/intel-gemma-correctness-d611994a-2026-11-01.json) on the Arc Pro B70 system. Gemma 4 26B-A4B could load and generate text, but its continuation disagreed with the pinned reference beyond our allowed near-tie policy. The model file was present, its checksum was correct, and the tokenizer was not the explanation.

Following that failure took us from a seemingly harmless shader reduction to the number of experts the model actually executes. It also made our work toward the ZINC release much more concrete: Intel Arc support has to survive the model that breaks our assumptions.

This is the engineering story behind that work, and why the **Intel Arc Pro B70 and B65**, along with **Arc Pro B60/B50 and Arc B580/B570**, are interesting targets for local LLM inference. The investigation used one B70/Xe2 system; the other cards appear below as hardware context, with their individual qualification still separate.

## Why Intel Arc Pro B70 and B65 matter for local LLMs

For a local inference engine, GPU memory is a working budget. Weights need to fit alongside the key/value cache, activations and temporary buffers. A model loading successfully only establishes the first part of that budget.

Intel’s current discrete Battlemage lineup gives us several useful capacity tiers. The newer Arc Pro B70 and B65 both provide 32 GB of GDDR6 and 608 GB/s of published memory bandwidth, with different compute resources. Intel introduced them in its [2026 workstation announcement](https://www.intel.com/content/www/us/en/newsroom/news/client-computing/intel-core-ultra-series-3-with-vpro-powers-next-gen-pcs-on-18a.html).

<div style="max-width: 100%; overflow-x: auto; white-space: nowrap;" role="region" aria-label="Intel Arc B-series memory specifications" tabindex="0">

| GPU                                                                                                                             |  VRAM | Memory bandwidth | Xe cores |
| ------------------------------------------------------------------------------------------------------------------------------- | ----: | ---------------: | -------: |
| [Intel Arc Pro B70](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html) | 32 GB |         608 GB/s |       32 |
| [Intel Arc Pro B65](https://www.intel.com/content/www/us/en/products/sku/245796/intel-arc-pro-b65-graphics/specifications.html) | 32 GB |         608 GB/s |       20 |
| [Intel Arc Pro B60](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) | 24 GB |         456 GB/s |       20 |
| [Intel Arc Pro B50](https://www.intel.com/content/www/us/en/products/sku/242615/intel-arc-pro-b50-graphics/specifications.html) | 16 GB |         224 GB/s |       16 |
| [Intel Arc B580](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html)       | 12 GB |         456 GB/s |       20 |
| [Intel Arc B570](https://www.intel.com/content/www/us/en/products/sku/241676/intel-arc-b570-graphics/specifications.html)       | 10 GB |         380 GB/s |       18 |

</div>

_Intel’s published specifications, checked November 1, 2026. These are hardware specifications, not ZINC throughput results or a list of individually qualified boards._

The B70/B65 pairing is especially interesting. The same published memory bandwidth makes the B65 worth investigating for workloads dominated by reading quantized weights. The B70’s additional compute resources could matter more during prompt prefill and other arithmetic-heavy work. That is a hypothesis to measure on both cards; equal bandwidth alone does not establish equal token throughput.

The smaller cards ask different questions. Arc Pro B60 provides 24 GB for weights and runtime state. Arc Pro B50 offers 16 GB with a lower bandwidth budget. Arc B580 and B570 bring 12 GB and 10 GB respectively, making model size and context allocation particularly important. None inherits the large-model fit of a 32 GB board simply by sharing the Xe2 architecture.

Our [earlier B70 architecture article](/blog/2026-05-18-intel-arc-pro-b70-deep-dive-and-zincs-t-intel-plan/) explored the hardware opportunity. The release work has been about making the actual execution paths answer to it.

## The first trap: a 64-thread workgroup is not a 64-lane subgroup

ZINC’s Intel path uses **Vulkan on Linux**, with Mesa’s Intel driver in these tests. Vulkan gives us a common API across vendors, while the shader still has to respect the execution properties of each device.

This declaration creates a workgroup with 64 invocations:

```glsl
layout(local_size_x = 64) in;
```

It does not guarantee that all 64 belong to one subgroup. A subgroup is the smaller set of invocations that participates in operations such as `subgroupAdd`. The distinction is defined in the [Vulkan shader execution model](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-scope-subgroup).

Our tested Xe2 driver reported a default subgroup width of 32 and supported widths from 16 to 32. Some ZINC attention kernels had been written around a complete 64-lane wave. Their pipelines could be created, but the dispatch decision treated a pipeline handle as enough evidence that the specialization was usable.

The result was a 27B prefill correctness failure on particular prompts. Shorter prompts could take another path and look fine.

The host now checks the actual capability before selecting those tiled and grouped attention kernels. Where 64 lanes are unavailable, Intel takes generic batched attention while retaining the other accelerated prefill stages. The [source check](https://github.com/zolotukhin/zinc/blob/10f2141b92158b6801239899d608a77d188ccff8/src/compute/forward.zig) is small:

```zig
fn wave64BatchedAttentionAvailable(self: *const InferenceEngine) bool {
    return self.instance.caps.supportsRequiredSubgroupSize(64) or
        self.instance.caps.default_subgroup_size == 64;
}
```

That fixed one assumption. Gemma found another.

## The router was adding only part of the workgroup

Gemma’s mixture-of-experts layers use a router to choose which expert networks process a token. ZINC fuses several steps of that router: RMS normalization of the hidden state, a scale vector and a matrix-vector product producing expert scores.

Three shader variants used subgroup reductions where the algorithm required a sum across the whole 64-thread workgroup. On a 32-lane subgroup, each reduction saw only part of the work. The RMS normalization and router scores could both be wrong.

This was larger than the last-bit variation that can accompany a different floating-point reduction order. The focused regression caught a router value of **−0.313987255 where the reference expected 0.0682756081**. A score can change sign before the model has even chosen its experts.

<figure class="diagram-card">
  <img src="/blog/2026-11-01-intel-arc-workgroup-reduction.svg" alt="A 64-thread workgroup split into two 32-lane subgroups. Each produces a partial sum; shared memory and workgroup barriers combine them into the complete sum." width="360" height="460" style="width: 360px; max-width: 100%; margin-inline: auto;" loading="lazy" />
  <figcaption>Example with two 32-lane subgroups. A subgroup reduction produces a partial sum; the router needs the complete workgroup sum. The helper also handles more subgroups.</figcaption>
</figure>

The corrected [router helper](https://github.com/zolotukhin/zinc/blob/10f2141b92158b6801239899d608a77d188ccff8/src/shaders/rms_norm_scale_dmmv_f32.comp) combines the subgroup totals through shared memory:

```glsl
shared float s_router_sums[64];

float workgroup_sum(float value) {
    float total = subgroupAdd(value);
    if (gl_NumSubgroups > 1u) {
        barrier();
        if (subgroupElect()) s_router_sums[gl_SubgroupID] = total;
        barrier();
        total = 0.0;
        for (uint sg = 0u; sg < gl_NumSubgroups; sg++)
            total += s_router_sums[sg];
    }
    return total;
}
```

All participating invocations reach the barriers. One elected invocation per subgroup stores its subtotal, and the workgroup combines those subtotals. The first barrier also protects shared-memory reads from the previous call: this helper is used for both normalization energy and the router dot product.

We added a numerical regression covering the router variants, output guards and repeated dispatch. The original Intel shaders fail it. The corrected shaders pass **36 cases on Intel and the same 36 cases on the R9700**, preserving the checks on the backend where the original assumption had been comfortable. The [public evidence](/zinc/validation/intel-gemma-correctness-d611994a-2026-11-01.json) retains both outcomes.

## A fast expert-count shortcut was changing the model

Correcting the router did not finish the investigation.

An Intel-specific default capped Gemma MoE at two experts per token. That makes less work for the GPU, but it changes the computation specified by the model’s metadata. Correct tokenization and plausible prose cannot tell you whether the intended experts actually ran.

ZINC now preserves the metadata’s expert count by default. An explicit expert-count override is logged as approximate execution.

We also separated weight quantization from an additional activation approximation. A Q4_K_M GGUF is still a quantized-weight model when its working activations use F32. Quantizing those activations further for an integer-dot path is another choice, with another possible effect on the continuation. The affected Gemma shortcuts are now opt-in.

The remaining variable was prefill. Grouped prompt execution still disagreed with the native reference beyond our existing policy, even after the router correction and full expert selection. Serial prefill with F32 activations passed all three frozen 16-token continuations: **48 of 48 exact choices**.

<div style="max-width: 100%; overflow-x: auto;" role="region" aria-label="Gemma 26B diagnostic controls" tabindex="0">

| Diagnostic control                                                | Native continuation policy |
| ----------------------------------------------------------------- | -------------------------- |
| Correct router reductions, otherwise earlier defaults             | Failed                     |
| Correct router and full metadata expert count                     | Failed                     |
| Those fixes plus F32 decode activations                           | Failed                     |
| Correct router, full experts, serial prefill and F32 activations  | Passed: 48/48 exact        |
| Correct router, full experts, grouped prefill and F32 activations | Failed                     |

</div>

_These are controls on the recorded preview with rebuilt shader overlays and explicit settings. They establish the passing combination, not qualification of a new archive._

The merged Intel Gemma MoE defaults follow that conservative combination, and startup logging explains the serial prefill choice. Grouped prefill remains an experimental override. We can optimize it further once it satisfies the same reference gate; the failed controls remain available to tell us whether the next change actually improves things.

## What has passed, and what still needs to pass

For this article’s November 1 snapshot, the `d611994a` Vulkan archive passed five model rows together on the B70/Xe2 system. Their frozen continuations contain **239 exact choices out of 240**, plus one policy-qualified near-tie. Tokenizer and HTTP checks passed for those rows. Gemma 26B failed the continuation gate.

<div style="max-width: 100%; overflow-x: auto;" role="region" aria-label="Recorded Intel Vulkan preview model results" tabindex="0">

| Model                    | Recorded archive result                        |
| ------------------------ | ---------------------------------------------- |
| Qwen 3 0.6B Q8_0 starter | Passed                                         |
| Qwen 3.6 35B-A3B Q4_K_XL | Passed, including one permitted near-tie       |
| Qwen 3.6 27B Q4_K_M      | Passed                                         |
| Qwen 3.5 9B Q4_K_M       | Passed                                         |
| Gemma 4 31B Q4_K_M       | Passed                                         |
| Gemma 4 26B-A4B Q4_K_M   | Failed; corrected diagnostic passes separately |

</div>

The separate 48/48 Gemma diagnostic uses corrected shaders and explicit flags. The source fixes are merged at `10f2141b`, but a fresh packaged-default run still has to demonstrate the complete matrix on one archive. Combining five rows from an old archive with a sixth row from a modified diagnostic would erase the identity of what we tested.

The reference is pinned too: llama.cpp commit `5f436dddb`, using its native Vulkan backend, with executable and library identities recorded. The tests do not relax the near-tie allowance to accommodate the failing path. Cancellation and context/error recovery checks replay reference continuations after the disruptive request, so request history has a chance to expose another bug.

This still leaves actual coding tasks as a separate gate. A server can pass inference and API tests while a coding client fails to edit a workspace correctly. Our [OpenCode validation notes](/zinc/docs/opencode-release-validation/) track that distinction. The Intel diagnostic here is not a claim that those coding tasks passed on this executable.

## Trying the Intel Vulkan path

The [ZINC repository](https://github.com/zolotukhin/zinc) contains the current source and setup instructions. On a Linux machine with the card and a working Intel Vulkan driver, the source-build entry point is:

```sh
./scripts/zig.sh build -Dbackend=vulkan -Doptimize=ReleaseFast
./zig-out/bin/zinc --check
./zig-out/bin/zinc model list
```

The compiler wrapper selects ZINC’s pinned Zig release. Check the detected device, then choose a model and context budget that fit its memory. The [hardware guide](/zinc/docs/hardware-requirements/) and [Intel GPU reference](/zinc/docs/intel-gpu-reference/) provide the maintained details.

Driver support is specific to the GPU. Intel’s [Xe driver hardware table](https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html) lists B70/B65 separately from B60/B50 and B580/B570, including kernel and distribution support. Our recorded matrix used Mesa 26.2.3 on Linux; it does not establish an older-driver minimum. These results concern Linux Vulkan, with Windows and other execution backends outside this test’s scope.

For the other B-series cards, the useful next evidence is the exact card, driver, GGUF, context and packaged executable used in a reproducible check. The shared architecture makes the fixes relevant; each board still needs its own fit and execution evidence.

The [release readiness plan](https://github.com/zolotukhin/zinc/blob/main/docs/releases/0.1.0-plan.md) is maintained in the repository. At this snapshot the release remains unqualified, with the complete Intel archive run and its isolated worker among the remaining gates.

The most valuable result from this work is a narrower set of assumptions: subgroup scope is explicit, expert selection follows the model, approximation requires a choice, and the report identifies the executable that ran. That gives the next Intel optimization a solid place to begin.
