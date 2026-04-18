---
title: "Qwen3.6-35B-A3B GGUF on AMD and Metal"
date: "2026-04-17"
tags:
  - zinc
  - qwen3-6
  - qwen
  - llm-inference
  - gguf
  - amd
  - rdna4
  - apple-silicon
  - metal
  - vulkan
keywords:
  - Qwen 3.6 ZINC
  - Qwen3.6-35B-A3B
  - Qwen3.6-35B-A3B-UD-Q4_K_XL
  - Qwen3.6 GGUF
  - Qwen3.6 local inference
  - Qwen3.6 AMD RDNA4
  - Qwen3.6 Apple Silicon
  - Qwen3.6 Metal
  - Qwen3.6 Vulkan
  - Qwen3.6 RDNA4
  - Qwen3.6 managed model
  - qwen36-35b-a3b-q4k-xl
  - ZINC Qwen 3.6
faqs:
  - question: "Is Qwen 3.6 supported in ZINC now?"
    answer: "Yes. The managed Qwen 3.6 entry in ZINC is now treated as supported, not experimental. The exact managed id is `qwen36-35b-a3b-q4k-xl`."
  - question: "Which Qwen 3.6 model is generally available in ZINC?"
    answer: "The current generally available local model is `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, exposed through ZINC's managed catalog as `qwen36-35b-a3b-q4k-xl`."
  - question: "Where has Qwen 3.6 been validated in ZINC?"
    answer: "The managed catalog marks it as tested on `amd-rdna4-32gb` and `apple-silicon`, with chat recommended and thinking marked stable."
  - question: "Can I run Qwen3.6-35B-A3B on AMD RDNA4 in ZINC?"
    answer: "Yes. The supported managed Qwen 3.6 entry is validated for `amd-rdna4-32gb`, which is ZINC's current RDNA4 target profile on the Vulkan backend."
  - question: "Does Qwen 3.6 run on Apple Silicon Metal in ZINC?"
    answer: "Yes. The same managed Qwen 3.6 entry is also marked as tested on `apple-silicon`, which maps to ZINC's Metal backend."
excerpt: "Qwen3.6-35B-A3B-UD-Q4_K_XL is now a supported ZINC managed model for AMD RDNA4 Vulkan and Apple Silicon Metal local inference, with one-command pull and chat-ready defaults."
---

If you want the blunt version, here it is: **`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` now runs in ZINC on the two local targets we care about most, AMD RDNA4 through Vulkan and Apple Silicon through Metal.**

Two weeks ago, Qwen 3.6 in ZINC was still a forward-looking architecture post. Today it is one command away from a local prompt:

```bash
./zig-out/bin/zinc model pull qwen36-35b-a3b-q4k-xl
```

That is the moment when a model stops being discourse and becomes software.

The important distinction is that this is not a vague "we think we can support the family eventually" announcement. It is a concrete local release path inside the engine: **Qwen 3.6 is now generally available in ZINC as the managed model `qwen36-35b-a3b-q4k-xl`**, backed by the exact GGUF `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`.

This post is the follow-through to [our earlier Qwen 3.6 architecture piece](/blog/2026-04-05-qwen-3-6-architecture-and-what-it-would-take-to-bring-it-into-zinc). If you want the broader engine context first, the best companions are [how MoE models work in ZINC](/blog/2026-04-04-how-moe-models-work-in-zinc), [every design decision behind ZINC](/blog/2026-04-03-every-design-decision-behind-zinc), and the practical path in [Getting Started](/zinc/docs/getting-started).

That matters because local inference projects get into trouble when they announce support too early. A model is not real support because it once produced a decent answer on one machine. In ZINC, "generally available" means something narrower and more operational:

- it exists in the built-in managed catalog
- it has an exact managed model id and download source
- it is marked as supported rather than experimental
- it is validated for the GPU profiles we actually care about
- it is chat-ready and thinking-stable in the current runtime
- it is covered by the smoke path that launches the real binary

That is the bar Qwen 3.6 now clears in this repo.

On AMD, that matters because most "local inference" conversations still collapse into ROCm assumptions, which leaves consumer RDNA4 owners in the wrong stack entirely. ZINC's path is different: native Vulkan on AMD, native Metal on Apple Silicon, and one managed model surface that spans both. That gives this Qwen3.6 release a much more practical shape than "the weights exist somewhere and a benchmark tweet looked promising."

<figure class="diagram-card diagram-compact">
  <div class="flow-diagram" role="img" aria-label="A flow diagram showing Qwen3.6-35B-A3B support moving from the exact GGUF through the managed catalog entry, supported status, one-command pull flow, and final local use on AMD RDNA4 Vulkan and Apple Silicon Metal through the CLI, server, and chat UI.">
    <div class="flow-main">
      <div class="flow-node">Qwen3.6-35B-A3B GGUF</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node flow-node-accent">Managed id `qwen36-35b-a3b-q4k-xl`</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node">Supported catalog status</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node">`model pull` + `--model-id`</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node flow-node-output">Local CLI, API, and chat</div>
    </div>
    <div class="flow-branches">
      <div class="flow-branch-label">Validated current targets</div>
      <div class="flow-branch-row">
        <span class="flow-pill">RDNA4 32 GB</span>
        <span class="flow-plus">+</span>
        <span class="flow-pill">Apple Silicon</span>
      </div>
    </div>
  </div>
  <figcaption>Qwen 3.6 in ZINC is not just a blog claim. It is a managed path that goes all the way from catalog metadata to a real local prompt surface.</figcaption>
</figure>

## The exact Qwen3.6-35B-A3B model ZINC supports

The supported local model is:

| Field | Value |
| --- | --- |
| Managed id | `qwen36-35b-a3b-q4k-xl` |
| Display name | `Qwen3.6 35B-A3B UD Q4_K_XL` |
| GGUF file | [`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| Family | `qwen3.6` |
| Quantization | `UD-Q4_K_XL` |
| Backends | AMD RDNA4 via Vulkan, Apple Silicon via Metal |
| Default context in ZINC | `4096` tokens |
| Recommended for chat | `true` |
| Thinking stable | `true` |
| Tested profiles | `amd-rdna4-32gb`, `apple-silicon` |

That description is intentionally concrete because Qwen 3.6 can mean different things depending on who is talking.

There is the broader Qwen 3.6 story as a hosted product line and architecture direction. Then there is the specific local open-weight path that matters for a real engine. ZINC is announcing the second one: the local 35B-A3B GGUF you can run through the managed catalog today.

If you read our earlier post, [Qwen 3.6 architecture and local inference in ZINC](/blog/2026-04-05-qwen-3-6-architecture-and-what-it-would-take-to-bring-it-into-zinc), that piece was about what the family implied and what a port might require. This post is the follow-through: a compatible local Qwen 3.6 line exists now, and ZINC has moved it into the supported managed path.

## Why this matters for AMD RDNA4 and Apple Silicon Metal

The easiest way to miss the significance of this release is to treat it like metadata work.

It is not.

Qwen 3.6 is important because it sits right on the boundary between "interesting architecture discourse" and "a model people will actually try to use for daily work." It is a modern sparse model family, it carries real coding expectations, and it arrives with the kind of momentum that makes users immediately ask the only question that matters:

Can I run it locally on the machine I already own?

For ZINC, the answer is now yes, with the exact local path made explicit.

That answer lands differently on the two platforms ZINC is built around:

- on AMD, it means a real local Qwen3.6 story on RDNA4 without collapsing into "just use ROCm" advice that does not fit consumer cards
- on Apple Silicon, it means the same Qwen3.6 GGUF is part of the native Metal path instead of being treated as a side experiment

That is the practical SEO-shaped query users are actually typing: can I run Qwen3.6 on AMD, and can I run Qwen3.6 on Metal?

That answer also says something useful about the engine. It means the existing hybrid Qwen runtime was not a one-off hack for Qwen 3.5. The catalog wiring, chat behavior, smoke coverage, and managed install flow were all close enough to absorb this Qwen 3.6 local release without inventing a separate product lane around it.

That is the part I care about most. Good inference engines do not just chase whatever the last benchmark thread got excited about. They build a runtime that can absorb adjacent model generations without turning each one into a new fork of reality.

## Why ZINC can support this Qwen3.6 GGUF cleanly

One of the more interesting implementation details is that the current Qwen 3.6 family entry in ZINC reuses the existing Qwen 3.5 GGUF architecture mapping internally.

That sounds small, but it is why this announcement is possible without pretending we solved a completely new backend research problem overnight.

In practice, the managed Qwen 3.6 path already benefits from:

- the hybrid Qwen decode machinery already used for the 35B-A3B class
- managed-model catalog support with exact fit metadata
- model-path matching for both managed-cache and raw-file loads
- chat-facing thinking support marked stable in the catalog
- smoke coverage that launches the real CLI against the actual GGUF
- both major local targets we care about: RDNA4 on Vulkan and Apple Silicon on Metal

If you want the platform-specific background behind those two targets, the best references are [RDNA4 tuning](/zinc/docs/rdna4-tuning), [the Apple Silicon reference](/zinc/docs/apple-silicon-reference), and [bringing ZINC to Apple Silicon](/blog/2026-04-01-bringing-zinc-to-apple-silicon).

That is a much better place to launch from than "we have a draft branch that kind of works if you know which flags not to touch."

It also lets us be honest about the shape of the support. This is not an everything-everywhere Qwen 3.6 announcement. It is a specific, validated, local GGUF path inside the current ZINC engine.

## Run Qwen3.6-35B-A3B on AMD and Metal

If you want the shortest path from announcement to prompt, this is it. If you want the full operational detail around managed models, model switching, and server behavior, the companion docs are [Getting Started](/zinc/docs/getting-started), [Running ZINC](/zinc/docs/running-zinc), and the [API reference](/zinc/docs/api).

On a local ZINC build:

```bash
cd /path/to/zinc
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc model pull qwen36-35b-a3b-q4k-xl
./zig-out/bin/zinc --model-id qwen36-35b-a3b-q4k-xl --prompt "What changed in Qwen 3.6?" --chat
```

On RDNA4 Linux, keep the normal Vulkan environment hint:

```bash
export RADV_PERFTEST=coop_matrix
./zig-out/bin/zinc --model-id qwen36-35b-a3b-q4k-xl --prompt "Write a concise Zig code review checklist." --chat
```

If you want the server path instead of the one-shot CLI:

```bash
./zig-out/bin/zinc --model-id qwen36-35b-a3b-q4k-xl --port 9090
```

Then point your OpenAI-compatible client at `http://127.0.0.1:9090/v1`.

That is part of what makes this a real release in ZINC. The model is not trapped inside one benchmark harness. It is reachable through the managed install path, the CLI path, the server path, and the built-in chat flow.

It also makes the competitive positioning clearer. If you are searching for a **Qwen3.6 AMD GPU** path, ZINC is solving a different problem than stacks that assume ROCm first and consumer RDNA support second. If you are searching for **Qwen3.6 Metal** or **Qwen3.6 Apple Silicon**, the point is the same: this is a native Metal runtime path, not a placeholder mention in a roadmap.

## What "generally available" does not mean

This announcement is stronger than "experimental," but it is not marketing fiction.

A few things are still worth saying plainly:

- ZINC as a whole is still an experimental engine under active development
- the current managed default context for this model is still `4096`, not the biggest number associated with the broader Qwen 3.6 family
- this post is about the local 35B-A3B GGUF path, not every hosted Qwen 3.6 capability
- generation in the current server is still serialized, so multi-request latency behavior is not the same thing as a fully scheduled batching runtime

None of those caveats weaken the release. They make it credible.

The point of general availability is not to pretend the engine is finished. The point is that the model has crossed from "interesting if you are willing to babysit it" into "officially supported path in the product surface we ship."

That is a much more useful threshold.

## Why I wanted this announcement to be boring in the best way

There is a flashy version of this post where I would say Qwen 3.6 is here, the future arrived, and every local engine should panic.

That version is less interesting than the real one.

The real win is that the announcement is almost boring:

- the model has a managed id
- the model has a tested profile story
- the model has a supported status
- the model has a predictable pull-and-run workflow
- the model fits into the same ZINC shape users already know

That kind of boring is what software should optimize for.

If you are building a local inference engine, every new model family is a chance to discover whether your architecture is composable or fragile. Qwen 3.6 is a good outcome for ZINC because the answer here was not "drop everything and invent a new stack." The answer was "the engine is coherent enough that this model can graduate into the supported lane."

That is the right kind of progress.

## The short version

Qwen 3.6 is now generally available in ZINC as the managed model `qwen36-35b-a3b-q4k-xl`.

It is the exact local GGUF path `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, it is marked supported in the managed catalog, it is validated on RDNA4 and Apple Silicon, and it plugs into the same CLI, server, and chat surfaces as the rest of the current ZINC model line.

That is what local model support is supposed to look like.

If you want the background on the runtime pieces that make this possible, the most relevant reads inside the ZINC blog are [the earlier Qwen 3.6 architecture post](/blog/2026-04-05-qwen-3-6-architecture-and-what-it-would-take-to-bring-it-into-zinc), [how MoE models work in ZINC](/blog/2026-04-04-how-moe-models-work-in-zinc), [bringing ZINC to Apple Silicon](/blog/2026-04-01-bringing-zinc-to-apple-silicon), and [every design decision behind ZINC](/blog/2026-04-03-every-design-decision-behind-zinc). If you just want to run the model, start with [Getting Started](/zinc/docs/getting-started) and [Running ZINC](/zinc/docs/running-zinc).
