---
title: "Why JSON-constrained decode no longer scans a 151k-token bitmask"
date: "2026-05-03"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - structured-output
  - constrained-decoding
  - xgrammar
  - llguidance
  - outlines
  - json-schema
  - llm-inference
  - qwen3
keywords:
  - constrained decoding RDNA4
  - JSON schema local LLM inference
  - XGrammar adaptive token mask cache
  - llguidance Earley parser tokens
  - Outlines FSM token mask
  - llama.cpp GBNF grammar overhead
  - Qwen3 151936 vocabulary mask
  - structured output local inference
  - vocab bitmask per step decode
  - 32 GB RDNA4 JSON tool calling
excerpt: "Constrained JSON decoding used to add a millisecond-class tax to every step of a local Qwen3 generation, because the engine had to scan a 151,936-bit logit mask against a grammar before each sample. XGrammar and llguidance moved that work off the hot path with adaptive caches and a Rust Earley parser that runs in tens of microseconds. On a bandwidth-bound RDNA4 decode loop the change turns a noisy ten percent overhead into something a profiler will not flag, but only if the engine actually adopts the new path."
---

A 200-token JSON response out of [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) on the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) takes about 6.7 seconds without a grammar constraint. The same 200 tokens through llama.cpp's GBNF JSON-schema constraint take roughly 7.9 seconds on the same machine. The grammar did not change the model. It did not change the prompt. It did not change the kernel choice on any of the linear layers or the attention path. It added 200 small CPU-side checks that each scanned a 151,936-entry logit mask against the current parser state and zeroed every disallowed token. The bitmask is the bill.

That bill is what XGrammar and llguidance retired. Both projects landed inside the last eighteen months, both are integrated into the major server-side engines, and both are still missing from the local-engine trees most people use on a single 32 GB workstation card. The reason to care, at a workstation seat, is that constrained generation is the entry point for tool-calling and agentic loops, and the constrained-generation tax used to scale with vocabulary size. The Qwen3 family ships a 151,936-token vocabulary, four to five times the size of the Llama-2 vocabulary the original constrained-decoding implementations were tuned against. The tax got worse before it got better.

This post is the structural reason the per-step grammar mask used to cost one to ten milliseconds, the architectural shift that took it down to under fifty microseconds, and what that means for a local inference engine that runs decode at thirty tokens per second on a bandwidth-bound R9700.

## What the per-step grammar mask actually does

A constrained-decoding engine wraps every sampling step with one extra job: given the current parser state, produce a vector of booleans the same length as the model's vocabulary, where each entry says whether sampling that token would still leave the partial output parseable. The standard implementation in [llama.cpp's GBNF path](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) walks the grammar's stacked rules against every candidate token in the vocabulary and writes a logit penalty of negative infinity wherever the rule cannot accept the token's bytes. The check is correct. It is also linear in the vocabulary, which on Qwen3 means 151,936 grammar evaluations per decoded token even when the parser state only has three or four valid continuations.

The early Outlines implementation took a different shape. The [Outlines core blog post](https://huggingface.co/blog/outlines-core) describes the original design as a per-state mask precomputed against the tokenizer once, with the lookup at sample time reduced to a tensor index. That removed the per-token grammar walk from the hot path but moved the cost into the precompilation step, which scales with the cross product of grammar states and vocabulary entries. On large grammars over a 152k-token vocabulary the precompile became the bottleneck. The early Outlines releases routinely took tens of seconds to compile a moderately deep JSON schema, and the cached mask sometimes accidentally got reused across tokenizers in ways that quietly corrupted output, as the project's own [issue tracker for the state-mapping cache](https://github.com/dottxt-ai/outlines/issues/872) documents.

The mask, in either approach, is the same shape. It is a 151,936-bit vector that gets ANDed against the logit vector before sampling. What differs is where the cost lives: per-step grammar walking in the GBNF case, or per-state precompilation in the original Outlines case. Both implementations end up writing essentially the same mask for the same parser state. Both implementations spend the bulk of their time on entries the mask will never need.

## What the mask actually contains

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-03-vocab-bitmask-context-split.svg" alt="A two-row diagram comparing the naive per-step vocab bitmask used by llama.cpp's legacy GBNF path against the adaptive token mask cache used by XGrammar. The top row shows a single horizontal strip 151,936 cells wide, mostly red, with a sparse scattering of green ticks representing the few-dozen grammar-valid tokens at one mid-decode JSON parser state. A label below reads cost per step: O(vocab) grammar walk, four to eight milliseconds on a 152k-token vocab. The bottom row shows the same 151,936-cell strip split into two regions: a long dark grey region labeled context-independent mask cache, hit by a single state-keyed lookup, and a short red region on the right labeled context-dependent tokens, resolved per step by walking the pushdown automaton against the small remaining set, with a per-step cost of thirty to fifty microseconds independent of vocab size." loading="lazy" />
  <figcaption>Two views of the same vocab-sized mask. In the legacy path every entry has to be checked against the grammar on every step. In the adaptive-cache path the bulk of the entries resolve from a precomputed lookup keyed by the parser state, and only a small remainder needs runtime evaluation.</figcaption>
</figure>

What is uncomfortable about the mask is how empty it tends to be. At a typical mid-decode JSON parser state, the set of valid next tokens is in the dozens. A state that expects a property name allows perhaps fifty tokens that begin with a quote. A state that expects a colon allows a single token. A state that expects the closing brace of an object allows a handful of whitespace tokens plus the brace itself. The grammar's constraints are nearly always sparse against a vocabulary the size of Qwen3's. The dense bitmask was the wrong representation; the work it forced on the engine was almost entirely on entries the parser would have rejected anyway.

The exception is the state inside an open string literal, where the grammar accepts any token whose bytes do not contain an unescaped quote. That set covers most of the vocabulary, and the grammar walk is genuinely linear in vocab size for the duration of that string. This is the case the adaptive-cache approach has to handle without falling back to the slow path.

## What XGrammar and llguidance actually changed

The [XGrammar paper](https://arxiv.org/abs/2411.15100) calls the central data structure an adaptive token mask cache. The trick is to split the vocabulary into the context-independent set, where the validity of each token is fully determined by the parser state, and the context-dependent set, where the token's validity depends on the byte-level content of what was sampled before. The context-independent partition is precomputed once per grammar plus tokenizer pair, keyed by the parser's pushdown-automaton state, and stored as a bitset. The context-dependent partition is small and is evaluated on demand using the pushdown automaton.

The numbers in the paper are concrete. XGrammar reports up to one hundred times speedup on context-free grammar evaluation against the prior baselines and up to three times on JSON Schema specifically. The end-to-end serving overhead for structured generation drops to under forty microseconds per token on JSON Schema and CFG paths. The library has shipped inside [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) as the default constrained-decoding backend, replacing the older Outlines integration in both engines.

[llguidance from the guidance-ai project](https://github.com/guidance-ai/llguidance) takes a different route to a similar number. Instead of a precomputed cache, it runs an Earley parser on top of a lexer built from regular-expression derivatives, and uses a slicer optimization to avoid evaluating the parser against tokens whose bytes cannot possibly match the next lexer state. The library's [own benchmark notes](https://guidance-ai.github.io/llguidance/llg-go-brrr) report average mask computation under fifty microseconds per token for a 128k-class tokenizer, with less than one percent of masks taking over one millisecond and a 0.001 percent tail under thirty milliseconds. There is no precompile step, which is the deciding factor for agentic workloads where the grammar changes on every tool call.

The implementations differ in where the work happens but agree on the shape of the answer. The vocab-sized scan was avoidable because the grammar's actual state space is small. Both libraries take the parser state seriously enough to avoid touching most of the mask on most steps.

## What the wall-time math looks like on RDNA4 decode

The bandwidth-bound floor for a Qwen3-30B-A3B decode step on the R9700 sits around thirty-three milliseconds per token at the Q4_K_M quantization the engine ships. The active-weight read is roughly two gigabytes, the KV cache reads scale with the resident context, and the [16k crossover post](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode) walked through how the KV term overtakes the weight term once the resident context grows past sixteen thousand tokens. Anything the engine adds on the CPU side has to be measured against that thirty-three-millisecond budget.

| Constrained-decoding backend | Per-token mask time | As share of 33 ms decode | Notes |
| --- | ---: | ---: | --- |
| llama.cpp GBNF (legacy walk) | 4 to 8 ms | 12 to 24 percent | Vocab-linear per step on the parser stack |
| Outlines pre-XGrammar (FSM cache) | 0.6 to 1.2 ms | 2 to 4 percent | Precompile expensive, hot path is one tensor index |
| XGrammar (adaptive mask cache) | 30 to 50 us | under 0.2 percent | Precompile cached per grammar plus tokenizer |
| llguidance (Earley plus slicer) | 40 to 60 us | under 0.2 percent | No precompile, per-call grammar swaps are free |

The two upper rows used to be the entire menu. They are also the two rows still in the local-engine trees most workstation users run today. The two lower rows are the rows the server-side engines have already converged on. The gap between them is between five percent and twenty-five percent of decode wall time on a workstation card, and the cost is paid on every constrained step, not just the cold first call.

The point that matters for an engine that targets local single-card decode is that the savings only show up on workloads that actually use a grammar, and most local workloads still do not. The pressure to adopt one of the two new backends comes from tool-calling and agent loops, where every model response is constrained and the per-token tax compounds across thousands of decoded steps in a session.

## The shape of the port for a local engine

A Vulkan or Metal local inference engine does not usually have to write the constrained-decoding logic itself. Both XGrammar and llguidance ship as libraries with thin C ABIs that the engine calls between the logits-out and the sampler. The integration is structurally identical to wiring in a new logit warper. What is real work is the engine's ownership of the tokenizer: the adaptive cache is keyed by the tokenizer's full vocabulary, and the cache is invalidated if the tokenizer is rebuilt with a different special-token configuration. For an engine that ships its own [Hugging Face tokenizer](https://github.com/huggingface/tokenizers) bridge, that means an explicit lifecycle for the cache that mirrors the tokenizer's, and a fingerprint that the cache can compare against on load.

The other piece that needs care is the contract with the kernel. The mask comes back as a bitset over the vocabulary; the engine has to apply it before the temperature and top-p sampler runs on the GPU. On a Vulkan backend the cleanest path is to upload the bitset as a small storage buffer and apply it in the same compute shader that does the temperature scale, which keeps the per-step round-trip to a single submit. On Metal the same shape works through a tiny argument buffer. Neither integration is hard. Both are still on the list.

The downside, named clearly, is that a precompiled mask cache is not free in memory. A 151,936-bit mask is nineteen kilobytes per state, and a deep JSON grammar can have on the order of ten thousand states once flattened, putting the cached mask near two hundred megabytes. That is a small fraction of a 32 GB card but a real fraction of a 16 GB one, and the engine has to be willing to spend it. llguidance's no-precompile approach trades the upfront memory for slightly higher worst-case per-token latency, which is the right trade for an agentic workload that rebuilds the grammar on every turn.

## What we are watching

Two things on the horizon are worth tracking from a local-inference seat. The first is whether the [XGrammar 2 work](https://arxiv.org/abs/2601.04426) on dynamic per-call grammars closes the gap with llguidance on agentic loops where the grammar changes per tool call. The second is whether the next round of open-weight models keeps growing vocabulary the way Qwen3 and gpt-oss have, which would push the linear-in-vocab cost of the legacy backends from a noticeable tax into something that genuinely competes with the bandwidth-bound decode floor for wall time. Either direction makes the case for moving the constrained-decoding path off the legacy walk and onto a cached or sliced parser more pressing.

For now, the practical answer is that JSON-constrained decode on a workstation R9700 does not have to scan a 151,936-bit mask per step, and the two libraries that proved it are sitting on the shelf waiting to be wired in. The hardest part of the port is the cache-lifecycle plumbing around the tokenizer. The wall-time win on a constrained workload is between five and twenty-five percent of decode time, and it shows up on every step. That is the kind of clean, model-agnostic, kernel-agnostic win that local inference engines should be willing to take whenever it lands on the doorstep.
