---
title: "Constrained decoding makes a local agent scan the vocabulary twice per token"
seoTitle: "Grammar-Constrained Decoding Overhead on a Local GPU: The Per-Token Vocabulary Mask"
date: "2026-07-22"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - constrained-decoding
  - grammar
  - gbnf
  - xgrammar
  - structured-output
  - tool-calling
  - agents
  - local-llm
  - llm-inference
keywords:
  - grammar constrained decoding overhead
  - structured output local LLM
  - XGrammar token mask cache
  - llama.cpp GBNF JSON schema grammar
  - vocabulary mask per decode token
  - tool call JSON constrained decoding
  - context-independent tokens vocabulary
  - RX 9070 XT decode sampler cost
  - Outlines finite state machine index
  - agent tool schema grammar RDNA4
excerpt: "An agent's tool call is only useful if it parses as valid JSON, and grammar-constrained decoding is what guarantees that. The catch on a local card is that the naive version re-examines all 151,936 tokens of the Qwen3.5-9B vocabulary at every decode step, a second CPU pass the size of the LM head, sitting on the critical path of a token that is already too slow."
seoDescription: "Grammar-constrained decoding forces every agent tool call to be valid JSON by masking the logits each step. The naive implementation, like llama.cpp's grammar sampler, walks all 151,936 tokens of the Qwen3.5-9B vocabulary per token on the CPU, an overhead on the order of the 1.5 ms LM head that ZINC already measures. XGrammar's fix splits the vocabulary into context-independent tokens that are precomputed into a mask cache and a small context-dependent set checked at runtime, cutting the per-token cost to near zero. In a local agent swarm the eight agents usually share one tool schema, so they share one compiled grammar and one mask cache."
faqs:
  - question: "Why does a local agent need constrained decoding at all?"
    answer: "Because a tool call is only actionable if it parses. If the model emits a stray brace, an unquoted key, or an argument that is not in the schema, the harness cannot execute the call and the whole decode turn is wasted. Grammar-constrained decoding guarantees the output matches the tool schema by masking every token that would break it, so the JSON is valid by construction rather than by luck."
  - question: "What does the grammar mask actually cost per token?"
    answer: "The naive implementation checks the grammar against every token in the vocabulary at every step. For Qwen3.5-9B that is 151,936 tokens per token generated, done on the CPU while the GPU waits. That pass is the same size as the LM head, which ZINC measures at about 1.5 ms of a 25.2 ms decode token, so a naive grammar mask can rival the LM head and it sits squarely on the critical path."
  - question: "How does XGrammar make it near-zero?"
    answer: "XGrammar splits the vocabulary into context-independent tokens, whose validity does not depend on the grammar's current stack state and can be precomputed into a mask cache when the grammar compiles, and a small set of context-dependent tokens that still need a runtime check. Most of the 151,936 tokens fall in the cached set, so the per-token work collapses from a full-vocabulary walk to a small check plus a bitmask lookup. The paper reports up to a 100x speedup and near-zero end-to-end overhead."
  - question: "Does the swarm change the grammar cost?"
    answer: "It helps. Eight coding agents launched from the same harness usually share the same tool schemas, which means they compile to the same grammar and can share one mask cache. The expensive part, building the grammar and its context-independent cache, is paid once and amortized across every agent, the same way the shared system prompt is stored once instead of eight times."
  - question: "Is constrained decoding free correctness?"
    answer: "It guarantees syntactic validity, not semantic correctness. A grammar can force well-formed JSON that matches the schema, but it cannot make the model pass the right file path or a sensible argument value. It also cannot fix a grammar that is too permissive. The win is that the class of failure where a turn is thrown away because the JSON did not parse disappears, which for an agent loop is a large class."
draft: false
---

An agent's tool call is worth exactly nothing if it does not parse. The model can pick the right tool and reason its way to the right arguments, and if the closing brace lands in the wrong place or a key comes out unquoted, the harness rejects the call and the entire decode turn is wasted. On a single local card that is not a rounding error. It is tens of seconds of decode thrown away because one of a few hundred tokens was structurally wrong.

Grammar-constrained decoding is the standard fix, and it is a good one. You describe the legal output as a grammar, usually derived from the tool's JSON schema, and at every step the sampler masks out every token that would violate it. The model literally cannot emit a broken tool call, because the tokens that would break it have been set to negative infinity before sampling. The JSON is valid by construction.

The part that gets glossed over is what that mask costs. To decide which tokens are legal right now, the naive implementation checks the grammar against every token in the vocabulary, one at a time, on the CPU, while the GPU sits idle waiting for the next input. For Qwen3.5-9B that is 151,936 checks per token generated. It is a second pass over the whole vocabulary, the same width as the [LM head](/blog/2026-05-16-what-qwen3-151k-lmhead-costs-on-rdna4-decode/) that the GPU just finished, and it lands directly on the critical path of a decode token that is [already too slow](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/).

## The tool schema is the grammar

Start with why an agent needs this at all. A tool is defined by a JSON schema: a name, a set of arguments, their types, which ones are required. When the model wants to call it, it has to produce a JSON object that matches. Free-form generation gets this right most of the time and wrong often enough to matter, and "often enough to matter" is fatal in a loop where a rejected call means the agent stalls or retries.

llama.cpp handles this with GBNF, its own grammar format, and it will convert a JSON schema into a grammar for you. The [GBNF guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) shows the machinery: hand it a schema with a `name` string and an `age` integer, and it emits production rules that force a leading brace, then the quoted key `"name"`, then a colon, then a string, and so on, down to the ranges of digits an integer may contain. Any token that does not fit the current position in that grammar is masked. The output cannot be anything but a matching object.

This is the right default for agents, and both major serving stacks ship it. The approach goes back to reformulating generation as walking a finite-state machine, which is the framing in Willard and Louf's [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702), the paper behind the Outlines library. Their insight was that you can build an index over the vocabulary keyed by FSM state, so guiding generation with a regex or grammar "adds little overhead to the token sequence generation process." That last clause is the whole game, and whether it holds depends entirely on how the mask is computed.

## The mask is the size of the LM head

Here is the cost the FSM framing is trying to avoid. A context-free grammar, the kind you need for nested JSON, is not a flat state machine. Executing it means tracking a stack, and to decide whether a given token is legal you may have to walk several stack states. The XGrammar authors put the problem plainly in their [paper](https://arxiv.org/abs/2411.15100): executing a context-free grammar "requires going through several stack states over all tokens in vocabulary during runtime, bringing non-negligible overhead for structured generation."

Over all tokens in the vocabulary. That is the phrase that matters on a local card. For every single token the agent decodes, the naive grammar engine iterates the entire 151,936-token vocabulary and, for each token, asks the grammar whether it is currently allowed. llama.cpp does exactly this: its grammar sampler walks the candidate set and sets the logit to negative infinity for any token that would violate the current grammar state. The [GBNF guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) even links its own performance issue, noting that grammars "currently have performance gotchas."

Put that next to what a decode token already costs. On one RX 9070 XT, a Qwen3.5-9B decode token is about 25.2 ms, and the LM head over the 151k-row output embedding plus sampling is roughly [1.5 ms of it](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/). The LM head is a single dense pass over the vocabulary on the GPU. A naive grammar mask is a single pass over the same vocabulary on the CPU, with per-token stack work instead of a dot product. It is the same shape and the same width, running on slower silicon, and it cannot overlap the GPU because the GPU needs its result before it can sample. When people say constrained decoding "slowed everything down," this is what they hit.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-22-grammar-token-mask-vocab-split.svg" alt="A two-panel diagram on a dark indigo background. The left panel, titled 'Every decode step, the grammar masks the whole vocabulary', shows the 151,936-token Qwen3.5-9B vocabulary drawn as a long horizontal band of small cells. The model is mid tool call, having emitted the characters open-brace quote, so the grammar is in a state expecting a JSON key. Almost the entire band is shaded dark grey and labelled 'forbidden, logit set to minus infinity'; a thin slice of about a dozen cells near the left is gold and labelled 'legal now: the tokens that continue a quoted key'. A bracket under the whole band reads 'the naive engine asks the grammar about all 151,936 tokens, one at a time, on the CPU'. The right panel, titled 'Modeled grammar cost per decode token, RX 9070 XT', is four horizontal bars against a faint vertical line marked '25.2 ms decode token'. From top: 'LM head plus sampling, measured' in slate at 1.5 milliseconds; 'naive full-vocabulary walk' in coral at 2.4 milliseconds, labelled 'checks all 151,936 tokens'; 'XGrammar context-split' in mint at 0.05 milliseconds, labelled 'most tokens cached, small set checked live'; and a thin reference bar 'grammar compile, once per schema' in dashed gold at 9 milliseconds with a note that it is paid once and shared across agents, off the per-token path. A caption strip reads 'context-independent tokens are precomputed into a mask cache; only context-dependent tokens are walked at runtime'." loading="lazy" />
  <figcaption>Modeled per-token grammar cost for Qwen3.5-9B on one RX 9070 XT. The LM head figure is measured; the mask bars are a model pinned to the 151,936-token vocabulary width and the up-to-100x speedup XGrammar reports. The naive walk rivals the LM head; the context-split version disappears into the noise.</figcaption>
</figure>

The bars are a model, not a profiler trace, and the compile bar is deliberately off the per-token axis because it is paid once. The point does not rest on whether the naive walk is 2.4 ms or 1.8. It rests on the walk being the width of the vocabulary, which is arithmetic, and the vocabulary being the same one the LM head just scanned, which is the observation. A per-token cost on the order of the LM head, hidden inside the sampler, is worth removing.

## Most of the vocabulary never changes its mind

The fix is to notice that the naive walk asks the same question 151,936 times when it only needs to ask it a few hundred. XGrammar's central move is to split the vocabulary into two classes. Context-independent tokens are ones whose legality does not depend on the grammar's current stack state at all. A raw byte in the middle of a long string, a token that can never appear inside a JSON structure, a token that is always illegal here: its answer is fixed the moment the grammar is compiled. Context-dependent tokens are the small remainder whose legality genuinely turns on where you are in the parse.

The context-independent tokens, which are the overwhelming majority, get precomputed into a mask cache at grammar-compile time. At decode time the engine looks them up instead of re-deriving them, and only the small context-dependent set is walked live. XGrammar layers a persistent stack to make those live checks cheap and overlaps the grammar computation with the GPU. The measured result is up to a 100x speedup over the walk-everything approach and, once folded into a serving engine, "near-zero overhead structure generation."

The table is the same story the FSM index tells, from the other direction. Outlines precomputes a per-state map from FSM state to the set of allowed tokens, so guiding generation is a lookup rather than a scan. XGrammar generalizes that to the stack machine a context-free grammar needs. Either way the expensive work moves off the per-token path and onto a one-time compile.

| Per token, Qwen3.5-9B decode | Tokens examined at runtime | Where it runs | Modeled cost |
| --- | ---: | --- | ---: |
| LM head + sampling (measured) | 151,936 | GPU | 1.5 ms |
| Naive grammar walk | 151,936 | CPU | ~2.4 ms |
| XGrammar context-split | a few hundred | CPU | ~0.05 ms |

Read the middle row against the top. The naive grammar walk does the same amount of vocabulary work as the LM head, off the GPU, on the critical path, every token. The bottom row is what happens when you refuse to redo the settled part of that work on every step. The compile cost that buys the cache is real, on the order of milliseconds per schema, but it is paid once when the agent's tool set is registered, not once per token.

## The swarm makes the cache cheaper, not more expensive

There is a version of this cost that looks worse at first for a local agent swarm and turns out to be better. Eight agents decoding at once means eight grammar states advancing in parallel, which sounds like eight times the masking work. It is, per token, but the expensive part is not the per-token check. It is the compile, and the eight agents launched from one harness almost always share the same tool schemas.

Same schemas mean the same grammar, which means one compiled grammar and one context-independent mask cache serving all eight agents. This is the same shape as the [shared system prompt](/blog/2026-07-21-the-system-prompt-a-local-agent-swarm-caches-eight-times-over/): the thing every agent has in common gets built once and reused, instead of paid for N times. The per-token context-dependent checks stay per-agent because each agent is at a different point in its own JSON, but those are the cheap part once the cache exists. The costly compile amortizes across the whole swarm.

That reframes constrained decoding for a local box. On a server it is a feature you weigh against throughput. On a single-user card running an agent loop it is closer to mandatory, because the alternative is a decode turn that does not parse, and it is affordable precisely when it is naive-free, because the swarm shares the one artifact that is expensive to build.

## What I am building toward

ZINC does not have a grammar engine yet, and the shape of the one it needs is now clear. It has to convert a tool schema to a grammar the way llama.cpp does, and then it has to mask like XGrammar rather than like the naive sampler, because a full-vocabulary CPU walk per token would give back a good chunk of the [decode overhead](/blog/2026-07-17-decode-kernel-fusion-on-rdna4-deletes-the-barrier-not-the-launch/) the last few posts have been trying to claw out of the token. The context-independent split is the whole difference between a mask that costs an LM head and a mask that costs nothing.

The swarm angle is the part that makes it worth doing well rather than doing minimally. Compile the grammar once per distinct tool set, share the mask cache across every agent that uses it, and keep only the tiny context-dependent check per sequence. Structured that way, guaranteeing that every one of eight agents emits valid JSON on every tool call costs a one-time compile and a bitmask lookup per token. The correctness is free by construction. The trick is making the correctness cheap by construction too.
