---
title: "One RDNA4 verify pass has slack for about two dozen speculative tokens"
seoTitle: "Token-Tree Verification Turns Prompt Lookup's Free Draft Into a Bigger Local Decode Win on RDNA4"
date: "2026-07-25"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - agents
  - speculative-decoding
  - prompt-lookup
  - token-tree
  - tree-attention
  - decode
  - kv-cache
  - local-llm
  - llm-inference
keywords:
  - token tree speculative decoding local LLM
  - tree attention verification RDNA4
  - prompt lookup multiple matches token tree
  - SpecInfer Medusa EAGLE token tree
  - verification pass cost decode step RX 9070 XT
  - how many speculative tokens per forward pass
  - coding agent duplicate lines speculative decode
  - lossless tree verification local inference
  - Qwen3.5-9B speculative decode budget
  - ZINC prompt lookup token tree
excerpt: "Prompt lookup gives a local coding agent a free draft, but it bets everything on the single most recent match in the context. A verification pass on one RX 9070 XT is priced like one decode step and has room for roughly two dozen candidate tokens, so the engine can stop betting: propose several matches at once as a token tree and let one pass confirm whichever branch the model actually wanted."
seoDescription: "Yesterday's prompt lookup proposes one continuation, the most recent n-gram match, and verifies it in a single forward pass. Because a decode step on one RX 9070 XT running Qwen3.5-9B streams the weights once and pays a fixed launch cost, a verification pass over many candidate positions costs about the same as one token until the tree crosses a modeled knee near 24 tokens. That slack is enough to spend on a token tree, the SpecInfer, Medusa, and EAGLE-2 idea, so instead of guessing which duplicate line in the context the model will copy, the engine proposes several matches as branches and a single pass confirms the right one. For a coding agent whose context holds many near-identical lines, and for a swarm bottlenecked on forward passes, hedging across matches is close to free."
faqs:
  - question: "What is a token tree in speculative decoding?"
    answer: "It is a set of candidate continuations arranged as a tree that share a common prefix, verified together in one forward pass using a tree-structured attention mask. Instead of proposing a single linear block of guessed tokens, you propose several branches, and the attention mask lets each branch attend only to its own ancestors so one pass scores all of them at once. The model then accepts the longest branch it agrees with. SpecInfer introduced tree-based verification, and Medusa and EAGLE-2 build draft trees on top of the model itself. It is lossless: the output is identical to normal decoding."
  - question: "Why can one verification pass check many tokens for about the price of one?"
    answer: "Because decode on a local card is bound by weight streaming and per-step launch overhead, not arithmetic. A decode step on one RX 9070 XT running Qwen3.5-9B lands about 39.6 tokens per second, roughly 25 ms, and most of that is streaming the weights once and paying the fixed cost of launching the step. A verification pass over many candidate positions streams those same weights once and pays that launch once, so it stays near 25 ms until the extra compute for the candidates catches up with the weight stream, which the roofline puts near 24 tokens for this model."
  - question: "How does a token tree help prompt lookup specifically?"
    answer: "Prompt lookup drafts by searching the context for where the last few tokens appeared before and proposing what followed. In a coding context the same short n-gram often appears in several places, near-identical lines, repeated identifiers, similar call sites, so there are several plausible continuations and the current implementation just takes the most recent one. A token tree lets the engine propose several of those matches as branches and confirm the correct one in a single pass, instead of betting on the most recent match and re-decoding when it is wrong."
  - question: "How big should the tree be on a single RX 9070 XT?"
    answer: "About two dozen candidate tokens total, spread across a handful of branches. The modeled knee for Qwen3.5-9B on one RX 9070 XT is near 24 tokens, where the compute for the candidates stops hiding under the weight stream and the pass starts to cost more than one decode step. Below that budget the tree is close to free; above it each extra token is priced at the compute-bound prefill rate. So a sensible shape is three or four branches of depth six to eight, not one deep branch or a wide bushy tree."
  - question: "Does token-tree verification change the model's output?"
    answer: "No. Like plain speculative decoding and prompt lookup, tree verification is lossless. Every candidate token is only accepted if the model itself would have produced it, so the final text is exactly what standard decoding would emit. The tree only changes how many forward passes it takes to get there. SpecInfer, Medusa, and EAGLE-2 all preserve the output distribution while cutting the number of decode steps."
draft: false
---

Yesterday's post left a bet on the table. [Prompt lookup decoding](/blog/2026-07-24-a-coding-agents-speculative-draft-is-the-context-it-already-read-back/) turns a coding agent's own context into a free draft model: take the last few tokens the model emitted, find where that same n-gram appeared earlier in the context, and propose whatever followed it as a candidate block for the next forward pass to confirm. It works because an agent copies most of its output from the file it just read. But it makes one quiet assumption, and that assumption is where the value leaks out.

When the last n-gram appears in several places, prompt lookup takes the most recent match. In prose that is fine, because text rarely repeats. In code it is a bad bet. A coding agent's context is full of near-identical lines: the same import written three ways, a variable name that shows up in a dozen call sites, a function signature echoed in its definition, its call, and its test. The most recent match is one of several plausible continuations, and picking it means the engine is guessing which duplicate the model is about to copy, then paying full price when it guesses wrong.

The fix is to stop guessing. You do not have to pick one match if a single forward pass has room to check several of them at once, and on a local card it does. A verification pass on one RX 9070 XT is priced like one decode step, and one decode step has slack for about two dozen candidate tokens before it costs any more. That slack is exactly enough to hedge.

## Why a verification pass is priced like one decode step

The reason runs through the last two weeks of posts. A decode step on one RX 9070 XT running Qwen3.5-9B lands about [39.6 tokens per second](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/), roughly 25 milliseconds a token, and almost none of that is arithmetic. It goes to streaming the model's weights across the bus once and to the fixed overhead of launching the step: the [command submission](/blog/2026-07-16-command-buffer-reuse-is-rdna4s-version-of-the-cuda-graph-decode-win/) and the [barrier between kernels](/blog/2026-07-17-decode-kernel-fusion-on-rdna4-deletes-the-barrier-not-the-launch/) that a single token pays in full.

A verification pass processes several candidate positions at once, but it streams those same weights once and pays that same launch once, for all of them. The only thing that grows with the number of candidates is the matrix work, and decode is not compute bound. So the pass stays flat, near the cost of one token, until the compute for the candidates finally catches up with the time spent streaming the weights. Past that point the pass is doing real extra arithmetic and starts to cost more.

That crossover is a roofline, and it has a number.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-25-rdna4-token-tree-verify-budget.svg" alt="A line chart on a deep teal-charcoal background titled 'One RDNA4 verify pass has slack for about two dozen speculative tokens'. The horizontal axis is candidate tokens verified in one forward pass, the token-tree size, from 0 to 48. The vertical axis is verification pass wall clock in milliseconds from 0 to 80. A coral curve stays flat at about 25 milliseconds from 0 up to a knee near 24 tokens, then rises linearly to about 50 milliseconds at 48 tokens. A soft mint-green band shades the region from 0 to 24 tokens, labelled 'free budget, one pass approximately one decode step'. A grey dashed horizontal line at 25.3 milliseconds is labelled 'one decode step, 39.6 tok/s'. An amber dashed line rising from the origin is labelled 'compute bound, 962 tok/s prefill'; the coral curve tracks it after the knee. A dot marks the knee at about 24 tokens, annotated 'tree stops being free past here'. A small box reads '24 tokens, one pass: about 25 ms; 24 tokens, one at a time: about 606 ms; about 24 fewer launches for the same tokens'. A footnote explains the roofline model: a pass streams the weights once, the decode floor, and does compute proportional to tree size, and below the knee the compute hides under the weight stream." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B on one RX 9070 XT from two measured rates: 39.6 tok/s decode sets the flat floor, 962 tok/s prefill sets the sloped compute bound. A single verification pass costs about one decode step until the candidate count crosses a knee near 24 tokens, after which each extra token is priced at the compute-bound rate.</figcaption>
</figure>

Read the flat part first. From one candidate up to about twenty-four, the pass costs roughly what a single decode token costs, because the candidates ride along under the weight stream that was going to happen anyway. The knee is where the compute for the tree grows past the weight-stream time, computed simply as the 25.3 millisecond decode floor divided by the 1.04 milliseconds of compute each candidate adds at the prefill rate. After the knee the curve joins the compute-bound line and the free ride is over.

The box on the right is the whole argument in three numbers. Confirming twenty-four tokens in one pass costs about 25 milliseconds. Generating those same twenty-four tokens one at a time costs twenty-four decode steps, about 606 milliseconds. Same tokens, roughly one twenty-fourth the launches, if you can arrange for the model to actually accept them.

## Spend the budget on a tree, not a longer guess

Prompt lookup spends that budget on depth. It proposes one continuation and makes it a bit longer. But a longer single guess only pays when acceptance is high the whole way down, and yesterday's curve showed that the tail of a long block is where acceptance falls off. Spending all twenty-four tokens on one branch means most of them are speculative continuations of a continuation, and the deeper ones rarely land.

A [token tree](https://arxiv.org/abs/2305.09781) spends the same budget on breadth. Instead of one block, you propose several candidate continuations that share a prefix and arrange them as a tree, then verify the whole tree in one pass using a tree-structured attention mask so each branch attends only to its own ancestors. SpecInfer introduced this tree-based verification, and [Medusa](https://arxiv.org/abs/2401.10774) made it mainstream by generating the branches from extra decoding heads on the model itself and confirming them with the same masked pass. The model scores every branch at once and accepts the longest one it agrees with. It is lossless, exactly as prompt lookup is, because a branch is only taken where the model would have produced it anyway.

For prompt lookup the branches are free to find. The context already contains several places the current n-gram appeared, and each one proposes a different continuation. Rather than throwing away all but the most recent, the engine keeps the top few matches and makes each a branch. One of them is usually the line the model is about to copy, and the tree lets a single pass confirm which.

| One verify pass, the ~24-token budget on RX 9070 XT | Candidate tokens | What it covers |
| --- | ---: | --- |
| Linear prompt lookup, most recent match, depth 8 | 8 | one continuation, right only if the guess was right |
| Deeper linear block, depth 24 | 24 | one continuation, tail rarely accepted |
| Token tree, 3 matches at depth 8 | 24 | three continuations, one pass picks the right one |

The three rows spend the same forward pass. The first uses a quarter of the budget and bets it all on the most recent match. The second uses the whole budget on depth and wastes most of it on low-acceptance tail tokens. The third uses the whole budget on breadth, so when the model was going to copy the second-most-recent line instead of the last one, the tree already has that branch and confirms it in the pass that would otherwise have been thrown away.

## Where the tree earns its keep, and where it does not

The honest limit is the knee. Past roughly two dozen candidate tokens the pass is no longer free, so a tree cannot be both wide and deep on a single card. That rules out the big bushy trees some server-side systems build, and it means the sensible shape here is a handful of branches at modest depth, three or four matches at depth six to eight, not sixteen branches or one branch of length forty. [EAGLE-2](https://arxiv.org/abs/2406.16858) makes the same point from the other direction: the acceptance rate of a draft token is context dependent, so a static tree wastes positions, and the win comes from shaping the tree to where acceptance is actually high. For prompt lookup that shaping is cheap, because the number of times an n-gram recurs in the context is a decent proxy for which branches are worth keeping.

There is also bookkeeping the linear version does not have. A tree needs the masked attention, and it needs the KV cache to hold the branches without corrupting the committed context, which means writing candidate keys and values to scratch slots and keeping only the accepted branch. On a card where [KV space is the constraint](/blog/2026-07-23-a-local-coding-agent-reads-back-eighteen-tokens-for-every-one-it-writes/), that scratch has to be small and reused, so the tree budget is bounded by memory as well as by the compute knee. Two dozen tokens is comfortably inside both.

For a swarm the direction is the same as it was for plain prompt lookup, only stronger. [Eight agents on one card](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) are bottlenecked on forward passes, because every decode step streams the shared weights, so anything that confirms more accepted tokens per pass is spending the scarce resource better. A tree that accepts the right branch where a linear guess would have missed is a whole re-decode the scheduler never has to run. The extra compute per pass is close to free on a bandwidth-bound card, which is precisely the trade a swarm wants: more arithmetic that was going to waste anyway, in exchange for fewer launches.

## What I am building toward

ZINC already keeps every agent's context resident as [KV cache](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/), and yesterday's plan was to add an n-gram index over each agent's own token stream. The tree changes what that index returns. Instead of the single most recent match, it returns the top few matches, ranked by how often the n-gram recurs, and the engine assembles them into a small tree that fits inside the two-dozen-token budget. The verification step then needs a tree-structured mask folded into the decode kernel and a scratch region in the KV cache for the branches.

The framing I want to keep is that the budget is the whole point. A local card gives you one nearly-free verification pass per token, and that pass has room for about two dozen candidates. Prompt lookup was leaving most of that room empty by betting on a single match. Filling it with a tree costs nothing the card was not already spending, and it turns the moments a coding agent's context is ambiguous, several near-identical lines it might copy, from a wrong guess and a re-decode into one pass that quietly picks the right line.
