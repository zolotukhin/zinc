---
title: "How we moved ZINC from 7 tok/s to 33 tok/s on AMD RDNA4"
date: "2026-03-30"
tags:
  - zinc
  - vulkan
  - rdna4
  - performance
  - llm-inference
  - qwen3-5
  - moe
  - ssm
  - local-llm-inference
keywords:
  - ZINC performance
  - AMD RDNA4 LLM inference
  - Vulkan inference engine optimization
  - Qwen3.5-35B-A3B performance
  - GPU argmax optimization
  - MoE routing on GPU
  - local LLM tok/s optimization
  - Vulkan descriptor overhead
  - GPU bandwidth utilization
  - ReleaseFast benchmarking
excerpt: "ZINC did not get from 7 tok/s to 33 tok/s because of one magical kernel. The path was a mix of real runtime changes, better GPU residency for MoE and sampling, and fixing the benchmark itself so we stopped measuring debug-heavy and contended runs as if they were the real product."
---

When people hear that an inference engine went from roughly **7 tok/s** to more than **33 tok/s**, they usually imagine one giant breakthrough. A better matrix multiply kernel. A heroic shader rewrite. A driver bug fix. Something dramatic and singular.

That is not what happened here.

The honest story is more interesting than that.

ZINC got faster because we did three different kinds of work at the same time:

1. we removed real decode-path overhead from the runtime,
2. we stopped dragging data back to the CPU for operations that belonged on the GPU,
3. and we fixed the benchmark methodology so we were no longer mistaking a debug-heavy, sometimes contended path for the actual throughput baseline.

That last part matters more than most performance write-ups admit. Some of the old `7–8 tok/s` numbers were real. Some were artifacts of how we were measuring. The final `33.58 tok/s` baseline is real too, but it is a **plain decode** number on a clean `ReleaseFast` build, not a thinking-enabled chat number.

This post is the detailed version of that timeline.

## The concrete code changes that mattered most

Before getting into the timeline, it is worth being explicit about the actual source-level changes that carried most of the gain.

These were the important ones:

- MoE routing stopped being "GPU produce logits, CPU pick experts, GPU run experts" and became a GPU-resident path built around `softmax_topk.comp`, `dmmv_*_moe.comp`, and `moe_weighted_acc.comp`.
- Expert execution stopped being serialized as lots of tiny per-expert dispatches and became batched expert work inside the same layer pass.
- Greedy sampling stopped copying the full logits buffer back to the CPU every token and switched to GPU argmax plus a 4-byte token readback.
- Prefill stopped collecting output on every prompt token and only kept output on the last prompt token.
- The benchmark path stopped treating debug-heavy runs as the truth and standardized on `zig build -Doptimize=ReleaseFast` plus the clean no-debug decode path.

If you prefer to think in terms of files rather than abstract changes, the center of gravity was:

- `src/compute/forward.zig`
- `src/compute/elementwise.zig`
- `src/compute/argmax.zig`
- `src/shaders/softmax_topk.comp`
- `src/shaders/moe_weighted_acc.comp`
- `src/shaders/dmmv_*_moe.comp`
- `loops/optimize_zinc.ts`

## The starting point was already a partial recovery

The popular shorthand inside the repo became "we were stuck at 7 tok/s," but even that compresses a lot of context.

By the time we started the March 29 optimization run, ZINC had already climbed from an earlier **~4.3 tok/s** stage into the **5.8-7.6 tok/s** range. The forward pass was coherent, the GPU SSM path was working, and the engine was no longer obviously broken. The remaining problem was that decode still looked far too serialized for RDNA4.

One of the loop prompts captured the state of the world pretty clearly:

- coherent output
- about `5.8 tok/s` on the measured run
- roughly `42 submits/token`
- modeled bandwidth utilization around `0.5-0.7%`
- a strong suspicion that the GPU was not compute-bound so much as synchronization-bound

At that point, the working theory was not crazy. The engine was still paying a per-layer tax around MoE routing and expert dispatch. That was the first real bottleneck we attacked.

## The first genuine speedup was keeping MoE routing on the GPU

The biggest early decode win was not in the LM head and not in attention. It was in the Mixture-of-Experts path.

At the start of that phase, the decode loop still effectively did this for every layer:

1. run router work on the GPU,
2. read expert IDs back to the CPU,
3. let the CPU decide which experts to dispatch,
4. submit more GPU work for the selected experts.

On a 40-layer MoE-heavy model, that architecture is poison for single-token latency. The actual tensor math might be fine, but the queue/host overhead compounds.

The first optimization cycle that stuck did one important thing: it eliminated the host-visible expert-ID handoff and let the GPU-side routing output feed the GPU-side expert work directly.

That change moved the measured loop result from about **5.8 tok/s** to **7.33 tok/s**.

What changed structurally:

- MoE expert IDs stopped being a per-layer CPU rendezvous point.
- The decode path stopped needing a mid-pass submit/wait for every layer just to learn which experts to run.
- The engine got much closer to "record the work once, keep the token on device, and only touch host-visible memory at the true output boundary."

That was the first proof that the `7 tok/s` story was not just "RDNA4 is slow" or "our DMMV kernels are bad." The engine had been spending too much time on choreography.

At source level, the decision boundary changed from "read router output back and let the CPU decide" to "if the GPU MoE path is available, keep the whole routing-and-expert chain on device":

```zig
const use_gpu_moe = self.dmmv.moePipelineForType(gate_quant) != null and
    self.dmmv.moePipelineForType(down_quant) != null and
    self.elementwise.pipeline_softmax_topk != null and
    self.elementwise.pipeline_moe_weighted_acc != null;

if (use_gpu_moe) {
    // softmax_topk writes expert_ids + weights to router_output_buf
    try self.elementwise.recordSoftmaxTopk(&self.decode_cmd, ds, config.n_experts, n_used);
    self.decode_cmd.computeBarrier();
    // ... expert work stays on GPU from here ...
} else {
    // CPU fallback: readback router logits, CPU softmax+topk
}
```

That looks like a small branch, but architecturally it removed one of the ugliest recurring sync points in decode.

## The second real speedup was batching expert work instead of dribbling it out

The next cycle that stuck moved the measured result from **7.33 tok/s** to **7.85 tok/s**.

That gain came from a different problem in the same neighborhood.

Even after routing stayed on the GPU, the expert path was still too fragmented. Small per-expert DMMV dispatches on RDNA4 meant poor occupancy, too many barriers, and too much repeated descriptor churn for work that shared the same input vector.

The second change batched all 8 selected experts into the same dispatch regime instead of serializing them as a chain of tiny GPU jobs.

In practical terms, that meant:

- more workgroups in flight per expert phase,
- fewer barriers per MoE layer,
- fewer descriptor set allocations and updates,
- and less time spent making the GPU look busy in theory while actually underfilling the machine.

The numerical gain looked small compared to the eventual 33 tok/s baseline, but this was still a real throughput win. More importantly, it taught us that ZINC's decode bottleneck was strongly shaped by **medium and small dispatch structure**, not just by the largest tensor in the model.

The batched expert path is the kind of change that matters more than its line count suggests. Instead of launching gate/up/down work expert by expert, the decode loop now launches all selected experts together:

```zig
// === GPU MoE path: BATCHED expert dispatch — all experts in parallel ===
// All 8 experts' gate/up/down DMMVs run as Y workgroups in a single dispatch.
// This gives ~8× better GPU utilization vs serial per-expert dispatch.
// Reduces dispatches from 32 to 5, barriers from 32 to 4 per MoE layer.

try self.dmmv.recordMoeDispatch(&self.decode_cmd, qt, ds, inter_dim, hidden_dim, expert_gate_row_bytes, n_used, 0, 0, 0);
try self.dmmv.recordMoeDispatch(&self.decode_cmd, qt, ds, inter_dim, hidden_dim, expert_gate_row_bytes, n_used, 0, 0, 0);
try self.elementwise.recordSwiglu(&self.decode_cmd, ds, n_used * inter_dim);
try self.dmmv.recordMoeDispatch(&self.decode_cmd, qt, ds, hidden_dim, inter_dim, expert_down_row_bytes, n_used, inter_dim, 0, 0);
try self.elementwise.recordMoeWeightedAcc(&self.decode_cmd, ds, hidden_dim, n_used, hidden_dim);
```

That was one of the key places where the speedup was not "better math" so much as "less fragmented work."

## Then we flatlined, and the flatline turned out to be partly fake

After those two wins, the loop spent a long time hovering around `7.8-8.0 tok/s`.

If you only looked at the topline result, the natural conclusion was: "the engine is still slow, and the remaining work must be deeper kernel tuning."

That conclusion was incomplete.

What the plateau actually revealed was a measurement problem:

- the optimization loop was benchmarking the `--debug` path,
- the profiling path was intrusive enough to distort the runtime,
- the printed bandwidth number was only modeling the LM-head tail, not full-token decode traffic,
- the loop had acceptance rules that rejected many small positive improvements,
- and some runs that really finished still got classified as failures because the wrapper timed out or misread the process exit.

In other words, we were trying to optimize a moving target while staring at the wrong dashboard.

This is one of the least glamorous but most important parts of the whole story. The engine was not going to get 4x faster until the benchmark stopped lying.

## The benchmark was measuring the wrong binary and the wrong mode

One of the first clean measurement passes broke the old narrative immediately.

When we separated the modes and measured them directly, the same codebase produced very different numbers:

- clean run: `10.76 tok/s`
- `--debug`: `8.89 tok/s`
- `--profile --debug`: `5.52 tok/s`

That one comparison changed the shape of the work.

It told us:

- the loop had been optimizing a slower path than the one users actually care about,
- `--profile` was not trustworthy as an apples-to-apples scoreboard,
- and the "we are still at 7 tok/s" claim was already stale once the debug tax was removed from the measurement path.

This is why we later standardized performance work around:

```bash
zig build -Doptimize=ReleaseFast
RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --prompt "The capital of France is"
```

That build choice turned out to be part of the real performance story, not just a documentation cleanup. Measuring `zig build` and then claiming the result as the engine baseline was simply not fair.

The loop itself now uses that same shape remotely:

```ts
async function remoteBuild(): Promise<{ exitCode: number; output: string }> {
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-p", String(ZINC_PORT),
      `${ZINC_USER}@${ZINC_HOST}`,
      `cd ${REMOTE_ZINC_DIR} && zig build -Doptimize=ReleaseFast 2>&1`,
    ],
  );
}

async function remoteRun(modelPath: string, prompt: string) {
  const runCmd =
    `cd ${REMOTE_ZINC_DIR} && RADV_PERFTEST=coop_matrix timeout 90 ` +
    `./zig-out/bin/zinc -m ${modelPath} --prompt "${prompt}" 2>&1`;
}
```

That part of the story is not glamorous, but it is critical. A performance loop that benchmarks the wrong mode will confidently optimize the wrong system.

## We also had to stop benchmarking on a dirty node

The other benchmarking problem was operational rather than architectural.

At one point the RDNA4 box had multiple `zinc` and `llama` processes running on it at the same time. That contaminated throughput numbers badly enough that it changed the whole interpretation of the engine.

Once we explicitly stopped the competing processes and re-ran the same current checkout on an idle node, the clean result jumped to:

- **22.14 tok/s**
- repeat run **22.21 tok/s**
- profiled run **22.21 tok/s**

That same measurement pass also showed:

- profiled GPU decode time around `43.87 ms/token`
- end-to-end decode time around `45.0 ms/token`

That gap is small. It means by that stage, host overhead was no longer dominating the token the way it had in the earlier `7 tok/s` world.

This was a huge interpretive milestone.

It meant:

- the engine was no longer primarily dying from CPU submission overhead,
- the remaining time was mostly real GPU-side decode work,
- and a big chunk of the apparent `7 tok/s` wall had been node contention, not just runtime architecture.

If you write performance software, this is the part worth underlining: sometimes the optimization is correct, but the machine is dirty. If you do not control the node, the graph lies.

## The tail of decode still had a CPU-shaped hole in it

Even after the node cleanup and the earlier MoE wins, the token tail still had an obvious structural flaw.

The fast path was doing all the hard model work on the GPU and then ending with:

1. final norm,
2. LM head,
3. full logits readback,
4. CPU argmax.

That is a bad deal for greedy sampling. You do not need the whole logits vector on the host just to learn the winning token ID.

So the next meaningful runtime change was to move argmax fully onto the GPU and reduce the output handoff from "copy back the whole vocabulary" to "copy back four bytes."

That changed the decode tail in two important ways:

- the normal greedy path now reads back a single token ID instead of a full logits buffer,
- and full logits staging became opt-in for debug and diagnostics instead of mandatory on every token.

This did not create the whole 4x speedup by itself, but it removed one of the last obviously wasteful host-visible boundaries in the normal decode path.

The deeper architectural point is that this was not just "argmax is faster on GPU." The real win was that the engine could finally treat the output boundary like a scalar result instead of a giant debugging artifact.

The decode tail now looks like this:

```zig
const use_gpu_argmax = collect_output and self.argmax.pipeline != null and self.argmax_descriptor_set != null;
if (use_gpu_argmax) {
    self.decode_cmd.computeBarrier();
    try self.argmax.record(
        &self.decode_cmd,
        self.argmax_descriptor_set.?,
        self.model.config.vocab_size,
        self.argmax_phase0_workgroups,
    );
}

const need_logits_readback = collect_output and
    (self.logits_readback_enabled or self.validation_diagnostics_enabled or !use_gpu_argmax);

if (use_gpu_argmax) {
    const token_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @sizeOf(u32) };
    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.argmax_result_buf.handle, self.argmax_result_staging.handle, 1, &token_region);
}
```

The important change is not just the argmax shader. It is that the default fast path now reads back **one `u32`** instead of an entire vocabulary-sized logits buffer.

## We also stopped doing extra work during prefill

Another fix was smaller in isolation but important for end-to-end honesty.

During prefill, the engine had been collecting output work more often than necessary. The current path only asks for output on the **last prompt token**, and then the first generated token is sampled from those existing logits instead of re-running the last prompt token through decode.

That matters because the first output token after prefill is easy to mishandle:

- duplicate the last prompt token's work,
- disturb the cache/state progression,
- and pay extra latency for something the model already computed.

The cleaned-up flow is simpler:

1. prefill all prompt tokens,
2. collect output only on the final prompt step,
3. sample the first generated token from those logits,
4. continue normal decode one token at a time from there.

This was not the headline win, but it was part of removing the kinds of hidden inefficiencies that keep turning "the engine feels slower than it should" into a weeks-long mystery.

The prefill-side implementation is now deliberately narrow:

```zig
for (prompt_tokens, 0..) |token_id, i| {
    const collect_output = i + 1 == prompt_tokens.len;
    try self.decodeStep(state, token_id, collect_output);
}
```

That `collect_output = i + 1 == prompt_tokens.len` detail matters. It means prefill still builds context and caches exactly as before, but only the last prompt token pays the output-side cost.

## The graph report fixed our bandwidth story

Another important change was analytical rather than directly executable.

The old bandwidth line in the runtime looked precise, but it was not modeling full-token decode. It was effectively dominated by the tail:

- final RMS norm,
- LM head,
- logits readback.

That made the printed "effective GB/s" number easy to misread as overall memory-bandwidth utilization, which in turn made the GPU look absurdly underused for reasons we could not quite explain.

Once the decode graph report became model-aware, we finally had a better whole-token estimate:

- roughly **3.35 GB/token** for the 35B-A3B decode graph

Using that model:

- `22.2 tok/s` implied about **74.4 GB/s**
- `33.58 tok/s` implied about **112.5 GB/s**

On a `576 GB/s` RDNA4 card, that is only about **19.5%** of peak.

That sounds low until you understand what it means.

It does **not** mean the engine is obviously broken.

It means single-stream decode on this workload is not a pure DRAM streaming benchmark. The limiting factors are:

- graph depth,
- medium/small kernel regimes,
- launch/setup structure,
- and the fact that one sequence at a time simply does not create enough parallel pressure to saturate the card's external memory bandwidth.

That realization mattered a lot, because it changed the optimization target from:

"Why are we not using 100% of 576 GB/s?"

to:

"How do we minimize single-token latency, and when do we need concurrency or batching to drive higher aggregate utilization?"

Those are very different engineering questions.

## The last big jump came from measuring the release path as the product

By the end of the cleanup, the current baseline on the idle RDNA4 node was no longer a 20-ish tok/s story either.

Once we made `ReleaseFast` the standard build for performance work and deployment docs, the clean baseline moved again:

- **33.58 tok/s** on the CLI plain decode path
- **33.55 tok/s** on raw `/v1/completions`
- **33.98 tok/s** aggregate on raw `/v1/completions` at `concurrency=4`

Those are the numbers that now belong in the README, not the old `7-16 tok/s` figures from debug-heavy loop runs.

There is one important caveat, and it should be said every time:

- `33.58 tok/s` is a **plain decode** number
- it is **not** a thinking-enabled or reasoning-chat number

The current reasoning-chat path for the 35B model lands around:

- **24.94-28.56 tok/s**

That gap is real and still worth optimizing. But it is a different problem from the original decode bottleneck.

## So what actually produced the 4x improvement?

If you compress the whole timeline down to the causes that mattered most, the answer looks like this.

### 1. We removed real host round trips from the decode loop

The two earliest wins were structural:

- MoE routing stayed on the GPU
- expert work got batched instead of serialized as tiny per-expert dispatches

Those were real engine improvements, not benchmark tricks.

### 2. We removed waste at the output boundary

Greedy sampling stopped copying full logits to the CPU just to discover one token ID.

That mattered because the tail of decode runs every token. A small inefficiency there is a permanent tax.

### 3. We fixed prefill/output sequencing

Only the final prompt token now collects output, and the first generated token is sampled from the logits already produced by prefill.

That removed unnecessary duplicate work and made the decode path cleaner to reason about.

### 4. We stopped benchmarking debug mode as if it were production

This was one of the most consequential changes even though it was not a new kernel:

- clean runs were much faster than `--debug`
- intrusive profiling distorted throughput
- `ReleaseFast` mattered

The old loop was partly measuring instrumentation overhead.

### 5. We stopped benchmarking on a contended machine

The clean-node reruns were the moment the engine's actual state became visible.

What looked like a stubborn `7-8 tok/s` ceiling turned out to contain a lot of shared-node contamination.

### 6. We fixed the mental model of bandwidth

The old LM-head-centric metric made the whole engine look stranger than it was.

Once we had a better full-token byte model, the numbers made more sense:

- single-stream decode at `33.58 tok/s` is fast,
- `112.5 GB/s` modeled bandwidth is plausible,
- and "not saturating 576 GB/s" is not a contradiction.

## The short version is that the engine got faster and the benchmark got honest

That is the most compact truthful summary I know.

The runtime really did improve:

- fewer host round trips
- fewer fragmented expert dispatches
- GPU-side argmax
- less unnecessary output work

But the measurement story also improved:

- `ReleaseFast` instead of generic builds
- clean decode instead of `--debug`
- idle node instead of a box full of competing inference processes
- full-token modeled bandwidth instead of a misleading tail-only metric

If you ignore the measurement half of that story, you end up telling a fake optimization narrative. If you ignore the runtime half, you end up claiming the whole gain was just benchmark hygiene. Neither is true.

It was both.

## Where we ended up

For the `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` path on the RDNA4 test node, the current practical baseline is:

- **33.58 tok/s** plain CLI decode
- **33.55 tok/s** raw HTTP plain decode
- **24.94-28.56 tok/s** reasoning-style chat

That is more than a **4x improvement** over the old `~7 tok/s` headline.

It is also a better engine than the old number suggests, because the current system is not just faster. It is structurally healthier:

- more GPU-resident,
- less dependent on host-visible synchronization,
- less confusing to profile,
- and much easier to benchmark honestly.

The next challenge is no longer "can ZINC escape 7 tok/s?"

The next challenge is:

- closing the gap between raw decode and reasoning chat,
- reducing hot-path Vulkan binding churn further,
- and using concurrency or batching when the actual goal is higher aggregate GPU utilization rather than lower single-token latency.

That is a much better class of problem to have.
