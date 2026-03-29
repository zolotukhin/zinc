---
title: "The Karpathy loop, autoresearch, and the self-improving AI loop behind ZINC"
date: "2026-03-28"
tags:
  - zinc
  - karpathy-loop
  - autoresearch
  - self-improving-loop
  - ai-loop
  - ai-self-improve
  - agentic-coding
keywords:
  - Karpathy autoresearch
  - Karpathy loop
  - self-improving AI loop
  - self-improving loop
  - AI loop
  - AI self-improve
  - autonomous coding loop
  - agentic coding loop
  - overnight optimization loop
  - ZINC optimize_zinc
  - autoresearch vs ZINC
  - GPU verification loop
  - remote GPU coding loop
  - AI research loop
excerpt: "Why Karpathy's autoresearch is viral right now, how a self-improving AI loop actually works, and how ZINC uses a remote GPU verification loop to compress brutal debugging cycles into repeatable overnight search."
---

The same coding agent gave us two completely different outcomes. In one ZINC loop run, it burned **43 cycles and kept nothing**. In the next, with a better controller around it, it kept **40 of 44**. Same repo. Same remote RDNA4 box. Same basic model. What changed was not the agent. What changed was the loop.

That is why [Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch) is spreading so fast. On March 28, 2026, its GitHub page showed **59.4k stars and 8.2k forks**. The repo makes the whole idea legible: give an agent a small but real training harness, make it edit one file, evaluate on a fixed budget, keep the good mutations, and discard the bad ones. The README and [`program.md`](https://github.com/karpathy/autoresearch/blob/master/program.md) reduce "AI self-improve" from a slogan into an operating loop.

We have been building a much harsher version of that idea in ZINC. Our loop lives in [`loops/optimize_zinc.ts`](https://github.com/zolotukhin/zinc/blob/main/loops/optimize_zinc.ts), and the ugly details are in [`writing/LOOP_CHALLENGES.md`](https://github.com/zolotukhin/zinc/blob/main/writing/LOOP_CHALLENGES.md). It does not mutate a single Python file and read back one loss number. It edits Zig and GLSL, syncs them to a remote AMD GPU node, builds on the target machine, runs real inference, checks whether the output got more coherent or more broken, and only advances when the hardware says yes.

That distinction matters. A self-improving AI loop is not mystical recursion. It is a controlled search process with four hard properties: a bounded mutation surface, an evaluator you trust, memory across attempts, and a cheap way to revert bad ideas. When those four properties are present, the development cycle gets much shorter without pretending the underlying engineering got easy.

If you want to read the exact controller, this is the file: [`loops/optimize_zinc.ts` on GitHub](https://github.com/zolotukhin/zinc/blob/main/loops/optimize_zinc.ts).

## Why the bottleneck was never just the GPU

Before the loop, debugging ZINC was expensive in a very specific way. The GPU run itself took time, but the real cost was everything wrapped around it.

ZINC is a native Zig inference engine with Vulkan compute shaders, GGUF loading, quantized kernels, and a Qwen3.5 forward pass that mixes attention, mixture-of-experts routing, and state space model layers. One serious debugging attempt meant editing local code, syncing it to the remote machine, rebuilding, rerunning the model, reading logs, remembering what already failed, deciding whether the change actually helped, and then cleaning up whatever should not survive to the next try.

That is tolerable when the bug is a crash. It is much worse when the engine loads, tokens come out, throughput exists, and the output is still nonsense. That was exactly the phase we were in after the work described in [what broke first in local LLM inference on AMD RDNA4](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4). The tokenizer matched llama.cpp. The embedding path was sane. The DMMV dispatch bug that zeroed most logits was fixed. The model still produced multilingual word soup.

The loop did not make the GPU faster. It made that whole debugging cycle much cheaper to repeat.

<figure class="diagram-card diagram-wide">
  <img src="/blog/zinc-loop-cycle.svg" alt="A simple diagram showing the ZINC loop compressing ship, build, run, diagnose, mutate, verify, keep or revert, and memory into one controlled cycle." loading="lazy" />
  <figcaption>The big win is not that the GPU runs faster. The big win is that the loop owns the dead time between GPU runs: sync, verify, compare, remember, and revert.</figcaption>
</figure>

What the diagram leaves out is the emotional part: loops do not get bored. Humans do. A systems bug that survives five or six careful attempts starts to poison judgment. The loop keeps doing the same disciplined thing long after a person would start improvising.

## How the ZINC loop actually works

At the center of [`optimize_zinc.ts`](https://github.com/zolotukhin/zinc/blob/main/loops/optimize_zinc.ts) is a very plain idea. Every cycle runs the same path: ship code to the RDNA4 node, build, run inference, parse the result, ask the agent for one focused change, verify again, then keep or revert.

In our case, that is not an abstraction. One real cycle looks roughly like this:

1. `rsync` the repo to the remote node, with heavy exclusions like `.git`, `site`, `node_modules`, `.zig-cache`, `zig-out`, and old optimize artifacts, because the loop only needs the inference engine and benchmark code on the box.
2. Run `zig build -Doptimize=ReleaseFast` remotely, then launch ZINC against a fixed prompt, `"The capital of France is"`, under `timeout 60` with `RADV_PERFTEST=coop_matrix` enabled.
3. Serialize the hardware step with `flock /tmp/zinc-gpu.lock` so only one loop touches the GPU at a time, then parse the result into build status, runtime status, tokens generated, decode tok/s, bandwidth, and whether the output looks coherent or like repeated garbage.
4. Commit a local pre-cycle checkpoint, build a prompt from the current diagnosis plus the last 15 cycles, then run the agent and extract `@@@DESCRIPTION`, `@@@SELF_ANALYSIS`, and `@@@NEXT_IDEAS` out of its response.
5. Verify the patch by syncing again, running `zig build test --summary all`, and only then rebuilding and rerunning inference on the remote machine.
6. Keep the change only if reality improved. In fix mode that can mean fewer errors, more generated tokens, or coherent text. In optimize mode it means a real tok/s gain above threshold without degrading output quality. Otherwise the loop hard-resets to the checkpoint.

That is the practical shape of the system. It is less like an autonomous researcher wandering around a codebase and more like a strict harness that forces every change through the same narrow tunnel.

That sounds obvious, but the implementation detail is the point. The loop does not judge code in the abstract. It judges code on the target machine, with serialized access to the GPU so two experiments do not step on each other:

```ts
async function remoteRun(
  modelPath: string,
  prompt: string,
): Promise<{ exitCode: number; output: string }> {
  const runCmd =
    `cd ${REMOTE_ZINC_DIR} && ZINC_DEBUG=1 RADV_PERFTEST=coop_matrix ` +
    `timeout 60 ./zig-out/bin/zinc -m ${modelPath} --prompt "${prompt}" --debug 2>&1`;

  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-p", String(ZINC_PORT),
      "-o", "StrictHostKeyChecking=no",
      `${ZINC_USER}@${ZINC_HOST}`,
      `flock /tmp/zinc-gpu.lock -c '${runCmd.replace(/'/g, "'\\''")}'`,
    ],
    { streamOutput: true, timeout: 180_000 },
  );

  return { exitCode, output: stdout + "\n" + stderr };
}
```

That `flock` call looks like a small detail, but it is part of the design. A real AI loop cannot be sloppy about shared hardware. If multiple agents can all "self-improve" by corrupting the same GPU run, you do not have a loop. You have a race condition with branding.

The other important part is that the loop does not let the agent spray changes everywhere. It asks for one focused mutation, reruns the whole path, and only advances when the result is measurably better. That is the reason the diffs stay reviewable and the branch stays coherent. Even the checkpointing is explicit: before the agent edits anything, the loop makes a local `zinc-loop: pre-cycle-*` commit so a failed idea can be thrown away with one reset instead of a human trying to undo half a diff tree by hand.

The next hard part is prompt construction. A lot of bad AI loops fail because every cycle starts from near-amnesia. ZINC keeps a compact history, carries forward the last self-analysis block, and detects when the output has stalled so the next cycle gets a different search hint instead of another polite no-op:

```ts
const historyBlock =
  cycles.length > 0
    ? cycles
        .slice(-15)
        .map((h) => {
          const snippet = h.outputSnippet ? ` out="${trunc(h.outputSnippet, 30)}"` : "";
          const coherent = h.coherentText ? " ✅COHERENT" : "";
          return `#${h.cycle}: ${h.description} → ${h.kept ? "KEPT" : "REVERTED"}${snippet}${coherent}`;
        })
        .join("\n")
    : "  (none yet)";

const lastAnalysisBlock = lastCycle?.selfAnalysis
  ? `## Last Cycle's Analysis\n${lastCycle.selfAnalysis}`
  : "";
```

That is not glamorous code, but it is one of the main reasons Run 3 behaved so differently from Run 2. The controller stopped making the agent rediscover the repo from scratch every cycle.

## How much the cycle actually shrank

The cleanest proof came from the three major runs in [`LOOP_CHALLENGES.md`](https://github.com/zolotukhin/zinc/blob/main/writing/LOOP_CHALLENGES.md). The first run focused on DMMV performance and kept 9 of 29 cycles. The second run used a weak prompt with poor architectural context and kept 0 of 43. The third run carried full architecture notes, failed approaches, previously fixed bugs, and the agent's own self-analysis, and kept 40 of 44.

That difference is why I take the loop seriously. The step change did not come from swapping models or adding more hype. It came from making the controller more honest about the problem.

<figure class="diagram-card diagram-wide">
  <img src="/blog/zinc-loop-runs.svg" alt="Three clean scorecards showing ZINC loop run 1 at 29 cycles with 9 kept, run 2 at 43 cycles with 0 kept, and run 3 at 44 cycles with 40 kept after better context and memory." loading="lazy" />
  <figcaption>The dramatic shift was not model magic. The same agent became useful once the loop carried forward architecture, failures, and stricter keep-or-revert logic.</figcaption>
</figure>

Run 3 started surfacing bugs that humans often miss because they sit between layers of the system: wave32 subgroup reduction loss, wrong Q4_K sub-block pairing, the shared expert dimension mismatch, conv1d split ordering mistakes, Q5_K layout errors, and attention buffer sizing bugs. Those are not "the AI found a typo" bugs. They are bugs at the boundary between tensor layout, shader assumptions, dispatch code, and model architecture.

That is what a good loop buys you. It removes the human tax between experiments, and it keeps pushing through a search space that would otherwise get abandoned halfway through because every iteration feels too expensive.

## Why this actually counts as self-improvement

I do not think "self-improving" should mean "the agent changed some code and happened to get lucky." In ZINC, the loop improves because the controller gets better at searching.

Each cycle carries forward history: output snippets, failed approaches, whether the text got more coherent, the agent's own `@@@SELF_ANALYSIS`, and ideas tagged as `@@@NEXT_IDEAS`. That state becomes the next prompt:

```ts
const lastCycle = cycles.length > 0 ? cycles[cycles.length - 1] : null;
const lastAnalysisBlock = lastCycle?.selfAnalysis
  ? `## Last Cycle's Analysis (cycle #${lastCycle.cycle})\n${lastCycle.selfAnalysis}`
  : "";

let stall_count = 0;
if (cycles.length >= 3) {
  const currentSnippet =
    lastResult.runOutput.match(/Output text:\s*(.{0,80})/)?.[1]?.trim() ?? "";
  for (let i = cycles.length - 1; i >= Math.max(0, cycles.length - 10); i--) {
    const prev = cycles[i].outputSnippet ?? "";
    if (prev === currentSnippet || !cycles[i].kept) stall_count++;
    else break;
  }
}
```

That is the useful kind of AI self-improve. The base model is not recursively rewriting its own weights. The search process is getting less forgetful and less gullible.

That second part matters just as much. A bad loop accepts churn as progress. A good loop gets stricter as it accumulates evidence. In our case, that meant rejecting no-op changes, detecting stalls, and refusing to keep patches that raised throughput while leaving the output equally broken. Once the loop learned to distinguish "different" from "better," the quality of the search changed.

The keep-or-revert gate is where that becomes concrete. This is the hardest part of the whole controller because it has to reject fake wins:

```ts
if (
  verifyResult.buildExitCode === 0 &&
  verifyResult.runExitCode === 0 &&
  buildRun.buildExitCode === 0 &&
  buildRun.runExitCode === 0
) {
  const oldOut = buildRun.runOutput.match(/Output text:\s*(.+)/)?.[1]?.trim() ?? "";
  const newOut = verifyResult.runOutput.match(/Output text:\s*(.+)/)?.[1]?.trim() ?? "";
  if (newOut !== oldOut && newOut.length > 0) keep = true;
}

if (keep && verifyResult.garbageOutput && !buildRun.garbageOutput) keep = false;
if (keep && buildRun.coherentText && !verifyResult.coherentText) keep = false;
```

That little block is doing a lot of work. It stops the loop from accepting unchanged output as progress, and it blocks speed regressions disguised as correctness regressions. Without this layer, a self-improving loop quickly turns into a commit generator.

One practical trick for anyone building a loop like this is to save the cycle record as if you already know you will need a post-mortem tomorrow. ZINC writes `build.log`, `run.log`, `prompt.md`, `agent_stdout.txt`, and `result.json` for every cycle, plus a resumable `state.json` for the whole run. That is boring infrastructure, but it is the reason we can improve the controller itself instead of only staring at the model code.

## What Karpathy's autoresearch gets exactly right

The reason [`autoresearch`](https://github.com/karpathy/autoresearch) is so compelling is that it makes all of this unusually clean.

According to the README, only three files really matter: `prepare.py`, `train.py`, and `program.md`. The mutation surface is compressed into `train.py`. The human edits `program.md`, not the training code directly. Training always runs for a fixed 5-minute wall-clock budget, and the metric is `val_bpb`, lower is better. [`program.md`](https://github.com/karpathy/autoresearch/blob/master/program.md) then turns that into a literal operating manual: run a baseline first, edit one file, launch the experiment, log the metric, keep improvements, reset worse changes, and keep going.

That is excellent loop design. It is why the repo reads as more than a demo. It is auditable. The diffs are reviewable. The metric is stable enough to compare runs. The human control plane is explicit. Even the README framing is useful here: the repo is small enough that you can understand where the "research org" logic lives instead of pretending the magic sits inside a giant opaque framework.

There is also a deeper lesson in the repo's constraints. `autoresearch` is not trying to solve everything at once. One file to mutate, one GPU, one time budget, one score. That is exactly how you make an AI loop real instead of theatrical.

## Where ZINC is harder than autoresearch

The clean shape of `autoresearch` also explains why ZINC needs more guardrails.

`autoresearch` optimizes inside a mostly stable harness. In ZINC, the harness is part of the problem. The loop is editing Zig dispatch code, GLSL shaders, quantized kernels, and model wiring while also testing on a remote AMD inference node. The evaluator is noisier too. We care about build success, runtime success, token count, coherence, tok/s, bandwidth clues, and whether the output is merely changing or actually getting closer to correct text.

That means the loop controller has to do more work than Karpathy's design. It needs checkpoints, remote hardware locks, prompt memory, stronger revert rules, and enough architectural context to stop the agent from rediscovering solved bugs. That was the entire story of Run 2 versus Run 3. The agent did not suddenly become brilliant. It stopped operating with partial context.

This is also why I think the two projects fit together well in one conversation. `autoresearch` is the clean research-loop template. ZINC is what happens when you take the same idea into a messier systems environment where the evaluator is attached to real hardware and the failures are often silent numerical errors instead of clean crashes.

## How we review the loop day to day

The loop is not fire-and-forget for us. We usually review it every day, and a lot of the gains come from that review process rather than from a single lucky patch.

The first thing we look at is not the kept count. We open the latest run directory under `.zinc_optimize`, read `state.json`, and check the last several `result.json` files in sequence. That tells us the phase, the best tok/s seen so far, which ideas were accumulating, and whether the loop is actually exploring or just bouncing around the same output.

If something looks off, the next stop is `prompt.md`. That is usually where the failure becomes obvious. Sometimes the diagnosis is too vague. Sometimes the loop is carrying too much stale history. Sometimes a known fixed bug is not stated strongly enough, so the agent wastes cycles rediscovering it. We then check `run.log` and `build.log` to make sure the evaluator was not fooled by a partial success, a timeout after prefill, or output that changed cosmetically but not semantically.

Most mornings, the improvement is not "write more model code." It is "make the controller stricter." We tighten a threshold, add one more invariant, promote a repeated idea into the diagnosis, or teach the loop a new pattern to treat as a stall. That is how Run 2 turned into Run 3. The agent did not get better overnight. The review loop around the loop got better.

If you are building your own AI loop, I would copy this part before anything flashy. Save the raw logs, save the prompt that produced the patch, save a structured result file, and make the run resumable. Without that paper trail, you cannot really improve the controller. You can only rerun it.

## What we want the loop to learn next

Right now, I would call ZINC partially self-improving, not fully self-improving. Inside a run, it clearly learns. It carries history forward, detects stalls, keeps ideas, rejects some fake wins, and searches more intelligently than a stateless agent. Across runs and across weeks, though, a lot of the controller improvement still comes from us reviewing the output and editing the harness by hand.

The next step is to make more of that controller learning durable. The obvious move is failure classification: build errors, runtime crashes, repeated-token garbage, low-bandwidth performance stalls, unchanged output, and reference mismatches should each push the loop into a different playbook instead of a single giant prompt. Another step is memory distillation. The loop should be able to mine its own successful fixes, turn them into reusable tactics, and promote them into future prompts automatically instead of waiting for us to rewrite the diagnosis manually.

I also want it to get better at choosing when to change modes. When the loop stalls, it should not just "try harder." It should switch behavior. That could mean entering a reference-check mode, adding CPU-side comparisons automatically, narrowing mutation to one subsystem, or running a targeted microbenchmark instead of a full inference pass. In other words, the loop should learn not only what to change, but what kind of experiment it needs next.

There is a more mechanical layer too. We should be able to auto-summarize every overnight run into one useful morning report, auto-rank ideas by whether they led to kept changes, and gradually build a map from failure type to likely files and likely tactics. That would make the loop more than a repeated agent call. It would make it a controller that accumulates operating knowledge.

## A few tricks that mattered

The most useful tricks in this repo are not especially glamorous. Use a fixed prompt and fixed environment so the score means something. Separate fix mode from optimize mode so correctness bugs do not masquerade as performance work. Checkpoint before every agent edit. Never accept a throughput win that breaks coherence. And when in doubt, bias toward narrower mutations and better logging, not broader search.

## What the viral framing still gets wrong

The viral version of this idea is "the agent improves itself while you sleep." That is catchy, but it blurs the important boundary.

What improves is not some mystical agent essence. What improves is a bounded search process over a measurable environment. Karpathy's repo makes that visible by keeping the environment extremely small and clean. Our ZINC loop makes the same point from the opposite direction. Once the environment gets noisy, the controller starts to matter more than the slogan.

The pattern I expect to survive the current wave of interest around the Karpathy loop, self-improving AI loops, and AI self-improve discourse is much simpler than the hype:

1. Keep the mutation surface small enough to search.
2. Use an evaluator you actually trust.
3. Make bad changes cheap to discard.
4. Carry memory from one cycle into the next.

That is the durable part. In ZINC, it turned a miserable remote GPU debugging workflow into a repeatable overnight optimization loop. It did not replace engineering judgment. It replaced the slowest part of engineering, the repetition between judgments.

If you want the rest of the story around this project, the earlier posts fill in the adjacent pieces: [why we're building ZINC](/blog/2026-03-25-why-we-are-building-zinc), [the home AI rig the loop runs on](/blog/2026-03-26-building-a-local-ai-rig), and [the early forward-pass failures that made the loop necessary](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4). Together they make the same point from different angles. Local AI systems work gets much more interesting once the loop around the model becomes part of the product.
