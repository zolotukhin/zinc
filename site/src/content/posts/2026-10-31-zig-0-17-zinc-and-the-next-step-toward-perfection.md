---
title: 'Zig 0.17, ZINC, and the next step toward perfection'
seoTitle: 'Zig 0.17 in Practice: What Building ZINC Taught Us'
date: '2026-10-31'
tags:
  - zinc
  - zig
  - systems-programming
  - llm-inference
  - performance
keywords:
  - Zig 0.17 examples
  - Zig 0.17 migration
  - Zig std.Io
  - Zig process.Init
  - Zig allocators
  - Zig GPU inference
  - ZINC inference engine
excerpt: 'A faster tokenizer, a borrowed slice, a vectored network write and a GPU assumption: real ZINC code shows why Zig 0.17 feels like another step toward a better systems language.'
seoDescription: 'Zig 0.17 through real ZINC examples: process-provided I/O, explicit allocation, vectored streaming, GPU contracts and a more inspectable build system.'
faqs:
  - question: 'Did Zig 0.17 alone make ZINC 37.7% faster?'
    answer: 'The complete migration reduced long-prompt first-visible-output latency by 37.7% on the measured Gemma 26B-A4B workload. That includes allocator, tokenizer and network changes. The compiler-only port was neutral on GPU decode; this is not a compiler-only speedup or an across-the-board throughput claim.'
  - question: 'Are std.Io and process.Init new in Zig 0.17?'
    answer: 'They arrived in Zig 0.16. ZINC adopted them while moving from 0.15.2 to 0.17. Changes specific to 0.17 include the build-system redesign, Build Server Protocol and progress on incremental compilation for x86_64 Linux.'
  - question: 'Does ZINC now write its Vulkan shaders in Zig?'
    answer: 'Its production Vulkan path still compiles GLSL to SPIR-V with glslc. Zig implements host orchestration and the build graph. Improvements to Zig’s own SPIR-V backend do not automatically change ZINC’s shader pipeline.'
draft: false
---

The most satisfying performance improvement in our Zig 0.17 migration began with a paragraph of text.

ZINC is a local LLM inference engine. We spend plenty of time thinking about GPU memory traffic, attention kernels and quantized matrix multiplication. Yet a user can wait on the CPU before the GPU gets anything useful to do. In this case, our tokenizer kept searching for the next pair of symbols to merge and allocating strings along the way.

Moving from Zig 0.15.2 to 0.17 made us revisit that work. The complete migration eventually took Gemma 26B-A4B’s long-prompt wait for first visible output from **1,917.1 ms to 1,194.2 ms**, a **37.7% reduction**, on our Radeon AI PRO R9700. The GPU shaders were unchanged.

That result includes application changes. The compiler-only port was neutral on GPU decode. The interesting part is how naturally Zig lets you follow a cost from an API down to the memory and operations responsible for it.

For me, that is what makes 0.17 exciting. The language and toolchain keep moving toward a program whose behavior you can explain by reading its parts. In ZINC, the parts include an allocator, an I/O implementation, a few borrowed slices and the exact capabilities of a GPU.

Perfection is an ambitious word for software that still has bugs. I mean it as a direction: fewer assumptions hidden between what the code says and what the machine does.

## A main function that tells you what it needs

Here is the opening of ZINC’s [main function](https://github.com/zolotukhin/zinc/blob/c1d2b7b416cfb407116cc3f7abd67f818468783b/src/main.zig#L2819), with the rest of the body omitted:

```zig
pub fn main(process_init: std.process.Init) !void {
    host_io.init(process_init);
    const allocator = process_init.gpa;

    const args = try host_io.argsAlloc(allocator);
    defer host_io.argsFree(allocator, args);

    // Parse arguments and run the selected command.
}
```

The process supplies the general-purpose allocator and the I/O implementation. Argument storage has an owner and a visible cleanup point. The `try` preserves the possibility that allocating those arguments fails.

These ideas have different ages. Explicit allocators and `defer` are longstanding Zig features; `std.Io` and the process-provided initialization object arrived in [Zig 0.16](https://ziglang.org/download/0.16.0/release-notes.html#IO-as-an-Interface). We adopted that design on the way from 0.15.2 to 0.17.

The I/O change matters because opening a file, waiting on a socket and coordinating tasks now share an explicit interface. The application can choose how those operations execute. A library can accept that choice instead of quietly constructing its own runtime.

ZINC currently installs the process’s I/O implementation in a small shared [host adapter](https://github.com/zolotukhin/zinc/blob/c1d2b7b416cfb407116cc3f7abd67f818468783b/src/host_io.zig). Tests use `std.testing.io`. We still have a shared global there; passing the context through more of the code is a useful next cleanup. The migration gives us a place to see that dependency and improve it.

## A streamed token is also a network operation

Consider what happens after the model produces text. A browser receives it through a server-sent event, wrapped in HTTP chunk framing. One event contains several pieces: a chunk length, the `data: ` prefix, JSON and the trailing separators.

Those pieces already exist. Concatenating them would allocate or copy another buffer; writing each one separately would create more individual write calls. ZINC’s [SSE implementation](https://github.com/zolotukhin/zinc/blob/c1d2b7b416cfb407116cc3f7abd67f818468783b/src/server/http.zig#L206) assembles slice descriptors instead:

```zig
var parts = [_][]const u8{
    size_str,
    event_prefix,
    data,
    event_suffix ++ "\r\n",
};
try host_io.writeSocketVecAll(self.stream, &parts);
```

The adapter is small enough to show in full:

```zig
pub fn writeSocketVecAll(
    stream: std.Io.net.Stream,
    buffers: [][]const u8,
) !void {
    var writer = stream.writer(io, &.{});
    try writer.interface.writeVecAll(buffers);
}
```

The payload stays in its existing storage. The writer handles partial writes; a full socket buffer can still require multiple underlying operations. The useful abstraction is “finish writing these slices,” with failures propagated to the caller.

This detail belongs in an inference engine because the user experiences the entire request:

<figure class="diagram-card">
  <img src="/blog/2026-10-31-zig-017-request-path.svg" alt="Prompt bytes pass through CPU tokenization, GPU prefill, GPU decode and HTTP or SSE writes before becoming visible text." width="280" height="548" style="width: 280px; max-width: 100%; margin-inline: auto;" loading="lazy" />
  <figcaption>The path to visible output crosses both host and GPU code. Improving one stage changes the request only to the extent that stage contributes to the wait.</figcaption>
</figure>

Our tests include a 512 KiB JSON response sent through a constrained socket buffer. That forces the write loop to cope with pressure. A tiny localhost response would be a comfortable way to miss a short-write bug.

## Explicit memory makes a better algorithm easier to express

The tokenizer was the largest win.

Byte-pair encoding repeatedly merges adjacent symbols according to vocabulary ranks. Our original implementation rescanned the remaining pairs after each merge. For short word-sized chunks, a simple scan is quite reasonable. For a long Gemma paragraph, it repeats a lot of work.

The replacement keeps a priority queue for chunks longer than 32 initial symbols. After a merge, it revisits the two neighboring pairs whose answers may have changed. Stale queue entries are discarded, and original positions preserve the leftmost choice when ranks tie. Short chunks retain the scan.

The memory representation makes the improvement especially pleasant. Initial symbols occupy consecutive regions of one byte buffer. Merging two adjacent symbols can extend a slice over those bytes:

```zig
fn joinedSymbols(left: []const u8, right: []const u8) []const u8 {
    std.debug.assert(@intFromPtr(left.ptr) + left.len == @intFromPtr(right.ptr));
    return left.ptr[0 .. left.len + right.len];
}
```

That is the actual [helper in the tokenizer](https://github.com/zolotukhin/zinc/blob/c1d2b7b416cfb407116cc3f7abd67f818468783b/src/model/tokenizer.zig#L1070). Its preconditions matter: both slices must be adjacent regions of the same live backing allocation, and the combined range must remain inside it. The address assertion checks adjacency in safety builds; the caller maintains the allocation and lifetime contract.

There is no new string to own after each merge. The original storage lives until the final pieces have become token IDs. You can follow the allocation, the slices that borrow it and the point where it is freed.

That visibility is one of Zig’s best qualities. You can select the production allocator, use a checking allocator in tests and deliberately exercise allocation failure. You still have to design ownership correctly, but the language gives you direct ways to express and test the design.

The complete migration comparison produced these long-prompt first-output medians:

<div style="max-width: 100%; overflow-x: auto; white-space: nowrap;" role="region" aria-label="Long-prompt first-output latency comparison" tabindex="0">

| Model            | Zig 0.15.2 |   Zig 0.17 | Less waiting |
| ---------------- | ---------: | ---------: | -----------: |
| Qwen 3.5 9B      | 1,399.7 ms | 1,358.6 ms |         2.9% |
| Qwen 3.6 35B-A3B | 1,355.7 ms | 1,285.8 ms |         5.2% |
| Gemma 4 26B-A4B  | 1,917.1 ms | 1,194.2 ms |        37.7% |

</div>

_Historical migration measurements on the R9700, using the same models, prompts and generation settings between builds. These measure request latency, with allocator, tokenizer and network changes included._

Our [performance post](/blog/2026-10-24-zig-0-17-faster-prompts-on-rdna4/) records the two alternating server rounds, 108 measured requests and separate comparisons between the initial 0.17 port and the later BPE optimization. Decode stayed essentially level. Complete token arrays were also checked against llama.cpp across eight GGUFs.

The payoff is very concrete: a user gets the same checked tokenization with less CPU work before generation begins.

## Safety checks can reveal a contract you forgot to write down

The migration also caught a streaming bug that is more educational than a benchmark.

Our tool-call detector returns a slice of pending content for the server to send. The detector owns the storage; the server borrows it until the next feed. We used to clear the list with `clearRetainingCapacity()` before returning that slice.

In safety builds, clearing the list poisoned the removed bytes. The returned slice still pointed to them. Keeping the allocation had felt like keeping the data, but the container operation had invalidated that assumption.

The [current implementation](https://github.com/zolotukhin/zinc/blob/c1d2b7b416cfb407116cc3f7abd67f818468783b/src/server/tool_format.zig#L150) preserves the borrowed bytes while resetting the logical length:

```zig
pub fn takeContentDelta(self: *StreamingDetector) []const u8 {
    const out = self.content_pending.items;
    self.content_pending.items.len = 0;
    return out;
}
```

The caller must consume that slice before the next feed overwrites the buffer. The function’s documentation now says so.

Zig does not prove every borrowed lifetime for us. Here, its safety behavior made an ownership mistake visible, and the small representation made the correction easy to inspect. That is a useful combination in code where a copy has a cost and a dangling slice has a much larger one.

## A GPU optimization needs an explicit eligibility rule

Another recent ZINC fix puts the same philosophy on the GPU boundary.

Two optimized Vulkan attention kernels assume a 64-lane subgroup. An Intel Xe2 device in our tests supported subgroup sizes from 16 to 32. The pipeline could still be created, so checking for a loaded pipeline was insufficient. Some prompt shapes produced incorrect text.

The [host-side eligibility check](https://github.com/zolotukhin/zinc/blob/c1d2b7b416cfb407116cc3f7abd67f818468783b/src/compute/forward.zig#L13383) now includes the actual requirement:

```zig
fn wave64BatchedAttentionAvailable(self: *const InferenceEngine) bool {
    return self.instance.caps.supportsRequiredSubgroupSize(64) or
        self.instance.caps.default_subgroup_size == 64;
}
```

The specialized paths use that predicate alongside their shape checks. Intel takes the generic batched attention path, while eligible RDNA hardware retains the optimized path.

This fix uses ordinary Zig control flow; it is not a feature introduced by 0.17. It illustrates why we like the language for this job. Device capabilities, tensor dimensions and pipeline handles remain visible in the host code. A fast kernel comes with a condition you can read, challenge and test on another device.

## The build deserves the same attention as the hot loop

ZINC calls C interfaces for Vulkan and its Metal shim. During the port, we moved the bindings into explicit translated modules. This excerpt from [build.zig](https://github.com/zolotukhin/zinc/blob/c1d2b7b416cfb407116cc3f7abd67f818468783b/build.zig#L48) creates the Metal binding module:

```zig
const translated = b.addTranslateC(.{
    .root_source_file = b.path("src/metal/shim.h"),
    .target = target,
    .optimize = module.optimize.?,
});
module.addImport("metal_c", translated.createModule());
```

Source code imports `metal_c` as a named dependency. Header translation, target selection and module wiring belong to the build graph, where we can inspect how they connect.

There is a transitional API in that example. Zig 0.17 removes `@cImport` and deprecates `Build.Step.TranslateC` in favor of the separately versioned official `translate-c` package. Moving ZINC to that package remains a follow-up. The [0.17 release notes](https://ziglang.org/download/0.17.0/release-notes.html#C-Translation-Moving-to-External-Package) spell out the migration.

Other 0.17 changes make the build itself more accessible to tools: separate configuration and execution processes, and a Build Server Protocol through which clients can inspect the graph, request steps and observe results. Incremental compilation has also improved for x86_64 Linux, with `-fincremental --watch` as the documented entry point. These are toolchain capabilities; we have not published a ZINC rebuild-speed benchmark. The same [release notes](https://ziglang.org/download/0.17.0/release-notes.html#Build-Server-Protocol) describe the ZLS integration work still needed at release time.

For a project that compiles host code, translates headers and builds shaders, an inspectable build graph is valuable. Our production Vulkan shaders still go through GLSL and `glslc`; upgrading Zig does not silently replace that pipeline.

## Why this feels like progress toward perfection

The pleasure of working in Zig is often in a small piece of code whose cost you can explain completely: a frame assembled from four slices, a symbol merge that changes a view, a cleanup tied to scope, a GPU path guarded by its real requirements.

Zig 0.17 extends that ambition into more of the toolchain. Adopting it also gave ZINC a reason to bring allocation, I/O and concurrency choices into clearer view. Some of that work exposed bugs. Some removed measurable waiting. Some left an obvious next cleanup.

There is plenty left to do. ZINC still needs its [full signed release qualification](/blog/2026-10-31-zinc-0-1-0-and-the-refrigerator-that-changed-our-tests/). Our shared I/O adapter can become more explicit, and the C translation dependency can finish its move. Each performance change still needs output checks and measurements on the hardware it targets.

I am excited about that work because the direction is coherent. We can keep making the program easier to account for: which bytes it owns, where it waits, what a kernel assumes, and which executable a test actually ran.

A language gets closer to perfection when it helps you ask those questions precisely. An engine gets closer when its answers survive contact with the machine. Zig 0.17 gives us more room to do both.
