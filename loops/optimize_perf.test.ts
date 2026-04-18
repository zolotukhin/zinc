import { describe, expect, test } from "bun:test";
import { mkdir, writeFile } from "node:fs/promises";
import {
  benchmarkSignatureForSpec,
  buildAnalysisReport,
  buildAgentPrompt,
  buildSelfReview,
  cleanupPreviousRunArtifacts,
  classifyApproachTags,
  codexExecArgs,
  effortArtifactPaths,
  formatCodexStreamLine,
  formatCoherenceFailureList,
  formatToolInput,
  formatClaudeStreamLine,
  getEffortSpec,
  improvementThreshold,
  isResumeStateCompatible,
  isMaterialImprovement,
  loadPreviousRun,
  mergeUniqueEntries,
  median,
  parseAgentReport,
  shouldKeepFoundationStep,
  summarizeCoherenceRegression,
  type ClaudeStreamState,
} from "./optimize_perf";

// -- Codex stream formatter ---------------------------------------------------

describe("formatCodexStreamLine", () => {
  test("formats shell command", () => {
    const line = JSON.stringify({ type: "action", name: "shell", command: "zig build 2>&1" });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("shell");
    expect(out).toContain("zig build");
  });

  test("formats file write", () => {
    const line = JSON.stringify({ type: "action", name: "write", input: { file_path: "/src/foo.zig" } });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("write");
    expect(out).toContain("foo.zig");
  });

  test("formats file read", () => {
    const line = JSON.stringify({ type: "action", name: "read", input: { file_path: "/a/b/c.zig" } });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("read");
    expect(out).toContain("c.zig");
  });

  test("formats agent message", () => {
    const line = JSON.stringify({ type: "message", content: "I will now edit the file." });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("I will now edit the file.");
  });

  test("skips tool output", () => {
    const line = JSON.stringify({ type: "function_call_output", output: "lots of text..." });
    expect(formatCodexStreamLine(line)).toBeNull();
  });

  test("skips empty lines", () => {
    expect(formatCodexStreamLine("")).toBeNull();
    expect(formatCodexStreamLine("   ")).toBeNull();
  });

  test("returns null for non-JSON", () => {
    expect(formatCodexStreamLine("not json at all")).toBeNull();
  });

  test("shows thinking indicator", () => {
    const line = JSON.stringify({ type: "thinking" });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("thinking");
  });
});

// -- Claude stream formatter --------------------------------------------------

describe("formatClaudeStreamLine", () => {
  function freshState(): ClaudeStreamState {
    return {
      currentToolName: null,
      currentBlockIsToolUse: false,
      inputJsonBuffer: "",
      inTextBlock: false,
      sawTextDeltaInCurrentMessage: false,
    };
  }

  test("shows tool name on content_block_start", () => {
    const state = freshState();
    const line = JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_start", content_block: { type: "tool_use", name: "bash" } },
    });
    const out = formatClaudeStreamLine(line, state);
    expect(out).toContain("bash");
    expect(state.currentToolName).toBe("bash");
    expect(state.currentBlockIsToolUse).toBe(true);
  });

  test("streams text delta", () => {
    const state = freshState();
    // Start text block
    formatClaudeStreamLine(JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_start", content_block: { type: "text" } },
    }), state);

    const out = formatClaudeStreamLine(JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_delta", delta: { type: "text_delta", text: "hello world" } },
    }), state);
    expect(out).toBe("hello world");
  });

  test("accumulates input_json_delta silently", () => {
    const state = freshState();
    // Start tool block
    formatClaudeStreamLine(JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_start", content_block: { type: "tool_use", name: "edit" } },
    }), state);

    const out = formatClaudeStreamLine(JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_delta", delta: { type: "input_json_delta", partial_json: '{"file' } },
    }), state);
    expect(out).toBeNull();
    expect(state.inputJsonBuffer).toBe('{"file');
  });

  test("returns null for empty line", () => {
    expect(formatClaudeStreamLine("", freshState())).toBeNull();
  });

  test("returns raw line for non-JSON", () => {
    const out = formatClaudeStreamLine("some random text", freshState());
    expect(out).toBe("some random text\n");
  });
});

// -- formatToolInput ----------------------------------------------------------

describe("formatToolInput", () => {
  test("formats bash command", () => {
    const out = formatToolInput("bash", JSON.stringify({ command: "ls -la" }));
    expect(out).toContain("ls -la");
  });

  test("formats edit with file path", () => {
    const out = formatToolInput("edit", JSON.stringify({ file_path: "/Users/me/project/src/main.zig" }));
    expect(out).toContain("src/main.zig");
  });

  test("formats write with line count", () => {
    const out = formatToolInput("write", JSON.stringify({ file_path: "/a/b.zig", content: "line1\nline2\nline3" }));
    expect(out).toContain("b.zig");
    expect(out).toContain("3 lines");
  });

  test("formats read with short path", () => {
    const out = formatToolInput("read", JSON.stringify({ file_path: "/long/path/to/file.zig" }));
    expect(out).toContain("file.zig");
  });

  test("formats grep with pattern", () => {
    const out = formatToolInput("grep", JSON.stringify({ pattern: "computeBarrier" }));
    expect(out).toContain("/computeBarrier/");
  });

  test("returns empty for unknown tool", () => {
    expect(formatToolInput("unknown_tool", "{}")).toBe("");
  });
});

// -- loadPreviousRun ----------------------------------------------------------

describe("loadPreviousRun", () => {
  test("returns empty state for nonexistent effort", async () => {
    // effort 99 won't have a log file
    const result = await loadPreviousRun(99);
    expect(result.history).toBe("");
    expect(result.bestTokPerSec).toBe(0);
    expect(result.lastCycle).toBe(0);
    expect(result.bestCycle).toBeNull();
    expect(result.bestCommitHash).toBeNull();
  });
});

describe("run artifact cleanup", () => {
  test("cleanupPreviousRunArtifacts removes only the requested effort files", async () => {
    const targetEffort = 98761;
    const otherEffort = 98762;
    const targetPaths = effortArtifactPaths(targetEffort);
    const otherPaths = effortArtifactPaths(otherEffort);

    await mkdir(".perf_optimize", { recursive: true });
    await Promise.all(targetPaths.map((path, index) => writeFile(path, `target-${index}`)));
    await Promise.all(otherPaths.map((path, index) => writeFile(path, `other-${index}`)));

    const removedPaths = await cleanupPreviousRunArtifacts(targetEffort);

    expect(removedPaths.sort()).toEqual([...targetPaths].sort());
    for (const path of targetPaths) {
      expect(Bun.file(path).exists()).resolves.toBe(false);
    }
    for (const path of otherPaths) {
      expect(Bun.file(path).exists()).resolves.toBe(true);
    }

    await cleanupPreviousRunArtifacts(otherEffort);
  });
});

describe("codexExecArgs", () => {
  test("pins the configured reasoning effort", () => {
    expect(codexExecArgs("optimize")).toEqual([
      "exec",
      "-c",
      'model_reasoning_effort="xhigh"',
      "--dangerously-bypass-approvals-and-sandbox",
      "--json",
      "optimize",
    ]);
  });
});

// -- Controller helpers ------------------------------------------------------

describe("controller helpers", () => {
  test("median returns the middle sample", () => {
    expect(median([37.4, 38.5, 37.2])).toBe(37.4);
  });

  test("improvement threshold uses absolute floor", () => {
    expect(improvementThreshold(37.2)).toBe(0.5);
  });

  test("material improvement rejects noisy outlier-sized deltas below threshold", () => {
    const currentBest = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 37.28,
      tokPerSecSamples: [37.1, 37.3, 37.4],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: 21.7,
      bandwidthSamples: [21.6, 21.7, 21.8],
      error: null,
    };
    const candidate = {
      ...currentBest,
      tokPerSec: 37.62,
      tokPerSecSamples: [37.5, 37.6, 37.7],
    };
    expect(isMaterialImprovement(candidate, currentBest)).toBe(false);
  });

  test("material improvement accepts clear gains over the current accepted baseline", () => {
    const currentBest = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 37.28,
      tokPerSecSamples: [37.1, 37.3, 37.4],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: 21.7,
      bandwidthSamples: [21.6, 21.7, 21.8],
      error: null,
    };
    const candidate = {
      ...currentBest,
      tokPerSec: 38.1,
      tokPerSecSamples: [38.0, 38.1, 38.2],
    };
    expect(isMaterialImprovement(candidate, currentBest)).toBe(true);
  });

  test("agent prompt uses current accepted baseline rather than original baseline", () => {
    const originalBaseline = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 37.28,
      tokPerSecSamples: [37.0, 37.3, 37.4],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: 21.7,
      bandwidthSamples: [21.6, 21.7, 21.8],
      error: null,
    };
    const currentBest = {
      ...originalBaseline,
      tokPerSec: 38.52,
      tokPerSecSamples: [38.4, 38.5, 38.6],
      bandwidthUtil: 22.4,
      bandwidthSamples: [22.3, 22.4, 22.5],
    };

    const prompt = buildAgentPrompt(
      "Step 1",
      originalBaseline,
      currentBest,
      2,
      "\nCycle 1: KEPT — 38.52 tok/s",
      "qwen35b",
      {
        cycles: [],
        failedApproaches: ["descriptor plumbing variant regressed 0.2 tok/s"],
        ideas: ["convert dmmv path after helper exists"],
        stalledCycles: 4,
        consecutiveFoundationKeeps: 1,
        reviewSummary: "Repeated dead ends: descriptor(3).",
        bestPerf: {
          cycle: 1,
          tokPerSec: 38.52,
          tokPerSecSamples: [38.4, 38.5, 38.6],
          bandwidthUtil: 22.4,
          bandwidthSamples: [22.3, 22.4, 22.5],
          outputText: "Paris.",
          commitHash: "04d0942b9fe04aca9611691bea2a66f3394225c0",
        },
      },
    );

    expect(prompt).toContain("Current Checked-Out Code");
    expect(prompt).toContain("Best Accepted Performance Checkpoint");
    expect(prompt).toContain("38.52 tok/s [38.40, 38.50, 38.60]");
    expect(prompt).toContain("Original Run Baseline");
    expect(prompt).toContain("37.28 tok/s [37.00, 37.30, 37.40]");
    expect(prompt).toContain("must beat the best accepted performance checkpoint");
    expect(prompt).toContain("Failed Approaches");
    expect(prompt).toContain("@@@DESCRIPTION:");
  });

  test("prefill effort advertises the correct benchmark focus", () => {
    const spec = getEffortSpec(4);
    expect(spec?.primaryMetricLabel).toBe("prefill tok/s");
    expect(spec?.benchmarkMethod).toContain("long-context prefill");

    const baseline = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 73.7,
      tokPerSecSamples: [73.7],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: null,
    };

    const prompt = buildAgentPrompt(
      "Step 1",
      baseline,
      baseline,
      1,
      "",
      "qwen35b",
      null,
      {
        primaryMetricLabel: spec?.primaryMetricLabel,
        benchmarkMethod: spec?.benchmarkMethod,
      },
    );

    expect(prompt).toContain("Benchmark Focus");
    expect(prompt).toContain("prefill tok/s");
    expect(prompt).toContain("long-context prefill benchmark");
  });

  test("RDNA Qwen35 prefill effort is registered with the flagship benchmark contract", () => {
    const spec = getEffortSpec(6);
    expect(spec).not.toBeNull();
    expect(spec?.doc).toBe("MULTI_HOUR_EFFORT_6_RDNA_QWEN35_PREFILL.md");
    expect(spec?.primaryMetricLabel).toBe("prefill tok/s");
    expect(spec?.summary).toContain("RDNA Qwen35 prefill");
    expect(spec?.benchmarkMethod).toContain("Qwen3.5-35B flagship workload");
  });

  test("resume compatibility rejects state from older benchmark regimes", () => {
    const spec = getEffortSpec(3);
    expect(spec).not.toBeNull();
    const compatible = isResumeStateCompatible({
      effort: 3,
      planDoc: "MULTI_HOUR_EFFORT_3_BATCH_PREFILL.md",
      benchmarkSignature: benchmarkSignatureForSpec(spec!),
      runStartedAt: "2026-04-07T00:00:00.000Z",
      lastUpdatedAt: "2026-04-07T00:00:00.000Z",
      lastCycle: 0,
      bestTokPerSec: 73.7,
      bestCycle: 0,
      bestCommitHash: null,
      bestResult: null,
      stalledCycles: 0,
      consecutiveFoundationKeeps: 0,
      cycles: [],
      failedApproaches: [],
      ideas: [],
      reviewSummaries: [],
    }, spec!);
    expect(compatible).toBe(true);

    const legacyStateCompatible = isResumeStateCompatible({
      effort: 3,
      planDoc: "MULTI_HOUR_EFFORT_3_BATCH_PREFILL.md",
      runStartedAt: "2026-04-07T00:00:00.000Z",
      lastUpdatedAt: "2026-04-07T00:00:00.000Z",
      lastCycle: 98,
      bestTokPerSec: 34.04,
      bestCycle: 0,
      bestCommitHash: null,
      bestResult: null,
      stalledCycles: 98,
      consecutiveFoundationKeeps: 0,
      cycles: [],
      failedApproaches: [],
      ideas: [],
      reviewSummaries: [],
    }, spec!);
    expect(legacyStateCompatible).toBe(false);
  });
});

describe("controller memory helpers", () => {
  test("mergeUniqueEntries deduplicates normalized duplicates", () => {
    const merged = mergeUniqueEntries(
      ["descriptor plumbing regressed 0.2 tok/s"],
      ["Descriptor plumbing regressed 0.2 tok/s!", "switch to dmmv hotspot"],
      10,
    );
    expect(merged).toHaveLength(2);
    expect(merged[0]).toContain("descriptor plumbing");
    expect(merged[1]).toContain("switch to dmmv");
  });

  test("parseAgentReport extracts markers and ideas", () => {
    const stdout = [
      JSON.stringify({ type: "message", content: "@@@DESCRIPTION: Add push descriptor helper" }),
      JSON.stringify({ type: "message", content: "@@@STEP_KIND: enablement" }),
      JSON.stringify({ type: "message", content: "@@@SELF_ANALYSIS: This unlocks dmmv conversion next." }),
      JSON.stringify({ type: "message", content: "@@@NEXT_IDEAS: convert dmmv; measure flash attention" }),
    ].join("\n");

    const report = parseAgentReport(stdout);
    expect(report.description).toContain("Add push descriptor helper");
    expect(report.stepKind).toBe("enablement");
    expect(report.selfAnalysis).toContain("unlocks dmmv");
    expect(report.nextIdeas).toEqual(["convert dmmv", "measure flash attention"]);
  });

  test("classifyApproachTags tags descriptor and dmmv work", () => {
    const tags = classifyApproachTags(
      "Convert push descriptor plumbing for DMMV dispatch",
      ["src/compute/dmmv.zig", "src/vulkan/pipeline.zig"],
    );
    expect(tags).toContain("dmmv");
    expect(tags).toContain("descriptor");
  });

  test("buildSelfReview highlights repeated dead ends", () => {
    const review = buildSelfReview({
      stalledCycles: 5,
      consecutiveFoundationKeeps: 1,
      cycles: [
        {
          cycle: 1,
          timestamp: "",
          description: "descriptor attempt 1",
          selfAnalysis: "",
          nextIdeas: [],
          stepKind: "optimization",
          changedFiles: ["src/vulkan/pipeline.zig"],
          categoryTags: ["descriptor"],
          tokPerSec: 37.1,
          tokPerSecSamples: [37.1],
          bandwidthUtil: 21.5,
          bandwidthSamples: [21.5],
          correct: true,
          improved: false,
          broken: false,
          kept: false,
          foundationKeep: false,
          decisionReason: "no improvement",
          outputText: "Paris.",
          commitHash: null,
        },
        {
          cycle: 2,
          timestamp: "",
          description: "descriptor attempt 2",
          selfAnalysis: "",
          nextIdeas: [],
          stepKind: "optimization",
          changedFiles: ["src/vulkan/command.zig"],
          categoryTags: ["descriptor"],
          tokPerSec: 37.2,
          tokPerSecSamples: [37.2],
          bandwidthUtil: 21.5,
          bandwidthSamples: [21.5],
          correct: true,
          improved: false,
          broken: false,
          kept: false,
          foundationKeep: false,
          decisionReason: "no improvement",
          outputText: "Paris.",
          commitHash: null,
        },
      ],
    });

    expect(review).toContain("Repeated dead ends");
    expect(review).toContain("Stall warning");
    expect(review).toContain("Foundation debt");
  });

  test("shouldKeepFoundationStep accepts enablement within tight noise band", () => {
    const bestPerf = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 38.0,
      tokPerSecSamples: [37.9, 38.0, 38.1],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: 22.0,
      bandwidthSamples: [21.9, 22.0, 22.1],
      error: null,
    };
    const candidate = {
      ...bestPerf,
      tokPerSec: 37.84,
      tokPerSecSamples: [37.8, 37.84, 37.9],
    };
    const keep = shouldKeepFoundationStep(
      candidate,
      bestPerf,
      3,
      0,
      {
        description: "Add enablement helper for later DMMV conversion",
        selfAnalysis: "Plumbing only, follow-up converts hot call sites.",
        nextIdeas: [],
        stepKind: "enablement",
        rawText: "",
      },
      ["src/compute/forward.zig"],
    );
    expect(keep).toBe(true);
  });

  test("shouldKeepFoundationStep rejects larger regressions", () => {
    const bestPerf = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 38.0,
      tokPerSecSamples: [37.9, 38.0, 38.1],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: 22.0,
      bandwidthSamples: [21.9, 22.0, 22.1],
      error: null,
    };
    const candidate = {
      ...bestPerf,
      tokPerSec: 37.5,
      tokPerSecSamples: [37.4, 37.5, 37.6],
    };
    const keep = shouldKeepFoundationStep(
      candidate,
      bestPerf,
      4,
      0,
      {
        description: "Add enablement helper for later DMMV conversion",
        selfAnalysis: "Plumbing only, follow-up converts hot call sites.",
        nextIdeas: [],
        stepKind: "enablement",
        rawText: "",
      },
      ["src/compute/forward.zig"],
    );
    expect(keep).toBe(false);
  });

  test("coherence non-regression ignores already-accepted failing cases", () => {
    const candidate = {
      failures: [
        {
          id: "Qwen3-8B::The capital of France is",
          label: "Qwen3-8B [The capital of France is]",
          model: "Qwen3-8B",
          prompt: "The capital of France is",
          outputText: "",
          kind: "crash" as const,
        },
        {
          id: "Gemma4-12B::What is 2+2?",
          label: "Gemma4-12B [What is 2+2?]",
          model: "Gemma4-12B",
          prompt: "What is 2+2?",
          outputText: "What is 5-3?",
          kind: "mismatch" as const,
        },
      ],
      failureIds: [
        "Qwen3-8B::The capital of France is",
        "Gemma4-12B::What is 2+2?",
      ],
    };

    expect(summarizeCoherenceRegression(candidate, candidate.failureIds)).toBeNull();
  });

  test("coherence non-regression flags newly introduced failures", () => {
    const candidate = {
      failures: [
        {
          id: "Qwen3-8B::The capital of France is",
          label: "Qwen3-8B [The capital of France is]",
          model: "Qwen3-8B",
          prompt: "The capital of France is",
          outputText: "",
          kind: "crash" as const,
        },
        {
          id: "Qwen3.5-35B::What is 2+2?",
          label: "Qwen3.5-35B [What is 2+2?]",
          model: "Qwen3.5-35B",
          prompt: "What is 2+2?",
          outputText: "five",
          kind: "mismatch" as const,
        },
      ],
      failureIds: [
        "Qwen3-8B::The capital of France is",
        "Qwen3.5-35B::What is 2+2?",
      ],
    };

    const regression = summarizeCoherenceRegression(candidate, [
      "Qwen3-8B::The capital of France is",
    ]);
    expect(regression).toContain("New coherence failures vs accepted baseline");
    expect(regression).toContain("Qwen3.5-35B [What is 2+2?]");
  });

  test("coherence failure formatter renders crashes and mismatches", () => {
    const formatted = formatCoherenceFailureList([
      {
        id: "Qwen3-8B::The capital of France is",
        label: "Qwen3-8B [The capital of France is]",
        model: "Qwen3-8B",
        prompt: "The capital of France is",
        outputText: "",
        kind: "crash" as const,
      },
      {
        id: "Gemma4-12B::What is 2+2?",
        label: "Gemma4-12B [What is 2+2?]",
        model: "Gemma4-12B",
        prompt: "What is 2+2?",
        outputText: "What is 5-3?",
        kind: "mismatch" as const,
      },
    ]);

    expect(formatted).toContain("Qwen3-8B [The capital of France is]: crashed");
    expect(formatted).toContain('Gemma4-12B [What is 2+2?]: "What is 5-3?"');
  });

  test("buildAnalysisReport summarizes kept and reverted cycles", () => {
    const report = buildAnalysisReport({
      effort: 1,
      planDoc: "MULTI_HOUR_EFFORT_1_PUSH_DESCRIPTORS.md",
      runStartedAt: "2026-04-07T00:00:00.000Z",
      lastUpdatedAt: "2026-04-07T01:00:00.000Z",
      lastCycle: 2,
      bestTokPerSec: 38.02,
      bestCycle: 2,
      bestCommitHash: "04d0942b9fe04aca9611691bea2a66f3394225c0",
      bestResult: {
        cycle: 2,
        tokPerSec: 38.02,
        tokPerSecSamples: [38.0, 38.02, 38.1],
        bandwidthUtil: 22.1,
        bandwidthSamples: [22.0, 22.1, 22.2],
        outputText: "Paris.",
        commitHash: "04d0942b9fe04aca9611691bea2a66f3394225c0",
      },
      stalledCycles: 3,
      consecutiveFoundationKeeps: 0,
      failedApproaches: ["descriptor helper variant regressed"],
      ideas: ["switch to dmmv hot path"],
      reviewSummaries: ["Last 2 cycles: 1 perf keep, 1 reverted."],
      cycles: [
        {
          cycle: 1,
          timestamp: "2026-04-07T00:10:00.000Z",
          description: "Descriptor helper",
          selfAnalysis: "",
          nextIdeas: [],
          stepKind: "enablement",
          changedFiles: ["src/vulkan/command.zig"],
          categoryTags: ["descriptor"],
          tokPerSec: 37.7,
          tokPerSecSamples: [37.7, 37.8, 37.9],
          bandwidthUtil: 21.8,
          bandwidthSamples: [21.7, 21.8, 21.9],
          correct: true,
          improved: false,
          broken: false,
          kept: false,
          foundationKeep: false,
          decisionReason: "no improvement",
          outputText: "Paris.",
          commitHash: null,
        },
        {
          cycle: 2,
          timestamp: "2026-04-07T00:20:00.000Z",
          description: "Convert elementwise push descriptors",
          selfAnalysis: "",
          nextIdeas: [],
          stepKind: "optimization",
          changedFiles: ["src/compute/elementwise.zig"],
          categoryTags: ["descriptor", "elementwise"],
          tokPerSec: 38.02,
          tokPerSecSamples: [38.0, 38.02, 38.1],
          bandwidthUtil: 22.1,
          bandwidthSamples: [22.0, 22.1, 22.2],
          correct: true,
          improved: true,
          broken: false,
          kept: true,
          foundationKeep: false,
          decisionReason: "improved",
          outputText: "Paris.",
          commitHash: "04d0942b9fe04aca9611691bea2a66f3394225c0",
        },
      ],
    });

    expect(report).toContain("Cycles: 2 total, 1 perf keeps");
    expect(report).toContain("descriptor:");
    expect(report).toContain("Recent cycles:");
    expect(report).toContain("Failed approaches:");
  });
});

// -- Config sanity ------------------------------------------------------------

describe("config", () => {
  test("env vars are used for host config", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("ZINC_HOST");
    expect(src).toContain("ZINC_PORT");
    expect(src).toContain("ZINC_USER");
    expect(src).toContain("ZINC_RDNA_QWEN35_35B_MODEL");
    expect(src).toContain("ZINC_RDNA_QWEN36_35B_MODEL");
    expect(src).toContain("ZINC_RDNA_QWEN3_8B_MODEL");
    expect(src).toContain("ZINC_RDNA_GEMMA4_31B_MODEL");
    expect(src).toContain("ZINC_RDNA_GEMMA4_12B_MODEL");
    expect(src).toContain("ZINC_RDNA_GPT_OSS_20B_MODEL");
  });

  test("coherence checks include multiple prompts", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("capital of France");
    expect(src).toContain("2+2");
    expect(src).toContain("first four planets");
  });

  test("all six models are listed for coherence", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("Qwen3.5-35B");
    expect(src).toContain("Qwen3.6-35B");
    expect(src).toContain("Qwen3-8B");
    expect(src).toContain("Gemma4-31B");
    expect(src).toContain("Gemma4-12B");
    expect(src).toContain("GPT-OSS-20B");
  });

  test("GPT-OSS coherence sweep gets extra token budget", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("coherenceMaxTokens: 96");
    expect(src).toContain("coherenceMaxTokensForModel");
    expect(src).toContain("zincRemoteCommand(modelTarget, prompt, maxTokens, promptMode)");
  });

  test("Qwen coherence sweep uses chat prompts without changing benchmark mode", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain('promptMode: "raw"');
    expect(src).toContain('coherencePromptMode: "chat"');
    expect(src).toContain("coherencePromptModeForModel");
  });

  test("codex uses exec with sandbox bypass and json", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain('"exec"');
    expect(src).toContain("dangerously-bypass-approvals-and-sandbox");
    expect(src).toContain('"--json"');
  });

  test("blocked ops prevent agent from git push/commit", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("git push");
    expect(src).toContain("git commit");
  });

  test("startup banner shows cycles for the current run", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("Cycles this run:");
  });
});
