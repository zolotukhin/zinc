import { describe, expect, test } from "bun:test";
import { mkdir, writeFile } from "node:fs/promises";
import {
  benchmarkSignatureForSpec,
  buildAnalysisReport,
  buildAgentPrompt,
  buildSelfReview,
  classifyCycleMetrics,
  computeRunMetrics,
  cleanupPreviousRunArtifacts,
  classifyApproachTags,
  codexExecArgs,
  detectAgentRateLimit,
  detectCorrectnessStreak,
  extractOptimizePerfPidsFromPs,
  effortArtifactPaths,
  formatCodexStreamLine,
  formatCoherenceFailureList,
  formatCorrectnessStreakWarning,
  formatLlamaCppComparison,
  formatPhaseBudget,
  formatPhaseBudgetFreshness,
  formatToolInput,
  formatClaudeStreamLine,
  getEffortSpec,
  detectEchoChamber,
  formatEchoChamberWarning,
  hasFlagOnMeasurementEvidence,
  improvementThreshold,
  introducesRuntimeFlag,
  isMeasuredDeadRevert,
  passesNoiseAwareOverride,
  relativeSampleSpread,
  sampleStdev,
  shouldCollectExtraBenchSample,
  isResumeStateCompatible,
  isMaterialImprovement,
  loadPreviousRun,
  mergeUniqueEntries,
  median,
  parseAgentReport,
  shouldKeepFoundationStep,
  shouldRunPivotCycle,
  summarizeCoherenceRegression,
  type BenchResult,
  type ClaudeStreamState,
  type CycleRecord,
  zincCliArgs,
} from "./optimize_perf";

// -- Codex stream formatter ---------------------------------------------------

describe("zincCliArgs", () => {
  test("pins RDNA benchmark runs to the discrete Vulkan device by default", () => {
    const args = zincCliArgs({ path: "/root/models/model.gguf", promptMode: "raw" }, "hello", 8);

    expect(args).toContain("-m '/root/models/model.gguf'");
    expect(args).toContain("-d 1");
    expect(args).toContain("--prompt 'hello'");
    expect(args).toContain("-n 8");
  });

  test("keeps chat mode alongside the explicit device selection", () => {
    const args = zincCliArgs({ path: "/root/models/model.gguf", promptMode: "chat" }, "hello", 8);

    expect(args).toContain("-d 1 --chat");
  });

  test("adds the server-equivalent system turn for chat targets that declare one", () => {
    const args = zincCliArgs({
      path: "/root/models/model.gguf",
      promptMode: "chat",
      systemPrompt: "You are a helpful assistant. Answer directly. Do not show analysis.",
    }, "hello", 8);

    expect(args).toContain("--chat --system-prompt 'You are a helpful assistant. Answer directly. Do not show analysis.'");
  });

  test("does not apply a chat system turn to raw benchmark overrides", () => {
    const args = zincCliArgs({
      path: "/root/models/model.gguf",
      promptMode: "chat",
      systemPrompt: "be direct",
    }, "hello", 8, "raw");

    expect(args).not.toContain("--system-prompt");
  });
});

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

  test("formats current Codex agent message events", () => {
    const line = JSON.stringify({ type: "item.completed", item: { type: "agent_message", text: "DONE" } });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("DONE");
  });

  test("formats current Codex command execution start events", () => {
    const line = JSON.stringify({
      type: "item.started",
      item: { type: "command_execution", command: "/bin/zsh -lc pwd", status: "in_progress" },
    });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("shell");
    expect(out).toContain("/bin/zsh -lc pwd");
  });

  test("skips successful current Codex command execution output", () => {
    const line = JSON.stringify({
      type: "item.completed",
      item: { type: "command_execution", command: "/bin/zsh -lc pwd", aggregated_output: "lots of text", exit_code: 0 },
    });
    expect(formatCodexStreamLine(line)).toBeNull();
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
  test("pins the configured model and reasoning effort", () => {
    expect(codexExecArgs("optimize")).toEqual([
      "exec",
      "-c",
      'model_reasoning_effort="xhigh"',
      "--dangerously-bypass-approvals-and-sandbox",
      "--json",
      "--model",
      "gpt-5.5",
      "optimize",
    ]);
  });
});

// -- Controller helpers ------------------------------------------------------

describe("controller helpers", () => {
  test("median returns the middle sample", () => {
    expect(median([37.4, 38.5, 37.2])).toBe(37.4);
  });

  test("improvement threshold uses absolute floor below the 1%-relative crossover", () => {
    // At 15 tok/s, 1% relative (0.15) is below the absolute floor, so the
    // floor dominates.
    expect(improvementThreshold(15.0)).toBe(0.2);
  });

  test("improvement threshold uses 1%-relative above the crossover", () => {
    // At 37.2 tok/s, 1% relative (0.372) is above the absolute floor, so
    // the relative bound dominates.
    expect(improvementThreshold(37.2)).toBeCloseTo(0.372, 5);
  });

  test("material improvement rejects noisy deltas when gain < 3x stdev", () => {
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
    // Sample stdev is large (~0.3 tok/s), gap is 0.34 — well under the
    // 3x stdev noise-aware bar, and below the 0.372 relative threshold.
    const candidate = {
      ...currentBest,
      tokPerSec: 37.62,
      tokPerSecSamples: [37.0, 37.9, 38.0],
    };
    expect(isMaterialImprovement(candidate, currentBest)).toBe(false);
  });

  test("material improvement accepts a tight-variance gain that clears the noise floor", () => {
    const currentBest = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 27.76,
      tokPerSecSamples: [27.75, 27.76, 27.77],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: null,
    };
    // Reproduces effort-6 cycle 16: samples [28.06, 28.06, 28.05], gap 0.30.
    // Normal threshold at 27.76 is 0.278 → gain 0.30 > 0.278 → accepts via
    // the normal path. Kept as a regression guard on the combined behavior.
    const candidate = {
      ...currentBest,
      tokPerSec: 28.06,
      tokPerSecSamples: [28.06, 28.06, 28.05],
    };
    expect(isMaterialImprovement(candidate, currentBest)).toBe(true);
  });

  test("material improvement accepts via noise-aware override when gain < relative threshold but stdev is tight", () => {
    const currentBest = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 50.0,
      tokPerSecSamples: [49.98, 50.0, 50.02],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: null,
    };
    // Normal relative threshold at 50 is 0.5; candidate gain is 0.2 — below
    // the normal threshold. But the candidate stdev is 0.005 and the gain
    // is 40x noise, above the 3x bar. Accept via noise override.
    const candidate = {
      ...currentBest,
      tokPerSec: 50.2,
      tokPerSecSamples: [50.195, 50.200, 50.205],
    };
    expect(isMaterialImprovement(candidate, currentBest)).toBe(true);
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
      "qwen36b",
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
    expect(prompt).toContain("coherence tested with 3 prompts on 5 models");
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
      "qwen36b",
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

  test("RDNA Qwen36 prefill effort is registered with the flagship benchmark contract", () => {
    const spec = getEffortSpec(6);
    expect(spec).not.toBeNull();
    expect(spec?.doc).toBe("MULTI_HOUR_EFFORT_6_RDNA_QWEN36_PREFILL.md");
    expect(spec?.primaryMetricLabel).toBe("prefill tok/s");
    expect(spec?.summary).toContain("RDNA Qwen36 prefill");
    expect(spec?.benchmarkMethod).toContain("Qwen3.6-35B flagship workload");
    expect(spec?.knownFlatCategories?.join("\n")).toContain("POST-759 PLATEAU");
    expect(spec?.knownFlatCategories?.join("\n")).toContain("SSM-out DP4a");
    expect(spec?.structuralSwingIdeas?.join("\n")).toContain("POST-759 PROFILE-FIRST");
    expect(spec?.structuralSwingIdeas?.join("\n")).toContain("Cross-scenario guard");
  });

  test("RDNA Qwen38 27B effort uses the site context-medium benchmark contract", () => {
    const spec = getEffortSpec(15);
    expect(spec).not.toBeNull();
    expect(spec?.doc).toBe("MULTI_HOUR_EFFORT_15_RDNA_QWEN36_27B_PREFILL_DECODE.md");
    expect(spec?.primaryMetricLabel).toBe("Qwen3.8-27B prefill tok/s");
    expect(spec?.defaultModel).toBe("qwen3827b");
    expect(spec?.benchmarkMethod).toContain("context-medium Coding Review");
    expect(spec?.benchmarkPrompt).toContain("You are reviewing a small TypeScript helper");
    expect(spec?.benchmarkPrompt).toContain("src/cache.ts");
    expect(spec?.benchmarkPrompt).not.toContain("capital of France");
    expect(spec?.minHealthyTokPerSec).toBe(100);
    expect(spec?.knownFlatCategories?.join("\n")).toContain("PARTIAL_ATTN_NORM_STORE");
    expect(spec?.structuralSwingIdeas?.join("\n")).toContain("shaderstats");
    expect(spec?.llamaCppBaselines?.find((b) => b.isPrimary)?.prefillTokPerSec).toBe(308.38);
    expect(spec?.llamaCppSuccessRule).toContain("context-medium 402.92/31.86");
  });

  test("RDNA Qwen35 9B effort targets the largest published prefill gap", () => {
    const spec = getEffortSpec(17);
    expect(spec).not.toBeNull();
    expect(spec?.doc).toBe("MULTI_HOUR_EFFORT_17_RDNA_QWEN35_9B_PREFILL.md");
    expect(spec?.primaryMetricLabel).toBe("Qwen3.5-9B long-draft prefill tok/s");
    expect(spec?.defaultModel).toBe("qwen359b");
    expect(spec?.benchmarkMethod).toContain("decode-extended Long Coding Plan");
    expect(spec?.benchmarkPrompt).toContain("stable benchmark preset");
    expect(spec?.benchmarkPrompt).toContain("Plan:\n1.");
    expect(spec?.llamaCppBaselines?.find((b) => b.isPrimary)?.prefillTokPerSec).toBe(855.82);
    expect(spec?.knownFlatCategories?.join("\n")).toContain("decode first");
    expect(spec?.structuralSwingIdeas?.join("\n")).toContain("Qwen3.5-9B");
  });

  test("RDNA Gemma 26B MoE effort targets grouped prefill MoE", () => {
    const spec = getEffortSpec(18);
    expect(spec).not.toBeNull();
    expect(spec?.doc).toBe("MULTI_HOUR_EFFORT_18_RDNA_GEMMA26_PREFILL.md");
    expect(spec?.primaryMetricLabel).toBe("Gemma 4 26B-A4B long-draft prefill tok/s");
    expect(spec?.defaultModel).toBe("gemma426ba4b");
    expect(spec?.benchmarkMethod).toContain("decode-extended Long Coding Draft");
    expect(spec?.benchmarkPrompt).toContain("stable benchmark preset");
    expect(spec?.benchmarkPrompt).not.toContain("Plan:\n1.");
    expect(spec?.llamaCppBaselines?.find((b) => b.isPrimary)?.prefillTokPerSec).toBe(647.16);
    expect(spec?.knownFlatCategories?.join("\n")).toContain("decode-time Gemma MoE");
    expect(spec?.structuralSwingIdeas?.join("\n")).toContain("Group prompt tokens by selected expert");
  });

  test("RDNA Gemma 31B dense effort targets short-prompt dense prefill gaps", () => {
    const spec = getEffortSpec(19);
    expect(spec).not.toBeNull();
    expect(spec?.doc).toBe("MULTI_HOUR_EFFORT_19_RDNA_GEMMA31_PREFILL.md");
    expect(spec?.primaryMetricLabel).toBe("Gemma 4 31B long-draft prefill tok/s");
    expect(spec?.defaultModel).toBe("gemma431b");
    expect(spec?.benchmarkMethod).toContain("decode-extended Long Coding Draft");
    expect(spec?.benchmarkPrompt).toContain("stable benchmark preset");
    expect(spec?.benchmarkPrompt).not.toContain("Plan:\n1.");
    expect(spec?.llamaCppBaselines?.find((b) => b.isPrimary)?.prefillTokPerSec).toBe(242.37);
    expect(spec?.knownFlatCategories?.join("\n")).toContain("Gemma 26B MoE");
    expect(spec?.structuralSwingIdeas?.join("\n")).toContain("true tiled Q4_K GEMM/MMQ");
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

  test("resume compatibility includes the selected model key and path", () => {
    const spec = getEffortSpec(15);
    expect(spec).not.toBeNull();
    const saved = {
      effort: 15,
      planDoc: "MULTI_HOUR_EFFORT_15_RDNA_QWEN36_27B_PREFILL_DECODE.md",
      benchmarkSignature: benchmarkSignatureForSpec(spec!, "qwen3827b", "/root/models/Qwen3.8-27B-Q4_K_M.gguf"),
      runStartedAt: "2026-05-20T00:00:00.000Z",
      lastUpdatedAt: "2026-05-20T00:00:00.000Z",
      lastCycle: 16,
      bestTokPerSec: 38.55,
      bestCycle: 16,
      bestCommitHash: "d1b4033",
      bestResult: null,
      stalledCycles: 0,
      consecutiveFoundationKeeps: 0,
      cycles: [],
      failedApproaches: [],
      ideas: [],
      reviewSummaries: [],
    };

    expect(isResumeStateCompatible(saved, spec!, "qwen3827b", "/root/models/Qwen3.8-27B-Q4_K_M.gguf")).toBe(true);
    expect(isResumeStateCompatible(saved, spec!, "qwen36b", "/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")).toBe(false);
    expect(isResumeStateCompatible(saved, spec!, "qwen3827b", "/root/models/other.gguf")).toBe(false);

    const legacySignature = JSON.parse(saved.benchmarkSignature);
    delete legacySignature.benchmarkSystemPrompt;
    expect(isResumeStateCompatible({
      ...saved,
      benchmarkSignature: JSON.stringify(legacySignature),
    }, spec!, "qwen3827b", "/root/models/Qwen3.8-27B-Q4_K_M.gguf")).toBe(false);
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
          id: "Gemma4-26B-A4B::What is 2+2?",
          label: "Gemma4-26B-A4B [What is 2+2?]",
          model: "Gemma4-26B-A4B",
          prompt: "What is 2+2?",
          outputText: "What is 5-3?",
          kind: "mismatch" as const,
        },
      ],
      failureIds: [
        "Qwen3-8B::The capital of France is",
        "Gemma4-26B-A4B::What is 2+2?",
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
          id: "Qwen3.6-35B::What is 2+2?",
          label: "Qwen3.6-35B [What is 2+2?]",
          model: "Qwen3.6-35B",
          prompt: "What is 2+2?",
          outputText: "five",
          kind: "mismatch" as const,
        },
      ],
      failureIds: [
        "Qwen3-8B::The capital of France is",
        "Qwen3.6-35B::What is 2+2?",
      ],
    };

    const regression = summarizeCoherenceRegression(candidate, [
      "Qwen3-8B::The capital of France is",
    ]);
    expect(regression).toContain("New coherence failures vs accepted baseline");
    expect(regression).toContain("Qwen3.6-35B [What is 2+2?]");
  });

  test("coherence failure formatter renders crashes and mismatches", () => {
    const formatted = formatCoherenceFailureList([
      {
        id: "Qwen3-8B::The capital of France is",
        label: "Qwen3-8B [The capital of France is]",
        model: "Qwen3-8B",
        prompt: "The capital of France is",
        outputText: "run failed: timeout after 120000ms",
        kind: "crash" as const,
      },
      {
        id: "Gemma4-26B-A4B::What is 2+2?",
        label: "Gemma4-26B-A4B [What is 2+2?]",
        model: "Gemma4-26B-A4B",
        prompt: "What is 2+2?",
        outputText: "What is 5-3?",
        kind: "mismatch" as const,
      },
    ]);

    expect(formatted).toContain("Qwen3-8B [The capital of France is]: crashed");
    expect(formatted).toContain("timeout after 120000ms");
    expect(formatted).toContain('Gemma4-26B-A4B [What is 2+2?]: "What is 5-3?"');
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
    expect(src).toContain("ZINC_RDNA_QWEN36_35B_MODEL");
    expect(src).toContain("ZINC_RDNA_QWEN38_27B_MODEL");
    expect(src).toContain("ZINC_RDNA_QWEN3_8B_MODEL");
    expect(src).toContain("ZINC_RDNA_GEMMA4_31B_MODEL");
    expect(src).toContain("ZINC_RDNA_GEMMA4_12B_MODEL");
  });

  test("coherence checks include multiple prompts", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("capital of France");
    expect(src).toContain("2+2");
    expect(src).toContain("first four planets");
  });

  test("all five models are listed for coherence", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("Qwen3.6-35B");
    expect(src).toContain("Qwen3.8-27B");
    expect(src).toContain("Qwen3-8B");
    expect(src).toContain("Gemma4-31B");
    expect(src).toContain("Gemma4-26B-A4B");
  });

  test("Qwen3.8 27B optimization target uses chat and the canonical RDNA GGUF", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    const routes = await Bun.file(import.meta.dir + "/../src/server/routes.zig").text();
    const start = src.indexOf("qwen3827b: {");
    expect(start).toBeGreaterThanOrEqual(0);
    const block = src.slice(start, start + 500);
    expect(block).toContain('path: envOrDefault("ZINC_RDNA_QWEN38_27B_MODEL", "/root/models/Qwen3.8-27B-Q4_K_M.gguf")');
    expect(block).toContain('promptMode: "chat"');
    expect(block).toContain('systemPrompt: "You are a helpful assistant. Answer directly. Do not show analysis."');
    expect(block).not.toContain("QWEN36_27B_MODEL");
    expect(routes).toContain('"You are a helpful assistant. Answer directly. Do not show analysis."');
  });

  test("coherence sweep supports a per-model token budget", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("coherenceMaxTokens?: number");
    expect(src).toContain("coherenceMaxTokensForModel");
    expect(src).toContain("zincRemoteCommand(testCase.modelTarget, testCase.prompt, testCase.maxTokens, testCase.promptMode)");
  });

  test("Qwen coherence sweep uses chat prompts without changing benchmark mode", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain('promptMode: "raw"');
    expect(src).toContain('coherencePromptMode: "chat"');
    expect(src).toContain("coherencePromptModeForModel");
  });

  test("coherence crashes are retried with diagnostic details", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("runCoherenceCase");
    expect(src).toContain("cleaning RDNA node and retrying crashed cases once");
    expect(src).toContain("String(e).slice(-500)");
    expect(src).toContain("240_000");
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

  test("remote benchmark cleanup is enabled with an escape hatch", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("cleanRemoteBenchmarkNode");
    expect(src).toContain("ZINC_SKIP_REMOTE_CLEAN");
    expect(src).toContain("pkill -f '[z]ig-out/bin/zinc'");
    expect(src).toContain("pkill -f '[l]lama-server'");
  });

  test("remote rsync excludes local env and editor swap files", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain('"--exclude", ".env"');
    expect(src).toContain('"--exclude", ".env.*"');
    expect(src).toContain('"--exclude", "*.swp"');
    expect(src).toContain('"--exclude=\'.env\'"');
    expect(src).toContain('"--exclude=\'.env.*\'"');
    expect(src).toContain('"--exclude=\'*.swp\'"');
    expect(src).toContain('"--exclude=\'*.swo\'"');
    expect(src).not.toContain("--exclude .env --exclude .env.*");
  });

  test("claude invocations pin the 1M-context Opus model and max effort", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    // Default must be the 1M variant.
    expect(src).toContain(`"claude-opus-4-7[1m]"`);
    // Must be overridable via env so a future run can drop to Sonnet/Haiku.
    expect(src).toContain("ZINC_CLAUDE_MODEL");
    // Both the main agent call and the fix-up retry must pass --model.
    const modelFlagCount = (src.match(/"--model", CLAUDE_MODEL,/g) ?? []).length;
    expect(modelFlagCount).toBeGreaterThanOrEqual(2);
    // Effort must stay at max.
    expect(src).toContain(`const CLAUDE_EFFORT = "max"`);
  });
});

describe("formatPhaseBudget", () => {
  test("renders a placeholder when no budget has been captured", () => {
    expect(formatPhaseBudget(null, null)).toContain("no phase profile captured");
  });

  test("lists totals sorted descending and names the biggest bucket", () => {
    const out = formatPhaseBudget({
      perTokenMs: { attn: 4.5, moe: 10.4, ssm: 11.8, tail: 0.9 },
      totalsMs: { attn: 693.0, moe: 1601.6, ssm: 1817.2, tail: 138.6, embed: 0.3 },
      moeTotalsMs: { router: 301, topk: 120, gate_up: 480, swiglu: 80, down: 540, weighted_acc: 80 },
      denseTotalsMs: { gateup: 1200, gate: 550, up: 500, down: 617 },
      ssmTotalsMs: { proj: 1300, conv: 150, delta: 210, gnorm: 90, out: 67 },
      biggestBucket: { name: "ssm", totalMs: 1817.2 },
    }, 0);
    expect(out).toContain("Top-level totals");
    // Ordering: ssm > moe > attn > tail; embed is skipped.
    const ssmIdx = out.indexOf("ssm:");
    const moeIdx = out.indexOf("moe:");
    const attnIdx = out.indexOf("attn:");
    expect(ssmIdx).toBeGreaterThan(-1);
    expect(moeIdx).toBeGreaterThan(ssmIdx);
    expect(attnIdx).toBeGreaterThan(moeIdx);
    expect(out).not.toContain("embed:");
    expect(out).toContain("Biggest top-level bucket: ssm");
    expect(out).toContain("MoE sub-buckets");
    expect(out).toContain("Dense FFN sub-buckets");
    expect(out).toContain("SSM sub-buckets");
  });
});

describe("formatPhaseBudgetFreshness", () => {
  test("stays quiet for a fresh profile with no failed refreshes", () => {
    expect(formatPhaseBudgetFreshness(12, 10, 0, null, null)).toBeNull();
  });

  test("warns when profile data is stale and refreshes have failed", () => {
    const out = formatPhaseBudgetFreshness(
      65,
      50,
      4,
      "profile run completed but emitted no parseable prefill phase budget",
      64,
    );
    expect(out).toContain("STALE");
    expect(out).toContain("cycle 50");
    expect(out).toContain("current cycle is 65");
    expect(out).toContain("4 refresh attempts have failed");
    expect(out).toContain("cycle 64");
    expect(out).toContain("ZINC_PREFILL_PROFILE=1");
  });

  test("warns when no parseable profile has ever been captured", () => {
    const out = formatPhaseBudgetFreshness(1, null, 1, "profile command timed out", 0);
    expect(out).toContain("No parseable prefill phase profile");
    expect(out).toContain("profile command timed out");
  });
});

describe("buildAgentPrompt — effort-6 controller hints", () => {
  const baseline = {
    buildOk: true,
    buildOutput: "",
    tokPerSec: 25.67,
    tokPerSecSamples: [25.64, 25.67, 25.67],
    correct: true,
    outputText: "Paris.",
    bandwidthUtil: null,
    bandwidthSamples: [],
    error: null,
  };

  test("renders phase budget, known-flat, and swing ideas for prefill efforts", () => {
    const prompt = buildAgentPrompt(
      "Step 1",
      baseline,
      baseline,
      25,
      "",
      "qwen36b",
      {
        cycles: [],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 0,
        consecutiveFoundationKeeps: 0,
        reviewSummary: null,
        bestPerf: null,
        phaseBudget: {
          perTokenMs: { attn: 4.5, moe: 10.4, ssm: 11.8, tail: 0.9 },
          totalsMs: { attn: 693, moe: 1601.6, ssm: 1817.2, tail: 138.6 },
          moeTotalsMs: {},
          denseTotalsMs: { gateup: 1200, down: 600 },
          ssmTotalsMs: { proj: 1300 },
          biggestBucket: { name: "ssm", totalMs: 1817.2 },
        },
        phaseBudgetCycle: 20,
      },
      {
        primaryMetricLabel: "Qwen3.8-27B prefill tok/s",
        benchmarkMethod: "long-context prefill on RDNA",
        knownFlatCategories: ["narrowing compute→compute barriers is flat on RDNA4"],
        structuralSwingIdeas: ["wire recordBatchDispatch into SSM proj with num_cols=2"],
      },
    );
    expect(prompt).toContain("Current Prefill Phase Budget");
    expect(prompt).toContain("Biggest top-level bucket: ssm");
    expect(prompt).toContain("Phase Budget Freshness");
    expect(prompt).toContain("STALE: phase profile was captured at cycle 20");
    expect(prompt).toContain("Dominant Bucket Directive");
    expect(prompt).toContain("Largest named sub-buckets overall");
    expect(prompt).toContain("ssm.proj=1300.0 ms");
    expect(prompt).toContain("dense_ffn.gateup=1200.0 ms");
    expect(prompt).toContain("Known Flat Territory");
    expect(prompt).toContain("narrowing compute→compute barriers is flat on RDNA4");
    expect(prompt).toContain("Structural Swing Ideas");
    expect(prompt).toContain("wire recordBatchDispatch into SSM proj with num_cols=2");
    expect(prompt).toContain("structural swing required this cycle: no");
  });

  test("STEP_BACK stall triggers the structural-swing-required banner", () => {
    const prompt = buildAgentPrompt(
      "Step 1",
      baseline,
      baseline,
      26,
      "",
      "qwen36b",
      {
        cycles: [],
        failedApproaches: ["micro barrier narrowing again"],
        ideas: [],
        stalledCycles: 5,
        consecutiveFoundationKeeps: 1,
        reviewSummary: "Repeated dead ends: barrier(4).",
        bestPerf: null,
        phaseBudget: null,
        phaseBudgetCycle: null,
      },
      {
        primaryMetricLabel: "prefill tok/s",
        benchmarkMethod: "long-context prefill on RDNA",
        knownFlatCategories: ["narrowing compute→compute barriers is flat on RDNA4"],
        structuralSwingIdeas: ["wire recordBatchDispatch into SSM proj with num_cols=2"],
      },
    );
    expect(prompt).toContain("STRUCTURAL SWING REQUIRED");
    expect(prompt).toContain("structural swing required this cycle: YES");
    expect(prompt).toContain("known-flat pattern");
  });

  test("dominant bucket directive steers custom 27B prefill labels toward dense FFN", () => {
    const prompt = buildAgentPrompt(
      "Step 1",
      baseline,
      baseline,
      37,
      "",
      "qwen3827b",
      {
        cycles: [],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 4,
        consecutiveFoundationKeeps: 0,
        reviewSummary: null,
        bestPerf: null,
        phaseBudget: {
          perTokenMs: { dense_ffn: 8.87, ssm: 6.90, attn: 1.11 },
          totalsMs: { dense_ffn: 3087, ssm: 2400.5, attn: 387, tail: 1.9 },
          moeTotalsMs: {},
          denseTotalsMs: { gateup: 1900, down: 1187 },
          ssmTotalsMs: { proj: 982.1, delta: 804.6 },
          biggestBucket: { name: "dense_ffn", totalMs: 3087 },
        },
        phaseBudgetCycle: 34,
      },
      {
        primaryMetricLabel: "Qwen3.8-27B prefill tok/s",
        benchmarkMethod: "site-aligned context-medium Coding Review",
        structuralSwingIdeas: ["dense down+acc fusion"],
      },
    );
    expect(prompt).toContain("Dominant Bucket Directive");
    expect(prompt).toContain("largest top-level bucket is dense_ffn");
    expect(prompt).toContain("Avoid SSM-only work");
    expect(prompt).toContain("Dense FFN sub-buckets");
    expect(prompt).toContain("Largest named sub-buckets overall");
    expect(prompt).toContain("dense_ffn.gateup=1900.0 ms");
    expect(prompt).toContain("top-level buckets are close");
  });

  test("a freshly-banked foundation keep still demands a swing next cycle", () => {
    const prompt = buildAgentPrompt(
      "Step 1",
      baseline,
      baseline,
      27,
      "",
      "qwen36b",
      {
        cycles: [],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 2,
        consecutiveFoundationKeeps: 1,
        reviewSummary: null,
        bestPerf: null,
        phaseBudget: null,
        phaseBudgetCycle: null,
      },
      {
        primaryMetricLabel: "prefill tok/s",
        benchmarkMethod: "long-context prefill on RDNA",
        structuralSwingIdeas: ["wire recordBatchDispatch into SSM proj with num_cols=2"],
      },
    );
    expect(prompt).toContain("STRUCTURAL SWING REQUIRED");
  });

  test("decode efforts do not render the prefill phase budget block", () => {
    const prompt = buildAgentPrompt(
      "Step 1",
      baseline,
      baseline,
      1,
      "",
      "qwen36b",
      null,
      {
        primaryMetricLabel: "decode tok/s",
        benchmarkMethod: "200-token decode",
      },
    );
    expect(prompt).not.toContain("Current Prefill Phase Budget");
  });

  test("references block stays hidden until stall >= threshold", () => {
    const unstalled = buildAgentPrompt(
      "Step 1",
      baseline,
      baseline,
      5,
      "",
      "qwen36b",
      {
        cycles: [],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 2,
        consecutiveFoundationKeeps: 0,
        reviewSummary: null,
        bestPerf: null,
      },
      {
        primaryMetricLabel: "prefill tok/s",
        benchmarkMethod: "long-context prefill on RDNA",
        referenceImplementations: [
          { path: "/Users/zolotukhin/Workplace/llama.cpp", focus: "Vulkan backend" },
        ],
      },
    );
    expect(unstalled).not.toContain("Reference Implementations on Disk");

    const stalled = buildAgentPrompt(
      "Step 1",
      baseline,
      baseline,
      6,
      "",
      "qwen36b",
      {
        cycles: [],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 5,
        consecutiveFoundationKeeps: 0,
        reviewSummary: null,
        bestPerf: null,
      },
      {
        primaryMetricLabel: "prefill tok/s",
        benchmarkMethod: "long-context prefill on RDNA",
        referenceImplementations: [
          { path: "/Users/zolotukhin/Workplace/llama.cpp", focus: "Vulkan backend" },
        ],
      },
    );
    expect(stalled).toContain("Reference Implementations on Disk");
    expect(stalled).toContain("/Users/zolotukhin/Workplace/llama.cpp");
  });

  test("flag-on measurement rule is surfaced in the task block", () => {
    const prompt = buildAgentPrompt(
      "Step 1",
      baseline,
      baseline,
      5,
      "",
      "qwen36b",
      null,
      {
        primaryMetricLabel: "prefill tok/s",
        benchmarkMethod: "long-context prefill on RDNA",
      },
    );
    expect(prompt).toContain("Flag-gated changes must be measured in the same cycle");
    expect(prompt).toContain("ZINC_*");
  });
});

describe("introducesRuntimeFlag / hasFlagOnMeasurementEvidence", () => {
  function report(description: string, analysis: string, stepKind: "analysis" | "enablement" = "enablement"): {
    description: string;
    selfAnalysis: string;
    nextIdeas: string[];
    stepKind: "analysis" | "enablement";
    rawText: string;
  } {
    return {
      description,
      selfAnalysis: analysis,
      nextIdeas: [],
      stepKind,
      rawText: `${description}\n${analysis}`,
    };
  }

  test("introducesRuntimeFlag detects ZINC_ env flag mentions", () => {
    expect(
      introducesRuntimeFlag(
        report("wire behind ZINC_PREFILL_BATCH=1 flag", "flag defaults off"),
        ["src/compute/forward.zig"],
      ),
    ).toBe(true);
  });

  test("introducesRuntimeFlag returns false for a plain code change", () => {
    expect(
      introducesRuntimeFlag(
        report("rewrite SSM proj dispatch ordering", "no env vars, no flags, just reorder"),
        ["src/compute/forward.zig"],
      ),
    ).toBe(false);
  });

  test("introducesRuntimeFlag ignores the diagnostic prefill profile flag", () => {
    expect(
      introducesRuntimeFlag(
        report(
          "Repair ZINC_PREFILL_PROFILE=1 phase budget instrumentation",
          "Profile now emits parseable dense_ffn 293.0 ms and attention 159.8 ms buckets.",
        ),
        ["src/compute/forward.zig"],
      ),
    ).toBe(false);
  });

  test("hasFlagOnMeasurementEvidence requires both a flag-on phrase and a tok/s number", () => {
    expect(
      hasFlagOnMeasurementEvidence(report("wire X", "flag defaults off")),
    ).toBe(false);
    expect(
      hasFlagOnMeasurementEvidence(report("wire X", "measured with flag ON: no tok/s cited")),
    ).toBe(false);
    expect(
      hasFlagOnMeasurementEvidence(report("wire X", "flag-on path at 25.40 tok/s vs best 25.67")),
    ).toBe(true);
  });

  test("shouldKeepFoundationStep rejects a flag-gated foundation that did not measure flag-on", () => {
    const bestPerf = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 25.67,
      tokPerSecSamples: [25.64, 25.67, 25.67],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: null,
    };
    const candidate = { ...bestPerf, tokPerSec: 25.65, tokPerSecSamples: [25.62, 25.65, 25.66] };
    const keep = shouldKeepFoundationStep(
      candidate,
      bestPerf,
      3,
      0,
      report(
        "Wire behind ZINC_PREFILL_BATCH=1 flag (off by default)",
        "Plumbing only; defaults off. Next cycle will measure flag-on.",
      ),
      ["src/compute/forward.zig"],
    );
    expect(keep).toBe(false);
  });

  test("shouldKeepFoundationStep accepts a flag-gated foundation that measured flag-on in-cycle", () => {
    const bestPerf = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 25.67,
      tokPerSecSamples: [25.64, 25.67, 25.67],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: null,
    };
    const candidate = { ...bestPerf, tokPerSec: 25.65, tokPerSecSamples: [25.62, 25.65, 25.66] };
    const keep = shouldKeepFoundationStep(
      candidate,
      bestPerf,
      3,
      0,
      report(
        "Wire behind ZINC_PREFILL_BATCH=1 flag",
        "Plumbing; measured flag-on at 25.40 tok/s vs 25.65 flag-off, acceptable for a foundation step that unlocks cycle N+1.",
      ),
      ["src/compute/forward.zig"],
    );
    expect(keep).toBe(true);
  });

  test("shouldKeepFoundationStep keeps diagnostic profile repairs with concrete phase evidence", () => {
    const bestPerf = {
      buildOk: true,
      buildOutput: "",
      tokPerSec: 62.86,
      tokPerSecSamples: [62.82, 62.86, 62.90],
      correct: true,
      outputText: "Paris.",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: null,
    };
    const candidate = { ...bestPerf, tokPerSec: 62.80, tokPerSecSamples: [62.75, 62.80, 62.86] };
    const keep = shouldKeepFoundationStep(
      candidate,
      bestPerf,
      0,
      0,
      report(
        "Repair ZINC_PREFILL_PROFILE=1 phase budget instrumentation",
        "Profile smoke now emits parseable buckets: dense_ffn 293.0 ms, attention 159.8 ms, gate_up 140.0 ms, down 146.6 ms.",
        "analysis",
      ),
      ["src/compute/forward.zig"],
    );
    expect(keep).toBe(true);
  });
});

describe("sampleStdev + passesNoiseAwareOverride", () => {
  test("sampleStdev returns 0 for a single sample", () => {
    expect(sampleStdev([42])).toBe(0);
  });

  test("relativeSampleSpread reports spread around the median", () => {
    expect(relativeSampleSpread([100, 101, 99])).toBeCloseTo(0.02, 4);
    expect(relativeSampleSpread([100])).toBe(0);
  });

  test("shouldCollectExtraBenchSample requests extra samples only for noisy runs", () => {
    expect(shouldCollectExtraBenchSample([38.55, 38.56], 3, 5)).toBe(true);
    expect(shouldCollectExtraBenchSample([38.55, 38.56, 38.54], 3, 5)).toBe(false);
    expect(shouldCollectExtraBenchSample([38.00, 38.80, 38.40], 3, 5)).toBe(true);
    expect(shouldCollectExtraBenchSample([38.00, 38.80, 38.40, 38.35, 38.50], 3, 5)).toBe(false);
  });

  test("sampleStdev on cycle-16-like samples is near zero", () => {
    expect(sampleStdev([28.06, 28.06, 28.05])).toBeCloseTo(0.005, 2);
  });

  test("noise override accepts cycle-16-shape gains (60x stdev)", () => {
    const best = {
      buildOk: true, buildOutput: "", tokPerSec: 27.76,
      tokPerSecSamples: [27.75, 27.76, 27.77],
      correct: true, outputText: "Paris.", bandwidthUtil: null, bandwidthSamples: [], error: null,
    };
    const cand = { ...best, tokPerSec: 28.06, tokPerSecSamples: [28.06, 28.06, 28.05] };
    expect(passesNoiseAwareOverride(cand, best)).toBe(true);
  });

  test("noise override rejects cycle-19-shape gains (gap < stdev)", () => {
    const best = {
      buildOk: true, buildOutput: "", tokPerSec: 27.76,
      tokPerSecSamples: [27.75, 27.76, 27.77],
      correct: true, outputText: "Paris.", bandwidthUtil: null, bandwidthSamples: [], error: null,
    };
    // Samples [28.06, 27.78, 27.77] — median 27.78, gap only 0.02, noise large.
    const cand = { ...best, tokPerSec: 27.78, tokPerSecSamples: [28.06, 27.78, 27.77] };
    expect(passesNoiseAwareOverride(cand, best)).toBe(false);
  });

  test("noise override respects the 0.15 absolute floor", () => {
    const best = {
      buildOk: true, buildOutput: "", tokPerSec: 27.76,
      tokPerSecSamples: [27.76, 27.76, 27.76],
      correct: true, outputText: "Paris.", bandwidthUtil: null, bandwidthSamples: [], error: null,
    };
    // Tiny gain 0.05 — below the 0.15 abs floor, override should not trigger
    // even if stdev is zero.
    const cand = { ...best, tokPerSec: 27.81, tokPerSecSamples: [27.81, 27.81, 27.81] };
    expect(passesNoiseAwareOverride(cand, best)).toBe(false);
  });
});

describe("detectEchoChamber", () => {
  function mkCycle(n: number, desc: string, improved = false): any {
    return {
      cycle: n,
      timestamp: new Date(Date.UTC(2026, 3, 19, 0, n)).toISOString(),
      description: desc,
      selfAnalysis: "",
      nextIdeas: [],
      stepKind: "enablement",
      changedFiles: ["src/compute/forward.zig"],
      categoryTags: [],
      tokPerSec: improved ? 28.0 : 27.0,
      tokPerSecSamples: [improved ? 28.0 : 27.0],
      bandwidthUtil: null,
      bandwidthSamples: [],
      correct: true,
      improved,
      broken: false,
      kept: improved,
      foundationKeep: false,
      decisionReason: "",
      outputText: "Paris.",
      commitHash: null,
    };
  }

  test("flags an 8-cycle SSM dominance window with zero SSM perf keeps", () => {
    const cycles = [
      mkCycle(1, "MoE gate/up/down kparallel", true),
      mkCycle(2, "ssm proj fusion"),
      mkCycle(3, "ssm delta_net register array"),
      mkCycle(4, "ssm Q8_0 kpar"),
      mkCycle(5, "ssm alpha+beta fusion"),
      mkCycle(6, "ssm out DMMV fusion"),
      mkCycle(7, "ssm proj pair dispatch"),
      mkCycle(8, "ssm proj uint16 packing"),
    ];
    const warn = detectEchoChamber(cycles, []);
    expect(warn).not.toBeNull();
    expect(warn!.bucket).toBe("ssm");
    expect(warn!.count).toBe(7);
    expect(warn!.perfKeepsInBucketWindow).toBe(0);
  });

  test("does not flag a window where the dominant bucket is actually paying off", () => {
    const cycles = [
      mkCycle(1, "ssm proj", true),
      mkCycle(2, "ssm delta"),
      mkCycle(3, "ssm out"),
      mkCycle(4, "ssm kpar"),
      mkCycle(5, "ssm uint16"),
      mkCycle(6, "ssm proj hoist"),
      mkCycle(7, "ssm fusion"),
      mkCycle(8, "ssm reorder"),
    ];
    const warn = detectEchoChamber(cycles, []);
    // The perf keep in the window is in the same bucket → not an echo chamber.
    expect(warn).toBeNull();
  });

  test("does not flag until the window size is reached", () => {
    const cycles = [mkCycle(1, "ssm x"), mkCycle(2, "ssm y"), mkCycle(3, "ssm z")];
    expect(detectEchoChamber(cycles, [])).toBeNull();
  });

  test("format mentions the bucket and any other-bucket wins", () => {
    const warn = {
      bucket: "ssm",
      count: 7,
      window: 8,
      perfKeepsInBucketWindow: 0,
      perfKeepsFromOtherBuckets: 1,
    };
    const text = formatEchoChamberWarning(warn);
    expect(text).toContain("ssm");
    expect(text).toContain("7/8");
    expect(text).toContain("different");
  });
});

describe("isMeasuredDeadRevert", () => {
  function rep(description: string, analysis: string, stepKind: "rollback" | "enablement" | "optimization" | "fix" | "analysis" | "unknown" = "unknown") {
    return {
      description,
      selfAnalysis: analysis,
      nextIdeas: [],
      stepKind,
      rawText: `${description}\n${analysis}`,
    };
  }

  test("recognizes stepKind=rollback as measured-dead", () => {
    expect(isMeasuredDeadRevert(rep("tried X", "", "rollback"))).toBe(true);
  });

  test("recognizes revert-after-measurement prose with a tok/s number", () => {
    expect(
      isMeasuredDeadRevert(rep(
        "Tested 4-way fused SSM proj dispatch; measured flat (25.63 vs 25.66) and reverted",
        "Reverted all code. Path is flat on RDNA4.",
      )),
    ).toBe(true);
  });

  test("does not flag a generic no-op as measured-dead", () => {
    expect(
      isMeasuredDeadRevert(rep(
        "Agent did some exploration but made no concrete change",
        "Needs another cycle to decide direction.",
      )),
    ).toBe(false);
  });

  test("requires a tok/s number — revert prose alone is not evidence", () => {
    expect(
      isMeasuredDeadRevert(rep(
        "Reverted the pair-dispatch code because it seemed flat",
        "Reverted, no measurement number provided",
      )),
    ).toBe(false);
  });
});

describe("shouldRunPivotCycle", () => {
  test("fires at cycle 10 when stalled >= threshold", () => {
    expect(
      shouldRunPivotCycle(10, {
        cycles: [],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 5,
        consecutiveFoundationKeeps: 0,
        reviewSummary: null,
        bestPerf: null,
      }),
    ).toBe(true);
  });

  test("does not fire at cycle 10 when actively making progress", () => {
    expect(
      shouldRunPivotCycle(10, {
        cycles: [],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 0,
        consecutiveFoundationKeeps: 0,
        reviewSummary: null,
        bestPerf: null,
      }),
    ).toBe(false);
  });

  test("does not fire at cycle 7 even when stalled", () => {
    expect(
      shouldRunPivotCycle(7, {
        cycles: [],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 7,
        consecutiveFoundationKeeps: 0,
        reviewSummary: null,
        bestPerf: null,
      }),
    ).toBe(false);
  });

  test("fires again at cycle 20 if still stalled", () => {
    expect(
      shouldRunPivotCycle(20, {
        cycles: [],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 4,
        consecutiveFoundationKeeps: 0,
        reviewSummary: null,
        bestPerf: null,
      }),
    ).toBe(true);
  });
});

describe("classifyCycleMetrics", () => {
  const baseCycle = {
    cycle: 1,
    timestamp: new Date("2026-04-18T16:00:00Z").toISOString(),
    description: "",
    selfAnalysis: "",
    nextIdeas: [],
    stepKind: "enablement" as const,
    changedFiles: ["src/compute/forward.zig"],
    categoryTags: ["dmmv"],
    tokPerSec: 25.65,
    tokPerSecSamples: [25.65, 25.66, 25.64],
    bandwidthUtil: null,
    bandwidthSamples: [],
    correct: true,
    improved: false,
    broken: false,
    kept: true,
    foundationKeep: true,
    decisionReason: "",
    outputText: "Paris.",
    commitHash: "deadbeef",
  };

  test("flags a dormant foundation when self-analysis has no flag-on number", () => {
    const m = classifyCycleMetrics(
      { ...baseCycle, description: "wire behind ZINC_PREFILL_BATCH=1 flag", selfAnalysis: "defaults off, next cycle will measure" },
      new Date("2026-04-18T15:40:00Z").toISOString(),
      [],
    );
    expect(m.introducedFlag).toBe(true);
    expect(m.measuredFlagOn).toBe(false);
    expect(m.informationValue).toBe("dormant_keep");
    expect(m.durationMs).toBe(20 * 60 * 1000);
  });

  test("recognizes measured dead-end findings as valuable information", () => {
    const m = classifyCycleMetrics(
      {
        ...baseCycle,
        foundationKeep: true,
        description: "measure flag-on",
        selfAnalysis: "flag-on path at 24.86 tok/s vs 25.65 flag-off — confirmed net-negative",
      },
      baseCycle.timestamp,
      [],
    );
    expect(m.measuredFlagOn).toBe(true);
    expect(m.informationValue).toBe("measured_dead");
  });

  test("classifies attack buckets from the self-analysis", () => {
    const ssm = classifyCycleMetrics(
      { ...baseCycle, description: "wire SSM proj pair-dispatch", selfAnalysis: "" },
      baseCycle.timestamp,
      [],
    );
    expect(ssm.attackedBucket).toBe("ssm");
    const moe = classifyCycleMetrics(
      { ...baseCycle, description: "batch MoE router across tokens", selfAnalysis: "" },
      baseCycle.timestamp,
      [],
    );
    expect(moe.attackedBucket).toBe("moe");
  });

  test("zero-changed-files with stepKind=rollback maps to measured_dead, not no_op", () => {
    const m = classifyCycleMetrics(
      {
        ...baseCycle,
        stepKind: "rollback",
        kept: false,
        foundationKeep: false,
        changedFiles: [],
        description: "tested fused SSM proj dispatch, measured flat, reverted",
        selfAnalysis: "25.63 vs 25.66 flat, reverted",
        decisionReason: "measured-dead: agent explored, measured, and reverted",
      },
      baseCycle.timestamp,
      [],
    );
    expect(m.informationValue).toBe("measured_dead");
  });

  test("genuine no-op (no rollback, no changes) still classifies as no_op", () => {
    const m = classifyCycleMetrics(
      {
        ...baseCycle,
        stepKind: "unknown",
        kept: false,
        foundationKeep: false,
        changedFiles: [],
        description: "agent produced no changes",
        selfAnalysis: "",
        decisionReason: "no source changes",
      },
      baseCycle.timestamp,
      [],
    );
    expect(m.informationValue).toBe("no_op");
  });

  test("detects citation of reference implementations", () => {
    const m = classifyCycleMetrics(
      { ...baseCycle, description: "port llama.cpp mul_mat_vec_max_cols=8 specialization", selfAnalysis: "" },
      baseCycle.timestamp,
      ["/Users/zolotukhin/Workplace/llama.cpp"],
    );
    expect(m.citedReference).toBe(true);
  });
});

describe("computeRunMetrics", () => {
  test("produces a run-health summary from cycle history", () => {
    const now = (min: number) => new Date(Date.UTC(2026, 3, 18, 15, 30 + min)).toISOString();
    const mkCycle = (n: number, overrides: Record<string, unknown>) => ({
      cycle: n,
      timestamp: now(n * 20),
      description: "",
      selfAnalysis: "",
      nextIdeas: [],
      stepKind: "enablement" as const,
      changedFiles: ["src/compute/forward.zig"],
      categoryTags: ["dmmv"],
      tokPerSec: 25.65,
      tokPerSecSamples: [25.65],
      bandwidthUtil: null,
      bandwidthSamples: [],
      correct: true,
      improved: false,
      broken: false,
      kept: false,
      foundationKeep: false,
      decisionReason: "",
      outputText: "Paris.",
      commitHash: null,
      ...overrides,
    });
    const state = {
      effort: 6,
      planDoc: "",
      runStartedAt: now(0),
      lastUpdatedAt: now(160),
      lastCycle: 8,
      bestTokPerSec: 25.65,
      bestCycle: 0,
      bestCommitHash: null,
      bestResult: null,
      stalledCycles: 8,
      consecutiveFoundationKeeps: 0,
      cycles: [
        mkCycle(1, { kept: true, foundationKeep: true, description: "wire helper", selfAnalysis: "" }),
        mkCycle(2, { description: "3-deep pipeline", selfAnalysis: "" }),
        mkCycle(3, {
          kept: true,
          foundationKeep: true,
          description: "wire SSM proj pair behind ZINC_PREFILL_BATCH=1",
          selfAnalysis: "defaults off, next cycle will measure",
        }),
        mkCycle(8, {
          description: "flag on measurement",
          selfAnalysis: "flag-on path 24.86 tok/s vs 25.65 baseline — net-negative, reverting",
        }),
      ],
      failedApproaches: [],
      ideas: [],
      reviewSummaries: [],
    };
    const rm = computeRunMetrics(state as any, []);
    expect(rm.totalCycles).toBe(4);
    expect(rm.foundationKeeps).toBe(2);
    expect(rm.dormantFoundations).toBeGreaterThanOrEqual(1);
    expect(rm.bucketCoverage.ssm).toBeGreaterThanOrEqual(1);
    expect(rm.cyclesProducingInformation).toBeGreaterThanOrEqual(1);
    expect(rm.totalCycleMs).toBeGreaterThan(0);
  });
});

describe("buildAgentPrompt pivot mode", () => {
  const baseline = {
    buildOk: true,
    buildOutput: "",
    tokPerSec: 25.65,
    tokPerSecSamples: [25.62, 25.65, 25.67],
    correct: true,
    outputText: "Paris.",
    bandwidthUtil: null,
    bandwidthSamples: [],
    error: null,
  };

  test("renders pivot instructions and lists committed foundations", () => {
    const prompt = buildAgentPrompt(
      "Plan body",
      baseline,
      baseline,
      10,
      "",
      "qwen36b",
      {
        cycles: [
          {
            cycle: 1,
            timestamp: "t",
            description: "add helper",
            selfAnalysis: "",
            nextIdeas: [],
            stepKind: "enablement",
            changedFiles: ["src/compute/dmmv.zig"],
            categoryTags: ["dmmv"],
            tokPerSec: 25.65,
            tokPerSecSamples: [25.64, 25.65, 25.66],
            bandwidthUtil: null,
            bandwidthSamples: [],
            correct: true,
            improved: false,
            broken: false,
            kept: true,
            foundationKeep: true,
            decisionReason: "",
            outputText: "Paris.",
            commitHash: "20c0ea8f6fb563a1fce2fc48824412ad8d08bd05",
          },
        ],
        failedApproaches: [],
        ideas: [],
        stalledCycles: 6,
        consecutiveFoundationKeeps: 1,
        reviewSummary: null,
        bestPerf: null,
        phaseBudget: {
          perTokenMs: { attn: 4.5, ssm: 11.8 },
          totalsMs: { attn: 693, ssm: 1817.2 },
          moeTotalsMs: {},
          denseTotalsMs: {},
          ssmTotalsMs: { proj: 1300 },
          biggestBucket: { name: "ssm", totalMs: 1817.2 },
        },
        phaseBudgetCycle: 8,
      },
      {
        primaryMetricLabel: "Qwen3.8-27B prefill tok/s",
        benchmarkMethod: "long-context prefill on RDNA",
        knownFlatCategories: ["barrier narrowing is flat"],
        structuralSwingIdeas: ["port llama.cpp 8-variant DMMV"],
        referenceImplementations: [
          { path: "/Users/zolotukhin/Workplace/llama.cpp", focus: "Vulkan matmul" },
        ],
        mode: "pivot",
      },
    );
    expect(prompt).toContain("PIVOT cycle");
    expect(prompt).toContain("Dead-end audit");
    expect(prompt).toContain("Pivot proposal");
    expect(prompt).toContain("Committed Foundations");
    expect(prompt).toContain("Current Prefill Phase Budget");
    expect(prompt).toContain("Dominant Bucket Directive");
    expect(prompt).toContain("Biggest top-level bucket: ssm");
    expect(prompt).toContain("20c0ea8f");
    expect(prompt).toContain("llama.cpp");
    expect(prompt).toContain("barrier narrowing is flat");
  });
});

describe("formatLlamaCppComparison", () => {
  const baselines = [
    { scenario: "core",            prefillTokPerSec: 61.12,  decodeTokPerSec: 34.43 },
    { scenario: "context-medium",  prefillTokPerSec: 195.01, decodeTokPerSec: 34.40, isPrimary: true },
    { scenario: "context-long",    prefillTokPerSec: 69.89,  decodeTokPerSec: 44.33 },
    { scenario: "decode-extended", prefillTokPerSec: 97.29,  decodeTokPerSec: 31.29 },
  ];

  test("shows primary ratio, gap-to-beat, and the other-scenario block", () => {
    const out = formatLlamaCppComparison(baselines, "Qwen3.8-27B prefill tok/s", "prefill", 150.95);
    expect(out).toContain("150.95");
    expect(out).toContain("195.01");
    expect(out).toContain("77.4%"); // 150.95/195.01
    expect(out).toContain("+44.06"); // 195.01-150.95
    expect(out).toContain("+29.2%"); // (195.01-150.95)/150.95 = 29.2%
    expect(out).toContain("core");
    expect(out).toContain("context-long");
    expect(out).toContain("decode-extended");
  });

  test("classifies tier as 'closing the gap (70-90%)' at 77%", () => {
    const out = formatLlamaCppComparison(baselines, "x", "prefill", 150.95);
    expect(out).toContain("closing the gap");
  });

  test("classifies tier as 'BEATING llama.cpp ✓' when ahead", () => {
    const out = formatLlamaCppComparison(baselines, "x", "prefill", 200.0);
    expect(out).toContain("BEATING llama.cpp");
    expect(out).toContain("margin over llama.cpp");
    expect(out).toContain("next target");
    expect(out).not.toContain("gap to beat:     +-");
  });

  test("classifies tier as 'within striking distance' at >=90%", () => {
    const out = formatLlamaCppComparison(baselines, "x", "prefill", 180.0);
    expect(out).toContain("within striking distance");
  });

  test("handles null/zero bestTokPerSec without crashing or fabricating numbers", () => {
    const out = formatLlamaCppComparison(baselines, "x", "prefill", null);
    expect(out).toContain("—");
    expect(out).not.toContain("NaN");
    expect(out).not.toContain("Infinity");
  });

  test("uses decode baseline when metricMode is decode", () => {
    const out = formatLlamaCppComparison(baselines, "decode tok/s", "decode", 28.0);
    expect(out).toContain("34.40"); // primary decode baseline
    expect(out).not.toContain("195.01"); // not the prefill row
  });

  test("renders scenario prompt-token shapes when available", () => {
    const out = formatLlamaCppComparison(
      [
        { scenario: "core", promptTokens: 36, prefillTokPerSec: 61.12, decodeTokPerSec: 34.43 },
        { scenario: "decode-extended", promptTokens: 64, prefillTokPerSec: 855.82, decodeTokPerSec: 84.96, isPrimary: true },
      ],
      "Qwen3.5-9B long-draft prefill tok/s",
      "prefill",
      817.62,
    );
    expect(out).toContain("decode-extended (64 prompt tokens)");
    expect(out).toContain("prompt:  36 tok");
  });

  test("renders an effort-specific success rule when provided", () => {
    const out = formatLlamaCppComparison(baselines, "x", "prefill", 150.95, "Project success rule: custom effort rule.");
    expect(out).toContain("Project success rule: custom effort rule.");
    expect(out).not.toContain("MULTI_HOUR_EFFORT_15");
  });
});

describe("extractOptimizePerfPidsFromPs", () => {
  const sample = [
    " 47142 /Users/zolotukhin/.bun/bin/bun loops/optimize_perf.ts --effort 15 --model qwen3827b --agent claude --cycles 50",
    " 47140 zsh -lc bun loops/optimize_perf.ts --effort 15 --model qwen3827b --agent claude --cycles 50 2>&1 | tee log",
    " 47139 login -pflq zolotukhin /bin/zsh -lc bun loops/optimize_perf.ts",
    " 47137 SCREEN -dmS zinc_effort15_50 zsh -lc bun loops/optimize_perf.ts --effort 15 --model qwen3827b",
    "  1234 some other process",
  ].join("\n");

  test("returns just the bun-leader PID when other instance is running", () => {
    expect(extractOptimizePerfPidsFromPs(sample, 99999)).toEqual([47142]);
  });

  test("excludes the current process by PID match", () => {
    expect(extractOptimizePerfPidsFromPs(sample, 47142)).toEqual([]);
  });

  test("returns empty on empty input", () => {
    expect(extractOptimizePerfPidsFromPs("", 1)).toEqual([]);
  });

  test("returns empty when no bun command is present", () => {
    const out = " 1234 node something.js\n 5678 python foo.py\n";
    expect(extractOptimizePerfPidsFromPs(out, 99999)).toEqual([]);
  });

  test("finds multiple bun instances if there really are several", () => {
    const out = " 1111 /usr/local/bin/bun loops/optimize_perf.ts --effort 1\n 2222 bun loops/optimize_perf.ts --effort 2\n";
    expect(extractOptimizePerfPidsFromPs(out, 99999)).toEqual([1111, 2222]);
  });
});

describe("detectCorrectnessStreak", () => {
  function rec(cycle: number, correct: boolean, files: string[] = []): CycleRecord {
    return {
      cycle,
      timestamp: new Date(2026, 4, 29, 10, cycle).toISOString(),
      description: "", selfAnalysis: "", nextIdeas: [], stepKind: "optimization",
      changedFiles: files, categoryTags: [],
      tokPerSec: correct ? 100 : 0, tokPerSecSamples: [], bandwidthUtil: null, bandwidthSamples: [],
      correct, improved: false, broken: false, kept: false, foundationKeep: false,
      decisionReason: "", outputText: "", commitHash: null,
    };
  }

  test("returns null when fewer than 3 cycles overall", () => {
    expect(detectCorrectnessStreak([rec(1, false), rec(2, false)])).toBeNull();
  });

  test("returns null when failures are sparse in the window", () => {
    // 2 of last 6 failed (below threshold 3).
    const cycles = [rec(1, false), rec(2, true), rec(3, true), rec(4, true), rec(5, false), rec(6, true)];
    expect(detectCorrectnessStreak(cycles)).toBeNull();
  });

  test("fires when 3+ of the last 6 cycles failed correctness", () => {
    // Exactly the effort-15 run-2 cycles 11-15 pattern.
    const cycles = [
      rec(10, true),
      rec(11, false, ["src/compute/forward.zig"]),
      rec(12, false, ["src/compute/forward.zig"]),
      rec(13, false, ["src/compute/forward.zig"]),
      rec(14, false, ["build.zig", "src/compute/dmmv.zig"]),
      rec(15, false, ["build.zig", "src/compute/dmmv.zig", "src/compute/forward.zig"]),
    ];
    const w = detectCorrectnessStreak(cycles);
    expect(w).not.toBeNull();
    expect(w!.failedCount).toBe(5);
    expect(w!.windowSize).toBe(6);
    expect(w!.recentFailedCycles).toEqual([11, 12, 13, 14, 15]);
    // Files touched by 2+ failed cycles, sorted by frequency.
    expect(w!.sharedFiles).toContain("src/compute/forward.zig");
    expect(w!.sharedFiles).toContain("build.zig");
    expect(w!.sharedFiles).toContain("src/compute/dmmv.zig");
  });

  test("formatted warning calls out the cycle-16 unlock and the files hint", () => {
    const w = {
      failedCount: 4, windowSize: 6, recentFailedCycles: [11, 12, 13, 15],
      sharedFiles: ["src/compute/forward.zig"],
    };
    const out = formatCorrectnessStreakWarning(w);
    expect(out).toContain("4/6 recent cycles");
    expect(out).toContain("DIAGNOSE the failing path");
    expect(out).toContain("src/compute/forward.zig");
    expect(out).toContain("cycle 16"); // narrative anchor
  });
});

describe("improvementThreshold stall-aware halving", () => {
  test("non-stalled uses the standard 1% / 0.2 tps floor", () => {
    // At 103 tok/s, 1% = 1.03 — well above the 0.2 floor.
    expect(improvementThreshold(103, 0)).toBeCloseTo(1.03, 4);
    // At 5 tok/s, 1% = 0.05 — clamped to the 0.2 floor.
    expect(improvementThreshold(5, 0)).toBe(0.2);
  });

  test("stalled past the warning threshold halves the bar (floor preserved)", () => {
    // Stall above STALL_WARNING_THRESHOLD (=4) should halve the bar:
    // 103 → 1.03 / 2 = 0.515.
    expect(improvementThreshold(103, 5)).toBeCloseTo(0.515, 4);
    // Floor of 0.2 is also halved to 0.1 — so a 5 tok/s baseline now
    // bars at 0.1 instead of 0.2, letting tiny-but-clear wins pass.
    expect(improvementThreshold(5, 5)).toBe(0.1);
  });

  test("just-below-threshold stall does not halve", () => {
    // Exactly at warning threshold (4) does halve; just below (3) does not.
    expect(improvementThreshold(103, 3)).toBeCloseTo(1.03, 4);
    expect(improvementThreshold(103, 4)).toBeCloseTo(0.515, 4);
  });

  test("isMaterialImprovement uses stalled bar when stalled", () => {
    const best: BenchResult = {
      buildOk: true, buildOutput: "", tokPerSec: 103, tokPerSecSamples: [103, 103, 103],
      correct: true, outputText: "", bandwidthUtil: null, bandwidthSamples: [], error: null,
    };
    // +0.6 tok/s gain with realistic noise (stdev ~1.5) so the 3-sigma
    // noise override does NOT fire (0.6 < 3*1.5 = 4.5). Then only the
    // flat threshold path matters: 0.6 is below the 1.03 normal bar but
    // above the 0.515 stalled bar.
    const candidate: BenchResult = { ...best, tokPerSec: 103.6, tokPerSecSamples: [103.6, 102.0, 105.0] };
    expect(isMaterialImprovement(candidate, best, 0)).toBe(false);
    expect(isMaterialImprovement(candidate, best, 5)).toBe(true);
  });
});

describe("detectAgentRateLimit", () => {
  const NOW = Date.UTC(2026, 4, 28, 18, 0, 0); // 2026-05-28 18:00 UTC

  test("parses claude api rate_limit_event resetsAt", () => {
    const stdout =
      '{"type":"rate_limit_event","rate_limit_info":{"status":"rejected","resetsAt":1779763200,"rateLimitType":"five_hour"}}';
    const hit = detectAgentRateLimit(stdout, "", NOW);
    expect(hit).not.toBeNull();
    expect(hit?.source).toBe("claude api");
    expect(hit?.resetsAtMs).toBe(1779763200 * 1000);
  });

  test("parses claude plain-text 'session limit · resets HH:MMpm'", () => {
    const stdout = "You've hit your session limit · resets 7:40pm (America/Los_Angeles)";
    const hit = detectAgentRateLimit(stdout, "", NOW);
    expect(hit).not.toBeNull();
    expect(hit?.source).toBe("claude text");
    // future-or-same-day; never in the past
    expect(hit!.resetsAtMs).toBeGreaterThan(NOW);
  });

  test("parses codex 'try again at <Month Day, Year HH:MM AM>'", () => {
    const stdout =
      "You've hit your usage limit. Visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at May 30th, 2026 11:33 AM.";
    const hit = detectAgentRateLimit(stdout, "", NOW);
    expect(hit).not.toBeNull();
    expect(hit?.source).toBe("codex text");
    expect(hit!.resetsAtMs).toBeGreaterThan(NOW);
  });

  test("falls back to +60 min on an unrecognized 429 timing", () => {
    const stdout = '{"error":"rate_limit","api_error_status": 429,"resetsAt": "tomorrow"}';
    const hit = detectAgentRateLimit(stdout, "", NOW);
    expect(hit).not.toBeNull();
    expect(hit?.source).toBe("fallback (+60m)");
    expect(hit!.resetsAtMs).toBe(NOW + 60 * 60 * 1000);
  });

  test("returns null on a genuine no-op (agent ran fine, no changes)", () => {
    const stdout = "Read 12 files\nGrep ...\n(no edits)";
    expect(detectAgentRateLimit(stdout, "", NOW)).toBeNull();
  });
});
