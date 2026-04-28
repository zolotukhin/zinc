import { describe, expect, test } from "bun:test";
import {
  buildPrompt,
  buildReflectionSummary,
  buildSelfReview,
  decideKeep,
  detectPhase,
  evaluateOutputText,
  mergeUniqueEntries,
  snapshotFromResult,
} from "./implement_metal";
import type { BuildRunResult, ControllerState, CycleResult, RunState } from "./implement_metal";

function makeResult(overrides: Partial<BuildRunResult> = {}): BuildRunResult {
  return {
    buildExitCode: 0,
    buildOutput: "",
    testExitCode: 0,
    testOutput: "",
    runExitCode: 0,
    runOutput: "",
    phase: "implement",
    tokPerSec: null,
    tokPerSecSamples: [],
    tokensGenerated: 5,
    outputText: "",
    containsReference: false,
    strongAnswer: false,
    outputQualityScore: 0,
    offTopic: false,
    evaluationNotes: [],
    error: null,
    ...overrides,
  };
}

function makeCycle(overrides: Partial<CycleResult> = {}): CycleResult {
  return {
    cycle: 1,
    timestamp: new Date().toISOString(),
    phase: "optimize",
    description: "Test change",
    kept: true,
    tokPerSec: 36,
    tokensGenerated: 64,
    containsReference: true,
    buildExitCode: 0,
    testExitCode: 0,
    runExitCode: 0,
    outputText: "ĠParis.ĠTheĠcapitalĠof",
    selfAnalysis: "",
    nextIdeas: [],
    ...overrides,
  };
}

function makeState(overrides: Partial<RunState> = {}): RunState {
  return {
    runId: "test-run",
    cycles: [],
    failedApproaches: [],
    ideas: [],
    phase: "optimize",
    currentBest: null,
    stalledCycles: 0,
    bestTokPerSec: 36,
    lastProfileOutput: null,
    lastProfileCycle: null,
    reviewSummaries: [],
    ...overrides,
  };
}

// ── evaluateOutputText ──────────────────────────────────────────────

describe("evaluateOutputText", () => {
  test("accepts BPE-marked Paris prefix as a strong answer", () => {
    const result = evaluateOutputText("ĠParis.ĠTheĠcapitalĠof");
    expect(result.normalizedText).toBe("Paris. The capital of");
    expect(result.containsReference).toBe(true);
    expect(result.strongAnswer).toBe(true);
    expect(result.offTopic).toBe(false);
  });

  test("penalizes contradictory continuations", () => {
    const result = evaluateOutputText("ĠParis.ĠTheĠcapitalĠofĠGermanyĠisĠBerlin");
    expect(result.containsReference).toBe(true);
    expect(result.strongAnswer).toBe(false);
    expect(result.offTopic).toBe(true);
  });

  test("handles empty string", () => {
    const result = evaluateOutputText("");
    expect(result.normalizedText).toBe("");
    expect(result.containsReference).toBe(false);
    expect(result.strongAnswer).toBe(false);
  });

  test("detects Paris without BPE markers", () => {
    const result = evaluateOutputText("Paris is the capital");
    expect(result.containsReference).toBe(true);
    expect(result.strongAnswer).toBe(true);
  });
});

// ── detectPhase ─────────────────────────────────────────────────────

describe("detectPhase", () => {
  test("stays in optimize when answer is correct but tok/s is missing", () => {
    expect(
      detectPhase(
        makeResult({
          outputText: "ĠParis.",
          containsReference: true,
          strongAnswer: true,
          outputQualityScore: 4,
        }),
      ),
    ).toBe("optimize");
  });

  test("stays in implement for partial Paris mentions", () => {
    expect(
      detectPhase(
        makeResult({
          outputText: "Somewhere near Paris",
          containsReference: true,
          strongAnswer: false,
          outputQualityScore: 1,
        }),
      ),
    ).toBe("implement");
  });

  test("returns fix on build failure", () => {
    expect(detectPhase(makeResult({ buildExitCode: 1 }))).toBe("fix");
  });

  test("returns fix on test failure", () => {
    expect(detectPhase(makeResult({ testExitCode: 1 }))).toBe("fix");
  });

  test("returns fix on runtime crash", () => {
    expect(detectPhase(makeResult({ runExitCode: 139 }))).toBe("fix");
  });

  test("returns fix on error string", () => {
    expect(detectPhase(makeResult({ error: "segfault" }))).toBe("fix");
  });
});

// ── decideKeep ──────────────────────────────────────────────────────

describe("decideKeep", () => {
  test("keeps the first strong correct output", () => {
    const baseline = snapshotFromResult(
      1,
      makeResult({ tokensGenerated: 5, outputQualityScore: 0 }),
    );
    const verify = makeResult({
      tokPerSec: 28,
      tokPerSecSamples: [28, 28.5, 27.8],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const state: ControllerState = { lastAccepted: null, bestSoFar: null, bestCorrect: null };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(true);
    expect(decision.improvedBestCorrect).toBe(true);
  });

  test("rejects slower correct output that does not beat the best", () => {
    const baselineResult = makeResult({
      tokPerSec: 30,
      tokPerSecSamples: [29.5, 30, 30.5],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const baseline = snapshotFromResult(2, baselineResult);
    const verify = makeResult({
      tokPerSec: 29.4,
      tokPerSecSamples: [29.2, 29.4, 29.5],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const state: ControllerState = { lastAccepted: baseline, bestSoFar: baseline, bestCorrect: baseline };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(false);
  });

  test("keeps significant correct-throughput gains", () => {
    const baselineResult = makeResult({
      tokPerSec: 30,
      tokPerSecSamples: [30, 30.2, 29.8],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const baseline = snapshotFromResult(3, baselineResult);
    const verify = makeResult({
      tokPerSec: 31.5,
      tokPerSecSamples: [31.4, 31.5, 31.6],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const state: ControllerState = { lastAccepted: baseline, bestSoFar: baseline, bestCorrect: baseline };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(true);
    expect(decision.improvedBestCorrect).toBe(true);
  });

  test("rejects loss of correctness after a correct baseline exists", () => {
    const baselineResult = makeResult({
      tokPerSec: 30,
      tokPerSecSamples: [30, 30.1, 29.9],
      outputText: "ĠParis.ĠTheĠcapitalĠof",
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
    });
    const baseline = snapshotFromResult(4, baselineResult);
    const verify = makeResult({
      tokPerSec: 40,
      tokPerSecSamples: [40, 40, 40],
      outputText: "ĠBerlin",
      containsReference: false,
      strongAnswer: false,
      outputQualityScore: 0,
    });
    const state: ControllerState = { lastAccepted: baseline, bestSoFar: baseline, bestCorrect: baseline };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(false);
  });

  test("keeps pre-correctness progress when tokens increase materially", () => {
    const baselineResult = makeResult({
      tokensGenerated: 5,
      outputText: "ĠThe",
      outputQualityScore: 0,
    });
    const baseline = snapshotFromResult(5, baselineResult);
    const verify = makeResult({
      tokensGenerated: 8,
      outputText: "ĠTheĠcapital",
      outputQualityScore: 1,
    });
    const state: ControllerState = { lastAccepted: baseline, bestSoFar: baseline, bestCorrect: null };
    const decision = decideKeep(verify, baseline, state);
    expect(decision.keep).toBe(true);
    expect(decision.improvedBestCorrect).toBe(false);
  });
});

// ── mergeUniqueEntries ──────────────────────────────────────────────

describe("mergeUniqueEntries", () => {
  test("dedupes near-duplicate ideas and caps memory", () => {
    const merged = mergeUniqueEntries(
      ["add real Metal per-dispatch timing behind --profile"],
      [
        "add real per-dispatch Metal timing behind `--profile`",
        "benchmark flash_attn vs MoE down-projection on-device",
      ],
      5,
    );
    expect(merged.length).toBe(2);
    expect(merged[1]).toContain("flash_attn");
  });

  test("respects maxEntries cap", () => {
    const merged = mergeUniqueEntries(
      ["a", "b", "c"],
      ["d", "e"],
      3,
    );
    expect(merged.length).toBe(3);
  });

  test("skips empty strings", () => {
    const merged = mergeUniqueEntries(["", "  ", "real idea"], [], 10);
    expect(merged.length).toBe(1);
    expect(merged[0]).toBe("real idea");
  });
});

// ── buildReflectionSummary ──────────────────────────────────────────

describe("buildReflectionSummary", () => {
  test("summarizes the last 20 cycles and highlights repeated failure basins", () => {
    const cycles = Array.from({ length: 20 }, (_, idx) => ({
      cycle: idx + 1,
      phase: "implement",
      shortTokPerSec: null,
      shortTokPerSecSamples: [],
      shortTokensGenerated: 32,
      shortContainsReference: true,
      shortStrongAnswer: false,
      shortOutputQualityScore: 2,
      shortOutputText: "ĠParis.ĠTheĠcapitalĠofĠGermanyĠisĠBerlin",
      longTokPerSec: 22,
      longTokPerSecSamples: [22],
      longTokensGenerated: 32,
      longContainsReference: true,
      longStrongAnswer: false,
      longOutputQualityScore: 2,
      longOutputText: "ĠParis.ĠTheĠcapitalĠofĠGermanyĠisĠBerlin",
      timestamp: new Date().toISOString(),
      description: "Attempted a speculative attention fix",
      kept: false,
      buildExitCode: 0,
      testExitCode: 0,
      runExitCode: 0,
      offTopic: true,
      evaluationNotes: ["contains contradictory capital/country terms"],
      decisionReason: "lost short-benchmark correctness relative to accepted baseline",
      selfAnalysis: "",
      nextIdeas: [],
    }));
    const summary = buildReflectionSummary({
      runId: "r",
      cycles,
      failedApproaches: [],
      ideas: [],
      phase: "implement",
      lastAccepted: null,
      bestSoFar: null,
      bestCorrect: null,
      currentBest: null,
      stalledCycles: 20,
      reviewSummaries: [],
      lastProfileExcerpt: null,
      lastProfileCycle: null,
      acceptedCommit: null,
    } as any);
    expect(summary).toContain("Last 20 cycles");
    expect(summary).toContain("Paris->Germany list drift");
    expect(summary).toContain("Prioritize parity tests");
  });
});

// ── buildSelfReview ─────────────────────────────────────────────────

describe("buildSelfReview", () => {
  test("returns empty string with no cycles", () => {
    const state = makeState({ cycles: [] });
    expect(buildSelfReview(state)).toBe("");
  });

  test("categorizes shader changes and reports success rate", () => {
    const cycles = Array.from({ length: 10 }, (_, i) => makeCycle({
      cycle: i + 1,
      description: i < 6 ? "Tune shader threadgroup size" : "Rearrange buffer alloc",
      kept: i < 4, // 4 kept, 6 reverted
      tokPerSec: 36 + (i < 4 ? i * 0.5 : 0),
    }));
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("Self-Review");
    expect(review).toContain("4/10 changes");
    expect(review).toContain("shader");
    expect(review).toContain("memory");
  });

  test("reports positive tok/s progress", () => {
    const cycles = Array.from({ length: 10 }, (_, i) => makeCycle({
      cycle: i + 1,
      description: "Optimize dispatch batching",
      kept: true,
      tokPerSec: 36 + i * 0.8,
    }));
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("Progress is positive");
    expect(review).toContain("dispatch");
  });

  test("warns on low progress", () => {
    const cycles = Array.from({ length: 10 }, (_, i) => makeCycle({
      cycle: i + 1,
      description: "Try random tweak",
      kept: i % 3 === 0,
      tokPerSec: 36 + (i % 3 === 0 ? 0.1 : -0.2),
    }));
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("Low progress");
    expect(review).toContain("strategic pivot");
  });

  test("does not count reverted-cycle movement as accepted progress", () => {
    const cycles = Array.from({ length: 10 }, (_, i) => makeCycle({
      cycle: i + 1,
      description: "Retune Q8 threadgroup",
      kept: false,
      tokPerSec: 34 + i * 0.4,
    }));
    const state = makeState({
      cycles,
      bestTokPerSec: 37.8,
      currentBest: { tokPerSec: 37.8, containsReference: true },
    });
    const review = buildSelfReview(state);
    expect(review).toContain("No accepted progress");
    expect(review).toContain("Do NOT treat faster reverted candidates as progress");
    expect(review).not.toContain("Progress is positive");
  });

  test("shows top performing changes", () => {
    const cycles = [
      makeCycle({ cycle: 1, description: "Fuse RMS+DMMV kernels", kept: true, tokPerSec: 38.5 }),
      makeCycle({ cycle: 2, description: "Batch MoE expert dispatch", kept: true, tokPerSec: 40.2 }),
      makeCycle({ cycle: 3, description: "Reduce encoder switches", kept: false, tokPerSec: 35 }),
      makeCycle({ cycle: 4, description: "Use bfloat16 intermediates", kept: true, tokPerSec: 41.0 }),
    ];
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("Top performing");
    expect(review).toContain("bfloat16");
    expect(review).toContain("41.0");
  });

  test("categorizes MoE and attention changes", () => {
    const cycles = [
      makeCycle({ cycle: 1, description: "Batch MoE expert routing with topk", kept: true }),
      makeCycle({ cycle: 2, description: "Optimize flash attention KV cache", kept: false }),
    ];
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("moe");
    expect(review).toContain("attention");
  });

  test("categorizes fusion changes", () => {
    const cycles = [
      makeCycle({ cycle: 1, description: "Fused SwiGLU kernel to avoid write-back", kept: true }),
    ];
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("fusion");
  });

  test("falls back to 'other' category for unrecognized descriptions", () => {
    const cycles = [
      makeCycle({ cycle: 1, description: "Reorder loop iterations", kept: true }),
    ];
    const state = makeState({ cycles });
    const review = buildSelfReview(state);
    expect(review).toContain("other");
  });
});

// ── buildPrompt ─────────────────────────────────────────────────────

describe("buildPrompt", () => {
  test("optimize phase includes bandwidth analysis and optimization targets", () => {
    const state = makeState({
      phase: "optimize",
      currentBest: { tokPerSec: 36, containsReference: true },
    });
    const result = makeResult({
      tokPerSec: 36,
      tokPerSecSamples: [35.8, 36, 36.2],
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("OPTIMIZE");
    expect(prompt).toContain("Bandwidth Analysis");
    expect(prompt).toContain("Reduce command buffer");
    expect(prompt).toContain("Threadgroup size");
    expect(prompt).toContain("MoE expert dispatch");
    expect(prompt).toContain("Fused kernels");
    expect(prompt).not.toContain("What Needs Implementation");
  });

  test("fix phase includes project structure but not bandwidth analysis", () => {
    const state = makeState({ phase: "fix" });
    const result = makeResult({ buildExitCode: 1, buildOutput: "error: undefined symbol", phase: "fix" });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("FIX");
    expect(prompt).toContain("BUILD FAILURE");
    expect(prompt).toContain("Project Structure");
    expect(prompt).not.toContain("Bandwidth Analysis");
  });

  test("includes stall warning after threshold", () => {
    const state = makeState({ stalledCycles: 5 });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("STALL");
    expect(prompt).toContain("llama.cpp");
    expect(prompt).toContain("vllm");
    expect(prompt).toContain("ggml-metal");
  });

  test("Gemma effort prompt uses Gemma model facts instead of Qwen facts", () => {
    const state = makeState({ effortId: 11 });
    const result = makeResult({
      tokPerSec: 37.8,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "The capital of France is Paris.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Model (Gemma 4 12B Q4_K_M)");
    expect(prompt).toContain("hidden_dim=2816");
    expect(prompt).not.toContain("Model (Qwen3.5-35B");
  });

  test("includes pre-stall warning at 3 cycles", () => {
    const state = makeState({ stalledCycles: 3 });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("3/5 cycles without improvement");
    expect(prompt).toContain("reference study");
  });

  test("no stall warning when stalledCycles is low", () => {
    const state = makeState({ stalledCycles: 1 });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).not.toContain("STALL");
    expect(prompt).not.toContain("reference study");
  });

  test("includes profile output when available", () => {
    const state = makeState({
      lastProfileOutput: "Phase: decode_step total=12.5ms\n  rms_norm: 0.8ms\n  dmmv_q4k: 5.2ms",
      lastProfileCycle: 5,
    });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Profile Output (cycle 5)");
    expect(prompt).toContain("dmmv_q4k: 5.2ms");
  });

  test("includes latest review summary", () => {
    const state = makeState({
      reviewSummaries: ["## Self-Review (last 10 cycles)\n\nshader: 3/5 kept"],
    });
    const result = makeResult({
      tokPerSec: 36,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Self-Review (last 10 cycles)");
    expect(prompt).toContain("shader: 3/5 kept");
  });

  test("correctness regression prompt tells agent to restore output", () => {
    const state = makeState({
      currentBest: { tokPerSec: 36, containsReference: true },
    });
    const result = makeResult({
      tokPerSec: 42,
      containsReference: false,
      strongAnswer: false,
      outputText: "ĠBerlin",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("CORRECTNESS REGRESSION");
    expect(prompt).toContain("restore correct output");
  });

  test("target reached status", () => {
    const state = makeState();
    const result = makeResult({
      tokPerSec: 52,
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("TARGET REACHED");
    expect(prompt).toContain("52.0");
  });

  test("includes benchmark samples in diagnosis", () => {
    const state = makeState();
    const result = makeResult({
      tokPerSec: 36,
      tokPerSecSamples: [35.5, 36.0, 36.5],
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "ĠParis.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("35.5");
    expect(prompt).toContain("36.5");
  });

  test("warns when benchmark samples are too noisy for direction", () => {
    const state = makeState();
    const result = makeResult({
      tokPerSec: 30.2,
      tokPerSecSamples: [37.6, 25.4, 30.2],
      containsReference: true,
      strongAnswer: true,
      outputQualityScore: 4,
      outputText: "The capital of France is Paris.",
    });
    const prompt = buildPrompt(state, result);
    expect(prompt).toContain("Benchmark variance warning");
    expect(prompt).toContain("too wide for reliable direction");
  });
});
