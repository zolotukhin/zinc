import { describe, expect, test } from "bun:test";
import {
  buildReflectionSummary,
  decideKeep,
  detectPhase,
  evaluateOutputText,
  mergeUniqueEntries,
  snapshotFromResult,
} from "./implement_metal";
import type { BuildRunResult, ControllerState } from "./implement_metal";

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
});

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
});

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
});

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
