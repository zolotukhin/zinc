import { describe, expect, test } from "bun:test";
import { parseTokPerSec, parseTokensGenerated, detectPhase, isGarbageOutput, isCoherentText } from "./optimize_zinc";
import type { BuildRunResult, Phase } from "./optimize_zinc";

describe("parseTokPerSec", () => {
  test("prefers decode tok/s over prefill tok/s", () => {
    const output = [
      "info(forward): Prefill complete: 10 tokens in 1.3 ms (7459.27 tok/s)",
      "info(forward): Generated 256 tokens in 5635.3 ms — 45.43 tok/s (22.0 ms/tok)",
    ].join("\n");
    expect(parseTokPerSec(output)).toBeCloseTo(45.43, 1);
  });

  test("extracts decode tok/s with em dash", () => {
    expect(parseTokPerSec("Generated 256 tokens in 2500.0 ms — 102.4 tok/s")).toBeCloseTo(102.4, 1);
  });

  test("extracts decode tok/s with en dash", () => {
    expect(parseTokPerSec("Generated 256 tokens in 2500.0 ms – 102.4 tok/s")).toBeCloseTo(102.4, 1);
  });

  test("extracts decode tok/s with hyphen", () => {
    expect(parseTokPerSec("Generated 256 tokens in 2500.0 ms - 102.4 tok/s")).toBeCloseTo(102.4, 1);
  });

  test("computes from generated tokens + time when no inline tok/s", () => {
    expect(parseTokPerSec("Generated 256 tokens in 2.5 s")).toBeCloseTo(102.4, 0);
  });

  test("computes from ms timing", () => {
    expect(parseTokPerSec("Generated 100 tokens in 1000 ms")).toBeCloseTo(100, 0);
  });

  test("falls back to any tok/s when no Generated line", () => {
    expect(parseTokPerSec("Running at 98 tok/s")).toBe(98);
  });

  test("returns null for no tok/s", () => {
    expect(parseTokPerSec("Build successful")).toBeNull();
  });
});

describe("parseTokensGenerated", () => {
  test("extracts token count from ZINC output", () => {
    expect(parseTokensGenerated("info(forward): Generated 256 tokens")).toBe(256);
  });

  test("extracts 1 token", () => {
    expect(parseTokensGenerated("Generated 1 tokens")).toBe(1);
  });

  test("returns 0 for no match", () => {
    expect(parseTokensGenerated("Build failed")).toBe(0);
  });

  test("extracts from full output", () => {
    const output = `info(loader): Loaded 733 tensors
info(forward): Generating: 0 prompt tokens, max 256 output tokens
info(forward): Prefill complete at position 0
info(forward): Generated 256 tokens`;
    expect(parseTokensGenerated(output)).toBe(256);
  });
});

describe("isGarbageOutput", () => {
  test("detects repeated tokens", () => {
    expect(isGarbageOutput(
      "info(zinc): Output tokens (20): { 6160, 1136, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047 }"
    )).toBe(true);
  });

  test("accepts diverse tokens", () => {
    expect(isGarbageOutput(
      "info(zinc): Output tokens (10): { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 }"
    )).toBe(false);
  });

  test("returns false when no Output tokens line", () => {
    expect(isGarbageOutput("info(forward): Generated 256 tokens")).toBe(false);
  });

  test("handles short token lists", () => {
    expect(isGarbageOutput("info(zinc): Output tokens (2): { 100, 100 }")).toBe(false);
  });

  test("detects short repeating patterns", () => {
    expect(isGarbageOutput(
      "info(zinc): Output tokens (20): { 7471, 6852, 4547, 7398, 514, 4402, 536, 4547, 7398, 514, 4402, 536, 4547, 7398, 514, 4402, 536, 4547, 7398, 514 }"
    )).toBe(true);
  });
});

describe("isCoherentText", () => {
  test("detects garbage BPE subwords", () => {
    expect(isCoherentText(
      "info(zinc): Output text: endregionoeseriescpyĠintandidieseriescpyĠintandid"
    )).toBe(false);
  });

  test("detects coherent English", () => {
    expect(isCoherentText(
      "info(zinc): Output text: The capital of France is Paris. It is the largest city in the country and one of the most visited cities in the world."
    )).toBe(true);
  });

  test("detects reasoning output", () => {
    expect(isCoherentText(
      "info(zinc): Output text: <think>The user is asking about the capital of France. The answer is Paris, which is the capital and largest city of France.</think>Paris."
    )).toBe(true);
  });

  test("returns false for no text line", () => {
    expect(isCoherentText("info(forward): Generated 256 tokens")).toBe(false);
  });
});

describe("detectPhase", () => {
  const base: BuildRunResult = {
    buildExitCode: 0,
    buildOutput: "",
    runExitCode: 0,
    runOutput: "",
    phase: "fix",
    tokPerSec: null,
    tokensGenerated: 0,
    garbageOutput: false,
    coherentText: false,
    bandwidthUtil: null,
    effectiveBW: null,
    error: null,
  };

  test("fix when build fails", () => {
    expect(detectPhase({ ...base, buildExitCode: 1 })).toBe("fix");
  });

  test("fix when run fails", () => {
    expect(detectPhase({ ...base, runExitCode: 1 })).toBe("fix");
  });

  test("fix when error set", () => {
    expect(detectPhase({ ...base, error: "segfault" })).toBe("fix");
  });

  test("optimize when running with tok/s", () => {
    expect(detectPhase({ ...base, tokPerSec: 110.5 })).toBe("optimize");
  });

  test("fix when running but no tok/s yet", () => {
    expect(detectPhase({ ...base, tokPerSec: null })).toBe("fix");
  });
});

describe("no hardcoded private info", () => {
  test("env vars are used for host config", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_zinc.ts").text();
    expect(src).toContain("ZINC_HOST");
    expect(src).toContain("ZINC_PORT");
    expect(src).toContain("ZINC_USER");
    expect(src).not.toContain('"71.');
    expect(src).not.toContain("'71.");
  });
});
