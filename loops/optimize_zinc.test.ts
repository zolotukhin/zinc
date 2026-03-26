import { describe, expect, test } from "bun:test";
import { parseTokPerSec, parseTokensGenerated, detectPhase } from "./optimize_zinc";
import type { BuildRunResult, Phase } from "./optimize_zinc";

describe("parseTokPerSec", () => {
  test("extracts tok/s from output", () => {
    expect(parseTokPerSec("Generation: 110.5 tok/s")).toBe(110.5);
  });

  test("extracts tok/s with integer", () => {
    expect(parseTokPerSec("Running at 98 tok/s")).toBe(98);
  });

  test("returns null for no tok/s", () => {
    expect(parseTokPerSec("Build successful")).toBeNull();
  });

  test("extracts from generated tokens + time", () => {
    expect(parseTokPerSec("Generated 256 tokens in 2.5s")).toBeCloseTo(102.4, 0);
  });

  test("handles ms timing", () => {
    expect(parseTokPerSec("Generated 100 tokens in 1000ms")).toBeCloseTo(100, 0);
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

describe("detectPhase", () => {
  const base: BuildRunResult = {
    buildExitCode: 0,
    buildOutput: "",
    runExitCode: 0,
    runOutput: "",
    phase: "fix",
    tokPerSec: null,
    tokensGenerated: 0,
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
