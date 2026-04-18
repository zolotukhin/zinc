import { describe, expect, test } from "bun:test";
import {
  parseTokPerSec,
  parseTokensGenerated,
  parseBandwidthUtil,
  parseEffectiveBW,
  parsePrefillPhaseBudget,
  detectPhase,
  isGarbageOutput,
  isCoherentText,
} from "./optimize_zinc";
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
    expect(
      parseTokPerSec("Generated 256 tokens in 2500.0 ms — 102.4 tok/s"),
    ).toBeCloseTo(102.4, 1);
  });

  test("extracts decode tok/s with en dash", () => {
    expect(
      parseTokPerSec("Generated 256 tokens in 2500.0 ms – 102.4 tok/s"),
    ).toBeCloseTo(102.4, 1);
  });

  test("extracts decode tok/s with hyphen", () => {
    expect(
      parseTokPerSec("Generated 256 tokens in 2500.0 ms - 102.4 tok/s"),
    ).toBeCloseTo(102.4, 1);
  });

  test("computes from generated tokens + time when no inline tok/s", () => {
    expect(parseTokPerSec("Generated 256 tokens in 2.5 s")).toBeCloseTo(
      102.4,
      0,
    );
  });

  test("computes from ms timing", () => {
    expect(parseTokPerSec("Generated 100 tokens in 1000 ms")).toBeCloseTo(
      100,
      0,
    );
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
    expect(parseTokensGenerated("info(forward): Generated 256 tokens")).toBe(
      256,
    );
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

describe("modeled bandwidth parsing", () => {
  test("extracts modeled utilization and effective bandwidth", () => {
    const output =
      "info(forward): Modeled decode bandwidth: 36.0 GB/s effective, 576 GB/s theoretical (6.3% utilization, ~3.3 MB/token)";
    expect(parseEffectiveBW(output)).toBeCloseTo(36.0, 1);
    expect(parseBandwidthUtil(output)).toBeCloseTo(6.3, 1);
  });

  test("returns null when no modeled bandwidth line is present", () => {
    expect(
      parseEffectiveBW("Generated 256 tokens in 23792.8 ms — 10.76 tok/s"),
    ).toBeNull();
    expect(
      parseBandwidthUtil("Generated 256 tokens in 23792.8 ms — 10.76 tok/s"),
    ).toBeNull();
  });
});

describe("isGarbageOutput", () => {
  test("detects repeated tokens", () => {
    expect(
      isGarbageOutput(
        "info(zinc): Output tokens (20): { 6160, 1136, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047, 6047 }",
      ),
    ).toBe(true);
  });

  test("accepts diverse tokens", () => {
    expect(
      isGarbageOutput(
        "info(zinc): Output tokens (10): { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 }",
      ),
    ).toBe(false);
  });

  test("returns false when no Output tokens line", () => {
    expect(isGarbageOutput("info(forward): Generated 256 tokens")).toBe(false);
  });

  test("handles short token lists", () => {
    expect(isGarbageOutput("info(zinc): Output tokens (2): { 100, 100 }")).toBe(
      false,
    );
  });

  test("detects numbers-only output as garbage", () => {
    expect(
      isGarbageOutput(
        "info(zinc): Output tokens (32): { 11, 220, 16, 16, 16 }\ninfo(zinc): Output text: ,Ġ11111Ġ110ĠĠ31Ġ311Ġ311111122Ġ3Ġ",
      ),
    ).toBe(true);
  });

  test("detects short repeating patterns", () => {
    expect(
      isGarbageOutput(
        "info(zinc): Output tokens (20): { 7471, 6852, 4547, 7398, 514, 4402, 536, 4547, 7398, 514, 4402, 536, 4547, 7398, 514, 4402, 536, 4547, 7398, 514 }",
      ),
    ).toBe(true);
  });
});

describe("isCoherentText", () => {
  test("detects garbage BPE subwords", () => {
    expect(
      isCoherentText(
        "info(zinc): Output text: endregionoeseriescpyĠintandidieseriescpyĠintandid",
      ),
    ).toBe(false);
  });

  test("detects coherent English", () => {
    expect(
      isCoherentText(
        "info(zinc): Output text: The capital of France is Paris. It is the largest city in the country and one of the most visited cities in the world.",
      ),
    ).toBe(true);
  });

  test("detects reasoning output", () => {
    expect(
      isCoherentText(
        "info(zinc): Output text: <think>The user is asking about the capital of France. The answer is Paris, which is the capital and largest city of France.</think>Paris.",
      ),
    ).toBe(true);
  });

  test("rejects repeating-prompt output", () => {
    expect(
      isCoherentText(
        "info(zinc): Output text: The capital of France is is not. The capital of France is is not. The capital of France is is not.",
      ),
    ).toBe(false);
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

  test("optimize when running with tok/s and coherent text", () => {
    expect(detectPhase({ ...base, tokPerSec: 110.5, coherentText: true })).toBe(
      "optimize",
    );
  });

  test("fix when running with tok/s but NOT coherent", () => {
    expect(
      detectPhase({ ...base, tokPerSec: 110.5, coherentText: false }),
    ).toBe("fix");
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
  });
});

describe("parsePrefillPhaseBudget", () => {
  const sample = [
    "info(forward): Prefill: 154 tokens in 6000.0 ms (25.67 tok/s)",
    "info(forward): Prefill profile: samples=154 avg embed=0.001 ms record=1.50 ms submit+wait=35.00 ms | totals embed=0.1 ms record=231.0 ms submit+wait=5390.0 ms",
    "info(forward): Prefill GPU phases: per-tok attn=4.50 ms moe=10.40 ms shared=0.50 ms ssm=11.80 ms tail=0.90 ms embed=0.002 ms | totals attn=693.0 moe=1601.6 shared=77.0 ssm=1817.2 tail=138.6 embed=0.3",
    "info(forward): Prefill MoE subphases totals: router=301.0 topk=120.0 gate_up=480.0 swiglu=80.0 down=540.0 weighted_acc=80.6 ms",
    "info(forward): Prefill SSM subphases totals: proj=1300.0 conv=150.0 delta=210.0 gnorm=90.0 out=67.2 ms",
  ].join("\n");

  test("parses per-token averages, totals, MoE and SSM sub-buckets", () => {
    const budget = parsePrefillPhaseBudget(sample);
    expect(budget).not.toBeNull();
    expect(budget!.perTokenMs.attn).toBeCloseTo(4.5, 2);
    expect(budget!.perTokenMs.ssm).toBeCloseTo(11.8, 2);
    expect(budget!.totalsMs.moe).toBeCloseTo(1601.6, 1);
    expect(budget!.totalsMs.ssm).toBeCloseTo(1817.2, 1);
    expect(budget!.moeTotalsMs.gate_up).toBeCloseTo(480.0, 1);
    expect(budget!.ssmTotalsMs.proj).toBeCloseTo(1300.0, 1);
  });

  test("biggestBucket picks the largest non-embed total", () => {
    const budget = parsePrefillPhaseBudget(sample)!;
    expect(budget.biggestBucket?.name).toBe("ssm");
    expect(budget.biggestBucket?.totalMs).toBeCloseTo(1817.2, 1);
  });

  test("returns null when the Prefill GPU phases line is missing", () => {
    expect(parsePrefillPhaseBudget("no phase data here")).toBeNull();
  });

  test("tolerates ANSI color codes around the log line", () => {
    const colored = `\x1b[2m${sample}\x1b[0m`;
    expect(parsePrefillPhaseBudget(colored)).not.toBeNull();
  });
});
