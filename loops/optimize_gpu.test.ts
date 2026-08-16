import { describe, expect, test } from "bun:test";
import {
  buildAgentPrompt,
  buildRemoteLlamaBenchCommand,
  buildRemoteLlamaCliCommand,
  buildRemoteZincCommand,
  buildSshOptions,
  changedSince,
  combinedCommandOutput,
  isAgentAuthFailure,
  managedCachePath,
  parseArgsFrom,
  parseLlamaBenchMetrics,
  parseLlamaCliMetrics,
  parseZincMetrics,
  resolveModelTarget,
  sshFailureSummary,
  stateTargetMismatchReason,
  type LoopOptions,
} from "./optimize_gpu";

const env = {
  ZINC_HOST: "gpu.local",
  ZINC_PORT: "8888",
  ZINC_USER: "tempuser",
  ZINC_REMOTE_HOME: "/home/tempuser",
  ZINC_REMOTE_XDG_CACHE_HOME: "/home/tempuser/.cache",
  ZINC_REMOTE_DIR: "/home/tempuser/zinc-loop",
};

describe("optimize_gpu args and model resolution", () => {
  test("defaults to the smaller Gemma 4 preset", () => {
    const opts = parseArgsFrom([], env);
    const target = resolveModelTarget(opts);

    expect(opts.model).toBe("gemma4-26b-a4b-q4k-m");
    expect(target.label).toBe("Gemma 4 26B-A4B MoE Q4_K_M");
    expect(target.modelId).toBe("gemma4-26b-a4b-q4k-m");
    expect(target.promptMode).toBe("chat");
  });

  test("allows env to choose the default model preset", () => {
    const opts = parseArgsFrom([], {
      ...env,
      ZINC_GPU_MODEL: "qwen35-9b-q4k-m",
    });
    const target = resolveModelTarget(opts);

    expect(opts.model).toBe("qwen35-9b-q4k-m");
    expect(target.label).toBe("Qwen3.5 9B Q4_K_M");
  });

  test("rejects reusing a run id with a different model target", () => {
    const opts = parseArgsFrom([], env);
    const target = resolveModelTarget(opts);
    const previous = {
      options: {
        modelKey: "qwen35-9b-q4k-m",
        modelPath: "/home/tempuser/.cache/zinc/models/models/qwen35-9b-q4k-m/model.gguf",
        metric: "decode",
        host: "gpu.local",
        user: "tempuser",
        port: 8888,
        remoteDir: "/home/tempuser/zinc-loop",
      },
    };

    expect(stateTargetMismatchReason(previous, target, opts)).toContain("run was created for qwen35-9b-q4k-m");
  });

  test("supports agent, resume, and model selection", () => {
    const opts = parseArgsFrom([
      "--agent", "claude",
      "--resume",
      ".gpu_optimize/old-run",
      "--model-id", "qwen35-9b-q4k-m",
      "--metric", "prefill",
      "--context", "4096",
      "--cycles", "7",
      "--max-stall-cycles", "12",
      "--skip-llama",
      "--continue-after-llama",
    ], env);

    expect(opts.agent).toBe("claude");
    expect(opts.resume).toBe(true);
    expect(opts.resumeDir).toBe(".gpu_optimize/old-run");
    expect(opts.modelId).toBe("qwen35-9b-q4k-m");
    expect(opts.metric).toBe("prefill");
    expect(opts.contextTokens).toBe(4096);
    expect(opts.cycles).toBe(7);
    expect(opts.maxStallCycles).toBe(12);
    expect(opts.skipLlama).toBe(true);
    expect(opts.continueAfterLlama).toBe(true);
  });

  test("maps managed model ids to the ZINC cache layout", () => {
    expect(managedCachePath("/cache", "qwen35-9b-q4k-m")).toBe("/cache/zinc/models/models/qwen35-9b-q4k-m/model.gguf");
  });

  test("resolves presets to same-machine GGUF paths", () => {
    const opts = parseArgsFrom(["--model", "qwen35-9b-q4k-m"], env);
    const target = resolveModelTarget(opts);

    expect(target.modelId).toBe("qwen35-9b-q4k-m");
    expect(target.modelPath).toBe("/home/tempuser/.cache/zinc/models/models/qwen35-9b-q4k-m/model.gguf");
    expect(target.promptMode).toBe("raw");
    expect(target.maxTokens).toBe(128);
    expect(target.prompt).toContain("Important fact: Paris is the capital of France.");
  });

  test("uses Qwen3.8 for the active dense 27B preset", () => {
    const opts = parseArgsFrom(["--model", "qwen38-27b-q4k-m"], env);
    const target = resolveModelTarget(opts);

    expect(target.label).toBe("Qwen3.8 27B Dense Q4_K_M");
    expect(target.modelId).toBe("qwen38-27b-q4k-m");
    expect(target.modelPath).toBe("/home/tempuser/.cache/zinc/models/models/qwen38-27b-q4k-m/model.gguf");
    expect(target.promptMode).toBe("chat");
  });

  test("prefers Intel env defaults for the generic GPU loop", () => {
    const opts = parseArgsFrom([], {
      ZINC_HOST: "rdna.local",
      ZINC_PORT: "2222",
      ZINC_USER: "root",
      ZINC_INTEL_HOST: "intel.local",
      ZINC_INTEL_PORT: "8888",
      ZINC_INTEL_USER: "tempuser",
      ZINC_INTEL_WORKDIR: "/home/tempuser/zinc-intel-loop",
      ZINC_INTEL_XDG_CACHE_HOME: "/home/tempuser/.cache",
      ZINC_INTEL_REMOTE_LIBC_CONF: "/workspace/zinc/.build-support/libc.conf",
    });

    expect(opts.host).toBe("intel.local");
    expect(opts.port).toBe(8888);
    expect(opts.user).toBe("tempuser");
    expect(opts.remoteDir).toBe("/home/tempuser/zinc-intel-loop");
    expect(opts.xdgCacheHome).toBe("/home/tempuser/.cache");
    expect(opts.remoteLibcConf).toBe("/workspace/zinc/.build-support/libc.conf");
  });

  test("supports temporary password auth for Intel bootstrap nodes", () => {
    const opts = parseArgsFrom([], {
      ZINC_INTEL_HOST: "intel.local",
      ZINC_INTEL_PORT: "8888",
      ZINC_INTEL_USER: "tempuser",
      ZINC_INTEL_SSH_PASSWORD: "not-for-command-lines",
    });

    expect(opts.sshPasswordEnvVar).toBe("ZINC_INTEL_SSH_PASSWORD");
    expect(opts.sshPasswordFile).toBeNull();
    expect(buildSshOptions(opts)).toContain("BatchMode=no");
    expect(buildSshOptions(opts)).toContain("NumberOfPasswordPrompts=1");
    expect(buildSshOptions(opts)).toContain("ConnectTimeout=15");
  });

  test("keeps keyless default SSH in batch mode when no password auth is configured", () => {
    const opts = parseArgsFrom([], {
      ZINC_INTEL_HOST: "intel.local",
      ZINC_INTEL_PORT: "8888",
      ZINC_INTEL_USER: "tempuser",
    });

    expect(opts.sshPasswordEnvVar).toBeNull();
    expect(opts.sshPasswordFile).toBeNull();
    expect(buildSshOptions(opts)).toContain("BatchMode=yes");
    expect(buildSshOptions(opts)).toContain("ConnectTimeout=15");
  });

  test("derives remote defaults from an explicit SSH user", () => {
    const opts = parseArgsFrom(["--user", "alice"], { ZINC_HOST: "gpu.local" });

    expect(opts.remoteHome).toBe("/home/alice");
    expect(opts.remoteDir).toBe("/home/alice/zinc-gpu-loop");
    expect(opts.xdgCacheHome).toBe("/home/alice/.cache");
    expect(opts.llamaDir).toBe("/home/alice/llama.cpp");
  });

  test("derives cache and llama defaults from an explicit remote home", () => {
    const opts = parseArgsFrom(["--remote-home", "/mnt/work"], { ZINC_USER: "tempuser" });

    expect(opts.remoteHome).toBe("/mnt/work");
    expect(opts.remoteDir).toBe("/mnt/work/zinc-gpu-loop");
    expect(opts.xdgCacheHome).toBe("/mnt/work/.cache");
    expect(opts.llamaDir).toBe("/mnt/work/llama.cpp");
  });
});

describe("optimize_gpu remote commands", () => {
  test("builds a managed ZINC command with model-id and raw mode", () => {
    const opts = parseArgsFrom(["--model", "qwen35-9b-q4k-m", "--remote-env", "ZINC_DEBUG=1", "--context", "4096"], env);
    const target = resolveModelTarget(opts);
    const command = buildRemoteZincCommand(opts, target);

    expect(command).toContain("cd '/home/tempuser/zinc-loop'");
    expect(command).toContain("XDG_CACHE_HOME='/home/tempuser/.cache'");
    expect(command).toContain("ZINC_DEBUG=1");
    expect(command).toContain("--model-id 'qwen35-9b-q4k-m'");
    expect(command).toContain("--raw");
    expect(command).toContain("-n 128");
    expect(command).toContain("-c 4096");
  });

  test("builds llama-bench against the same resolved model file", () => {
    const opts = parseArgsFrom(["--model", "qwen35-9b-q4k-m", "--samples", "5"], env);
    const target = resolveModelTarget(opts);
    const command = buildRemoteLlamaBenchCommand(opts, target);

    expect(command).toContain("/home/tempuser/llama.cpp/build/bin/llama-bench");
    expect(command).toContain("-m '/home/tempuser/.cache/zinc/models/models/qwen35-9b-q4k-m/model.gguf'");
    expect(command).toContain("-r 5");
    expect(command).toContain("-fa 1");
  });

  test("builds same-prompt llama.cpp command by default", () => {
    const opts = parseArgsFrom(["--model", "qwen35-9b-q4k-m", "--context", "4096"], env);
    const target = resolveModelTarget(opts);
    const command = buildRemoteLlamaCliCommand(opts, target);

    expect(command).toContain("llama-completion");
    expect(command).toContain("llama-cli");
    expect(command).toContain("--no-conversation");
    expect(command).toContain("-m '/home/tempuser/.cache/zinc/models/models/qwen35-9b-q4k-m/model.gguf'");
    expect(command).toContain("-p 'Benchmark context only.");
    expect(command).toContain("-n 128");
    expect(command).toContain("--device Vulkan1");
    expect(command).toContain("-st");
    expect(command).toContain("-fa 1");
    expect(command).toContain("-c 4096");
    expect(command).toContain("-b 256");
    expect(command).toContain("-ub 128");
    expect(command).toContain("timeout 300s");
  });
});

describe("optimize_gpu parsers", () => {
  test("detects fatal agent authentication failures", () => {
    expect(isAgentAuthFailure({
      exitCode: 1,
      signal: null,
      stdout: "",
      stderr: "Failed to authenticate. API Error: 401 Invalid authentication credentials",
    })).toBe(true);

    expect(isAgentAuthFailure({
      exitCode: 1,
      signal: null,
      stdout: "remote command failed (1): err: FenceWaitFailed",
      stderr: "",
    })).toBe(false);

    expect(isAgentAuthFailure({
      exitCode: 0,
      signal: null,
      stdout: "API Error: 401 Invalid authentication credentials",
      stderr: "",
    })).toBe(false);
  });

  test("extracts distinct ZINC decode and prefill rates", () => {
    const metrics = parseZincMetrics(`
info(forward): Prefill complete: 12 tokens in 60.0 ms (200.00 tok/s)
info(forward): Generated 32 tokens in 320.0 ms — 100.00 tok/s (10.0 ms/tok)
Output: "Paris"
`, ["Paris"]);

    expect(metrics.prefillTokPerSec).toBe(200);
    expect(metrics.decodeTokPerSec).toBe(100);
    expect(metrics.promptTokens).toBe(12);
    expect(metrics.outputText).toBe("Paris");
    expect(metrics.coherent).toBe(true);
  });

  test("does not report a decode rate from prefill-only output", () => {
    const metrics = parseZincMetrics(`
info(forward): Prefill complete: 12 tokens in 60.0 ms (200.00 tok/s)
Output: "Paris"
`, ["Paris"]);

    expect(metrics.prefillTokPerSec).toBe(200);
    expect(metrics.decodeTokPerSec).toBeNull();
    expect(metrics.promptTokens).toBe(12);
  });

  test("extracts llama-bench JSON tg and pp rows", () => {
    const llama = parseLlamaBenchMetrics(JSON.stringify([
      { test: "pp128", n_prompt: 128, n_gen: 0, avg_ts: 512.5 },
      { test: "tg128", n_prompt: 0, n_gen: 128, avg_ts: 93.25 },
    ]));

    expect(llama.prefillTokPerSec).toBe(512.5);
    expect(llama.decodeTokPerSec).toBe(93.25);
    expect(llama.promptTokens).toBe(128);
    expect(llama.source).toBe("llama-bench");
  });

  test("extracts llama.cpp prompt and decode timings", () => {
    const llama = parseLlamaCliMetrics(`
ZINC_LLAMA_SOURCE=llama-completion
llama_perf_context_print: prompt eval time =   152.58 ms /    16 tokens (    9.54 ms per token,   104.87 tokens per second)
llama_perf_context_print:        eval time =   474.72 ms /    15 runs   (   31.65 ms per token,    31.60 tokens per second)
`);

    expect(llama.prefillTokPerSec).toBe(104.87);
    expect(llama.decodeTokPerSec).toBe(31.60);
    expect(llama.promptTokens).toBe(16);
    expect(llama.source).toBe("llama-completion");
  });

  test("combines stdout and stderr so ZINC logs are parsed", () => {
    const output = combinedCommandOutput({
      stdout: "info(zinc): Output text: Paris\n",
      stderr: "info(forward): Generated 16 tokens in 160.0 ms — 100.00 tok/s (10.0 ms/tok)\n",
    });
    const metrics = parseZincMetrics(output, ["Paris"]);

    expect(metrics.decodeTokPerSec).toBe(100);
    expect(metrics.coherent).toBe(true);
  });

  test("summarizes SSH failures without exposing endpoints", () => {
    expect(sshFailureSummary({
      stdout: "",
      stderr: "ssh: connect to host bench-host.example.invalid port 2222: Operation timed out\n",
    })).toBe("ssh: connect to host <host> port <port>: Operation timed out");

    expect(sshFailureSummary({
      stdout: "err: FenceWaitFailed\nextra diagnostic",
      stderr: "",
    })).toBe("err: FenceWaitFailed");
  });
});

describe("optimize_gpu dirty-tree helpers", () => {
  test("reports only files introduced after the cycle starts", () => {
    expect(changedSince(["README.md", "src/main.zig"], ["README.md", "src/main.zig", "src/new.zig"])).toEqual(["src/new.zig"]);
  });
});

describe("optimize_gpu agent prompt", () => {
  test("includes remote target and guardrails", () => {
    const opts = parseArgsFrom(["--model", "qwen35-9b-q4k-m"], env);
    const target = resolveModelTarget(opts);
    const baseline = {
      metric: "decode",
      value: 75,
      samples: [74, 75, 76],
      decodeSamples: [74, 75, 76],
      prefillSamples: [210, 211, 212],
      promptTokenSamples: [128, 128, 128],
      outputText: "Paris",
      coherent: true,
    } satisfies Parameters<typeof buildAgentPrompt>[3];
    const state = {
      startedAt: "2026-05-12T00:00:00.000Z",
      updatedAt: "2026-05-12T00:00:00.000Z",
      runId: "test",
      options: {
        modelKey: target.key,
        modelPath: target.modelPath,
        metric: "decode",
        host: opts.host,
        user: opts.user,
        port: opts.port,
        remoteDir: opts.remoteDir,
      },
      best: baseline,
      llamaBaseline: {
        decodeTokPerSec: 50,
        prefillTokPerSec: 500,
        promptTokens: 128,
        raw: "[]",
      },
      cycles: [],
      failedApproaches: [],
    } satisfies Parameters<typeof buildAgentPrompt>[0];

    const prompt = buildAgentPrompt(state, opts, target, baseline, state.llamaBaseline);

    expect(prompt).toContain("remote: tempuser@gpu.local:8888 /home/tempuser/zinc-loop");
    expect(prompt).toContain("Objective: beat llama.cpp on both sustained decode and prompt prefill for Qwen3.5 9B Q4_K_M.");
    expect(prompt).toContain("Current attack metric: prefill");
    expect(prompt).toContain("All-cycle memory:");
    expect(prompt).toContain("decode=75.00 [74.00, 75.00, 76.00]");
    expect(prompt).toContain("prefill=211.00 [210.00, 211.00, 212.00]");
    expect(prompt).toContain("measured prompt shape: 128tok [128, 128, 128]");
    expect(prompt).toContain("Do not commit, push, reset, stash, or edit secrets.");
    expect(prompt).toContain("STEP_KIND");
  });
});
