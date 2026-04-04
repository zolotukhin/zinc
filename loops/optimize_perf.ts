#!/usr/bin/env bun
/**
 * ZINC Performance Optimization Loop
 *
 * Implements multi-hour optimization efforts defined in MULTI_HOUR_EFFORT_*.md
 * documents. Each cycle:
 *   1. Read the optimization plan document
 *   2. Build & benchmark baseline on remote RDNA4 node
 *   3. Spawn AI agent to implement ONE concrete step from the plan
 *   4. Build, run tests, benchmark
 *   5. If tok/s improved AND output correct → commit, update plan
 *   6. If regressed or broken → revert, log what went wrong
 *   7. Loop back to 3
 *
 * Usage:
 *   bun loops/optimize_perf.ts --effort 1              # Push descriptors
 *   bun loops/optimize_perf.ts --effort 2              # Fused gate+up
 *   bun loops/optimize_perf.ts --effort 3              # Batch prefill
 *   bun loops/optimize_perf.ts --effort 1 --cycles 10  # Max 10 cycles
 *   bun loops/optimize_perf.ts --effort 1 --dry-run    # Build+bench only
 *   bun loops/optimize_perf.ts --effort 3 --model gemma4  # Use Gemma 4 31B
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
  parseTokPerSec,
  parseTokensGenerated,
  isGarbageOutput,
  isCoherentText,
  parseBandwidthUtil,
  parseEffectiveBW,
} from "./optimize_zinc";

// ── Config ──────────────────────────────────────────────────────────

const REPO_ROOT = resolve(import.meta.dir, "..");
const RESULTS_DIR = resolve(REPO_ROOT, ".perf_optimize");

function loadEnv(): Record<string, string> {
  const envPath = join(REPO_ROOT, ".env");
  const vars: Record<string, string> = {};
  if (existsSync(envPath)) {
    const content = require("fs").readFileSync(envPath, "utf8") as string;
    for (const line of content.split("\n")) {
      const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.+?)\s*$/);
      if (m) vars[m[1]] = m[2];
    }
  }
  return vars;
}

const ENV = loadEnv();
const ZINC_HOST = process.env.ZINC_HOST ?? ENV.ZINC_HOST ?? "127.0.0.1";
const ZINC_PORT = Number(process.env.ZINC_PORT ?? ENV.ZINC_PORT ?? "22");
const ZINC_USER = process.env.ZINC_USER ?? ENV.ZINC_USER ?? "root";
const REMOTE_DIR = "/root/zinc";

const MODELS: Record<string, string> = {
  gemma3: "/root/models/gemma-3-12b-it-Q4_K_M.gguf",
  gemma4: "/root/models/gemma-4-31B-it-Q4_K_M.gguf",
};

const EFFORT_DOCS: Record<number, string> = {
  1: "MULTI_HOUR_EFFORT_1_PUSH_DESCRIPTORS.md",
  2: "MULTI_HOUR_EFFORT_2_FUSED_GATE_UP.md",
  3: "MULTI_HOUR_EFFORT_3_BATCH_PREFILL.md",
};

const CORRECTNESS_PROMPT = "What is the capital of France?";
const CORRECTNESS_EXPECT = "Paris";

const BLOCKED_FILE_OPS = [
  "Edit(loops/*)", "Write(loops/*)", "Edit(site/*)", "Write(site/*)",
  "Edit(docs/*)", "Write(docs/*)", "Edit(.env)", "Write(.env)",
  "Edit(AGENTS.md)", "Write(AGENTS.md)", "Edit(CLAUDE.md)", "Write(CLAUDE.md)",
  "Edit(MULTI_HOUR_EFFORT_*)", "Write(MULTI_HOUR_EFFORT_*)",
];

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)", "Bash(git revert:*)", "Bash(git restore:*)",
  "Bash(git reset:*)", "Bash(git stash:*)", "Bash(git clean:*)",
  "Bash(git push:*)", "Bash(git commit:*)",
];

// ── CLI parsing ─────────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  let effort = 0;
  let cycles = 20;
  let dryRun = false;
  let model = "gemma3";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--effort" && args[i + 1]) effort = parseInt(args[++i], 10);
    else if (args[i] === "--cycles" && args[i + 1]) cycles = parseInt(args[++i], 10);
    else if (args[i] === "--dry-run") dryRun = true;
    else if (args[i] === "--model" && args[i + 1]) model = args[++i];
  }
  if (!effort || !EFFORT_DOCS[effort]) {
    console.error("Usage: bun loops/optimize_perf.ts --effort <1|2|3> [--cycles N] [--model gemma3|gemma4]");
    console.error("  1 = Push descriptors (~3% decode speedup)");
    console.error("  2 = Fused gate+up DMMV (~1.2% decode speedup)");
    console.error("  3 = Batch prefill (~4x prefill speedup)");
    process.exit(1);
  }
  return { effort, cycles, dryRun, model };
}

// ── Helpers ─────────────────────────────────────────────────────────

const CLR = process.stdout.isTTY && !("NO_COLOR" in process.env);
const c = (code: string, t: string) => CLR ? `\x1b[${code}m${t}\x1b[0m` : t;

type RunResult = { exitCode: number; stdout: string; stderr: string };

async function run(cmd: string, args: string[], opts: { cwd?: string; timeout?: number } = {}): Promise<RunResult> {
  return new Promise((res, rej) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout ?? 120_000,
    });
    let stdout = "", stderr = "";
    child.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
    child.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });
    child.on("error", rej);
    child.on("close", (code) => res({ exitCode: code ?? 1, stdout, stderr }));
  });
}

async function ssh(command: string, timeout = 120_000): Promise<string> {
  const { stdout, stderr, exitCode } = await run("ssh", [
    "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
    "-p", String(ZINC_PORT), `${ZINC_USER}@${ZINC_HOST}`, command,
  ], { timeout });
  if (exitCode !== 0 && !stderr.includes("Warning"))
    throw new Error(`SSH failed (${exitCode}): ${stderr.slice(0, 500)}`);
  return stdout.trim();
}

async function rsyncToRemote(): Promise<void> {
  const { exitCode, stderr } = await run("rsync", [
    "-avz", "--checksum", "--delete",
    "-e", `ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no`,
    "--exclude", ".zig-cache", "--exclude", "zig-out", "--exclude", "node_modules",
    "--exclude", ".git", "--exclude", ".perf_optimize", "--exclude", ".zinc_optimize",
    "--exclude", "site", "--exclude", ".DS_Store",
    `${REPO_ROOT}/`, `${ZINC_USER}@${ZINC_HOST}:${REMOTE_DIR}/`,
  ], { timeout: 120_000 });
  if (exitCode !== 0) throw new Error(`rsync failed: ${stderr.slice(0, 300)}`);
}

// ── Build & benchmark ───────────────────────────────────────────────

type BenchResult = {
  buildOk: boolean;
  buildOutput: string;
  tokPerSec: number | null;
  correct: boolean;
  outputText: string;
  bandwidthUtil: number | null;
  error: string | null;
};

async function buildAndBench(modelPath: string): Promise<BenchResult> {
  // Compile shaders
  console.log(c("2", "  Compiling shaders..."));
  try {
    await ssh(`cd ${REMOTE_DIR}/src/shaders && for f in *.comp; do glslc --target-env=vulkan1.3 -fshader-stage=compute $f -o \${f%.comp}.spv 2>&1; done`, 60_000);
  } catch (e) {
    return { buildOk: false, buildOutput: String(e), tokPerSec: null, correct: false, outputText: "", bandwidthUtil: null, error: "shader compile failed" };
  }

  // Zig build
  console.log(c("2", "  Building..."));
  let buildOutput: string;
  try {
    buildOutput = await ssh(`cd ${REMOTE_DIR} && zig build 2>&1`, 300_000);
  } catch (e) {
    return { buildOk: false, buildOutput: String(e), tokPerSec: null, correct: false, outputText: "", bandwidthUtil: null, error: "zig build failed" };
  }
  if (buildOutput.includes("error:")) {
    return { buildOk: false, buildOutput, tokPerSec: null, correct: false, outputText: "", bandwidthUtil: null, error: "build errors" };
  }

  // Run correctness test
  console.log(c("2", "  Running correctness test..."));
  let runOutput: string;
  try {
    runOutput = await ssh(
      `cd ${REMOTE_DIR} && ./zig-out/bin/zinc -m ${modelPath} --prompt '${CORRECTNESS_PROMPT}' --chat -n 32 2>&1`,
      180_000,
    );
  } catch (e) {
    return { buildOk: true, buildOutput, tokPerSec: null, correct: false, outputText: "", bandwidthUtil: null, error: `run failed: ${e}` };
  }

  const tokPerSec = parseTokPerSec(runOutput);
  const textMatch = runOutput.match(/Output text:\s*(.+)/i);
  const outputText = textMatch ? textMatch[1].trim() : "";
  const correct = outputText.toLowerCase().includes(CORRECTNESS_EXPECT.toLowerCase());
  const bandwidthUtil = parseBandwidthUtil(runOutput);

  return { buildOk: true, buildOutput, tokPerSec, correct, outputText, bandwidthUtil, error: null };
}

// ── Agent spawn ─────────────────────────────────────────────────────

async function spawnAgent(
  effortDoc: string,
  plan: string,
  baseline: BenchResult,
  cycleNum: number,
  history: string,
): Promise<void> {
  const prompt = `You are implementing a performance optimization for the ZINC Vulkan inference engine.

## Optimization Plan
${plan}

## Current Baseline
- tok/s: ${baseline.tokPerSec?.toFixed(2) ?? "unknown"}
- bandwidth utilization: ${baseline.bandwidthUtil?.toFixed(1) ?? "unknown"}%
- output: "${baseline.outputText}" (must still contain "${CORRECTNESS_EXPECT}" after changes)

## Previous Attempts
${history || "None yet."}

## Your Task (Cycle ${cycleNum})
Implement ONE concrete step from the optimization plan above. Pick the next unfinished step.

Rules:
- Make ONE focused change. Don't try to do everything at once.
- The change must compile (zig build) and produce correct output ("${CORRECTNESS_EXPECT}").
- Test your change by building and running on the remote node:
  ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build && ./zig-out/bin/zinc -m ${MODELS.gemma3} --prompt '${CORRECTNESS_PROMPT}' --chat -n 16"
- If you need to compile shaders: glslc --target-env=vulkan1.3 -fshader-stage=compute file.comp -o file.spv
- After verifying it works, stop. The loop will benchmark and decide whether to keep or revert.

Files you may edit:
- src/compute/*.zig (forward.zig, dmmv.zig, elementwise.zig, attention.zig, argmax.zig)
- src/vulkan/*.zig (pipeline.zig, command.zig, buffer.zig, instance.zig)
- src/model/*.zig (tokenizer.zig, loader.zig, config.zig, architecture.zig)
- src/server/*.zig (routes.zig, runtime.zig)
- src/server/chat.html
- src/shaders/*.comp (GLSL compute shaders)
- src/main.zig`;

  console.log(c("1;36", `  Spawning agent for cycle ${cycleNum}...`));

  const { exitCode } = await run("claude", [
    "--print", "--dangerously-skip-permissions",
    "--allowedTools", [
      "Read", "Write", "Edit", "Glob", "Grep", "Bash",
      ...BLOCKED_GIT_OPS.map(b => `!${b}`),
      ...BLOCKED_FILE_OPS.map(b => `!${b}`),
    ].join(","),
    "-p", prompt,
  ], { cwd: REPO_ROOT, timeout: 600_000 });

  if (exitCode !== 0) {
    console.log(c("1;31", `  Agent exited with code ${exitCode}`));
  }
}

// ── Main loop ───────────────────────────────────────────────────────

async function main() {
  const { effort, cycles, dryRun, model } = parseArgs();
  const modelPath = MODELS[model] ?? MODELS.gemma3;
  const effortFile = EFFORT_DOCS[effort];
  const plan = await readFile(join(REPO_ROOT, effortFile), "utf8");

  await mkdir(RESULTS_DIR, { recursive: true });

  console.log(c("1;37", `\n╔══════════════════════════════════════════════════════════╗`));
  console.log(c("1;37", `║  ZINC Performance Optimization Loop — Effort ${effort}          ║`));
  console.log(c("1;37", `║  ${effortFile.padEnd(54)}║`));
  console.log(c("1;37", `║  Model: ${model.padEnd(49)}║`));
  console.log(c("1;37", `╚══════════════════════════════════════════════════════════╝\n`));

  // Step 1: Sync and get baseline
  console.log(c("1;33", "── Baseline ──────────────────────────────────────────────"));
  await rsyncToRemote();
  const baseline = await buildAndBench(modelPath);

  if (!baseline.buildOk) {
    console.error(c("1;31", "Baseline build failed! Fix build errors first."));
    process.exit(1);
  }
  if (!baseline.correct) {
    console.error(c("1;31", `Baseline output incorrect: "${baseline.outputText}". Fix correctness first.`));
    process.exit(1);
  }

  console.log(c("1;32", `  Baseline: ${baseline.tokPerSec?.toFixed(2)} tok/s, BW: ${baseline.bandwidthUtil?.toFixed(1)}%`));
  console.log(c("1;32", `  Output: "${baseline.outputText.slice(0, 80)}"`));

  let bestTokPerSec = baseline.tokPerSec ?? 0;
  let history = "";

  // Step 2: Optimization cycles
  for (let cycle = 1; cycle <= cycles; cycle++) {
    console.log(c("1;33", `\n── Cycle ${cycle}/${cycles} ──────────────────────────────────────────`));

    if (dryRun) {
      console.log(c("2", "  Dry run — skipping agent."));
      break;
    }

    // Save current state for revert
    const { stdout: stashRef } = await run("git", ["stash", "create"], { cwd: REPO_ROOT });
    const canRevert = stashRef.trim().length > 0;

    // Spawn agent
    await spawnAgent(effortFile, plan, baseline, cycle, history);

    // Sync and benchmark
    console.log(c("2", "  Syncing changes..."));
    await rsyncToRemote();
    const result = await buildAndBench(modelPath);

    const improved = result.tokPerSec != null && result.tokPerSec > bestTokPerSec * 1.001;
    const correct = result.correct;
    const broken = !result.buildOk || !correct;

    const delta = result.tokPerSec != null && bestTokPerSec > 0
      ? ((result.tokPerSec - bestTokPerSec) / bestTokPerSec * 100).toFixed(2)
      : "?";

    if (broken) {
      console.log(c("1;31", `  ❌ BROKEN: ${result.error ?? "incorrect output"}`));
      console.log(c("1;31", `     Output: "${result.outputText?.slice(0, 80)}"`));
      history += `\nCycle ${cycle}: REVERTED — ${result.error ?? "incorrect output"}`;

      // Revert
      if (canRevert) {
        await run("git", ["checkout", "."], { cwd: REPO_ROOT });
        console.log(c("2", "  Reverted changes."));
      }
    } else if (improved) {
      console.log(c("1;32", `  ✅ IMPROVED: ${result.tokPerSec?.toFixed(2)} tok/s (+${delta}%)`));
      bestTokPerSec = result.tokPerSec!;
      history += `\nCycle ${cycle}: KEPT — ${result.tokPerSec?.toFixed(2)} tok/s (+${delta}%)`;

      // Commit
      await run("git", ["add", "-A"], { cwd: REPO_ROOT });
      await run("git", ["commit", "-m", `perf(effort-${effort}): cycle ${cycle} — ${result.tokPerSec?.toFixed(2)} tok/s (+${delta}%)`], { cwd: REPO_ROOT });
      console.log(c("2", "  Committed."));
    } else {
      console.log(c("1;33", `  ⚠ NO IMPROVEMENT: ${result.tokPerSec?.toFixed(2)} tok/s (${delta}%)`));
      history += `\nCycle ${cycle}: REVERTED — no improvement (${result.tokPerSec?.toFixed(2)} tok/s, ${delta}%)`;

      // Revert
      if (canRevert) {
        await run("git", ["checkout", "."], { cwd: REPO_ROOT });
        console.log(c("2", "  Reverted changes."));
      }
    }

    // Log cycle result
    const logEntry = {
      cycle, effort,
      tokPerSec: result.tokPerSec,
      bandwidthUtil: result.bandwidthUtil,
      correct: result.correct,
      improved, broken,
      outputText: result.outputText?.slice(0, 200),
      timestamp: new Date().toISOString(),
    };
    const logPath = join(RESULTS_DIR, `effort_${effort}_log.jsonl`);
    await writeFile(logPath, JSON.stringify(logEntry) + "\n", { flag: "a" });
  }

  // Summary
  console.log(c("1;37", `\n══════════════════════════════════════════════════════════`));
  console.log(c("1;37", `  Effort ${effort} complete.`));
  console.log(c("1;37", `  Baseline: ${baseline.tokPerSec?.toFixed(2)} tok/s`));
  console.log(c("1;37", `  Best:     ${bestTokPerSec.toFixed(2)} tok/s`));
  if (bestTokPerSec > (baseline.tokPerSec ?? 0)) {
    const gain = ((bestTokPerSec - (baseline.tokPerSec ?? 0)) / (baseline.tokPerSec ?? 1) * 100).toFixed(1);
    console.log(c("1;32", `  Gain:     +${gain}%`));
  }
  console.log(c("1;37", `══════════════════════════════════════════════════════════\n`));
}

main().catch((e) => {
  console.error(c("1;31", `Fatal: ${e}`));
  process.exit(1);
});
