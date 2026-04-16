#!/usr/bin/env bun

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type SmokeCase = {
  label: string;
  modelPath: string;
  prompt: string;
  expectedFirstToken: number;
  expectedTextSubstrings: string[];
};

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function getEnv(name: string): string | null {
  const value = process.env[name];
  return value && value.length > 0 ? value : null;
}

const MANAGED_MODEL_ROOT = join(homedir(), "Library", "Caches", "zinc", "models", "models");

function managedModelPath(id: string): string {
  return join(MANAGED_MODEL_ROOT, id, "model.gguf");
}

export function resolveSmokeModel(envName: string, managedId: string): string | null {
  const fromEnv = getEnv(envName);
  if (fromEnv) return fromEnv;
  const candidate = managedModelPath(managedId);
  return existsSync(candidate) ? candidate : null;
}

export const QWEN_SMOKE_CASES: Array<{
  label: string;
  envName: string;
  managedId: string;
  expectedFirstToken: number;
  expectedTextSubstrings: string[];
}> = [
  {
    label: "Qwen3 8B smoke",
    envName: "ZINC_QWEN3_8B_MODEL",
    managedId: "qwen3-8b-q4k-m",
    expectedFirstToken: 12095,
    expectedTextSubstrings: ["Paris"],
  },
  {
    label: "Qwen3.5 35B smoke",
    envName: "ZINC_QWEN35_35B_MODEL",
    managedId: "qwen35-35b-a3b-q4k-xl",
    expectedFirstToken: 11751,
    expectedTextSubstrings: ["Paris"],
  },
];

export function smokeTimeoutMs(): number {
  return Number(getEnv("ZINC_QWEN_SMOKE_TIMEOUT_MS") ?? "120000");
}

async function readStream(stream: ReadableStream<Uint8Array> | null): Promise<string> {
  if (!stream) return "";
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let out = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    out += decoder.decode(value, { stream: true });
  }
  out += decoder.decode();
  return out;
}

function parseFirstToken(output: string): number {
  const match = output.match(/gen\[0\]: id=(\d+)/) ?? output.match(/decode\[0\]: token=(\d+)/);
  assert(match, `Missing first-token line in output:\n${output.slice(0, 4000)}`);
  return Number(match[1]);
}

function parseOutputText(output: string): string {
  const match = output.match(/Output text:\s*(.*)/) ?? output.match(/Output \(\d+ tokens\):\s*(.*)/);
  assert(match, `Missing Output line in output:\n${output.slice(0, 4000)}`);
  return match[1] ?? "";
}

async function runCli(binary: string, smokeCase: SmokeCase, timeoutMs: number): Promise<{ output: string; exitCode: number }> {
  const proc = Bun.spawn({
    cmd: [
      binary,
      "--debug",
      "--max-tokens",
      "8",
      "-m",
      smokeCase.modelPath,
      "--prompt",
      smokeCase.prompt,
    ],
    stdout: "pipe",
    stderr: "pipe",
    env: process.env,
  });

  const killer = setTimeout(() => proc.kill(), timeoutMs);
  try {
    const [stdout, stderr, exitCode] = await Promise.all([
      readStream(proc.stdout),
      readStream(proc.stderr),
      proc.exited,
    ]);
    return { output: `${stdout}\n${stderr}`, exitCode: exitCode ?? 1 };
  } finally {
    clearTimeout(killer);
  }
}

export async function runSmokeCase(binary: string, smokeCase: SmokeCase, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let result = await runCli(binary, smokeCase, timeoutMs);
  while (result.exitCode !== 0 && /already reserved by another zinc process/.test(result.output) && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 2_000));
    result = await runCli(binary, smokeCase, Math.max(10_000, deadline - Date.now()));
  }
  const { output, exitCode } = result;
  {
    assert(exitCode === 0, `${smokeCase.label} exited with code ${exitCode}\n${output.slice(0, 6000)}`);

    const firstToken = parseFirstToken(output);
    assert(
      firstToken === smokeCase.expectedFirstToken,
      `${smokeCase.label} first token mismatch: expected ${smokeCase.expectedFirstToken}, got ${firstToken}\n${output.slice(0, 6000)}`,
    );

    const text = parseOutputText(output);
    for (const needle of smokeCase.expectedTextSubstrings) {
      assert(
        text.includes(needle),
        `${smokeCase.label} output missing '${needle}'. Got:\n${text}`,
      );
    }

    console.log(`  PASS: ${smokeCase.label} → token ${firstToken}, text='${text.slice(0, 80)}'`);
  }
}

export function hasSmokeEnv(): boolean {
  return QWEN_SMOKE_CASES.every((entry) => resolveSmokeModel(entry.envName, entry.managedId) !== null);
}

export async function runSmokeSuite(binary: string): Promise<void> {
  const prompt = "The capital of France is";
  const timeoutMs = smokeTimeoutMs();

  console.log(`Testing Qwen smoke cases with ${binary}\n`);
  for (const entry of QWEN_SMOKE_CASES) {
    const modelPath = resolveSmokeModel(entry.envName, entry.managedId);
    assert(modelPath, `Missing ${entry.envName} and no managed model at ${managedModelPath(entry.managedId)}`);
    await runSmokeCase(binary, {
      label: entry.label,
      modelPath,
      prompt,
      expectedFirstToken: entry.expectedFirstToken,
      expectedTextSubstrings: entry.expectedTextSubstrings,
    }, timeoutMs);
  }
  console.log("\nQwen smoke suite passed.");
}

if (import.meta.main) {
  const binary = getEnv("ZINC_CLI_BIN") ?? "./zig-out/bin/zinc";
  runSmokeSuite(binary).catch((error) => {
    console.error(`FAILED: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  });
}
