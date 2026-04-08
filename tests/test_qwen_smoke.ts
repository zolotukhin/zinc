#!/usr/bin/env bun

type SmokeCase = {
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
  const match = output.match(/decode\[0\]: token=(\d+)/);
  assert(match, `Missing decode[0] in output:\n${output.slice(0, 4000)}`);
  return Number(match[1]);
}

function parseOutputText(output: string): string {
  const match = output.match(/Output text:\s*(.*)/);
  assert(match, `Missing Output text in output:\n${output.slice(0, 4000)}`);
  return match[1] ?? "";
}

async function runSmokeCase(binary: string, smokeCase: SmokeCase, timeoutMs: number): Promise<void> {
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
    const output = `${stdout}\n${stderr}`;
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
  } finally {
    clearTimeout(killer);
  }
}

export function hasSmokeEnv(): boolean {
  return !!getEnv("ZINC_QWEN3_8B_MODEL") && !!getEnv("ZINC_QWEN35_35B_MODEL");
}

export async function runSmokeSuite(binary: string): Promise<void> {
  const model8b = getEnv("ZINC_QWEN3_8B_MODEL");
  const model35b = getEnv("ZINC_QWEN35_35B_MODEL");
  assert(model8b, "Missing ZINC_QWEN3_8B_MODEL");
  assert(model35b, "Missing ZINC_QWEN35_35B_MODEL");

  const timeoutMs = Number(getEnv("ZINC_QWEN_SMOKE_TIMEOUT_MS") ?? "120000");
  const prompt = "The capital of France is";
  const cases: SmokeCase[] = [
    {
      label: "Qwen3 8B smoke",
      modelPath: model8b,
      prompt,
      expectedFirstToken: 12095,
      expectedTextSubstrings: ["Paris"],
    },
    {
      label: "Qwen3.5 35B smoke",
      modelPath: model35b,
      prompt,
      expectedFirstToken: 11751,
      expectedTextSubstrings: ["Paris"],
    },
  ];

  console.log(`Testing Qwen smoke cases with ${binary}\n`);
  for (const smokeCase of cases) {
    await runSmokeCase(binary, smokeCase, timeoutMs);
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
