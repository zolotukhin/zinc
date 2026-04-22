import { afterAll, beforeAll, describe, test } from "bun:test";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { QWEN_SMOKE_CASES, resolveSmokeModel, runSmokeCase, smokeTimeoutMs } from "./test_qwen_smoke";
import { runSuite } from "./test_openai_sdk";

const MANAGED_MODEL_ROOT = join(homedir(), "Library", "Caches", "zinc", "models", "models");
const SERVER_MODEL_IDS = [
  "qwen3-8b-q4k-m",
  "gemma4-12b-q4k-m",
  "gpt-oss-20b-q4k-m",
];
const QWEN_CHAT_MODEL_IDS = [
  "qwen35-35b-a3b-q4k-xl",
  "qwen36-35b-a3b-q4k-xl",
];

const binary = process.env.ZINC_CLI_BIN ?? "./zig-out/bin/zinc";
const requireFull = process.env.ZINC_REQUIRE_FULL_TESTS === "1";
const prompt = "The capital of France is";
const thinkingPrompt = "tell me about zig";

function firstInstalledManagedModel(): string | null {
  for (const id of SERVER_MODEL_IDS) {
    if (existsSync(join(MANAGED_MODEL_ROOT, id, "model.gguf"))) return id;
  }
  return null;
}

function hasManagedModel(id: string): boolean {
  return existsSync(join(MANAGED_MODEL_ROOT, id, "model.gguf"));
}

async function waitForHealth(baseUrl: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  const healthUrl = `${baseUrl.replace(/\/v1\/?$/, "")}/health`;
  let lastError: unknown = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(healthUrl, { signal: AbortSignal.timeout(1_000) });
      if (response.status === 200) return;
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Server did not become healthy within ${timeoutMs} ms: ${String(lastError)}`);
}

async function pickOpenPort(): Promise<number> {
  return await new Promise((resolve, reject) => {
    const socket = Bun.listen({
      hostname: "127.0.0.1",
      port: 0,
      socket: { data() {}, open() {}, close() {} },
    });
    try {
      const port = socket.port;
      socket.stop(true);
      resolve(port);
    } catch (error) {
      socket.stop(true);
      reject(error);
    }
  });
}

// Large models (35B-class) take ~80s to load on Apple M1/M2 Pro because
// the entire 21 GB Q4_K_XL GGUF has to be mmap-paged into the unified
// memory budget plus the KV cache allocated. Give each attempt 180s
// before killing + retrying, and the outer deadline 240s so a single
// slow cold-start does not fail the whole test.
const HEALTH_WAIT_MS = 180_000;
const SERVER_ACQUIRE_MS = 240_000;

async function launchManagedServer(modelId: string): Promise<{ process: Bun.Subprocess; baseUrl: string }> {
  const deadline = Date.now() + SERVER_ACQUIRE_MS;
  while (Date.now() < deadline) {
    const port = await pickOpenPort();
    const candidateBase = `http://127.0.0.1:${port}/v1`;
    const child = Bun.spawn({
      cmd: [binary, "--model-id", modelId, "--port", String(port)],
      stdout: "pipe",
      stderr: "pipe",
      env: process.env,
    });
    try {
      await waitForHealth(candidateBase, HEALTH_WAIT_MS);
      return { process: child, baseUrl: candidateBase };
    } catch (error) {
      child.kill();
      await child.exited.catch(() => {});
      const exited = await Promise.race([child.exited, Promise.resolve(null)]);
      if (exited === null) throw error;
      await new Promise((resolve) => setTimeout(resolve, 2_000));
    }
  }
  throw new Error(`Local server could not acquire GPU for ${modelId} within ${SERVER_ACQUIRE_MS / 1000}s`);
}

async function stopManagedServer(process: Bun.Subprocess | null): Promise<void> {
  if (!process) return;
  process.kill();
  await process.exited.catch(() => {});
}

async function postJson(baseUrl: string, path: string, body: unknown): Promise<any> {
  const response = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(120_000),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${path}: ${text.slice(0, 400)}`);
  }
  return JSON.parse(text);
}

async function getJson(url: string): Promise<any> {
  const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${url}: ${text.slice(0, 400)}`);
  }
  return JSON.parse(text);
}

function assistantContent(body: any): string {
  return String(body?.choices?.[0]?.message?.content ?? "").trim();
}

function assistantMessageDebug(body: any): string {
  try {
    return JSON.stringify(body?.choices?.[0]?.message ?? body, null, 2);
  } catch {
    return String(body);
  }
}

function extractVisibleAnswer(text: string): string {
  const closeIdx = text.lastIndexOf("</think>");
  if (closeIdx === -1) return text;
  return text.slice(closeIdx + "</think>".length).trim();
}

function assertUsefulZigAnswer(label: string, text: string): void {
  if (!text) throw new Error(`${label} returned an empty answer`);
  const answer = extractVisibleAnswer(text);
  if (!answer) {
    throw new Error(`${label} returned reasoning but no post-</think> answer:\n${text}`);
  }
  const lower = answer.toLowerCase();
  if (!lower.includes("zig")) {
    throw new Error(`${label} did not mention Zig:\n${answer}`);
  }
  for (const marker of [
    "here's a thinking process",
    "thinking process:",
    "analyze the request",
    "user says:",
  ]) {
    if (lower.includes(marker)) {
      throw new Error(`${label} leaked meta-planning instead of answering:\n${answer}`);
    }
  }
}

for (const entry of QWEN_SMOKE_CASES) {
  const modelPath = resolveSmokeModel(entry.envName, entry.managedId);

  if (modelPath) {
    test(entry.label, async () => {
      await runSmokeCase(binary, {
        label: entry.label,
        modelPath,
        prompt,
        expectedFirstToken: entry.expectedFirstToken,
        expectedTextSubstrings: entry.expectedTextSubstrings,
      }, smokeTimeoutMs());
    }, 300_000);
  } else if (requireFull) {
    test(entry.label, () => {
      throw new Error(`Missing ${entry.envName} and no managed model for ${entry.managedId}`);
    });
  } else {
    test.skip(`${entry.label} requires ${entry.envName} or managed model ${entry.managedId}`, () => {});
  }
}

const externalBase = process.env.ZINC_API_BASE_URL;
const managedModelId = externalBase ? null : firstInstalledManagedModel();
const canRunLocalServer = !externalBase && managedModelId !== null && existsSync(binary);

let managedProcess: Bun.Subprocess | null = null;
let baseUrl: string | null = externalBase ?? null;

describe("OpenAI API smoke", () => {
  if (externalBase) {
    test("external server", async () => {
      await runSuite(externalBase);
    }, 300_000);
    return;
  }

  if (!canRunLocalServer) {
    if (requireFull) {
      test("OpenAI API smoke", () => {
        throw new Error("Missing ZINC_API_BASE_URL and no managed model installed for a local server");
      });
    } else {
      test.skip("OpenAI API smoke requires ZINC_API_BASE_URL or an installed managed model", () => {});
    }
    return;
  }

  beforeAll(async () => {
    const deadline = Date.now() + SERVER_ACQUIRE_MS;
    while (Date.now() < deadline) {
      const port = await pickOpenPort();
      baseUrl = `http://127.0.0.1:${port}/v1`;
      managedProcess = Bun.spawn({
        cmd: [binary, "--model-id", managedModelId!, "--port", String(port)],
        stdout: "pipe",
        stderr: "pipe",
        env: process.env,
      });
      try {
        await waitForHealth(baseUrl, HEALTH_WAIT_MS);
        return;
      } catch (error) {
        const exited = await Promise.race([managedProcess.exited, Promise.resolve(null)]);
        if (exited === null) throw error;
        await new Promise((resolve) => setTimeout(resolve, 2_000));
      }
    }
    throw new Error(`Local server could not acquire GPU within ${SERVER_ACQUIRE_MS / 1000}s`);
  }, 300_000);

  afterAll(async () => {
    if (managedProcess) {
      managedProcess.kill();
      await managedProcess.exited;
      managedProcess = null;
    }
  });

  test("local server", async () => {
    if (!baseUrl) throw new Error("Local server did not initialize");
    await runSuite(baseUrl);
  }, 300_000);
});

for (const modelId of QWEN_CHAT_MODEL_IDS) {
  if (!hasManagedModel(modelId)) {
    if (requireFull) {
      test(`${modelId} chat thinking smoke`, () => {
        throw new Error(`Missing managed model ${modelId}`);
      });
    } else {
      test.skip(`${modelId} chat thinking smoke requires managed model ${modelId}`, () => {});
    }
    continue;
  }

  test(`${modelId} chat thinking smoke`, async () => {
    const { process, baseUrl } = await launchManagedServer(modelId);
    try {
      const disabled = await postJson(baseUrl, "/chat/completions", {
        model: modelId,
        messages: [{ role: "user", content: thinkingPrompt }],
        max_tokens: 256,
        stream: false,
        enable_thinking: false,
      });
      const disabledText = assistantContent(disabled);
      if (!disabledText) {
        throw new Error(`${modelId} thinking disabled returned empty content\n${assistantMessageDebug(disabled)}`);
      }
      assertUsefulZigAnswer(`${modelId} thinking disabled`, disabledText);

      const enabled = await postJson(baseUrl, "/chat/completions", {
        model: modelId,
        messages: [{ role: "user", content: thinkingPrompt }],
        max_tokens: 2048,
        stream: false,
        enable_thinking: true,
      });
      const enabledText = assistantContent(enabled);
      if (!enabledText) {
        throw new Error(`${modelId} thinking enabled returned empty content\n${assistantMessageDebug(enabled)}`);
      }
      assertUsefulZigAnswer(`${modelId} thinking enabled`, enabledText);

      const models = await getJson(`${baseUrl.replace(/\/v1\/?$/, "")}/v1/models`);
      const active = models?.data?.find((entry: any) => entry?.id === modelId);
      if (!active) throw new Error(`Active model ${modelId} missing from /v1/models`);
      if (active.supports_thinking_toggle !== true) {
        throw new Error(`${modelId} should expose a thinking toggle once chat thinking is wired correctly`);
      }
    } finally {
      await stopManagedServer(process);
    }
  }, 300_000);
}
