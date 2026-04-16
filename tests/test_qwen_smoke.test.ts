import { afterAll, beforeAll, describe, test } from "bun:test";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { QWEN_SMOKE_CASES, resolveSmokeModel, runSmokeCase, smokeTimeoutMs } from "./test_qwen_smoke";
import { runSuite } from "./test_openai_sdk";

const MANAGED_MODEL_ROOT = join(homedir(), "Library", "Caches", "zinc", "models", "models");
const SERVER_MODEL_IDS = [
  "qwen3-8b-q4k-m",
  "gemma3-12b-q4k-m",
  "gemma4-12b-q4k-m",
  "gpt-oss-20b-q4k-m",
];

const binary = process.env.ZINC_CLI_BIN ?? "./zig-out/bin/zinc";
const requireFull = process.env.ZINC_REQUIRE_FULL_TESTS === "1";
const prompt = "The capital of France is";

function firstInstalledManagedModel(): string | null {
  for (const id of SERVER_MODEL_IDS) {
    if (existsSync(join(MANAGED_MODEL_ROOT, id, "model.gguf"))) return id;
  }
  return null;
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
    const deadline = Date.now() + 120_000;
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
        await waitForHealth(baseUrl, 60_000);
        return;
      } catch (error) {
        const exited = await Promise.race([managedProcess.exited, Promise.resolve(null)]);
        if (exited === null) throw error;
        await new Promise((resolve) => setTimeout(resolve, 2_000));
      }
    }
    throw new Error("Local server could not acquire GPU within 120s");
  }, 180_000);

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
