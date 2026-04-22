#!/usr/bin/env bun

/**
 * Integration test for ZINC's OpenAI-compatible API server.
 * Run with: bun tests/test_openai_sdk.ts [--base-url http://localhost:8080/v1]
 *
 * Optional: bun add openai
 */

// The first inference request after server startup triggers pipeline-state
// compilation for every Metal / Vulkan shader the decode graph touches —
// ~3s extra on Qwen3-8B on M1 Pro. 35B-class chat-thinking tests can emit
// up to max_tokens=2048 and take much longer, so keep the ceiling high.
// Metadata endpoints (/health, /models, error paths) stay at 5s.
const INFERENCE_TIMEOUT_MS = 180_000;

type ModelsResponse = {
  object: string;
  data: Array<{ id: string; active?: boolean }>;
};

type HealthResponse = {
  status: string;
  model: string;
  active_requests?: number;
  queued_requests?: number;
  uptime_seconds?: number;
};

type ChatCompletionResponse = {
  object: string;
  choices: Array<{ message: { content: string } }>;
  usage?: { completion_tokens: number };
};

type ChatChunk = {
  object: string;
  choices: Array<{ delta?: { role?: string; content?: string } }>;
};

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function getArgValue(name: string, fallback: string): string {
  const args = Bun.argv.slice(2);
  const index = args.indexOf(name);
  if (index !== -1 && index + 1 < args.length) {
    return args[index + 1]!;
  }
  return fallback;
}

function healthBase(base: string): string {
  return base.replace(/\/v1\/?$/, "");
}

async function parseJson(response: Response): Promise<any> {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`Invalid JSON (${response.status}): ${text.slice(0, 200)} | ${String(error)}`);
  }
}

function spawnStreamingCurl(base: string, modelId: string, message: string): Bun.Subprocess {
  return Bun.spawn({
    cmd: [
      "curl",
      "-N",
      "-sS",
      `${base}/chat/completions`,
      "-H",
      "Content-Type: application/json",
      "-d",
      JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: message }],
        stream: true,
        max_tokens: 256,
        temperature: 0,
      }),
    ],
    stdout: "pipe",
    stderr: "pipe",
    env: process.env,
  });
}

async function testHealth(base: string): Promise<void> {
  const response = await fetch(`${healthBase(base)}/health`, {
    signal: AbortSignal.timeout(5_000),
  });
  assert(response.status === 200, `Health check failed: ${response.status}`);
  const data = (await parseJson(response)) as HealthResponse;
  assert(data.status === "ok", `Health status: ${JSON.stringify(data)}`);
  assert(typeof data.active_requests === "number", "Health missing active_requests");
  assert(typeof data.queued_requests === "number", "Health missing queued_requests");
  assert(typeof data.uptime_seconds === "number", "Health missing uptime_seconds");
  console.log(`  PASS: /health → ${data.status}, model=${data.model ?? "?"}`);
}

async function fetchHealth(base: string): Promise<HealthResponse> {
  const response = await fetch(`${healthBase(base)}/health`, {
    signal: AbortSignal.timeout(5_000),
  });
  assert(response.status === 200, `Health check failed: ${response.status}`);
  return (await parseJson(response)) as HealthResponse;
}

async function waitForHealth(
  base: string,
  predicate: (health: HealthResponse) => boolean,
  label: string,
  timeoutMs = 10_000,
): Promise<HealthResponse> {
  const deadline = Date.now() + timeoutMs;
  let last: HealthResponse | null = null;
  while (Date.now() < deadline) {
    last = await fetchHealth(base);
    if (predicate(last)) return last;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for health condition: ${label}. Last health=${JSON.stringify(last)}`);
}

async function testModels(base: string): Promise<string> {
  const response = await fetch(`${base}/models`, {
    signal: AbortSignal.timeout(5_000),
  });
  assert(response.status === 200, `Models failed: ${response.status}`);
  const data = (await parseJson(response)) as ModelsResponse;
  assert(data.object === "list", `Expected list, got: ${data.object}`);
  assert(data.data.length > 0, "No models returned");
  // /v1/models returns the whole catalog (installed + available). Pick the
  // entry with active=true — it is the one currently loaded in VRAM. The
  // old behavior (data[0]) picked whatever happened to sort first, which
  // on this catalog is gpt-oss-20b — triggering a multi-GB model switch
  // on every chat request and timing out before first token.
  const model = data.data.find((m) => m.active === true) ?? data.data[0]!;
  assert(typeof model.id === "string" && model.id.length > 0, "Model missing id");
  console.log(`  PASS: /v1/models → ${model.id}${model.active ? " (active)" : " (first listed)"}`);
  return model.id;
}

async function testChatCompletionNonStreaming(base: string, modelId: string): Promise<void> {
  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: modelId,
      messages: [{ role: "user", content: "What is 2+2? Answer in one word." }],
      max_tokens: 16,
      temperature: 0,
    }),
    signal: AbortSignal.timeout(INFERENCE_TIMEOUT_MS),
  });
  const body = await parseJson(response);
  assert(response.status === 200, `Chat failed: ${response.status} ${JSON.stringify(body).slice(0, 200)}`);
  const data = body as ChatCompletionResponse;
  assert(data.object === "chat.completion", `Wrong object: ${data.object}`);
  assert(data.choices.length > 0, "No choices");
  const content = data.choices[0]?.message?.content ?? "";
  assert(content.length > 0, "Empty response content");
  assert(data.usage, "Missing usage");
  console.log(`  PASS: chat (non-streaming) → '${content.slice(0, 50)}' (${data.usage!.completion_tokens} tokens)`);
}

async function streamChat(
  base: string,
  modelId: string,
  message: string,
  options: { maxTokens?: number; timeoutMs?: number } = {},
): Promise<{ text: string; chunks: string[] }> {
  const response = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: modelId,
      messages: [{ role: "user", content: message }],
      stream: true,
      max_tokens: options.maxTokens ?? 32,
      temperature: 0,
    }),
    signal: AbortSignal.timeout(options.timeoutMs ?? INFERENCE_TIMEOUT_MS),
  });
  assert(response.status === 200, `Stream failed: ${response.status}`);
  assert(response.headers.get("content-type")?.includes("text/event-stream"), "Not SSE");
  assert(response.body, "Missing response body");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let gotDone = false;
  let sawRole = false;
  const chunks: string[] = [];

  outer: while (true) {
    const { value, done } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      break;
    }

    buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");
    while (true) {
      const eventEnd = buffer.indexOf("\n\n");
      if (eventEnd === -1) break;

      const event = buffer.slice(0, eventEnd);
      buffer = buffer.slice(eventEnd + 2);

      for (const line of event.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") {
          gotDone = true;
          break outer;
        }

        const chunk = JSON.parse(payload) as ChatChunk;
        assert(chunk.object === "chat.completion.chunk", `Wrong chunk object: ${chunk.object}`);
        const delta = chunk.choices[0]?.delta ?? {};
        if (delta.role === "assistant") sawRole = true;
        if (typeof delta.content === "string") {
          chunks.push(delta.content);
        }
      }
    }
  }

  assert(gotDone, "Never received [DONE]");
  assert(sawRole, "Streaming response never emitted assistant role chunk");
  const fullText = chunks.join("");
  assert(fullText.length > 0, "Empty streaming response");
  assert(!fullText.includes("<|im_start|>"), `Leaked chat start token: ${fullText.slice(0, 200)}`);
  assert(!fullText.includes("<|im_end|>"), `Leaked chat end token: ${fullText.slice(0, 200)}`);
  return { text: fullText, chunks };
}

async function testChatCompletionStreaming(base: string, modelId: string): Promise<void> {
  const { text, chunks } = await streamChat(base, modelId, "Count to 3.");
  console.log(`  PASS: chat (streaming) → '${text.slice(0, 50)}' (${chunks.length} chunks)`);
}

async function testChatCompletionStreamingSequential(base: string, modelId: string): Promise<void> {
  for (const run of [1, 2]) {
    const { text, chunks } = await streamChat(base, modelId, "hello");
    assert(chunks[0] && chunks[0].length > 0, `Sequential stream ${run} first assistant chunk was empty`);
    console.log(`  PASS: sequential chat stream ${run} → '${text.slice(0, 50)}'`);
  }
}

async function testChatCompletionStreamingOverlapped(base: string, modelId: string): Promise<void> {
  const first = streamChat(base, modelId, "Reply with alpha.", {
    maxTokens: 16,
    timeoutMs: 60_000,
  });

  await new Promise((resolve) => setTimeout(resolve, 250));

  const second = streamChat(base, modelId, "Reply with beta.", {
    maxTokens: 16,
    timeoutMs: 60_000,
  });

  const [firstResult, secondResult] = await Promise.all([first, second]);
  assert(firstResult.text.length > 0, "First overlapped stream returned empty content");
  assert(secondResult.text.length > 0, "Second overlapped stream returned empty content");
  console.log(`  PASS: overlapped chat streams → '${firstResult.text.slice(0, 24)}' | '${secondResult.text.slice(0, 24)}'`);
}

async function testHealthReflectsQueuedLoad(base: string, modelId: string): Promise<void> {
  const firstProc = spawnStreamingCurl(base, modelId, "The capital of France is");

  const activeHealth = await waitForHealth(
    base,
    (health) => (health.active_requests ?? 0) >= 1,
    "active_requests >= 1",
    15_000,
  );

  const secondProc = spawnStreamingCurl(base, modelId, "The capital of France is");

  const queuedHealth = await waitForHealth(
    base,
    (health) => (health.active_requests ?? 0) >= 1 && (health.queued_requests ?? 0) >= 1,
    "active_requests >= 1 and queued_requests >= 1",
    15_000,
  );

  firstProc.kill();
  secondProc.kill();
  await Promise.allSettled([firstProc.exited, secondProc.exited]);

  const idleHealth = await waitForHealth(
    base,
    (health) => (health.active_requests ?? 0) === 0 && (health.queued_requests ?? 0) === 0,
    "active_requests == 0 and queued_requests == 0",
    15_000,
  );

  console.log(
    `  PASS: health under load → active=${activeHealth.active_requests}, queued=${queuedHealth.queued_requests}, uptime=${idleHealth.uptime_seconds}`,
  );
}

async function testCompletion(base: string, modelId: string): Promise<void> {
  const response = await fetch(`${base}/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: modelId,
      prompt: "The capital of France is",
      max_tokens: 8,
      temperature: 0,
    }),
    signal: AbortSignal.timeout(INFERENCE_TIMEOUT_MS),
  });
  assert(response.status === 200, `Completion failed: ${response.status}`);
  const data = await parseJson(response);
  assert(data.object === "text_completion", `Wrong object: ${data.object}`);
  const text = data.choices?.[0]?.text ?? "";
  assert(text.length > 0, "Empty completion");
  console.log(`  PASS: completion → '${text.slice(0, 50)}'`);
}

async function testErrorHandling(base: string): Promise<void> {
  const missingMessages = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "x" }),
    signal: AbortSignal.timeout(5_000),
  });
  assert(missingMessages.status === 400, `Expected 400, got ${missingMessages.status}`);
  const errorBody = await parseJson(missingMessages);
  assert(errorBody.error, "Missing error field");
  console.log(`  PASS: error format → ${String(errorBody.error.message).slice(0, 60)}`);

  const unknownEndpoint = await fetch(`${base}/unknown`, {
    signal: AbortSignal.timeout(5_000),
  });
  assert(unknownEndpoint.status === 404, `Expected 404, got ${unknownEndpoint.status}`);
  console.log("  PASS: 404 for unknown endpoint");
}

async function testOpenAiSdk(base: string, modelId: string): Promise<void> {
  let OpenAI: any;
  try {
    ({ default: OpenAI } = await import("openai"));
  } catch {
    console.log("  SKIP: openai package not installed (bun add openai)");
    return;
  }

  const client = new OpenAI({
    baseURL: base,
    apiKey: "unused",
  });

  const response = await client.chat.completions.create({
    model: modelId,
    messages: [{ role: "user", content: "Say hello." }],
    max_tokens: 8,
  });
  const nonStreaming = response.choices?.[0]?.message?.content ?? "";
  assert(nonStreaming.length > 0, "SDK non-streaming empty");
  console.log(`  PASS: OpenAI SDK (non-streaming) → '${nonStreaming.slice(0, 40)}'`);

  const chunks: string[] = [];
  const stream = await client.chat.completions.create({
    model: modelId,
    messages: [{ role: "user", content: "Say hi." }],
    max_tokens: 8,
    stream: true,
  });
  for await (const chunk of stream) {
    const content = chunk.choices?.[0]?.delta?.content;
    if (content) chunks.push(content);
  }
  assert(chunks.length > 0, "SDK streaming empty");
  console.log(`  PASS: OpenAI SDK (streaming) → '${chunks.join("").slice(0, 40)}'`);
}

export async function runSuite(base: string): Promise<void> {
  console.log(`Testing ZINC API at ${base}\n`);

  await testHealth(base);
  const modelId = await testModels(base);
  await testChatCompletionNonStreaming(base, modelId);
  await testChatCompletionStreaming(base, modelId);
  await testChatCompletionStreamingSequential(base, modelId);
  await testChatCompletionStreamingOverlapped(base, modelId);
  await testHealthReflectsQueuedLoad(base, modelId);
  await testCompletion(base, modelId);
  await testErrorHandling(base);
  await testOpenAiSdk(base, modelId);
  console.log("\nAll tests passed!");
}

if (import.meta.main) {
  const base = getArgValue("--base-url", "http://localhost:8080/v1");
  runSuite(base).catch((error) => {
    console.error(`\nFAILED: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  });
}
