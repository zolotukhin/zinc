import { performance } from "node:perf_hooks";

const shortPrompt = "The capital of France is";
const mediumPrompt =
  "Context for load testing only. " +
  "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. ".repeat(12) +
  "\nNow continue this text naturally: The capital of France is";
const longPrompt =
  "Longer context for API benchmark only. " +
  "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. ".repeat(64) +
  "\nNow continue this text naturally: The capital of France is";

export function percentile(values, p) {
  if (values.length === 0) return 0;
  if (values.length === 1) return values[0];
  const xs = [...values].sort((a, b) => a - b);
  const rank = (xs.length - 1) * p;
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return xs[lo];
  const frac = rank - lo;
  return xs[lo] * (1 - frac) + xs[hi] * frac;
}

export function defaultScenarios(mode) {
  const chat = [
    { name: "short_c1_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 1 },
    { name: "short_c2_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 2 },
    { name: "short_c4_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 4 },
    { name: "medium_c1_t64", kind: "chat", prompt: mediumPrompt, maxTokens: 64, concurrency: 1 },
    { name: "medium_c2_t64", kind: "chat", prompt: mediumPrompt, maxTokens: 64, concurrency: 2 },
    { name: "medium_c4_t64", kind: "chat", prompt: mediumPrompt, maxTokens: 64, concurrency: 4 },
    { name: "long_c1_t64", kind: "chat", prompt: longPrompt, maxTokens: 64, concurrency: 1 },
    { name: "long_c2_t64", kind: "chat", prompt: longPrompt, maxTokens: 64, concurrency: 2 },
    { name: "long_c4_t64", kind: "chat", prompt: longPrompt, maxTokens: 64, concurrency: 4 },
    { name: "short_c1_t256", kind: "chat", prompt: shortPrompt, maxTokens: 256, concurrency: 1 },
    { name: "short_c4_t256", kind: "chat", prompt: shortPrompt, maxTokens: 256, concurrency: 4 },
    { name: "short_stream_c1_t64", kind: "chat", prompt: shortPrompt, maxTokens: 64, concurrency: 1, stream: true },
  ];
  const raw = [
    { name: "raw_c1_t256", kind: "raw", prompt: shortPrompt, maxTokens: 256, concurrency: 1 },
    { name: "raw_c4_t256", kind: "raw", prompt: shortPrompt, maxTokens: 256, concurrency: 4 },
  ];
  if (mode === "chat") return chat;
  if (mode === "raw") return raw;
  return [...chat, ...raw];
}

export function summarizeResult(result) {
  if (result.stream) {
    return `${result.name}: latency_avg=${result.latency_avg_s.toFixed(2)}s p95=${result.latency_p95_s.toFixed(2)}s ttft_avg=${result.ttft_avg_s.toFixed(2)}s chunks_avg=${result.chunks_avg.toFixed(2)}`;
  }
  return `${result.name}: latency_avg=${result.latency_avg_s.toFixed(2)}s p95=${result.latency_p95_s.toFixed(2)}s agg_completion_tps=${result.aggregate_completion_tps.toFixed(2)} per_req_tps_avg=${result.completion_tps_avg.toFixed(2)}`;
}

export function parseArgs(argv) {
  let base = "http://127.0.0.1:9090/v1";
  let mode = "both";
  let output = `/tmp/zinc_api_benchmark_${Date.now()}.json`;
  let timeoutMs = 600_000;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--base":
        base = argv[++i] ?? base;
        break;
      case "--mode": {
        const value = argv[++i];
        if (value === "chat" || value === "raw" || value === "both") mode = value;
        else throw new Error(`Invalid --mode '${value}'. Expected chat, raw, or both.`);
        break;
      }
      case "--output":
        output = argv[++i] ?? output;
        break;
      case "--timeout-ms": {
        const raw = argv[++i] ?? "";
        const value = Number(raw);
        if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid --timeout-ms '${raw}'`);
        timeoutMs = value;
        break;
      }
      case "-h":
      case "--help":
        printUsageAndExit();
        break;
      default:
        throw new Error(`Unknown argument '${arg}'`);
    }
  }

  return { base, mode, output, timeoutMs };
}

function printUsageAndExit() {
  console.log(`Usage: bun tools/benchmark_api.mjs [options]
  --base <url>         Base /v1 URL (default: http://127.0.0.1:9090/v1)
  --mode <mode>        chat | raw | both (default: both)
  --output <path>      JSON artifact path (default: /tmp/zinc_api_benchmark_<ts>.json)
  --timeout-ms <ms>    Per-request timeout in milliseconds (default: 600000)
  -h, --help           Show this help`);
  process.exit(0);
}

function makeBarrier(count) {
  let waiting = 0;
  let release = null;
  const gate = new Promise((resolve) => {
    release = resolve;
  });

  return async () => {
    waiting += 1;
    if (waiting === count && release) release();
    await gate;
  };
}

async function postJson(url, payload, timeoutMs) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

async function warmup(base, timeoutMs, kind) {
  if (kind === "chat") {
    const resp = await postJson(`${base}/chat/completions`, {
      model: "q",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 8,
      temperature: 0,
      stream: false,
    }, timeoutMs);
    await resp.text();
    return;
  }
  const resp = await postJson(`${base}/completions`, {
    model: "q",
    prompt: "hi",
    max_tokens: 8,
  }, timeoutMs);
  await resp.text();
}

async function runNonStreamingRequest(scenario, base, timeoutMs, waitForStart) {
  const url = scenario.kind === "chat" ? `${base}/chat/completions` : `${base}/completions`;
  const payload =
    scenario.kind === "chat"
      ? {
          model: "q",
          messages: [{ role: "user", content: scenario.prompt }],
          max_tokens: scenario.maxTokens,
          temperature: 0,
          stream: false,
        }
      : {
          model: "q",
          prompt: scenario.prompt,
          max_tokens: scenario.maxTokens,
        };

  await waitForStart();
  const t0 = performance.now();
  const resp = await postJson(url, payload, timeoutMs);
  const body = await resp.json();
  const t1 = performance.now();
  const promptTokens = body.usage?.prompt_tokens ?? 0;
  const completionTokens = body.usage?.completion_tokens ?? 0;
  return {
    latencyS: (t1 - t0) / 1000,
    promptTokens,
    completionTokens,
    completionTps: completionTokens / Math.max((t1 - t0) / 1000, 1e-9),
  };
}

async function runStreamingRequest(scenario, base, timeoutMs, waitForStart) {
  await waitForStart();
  const t0 = performance.now();
  const resp = await postJson(`${base}/chat/completions`, {
    model: "q",
    messages: [{ role: "user", content: scenario.prompt }],
    max_tokens: scenario.maxTokens,
    temperature: 0,
    stream: true,
  }, timeoutMs);
  if (!resp.body) throw new Error(`Streaming response for ${scenario.name} had no body`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let firstTokenS = null;
  let chunks = 0;
  while (true) {
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
          const t1 = performance.now();
          return {
            latencyS: (t1 - t0) / 1000,
            ttftS: firstTokenS ?? (t1 - t0) / 1000,
            chunks,
          };
        }
        const chunk = JSON.parse(payload);
        const content = chunk.choices?.[0]?.delta?.content;
        if (typeof content === "string" && content.length > 0) {
          if (firstTokenS == null) firstTokenS = (performance.now() - t0) / 1000;
          chunks += 1;
        }
      }
    }
  }
  throw new Error(`Streaming response for ${scenario.name} ended without [DONE]`);
}

async function runScenario(scenario, base, timeoutMs) {
  const waitForStart = makeBarrier(scenario.concurrency);
  const started = performance.now();

  if (scenario.stream) {
    const rows = await Promise.all(
      Array.from({ length: scenario.concurrency }, () =>
        runStreamingRequest(scenario, base, timeoutMs, waitForStart),
      ),
    );
    const ended = performance.now();
    const latencies = rows.map((row) => row.latencyS);
    const ttfts = rows.map((row) => row.ttftS);
    return {
      name: scenario.name,
      kind: scenario.kind,
      prompt_chars: scenario.prompt.length,
      max_tokens: scenario.maxTokens,
      concurrency: scenario.concurrency,
      stream: true,
      requests: rows.length,
      makespan_s: (ended - started) / 1000,
      latency_avg_s: average(latencies),
      latency_p50_s: percentile(latencies, 0.5),
      latency_p95_s: percentile(latencies, 0.95),
      ttft_avg_s: average(ttfts),
      ttft_p50_s: percentile(ttfts, 0.5),
      ttft_p95_s: percentile(ttfts, 0.95),
      chunks_avg: average(rows.map((row) => row.chunks)),
    };
  }

  const rows = await Promise.all(
    Array.from({ length: scenario.concurrency }, () =>
      runNonStreamingRequest(scenario, base, timeoutMs, waitForStart),
    ),
  );
  const ended = performance.now();
  const latencies = rows.map((row) => row.latencyS);
  const promptTokens = rows.map((row) => row.promptTokens);
  const completionTokens = rows.map((row) => row.completionTokens);
  const completionTps = rows.map((row) => row.completionTps);

  return {
    name: scenario.name,
    kind: scenario.kind,
    prompt_chars: scenario.prompt.length,
    max_tokens: scenario.maxTokens,
    concurrency: scenario.concurrency,
    requests: rows.length,
    makespan_s: (ended - started) / 1000,
    latency_avg_s: average(latencies),
    latency_p50_s: percentile(latencies, 0.5),
    latency_p95_s: percentile(latencies, 0.95),
    prompt_tokens_avg: average(promptTokens),
    completion_tokens_avg: average(completionTokens),
    completion_tps_avg: average(completionTps),
    aggregate_completion_tps: sum(completionTokens) / Math.max((ended - started) / 1000, 1e-9),
    aggregate_total_tps: (sum(promptTokens) + sum(completionTokens)) / Math.max((ended - started) / 1000, 1e-9),
  };
}

function average(values) {
  return sum(values) / Math.max(values.length, 1);
}

function sum(values) {
  return values.reduce((acc, value) => acc + value, 0);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const scenarios = defaultScenarios(args.mode);
  const seenKinds = new Set(scenarios.map((scenario) => scenario.kind));
  for (const kind of seenKinds) {
    await warmup(args.base, args.timeoutMs, kind);
  }

  const results = [];
  for (const scenario of scenarios) {
    const result = await runScenario(scenario, args.base, args.timeoutMs);
    results.push(result);
    console.log(summarizeResult(result));
  }

  const artifact = {
    base: args.base,
    mode: args.mode,
    generated_at_unix: Math.floor(Date.now() / 1000),
    results,
  };
  await Bun.write(args.output, `${JSON.stringify(artifact, null, 2)}\n`);
  console.log(`ARTIFACT ${args.output}`);
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
