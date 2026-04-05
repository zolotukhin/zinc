#!/usr/bin/env bun
/**
 * ZINC Guided Change
 *
 * Takes a prompt (inline, file, or stdin), establishes performance baselines
 * across all locally installed models, spawns an AI agent to implement the
 * change, then re-benchmarks to verify no model regressed. If regressions
 * are detected, feeds the data back to the agent for selective adjustment
 * rather than a full revert.
 *
 * Usage:
 *   bun loops/guided_change.ts --prompt "Refactor the MoE dispatch"
 *   bun loops/guided_change.ts --prompt-file plan.md
 *   echo "Add bfloat16 support" | bun loops/guided_change.ts
 *   bun loops/guided_change.ts                      # interactive prompt
 */

import { spawn } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline";

// ── Color & display ──────────────────────────────────────────────────

const TTY = process.stdout.isTTY ?? false;
const NO_COLOR = "NO_COLOR" in process.env;
const FORCE_COLOR = process.env.FORCE_COLOR === "1" || process.env.CLICOLOR_FORCE === "1";
const COLOR_ENABLED = !NO_COLOR && (TTY || FORCE_COLOR);

function clr(code: string, text: string): string {
  return COLOR_ENABLED ? `\x1b[${code}m${text}\x1b[0m` : text;
}

const SEP = "─".repeat(64);

// ── Constants ────────────────────────────────────────────────────────

const REPO_ROOT = resolve(import.meta.dir, "..");
const RESULTS_DIR = resolve(REPO_ROOT, ".guided_change");
const BENCHMARK_PROMPT = "The capital of France is";
const BENCHMARK_TOKENS = 32;
const BENCHMARK_RUNS = 3;
const REGRESSION_PCT = 0.03; // 3% regression tolerance
const REGRESSION_FLOOR = 0.5; // minimum tok/s drop to count as regression
const MAX_RETRIES = 3; // retries before full revert

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)",
  "Bash(git revert:*)",
  "Bash(git restore:*)",
  "Bash(git reset:*)",
  "Bash(git stash:*)",
  "Bash(git clean:*)",
];

type AgentKind = "claude" | "codex";

// ── Types ────────────────────────────────────────────────────────────

type ModelBaseline = {
  id: string;
  path: string;
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  correct: boolean;
  outputPreview: string;
};

type RegressionReport = {
  model: ModelBaseline;
  baselineTps: number;
  afterTps: number;
  deltaTps: number;
  deltaPct: number;
  regressed: boolean;
};

// ── Output parsing ──────────────────────────────────────────────────

function parseTokPerSec(output: string): number | null {
  const m = output.match(/Generated\s+(\d+)\s+tokens\s+in\s+(\d+\.?\d*)\s*(ms|s)/i);
  if (m) {
    const tokens = parseInt(m[1], 10);
    let seconds = parseFloat(m[2]);
    if (m[3] === "ms") seconds /= 1000;
    if (seconds > 0) return tokens / seconds;
  }
  const m2 = output.match(/(\d+\.?\d*)\s*tok\/s/i);
  return m2 ? parseFloat(m2[1]) : null;
}

function parseOutputText(output: string): string {
  const m = output.match(/Output\s*\(\d+\s*tokens?\)\s*:\s*(.+)/i);
  return m ? m[1].trim().slice(0, 200) : "";
}

// ── Command runner ──────────────────────────────────────────────────

type RunResult = { exitCode: number; stdout: string; stderr: string };

function formatElapsed(startMs: number): string {
  const s = ((Date.now() - startMs) / 1000) | 0;
  if (s < 60) return `${s}s`;
  return `${(s / 60) | 0}m${s % 60}s`;
}

async function runCommand(
  cmd: string,
  args: string[],
  opts: {
    cwd?: string;
    timeout?: number;
    streamOutput?: boolean;
    stdoutLineFormatter?: (line: string) => string | null;
    stderrLineFormatter?: (line: string) => string | null;
  } = {},
): Promise<RunResult> {
  return new Promise((res) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout,
    });
    let stdout = "", stderr = "", lineBuffer = "", stderrLineBuffer = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (!opts.streamOutput) return;
      if (opts.stdoutLineFormatter) {
        lineBuffer += text;
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const formatted = opts.stdoutLineFormatter(line);
          if (formatted != null) process.stdout.write(formatted);
        }
      } else {
        process.stdout.write(text);
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (!opts.streamOutput) return;
      if (opts.stderrLineFormatter) {
        stderrLineBuffer += text;
        const lines = stderrLineBuffer.split("\n");
        stderrLineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const formatted = opts.stderrLineFormatter(line);
          if (formatted != null) process.stderr.write(formatted);
        }
      } else {
        process.stderr.write(text);
      }
    });
    child.on("error", () => res({ exitCode: -1, stdout, stderr }));
    child.on("close", (code) => {
      if (opts.streamOutput && opts.stdoutLineFormatter && lineBuffer.trim()) {
        const formatted = opts.stdoutLineFormatter(lineBuffer);
        if (formatted != null) process.stdout.write(formatted);
      }
      if (opts.streamOutput && opts.stderrLineFormatter && stderrLineBuffer.trim()) {
        const formatted = opts.stderrLineFormatter(stderrLineBuffer);
        if (formatted != null) process.stderr.write(formatted);
      }
      res({ exitCode: code ?? -1, stdout, stderr });
    });
  });
}

// ── Agent stream formatters ─────────────────────────────────────────

type ClaudeStreamState = {
  currentToolName: string | null;
  currentBlockIsToolUse: boolean;
  inputJsonBuffer: string;
  inTextBlock: boolean;
  sawTextDelta: boolean;
};

function coerceDisplayText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    const parts = value.map((e) => coerceDisplayText(e)).filter((e) => e.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(value, null, 2); } catch { return ""; }
  }
  if (typeof value === "object") {
    const r = value as Record<string, unknown>;
    const parts = [r.text, r.message, r.output, r.stdout, r.stderr, r.content, r.result, r.summary, r.output_text]
      .map((e) => coerceDisplayText(e)).filter((e) => e.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(r, null, 2); } catch { return ""; }
  }
  return "";
}

function formatToolInput(name: string, jsonBuf: string): string {
  let input: Record<string, unknown> = {};
  try { input = JSON.parse(jsonBuf); } catch { return ""; }
  const out: string[] = [];
  const MAX_DIFF = 5;
  if (name === "edit") {
    const fp = (input.file_path as string | undefined) ?? "?";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}`));
    const oldLines = ((input.old_string as string | undefined) ?? "").split("\n");
    const newLines = ((input.new_string as string | undefined) ?? "").split("\n");
    for (const l of oldLines.slice(0, MAX_DIFF)) out.push(clr("31", `   - ${l}`));
    if (oldLines.length > MAX_DIFF) out.push(clr("2", `   - … (${oldLines.length - MAX_DIFF} more)`));
    for (const l of newLines.slice(0, MAX_DIFF)) out.push(clr("32", `   + ${l}`));
    if (newLines.length > MAX_DIFF) out.push(clr("2", `   + … (${newLines.length - MAX_DIFF} more)`));
  } else if (name === "write") {
    const fp = (input.file_path as string | undefined) ?? "?";
    const lineCount = ((input.content as string | undefined) ?? "").split("\n").length;
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")} (${lineCount} lines)`));
  } else if (name === "bash") {
    const cmd = (input.command as string | undefined) ?? "?";
    out.push(clr("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "…" : cmd}`));
  } else if (name === "read") {
    const fp = (input.file_path as string | undefined) ?? "?";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}`));
  } else if (name === "grep") {
    out.push(clr("2", ` → /${(input.pattern as string | undefined) ?? "?"}/`));
  } else if (name === "glob") {
    out.push(clr("2", ` → ${(input.pattern as string | undefined) ?? "?"}`));
  }
  return out.length > 0 ? out.join("\n") + "\n" : "";
}

function formatClaudeStreamLine(rawLine: string, state: ClaudeStreamState): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return null; }

  if (event.type === "stream_event") {
    const e = event.event as Record<string, unknown> | undefined;
    if (!e) return null;
    if (e.type === "content_block_start") {
      const block = e.content_block as Record<string, unknown> | undefined;
      if (block?.type === "tool_use") {
        state.currentToolName = (block.name as string) ?? "tool";
        state.currentBlockIsToolUse = true;
        state.inputJsonBuffer = "";
        state.inTextBlock = false;
        return `\n${clr("33", `🔧 ${state.currentToolName}`)}`;
      }
      if (block?.type === "text") {
        state.inTextBlock = true;
        state.currentBlockIsToolUse = false;
        return COLOR_ENABLED ? "\n\x1b[96m" : "\n";
      }
      state.inTextBlock = false;
      state.currentBlockIsToolUse = false;
      return null;
    }
    if (e.type === "content_block_delta") {
      const delta = e.delta as Record<string, unknown> | undefined;
      if (delta?.type === "input_json_delta") {
        state.inputJsonBuffer += (delta.partial_json as string) ?? "";
        return null;
      }
      if (delta?.type === "text_delta" && state.inTextBlock) {
        state.sawTextDelta = true;
        return delta.text as string;
      }
      return null;
    }
    if (e.type === "content_block_stop") {
      if (state.currentBlockIsToolUse) {
        state.currentBlockIsToolUse = false;
        const detail = formatToolInput(state.currentToolName ?? "", state.inputJsonBuffer);
        state.inputJsonBuffer = "";
        return detail || null;
      }
      if (state.inTextBlock) {
        state.inTextBlock = false;
        return COLOR_ENABLED ? "\x1b[0m\n" : "\n";
      }
      return null;
    }
    return null;
  }
  if (event.type === "user") {
    const result = event.tool_use_result as Record<string, unknown> | undefined;
    if (result) return clr("32", "   ☑ accepted") + "\n";
    return null;
  }
  if (event.type === "assistant") {
    const msg = event.message as Record<string, unknown> | undefined;
    if (!msg) return null;
    const content = msg.content;
    if (Array.isArray(content)) {
      const parts: string[] = [];
      for (const block of content) {
        const b = block as Record<string, unknown>;
        if (b?.type === "text" && typeof b.text === "string" && b.text.trim()) parts.push(b.text);
      }
      const text = parts.join("\n");
      if (!text.trim() || state.sawTextDelta) { state.sawTextDelta = false; return null; }
      return clr("96", text) + "\n";
    }
    return null;
  }
  return null;
}

type CodexStreamState = { startedCommandIds: Set<string> };

function formatCodexJsonEvent(rawLine: string, state: CodexStreamState): string | null | undefined {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return undefined; }
  const eventType = typeof event.type === "string" ? event.type : "";
  if (["thread.started", "turn.started", "turn.completed"].includes(eventType)) return null;
  if (eventType === "error") {
    const message = coerceDisplayText(event.message);
    return message ? `${clr("31", message)}\n` : null;
  }
  if (!eventType.startsWith("item.")) return null;
  const item = event.item as Record<string, unknown> | undefined;
  if (!item) return null;
  const itemType = typeof item.type === "string" ? item.type : "";
  const itemId = typeof item.id === "string" ? item.id : "";
  const phase = eventType.slice("item.".length);

  if (itemType === "reasoning" && phase === "completed") {
    const text = coerceDisplayText(item.summary ?? item.text ?? item.message ?? item.content);
    return text ? `${clr("2", `thinking: ${text}`)}\n` : null;
  }
  if (itemType === "command_execution") {
    const inp = item.input as Record<string, unknown> | undefined;
    const cmd = coerceDisplayText(item.command ?? inp?.command ?? "").trim();
    const output = coerceDisplayText(item.aggregated_output ?? item.output ?? item.stdout ?? "");
    const exitCode = typeof item.exit_code === "number" ? item.exit_code : null;
    const startedAlready = itemId ? state.startedCommandIds.has(itemId) : false;
    if (phase === "started") {
      if (itemId) state.startedCommandIds.add(itemId);
      return cmd ? `\n${clr("33", "🔧 bash")}\n${clr("2", `   $ ${cmd}`)}\n` : `\n${clr("33", "🔧 bash")}\n`;
    }
    let out = startedAlready ? "" : `\n${clr("33", "🔧 bash")}\n${cmd ? clr("2", `   $ ${cmd}`) + "\n" : ""}`;
    if (phase === "completed") {
      const lines = output.split("\n").filter((l) => l.trim());
      const tail = lines.slice(-3);
      const sc = exitCode === 0 ? "32" : exitCode == null ? "33" : "31";
      const st = exitCode === 0 ? "   ☑ accepted" : exitCode == null ? "   ⚠ completed" : `   ✖ exit ${exitCode}`;
      const body = tail.length > 0
        ? (lines.length > 3 ? clr("2", "   …\n") : "") + tail.map((l) => clr("2", `   ${l.trim()}`)).join("\n") + "\n"
        : "";
      out += `${clr(sc, st)}\n${body}`;
    }
    if (itemId) state.startedCommandIds.delete(itemId);
    return out || null;
  }
  if (itemType === "file_change" && phase === "completed") {
    const changesSource = [item.changes, item.file_changes, item.files];
    const changes = changesSource.flatMap((v) => {
      if (!Array.isArray(v)) return [];
      return v.map((e) => {
        const c = e as Record<string, unknown>;
        return { path: coerceDisplayText(c.path ?? c.file_path ?? "?"), action: coerceDisplayText(c.change_type ?? c.kind ?? "") };
      });
    });
    if (changes.length === 0) return null;
    let out = `\n${clr("35", "📝 file change")}\n`;
    for (const c of changes.slice(0, 6)) out += clr("2", `   ${c.action ? `${c.action}: ` : ""}${c.path}`) + "\n";
    return out;
  }
  if (itemType === "agent_message" && phase === "completed") {
    const text = coerceDisplayText(item.text ?? item.message ?? item.output_text ?? item.content);
    return text ? `${clr("96", text)}\n` : null;
  }
  if (itemType === "error" && phase === "completed") {
    const text = coerceDisplayText(item.text ?? item.message ?? item.content);
    return text ? `${clr("33", text)}\n` : null;
  }
  return null;
}

function formatCodexStreamLine(rawLine: string, state: CodexStreamState): string | null {
  const jsonFormatted = formatCodexJsonEvent(rawLine, state);
  if (jsonFormatted !== undefined) return jsonFormatted;
  if (!rawLine.trim()) return "\n";
  if (rawLine === "thinking") return `${clr("2", rawLine)}\n`;
  if (rawLine === "codex") return `${clr("1;35", rawLine)}\n`;
  if (rawLine.includes("still running")) return `${clr("2", rawLine)}\n`;
  return `${clr("2", `[codex] ${rawLine}`)}\n`;
}

function formatCodexStderrLine(rawLine: string): string | null {
  if (!rawLine.trim()) return "\n";
  return `${clr("2", `[codex] ${rawLine}`)}\n`;
}

// ── Agent invocation ────────────────────────────────────────────────

function buildClaudeArgs(prompt: string, model?: string): string[] {
  const args = [
    "-p", "--verbose",
    "--output-format", "stream-json",
    "--include-partial-messages",
    `--disallowed-tools=${BLOCKED_GIT_OPS.join(",")}`,
    "--permission-mode", "bypassPermissions",
    "--effort", "high",
  ];
  if (model) args.push("--model", model);
  args.push(prompt);
  return args;
}

function buildCodexArgs(prompt: string, model?: string): string[] {
  const args = [
    "exec", "--skip-git-repo-check", "--json",
    "--color", "never", "--sandbox", "workspace-write",
    "--cd", REPO_ROOT,
  ];
  if (model) args.push("--model", model);
  args.push(prompt);
  return args;
}

async function runAgent(agent: AgentKind, prompt: string, model?: string): Promise<RunResult> {
  const label = agent === "codex" ? "Codex" : "Claude";
  console.log(clr("1;34", SEP));
  console.log(clr("1;34", `  🧠 AGENT (${label})`));
  console.log(clr("1;34", SEP));
  const lines = prompt.split("\n");
  for (const line of lines.slice(0, 15)) process.stdout.write(clr("2", line) + "\n");
  if (lines.length > 15) process.stdout.write(clr("2", `… (${lines.length - 15} more lines)\n`));
  console.log(clr("1;34", SEP));

  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(clr("2", `\n⏳ agent running (${formatElapsed(startedAt)})...\n`));
  }, 30_000);

  console.log(clr("1;36", SEP));
  console.log(clr("1;36", `  💬 RESPONSE (${label})`));
  console.log(clr("1;36", SEP));

  let result: RunResult;
  if (agent === "codex") {
    const s: CodexStreamState = { startedCommandIds: new Set() };
    result = await runCommand("codex", buildCodexArgs(prompt, model), {
      streamOutput: true, timeout: 1_800_000,
      stdoutLineFormatter: (line) => formatCodexStreamLine(line, s),
      stderrLineFormatter: formatCodexStderrLine,
    });
  } else {
    const s: ClaudeStreamState = {
      currentToolName: null, currentBlockIsToolUse: false,
      inputJsonBuffer: "", inTextBlock: false, sawTextDelta: false,
    };
    result = await runCommand("claude", buildClaudeArgs(prompt, model), {
      streamOutput: true, timeout: 1_800_000,
      stdoutLineFormatter: (line) => formatClaudeStreamLine(line, s),
    });
  }

  clearInterval(heartbeat);
  console.log(clr("1;36", SEP));
  console.log(clr("1;32", `  ✅ ${label} done in ${formatElapsed(startedAt)}`));
  return result;
}

function extractAgentText(stdout: string): string {
  const lines = stdout.split("\n");
  const texts: string[] = [];
  for (const line of lines) {
    try {
      const evt = JSON.parse(line);
      if (evt?.type === "assistant") {
        const content = evt?.message?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            if (block?.type === "text" && typeof block.text === "string") texts.push(block.text);
          }
        }
      }
      if (evt?.type === "item.completed" && evt?.item?.type === "agent_message") {
        const text = coerceDisplayText(evt.item.text ?? evt.item.message ?? evt.item.output_text ?? evt.item.content);
        if (text.trim()) texts.push(text);
      }
    } catch { /* not JSON */ }
  }
  return texts.join("\n");
}

// ── Model discovery ─────────────────────────────────────────────────

function discoverModels(extraPaths: string[]): { id: string; path: string }[] {
  const seen = new Set<string>();
  const models: { id: string; path: string }[] = [];

  function add(path: string, id?: string) {
    const resolved = resolve(path);
    if (seen.has(resolved) || !existsSync(resolved)) return;
    seen.add(resolved);
    const name = id ?? resolved.split("/").pop()?.replace(/\.gguf$/i, "") ?? "unknown";
    models.push({ id: name, path: resolved });
  }

  // Explicit paths from --models flag
  for (const p of extraPaths) add(p);

  // ZINC_MODEL env
  if (process.env.ZINC_MODEL) add(process.env.ZINC_MODEL);

  // Managed models cache (macOS)
  const home = process.env.HOME ?? "";
  const cacheDir = join(home, "Library", "Caches", "zinc", "models");
  if (existsSync(cacheDir)) {
    for (const entry of readdirSync(cacheDir, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        const gguf = join(cacheDir, entry.name, "model.gguf");
        add(gguf, entry.name);
      }
    }
  }

  // Common local model directories
  const localModelDir = join(home, "models");
  if (existsSync(localModelDir)) {
    for (const entry of readdirSync(localModelDir)) {
      if (entry.endsWith(".gguf")) add(join(localModelDir, entry));
    }
  }

  return models;
}

// ── Build & benchmark ───────────────────────────────────────────────

async function buildProject(): Promise<{ ok: boolean; output: string }> {
  console.log(clr("1;33", "  🔨 Building..."));
  const build = await runCommand("zig", ["build"], { timeout: 120_000 });
  if (build.exitCode !== 0) {
    console.log(clr("1;31", "  ❌ Build failed"));
    return { ok: false, output: build.stderr + build.stdout };
  }
  console.log(clr("1;32", "  ✅ Build OK"));
  return { ok: true, output: build.stderr };
}

async function runTests(): Promise<{ ok: boolean; output: string }> {
  console.log(clr("1;33", "  🧪 Testing..."));
  const test = await runCommand("zig", ["build", "test"], { timeout: 120_000 });
  if (test.exitCode !== 0) {
    console.log(clr("1;31", "  ❌ Tests failed"));
    return { ok: false, output: test.stderr + test.stdout };
  }
  console.log(clr("1;32", "  ✅ Tests OK"));
  return { ok: true, output: test.stderr };
}

async function benchmarkModel(model: { id: string; path: string }): Promise<ModelBaseline> {
  const samples: number[] = [];
  let lastOutput = "";

  for (let i = 0; i < BENCHMARK_RUNS; i++) {
    const run = await runCommand("./zig-out/bin/zinc", [
      "-m", model.path,
      "--prompt", BENCHMARK_PROMPT,
      "-n", String(BENCHMARK_TOKENS),
    ], { timeout: 300_000 });
    const combined = run.stderr + run.stdout;
    lastOutput = combined;

    if (run.exitCode !== 0) break;

    const tps = parseTokPerSec(combined);
    if (tps != null) samples.push(tps);
  }

  const sorted = [...samples].sort((a, b) => a - b);
  const median = sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : null;
  const outputText = parseOutputText(lastOutput);

  return {
    id: model.id,
    path: model.path,
    tokPerSec: median,
    tokPerSecSamples: samples,
    correct: outputText.toLowerCase().includes("paris"),
    outputPreview: outputText.slice(0, 100),
  };
}

async function benchmarkAll(models: { id: string; path: string }[]): Promise<ModelBaseline[]> {
  const results: ModelBaseline[] = [];
  for (const model of models) {
    console.log(clr("1;33", `  📊 Benchmarking ${model.id}...`));
    const baseline = await benchmarkModel(model);
    if (baseline.tokPerSec != null) {
      const range = baseline.tokPerSecSamples.length > 1
        ? ` [${baseline.tokPerSecSamples.map(s => s.toFixed(1)).join(", ")}]`
        : "";
      console.log(clr("1;36", `     ${baseline.tokPerSec.toFixed(2)} tok/s${range} ${baseline.correct ? "✅" : "❌"}`));
    } else {
      console.log(clr("1;31", `     Failed to benchmark`));
    }
    results.push(baseline);
  }
  return results;
}

// ── Regression detection ────────────────────────────────────────────

function detectRegressions(baselines: ModelBaseline[], after: ModelBaseline[]): RegressionReport[] {
  const reports: RegressionReport[] = [];
  for (const baseline of baselines) {
    const match = after.find(a => a.id === baseline.id);
    if (!match || baseline.tokPerSec == null) continue;

    const baselineTps = baseline.tokPerSec;
    const afterTps = match.tokPerSec ?? 0;
    const deltaTps = afterTps - baselineTps;
    const deltaPct = baselineTps > 0 ? deltaTps / baselineTps : 0;
    const threshold = Math.max(baselineTps * REGRESSION_PCT, REGRESSION_FLOOR);
    const regressed = deltaTps < -threshold;

    // Also check correctness regression
    const lostCorrectness = baseline.correct && !match.correct;

    reports.push({
      model: baseline,
      baselineTps,
      afterTps,
      deltaTps,
      deltaPct,
      regressed: regressed || lostCorrectness,
    });
  }
  return reports;
}

function formatRegressionTable(reports: RegressionReport[]): string {
  const lines: string[] = [];
  const regressed = reports.filter(r => r.regressed);
  const ok = reports.filter(r => !r.regressed);

  if (regressed.length > 0) {
    lines.push("### REGRESSIONS DETECTED:");
    for (const r of regressed) {
      const delta = r.deltaTps >= 0 ? `+${r.deltaTps.toFixed(2)}` : r.deltaTps.toFixed(2);
      const pct = (r.deltaPct * 100).toFixed(1);
      lines.push(`  ❌ ${r.model.id}: ${r.baselineTps.toFixed(2)} → ${r.afterTps.toFixed(2)} tok/s (${delta}, ${pct}%)`);
    }
  }
  if (ok.length > 0) {
    lines.push("### No regression:");
    for (const r of ok) {
      const delta = r.deltaTps >= 0 ? `+${r.deltaTps.toFixed(2)}` : r.deltaTps.toFixed(2);
      lines.push(`  ✅ ${r.model.id}: ${r.baselineTps.toFixed(2)} → ${r.afterTps.toFixed(2)} tok/s (${delta})`);
    }
  }
  return lines.join("\n");
}

// ── Prompt builders ─────────────────────────────────────────────────

function buildAgentPrompt(
  userPrompt: string,
  baselines: ModelBaseline[],
  attempt: number,
  regressionFeedback: string | null,
  buildError: string | null,
  testError: string | null,
): string {
  const sections: string[] = [
    "# ZINC Change Request",
    "",
    "## Your Task",
    userPrompt,
    "",
  ];

  if (buildError) {
    sections.push(
      "## ⚠ BUILD FAILURE — Fix this first",
      "Your previous changes broke the build. Fix the compilation error:",
      "```",
      buildError.slice(-3000),
      "```",
      "",
    );
  }

  if (testError) {
    sections.push(
      "## ⚠ TEST FAILURE — Fix this first",
      "Your previous changes broke tests. Fix them:",
      "```",
      testError.slice(-3000),
      "```",
      "",
    );
  }

  if (regressionFeedback) {
    sections.push(
      `## ⚠ REGRESSION DETECTED (attempt ${attempt}/${MAX_RETRIES})`,
      "Your previous changes caused performance regressions. You must SELECTIVELY",
      "adjust your changes to fix these regressions while preserving the gains.",
      "Do NOT do a full revert — identify the specific part causing the regression.",
      "",
      regressionFeedback,
      "",
    );
  }

  sections.push(
    "## Performance Baselines (MUST NOT REGRESS)",
    `Regression threshold: ${(REGRESSION_PCT * 100).toFixed(0)}% or ${REGRESSION_FLOOR} tok/s, whichever is larger.`,
    "",
  );
  for (const b of baselines) {
    if (b.tokPerSec != null) {
      sections.push(`  - ${b.id}: ${b.tokPerSec.toFixed(2)} tok/s [${b.tokPerSecSamples.map(s => s.toFixed(1)).join(", ")}] ${b.correct ? "✅correct" : "❌wrong"}`);
    }
  }
  sections.push("");

  sections.push(
    "## Hardware",
    "- Mac Studio M4 Max, 64 GB unified memory, 40-core GPU, 546 GB/s bandwidth",
    "- Apple GPU family: Apple9 (M4), simdgroup_matrix = true, bfloat = true",
    "- macOS, Metal compute only",
    "",
    "## Project Layout",
    "```",
    "src/compute/forward_metal.zig — Metal inference engine (main decode loop)",
    "src/metal/   — shim.h, shim.m (ObjC C API), device.zig, buffer.zig, command.zig",
    "src/shaders/metal/ — MSL compute shaders",
    "src/model/   — config.zig, gguf.zig, loader_metal.zig, tokenizer.zig",
    "src/main.zig — CLI entry point",
    "```",
    "",
    "## Rules",
    "1. Do NOT modify src/vulkan/, loops/, or .env.",
    "2. Zig 0.15.2 API: ArrayList is unmanaged (pass allocator to append/deinit).",
    "3. MSL shaders use 'main0' as entry point (SPIRV-Cross convention).",
    "4. Metal push constants go in buffer[n_bufs] (see shim.m mtl_dispatch).",
    "5. All existing tests must continue passing.",
    "6. Performance must not regress on ANY model (see baselines above).",
    "7. Make the minimal change needed. Don't refactor unrelated code.",
    "",
    "## Output Format",
    "After making your changes, print:",
    "@@@DESCRIPTION: <one-line summary of what you changed>",
  );

  return sections.join("\n");
}

// ── Prompt input ────────────────────────────────────────────────────

async function readPromptInteractive(): Promise<string> {
  if (!process.stdin.isTTY) {
    // Piped input — read all of stdin
    const chunks: string[] = [];
    for await (const chunk of process.stdin) {
      chunks.push(chunk.toString());
    }
    const text = chunks.join("").trim();
    if (!text) {
      console.error("No prompt provided on stdin.");
      process.exit(1);
    }
    return text;
  }

  // Interactive TTY
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    console.log(clr("1;36", "Enter your prompt (single line):"));
    rl.question(clr("2", "> "), (answer) => {
      rl.close();
      const text = answer.trim();
      if (!text) {
        console.error("Empty prompt.");
        process.exit(1);
      }
      resolve(text);
    });
  });
}

// ── Main ────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  let promptText: string | null = null;
  let promptFile: string | null = null;
  let agent: AgentKind = "claude";
  let model: string | undefined;
  let extraModelPaths: string[] = [];

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--prompt":
        promptText = args[++i];
        break;
      case "--prompt-file":
        promptFile = args[++i];
        break;
      case "--agent": {
        const v = args[++i];
        if (v !== "claude" && v !== "codex") {
          console.error(`Invalid --agent: ${v}. Use claude or codex.`);
          process.exit(1);
        }
        agent = v;
        break;
      }
      case "--model":
        model = args[++i];
        break;
      case "--models":
        extraModelPaths = args[++i].split(",").map(s => s.trim()).filter(Boolean);
        break;
      case "--help":
        console.log([
          "Usage: bun loops/guided_change.ts [options]",
          "",
          "Options:",
          "  --prompt <text>          Inline prompt",
          "  --prompt-file <path>     Read prompt from file",
          "  --agent <claude|codex>   Agent to use (default: claude)",
          "  --model <name>           Model override for the AI agent",
          "  --models <path,path,...>  Explicit GGUF model paths to benchmark",
          "",
          "If no --prompt or --prompt-file is given, reads from stdin.",
        ].join("\n"));
        process.exit(0);
    }
  }

  // Resolve prompt
  let userPrompt: string;
  if (promptText) {
    userPrompt = promptText;
  } else if (promptFile) {
    if (!existsSync(promptFile)) {
      console.error(`Prompt file not found: ${promptFile}`);
      process.exit(1);
    }
    userPrompt = (await readFile(promptFile, "utf8")).trim();
  } else {
    userPrompt = await readPromptInteractive();
  }

  const agentLabel = agent === "codex" ? "Codex" : "Claude";

  console.log(clr("1;36", "╔══════════════════════════════════════════════════════════════╗"));
  console.log(clr("1;36", "║  ZINC Guided Change                                          ║"));
  console.log(clr("1;36", "╚══════════════════════════════════════════════════════════════╝"));
  console.log(`  Agent: ${clr("1", agentLabel)}${model ? ` (${model})` : ""}`);
  console.log(`  Prompt: ${clr("2", userPrompt.length > 80 ? userPrompt.slice(0, 77) + "…" : userPrompt)}`);

  // Set up results dir
  const runId = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const runDir = join(RESULTS_DIR, runId);
  await mkdir(runDir, { recursive: true });
  console.log(`  Results: ${clr("2", runDir)}`);

  // ── Step 1: Discover models ─────────────────────────────────────
  console.log(clr("1;35", "\n" + SEP));
  console.log(clr("1;35", "  STEP 1: Discover models"));
  console.log(clr("1;35", SEP));

  const models = discoverModels(extraModelPaths);
  if (models.length === 0) {
    console.error("No models found. Set ZINC_MODEL env or use --models.");
    process.exit(1);
  }
  for (const m of models) {
    console.log(`  ${clr("1", m.id)}: ${clr("2", m.path)}`);
  }

  // ── Step 2: Build ───────────────────────────────────────────────
  console.log(clr("1;35", "\n" + SEP));
  console.log(clr("1;35", "  STEP 2: Build & test"));
  console.log(clr("1;35", SEP));

  const build = await buildProject();
  if (!build.ok) {
    console.error("Initial build failed. Fix build errors first.");
    process.exit(1);
  }
  const test = await runTests();
  if (!test.ok) {
    console.error("Initial tests failed. Fix test failures first.");
    process.exit(1);
  }

  // ── Step 3: Baseline benchmarks ─────────────────────────────────
  console.log(clr("1;35", "\n" + SEP));
  console.log(clr("1;35", "  STEP 3: Baseline benchmarks"));
  console.log(clr("1;35", SEP));

  const baselines = await benchmarkAll(models);
  await writeFile(join(runDir, "baselines.json"), JSON.stringify(baselines, null, 2));

  const validBaselines = baselines.filter(b => b.tokPerSec != null);
  if (validBaselines.length === 0) {
    console.error("No models could be benchmarked. Check model paths and binary.");
    process.exit(1);
  }

  console.log(clr("1;32", `\n  Baselines established for ${validBaselines.length} model(s)`));

  // ── Step 4: Git snapshot ────────────────────────────────────────
  await runCommand("git", ["add", "-A", "src/", "build.zig"]).catch(() => {});
  await runCommand("git", ["commit", "--allow-empty", "-m", `guided-change: pre ${runId}`]).catch(() => {});
  const preCommit = await runCommand("git", ["rev-parse", "HEAD"]);
  const preHash = preCommit.stdout.trim();

  // ── Step 5: Agent loop with regression checking ─────────────────
  let regressionFeedback: string | null = null;
  let buildError: string | null = null;
  let testError: string | null = null;
  let success = false;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    console.log(clr("1;35", "\n" + "═".repeat(64)));
    console.log(clr("1;35", `  ATTEMPT ${attempt}/${MAX_RETRIES}`));
    console.log(clr("1;35", "═".repeat(64)));

    // Build agent prompt
    const agentPrompt = buildAgentPrompt(userPrompt, baselines, attempt, regressionFeedback, buildError, testError);
    await writeFile(join(runDir, `attempt-${attempt}-prompt.md`), agentPrompt);

    // Run agent
    const agentResult = await runAgent(agent, agentPrompt, model);
    await writeFile(join(runDir, `attempt-${attempt}-agent.log`), agentResult.stdout + agentResult.stderr);

    const agentText = extractAgentText(agentResult.stdout);
    const descMatch = agentText.slice(-2000).match(/@@@DESCRIPTION:\s*(.+)/im);
    const description = descMatch?.[1]?.trim() ?? "Agent made changes";

    // Verify: build
    console.log(clr("1;33", "\n  📊 Verifying build..."));
    const verifyBuild = await buildProject();
    if (!verifyBuild.ok) {
      buildError = verifyBuild.output.slice(-3000);
      testError = null;
      regressionFeedback = null;
      console.log(clr("1;31", `  ❌ Build broken — will retry (${attempt}/${MAX_RETRIES})`));
      continue;
    }
    buildError = null;

    // Verify: tests
    const verifyTest = await runTests();
    if (!verifyTest.ok) {
      testError = verifyTest.output.slice(-3000);
      regressionFeedback = null;
      console.log(clr("1;31", `  ❌ Tests broken — will retry (${attempt}/${MAX_RETRIES})`));
      continue;
    }
    testError = null;

    // Verify: benchmark all models
    console.log(clr("1;33", "  📊 Benchmarking after changes..."));
    const afterResults = await benchmarkAll(models);
    await writeFile(join(runDir, `attempt-${attempt}-benchmarks.json`), JSON.stringify(afterResults, null, 2));

    // Check regressions
    const reports = detectRegressions(baselines, afterResults);
    const regressions = reports.filter(r => r.regressed);

    if (regressions.length === 0) {
      // No regressions — success!
      console.log(clr("1;32", "\n  ✅ No regressions detected!"));
      const table = formatRegressionTable(reports);
      console.log(clr("2", table));

      // Commit
      await runCommand("git", ["add", "-A", "src/", "build.zig"]).catch(() => {});
      await runCommand("git", ["commit", "-m", `guided-change: ${description}`]).catch(() => {});

      success = true;
      console.log(clr("1;32", "\n" + "═".repeat(64)));
      console.log(clr("1;32", `  CHANGE APPLIED: ${description}`));
      console.log(clr("1;32", "═".repeat(64)));
      break;
    }

    // Regressions found
    console.log(clr("1;31", `\n  ❌ ${regressions.length} regression(s) detected`));
    const table = formatRegressionTable(reports);
    console.log(table);

    if (attempt < MAX_RETRIES) {
      regressionFeedback = table;
      console.log(clr("1;33", `  Will ask agent to selectively fix (attempt ${attempt + 1}/${MAX_RETRIES})`));
    } else {
      // Out of retries — full revert
      console.log(clr("1;31", `\n  ↩ FULL REVERT — exhausted ${MAX_RETRIES} attempts`));
      await runCommand("git", ["reset", "--hard", preHash]);
      console.log(clr("1;31", "  All changes reverted to pre-change state."));
    }
  }

  // Summary
  console.log(clr("1;36", "\n" + SEP));
  console.log(clr("1;36", `  Result: ${success ? "SUCCESS" : "FAILED (reverted)"}`));
  console.log(clr("1;36", `  Logs: ${runDir}`));
  console.log(clr("1;36", SEP));

  process.exit(success ? 0 : 1);
}

if (import.meta.main) {
  main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
  });
}
