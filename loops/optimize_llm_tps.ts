#!/usr/bin/env bun
/**
 * LLM TPS Optimization Loop
 *
 * Iteratively improves llama.cpp inference speed on AMD RDNA4 by
 * spawning an AI agent each cycle to make one optimization, then
 * benchmarking and keeping/reverting based on results.
 *
 * Usage:
 *   bun scripts/optimize_llm_tps.ts --agent claude "improve generation throughput"
 *   bun scripts/optimize_llm_tps.ts --agent codex --cycles 50 --target-tps 150
 *   bun scripts/optimize_llm_tps.ts --dry-run   # benchmark only
 */

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

// ── Color & display ──────────────────────────────────────────────────

const TTY = process.stdout.isTTY ?? false;
const NO_COLOR = "NO_COLOR" in process.env;
const FORCE_COLOR =
  process.env.FORCE_COLOR === "1" || process.env.CLICOLOR_FORCE === "1";
const COLOR_ENABLED = !NO_COLOR && (TTY || FORCE_COLOR);

function clr(code: string, text: string): string {
  return COLOR_ENABLED ? `\x1b[${code}m${text}\x1b[0m` : text;
}

const SEP = "─".repeat(64);
const MAX_DIFF_LINES = 8;

// ── Garbage detection (shared by extraction, state sanitization, idea filtering) ──

/** Returns true if a string looks like garbage (JSON fragments, code, tool output) rather than natural language. */
export function isGarbageString(s: string): boolean {
  if (!s || s.length > 200 || s.length < 5) return true;
  if (s.startsWith("<") && s.includes(">")) return true;
  if (s.includes("session_id") || s.includes("parent_tool_use_id") || s.includes("tool_use_result")) return true;
  if (s.includes("uuid")) return true;
  if (/^\s*[{[\\/|`#→$'"(]/.test(s)) return true;
  if ((s.match(/[{}[\]]/g)?.length ?? 0) > 2) return true;
  if (s.includes("\\n") || s.includes('\\"') || s.includes("\\\\")) return true;
  if (/→/.test(s)) return true;
  if (s.includes("console.") || s.includes("await ") || s.includes("const ")) return true;
  if (/\b(string|number|boolean|null|undefined)\b/.test(s) && s.includes(";")) return true;
  if (/\d{3,4}→/.test(s)) return true;
  const words = s.split(/\s+/).filter(w => w.length > 0);
  if (words.length < 2) return true;
  const alphaWords = words.filter(w => /^[a-zA-Z]/.test(w));
  if (alphaWords.length < words.length * 0.4) return true;
  return false;
}

// ── Constants ────────────────────────────────────────────────────────

const PROJECT_ROOT = resolve(import.meta.dir, "..");
const RESULTS_DIR = resolve(PROJECT_ROOT, ".llm_optimize");

const LLM_HOST = process.env.LLM_HOST ?? "127.0.0.1";
const LLM_SSH_PORT = Number(process.env.LLM_SSH_PORT ?? "22");
const LLM_PORT = Number(process.env.LLM_PORT ?? "8080");
const LLAMA_CPP_DIR = process.env.LLAMA_CPP_DIR ?? "/root/llama.cpp";
const BASE_URL = `http://${LLM_HOST}:${LLM_PORT}/v1/chat/completions`;

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)",
  "Bash(git revert:*)",
  "Bash(git restore:*)",
  "Bash(git reset:*)",
  "Bash(git stash:*)",
  "Bash(git clean:*)",
];

type AgentKind = "claude" | "codex";

// ── Claude stream formatter (stream-json output) ─────────────────────

type ClaudeStreamState = {
  currentToolName: string | null;
  currentBlockIsToolUse: boolean;
  inputJsonBuffer: string;
  inTextBlock: boolean;
  sawTextDeltaInCurrentMessage: boolean;
};

export function coerceDisplayText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  if (typeof value === "number" || typeof value === "boolean")
    return String(value);
  if (Array.isArray(value)) {
    const parts = value
      .map((e) => coerceDisplayText(e))
      .filter((e) => e.trim());
    if (parts.length > 0) return parts.join("\n");
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return "";
    }
  }
  if (typeof value === "object") {
    const r = value as Record<string, unknown>;
    const parts = [
      r.text,
      r.message,
      r.output,
      r.stdout,
      r.stderr,
      r.content,
      r.result,
    ]
      .map((e) => coerceDisplayText(e))
      .filter((e) => e.trim());
    if (parts.length > 0) return parts.join("\n");
    try {
      return JSON.stringify(r, null, 2);
    } catch {
      return "";
    }
  }
  return "";
}

function formatToolInput(toolName: string, inputJson: string): string {
  let input: Record<string, unknown> = {};
  try {
    input = JSON.parse(inputJson) as Record<string, unknown>;
  } catch {
    /* partial */
  }
  const name = toolName.toLowerCase();
  const out: string[] = [];

  if (name === "edit") {
    const fp = (input.file_path as string | undefined) ?? "?";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}`));
    const oldLines = ((input.old_string as string | undefined) ?? "").split(
      "\n",
    );
    const newLines = ((input.new_string as string | undefined) ?? "").split(
      "\n",
    );
    for (const l of oldLines.slice(0, MAX_DIFF_LINES))
      out.push(clr("31", `   - ${l}`));
    if (oldLines.length > MAX_DIFF_LINES)
      out.push(
        clr("2", `   - … (${oldLines.length - MAX_DIFF_LINES} more)`),
      );
    for (const l of newLines.slice(0, MAX_DIFF_LINES))
      out.push(clr("32", `   + ${l}`));
    if (newLines.length > MAX_DIFF_LINES)
      out.push(
        clr("2", `   + … (${newLines.length - MAX_DIFF_LINES} more)`),
      );
  } else if (name === "write") {
    const fp = (input.file_path as string | undefined) ?? "?";
    const lineCount = ((input.content as string | undefined) ?? "").split(
      "\n",
    ).length;
    out.push(
      clr("2", ` → ${fp.split("/").slice(-3).join("/")} (${lineCount} lines)`),
    );
  } else if (name === "bash") {
    const cmd = (input.command as string | undefined) ?? "?";
    out.push(
      clr("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "…" : cmd}`),
    );
  } else if (name === "read") {
    const fp = (input.file_path as string | undefined) ?? "?";
    const offset = input.offset != null ? ` @line ${input.offset}` : "";
    out.push(
      clr("2", ` → ${fp.split("/").slice(-3).join("/")}${offset}`),
    );
  } else if (name === "grep") {
    const pattern = (input.pattern as string | undefined) ?? "?";
    const path = (input.path as string | undefined) ?? "";
    out.push(
      clr(
        "2",
        ` → /${pattern}/${path ? ` in ${path.split("/").slice(-2).join("/")}` : ""}`,
      ),
    );
  } else if (name === "glob") {
    out.push(clr("2", ` → ${(input.pattern as string | undefined) ?? "?"}`));
  }
  return out.length > 0 ? out.join("\n") + "\n" : "";
}

function formatToolResult(result: Record<string, unknown>): string {
  const file = result.file as Record<string, unknown> | undefined;
  if (file)
    return (
      clr("32", `   ☑ accepted`) +
      clr("2", `  (${file.numLines ?? "?"} lines)`) +
      "\n"
    );
  const content = coerceDisplayText(result.content);
  if (!content.trim()) return clr("32", "   ☑ accepted") + "\n";
  const lines = content.split("\n").filter((l) => l.trim());
  const tail = lines.slice(-3);
  const ellipsis = lines.length > 3 ? clr("2", "   …\n") : "";
  const body = tail.map((l) => clr("2", `   ${l.trim()}`)).join("\n");
  return clr("32", "   ☑ accepted") + "\n" + ellipsis + body + "\n";
}

function formatClaudeStreamLine(
  rawLine: string,
  state: ClaudeStreamState,
): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try {
    event = JSON.parse(rawLine) as Record<string, unknown>;
  } catch {
    return rawLine + "\n";
  }

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
        state.sawTextDeltaInCurrentMessage = true;
        return delta.text as string;
      }
      return null;
    }
    if (e.type === "content_block_stop") {
      if (state.currentBlockIsToolUse) {
        state.currentBlockIsToolUse = false;
        const detail = formatToolInput(
          state.currentToolName ?? "",
          state.inputJsonBuffer,
        );
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
    const result = event.tool_use_result as
      | Record<string, unknown>
      | undefined;
    if (result) return formatToolResult(result);
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
        if (b?.type === "text" && typeof b.text === "string" && b.text.trim())
          parts.push(b.text);
      }
      const text = parts.join("\n");
      if (!text.trim()) return null;
      if (state.sawTextDeltaInCurrentMessage) {
        state.sawTextDeltaInCurrentMessage = false;
        return null;
      }
      return clr("96", text) + "\n";
    }
    return null;
  }
  return null;
}

// ── Codex stream formatter ───────────────────────────────────────────

type CodexStreamState = {
  inDiffBlock: boolean;
  diffLines: string[];
  lastDiffHash: string | null;
  suppressedDuplicateBlocks: number;
  startedCommandIds: Set<string>;
};

function formatCodexDiffLine(line: string): string {
  if (!line.trim()) return "\n";
  if (line.startsWith("file update:")) return `\n${clr("35", line)}\n`;
  if (line.startsWith("diff --git ")) return `${clr("1;36", line)}\n`;
  if (line.startsWith("index ")) return `${clr("2", line)}\n`;
  if (line.startsWith("--- ") || line.startsWith("+++ "))
    return `${clr("36", line)}\n`;
  if (line.startsWith("@@")) return `${clr("33", line)}\n`;
  if (line.startsWith("+") && !line.startsWith("+++"))
    return `${clr("32", line)}\n`;
  if (line.startsWith("-") && !line.startsWith("---"))
    return `${clr("31", line)}\n`;
  return `${line}\n`;
}

function isCodexDiffStart(line: string): boolean {
  return (
    line.startsWith("file update:") ||
    line.startsWith("diff --git ") ||
    line.startsWith("index ") ||
    line.startsWith("--- ") ||
    line.startsWith("+++ ") ||
    line.startsWith("@@")
  );
}

function isCodexDiffContinuation(line: string): boolean {
  if (!line.trim()) return true;
  return (
    line.startsWith("diff --git ") ||
    line.startsWith("index ") ||
    line.startsWith("--- ") ||
    line.startsWith("+++ ") ||
    line.startsWith("@@") ||
    (line.startsWith("+") && !line.startsWith("++")) ||
    (line.startsWith("-") && !line.startsWith("--"))
  );
}

function flushCodexDiffBlock(state: CodexStreamState): string | null {
  if (!state.inDiffBlock) return null;
  const blockRaw = state.diffLines.join("\n");
  const blockHash = createHash("sha256").update(blockRaw).digest("hex");
  state.inDiffBlock = false;
  state.diffLines = [];
  if (state.lastDiffHash === blockHash) {
    state.suppressedDuplicateBlocks += 1;
    return null;
  }
  let out = "";
  if (state.suppressedDuplicateBlocks > 0) {
    out += `${clr("2", `… identical diff repeated ${state.suppressedDuplicateBlocks} more time${state.suppressedDuplicateBlocks === 1 ? "" : "s"}`)}\n`;
    state.suppressedDuplicateBlocks = 0;
  }
  state.lastDiffHash = blockHash;
  for (const line of blockRaw.split("\n")) out += formatCodexDiffLine(line);
  return out;
}

function finalizeCodexStream(state: CodexStreamState): string | null {
  let out = "";
  const flushed = flushCodexDiffBlock(state);
  if (flushed) out += flushed;
  if (state.suppressedDuplicateBlocks > 0) {
    out += `${clr("2", `… identical diff repeated ${state.suppressedDuplicateBlocks} more time${state.suppressedDuplicateBlocks === 1 ? "" : "s"}`)}\n`;
    state.suppressedDuplicateBlocks = 0;
  }
  return out || null;
}

function formatCodexJsonEvent(
  rawLine: string,
  state: CodexStreamState,
): string | null | undefined {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try {
    event = JSON.parse(rawLine) as Record<string, unknown>;
  } catch {
    return undefined;
  }
  const eventType = typeof event.type === "string" ? event.type : "";
  if (["thread.started", "turn.started", "turn.completed"].includes(eventType))
    return null;
  if (eventType === "error") {
    const message = typeof event.message === "string" ? event.message : "";
    return message ? `${clr("31", message)}\n` : null;
  }
  if (eventType.startsWith("item.")) {
    const item = event.item as Record<string, unknown> | undefined;
    if (!item) return null;
    const itemType = typeof item.type === "string" ? item.type : "";
    const itemId = typeof item.id === "string" ? item.id : "";
    const phase = eventType.slice("item.".length);
    if (itemType === "reasoning") {
      const text = coerceDisplayText(
        item.summary ?? item.text ?? item.message ?? item.content,
      );
      return text ? `${clr("2", `thinking: ${text}`)}\n` : null;
    }
    if (itemType === "command_execution") {
      const cmd = coerceDisplayText(
        item.command ?? (item.input as any)?.command ?? "",
      ).trim();
      const output = coerceDisplayText(
        item.aggregated_output ?? item.output ?? item.stdout ?? "",
      );
      const exitCode =
        typeof item.exit_code === "number" ? item.exit_code : null;
      const startedAlready = itemId
        ? state.startedCommandIds.has(itemId)
        : false;
      if (phase === "started") {
        if (itemId) state.startedCommandIds.add(itemId);
        return cmd
          ? `\n${clr("33", "🔧 bash")}\n${clr("2", `   $ ${cmd}`)}\n`
          : `\n${clr("33", "🔧 bash")}\n`;
      }
      let out = "";
      if (!startedAlready) {
        out += `\n${clr("33", "🔧 bash")}\n`;
        if (cmd) out += clr("2", `   $ ${cmd}`) + "\n";
      }
      if (phase === "completed") {
        const lines = output.split("\n").filter((l) => l.trim());
        const tail = lines.slice(-3);
        const statusColor =
          exitCode === 0 ? "32" : exitCode == null ? "33" : "31";
        const statusText =
          exitCode === 0
            ? "   ☑ accepted"
            : exitCode == null
              ? "   ⚠ completed"
              : `   ✖ exit ${exitCode}`;
        const body =
          tail.length > 0
            ? (lines.length > 3 ? clr("2", "   …\n") : "") +
              tail.map((l) => clr("2", `   ${l.trim()}`)).join("\n") +
              "\n"
            : "";
        out += `${clr(statusColor, statusText)}\n${body}`;
      }
      if (itemId) state.startedCommandIds.delete(itemId);
      return out || null;
    }
    if (itemType === "file_change" && phase === "completed") {
      const changes = [item.changes, item.file_changes, item.files].flatMap(
        (c) => {
          if (!c) return [];
          if (Array.isArray(c))
            return c.map((e: any) => ({
              path: e?.path ?? e?.file_path ?? "?",
              action: e?.change_type ?? e?.kind ?? null,
            }));
          return [];
        },
      );
      if (changes.length === 0) return null;
      let out = `\n${clr("35", "📝 file change")}\n`;
      for (const c of changes.slice(0, 6))
        out +=
          clr("2", `   ${c.action ? c.action + ": " : ""}${c.path}`) + "\n";
      return out;
    }
    if (itemType === "agent_message" && phase === "completed") {
      const text = coerceDisplayText(
        item.text ?? item.message ?? item.output_text ?? item.content ?? "",
      );
      return text ? `${clr("96", text)}\n` : null;
    }
    if (itemType === "error" && phase === "completed") {
      const text = coerceDisplayText(
        item.text ?? item.message ?? item.content ?? "",
      );
      return text ? `${clr("33", text)}\n` : null;
    }
  }
  return null;
}

function formatCodexStreamLine(
  rawLine: string,
  state: CodexStreamState,
): string | null {
  const jsonFormatted = formatCodexJsonEvent(rawLine, state);
  if (jsonFormatted !== undefined) return jsonFormatted;
  if (isCodexDiffStart(rawLine)) {
    const flushed = flushCodexDiffBlock(state) ?? "";
    state.inDiffBlock = true;
    state.diffLines = [rawLine];
    return flushed || null;
  }
  if (state.inDiffBlock) {
    if (isCodexDiffContinuation(rawLine)) {
      state.diffLines.push(rawLine);
      return null;
    }
    const flushed = flushCodexDiffBlock(state) ?? "";
    const next = formatCodexStreamLine(rawLine, state) ?? "";
    return flushed + next || null;
  }
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

// ── Command runner with line-based stream formatting ─────────────────

type RunResult = { exitCode: number; stdout: string; stderr: string };

async function runCommand(
  cmd: string,
  args: string[],
  opts: {
    cwd?: string;
    env?: NodeJS.ProcessEnv;
    streamOutput?: boolean;
    timeout?: number;
    stdoutLineFormatter?: (line: string) => string | null;
    stderrLineFormatter?: (line: string) => string | null;
  } = {},
): Promise<RunResult> {
  const streamOutput = opts.streamOutput ?? true;
  return new Promise((res, rej) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? PROJECT_ROOT,
      env: opts.env,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout,
    });
    let stdout = "",
      stderr = "",
      lineBuffer = "",
      stderrLineBuffer = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (!streamOutput) return;
      if (opts.stdoutLineFormatter) {
        lineBuffer += text;
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const f = opts.stdoutLineFormatter(line);
          if (f !== null) process.stdout.write(f);
        }
      } else {
        process.stdout.write(text);
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (!streamOutput) return;
      if (opts.stderrLineFormatter) {
        stderrLineBuffer += text;
        const lines = stderrLineBuffer.split("\n");
        stderrLineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const f = opts.stderrLineFormatter(line);
          if (f !== null) process.stderr.write(f);
        }
      } else {
        process.stderr.write(text);
      }
    });
    child.on("error", rej);
    child.on("close", (code) => {
      if (streamOutput && opts.stdoutLineFormatter && lineBuffer.trim()) {
        const f = opts.stdoutLineFormatter(lineBuffer);
        if (f !== null) process.stdout.write(f);
      }
      if (
        streamOutput &&
        opts.stderrLineFormatter &&
        stderrLineBuffer.trim()
      ) {
        const f = opts.stderrLineFormatter(stderrLineBuffer);
        if (f !== null) process.stderr.write(f);
      }
      res({ exitCode: code ?? 1, stdout, stderr });
    });
  });
}

// ── Agent invocation ─────────────────────────────────────────────────

function buildClaudeArgs(prompt: string, model?: string): string[] {
  const args = [
    "-p",
    "--verbose",
    "--output-format",
    "stream-json",
    "--include-partial-messages",
    `--disallowed-tools=${BLOCKED_GIT_OPS.join(",")}`,
    "--permission-mode",
    "bypassPermissions",
    "--effort",
    "high",
  ];
  if (model) args.push("--model", model);
  args.push(prompt);
  return args;
}

function buildCodexArgs(prompt: string, model?: string): string[] {
  const args = [
    "exec",
    "--skip-git-repo-check",
    "--json",
    "--color",
    "never",
    "--sandbox",
    "workspace-write",
    "--cd",
    PROJECT_ROOT,
  ];
  if (model) args.push("--model", model);
  args.push(prompt);
  return args;
}

export function formatElapsed(startMs: number): string {
  const sec = Math.floor((Date.now() - startMs) / 1000);
  if (sec < 60) return `${sec}s`;
  return `${Math.floor(sec / 60)}m${sec % 60}s`;
}

async function runAgent(
  agent: AgentKind,
  prompt: string,
  model?: string,
): Promise<RunResult> {
  const label = agent === "codex" ? "Codex" : "Claude";
  console.log(clr("1;34", SEP));
  console.log(clr("1;34", `  🧠 PROMPT (${label})`));
  console.log(clr("1;34", SEP));
  // Show first 10 lines of prompt to keep output manageable
  const promptLines = prompt.split("\n");
  for (const line of promptLines.slice(0, 10))
    process.stdout.write(clr("2", line) + "\n");
  if (promptLines.length > 10)
    process.stdout.write(
      clr("2", `… (${promptLines.length - 10} more lines)\n`),
    );
  console.log(clr("1;34", SEP));

  console.log(clr("1;36", SEP));
  console.log(clr("1;36", `  💬 RESPONSE (${label})`));
  console.log(clr("1;36", SEP));

  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(
      clr("2", `\n⏳ still running (${formatElapsed(startedAt)} elapsed)...\n`),
    );
  }, 20_000);

  let result: RunResult;
  if (agent === "codex") {
    const codexState: CodexStreamState = {
      inDiffBlock: false,
      diffLines: [],
      lastDiffHash: null,
      suppressedDuplicateBlocks: 0,
      startedCommandIds: new Set(),
    };
    result = await runCommand("codex", buildCodexArgs(prompt, model), {
      streamOutput: true,
      timeout: 600_000,
      stdoutLineFormatter: (line) => formatCodexStreamLine(line, codexState),
      stderrLineFormatter: formatCodexStderrLine,
    });
    const tail = finalizeCodexStream(codexState);
    if (tail) process.stdout.write(tail);
  } else {
    const claudeState: ClaudeStreamState = {
      currentToolName: null,
      currentBlockIsToolUse: false,
      inputJsonBuffer: "",
      inTextBlock: false,
      sawTextDeltaInCurrentMessage: false,
    };
    result = await runCommand("claude", buildClaudeArgs(prompt, model), {
      streamOutput: true,
      timeout: 600_000,
      stdoutLineFormatter: (line) =>
        formatClaudeStreamLine(line, claudeState),
    });
  }

  clearInterval(heartbeat);
  console.log(clr("1;36", SEP));
  console.log(
    clr("1;32", `  ✅ ${label} done in ${formatElapsed(startedAt)}`),
  );
  console.log(clr("1;36", SEP));

  return result;
}

// ── Stream-JSON text extraction ──────────────────────────────────────

/** Extract assembled text content from Claude's stream-json stdout. */
export function extractTextFromStreamJson(raw: string): string {
  const parts: string[] = [];
  let hasStreamEvents = false;
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    try {
      const ev = JSON.parse(line) as Record<string, unknown>;
      if (ev.type === "stream_event") {
        hasStreamEvents = true;
        const e = ev.event as Record<string, unknown> | undefined;
        if (e?.type === "content_block_delta") {
          const delta = e.delta as Record<string, unknown> | undefined;
          if (delta?.type === "text_delta") {
            parts.push((delta.text as string) ?? "");
          }
        }
      }
    } catch {
      if (!hasStreamEvents) {
        parts.push(line);
      }
    }
  }
  return parts.join("");
}

/** Extract thinking content from Claude's stream-json stdout. */
function extractThinkingFromStreamJson(raw: string): string {
  const parts: string[] = [];
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    try {
      const ev = JSON.parse(line) as Record<string, unknown>;
      if (ev.type === "stream_event") {
        const e = ev.event as Record<string, unknown> | undefined;
        if (e?.type === "content_block_delta") {
          const delta = e.delta as Record<string, unknown> | undefined;
          if (delta?.type === "thinking_delta") {
            parts.push((delta.thinking as string) ?? "");
          }
        }
      }
    } catch { /* skip */ }
  }
  return parts.join("");
}

/** Extract SSH commands the agent executed from stream-json stdout. */
function extractSSHCommands(raw: string): string[] {
  const cmds: string[] = [];
  let currentTool: string | null = null;
  const inputParts: string[] = [];
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    try {
      const ev = JSON.parse(line) as Record<string, unknown>;
      if (ev.type === "stream_event") {
        const e = ev.event as Record<string, unknown> | undefined;
        if (e?.type === "content_block_start") {
          const cb = e.content_block as Record<string, unknown> | undefined;
          if (cb?.type === "tool_use" && cb?.name === "Bash") {
            currentTool = "Bash";
            inputParts.length = 0;
          }
        } else if (e?.type === "content_block_delta") {
          const delta = e.delta as Record<string, unknown> | undefined;
          if (delta?.type === "input_json_delta" && currentTool === "Bash") {
            inputParts.push((delta.partial_json as string) ?? "");
          }
        } else if (e?.type === "content_block_stop" && currentTool === "Bash") {
          try {
            const parsed = JSON.parse(inputParts.join("")) as Record<string, unknown>;
            const cmd = (parsed.command as string) ?? "";
            if (cmd.includes("ssh")) cmds.push(cmd);
          } catch { /* skip */ }
          currentTool = null;
        }
      }
    } catch { /* skip */ }
  }
  return cmds;
}

/** Summarize what the agent did from SSH commands (last resort description). */
export function summarizeFromSSHCommands(cmds: string[]): string {
  // Look for the last write/modify command (sed -i, tee, patch, cmake --build, systemctl)
  const writeCmds = cmds.filter(c =>
    c.includes("sed -i") || c.includes("tee") || c.includes("patch") ||
    c.includes("cmake --build") || c.includes("cat >") || c.includes("echo ") && c.includes(">") ||
    c.includes("systemctl") || c.includes("node -e")
  );
  if (writeCmds.length > 0) {
    // Extract the essence of the last modification
    const last = writeCmds[writeCmds.length - 1];
    // Try to extract the remote command after ssh ... "command"
    const m = last.match(/ssh\s.*?"(.+)"/s) ?? last.match(/ssh\s.*?'(.+)'/s);
    if (m) return `Agent modified server: ${m[1].slice(0, 80)}`;
    return `Agent ran: ${last.slice(0, 80)}`;
  }
  if (cmds.length > 0) return `Agent explored server (${cmds.length} SSH commands, no modifications detected)`;
  return "Unknown optimization";
}

// ── SSH helper ───────────────────────────────────────────────────────

async function ssh(
  command: string,
  timeout = 120_000,
): Promise<string> {
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    ["-p", String(LLM_SSH_PORT), `root@${LLM_HOST}`, command],
    { streamOutput: false, timeout },
  );
  if (exitCode !== 0 && !stderr.includes("Warning")) {
    throw new Error(`SSH failed (${exitCode}): ${stderr.slice(0, 500)}`);
  }
  return stdout.trim();
}

// ── Benchmark ────────────────────────────────────────────────────────

type BenchResult = {
  prefill_tps: number;
  generation_tps: number;
  cognitive_score: number;
  cognitive_total: number;
  total_time_s: number;
  details: string[];
};

async function callLLM(
  prompt: string,
  opts: { thinking?: boolean; maxTokens?: number } = {},
): Promise<{
  content: string;
  reasoning: string;
  prefill_tps: number;
  gen_tps: number;
  prompt_tokens: number;
  gen_tokens: number;
}> {
  const body: Record<string, unknown> = {
    model: "qwen",
    messages: [{ role: "user", content: prompt }],
    max_tokens: opts.maxTokens ?? 512,
    temperature: 0.3,
  };
  if (opts.thinking === false) {
    body.chat_template_kwargs = { enable_thinking: false };
  }

  const resp = await fetch(BASE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Connection": "keep-alive" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180_000),
  });
  const data = (await resp.json()) as Record<string, unknown>;
  const choice = (data.choices as Array<Record<string, unknown>>)?.[0];
  const msg = choice?.message as Record<string, string>;
  const timings = data.timings as Record<string, number>;

  return {
    content: msg?.content ?? "",
    reasoning: msg?.reasoning_content ?? "",
    prefill_tps: timings?.prompt_per_second ?? 0,
    gen_tps: timings?.predicted_per_second ?? 0,
    prompt_tokens: timings?.prompt_n ?? 0,
    gen_tokens: timings?.predicted_n ?? 0,
  };
}

const COGNITIVE_TESTS: Array<{
  name: string;
  prompt: string;
  check: (r: string) => boolean;
}> = [
  {
    name: "Arithmetic",
    prompt: "What is (17*23)+(45*12)-131? Just the number.",
    check: (r) => r.includes("800"),
  },
  {
    name: "Probability",
    prompt: "Roll two fair dice. P(sum=7) as simplified fraction?",
    check: (r) => r.includes("1/6"),
  },
  {
    name: "Syllogism",
    prompt:
      'All roses are flowers. Some flowers fade quickly. Can we conclude some roses fade quickly? Answer "no" or "yes" first.',
    check: (r) => r.toLowerCase().startsWith("no"),
  },
  {
    name: "Bug detection",
    prompt:
      "Bug in this? def binary_search(arr,t):\n low,high=0,len(arr)\n while low<=high:\n  mid=(low+high)//2\n  if arr[mid]==t:return mid\n  elif arr[mid]<t:low=mid+1\n  else:high=mid-1\n return -1\nOne sentence.",
    check: (r) =>
      r.toLowerCase().includes("len(arr)") ||
      (r.toLowerCase().includes("high") &&
        (r.toLowerCase().includes("index") ||
          r.toLowerCase().includes("bound"))),
  },
  {
    name: "Code output",
    prompt:
      "What prints? x=[1,2,3,4,5]; y=x[1:4]; y[0]=99; print(x); print(y)",
    check: (r) => r.includes("[1, 2, 3, 4, 5]") && r.includes("[99, 3, 4]"),
  },
  {
    name: "Diagnosis",
    prompt:
      "Patient: fever, joint pain, returned from tropics, mosquito bites. Two most likely diagnoses?",
    check: (r) =>
      ["dengue", "malaria", "chikungunya", "zika"].some((d) =>
        r.toLowerCase().includes(d),
      ),
  },
];

async function warmupServer(): Promise<void> {
  // Warm up to avoid cold-start measurement artifacts
  for (let i = 0; i < 2; i++) {
    try {
      await callLLM("Hello", { thinking: false, maxTokens: 16 });
    } catch {}
  }
}

async function runBenchmark(): Promise<BenchResult> {
  await warmupServer();
  const start = Date.now();
  const details: string[] = [];

  // 1. Prefill (large prompt, small output)
  const prefillResult = await callLLM(
    "You are a helpful coding assistant. Analyze this async service code:\n\n" +
      "import asyncio, aiohttp, time, hashlib, json, logging\n" +
      "from dataclasses import dataclass, field\nfrom typing import Optional, List, Dict, Any\n" +
      "from collections import defaultdict\n\nlogger = logging.getLogger(__name__)\n\n" +
      "@dataclass\nclass RequestMetrics:\n    total_requests: int = 0\n    failed_requests: int = 0\n" +
      "    total_latency: float = 0.0\n    min_latency: float = float('inf')\n" +
      "    max_latency: float = 0.0\n    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))\n\n" +
      "    @property\n    def avg_latency(self): return self.total_latency / max(self.total_requests, 1)\n" +
      "    @property\n    def success_rate(self): return (self.total_requests - self.failed_requests) / max(self.total_requests, 1)\n\n" +
      "class RateLimiter:\n    def __init__(self, max_rps=10): self.max_rps = max_rps; self.tokens = max_rps; self.last_refill = time.monotonic(); self._lock = asyncio.Lock()\n" +
      "    async def acquire(self):\n        async with self._lock:\n            now = time.monotonic(); elapsed = now - self.last_refill\n" +
      "            self.tokens = min(self.max_rps, self.tokens + elapsed * self.max_rps); self.last_refill = now\n" +
      "            if self.tokens < 1: await asyncio.sleep((1 - self.tokens) / self.max_rps); self.tokens = 0\n" +
      "            else: self.tokens -= 1\n\nBrief review.",
    { thinking: false, maxTokens: 512 },
  );
  details.push(
    `Prefill: ${prefillResult.prefill_tps.toFixed(1)} tok/s (${prefillResult.prompt_tokens} tok)`,
  );

  // 2. Generation (long output)
  const genResult = await callLLM(
    "Write a TypeScript BST with insert, delete, search, all traversals. Type annotations.",
    { thinking: false, maxTokens: 1024 },
  );
  details.push(
    `Generation: ${genResult.gen_tps.toFixed(1)} tok/s (${genResult.gen_tokens} tok)`,
  );

  // 3. Cognitive (no-thinking, fast)
  let passed = 0;
  for (const test of COGNITIVE_TESTS) {
    try {
      const r = await callLLM(test.prompt, { thinking: false, maxTokens: 256 });
      if (test.check(r.content || r.reasoning)) passed++;
      else details.push(`  FAIL: ${test.name}`);
    } catch {
      details.push(`  ERROR: ${test.name}`);
    }
  }
  details.push(`Cognitive: ${passed}/${COGNITIVE_TESTS.length}`);

  return {
    prefill_tps: prefillResult.prefill_tps,
    generation_tps: genResult.gen_tps,
    cognitive_score: passed,
    cognitive_total: COGNITIVE_TESTS.length,
    total_time_s: (Date.now() - start) / 1000,
    details,
  };
}

function printBench(label: string, b: BenchResult): void {
  console.log(clr("1;33", `\n  📊 ${label}`));
  console.log(
    `  Prefill:    ${clr("1", b.prefill_tps.toFixed(1) + " tok/s")}`,
  );
  console.log(
    `  Generation: ${clr("1", b.generation_tps.toFixed(1) + " tok/s")}`,
  );
  console.log(
    `  Cognitive:  ${clr("1", `${b.cognitive_score}/${b.cognitive_total}`)}`,
  );
  console.log(`  Time:       ${b.total_time_s.toFixed(1)}s`);
  for (const d of b.details.filter((l) => l.startsWith("  ")))
    console.log(clr("2", `  ${d}`));
}

// ── Cycle state (persisted between runs) ─────────────────────────────

type CycleResult = {
  cycle: number;
  timestamp: string;
  baseline: BenchResult;
  optimized: BenchResult | null;
  gen_delta_pct: number;
  prefill_delta_pct: number;
  kept: boolean;
  description: string;
  nextIdeas: string[];
  selfAnalysis: string;
  error?: string;
};

type RunState = {
  runId: string;
  target_tps: number;
  originalBaseline: BenchResult;
  currentBest: BenchResult;
  cycles: CycleResult[];
  ideas: string[];
  failedApproaches: string[];
};

async function loadState(runDir: string): Promise<RunState | null> {
  const p = join(runDir, "state.json");
  if (!existsSync(p)) return null;
  return JSON.parse(await readFile(p, "utf8")) as RunState;
}

async function saveState(runDir: string, state: RunState): Promise<void> {
  await writeFile(join(runDir, "state.json"), JSON.stringify(state, null, 2));
}

// ── Prompt builder ───────────────────────────────────────────────────

function buildOptimizationPrompt(state: RunState): string {
  const { currentBest: bl, cycles, ideas, failedApproaches, target_tps } = state;

  // Truncate helper to prevent garbage strings from bloating the prompt
  const trunc = (s: string, max: number) => s && s.length > max ? s.slice(0, max) + "…" : s;

  const historyBlock =
    cycles.length > 0
      ? cycles
          .slice(-15)
          .map(
            (h) => {
              const desc = trunc(h.description, 80);
              const analysis = trunc(h.selfAnalysis, 100);
              return `  #${h.cycle}: ${desc} → gen:${h.optimized?.generation_tps.toFixed(1) ?? "ERR"} prefill:${h.optimized?.prefill_tps.toFixed(0) ?? "ERR"} (${h.gen_delta_pct > 0 ? "+" : ""}${h.gen_delta_pct.toFixed(1)}%) ${h.kept ? "✅ KEPT" : "❌ REVERTED"}${analysis ? ` — ${analysis}` : ""}`;
            },
          )
          .join("\n")
      : "  (none yet)";

  const failedBlock =
    failedApproaches.length > 0
      ? failedApproaches.map((f, n) => `  ${n + 1}. ${trunc(f, 120)}`).join("\n")
      : "  (none yet)";

  const ideasBlock =
    ideas.length > 0
      ? ideas.map((i, n) => `  ${n + 1}. ${i}`).join("\n")
      : "  (none — suggest your own)";

  return [
    "# LLM Inference Optimization Task",
    "",
    `Target: ${target_tps} tok/s generation. Current: ${bl.generation_tps.toFixed(1)} tok/s (${((bl.generation_tps / target_tps) * 100).toFixed(0)}% of target).`,
    "",
    `You have SSH access to root@${LLM_HOST} (port ${LLM_SSH_PORT}) where llama.cpp is installed at ${LLAMA_CPP_DIR}.`,
    "",
    "## Hardware",
    "- GPU: AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32GB VRAM, 576 GB/s bandwidth)",
    "- CPU: AMD Ryzen 7 9800X3D (8c/16t Zen5, 96MB L3)",
    "- RAM: 92GB DDR5",
    "- Vulkan: Mesa 25.2.8 RADV, VK_KHR_cooperative_matrix 16x16x16",
    "",
    "## Current Setup",
    "- llama.cpp built: -DGGML_VULKAN=ON, -O3 -march=znver4",
    "- Model: Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf (MoE 35B/3B active, 22GB)",
    "- Server flags: -ngl 99 --device Vulkan0 --parallel 4 -c 32768 -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 --mlock",
    "- Env: RADV_PERFTEST=coop_matrix",
    "- Service: /etc/systemd/system/llama-server.service",
    "- systemctl daemon-reload && systemctl restart llama-server to apply changes",
    "",
    "## Current Performance",
    `- Generation: ${bl.generation_tps.toFixed(1)} tok/s`,
    `- Prefill: ${bl.prefill_tps.toFixed(1)} tok/s`,
    `- Cognitive: ${bl.cognitive_score}/${bl.cognitive_total}`,
    "",
    "## Optimization History (last 15 cycles)",
    historyBlock,
    "",
    "## FAILED APPROACHES — DO NOT REPEAT THESE",
    failedBlock,
    "",
    "## Ideas Queue (pick one, or propose your own)",
    ideasBlock,
    "",
    "## Key Source Files",
    "- Vulkan backend: /root/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp",
    "- Vulkan shaders: /root/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/",
    "  - mul_mm.comp, mul_mm_cm2.comp (matmul — prefill bottleneck)",
    "  - dequant_*.comp, mul_mat_vec*.comp (DMMV — generation bottleneck)",
    "  - flash_attn_cm1.comp (flash attention with coopmat)",
    "- CMake: /root/llama.cpp/ggml/src/ggml-vulkan/CMakeLists.txt",
    "",
    "## Theory",
    "- Generation is memory-bandwidth bound: 576 GB/s / ~1.5GB MoE = ~384 tok/s theoretical",
    "- Current ~100 tok/s ≈ 26% bandwidth utilization — big room for improvement",
    "- DMMV shaders (dequant + dot product per row) are the generation bottleneck",
    "- RDNA4 has 64 CUs, wave64, 32KB L1/CU, 6MB L2, VK_KHR_cooperative_matrix 16x16x16",
    "- The Vulkan backend treats RDNA4 as RDNA3 — no RDNA4-specific tuning exists yet",
    "",
    "## Rules",
    "1. Make ONE focused optimization. Do not try multiple things.",
    "2. After changes: cd /root/llama.cpp && cmake --build build --config Release -j8",
    "3. Then: systemctl daemon-reload && systemctl restart llama-server",
    "4. Wait ~15s after restart for model to load.",
    "5. Do NOT change the model, quantization, or context size.",
    "6. Cognitive quality must NOT degrade (≥" +
      `${Math.max(0, bl.cognitive_score - 1)}/${bl.cognitive_total}).`,
    "7. You CAN: modify source, shaders, build flags, server flags, Mesa env vars, system tuning.",
    "",
    "## Your Task",
    "1. Analyze the history above. Understand what worked, what failed, and why.",
    "2. Pick the MOST PROMISING approach you haven't tried yet (from Ideas Queue or your own).",
    "3. Implement it. Build. Restart server. Wait for load.",
    "4. Do NOT try anything from the FAILED APPROACHES list — those already didn't work.",
    "",
    "## Output Format",
    "At the very end, after all changes and restart, print exactly these 3 lines (with the @@@ prefixes):",
    "@@@DESCRIPTION: <one-line summary of what you changed>",
    "@@@SELF_ANALYSIS: <why you chose this approach and what you expect it to achieve>",
    "@@@NEXT_IDEAS: <comma-separated list of NEW ideas to try in future cycles, based on what you learned>",
    "",
    "IMPORTANT: These @@@ lines must appear as the LAST thing you output. Do not output anything after them.",
  ].join("\n");
}

// ── Cycle runner ─────────────────────────────────────────────────────

async function runCycle(
  runDir: string,
  state: RunState,
  agent: AgentKind,
  model?: string,
): Promise<CycleResult> {
  const cycleNum = state.cycles.length + 1;
  const cycleDir = join(runDir, `cycle-${String(cycleNum).padStart(3, "0")}`);
  await mkdir(cycleDir, { recursive: true });

  const baseline = state.currentBest;
  const prompt = buildOptimizationPrompt(state);
  await writeFile(join(cycleDir, "prompt.md"), prompt);

  console.log(clr("1;35", "\n" + "═".repeat(64)));
  console.log(
    clr(
      "1;35",
      `  CYCLE ${cycleNum} — gen:${baseline.generation_tps.toFixed(1)} → target:${state.target_tps} tok/s`,
    ),
  );
  console.log(clr("1;35", "═".repeat(64)));

  // Pre-check: verify server is reachable before spawning expensive agent
  console.log(clr("2", "\n  Checking server connectivity..."));
  try {
    await ssh("echo ok", 15_000);
    console.log(clr("1;32", "  Server reachable."));
  } catch {
    console.log(clr("1;31", "  ❌ Server unreachable — skipping cycle to save API costs."));
    const cycleResult: CycleResult = {
      cycle: cycleNum,
      timestamp: new Date().toISOString(),
      baseline,
      optimized: null,
      gen_delta_pct: 0,
      prefill_delta_pct: 0,
      kept: false,
      description: "Skipped — server unreachable",
      selfAnalysis: "",
      nextIdeas: [],
      error: "Server unreachable (pre-check failed)",
    };
    await writeFile(join(cycleDir, "result.json"), JSON.stringify(cycleResult, null, 2));
    return cycleResult;
  }

  // Snapshot llama.cpp before changes — use hard reset instead of stash to avoid stash accumulation
  // Just record the current HEAD so we can reset to it on revert
  let preCommit: string;
  try {
    preCommit = await ssh(`cd ${LLAMA_CPP_DIR} && git add -A 2>/dev/null; git stash drop 2>/dev/null; git rev-parse HEAD`);
  } catch {
    preCommit = "HEAD";
  }

  // Run agent
  const result = await runAgent(agent, prompt, model);
  await writeFile(join(cycleDir, "agent_stdout.txt"), result.stdout);
  await writeFile(join(cycleDir, "agent_stderr.txt"), result.stderr);

  // Extract description, self-analysis, and ideas from agent output.
  // Strategy: search text_delta first, then thinking_delta, then infer from SSH commands.
  // Agents rarely print DESCRIPTION: markers in their text output — they get consumed by tool calls.
  const assembledText = extractTextFromStreamJson(result.stdout);
  const thinkingText = extractThinkingFromStreamJson(result.stdout);

  // Search both text and thinking (last 3000 chars of each) for output markers
  const searchTexts = [assembledText.slice(-3000), thinkingText.slice(-3000)];
  let descMatch: RegExpMatchArray | null = null;
  let analysisMatch: RegExpMatchArray | null = null;
  let ideasMatch: RegExpMatchArray | null = null;
  for (const text of searchTexts) {
    if (!descMatch) descMatch = text.match(/^@@@DESCRIPTION:\s*(.+)/im) ?? text.match(/^DESCRIPTION:\s*(.+)/im);
    if (!analysisMatch) analysisMatch = text.match(/^@@@SELF_ANALYSIS:\s*(.+)/im) ?? text.match(/^SELF_ANALYSIS:\s*(.+)/im);
    if (!ideasMatch) ideasMatch = text.match(/^@@@NEXT_IDEAS:\s*(.+)/im) ?? text.match(/^NEXT_IDEAS:\s*(.+)/im);
  }

  const rawDesc = descMatch?.[1]?.trim() ?? "";
  let description = rawDesc && !isGarbageString(rawDesc) ? rawDesc : "";

  // Fallback: if no markers found, summarize from SSH commands
  if (!description) {
    const sshCmds = extractSSHCommands(result.stdout);
    description = summarizeFromSSHCommands(sshCmds);
  }

  const rawAnalysis = analysisMatch?.[1]?.trim() ?? "";
  const selfAnalysis = rawAnalysis && !isGarbageString(rawAnalysis) ? rawAnalysis : "";
  const newIdeas = (ideasMatch?.[1]
    ?.split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 3 && s.length < 120 && !isGarbageString(s)) ?? []);

  // Wait for server
  console.log(clr("2", "\n  Waiting for server to be ready..."));
  let serverReady = false;
  for (let i = 0; i < 12; i++) {
    await new Promise((r) => setTimeout(r, 5_000));
    try {
      const resp = await fetch(`http://${LLM_HOST}:${LLM_PORT}/health`, {
        signal: AbortSignal.timeout(5_000),
      });
      if (resp.ok) {
        serverReady = true;
        break;
      }
    } catch {}
  }

  if (!serverReady) {
    console.log(clr("1;31", "  ❌ Server not ready after 60s. Reverting."));
    await ssh(
      `cd ${LLAMA_CPP_DIR} && git reset --hard ${preCommit} 2>/dev/null; git clean -fd 2>/dev/null; cmake --build build --config Release -j8 2>&1 | tail -3; systemctl restart llama-server`,
      300_000,
    ).catch(() => {});
    await new Promise((r) => setTimeout(r, 25_000));
    const cycleResult: CycleResult = {
      cycle: cycleNum,
      timestamp: new Date().toISOString(),
      baseline,
      optimized: null,
      gen_delta_pct: 0,
      prefill_delta_pct: 0,
      kept: false,
      description,
      selfAnalysis,
      nextIdeas: newIdeas,
      error: "Server failed to start",
    };
    await writeFile(join(cycleDir, "result.json"), JSON.stringify(cycleResult, null, 2));
    return cycleResult;
  }

  // Benchmark
  console.log(clr("1;33", "\n  📊 Running post-optimization benchmark..."));
  let optimized: BenchResult;
  try {
    optimized = await runBenchmark();
  } catch (e) {
    console.log(clr("1;31", `  ❌ Benchmark failed: ${e}`));
    await ssh(
      `cd ${LLAMA_CPP_DIR} && git reset --hard ${preCommit} 2>/dev/null; git clean -fd 2>/dev/null; cmake --build build --config Release -j8 2>&1 | tail -3; systemctl restart llama-server`,
      300_000,
    ).catch(() => {});
    await new Promise((r) => setTimeout(r, 25_000));
    const cycleResult: CycleResult = {
      cycle: cycleNum,
      timestamp: new Date().toISOString(),
      baseline,
      optimized: null,
      gen_delta_pct: 0,
      prefill_delta_pct: 0,
      kept: false,
      description,
      selfAnalysis,
      nextIdeas: newIdeas,
      error: String(e),
    };
    await writeFile(join(cycleDir, "result.json"), JSON.stringify(cycleResult, null, 2));
    return cycleResult;
  }

  printBench("Post-Optimization", optimized);

  const gen_delta =
    ((optimized.generation_tps - baseline.generation_tps) /
      baseline.generation_tps) *
    100;
  const prefill_delta =
    ((optimized.prefill_tps - baseline.prefill_tps) / baseline.prefill_tps) *
    100;
  const cognitive_ok =
    optimized.cognitive_score >= baseline.cognitive_score - 1;

  // Keep if generation improved > 0.5% and cognitive didn't tank
  const keep = gen_delta > 0.5 && cognitive_ok;

  console.log(clr("1;33", SEP));
  console.log(
    clr(
      "1;33",
      `  ${description}`,
    ),
  );
  console.log(
    clr(
      "1;33",
      `  gen: ${baseline.generation_tps.toFixed(1)} → ${optimized.generation_tps.toFixed(1)} tok/s (${gen_delta > 0 ? "+" : ""}${gen_delta.toFixed(1)}%)`,
    ),
  );
  console.log(
    clr(
      "1;33",
      `  prefill: ${baseline.prefill_tps.toFixed(0)} → ${optimized.prefill_tps.toFixed(0)} tok/s (${prefill_delta > 0 ? "+" : ""}${prefill_delta.toFixed(1)}%)`,
    ),
  );
  console.log(
    clr(keep ? "1;32" : "1;31", `  → ${keep ? "✅ KEEPING" : "❌ REVERTING"}`),
  );
  console.log(clr("1;33", SEP));

  if (!keep) {
    console.log(clr("2", "  Reverting changes..."));
    await ssh(
      `cd ${LLAMA_CPP_DIR} && git reset --hard ${preCommit} 2>/dev/null; git clean -fd 2>/dev/null; cmake --build build --config Release -j8 2>&1 | tail -3; systemctl restart llama-server`,
      300_000,
    ).catch(() => {});
    await new Promise((r) => setTimeout(r, 25_000));
  } else {
    // Commit the successful change
    await ssh(
      `cd ${LLAMA_CPP_DIR} && git add -A && git commit -m "optimize: ${description}" 2>/dev/null || true`,
    ).catch(() => {});
  }

  const cycleResult: CycleResult = {
    cycle: cycleNum,
    timestamp: new Date().toISOString(),
    baseline,
    optimized,
    gen_delta_pct: gen_delta,
    prefill_delta_pct: prefill_delta,
    kept: keep,
    description,
    selfAnalysis,
    nextIdeas: newIdeas,
  };
  await writeFile(
    join(cycleDir, "result.json"),
    JSON.stringify(cycleResult, null, 2),
  );
  return cycleResult;
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  const rawArgs = process.argv.slice(2);
  let maxCycles = Infinity;
  let agent: AgentKind = "claude";
  let model: string | undefined;
  let targetTps = 150;
  let dryRun = false;
  let resumeDir: string | undefined;

  for (let i = 0; i < rawArgs.length; i++) {
    switch (rawArgs[i]) {
      case "--agent": {
        const val = rawArgs[++i];
        if (val !== "claude" && val !== "codex") {
          console.error(`Invalid --agent: ${val}. Use claude or codex.`);
          process.exit(1);
        }
        agent = val;
        break;
      }
      case "--model":
        model = rawArgs[++i];
        break;
      case "--cycles":
        maxCycles = parseInt(rawArgs[++i], 10);
        break;
      case "--target-tps":
        targetTps = parseInt(rawArgs[++i], 10);
        break;
      case "--dry-run":
        dryRun = true;
        break;
      case "--resume":
        resumeDir = rawArgs[++i];
        break;
      case "--help":
        console.log(
          [
            "Usage: bun scripts/optimize_llm_tps.ts --agent <claude|codex> [options]",
            "",
            "Options:",
            "  --agent <claude|codex>  Agent to use (required)",
            "  --model <name>          Model override for agent",
            "  --cycles N              Max cycles (default: infinite)",
            "  --target-tps N          Generation target tok/s (default: 150)",
            "  --dry-run               Benchmark only, no optimization",
            "  --resume <dir>          Resume a previous run from its directory",
          ].join("\n"),
        );
        process.exit(0);
    }
  }

  // Set up run directory
  const runId = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const runDir = resumeDir ?? join(RESULTS_DIR, runId);
  await mkdir(runDir, { recursive: true });

  const agentLabel = agent === "codex" ? "Codex" : "Claude";

  console.log(clr("1;35", "═".repeat(64)));
  console.log(clr("1;35", "  LLM TPS OPTIMIZATION LOOP"));
  console.log(clr("1;35", "═".repeat(64)));
  console.log(`  Agent:    ${clr("1", agentLabel)}${model ? ` (${model})` : ""}`);
  console.log(`  Target:   ${clr("1", `${targetTps} tok/s generation`)}`);
  console.log(`  Cycles:   ${maxCycles === Infinity ? "infinite" : maxCycles}`);
  console.log(`  Results:  ${runDir}`);
  console.log(clr("1;35", "═".repeat(64)));

  // Load or create state
  let state = await loadState(runDir);
  if (!state) {
    console.log(clr("1;33", "\n  📊 Running initial benchmark...\n"));
    let baseline: BenchResult;
    try {
      baseline = await runBenchmark();
    } catch (e) {
      console.error(`Failed to connect to LLM server: ${e}`);
      process.exit(1);
    }
    printBench("BASELINE", baseline);

    state = {
      runId,
      target_tps: targetTps,
      originalBaseline: baseline,
      currentBest: baseline,
      cycles: [],
      failedApproaches: [],
      ideas: [
        "Tune DMMV shader workgroup sizes for RDNA4 wave64",
        "Try f16 KV cache instead of q8_0 (remove dequant overhead in attention)",
        "Add RDNA4-specific warptile tuning path in ggml-vulkan.cpp",
        "Increase mul_mat_vec_max_cols for wider batch decode",
        "Try RADV_DEBUG=nothreadllvm or other RADV env vars",
        "Tune flash attention block sizes for 64KB shared memory",
        "Use HugeTLB pages for model memory",
        "Try CPU affinity pinning for server threads",
        "Check if Mesa 25.3 PPA has further RDNA4 patches",
        "Profile with VK_LAYER_MESA_OVERLAY to find shader bottlenecks",
        "Modify dequant shaders to use wider loads (uvec4 instead of uint)",
        "Try different batch/ubatch sizes",
      ],
    };
    await saveState(runDir, state);
  } else {
    console.log(
      clr(
        "1;33",
        `\n  Resuming from cycle ${state.cycles.length} (gen: ${state.currentBest.generation_tps.toFixed(1)} tok/s)\n`,
      ),
    );
    state.target_tps = targetTps;
    if (!state.failedApproaches) state.failedApproaches = [];

    // Sanitize state on resume — purge garbage from previous runs using shared isGarbageString
    const prevFailed = state.failedApproaches.length;
    const prevIdeas = state.ideas.length;
    state.failedApproaches = state.failedApproaches.filter((f) => !isGarbageString(f));
    state.ideas = state.ideas.filter((i) => !isGarbageString(i) && i.length < 120);
    for (const c of state.cycles) {
      if (isGarbageString(c.description)) c.description = "Unknown optimization";
      if (c.selfAnalysis && isGarbageString(c.selfAnalysis)) c.selfAnalysis = "";
      c.nextIdeas = c.nextIdeas?.filter((i) => !isGarbageString(i) && i.length < 120) ?? [];
    }
    if (prevFailed !== state.failedApproaches.length || prevIdeas !== state.ideas.length) {
      console.log(clr("33", `  Sanitized state: failedApproaches ${prevFailed}→${state.failedApproaches.length}, ideas ${prevIdeas}→${state.ideas.length}`));
      await saveState(runDir, state);
    }
  }

  if (dryRun) {
    console.log("\n  Dry run — stopping after benchmark.");
    return;
  }

  // Pre-loop: verify SSH connectivity to avoid burning cycles
  try {
    await ssh("echo ok", 10_000);
    console.log(clr("32", "  SSH connectivity: OK"));
  } catch {
    console.error(clr("31", "\n  ❌ Cannot reach server via SSH. Is it online?"));
    process.exit(1);
  }

  // Main loop
  let cyclesDone = 0;
  let consecutiveSSHFailures = 0;
  while (cyclesDone < maxCycles) {
    if (state.currentBest.generation_tps >= targetTps) {
      console.log(
        clr(
          "1;32",
          `\n  🎯 TARGET REACHED: ${state.currentBest.generation_tps.toFixed(1)} ≥ ${targetTps} tok/s!`,
        ),
      );
      break;
    }

    // Check SSH before each cycle to avoid burning API credits when server is down
    try {
      await ssh("echo ok", 15_000);
      consecutiveSSHFailures = 0;
    } catch {
      consecutiveSSHFailures++;
      if (consecutiveSSHFailures >= 3) {
        console.error(clr("31", `\n  ❌ SSH unreachable for ${consecutiveSSHFailures} consecutive checks. Waiting 5 minutes...`));
        await new Promise((r) => setTimeout(r, 300_000));
        continue;
      }
      console.log(clr("33", `  SSH check failed (${consecutiveSSHFailures}/3), retrying in 60s...`));
      await new Promise((r) => setTimeout(r, 60_000));
      continue;
    }

    const cycleResult = await runCycle(runDir, state, agent, model);
    state.cycles.push(cycleResult);

    if (cycleResult.kept && cycleResult.optimized) {
      state.currentBest = cycleResult.optimized;
    } else if (!cycleResult.kept && cycleResult.description !== "Unknown optimization" && !cycleResult.description.startsWith("Skipped")) {
      // Track failed approaches — only if description is clean (not skipped/error cycles)
      const desc = cycleResult.description.slice(0, 100);
      if (!isGarbageString(desc)) {
        const failEntry = `${desc} (gen: ${cycleResult.optimized?.generation_tps.toFixed(1) ?? "ERR"} tok/s, ${cycleResult.gen_delta_pct > 0 ? "+" : ""}${cycleResult.gen_delta_pct.toFixed(1)}%)`;
        state.failedApproaches.push(failEntry);
      }
      // Cap at 30 most recent failed approaches
      if (state.failedApproaches.length > 30) {
        state.failedApproaches = state.failedApproaches.slice(-30);
      }
    }

    // Merge new ideas (deduplicate, skip garbage, skip if already failed)
    const existingLower = new Set(state.ideas.map((i) => i.toLowerCase()));
    const failedLower = state.failedApproaches.map((f) => f.toLowerCase());
    for (const idea of cycleResult.nextIdeas) {
      const il = idea.toLowerCase();
      if (isGarbageString(idea) || idea.length > 120) continue;
      if (!existingLower.has(il) && !failedLower.some((f) => f.includes(il.slice(0, 20)))) {
        state.ideas.push(idea);
        existingLower.add(il);
      }
    }

    // Remove ideas that match what was just tried
    const triedLower = cycleResult.description.toLowerCase();
    if (triedLower !== "unknown optimization") {
      state.ideas = state.ideas.filter(
        (idea) => !triedLower.includes(idea.toLowerCase().slice(0, 15)),
      );
    }

    // Cap ideas at 50
    if (state.ideas.length > 50) {
      state.ideas = state.ideas.slice(0, 50);
    }

    // Cap cycles — keep all KEPT cycles + last 50 to prevent state bloat
    if (state.cycles.length > 60) {
      const kept = state.cycles.filter((c) => c.kept);
      const recent = state.cycles.slice(-50);
      const seen = new Set<number>();
      const merged: CycleResult[] = [];
      for (const c of [...kept, ...recent]) {
        if (!seen.has(c.cycle)) {
          seen.add(c.cycle);
          merged.push(c);
        }
      }
      state.cycles = merged.sort((a, b) => a.cycle - b.cycle);
    }

    await saveState(runDir, state);

    // Print running totals
    const origGen = state.originalBaseline.generation_tps;
    const currGen = state.currentBest.generation_tps;
    const totalPct = ((currGen - origGen) / origGen) * 100;
    const kept = state.cycles.filter((c) => c.kept).length;

    console.log(clr("1;35", "\n" + "═".repeat(64)));
    console.log(
      clr(
        "1;35",
        `  AFTER ${state.cycles.length} CYCLES:`,
      ),
    );
    console.log(
      clr(
        "1;35",
        `  Generation: ${origGen.toFixed(1)} → ${currGen.toFixed(1)} tok/s (${totalPct > 0 ? "+" : ""}${totalPct.toFixed(1)}%)  target: ${targetTps}`,
      ),
    );
    console.log(
      clr("1;35", `  Kept: ${kept}/${state.cycles.length} optimizations`),
    );
    console.log(
      clr("1;35", `  Ideas queued: ${state.ideas.length}`),
    );
    console.log(clr("1;35", "═".repeat(64)));

    cyclesDone++;

    // Brief cooldown
    await new Promise((r) => setTimeout(r, 3_000));
  }

  console.log(clr("1;32", "\n" + "═".repeat(64)));
  console.log(clr("1;32", "  OPTIMIZATION COMPLETE"));
  console.log(
    clr(
      "1;32",
      `  ${state.originalBaseline.generation_tps.toFixed(1)} → ${state.currentBest.generation_tps.toFixed(1)} tok/s`,
    ),
  );
  console.log(clr("1;32", `  Results: ${runDir}`));
  console.log(clr("1;32", "═".repeat(64)));
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(clr("31", `\nFatal error: ${err.message ?? err}`));
    process.exit(1);
  });
}
