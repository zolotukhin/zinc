import { describe, expect, test } from "bun:test";
import {
  formatCodexStreamLine,
  formatToolInput,
  formatClaudeStreamLine,
  loadPreviousRun,
  type ClaudeStreamState,
} from "./optimize_perf";

// -- Codex stream formatter ---------------------------------------------------

describe("formatCodexStreamLine", () => {
  test("formats shell command", () => {
    const line = JSON.stringify({ type: "action", name: "shell", command: "zig build 2>&1" });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("shell");
    expect(out).toContain("zig build");
  });

  test("formats file write", () => {
    const line = JSON.stringify({ type: "action", name: "write", input: { file_path: "/src/foo.zig" } });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("write");
    expect(out).toContain("foo.zig");
  });

  test("formats file read", () => {
    const line = JSON.stringify({ type: "action", name: "read", input: { file_path: "/a/b/c.zig" } });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("read");
    expect(out).toContain("c.zig");
  });

  test("formats agent message", () => {
    const line = JSON.stringify({ type: "message", content: "I will now edit the file." });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("I will now edit the file.");
  });

  test("skips tool output", () => {
    const line = JSON.stringify({ type: "function_call_output", output: "lots of text..." });
    expect(formatCodexStreamLine(line)).toBeNull();
  });

  test("skips empty lines", () => {
    expect(formatCodexStreamLine("")).toBeNull();
    expect(formatCodexStreamLine("   ")).toBeNull();
  });

  test("returns null for non-JSON", () => {
    expect(formatCodexStreamLine("not json at all")).toBeNull();
  });

  test("shows thinking indicator", () => {
    const line = JSON.stringify({ type: "thinking" });
    const out = formatCodexStreamLine(line);
    expect(out).toContain("thinking");
  });
});

// -- Claude stream formatter --------------------------------------------------

describe("formatClaudeStreamLine", () => {
  function freshState(): ClaudeStreamState {
    return {
      currentToolName: null,
      currentBlockIsToolUse: false,
      inputJsonBuffer: "",
      inTextBlock: false,
      sawTextDeltaInCurrentMessage: false,
    };
  }

  test("shows tool name on content_block_start", () => {
    const state = freshState();
    const line = JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_start", content_block: { type: "tool_use", name: "bash" } },
    });
    const out = formatClaudeStreamLine(line, state);
    expect(out).toContain("bash");
    expect(state.currentToolName).toBe("bash");
    expect(state.currentBlockIsToolUse).toBe(true);
  });

  test("streams text delta", () => {
    const state = freshState();
    // Start text block
    formatClaudeStreamLine(JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_start", content_block: { type: "text" } },
    }), state);

    const out = formatClaudeStreamLine(JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_delta", delta: { type: "text_delta", text: "hello world" } },
    }), state);
    expect(out).toBe("hello world");
  });

  test("accumulates input_json_delta silently", () => {
    const state = freshState();
    // Start tool block
    formatClaudeStreamLine(JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_start", content_block: { type: "tool_use", name: "edit" } },
    }), state);

    const out = formatClaudeStreamLine(JSON.stringify({
      type: "stream_event",
      event: { type: "content_block_delta", delta: { type: "input_json_delta", partial_json: '{"file' } },
    }), state);
    expect(out).toBeNull();
    expect(state.inputJsonBuffer).toBe('{"file');
  });

  test("returns null for empty line", () => {
    expect(formatClaudeStreamLine("", freshState())).toBeNull();
  });

  test("returns raw line for non-JSON", () => {
    const out = formatClaudeStreamLine("some random text", freshState());
    expect(out).toBe("some random text\n");
  });
});

// -- formatToolInput ----------------------------------------------------------

describe("formatToolInput", () => {
  test("formats bash command", () => {
    const out = formatToolInput("bash", JSON.stringify({ command: "ls -la" }));
    expect(out).toContain("ls -la");
  });

  test("formats edit with file path", () => {
    const out = formatToolInput("edit", JSON.stringify({ file_path: "/Users/me/project/src/main.zig" }));
    expect(out).toContain("src/main.zig");
  });

  test("formats write with line count", () => {
    const out = formatToolInput("write", JSON.stringify({ file_path: "/a/b.zig", content: "line1\nline2\nline3" }));
    expect(out).toContain("b.zig");
    expect(out).toContain("3 lines");
  });

  test("formats read with short path", () => {
    const out = formatToolInput("read", JSON.stringify({ file_path: "/long/path/to/file.zig" }));
    expect(out).toContain("file.zig");
  });

  test("formats grep with pattern", () => {
    const out = formatToolInput("grep", JSON.stringify({ pattern: "computeBarrier" }));
    expect(out).toContain("/computeBarrier/");
  });

  test("returns empty for unknown tool", () => {
    expect(formatToolInput("unknown_tool", "{}")).toBe("");
  });
});

// -- loadPreviousRun ----------------------------------------------------------

describe("loadPreviousRun", () => {
  test("returns empty state for nonexistent effort", async () => {
    // effort 99 won't have a log file
    const result = await loadPreviousRun(99);
    expect(result.history).toBe("");
    expect(result.bestTokPerSec).toBe(0);
    expect(result.lastCycle).toBe(0);
  });
});

// -- Config sanity ------------------------------------------------------------

describe("config", () => {
  test("env vars are used for host config", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("ZINC_HOST");
    expect(src).toContain("ZINC_PORT");
    expect(src).toContain("ZINC_USER");
  });

  test("coherence checks include multiple prompts", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("capital of France");
    expect(src).toContain("2+2");
    expect(src).toContain("Mercury");
  });

  test("all three models are listed for coherence", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("Qwen3.5-35B");
    expect(src).toContain("Qwen3.5-2B");
    expect(src).toContain("Gemma3-12B");
  });

  test("codex uses exec with sandbox bypass and json", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain('"exec"');
    expect(src).toContain("dangerously-bypass-approvals-and-sandbox");
    expect(src).toContain('"--json"');
  });

  test("blocked ops prevent agent from git push/commit", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_perf.ts").text();
    expect(src).toContain("git push");
    expect(src).toContain("git commit");
  });
});
