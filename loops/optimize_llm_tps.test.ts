import { describe, expect, test } from "bun:test";
import {
  isGarbageString,
  formatElapsed,
  coerceDisplayText,
  extractTextFromStreamJson,
  summarizeFromSSHCommands,
} from "./optimize_llm_tps";

describe("isGarbageString", () => {
  test("rejects empty/short strings", () => {
    expect(isGarbageString("")).toBe(true);
    expect(isGarbageString("hi")).toBe(true);
  });

  test("rejects strings over 200 chars", () => {
    expect(isGarbageString("a ".repeat(120))).toBe(true);
  });

  test("rejects HTML/XML-like strings", () => {
    expect(isGarbageString("<div>hello world</div>")).toBe(true);
  });

  test("rejects tool output artifacts", () => {
    expect(isGarbageString("some session_id found here")).toBe(true);
    expect(isGarbageString("parent_tool_use_id is set")).toBe(true);
  });

  test("rejects code-like strings", () => {
    expect(isGarbageString("console.log('hello world')")).toBe(true);
    expect(isGarbageString("await fetch(url)")).toBe(true);
    expect(isGarbageString("const x = 42;")).toBe(true);
  });

  test("accepts normal natural language", () => {
    expect(isGarbageString("Tune DMMV shader workgroup sizes for RDNA4")).toBe(false);
    expect(isGarbageString("Increased batch size from 512 to 1024")).toBe(false);
  });
});

describe("formatElapsed", () => {
  test("formats seconds", () => {
    const now = Date.now();
    expect(formatElapsed(now - 30_000)).toBe("30s");
  });

  test("formats minutes and seconds", () => {
    const now = Date.now();
    expect(formatElapsed(now - 90_000)).toBe("1m30s");
  });
});

describe("coerceDisplayText", () => {
  test("returns strings as-is", () => {
    expect(coerceDisplayText("hello")).toBe("hello");
  });

  test("returns empty string for null/undefined", () => {
    expect(coerceDisplayText(null)).toBe("");
    expect(coerceDisplayText(undefined)).toBe("");
  });

  test("stringifies numbers and booleans", () => {
    expect(coerceDisplayText(42)).toBe("42");
    expect(coerceDisplayText(true)).toBe("true");
  });

  test("extracts text from objects with known keys", () => {
    expect(coerceDisplayText({ text: "hello" })).toBe("hello");
    expect(coerceDisplayText({ message: "world" })).toBe("world");
  });
});

describe("extractTextFromStreamJson", () => {
  test("extracts text deltas from stream events", () => {
    const lines = [
      JSON.stringify({ type: "stream_event", event: { type: "content_block_delta", delta: { type: "text_delta", text: "Hello " } } }),
      JSON.stringify({ type: "stream_event", event: { type: "content_block_delta", delta: { type: "text_delta", text: "world" } } }),
    ].join("\n");
    expect(extractTextFromStreamJson(lines)).toBe("Hello world");
  });

  test("falls back to raw lines when no stream events", () => {
    expect(extractTextFromStreamJson("just plain text")).toBe("just plain text");
  });

  test("returns empty for empty input", () => {
    expect(extractTextFromStreamJson("")).toBe("");
  });
});

describe("summarizeFromSSHCommands", () => {
  test("reports no modifications for read-only commands", () => {
    const cmds = ['ssh -p 2223 root@host "cat /etc/os-release"'];
    expect(summarizeFromSSHCommands(cmds)).toContain("explored server");
  });

  test("detects modification commands", () => {
    const cmds = ['ssh -p 2223 root@host "sed -i s/old/new/ /etc/conf"'];
    expect(summarizeFromSSHCommands(cmds)).toContain("modified server");
  });

  test("returns unknown for empty array", () => {
    expect(summarizeFromSSHCommands([])).toBe("Unknown optimization");
  });
});

describe("no hardcoded private info", () => {
  test("env vars are used for host/port config", async () => {
    const src = await Bun.file(import.meta.dir + "/optimize_llm_tps.ts").text();
    // Should not contain hardcoded private IPs
    expect(src).not.toContain('"192.168.');
    expect(src).not.toContain("'192.168.");
    // SSH port and LLM port should come from env
    expect(src).toContain("process.env.LLM_HOST");
    expect(src).toContain("process.env.LLM_SSH_PORT");
    expect(src).toContain("process.env.LLM_PORT");
  });
});
