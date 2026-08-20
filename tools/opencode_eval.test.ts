import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import {
  buildPrompt,
  buildOpenCodeArgs,
  fixturesForOption,
  parseArgs,
  parseOpenCodeJsonl,
  parsePortListenPids,
  parseTestSummary,
  readOnlyViolations,
  renderSummary,
  runCommand,
  snapshotReadOnlyFiles,
  summarizeTraceFiles,
  writeFixtureProject,
} from "./opencode_eval.mjs";
import { FIXTURES, fixtureById } from "./opencode_eval_fixtures.mjs";

async function tempDir(prefix = "zinc-opencode-eval-test-") {
  return await mkdtemp(path.join(os.tmpdir(), prefix));
}

describe("opencode eval args", () => {
  test("parses provider, fixtures, proxy, and model options", () => {
    const opts = parseArgs([
      "--provider",
      "both",
      "--fixtures",
      "rate-limiter-single,cart-multi-file",
      "--manage-proxy",
      "--replace-proxy",
      "--model",
      "zinc/custom",
      "--timeout-ms",
      "1234",
    ]);

    expect(opts.provider).toBe("both");
    expect(opts.fixtureIds).toEqual(["rate-limiter-single", "cart-multi-file"]);
    expect(opts.manageProxy).toBe(true);
    expect(opts.replaceProxy).toBe(true);
    expect(opts.model).toBe("zinc/custom");
    expect(opts.timeoutMs).toBe(1234);
  });

  test("selects fixtures by id", () => {
    expect(fixturesForOption(["rate-limiter-single"]).map((fixture) => fixture.id)).toEqual([
      "rate-limiter-single",
    ]);
    expect(fixturesForOption(["all"]).length).toBe(FIXTURES.length);
  });

  test("parses listener pids from lsof output", () => {
    expect(parsePortListenPids("61961\n75393\nnot-a-pid\n")).toEqual(["61961", "75393"]);
  });
});

describe("fixture projects", () => {
  test("writes a project and builds a prompt with exact paths", async () => {
    const fixture = fixtureById("rate-limiter-single")!;
    const dir = await tempDir();
    await writeFixtureProject(fixture, dir);

    const source = await readFile(path.join(dir, "src/rate_limiter.mjs"), "utf8");
    expect(source).toContain("SlidingWindowRateLimiter");

    const prompt = buildPrompt(fixture, dir);
    expect(prompt).toContain(path.join(dir, "package.json"));
    expect(prompt).toContain("Change only source files");
    expect(prompt).toContain("npm test 2>&1");
  });

  test("includes baseline failing test output when supplied", async () => {
    const fixture = fixtureById("multi-run-duration")!;
    const dir = await tempDir();
    const prompt = buildPrompt(fixture, dir, "1 pass\n2 fail\nExpected: 180000\nReceived: 3000");

    expect(prompt).toContain("Initial failing test output");
    expect(prompt).toContain("Expected: 180000");
    expect(prompt).toContain("do not spend turns rereading files already shown");
  });

  test("builds OpenCode args with fixture directory as workspace", async () => {
    const fixture = fixtureById("rate-limiter-single")!;
    const dir = await tempDir();
    const args = buildOpenCodeArgs("zinc", fixture, dir, {
      model: "zinc/qwen",
    } as any);

    expect(args).toContain("--dir");
    expect(args[args.indexOf("--dir") + 1]).toBe(dir);
    expect(args).toContain("zinc-rate-limiter-single");
  });

  test("all seeded fixtures start with failing tests", async () => {
    for (const fixture of FIXTURES) {
      const dir = await tempDir(`zinc-opencode-${fixture.id}-`);
      await writeFixtureProject(fixture, dir);
      const result = await runCommand(fixture.testCommand, { cwd: dir, timeoutMs: 30000 });
      const summary = parseTestSummary(result.output);

      expect(result.exitCode).not.toBe(0);
      expect(summary.fail ?? 0).toBeGreaterThan(0);
    }
  }, 60_000);

  test("detects read-only file violations", async () => {
    const fixture = fixtureById("readonly-test-temptation")!;
    const dir = await tempDir();
    await writeFixtureProject(fixture, dir);
    const before = await snapshotReadOnlyFiles(dir, fixture);

    await writeFile(path.join(dir, "test/slugify.test.mjs"), "changed\n");

    expect(await readOnlyViolations(dir, before)).toEqual(["test/slugify.test.mjs"]);
  });
});

describe("output parsing", () => {
  test("parses OpenCode JSONL tool usage", () => {
    const jsonl = [
      JSON.stringify({
        type: "tool_use",
        part: {
          type: "tool",
          tool: "bash",
          state: {
            status: "completed",
            input: { command: "npm test 2>&1", description: "Run tests" },
            metadata: { exit: 1 },
          },
        },
      }),
      JSON.stringify({
        type: "tool_use",
        part: {
          type: "tool",
          tool: "write",
          state: { status: "completed", input: { filePath: "/tmp/project/src/app.mjs" } },
        },
      }),
      "{bad json",
    ].join("\n");

    const summary = parseOpenCodeJsonl(jsonl);

    expect(summary.toolCalls).toBe(2);
    expect(summary.toolCounts).toEqual({ bash: 1, write: 1 });
    expect(summary.bashCommands[0].command).toBe("npm test 2>&1");
    expect(summary.fileWrites[0].filePath).toBe("/tmp/project/src/app.mjs");
    expect(summary.malformedJsonLines).toBe(1);
  });

  test("flags malformed package-test redirection that escaped proxy repair", () => {
    const summary = parseOpenCodeJsonl(
      `${JSON.stringify({
        type: "tool_use",
        part: {
          type: "tool",
          tool: "bash",
          state: { status: "completed", input: { command: "npm test2>&1" }, metadata: { exit: 127 } },
        },
      })}\n`,
    );

    expect(summary.malformedCommands).toEqual(["npm test2>&1"]);
  });

  test("aggregates proxy traces", async () => {
    const dir = await tempDir();
    await mkdir(dir, { recursive: true });
    const a = path.join(dir, "a.json");
    const b = path.join(dir, "b.json");
    await writeFile(
      a,
      JSON.stringify({
        upstream: "http://127.0.0.1:9090/v1/chat/completions",
        shortcut: "prefetch_reads",
        response_metrics: { repaired_events: 2, suppressed_content_events: 3, suppressed_content_chars: 12, bytes: 100 },
      }),
    );
    await writeFile(
      b,
      JSON.stringify({
        upstream: "http://127.0.0.1:9090/v1/chat/completions",
        response_metrics: { repaired_events: 1, bytes: 25 },
      }),
    );

    const summary = await summarizeTraceFiles([a, b]);

    expect(summary.files).toBe(2);
    expect(summary.repairedEvents).toBe(3);
    expect(summary.suppressedContentEvents).toBe(3);
    expect(summary.suppressedContentChars).toBe(12);
    expect(summary.bytes).toBe(125);
    expect(summary.shortcuts).toEqual({ prefetch_reads: 1 });
  });

  test("renders compact markdown summary", () => {
    const out = renderSummary([
      {
        provider: "zinc",
        fixture: "rate-limiter-single",
        success: true,
        projectDir: "/tmp/project",
        finalTest: { tests: { pass: 3, fail: 0 } },
        opencode: { durationMs: 1500, summary: { toolCounts: { bash: 2, write: 1 } } },
        traces: { repairedEvents: 4, suppressedContentEvents: 2 },
      },
    ] as any);

    expect(out).toContain("| zinc | rate-limiter-single | pass | 3 pass / 0 fail | bash:2 write:1 | 4 | 2 | 1.5s |");
  });
});
