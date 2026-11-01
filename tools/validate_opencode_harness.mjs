#!/usr/bin/env bun
// CPU protocol/lifecycle checks with actual OpenCode and scripted responses.
// This does not load a model or measure coding quality or inference performance.
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import {
  buildConfig,
  openCodeRunFlags,
  parseOpenCodeVersion,
  configForOpenCode,
  runVerification,
  runVerifiedOpenCode,
  writeConfig,
} from './setup_opencode.mjs';

const exec = promisify(execFile);
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');

function parseArgs(args) {
  const opts = { opencode: process.env.OPENCODE_BIN || 'opencode', roundTimeoutMs: 15000 };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--help' || args[i] === '-h') {
      opts.help = true;
      continue;
    }
    const key = {
      '--opencode': 'opencode',
      '--output': 'output',
      '--round-timeout-ms': 'roundTimeoutMs',
      '--expect-version': 'expectedVersion',
    }[args[i]];
    if (!key || !args[i + 1] || args[i + 1].startsWith('--'))
      throw new Error('Invalid validation arguments');
    opts[key] = key === 'roundTimeoutMs' ? Number(args[++i]) : args[++i];
  }
  if (!opts.help && !opts.output) throw new Error('Pass --output with a fresh directory');
  if (
    !Number.isSafeInteger(opts.roundTimeoutMs) ||
    opts.roundTimeoutMs < 1000 ||
    opts.roundTimeoutMs > 60000
  )
    throw new Error('Round timeout must be between 1000 and 60000 ms');
  if (opts.opencode.includes('/') || opts.opencode.includes('\\'))
    opts.opencode = path.resolve(opts.opencode);
  return opts;
}

function capture() {
  const chunks = [];
  let bytes = 0;
  const limit = 2 * 1024 * 1024;
  return {
    write(chunk) {
      const data = Buffer.from(chunk);
      if (bytes < limit) chunks.push(data.subarray(0, limit - bytes));
      bytes += data.length;
      return true;
    },
    bytes: () => Buffer.concat(chunks),
    truncated: () => bytes > limit,
  };
}

function reply(body, delta, finish) {
  const base = { id: 'chatcmpl-zinc-protocol-fixture', created: 0, model: body.model };
  const usage = { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 };
  if (!body.stream) {
    const message = { role: 'assistant', ...delta };
    if (message.tool_calls)
      message.tool_calls = message.tool_calls.map(({ index, ...call }) => call);
    return Response.json({
      ...base,
      object: 'chat.completion',
      choices: [{ index: 0, message, finish_reason: finish }],
      usage,
    });
  }
  const event = (value) => `data: ${JSON.stringify(value)}\n\n`;
  return new Response(
    event({
      ...base,
      object: 'chat.completion.chunk',
      choices: [{ index: 0, delta: { role: 'assistant', ...delta }, finish_reason: null }],
    }) +
      event({
        ...base,
        object: 'chat.completion.chunk',
        choices: [{ index: 0, delta: {}, finish_reason: finish }],
        usage,
      }) +
      'data: [DONE]\n\n',
    { headers: { 'content-type': 'text/event-stream' } },
  );
}

async function fixture(name, root, opts, env) {
  const dir = path.join(root, name);
  await mkdir(dir, { mode: 0o700 });
  const project = path.join(dir, 'project');
  await mkdir(project, { mode: 0o700 });
  const config = configForOpenCode(
    buildConfig({ context: 16384, output: 1024, steps: 8 }),
    opts.runFlags,
  );
  await writeConfig(project, config);
  const configFile = path.join(project, 'opencode.zinc.json');
  const originalConfig = await readFile(configFile);
  const testFile = path.join(project, 'feature.test.mjs');
  await writeFile(
    path.join(project, 'feature.mjs'),
    `export const answer = ${name === 'api-error' ? 42 : -1};\n`,
    { mode: 0o600 },
  );
  await writeFile(
    testFile,
    `import { expect, test } from 'bun:test';
import { answer } from './feature.mjs';
test('protected feature acceptance', () => expect(answer).toBe(42));\n`,
    { mode: 0o600 },
  );
  const testHash = hash(await readFile(testFile));
  if (name === 'bounded-tools')
    await writeFile(
      path.join(project, 'large-source.txt'),
      Array.from({ length: 600 }, (_, i) => `READ_LINE_${String(i).padStart(4, '0')}`).join('\n'),
      { mode: 0o600 },
    );
  const stdout = capture(),
    stderr = capture();
  const sessions = new Set(),
    checks = [],
    controllers = new Set(),
    heartbeats = new Set();
  let firstWrite = false,
    repaired = false,
    stalled = 0,
    requests = 0,
    errors = 0,
    tools = 0,
    boundedStage = 0,
    boundedRead = false,
    boundedOutput = false;
  let requestSettings;
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    idleTimeout: 255,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === '/health') return Response.json({ status: 'ok', context_length: 16384 });
      if (url.pathname === '/v1/models')
        return Response.json({
          object: 'list',
          data: [{ id: config.model.slice(5), object: 'model' }],
        });
      if (req.method !== 'POST' || url.pathname !== '/v1/chat/completions')
        return new Response(null, { status: 404 });
      requests++;
      if (name === 'api-error')
        return Response.json(
          {
            error: {
              message: 'Protocol fixture rejected the request',
              type: 'invalid_request_error',
            },
          },
          { status: 400 },
        );
      const body = await req.json();
      if (body.tools?.length && !requestSettings)
        requestSettings = {
          keys: Object.keys(body),
          max_tokens: body.max_tokens,
          max_completion_tokens: body.max_completion_tokens,
          temperature: body.temperature,
          top_p: body.top_p,
          enable_thinking: body.enable_thinking,
        };
      if (!body.tools?.length) return reply(body, { content: 'Protocol fixture' }, 'stop');
      // Use the real client's schemas: OpenCode 2 calls bash "shell" and
      // uses "path" instead of "filePath" for read/write arguments.
      const fileArgs = (name, target) => {
        const properties = body.tools.find((tool) => tool.function?.name === name)?.function
          ?.parameters?.properties;
        if (!properties || (!properties.path && !properties.filePath))
          throw new Error('Missing advertised file tool schema');
        return { [properties.path ? 'path' : 'filePath']: target };
      };
      const shell = body.tools.some((tool) => tool.function?.name === 'shell') ? 'shell' : 'bash';
      if (name === 'bounded-tools') {
        const toolReply = (tool, args, id) =>
          reply(
            body,
            {
              tool_calls: [
                {
                  index: 0,
                  id,
                  type: 'function',
                  function: { name: tool, arguments: JSON.stringify(args) },
                },
              ],
            },
            'tool_calls',
          );
        const outputs = body.messages.filter((message) => message.role === 'tool');
        const last = JSON.stringify(outputs.at(-1)?.content ?? '');
        if (boundedStage++ === 0)
          return toolReply(
            'read',
            fileArgs('read', path.join(project, 'large-source.txt')),
            'call_bounded_read',
          );
        if (boundedStage === 2) {
          boundedRead = last.includes('READ_LINE_0079') && !last.includes('READ_LINE_0080');
          return toolReply(
            shell,
            {
              command: `bun -e 'console.log("BASH_HEAD" + "x".repeat(12000) + "BASH_TAIL")'`,
              description: 'Exercise bounded command output',
            },
            'call_bounded_bash',
          );
        }
        if (boundedStage === 3) {
          boundedOutput =
            last.length < 7000 &&
            last.includes('BASH_HEAD') &&
            last.includes('BASH_TAIL') &&
            last.includes('ZINC context limit');
          return toolReply(
            'write',
            {
              ...fileArgs('write', path.join(project, 'feature.mjs')),
              content: 'export const answer = 42;\n',
            },
            'call_bounded_write',
          );
        }
        return reply(
          body,
          { content: 'Focused edit complete; run independent verification.' },
          'stop',
        );
      }
      const repair = JSON.stringify(body.messages).includes('Independent verification failed');
      if ((!firstWrite && !repair) || (repair && !repaired)) {
        firstWrite = true;
        repaired = repair;
        if (!body.tools.some((tool) => tool.function?.name === 'write'))
          return new Response(null, { status: 400 });
        return reply(
          body,
          {
            tool_calls: [
              {
                index: 0,
                id: `call_fixture_${repair ? 'repair' : 'partial'}`,
                type: 'function',
                function: {
                  name: 'write',
                  arguments: JSON.stringify({
                    ...fileArgs('write', path.join(project, 'feature.mjs')),
                    content: `export const answer = ${repair ? 42 : 0};\n`,
                  }),
                },
              },
            ],
          },
          'tool_calls',
        );
      }
      if (repair)
        return reply(
          body,
          { content: 'Focused edit complete; run independent verification.' },
          'stop',
        );
      stalled++;
      // Keep transport alive so the editing deadline, rather than OpenCode
      // 2's socket-idle timeout, triggers cancellation and verification.
      return new Response(
        new ReadableStream({
          start(controller) {
            controllers.add(controller);
            const timer = setInterval(() => {
              try {
                controller.enqueue(new TextEncoder().encode(': fixture heartbeat\n\n'));
              } catch {
                clearInterval(timer);
                heartbeats.delete(timer);
              }
            }, 1000);
            heartbeats.add(timer);
            controller.enqueue(
              new TextEncoder().encode(
                'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
              ),
            );
          },
        }),
        { headers: { 'content-type': 'text/event-stream' } },
      );
    },
  });
  const started = performance.now();
  let result;
  try {
    const code = await runVerifiedOpenCode(
      config,
      {
        project,
        auto: true,
        prompt: ['Implement feature.mjs so answer is 42. Leave the acceptance test unchanged.'],
        verify: 'bun test',
        attempts: 3,
        roundTimeoutMs: opts.roundTimeoutMs,
        timeoutMs: opts.roundTimeoutMs * 3 + 15000,
      },
      {
        env: {
          ...env,
          OPENCODE_BIN: opts.opencode,
          ZINC_API_BASE_URL: `http://127.0.0.1:${server.port}/v1`,
        },
        stdout,
        stderr,
        onEvent(event, details) {
          if (event.sessionID) sessions.add(event.sessionID);
          if (event.type === 'error' && !details?.expectedTimeoutShutdown) errors++;
          if (event.type === 'tool_use') tools++;
        },
        async runCheck(command, checkOpts, deps) {
          const checked = await runVerification(command, checkOpts, deps);
          checks.push(checked.code);
          return checked;
        },
      },
    );
    const configUnchanged = hash(originalConfig) === hash(await readFile(configFile));
    const protectedTestUnchanged = testHash === hash(await readFile(testFile));
    const answerCorrect =
      (await readFile(path.join(project, 'feature.mjs'), 'utf8')).trim() ===
      'export const answer = 42;';
    const settingsMatch =
      requestSettings &&
      (requestSettings.max_completion_tokens ?? requestSettings.max_tokens) === 1024 &&
      (requestSettings.temperature ?? 0) === 0 &&
      (requestSettings.top_p ?? 1) === 1 &&
      requestSettings.enable_thinking === false;
    const passed =
      configUnchanged &&
      protectedTestUnchanged &&
      (name === 'api-error'
        ? code === 1 && errors > 0 && checks.length === 0
        : name === 'bounded-tools'
          ? settingsMatch &&
            code === 0 &&
            errors === 0 &&
            boundedRead &&
            boundedOutput &&
            tools >= 3 &&
            checks.length === 1 &&
            checks[0] === 0 &&
            answerCorrect
          : settingsMatch &&
            code === 0 &&
            errors === 0 &&
            stalled > 0 &&
            repaired &&
            sessions.size === 1 &&
            tools >= 2 &&
            checks.length === 2 &&
            checks[0] !== 0 &&
            checks[1] === 0 &&
            answerCorrect);
    result = {
      name,
      passed,
      run_exit_code: code,
      elapsed_ms: Math.round(performance.now() - started),
      requests,
      request_settings: requestSettings,
      ...(name === 'api-error' ? {} : { request_settings_valid: !!settingsMatch }),
      error_events: errors,
      tool_events: tools,
      unique_sessions: sessions.size,
      verification_exit_codes: checks,
      stalled_requests: stalled,
      configuration_unchanged: configUnchanged,
      protected_test_unchanged: protectedTestUnchanged,
      ...(name === 'bounded-tools'
        ? { read_limit_enforced: boundedRead, output_limit_enforced: boundedOutput }
        : {}),
      logs_truncated: stdout.truncated() || stderr.truncated(),
    };
  } catch (error) {
    result = { name, passed: false, error: 'Harness process did not complete' };
    await writeFile(path.join(dir, 'exception.log'), String(error.stack || error), { mode: 0o600 });
  } finally {
    server.stop(true);
    for (const timer of heartbeats) clearInterval(timer);
    for (const controller of controllers) {
      try {
        controller.close();
      } catch {}
    }
    await writeFile(path.join(dir, 'events.jsonl'), stdout.bytes(), { mode: 0o600 });
    await writeFile(path.join(dir, 'progress.log'), stderr.bytes(), { mode: 0o600 });
  }
  await writeFile(path.join(dir, 'result.json'), JSON.stringify(result, null, 2) + '\n', {
    mode: 0o600,
  });
  return result;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    console.log(`Usage: bun tools/validate_opencode_harness.mjs --output FRESH_DIR [options]

  --opencode FILE       OpenCode executable (default: OPENCODE_BIN or opencode)
  --expect-version N    Require this exact OpenCode version
  --round-timeout-ms N  Editing round budget (default: 15000; range: 1000..60000)

Runs actual OpenCode against scripted local HTTP responses, without model
inference. Checks timeout/repair, API failure handling, and real tool limits. Raw logs stay in the
new output directory; result.json contains no paths, endpoints, or credentials.`);
    return;
  }
  const bunVersion = globalThis.Bun?.version || '0.0.0';
  const [major, minor, patch] = bunVersion.split('.').map(Number.parseFloat);
  if (
    major < 1 ||
    (major === 1 &&
      (minor < 4 || (minor === 4 && (patch < 4 || (patch === 4 && bunVersion.includes('-'))))))
  )
    throw new Error('Bun 1.4.4 or newer is required');
  const root = path.resolve(opts.output);
  await mkdir(root, { mode: 0o700 });
  const env = {
    ...process.env,
    ZINC_API_KEY: 'local',
    OPENCODE_DISABLE_AUTOUPDATE: 'true',
    OPENCODE_DISABLE_MODELS_FETCH: 'true',
    XDG_CONFIG_HOME: path.join(root, 'config'),
    XDG_DATA_HOME: path.join(root, 'data'),
    XDG_STATE_HOME: path.join(root, 'state'),
    XDG_CACHE_HOME: path.join(root, 'cache'),
  };
  const version = parseOpenCodeVersion(
    (await exec(opts.opencode, ['--version'], { env, timeout: 10000 })).stdout,
  );
  if (opts.expectedVersion && version !== opts.expectedVersion)
    throw new Error('OpenCode returned a different version than requested');
  opts.runFlags = openCodeRunFlags({ env: { ...env, OPENCODE_BIN: opts.opencode } });
  const cases = [];
  for (const name of ['timeout-repair', 'api-error', 'bounded-tools']) {
    const result = await fixture(name, root, opts, env);
    cases.push(result);
    process.stderr.write(`${result.passed ? 'PASS' : 'FAIL'} ${name}\n`);
    if ([130, 143].includes(result.run_exit_code)) break;
  }
  const result = {
    scope: 'CPU protocol fixtures with actual OpenCode and scripted responses',
    model_inference: false,
    performance_measurement: false,
    opencode_version: version,
    helper_sha256: hash(await readFile(new URL('./setup_opencode.mjs', import.meta.url))),
    validator_sha256: hash(await readFile(new URL(import.meta.url))),
    passed: cases.length === 3 && cases.every((item) => item.passed),
    cases,
  };
  await writeFile(path.join(root, 'result.json'), JSON.stringify(result, null, 2) + '\n', {
    mode: 0o600,
  });
  console.log(JSON.stringify(result, null, 2));
  process.exitCode =
    cases.find((item) => [130, 143].includes(item.run_exit_code))?.run_exit_code ??
    (result.passed ? 0 : 1);
}

if (import.meta.main)
  main().catch((error) => {
    console.error(
      error.code === 'EEXIST'
        ? 'Output directory already exists; choose a fresh directory.'
        : /^(Pass --output|Invalid validation|Round timeout|Bun 1\.4\.4|OpenCode returned)/.test(
              error.message,
            )
          ? error.message
          : 'OpenCode harness validation failed. Check the executable and output directory.',
    );
    process.exitCode = 1;
  });
