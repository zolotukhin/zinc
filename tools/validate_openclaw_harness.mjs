#!/usr/bin/env node
// CPU-only: actual OpenClaw + a scripted HTTP server. This proves client wire
// compatibility and real tool execution, never model or GPU correctness.
import { createServer } from 'node:http';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { buildConfig, runOpenClaw, validateClient, writeConfig } from './setup_openclaw.mjs';
import { createFixture, TASK, verifyFixture } from './openclaw_eval.mjs';

function response(res, body, delta, finish) {
  const base = { id: 'chatcmpl-zinc-openclaw-protocol', model: body.model, created: 0 };
  const usage = { prompt_tokens: 1000, completion_tokens: 40, total_tokens: 1040 };
  if (!body.stream) {
    const message = { role: 'assistant', ...delta };
    if (message.tool_calls)
      message.tool_calls = message.tool_calls.map(({ index, ...call }) => call);
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        ...base,
        object: 'chat.completion',
        choices: [{ index: 0, message, finish_reason: finish }],
        usage,
      }),
    );
    return;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  const event = (value, reason = null) =>
    res.write(
      `data: ${JSON.stringify({ ...base, object: 'chat.completion.chunk', choices: [{ index: 0, delta: value, finish_reason: reason }] })}\n\n`,
    );
  event({ role: 'assistant' });
  if (delta.tool_calls) {
    const call = delta.tool_calls[0],
      args = call.function.arguments,
      midpoint = Math.floor(args.length / 2);
    // Exercise argument assembly across SSE chunks, including escaped content.
    event({
      tool_calls: [
        { ...call, function: { name: call.function.name, arguments: args.slice(0, midpoint) } },
      ],
    });
    event({ tool_calls: [{ index: 0, function: { arguments: args.slice(midpoint) } }] });
  } else event(delta);
  event({}, finish);
  res.write(
    `data: ${JSON.stringify({ ...base, object: 'chat.completion.chunk', choices: [], usage })}\n\n`,
  );
  res.end('data: [DONE]\n\n');
}

const args = process.argv.slice(2),
  opts = {};
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--help') {
    console.log(
      'node validate_openclaw_harness.mjs --output NEW_DIR [--openclaw-bin PATH]\nCPU protocol test using the actual installed client; does not load a model.',
    );
    process.exit(0);
  }
  const field = { '--output': 'output', '--openclaw-bin': 'binary' }[args[i]];
  if (!field || !args[i + 1]) throw new Error('Invalid arguments; use --help');
  opts[field] = args[++i];
}
if (!opts.output) throw new Error('Use a fresh --output directory');
const version = await promisify(execFile)(opts.binary || 'openclaw', ['--version'], {
  timeout: 30000,
});
if (!/^OpenClaw 2026\.11\.1(?:\s|$)/.test(version.stdout.trim()))
  throw new Error('This protocol fixture pins OpenClaw 2026.11.1');
const root = path.resolve(opts.output);
await mkdir(root, { mode: 0o700 });
const project = path.join(root, 'project'),
  fixture = await createFixture(project);
const requests = [];
let resolveCancellationClosed;
const cancellationClosed = new Promise((resolve) => {
  resolveCancellationClosed = resolve;
});
let failure,
  step = 0,
  phase = 'tools',
  cancelledConnectionClosed = false,
  apiErrorResponses = 0,
  cancelRequested = false;
const actions = [
  ['read', { path: 'contract.json' }],
  [
    'write',
    {
      path: 'feature.mjs',
      content:
        "export function normalizeTag(value) {return value.trim().toLowerCase().replace(/\\s+/g,'-');}\n",
    },
  ],
  ['write', { path: 'receipt.txt', content: fixture.receipt + '\n' }],
  ['exec', { command: 'node --test feature.test.mjs', yieldMs: 1000 }],
];
const server = createServer(async (req, res) => {
  try {
    if (req.url !== '/v1/chat/completions' || req.method !== 'POST')
      throw new Error('Unexpected client endpoint');
    let raw = '';
    for await (const chunk of req) {
      raw += chunk;
      if (raw.length > 1024 * 1024) throw new Error('Oversized request');
    }
    const body = JSON.parse(raw);
    requests.push({
      model: body.model,
      stream: body.stream,
      max_tokens: body.max_tokens,
      roles: body.messages.map((message) => message.role),
      tools: body.tools?.map((tool) => tool.function.name),
    });
    await writeFile(path.join(root, 'requests-private.json'), JSON.stringify(requests, null, 2));
    if (
      body.model !== 'zinc-protocol' ||
      body.messages.some((message) => message.role === 'developer')
    )
      throw new Error('Wrong model identity or unsupported developer role');
    if (phase === 'error') {
      apiErrorResponses++;
      res.writeHead(401, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          error: {
            message: 'ZINC_PROTOCOL_EXPECTED_AUTH_FAILURE',
            type: 'authentication_error',
            code: 'invalid_api_key',
          },
        }),
      );
      return;
    }
    if (phase === 'cancel') {
      // Cancel only after the actual client reaches HTTP. A fixed startup
      // delay can expire before a slower CI worker has opened a connection.
      res.once('close', () => {
        cancelledConnectionClosed = true;
        resolveCancellationClosed();
      });
      if (!cancelRequested) {
        cancelRequested = true;
        setImmediate(() => process.kill(process.pid, 'SIGINT'));
      }
      return;
    }
    if (phase === 'recovery') {
      response(res, body, { content: 'ZINC_PROTOCOL_RECOVERED' }, 'stop');
      return;
    }
    if (step > 0) {
      const call = body.messages
        .filter((message) => message.role === 'assistant')
        .flatMap((message) => message.tool_calls || [])
        .at(-1);
      const result = body.messages.filter((message) => message.role === 'tool').at(-1);
      // Clients may normalize call IDs. The replayed assistant and tool must
      // still agree, and the result must belong to the action we requested.
      if (
        !call ||
        !result ||
        call.id !== result.tool_call_id ||
        call.function.name !== actions[step - 1][0]
      )
        throw new Error('Missing correlated tool result');
      if (step === 1 && !JSON.stringify(result.content).includes(fixture.receipt))
        throw new Error('Read tool did not return the real fixture');
    }
    const action = actions[step++];
    if (!action) {
      response(res, body, { content: 'ZINC_PROTOCOL_OK' }, 'stop');
      return;
    }
    const [name, parameters] = action;
    const tool = body.tools?.find((tool) => tool.function.name === name);
    if (!tool) throw new Error(`Required tool is missing: ${name}`);
    const properties = tool.function.parameters?.properties || {};
    if (parameters.path && !properties.path) {
      const key = properties.file_path ? 'file_path' : properties.filePath ? 'filePath' : null;
      if (!key) throw new Error('Unknown file tool path parameter');
      parameters[key] = parameters.path;
      delete parameters.path;
    }
    response(
      res,
      body,
      {
        tool_calls: [
          {
            index: 0,
            id: `protocol-${step}`,
            type: 'function',
            function: { name, arguments: JSON.stringify(parameters) },
          },
        ],
      },
      'tool_calls',
    );
  } catch (error) {
    failure = error.message;
    await writeFile(path.join(root, 'server-failure-private.json'), JSON.stringify({ failure }));
    res.writeHead(500, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: failure } }));
  }
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
try {
  const { filename } = await writeConfig(
    project,
    buildConfig({
      baseUrl: `http://127.0.0.1:${server.address().port}/v1`,
      model: 'zinc-protocol',
      context: 32768,
    }),
  );
  await validateClient({ binary: opts.binary, filename, project });
  const client = await runOpenClaw({
    binary: opts.binary,
    filename,
    project,
    model: 'zinc-protocol',
    prompt: TASK,
    timeout: 60,
  });
  const verified = await verifyFixture(project, fixture);
  const toolRequests = [...requests];
  const failureChecks = {};
  let apiErrorEvidence;
  phase = 'error';
  try {
    // Use the helper's normal outer deadline. Client startup precedes its
    // 30-second task timer, particularly on smaller CI workers.
    await runOpenClaw({
      binary: opts.binary,
      filename,
      project,
      model: 'zinc-protocol',
      prompt: 'Reply OK.',
      timeout: 30,
    });
    failureChecks.apiErrorRejected = false;
  } catch (error) {
    const diagnostics = error.diagnostics;
    let report;
    try {
      report = JSON.parse(diagnostics?.stdout || '');
    } catch {}
    apiErrorEvidence = {
      responses: apiErrorResponses,
      exitCode: diagnostics?.code,
      signal: diagnostics?.signal,
      reportedOk: report?.ok,
      status: report?.status,
      helperError: error.message,
    };
    failureChecks.apiErrorRejected =
      apiErrorResponses > 0 &&
      Number.isInteger(diagnostics?.code) &&
      diagnostics.code !== 0 &&
      report?.ok === false;
    await writeFile(
      path.join(root, 'api-error-private.json'),
      JSON.stringify({ report, diagnostics }, null, 2),
      { mode: 0o600 },
    );
  }
  phase = 'cancel';
  try {
    await runOpenClaw(
      {
        binary: opts.binary,
        filename,
        project,
        model: 'zinc-protocol',
        prompt: 'Reply OK.',
        timeout: 60,
      },
      { deadlineMs: 30000, graceMs: 1000 },
    );
    failureChecks.cancellationRejected = false;
  } catch (error) {
    failureChecks.cancellationRejected = error.message === 'OpenClaw was interrupted';
  }
  // Socket-close delivery may trail the child's close event. Wait for that
  // event with a separate bound rather than accepting a stale flag.
  failureChecks.cancelledConnectionClosed =
    cancelledConnectionClosed ||
    (await new Promise((resolve) => {
      const timer = setTimeout(() => resolve(false), 2000);
      cancellationClosed.then(() => {
        clearTimeout(timer);
        resolve(true);
      });
    }));
  phase = 'recovery';
  const recovered = await runOpenClaw({
    binary: opts.binary,
    filename,
    project,
    model: 'zinc-protocol',
    prompt: 'Reply ZINC_PROTOCOL_RECOVERED.',
    timeout: 30,
  });
  failureChecks.recovery = recovered.report.final?.trim() === 'ZINC_PROTOCOL_RECOVERED';
  const summary = {
    kind: 'cpu-scripted-protocol',
    clientVersion: version.stdout.trim(),
    passed:
      !failure &&
      verified.passed &&
      client.report.final?.trim() === 'ZINC_PROTOCOL_OK' &&
      toolRequests.length === 5 &&
      toolRequests.every((request) => request.stream === true) &&
      Object.values(failureChecks).every(Boolean),
    requests: toolRequests,
    failureChecks,
    apiErrorEvidence,
    verification: verified,
    toolSummary: client.report.toolSummary,
  };
  await writeFile(path.join(root, 'client-private.json'), JSON.stringify(client, null, 2), {
    mode: 0o600,
  });
  await writeFile(path.join(root, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify(summary, null, 2));
  if (!summary.passed) process.exitCode = 1;
} catch (error) {
  await writeFile(
    path.join(root, 'failure-private.json'),
    JSON.stringify(
      { message: error.message, diagnostics: error.diagnostics, requests, failure },
      null,
      2,
    ),
  );
  throw error;
} finally {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
}
