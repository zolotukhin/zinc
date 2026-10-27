#!/usr/bin/env bun
// Run against an already running ZINC server. This script never starts a model.
import assert from 'node:assert/strict';
import { writeFile } from 'node:fs/promises';
import { buildConfig, runtimeConfig } from './setup_opencode.mjs';

export function parseCompletion(text, stream) {
  if (!stream) {
    const json = JSON.parse(text);
    assert.ok(!json.error);
    return {
      message: json.choices[0].message,
      finish: json.choices[0].finish_reason,
      usage: json.usage,
    };
  }
  assert.ok(text.includes('data: [DONE]'));
  const calls = new Map();
  let content = '',
    finish = null,
    usage = null;
  for (const line of text.split(/\r?\n/)) {
    if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
    const event = JSON.parse(line.slice(6));
    assert.ok(!event.error);
    const choice = event.choices?.[0];
    content += choice?.delta?.content || '';
    if (choice?.finish_reason) finish = choice.finish_reason;
    if (event.usage) usage = event.usage;
    for (const part of choice?.delta?.tool_calls || []) {
      const call = calls.get(part.index) || {
        id: '',
        type: 'function',
        function: { name: '', arguments: '' },
      };
      if (part.id) call.id = part.id;
      if (part.function?.name) call.function.name += part.function.name;
      call.function.arguments += part.function?.arguments || '';
      calls.set(part.index, call);
    }
  }
  assert.ok(finish);
  return {
    message: { role: 'assistant', content, tool_calls: [...calls.values()] },
    finish,
    usage,
  };
}

export async function validateBufferedToolCancellation({
  model,
  env = process.env,
  timeoutMs = 10000,
}) {
  const config = runtimeConfig(buildConfig({ model }), env);
  const url = `${config.provider.zinc.options.baseURL}/chat/completions`;
  const headers = {
    'content-type': 'application/json',
    authorization: `Bearer ${env.ZINC_API_KEY || 'local'}`,
  };
  const response = await fetch(url, {
    method: 'POST',
    headers,
    signal: AbortSignal.timeout(timeoutMs),
    body: JSON.stringify({
      model,
      messages: [
        {
          role: 'user',
          content:
            'Call buffered_probe with text containing copper repeated 5000 times. Do not abbreviate.',
        },
      ],
      tools: [
        {
          type: 'function',
          function: {
            name: 'buffered_probe',
            parameters: {
              type: 'object',
              properties: { text: { type: 'string', minLength: 20000 } },
              required: ['text'],
            },
          },
        },
      ],
      tool_choice: 'required',
      temperature: 0,
      enable_thinking: false,
      max_tokens: 4096,
      stream: true,
    }),
  });
  assert.equal(response.status, 200);
  await response.body.cancel();
  const started = performance.now();
  const recovered = await fetch(url, {
    method: 'POST',
    headers,
    signal: AbortSignal.timeout(timeoutMs),
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: 'Reply with OK.' }],
      temperature: 0,
      enable_thinking: false,
      max_tokens: 8,
      stream: false,
    }),
  });
  assert.equal(recovered.status, 200);
  const json = await recovered.json();
  assert.ok(!json.error && json.choices?.length > 0);
  assert.ok(performance.now() - started < timeoutMs);
  return { recovery_ms: Math.round(performance.now() - started) };
}

export async function validateApi({
  model = 'qwen38-27b-q4k-m',
  context = 16384,
  env = process.env,
  bufferedToolCancellation = false,
} = {}) {
  const config = runtimeConfig(buildConfig({ model, context }), env);
  const baseURL = config.provider.zinc.options.baseURL;
  const headers = {
    'content-type': 'application/json',
    authorization: `Bearer ${env.ZINC_API_KEY || 'local'}`,
  };
  const checks = [];
  const request = async (body, { signal = AbortSignal.timeout(180000) } = {}) => {
    const response = await fetch(`${baseURL}/chat/completions`, {
      method: 'POST',
      headers,
      body: typeof body === 'string' ? body : JSON.stringify(body),
      signal,
    });
    return response;
  };
  const complete = async (body) => {
    const response = await request({
      model,
      temperature: 0,
      top_p: 1,
      enable_thinking: false,
      stream: false,
      max_tokens: 64,
      ...body,
    });
    if (response.status !== 200)
      assert.fail(`HTTP ${response.status}: ${(await response.text()).slice(0, 1000)}`);
    const raw = new Uint8Array(await response.arrayBuffer());
    const text = new TextDecoder('utf-8', { fatal: true }).decode(raw);
    const result = parseCompletion(text, !!body.stream);
    assert.ok(['stop', 'length', 'tool_calls'].includes(result.finish));
    assert.ok(result.usage?.prompt_tokens > 0);
    assert.ok(result.usage?.completion_tokens > 0);
    assert.equal(
      result.usage.total_tokens,
      result.usage.prompt_tokens + result.usage.completion_tokens,
    );
    assert.ok(!String(result.message.content || '').includes('<|im_end|>'));
    return result;
  };
  const check = async (name, fn) => {
    const started = performance.now();
    try {
      await fn();
      checks.push({ name, pass: true, ms: Math.round(performance.now() - started) });
    } catch (error) {
      checks.push({ name, pass: false, ms: Math.round(performance.now() - started) });
      process.stderr.write(`${name}: ${String(error?.message ?? error).slice(0, 1000)}\n`);
    }
    process.stderr.write(`${checks.at(-1).pass ? 'PASS' : 'FAIL'} ${name}\n`);
  };

  for (const budget of [1, 2, 7]) {
    await check(`JSON/SSE parity at ${budget} output tokens`, async () => {
      const body = { messages: [{ role: 'user', content: 'Reply with OK.' }], max_tokens: budget };
      const json = await complete(body),
        sse = await complete({ ...body, stream: true });
      assert.equal(sse.message.content, json.message.content);
      assert.equal(sse.finish, json.finish);
      assert.ok(json.usage.completion_tokens <= budget);
    });
  }
  await check('Unicode JSON/SSE parity', async () => {
    const body = {
      messages: [{ role: 'user', content: 'Repeat exactly this text: café 東京 🚀' }],
    };
    const json = await complete(body),
      sse = await complete({ ...body, stream: true });
    assert.equal(sse.message.content, json.message.content);
    for (const word of ['café', '東京', '🚀']) assert.ok(json.message.content.includes(word));
    assert.ok(!json.message.content.includes('\uFFFD'));
  });
  for (const budget of [1, 2, 3]) {
    await check(`Unicode JSON/SSE parity at ${budget} output tokens`, async () => {
      const body = {
        messages: [{ role: 'user', content: 'Repeat exactly this text: 🚀' }],
        max_tokens: budget,
      };
      const json = await complete(body),
        sse = await complete({ ...body, stream: true });
      assert.equal(sse.message.content, json.message.content);
      assert.ok(!json.message.content.includes('\uFFFD'));
      assert.ok(json.usage.completion_tokens <= budget);
    });
  }

  const tools = [
    {
      type: 'function',
      function: {
        name: 'zinc_probe',
        description: 'Record a diagnostic message.',
        parameters: {
          type: 'object',
          properties: { message: { type: 'string' } },
          required: ['message'],
        },
      },
    },
  ];
  const toolBody = {
    messages: [{ role: 'user', content: 'Call zinc_probe with message "café 東京 🚀".' }],
    tools,
    tool_choice: 'required',
    max_tokens: 128,
  };
  let priorCall;
  const ids = new Set();
  for (const stream of [false, true]) {
    await check(`${stream ? 'SSE' : 'JSON'} tool arguments and unique ID`, async () => {
      const result = await complete({ ...toolBody, stream });
      assert.equal(result.finish, 'tool_calls');
      const call = result.message.tool_calls[0];
      assert.ok(call.id && !ids.has(call.id));
      ids.add(call.id);
      assert.equal(call.function.name, 'zinc_probe');
      assert.equal(JSON.parse(call.function.arguments).message, 'café 東京 🚀');
      priorCall = call;
    });
  }
  const typedArguments = {
    count: 2,
    enabled: true,
    labels: ['東京'],
    options: { safe: true },
    text: '0',
  };
  for (const stream of [false, true]) {
    await check(
      `${stream ? 'SSE' : 'JSON'} tool schema retains numbers booleans arrays objects and strings`,
      async () => {
        const result = await complete({
          messages: [
            {
              role: 'user',
              content: `Call typed_probe with exactly these arguments: ${JSON.stringify(typedArguments)}`,
            },
          ],
          tools: [
            {
              type: 'function',
              function: {
                name: 'typed_probe',
                description: 'Check typed diagnostic arguments.',
                parameters: {
                  type: 'object',
                  properties: {
                    count: { type: 'integer' },
                    enabled: { type: 'boolean' },
                    labels: { type: 'array', items: { type: 'string' } },
                    options: {
                      type: 'object',
                      properties: { safe: { type: 'boolean' } },
                      required: ['safe'],
                    },
                    text: { type: 'string' },
                  },
                  required: Object.keys(typedArguments),
                },
              },
            },
          ],
          tool_choice: 'required',
          max_tokens: 256,
          stream,
        });
        assert.equal(result.finish, 'tool_calls');
        const call = result.message.tool_calls[0];
        assert.equal(call.function.name, 'typed_probe');
        assert.deepEqual(JSON.parse(call.function.arguments), typedArguments);
      },
    );
  }
  await check('Truncated tool calls remain text without an executable call', async () => {
    for (const stream of [false, true]) {
      const result = await complete({ ...toolBody, stream, max_tokens: 1 });
      assert.equal(result.finish, 'length');
      assert.equal(result.message.tool_calls?.length || 0, 0);
    }
  });
  await check('Null assistant content and tool-result history', async () => {
    assert.ok(priorCall);
    const result = await complete({
      messages: [
        { role: 'user', content: 'Look up the test token.' },
        { role: 'assistant', content: null, tool_calls: [priorCall] },
        { role: 'tool', tool_call_id: priorCall.id, content: 'zinc-canary-57' },
        {
          role: 'user',
          content: 'What exact token did the tool return? Reply with that token only.',
        },
      ],
    });
    assert.ok(result.message.content.includes('zinc-canary-57'));
  });
  await check('tool_choice none', async () => {
    const result = await complete({
      messages: [{ role: 'user', content: 'Reply with OK.' }],
      tools,
      tool_choice: 'none',
    });
    assert.ok(!result.message.tool_calls?.length);
    assert.ok(result.message.content.includes('OK'));
  });
  await check('Text content arrays', async () => {
    const result = await complete({
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Reply with OK.' }] }],
    });
    assert.ok(result.message.content.includes('OK'));
  });
  for (const [name, body] of [
    ['Malformed JSON', '{"messages":'],
    ['Invalid message content', { messages: [{ role: 'user', content: 42 }] }],
    ['Empty messages', { messages: [] }],
    ['Negative token budget', { messages: [{ role: 'user', content: 'Hi' }], max_tokens: -1 }],
    [
      'Context overflow',
      { messages: [{ role: 'user', content: 'word '.repeat(context + 2048) }], max_tokens: 1 },
    ],
  ]) {
    await check(`${name} returns HTTP 400`, async () => {
      const response = await request(body);
      assert.equal(response.status, 400);
      assert.ok((await response.json()).error);
    });
  }
  await check('Queued requests keep conversations separate', async () => {
    const values = ['APPLE', 'BANANA', 'CHERRY'];
    const results = await Promise.all(
      values.map((value) =>
        complete({
          messages: [{ role: 'user', content: `Reply with only ${value}.` }],
          max_tokens: 16,
          stream: true,
        }),
      ),
    );
    for (let i = 0; i < values.length; i++)
      assert.ok(results[i].message.content.includes(values[i]));
  });
  await check('Streaming cancellation releases the slot', async () => {
    const controller = new AbortController();
    const response = await request(
      {
        model,
        messages: [
          {
            role: 'user',
            content:
              'Count the integers from 1 to 10000, separated by commas. Continue counting without summarizing.',
          },
        ],
        enable_thinking: false,
        max_tokens: 1024,
        stream: true,
      },
      { signal: controller.signal },
    );
    assert.equal(response.status, 200);
    const reader = response.body.getReader();
    await reader.read();
    controller.abort();
    await reader.cancel().catch(() => {});
    const started = performance.now();
    const result = await complete({
      messages: [{ role: 'user', content: 'Reply with OK.' }],
      max_tokens: 8,
    });
    assert.ok(result.message.content.includes('OK'));
    assert.ok(performance.now() - started < 10000);
  });
  await check('Long prompt above the former 8192-token SSM limit', async () => {
    const result = await complete({
      messages: [
        {
          role: 'user',
          content:
            'The following line is reference filler.\n' +
            'word '.repeat(9200) +
            '\nIgnore the reference filler and reply with OK.',
        },
      ],
      max_tokens: 16,
    });
    assert.ok(result.usage.prompt_tokens > 8192);
    assert.ok(result.message.content.includes('OK'));
  });
  if (bufferedToolCancellation) {
    await check('Buffered tool-call cancellation releases the slot', () =>
      validateBufferedToolCancellation({ model, env }),
    );
  }
  return {
    model,
    context,
    passed: checks.filter((item) => item.pass).length,
    total: checks.length,
    checks,
    complete: true,
  };
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  if (args.includes('--help')) {
    console.log(
      'Usage: bun tools/validate_opencode_api.mjs [--model ID] [--context N] [--output FILE] [--buffered-tool-cancel]\nUses ZINC_API_BASE_URL and ZINC_API_KEY at runtime. Run only where GPU requests are authorized.',
    );
  } else {
    try {
      const options = {};
      for (let i = 0; i < args.length; i += 1) {
        if (args[i] === '--buffered-tool-cancel') {
          options.bufferedToolCancellation = true;
          continue;
        }
        if (!['--model', '--context', '--output'].includes(args[i]) || !args[i + 1])
          throw new Error('arguments');
        options[args[i].slice(2)] = args[i] === '--context' ? Number(args[i + 1]) : args[i + 1];
        i += 1;
      }
      const result = await validateApi(options);
      const text = JSON.stringify(result, null, 2) + '\n';
      if (options.output) await writeFile(options.output, text);
      console.log(text);
      if (result.passed !== result.total) process.exitCode = 1;
    } catch {
      console.error('API validation failed; check arguments and the ZINC connection.');
      process.exitCode = 1;
    }
  }
}
