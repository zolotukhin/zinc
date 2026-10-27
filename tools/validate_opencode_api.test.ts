import { expect, test } from 'bun:test';
import { parseCompletion, validateBufferedToolCancellation } from './validate_opencode_api.mjs';

test('buffered cancellation does not wait for the tool response body before proving recovery', async () => {
  const requests: any[] = [];
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      const body = await request.json();
      requests.push(body);
      if (body.stream) {
        return new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(new TextEncoder().encode(': buffering\n\n'));
            },
          }),
          { headers: { 'content-type': 'text/event-stream' } },
        );
      }
      return Response.json({ choices: [{ message: { content: 'OK' } }] });
    },
  });
  try {
    const result = await validateBufferedToolCancellation({
      model: 'probe',
      env: { ZINC_API_BASE_URL: `http://127.0.0.1:${server.port}/v1` },
      timeoutMs: 1000,
    });
    expect(result.recovery_ms).toBeLessThan(1000);
    expect(requests).toHaveLength(2);
    expect(requests[0].tool_choice).toBe('required');
    expect(requests[0].tools[0].function.name).toBe('buffered_probe');
    expect(requests[1].stream).toBe(false);
  } finally {
    await server.stop(true);
  }
});

test('buffered cancellation fails when the next request remains queued', async () => {
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      const body = await request.json();
      if (!body.stream) return new Promise<Response>(() => {});
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(': buffering\n\n'));
          },
        }),
        { headers: { 'content-type': 'text/event-stream' } },
      );
    },
  });
  try {
    await expect(
      validateBufferedToolCancellation({
        model: 'probe',
        env: { ZINC_API_BASE_URL: `http://127.0.0.1:${server.port}/v1` },
        timeoutMs: 100,
      }),
    ).rejects.toThrow();
  } finally {
    await server.stop(true);
  }
});

test('assembles split tool arguments with Unicode and terminal usage', () => {
  const frames = [
    {
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, id: 'call_1', function: { name: 'probe', arguments: '{"message":' } },
            ],
          },
        },
      ],
    },
    { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"東京🚀"}' } }] } }] },
    {
      choices: [{ delta: {}, finish_reason: 'tool_calls' }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    },
  ];
  const text =
    frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join('') + 'data: [DONE]\n\n';
  const parsed = parseCompletion(text, true);
  expect(parsed.finish).toBe('tool_calls');
  expect(parsed.usage.total_tokens).toBe(15);
  expect(parsed.message.tool_calls[0].id).toBe('call_1');
  expect(JSON.parse(parsed.message.tool_calls[0].function.arguments)).toEqual({
    message: '東京🚀',
  });
});

test('rejects truncated streams and streamed server errors', () => {
  expect(() =>
    parseCompletion('data: {"choices":[{"delta":{"content":"OK"}}]}\n\n', true),
  ).toThrow();
  expect(() =>
    parseCompletion('data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n', true),
  ).toThrow();
});

test('keeps non-streaming tool calls and usage intact', () => {
  const json = {
    choices: [
      {
        message: { role: 'assistant', content: null, tool_calls: [{ id: 'call_1' }] },
        finish_reason: 'tool_calls',
      },
    ],
    usage: { total_tokens: 20 },
  };
  const parsed = parseCompletion(JSON.stringify(json), false);
  expect(parsed.message).toEqual(json.choices[0].message);
  expect(parsed.usage).toEqual(json.usage);
});
