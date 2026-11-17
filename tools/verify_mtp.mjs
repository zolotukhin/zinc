#!/usr/bin/env node
/**
 * Verify MTP request propagation against one already-running ZINC server.
 * Never activates/downloads models, restarts servers, or changes environment.
 * This is a behavioral check, not a throughput benchmark or GPU-reference test.
 *
 * node tools/verify_mtp.mjs --base http://127.0.0.1:9090 --output /tmp/mtp-check.json
 */
import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const PROMPT =
  'Write the positive integers from 1 to 20, separated by commas. Give only the numbers.';
const HELP = `Usage: node tools/verify_mtp.mjs [options]
  --base <url>        Running server root or /v1 URL (default: http://127.0.0.1:9090)
  --model <id>        Require this already-active model; never activates a model
  --output <path>     Optional JSON report (omits endpoint and generated text)
  --max-tokens <n>    Per-request budget, 8..64 (default: 32)
  --timeout-ms <ms>   Per-request timeout (default: 120000)
  --strict-parity     Fail if greedy MTP on/off completion text differs
  -h, --help          Show help

Checks stream/nonstream on/off/default propagation, queued choice isolation,
actual draft/acceptance metadata, and small token budgets. Output parity is
recorded but does not establish equivalence to a reference implementation.`;

export function parseArgs(argv) {
  const options = {
    base: 'http://127.0.0.1:9090',
    model: null,
    output: null,
    maxTokens: 32,
    timeoutMs: 120_000,
    strictParity: false,
    help: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (key === '--help' || key === '-h') {
      options.help = true;
      continue;
    }
    if (key === '--strict-parity') {
      options.strictParity = true;
      continue;
    }
    const names = {
      '--base': 'base',
      '--model': 'model',
      '--output': 'output',
      '--max-tokens': 'maxTokens',
      '--timeout-ms': 'timeoutMs',
    };
    if (!names[key]) throw new Error('Unknown argument; use --help for supported options.');
    const value = argv[++i];
    if (!value || value.startsWith('--'))
      throw new Error('An option value is missing; use --help.');
    options[names[key]] = ['maxTokens', 'timeoutMs'].includes(names[key]) ? Number(value) : value;
  }
  if (!Number.isInteger(options.maxTokens) || options.maxTokens < 8 || options.maxTokens > 64)
    throw new Error('--max-tokens must be an integer from 8 to 64.');
  if (
    !Number.isInteger(options.timeoutMs) ||
    options.timeoutMs < 1_000 ||
    options.timeoutMs > 600_000
  )
    throw new Error('--timeout-ms must be an integer from 1000 to 600000.');
  options.base = serverRoot(options.base);
  return options;
}

export function serverRoot(value) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error('--base must be an HTTP(S) server URL.');
  }
  if (
    !['http:', 'https:'].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  )
    throw new Error('--base must be an HTTP(S) URL without credentials, query, or fragment.');
  url.pathname = url.pathname.replace(/\/+$/, '').replace(/\/v1$/, '');
  return url.toString().replace(/\/$/, '');
}

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}
function safeText(value) {
  return String(value || '')
    .replace(/https?:\/\/[^\s"<>]+/gi, '[endpoint]')
    .replace(/\b(?:\d{1,3}\.){3}\d{1,3}\b/g, '[address]')
    .replace(/\b[\w.-]+\.(?:local|internal|lan)\b/gi, '[host]')
    .replace(/(?:[A-Za-z]:[\\/]|\/(?:Users|home|root|tmp|var|opt)\/)[^\s"<>]*/g, '[path]')
    .slice(0, 500);
}
function modelLabel(value) {
  return safeText(
    String(value || '')
      .split(/[\\/]/)
      .pop(),
  );
}
function digest(value) {
  return createHash('sha256').update(value).digest('hex');
}
function outputText(payload) {
  return payload?.choices?.[0]?.message?.content ?? payload?.choices?.[0]?.text ?? '';
}

export function validateMtp(result, setting, capability, budget, finishReason = null) {
  requireValue(result && typeof result === 'object', 'Completion omitted actual MTP metadata.');
  requireValue(
    result.requested === (setting === undefined ? null : setting),
    'MTP metadata did not preserve the requested setting.',
  );
  const expected = capability.supports_mtp && (setting ?? capability.mtp_default_enabled);
  requireValue(
    result.enabled === expected,
    'MTP effective setting differs from the requested/default policy.',
  );
  requireValue(typeof result.used === 'boolean', 'MTP metadata omitted actual usage.');
  for (const key of ['drafted', 'accepted', 'cycles'])
    requireValue(
      Number.isSafeInteger(result[key]) && result[key] >= 0,
      'MTP counters must be nonnegative safe integers.',
    );
  requireValue(result.accepted <= result.drafted, 'MTP accepted more tokens than were drafted.');
  requireValue(
    result.accepted < budget,
    'MTP accepted more draft tokens than fit after the initial output token.',
  );
  requireValue(
    result.fallback_reason === null || typeof result.fallback_reason === 'string',
    'MTP fallback reason must be a string or null.',
  );
  if (!result.enabled)
    requireValue(
      !result.used && result.drafted === 0 && result.accepted === 0 && result.cycles === 0,
      'Disabled MTP still performed speculative work.',
    );
  if (result.used)
    requireValue(
      result.enabled && result.cycles > 0 && result.drafted > 0,
      'MTP usage claim has no actual speculative cycle.',
    );
  else
    requireValue(
      result.drafted === 0 && result.accepted === 0 && result.cycles === 0,
      'Unused MTP reported speculative work.',
    );
  // EOS/EOT can finish a request before the first speculative cycle. That is
  // ordinary completion, not a failed MTP setup. The full matrix separately
  // requires at least one actual speculative cycle on a capable server.
  if (result.enabled && !result.used && budget > 1 && finishReason !== 'stop')
    requireValue(
      typeof result.fallback_reason === 'string' && result.fallback_reason.length > 0,
      'Enabled MTP neither ran nor reported a fallback reason.',
    );
  return {
    requested: result.requested,
    enabled: result.enabled,
    used: result.used,
    drafted: result.drafted,
    accepted: result.accepted,
    cycles: result.cycles,
    fallback_reason: result.fallback_reason === null ? null : safeText(result.fallback_reason),
  };
}

export async function readStreamingCompletion(response) {
  requireValue(response.body, 'Streaming completion has no response body.');
  const reader = response.body.getReader(),
    decoder = new TextDecoder();
  let pending = '',
    text = '',
    mtp = null,
    finishReason = null,
    doneMarker = false;
  const event = (block) => {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n');
    if (!data) return;
    if (data === '[DONE]') {
      doneMarker = true;
      return;
    }
    let payload;
    try {
      payload = JSON.parse(data);
    } catch {
      throw new Error('Streaming completion contains invalid JSON.');
    }
    requireValue(!payload.error, 'Streaming completion reported an API error.');
    if (payload.mtp !== undefined) mtp = payload.mtp;
    const choice = payload.choices?.[0];
    const chunk = choice?.delta?.content ?? choice?.text ?? '';
    requireValue(typeof chunk === 'string', 'Streaming completion text is malformed.');
    text += chunk;
    if (choice?.finish_reason) finishReason = choice.finish_reason;
  };
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        pending += decoder.decode();
        break;
      }
      pending += decoder.decode(value, { stream: true });
      let match;
      while ((match = pending.match(/\r?\n\r?\n/))) {
        event(pending.slice(0, match.index));
        pending = pending.slice(match.index + match[0].length);
      }
    }
    if (pending.trim()) event(pending);
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
  requireValue(doneMarker, 'Streaming completion omitted its DONE marker.');
  requireValue(finishReason, 'Streaming completion omitted its finish reason.');
  return { text, mtp, finishReason };
}

async function fetchApi(fetchImpl, url, options, timeoutMs) {
  try {
    return await fetchImpl(url, { ...options, signal: AbortSignal.timeout(timeoutMs) });
  } catch {
    throw new Error('Server request failed or timed out.');
  }
}

async function responseJson(response) {
  try {
    return await response.json();
  } catch {
    throw new Error('Server returned invalid JSON.');
  }
}

export async function runVerification(options, fetchImpl = fetch) {
  const root = serverRoot(options.base);
  const report = {
    schema_version: 1,
    checked_at: new Date().toISOString(),
    status: 'failed',
    backend: null,
    model: null,
    capability: null,
    checks: [],
    cases: [],
    output_parity: null,
    reference_correctness_validated: false,
  };
  const check = (name, fn) => {
    try {
      const value = fn();
      report.checks.push({ name, passed: true });
      return value;
    } catch (error) {
      report.checks.push({ name, passed: false, error: safeText(error.message) });
      return null;
    }
  };
  let model, capability;
  try {
    const healthResponse = await fetchApi(fetchImpl, root + '/health', {}, options.timeoutMs);
    requireValue(healthResponse.ok, 'Health endpoint failed.');
    const health = await responseJson(healthResponse);
    requireValue(health.status === 'ok', 'Server is not healthy.');
    const modelsResponse = await fetchApi(fetchImpl, root + '/v1/models', {}, options.timeoutMs);
    requireValue(modelsResponse.ok, 'Model catalog endpoint failed.');
    const catalog = await responseJson(modelsResponse);
    requireValue(
      Array.isArray(catalog.data) && catalog.data.length > 0,
      'No active model was reported.',
    );
    model = options.model
      ? catalog.data.find((entry) => entry.id === options.model)
      : (catalog.data.find((entry) => entry.active) ??
        (catalog.data.length === 1 ? catalog.data[0] : null));
    requireValue(
      model && (model.active === true || catalog.data.length === 1),
      'Requested model is not active; verification never changes models.',
    );
    capability = model;
    for (const value of [health, model]) {
      requireValue(
        typeof value.supports_mtp === 'boolean' &&
          typeof value.mtp_default_enabled === 'boolean' &&
          value.mtp_request_control === true,
        'Server does not report the MTP request-control contract; update the running server before testing propagation.',
      );
    }
    requireValue(
      health.supports_mtp === model.supports_mtp &&
        health.mtp_default_enabled === model.mtp_default_enabled,
      'Health/model MTP capability reports disagree.',
    );
    requireValue(
      !model.mtp_default_enabled || model.supports_mtp,
      'Server enables MTP by default while reporting it unsupported.',
    );
    report.backend = ['vulkan', 'metal', 'cuda', 'rocm', 'zinc_rt'].includes(health.backend)
      ? health.backend
      : 'reported';
    report.model = modelLabel(model.id);
    report.capability = {
      supports_mtp: model.supports_mtp,
      mtp_default_enabled: model.mtp_default_enabled,
      mtp_request_control: true,
      mtp_unavailable_reason: safeText(model.mtp_unavailable_reason),
    };
    report.checks.push({ name: 'capability_preflight', passed: true });
  } catch (error) {
    report.checks.push({
      name: 'capability_preflight',
      passed: false,
      error: safeText(error.message || 'Capability preflight failed.'),
    });
    return report;
  }
  const observations = new Map();
  const runCase = async (name, setting, stream, maxTokens = options.maxTokens) => {
    const result = {
      name,
      stream,
      requested: setting === undefined ? null : setting,
      budget: maxTokens,
      passed: false,
    };
    report.cases.push(result);
    try {
      const body = {
        model: model.id,
        messages: [{ role: 'user', content: PROMPT }],
        max_tokens: maxTokens,
        temperature: 0,
        enable_thinking: false,
        stream,
      };
      if (setting !== undefined) body.enable_mtp = setting;
      const response = await fetchApi(
        fetchImpl,
        root + '/v1/chat/completions',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
        options.timeoutMs,
      );
      if (setting === true && !capability.supports_mtp) {
        requireValue(
          response.status === 400,
          'Unsupported MTP opt-in was not rejected with HTTP 400.',
        );
        const rejected = await responseJson(response);
        requireValue(
          typeof rejected.error?.message === 'string' && rejected.error.message.length > 0,
          'Unsupported MTP rejection omitted its explanation.',
        );
        result.rejected = true;
        result.passed = true;
        return;
      }
      requireValue(response.ok, 'Completion request failed.');
      let completion;
      if (stream) completion = await readStreamingCompletion(response);
      else {
        const payload = await responseJson(response);
        requireValue(!payload.error, 'Completion reported an API error.');
        completion = {
          text: outputText(payload),
          mtp: payload.mtp,
          finishReason: payload.choices?.[0]?.finish_reason,
        };
        requireValue(completion.finishReason, 'Completion omitted its finish reason.');
      }
      requireValue(
        typeof completion.text === 'string' && completion.text.length > 0,
        'Completion returned no visible output.',
      );
      result.mtp = validateMtp(
        completion.mtp,
        setting,
        capability,
        maxTokens,
        completion.finishReason,
      );
      result.output_chars = completion.text.length;
      result.output_sha256 = digest(completion.text);
      result.finish_reason = safeText(completion.finishReason);
      result.passed = true;
      observations.set(name, completion.text);
    } catch (error) {
      result.error = safeText(error.message || 'Completion check failed.');
    }
  };
  for (const stream of [false, true]) {
    const suffix = stream ? 'stream' : 'json';
    for (const [name, setting] of [
      ['omitted', undefined],
      ['null', null],
      ['off', false],
      ['on', true],
    ])
      await runCase(name + '_' + suffix, setting, stream);
  }
  if (capability.supports_mtp) {
    await Promise.all([runCase('queued_off', false, true), runCase('queued_on', true, true)]);
    await runCase('on_budget_1', true, false, 1);
    await runCase('on_budget_2', true, true, 2);
  }
  await runCase('default_after_overrides', undefined, false);
  if (capability.supports_mtp)
    check('speculative_execution_observed', () => {
      requireValue(
        report.cases.some((item) => item.mtp?.used),
        'No actual MTP cycle was observed; speculative execution remains unverified.',
      );
    });
  if (capability.supports_mtp && observations.has('off_json') && observations.has('on_json')) {
    const equal = observations.get('off_json') === observations.get('on_json');
    report.output_parity = {
      greedy_text_equal: equal,
      strict: options.strictParity === true,
      note: 'Text comparison only; this does not validate output against llama.cpp or another target reference.',
    };
    check('greedy_text_parity', () => {
      requireValue(
        !options.strictParity || equal,
        'Greedy MTP on/off completion text differs; investigate numerical/state correctness before claiming parity.',
      );
    });
  }
  report.status =
    report.checks.every((item) => item.passed) && report.cases.every((item) => item.passed)
      ? 'passed'
      : 'failed';
  return report;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(HELP);
    return;
  }
  const report = await runVerification(options);
  if (options.output) await writeFile(options.output, JSON.stringify(report, null, 2) + '\n');
  console.log(
    `MTP propagation ${report.status}: ${report.cases.filter((item) => item.passed).length}/${report.cases.length} request checks passed.`,
  );
  for (const failure of [...report.checks, ...report.cases].filter((item) => !item.passed))
    console.error(`${failure.name}: ${failure.error}`);
  if (report.output_parity?.greedy_text_equal === false)
    console.log('Greedy MTP on/off text differed; no output-equivalence claim is made.');
  if (report.status !== 'passed') process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  main().catch(() => {
    console.error('MTP verification could not complete; use --help and check local configuration.');
    process.exitCode = 1;
  });
