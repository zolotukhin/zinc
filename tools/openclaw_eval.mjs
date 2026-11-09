#!/usr/bin/env node
// Actual-client acceptance test. Run against a server on an authorized GPU
// worker. Scripted protocol checks and model coding ability are separate claims.
import { execFile } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import {
  buildConfig,
  discoverServer,
  INTERRUPTED_CODE,
  runOpenClaw,
  validateClient,
  writeConfig,
} from './setup_openclaw.mjs';

const exec = promisify(execFile);
const hash = (value) => createHash('sha256').update(value).digest('hex');
export const TASK =
  'Read contract.json and feature.mjs. Implement normalizeTag(value): trim whitespace, lowercase, and replace each run of whitespace with a single hyphen. Write the exact receipt value from contract.json to receipt.txt, followed by a newline. Run node --test feature.test.mjs and fix failures. Do not modify contract.json or feature.test.mjs. Complete the file changes using tools, then briefly report the test result.';

export async function createFixture(project) {
  const contract = JSON.stringify({ receipt: `zinc-openclaw-${randomUUID()}` }) + '\n';
  const test = `import {test} from 'node:test';
import assert from 'node:assert/strict';
import {normalizeTag} from './feature.mjs';
test('normalize whitespace and case',()=>{
 for(const [input,want] of [['  Hello WORLD  ','hello-world'],['a\\tB\\nC','a-b-c'],['already-normal','already-normal'],['  ','']]) assert.equal(normalizeTag(input),want);
});
`;
  await mkdir(project, { mode: 0o700 });
  await writeFile(path.join(project, 'contract.json'), contract);
  await writeFile(path.join(project, 'feature.test.mjs'), test);
  await writeFile(
    path.join(project, 'feature.mjs'),
    'export function normalizeTag(value) { return value; }\n',
  );
  return {
    receipt: JSON.parse(contract).receipt,
    protectedHashes: { 'contract.json': hash(contract), 'feature.test.mjs': hash(test) },
  };
}

export async function verifyFixture(project, fixture, { node = 'node' } = {}) {
  const checks = {};
  for (const [name, expected] of Object.entries(fixture.protectedHashes)) {
    checks[name] = await readFile(path.join(project, name))
      .then((bytes) => hash(bytes) === expected)
      .catch(() => false);
  }
  checks.receipt = await readFile(path.join(project, 'receipt.txt'), 'utf8')
    .then((text) => text === fixture.receipt + '\n')
    .catch(() => false);
  // This second execution is independent of the assistant's claimed tool result.
  try {
    await exec(node, ['--test', 'feature.test.mjs'], {
      cwd: project,
      timeout: 10000,
      maxBuffer: 1024 * 1024,
    });
    checks.tests = true;
  } catch {
    checks.tests = false;
  }
  return { passed: Object.values(checks).every(Boolean), checks };
}

export async function evaluate(
  { output, baseUrl, binary = 'openclaw', context = 'auto', timeout = 300, apiKeyEnv },
  { discover = discoverServer, validate = validateClient, run = runOpenClaw } = {},
) {
  const directory = path.resolve(output);
  // A fresh directory prevents an earlier successful run from supplying files.
  await mkdir(directory, { mode: 0o700 });
  const project = path.join(directory, 'project');
  const fixture = await createFixture(project);
  const server = await discover({ baseUrl, context, apiKeyEnv });
  const { filename } = await writeConfig(
    project,
    buildConfig({ ...server, output: 1536, timeout, apiKeyEnv }),
  );
  await validate({ binary, filename, project });
  const results = [];
  const saveSummary = async (healthyAfter, extra = {}) => {
    const summary = {
      schema: 1,
      client: 'OpenClaw',
      model: server.model,
      context: server.context,
      results,
      healthyAfter,
      passed: healthyAfter === true && results.every((result) => result.passed),
      ...extra,
    };
    await writeFile(path.join(directory, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
    return summary;
  };
  for (const [name, prompt] of [
    ['text', 'Reply with exactly ZINC_OPENCLAW_OK. Do not use any tools.'],
    ['workspace', TASK],
    ['recovery', 'Reply with exactly ZINC_OPENCLAW_RECOVERED. Do not use any tools.'],
  ]) {
    const started = Date.now();
    try {
      const {
        report,
        stderr,
        preparationFailures = [],
      } = await run({
        binary,
        filename,
        project,
        model: server.model,
        prompt,
        timeout,
      });
      await writeFile(
        path.join(directory, `${name}-client-private.json`),
        JSON.stringify(report, null, 2) + '\n',
        { mode: 0o600 },
      );
      await writeFile(path.join(directory, `${name}-stderr-private.log`), stderr, { mode: 0o600 });
      if (preparationFailures.length)
        await writeFile(
          path.join(directory, `${name}-preparation-failures-private.json`),
          JSON.stringify(preparationFailures, null, 2) + '\n',
          { mode: 0o600 },
        );
      const verification =
        name === 'workspace'
          ? await verifyFixture(project, fixture)
          : {
              passed:
                report.final?.trim() ===
                (name === 'text' ? 'ZINC_OPENCLAW_OK' : 'ZINC_OPENCLAW_RECOVERED'),
            };
      results.push({
        name,
        completed: true,
        passed: verification.passed,
        ...(verification.checks ? { checks: verification.checks } : {}),
        toolSummary: report.toolSummary ?? null,
        ...(preparationFailures.length ? { preparationRetries: preparationFailures.length } : {}),
        seconds: (Date.now() - started) / 1000,
      });
    } catch (error) {
      await writeFile(
        path.join(directory, `${name}-failure-private.json`),
        JSON.stringify(
          {
            message: error.message,
            code: error.code,
            diagnostics: error.diagnostics,
            preparationFailures: error.preparationFailures,
          },
          null,
          2,
        ) + '\n',
        { mode: 0o600 },
      );
      results.push({
        name,
        completed: false,
        passed: false,
        ...(error.preparationFailures?.length
          ? { preparationRetries: error.preparationFailures.length }
          : {}),
        seconds: (Date.now() - started) / 1000,
      });
      // Cancellation ends the suite too; starting a recovery task here would
      // create a new client while the parent is trying to retire this run.
      if (error.code === INTERRUPTED_CODE) {
        await saveSummary(null, { interrupted: true });
        throw error;
      }
    }
  }
  const health = await discover({
    baseUrl,
    context: server.context,
    model: server.model,
    apiKeyEnv,
  })
    .then(() => true)
    .catch(() => false);
  return saveSummary(health);
}

async function main() {
  const opts = {};
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--help') {
      console.log(
        'node openclaw_eval.mjs --base-url URL --output NEW_DIR [--openclaw-bin PATH] [--context N] [--timeout SECONDS] [--api-key-env NAME]\nRun only on the authorized GPU worker. Verifies text, real file edits/tests, and recovery; raw client logs remain private.',
      );
      return;
    }
    const field = {
      '--base-url': 'baseUrl',
      '--output': 'output',
      '--openclaw-bin': 'binary',
      '--context': 'context',
      '--timeout': 'timeout',
      '--api-key-env': 'apiKeyEnv',
    }[args[i]];
    if (!field || !args[i + 1] || args[i + 1].startsWith('--'))
      throw new Error('Invalid arguments; use --help');
    opts[field] = args[++i];
  }
  if (!opts.output || !opts.baseUrl)
    throw new Error('Pass --base-url and a fresh --output directory');
  if (
    opts.timeout !== undefined &&
    (!Number.isSafeInteger(Number(opts.timeout)) ||
      Number(opts.timeout) < 1 ||
      Number(opts.timeout) > 7200)
  )
    throw new Error('Timeout must be 1–7200 seconds');
  if (opts.timeout !== undefined) opts.timeout = Number(opts.timeout);
  const summary = await evaluate(opts);
  console.log(JSON.stringify(summary, null, 2));
  if (!summary.passed) process.exitCode = 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch((error) => {
    console.error(`OpenClaw evaluation: ${error.message}`);
    process.exitCode = 1;
  });
