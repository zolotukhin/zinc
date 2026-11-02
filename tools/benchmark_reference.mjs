#!/usr/bin/env bun
import { execFile } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const pinFile = new URL('../config/benchmark-reference.json', import.meta.url);
const execute = promisify(execFile);

export async function latestBenchmarkReference() {
  const { stdout } = await execute(
    'git',
    ['ls-remote', 'https://github.com/ggml-org/llama.cpp.git', 'HEAD'],
    { timeout: 30000 },
  );
  const commit = stdout.match(/^([0-9a-f]{40})\s+HEAD\s*$/)?.[1];
  if (!commit) throw new Error('Cannot identify the current upstream llama.cpp HEAD');
  return commit;
}

export async function checkBenchmarkReference() {
  const pin = JSON.parse(await readFile(pinFile, 'utf8'));
  const latest = await latestBenchmarkReference();
  if (pin.commit !== latest)
    throw new Error(
      `llama.cpp advanced to ${latest}. Run bun tools/benchmark_reference.mjs update and rebuild the reference before benchmarking.`,
    );
  return latest;
}

if (import.meta.main) {
  const command = process.argv[2];
  if (command === 'check') console.log(await checkBenchmarkReference());
  else if (command === 'update') {
    const pin = JSON.parse(await readFile(pinFile, 'utf8'));
    pin.commit = await latestBenchmarkReference();
    pin.checked_at_utc = new Date().toISOString();
    await writeFile(fileURLToPath(pinFile), `${JSON.stringify(pin, null, 2)}\n`);
    console.log(
      `Pinned ${pin.commit}. Rebuild llama-server and llama-tokenize before benchmarking.`,
    );
  } else throw new Error('Usage: bun tools/benchmark_reference.mjs <check|update>');
}
