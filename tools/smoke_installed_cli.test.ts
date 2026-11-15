import { expect, test } from 'bun:test';
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { smokeInstalledCli } from './smoke_installed_cli.mjs';

async function fixture(schema = 1) {
  const directory = await mkdtemp(join(tmpdir(), 'zinc cli fixture '));
  await mkdir(join(directory, 'bin'));
  await mkdir(join(directory, 'share/zinc/shaders/metal'), { recursive: true });
  await writeFile(
    join(directory, 'share/zinc/shaders/metal/fixture.metal'),
    '// Package-layout fixture; never compiled.\n',
  );
  await writeFile(
    join(directory, 'bin/zinc'),
    `#!/usr/bin/env node
const fs=require('node:fs');
if(fs.readdirSync('.').length)process.exit(4);
const args=process.argv.slice(2);
const build={version:'dev',commit:'fixture',target:'fixture',optimize:'fast',backend:'metal'};
if(args.includes('--help')||args.includes('--help-all'))console.log('zinc serve --no-browser --check [--json] --version [--json]');
else if(args.includes('--version'))console.log(args.includes('--json')?JSON.stringify({schema_version:${schema},build}):'zinc dev');
else process.exit(5);
`,
  );
  await chmod(join(directory, 'bin/zinc'), 0o755);
  return directory;
}

test('installed CLI smoke uses an empty cwd, copies assets and leaves the original package intact', async () => {
  const directory = await fixture();
  try {
    const original = await readFile(join(directory, 'bin/zinc'));
    const result = await smokeInstalledCli(directory);
    expect(result.isolated_cli).toBe('pass');
    expect(result.assets).toBe('pass');
    expect(await readFile(join(directory, 'bin/zinc'))).toEqual(original);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('installed CLI smoke rejects a different JSON contract and incomplete runtime assets', async () => {
  const invalid = await fixture(99);
  const missing = await fixture();
  try {
    await expect(smokeInstalledCli(invalid)).rejects.toThrow();
    await rm(join(missing, 'share'), { recursive: true });
    await expect(smokeInstalledCli(missing)).rejects.toThrow();
  } finally {
    await rm(invalid, { recursive: true, force: true });
    await rm(missing, { recursive: true, force: true });
  }
});
