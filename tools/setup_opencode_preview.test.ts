import { afterEach, expect, test } from 'bun:test';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { buildConfig, CONFIG_NAME } from './setup_opencode.mjs';

const exec = promisify(execFile);
const helper = fileURLToPath(new URL('./setup_opencode.mjs', import.meta.url));
const directories: string[] = [];
afterEach(async () => {
  for (const dir of directories.splice(0)) await rm(dir, { recursive: true, force: true });
});
async function temp() {
  const dir = await mkdtemp(path.join(tmpdir(), 'zinc-preview-'));
  directories.push(dir);
  return dir;
}
async function cli(args: string[], env: Record<string, string> = {}) {
  try {
    return {
      code: 0,
      ...(await exec(process.execPath, [helper, ...args], {
        env: {
          ...process.env,
          OPENCODE_BIN: path.join(import.meta.dir, 'fixtures', 'uninstalled-opencode'),
          ...env,
        },
        timeout: 10000,
      })),
    };
  } catch (error: any) {
    return { code: error.code, stdout: error.stdout || '', stderr: error.stderr || '' };
  }
}

test('preview matches the written config byte for byte without creating a project', async () => {
  const root = await temp();
  const project = path.join(root, 'missing parent ✓', 'project');
  const result = await cli(['--dry-run', '--project', project]);
  expect(result.code).toBe(0);
  expect(result.stderr).toBe('');
  expect(result.stdout).toBe(JSON.stringify(buildConfig(), null, 2) + '\n');
  expect(await readdir(root)).toEqual([]);
  expect((await cli(['setup', '--project', project])).code).toBe(0);
  expect(await readFile(path.join(project, CONFIG_NAME), 'utf8')).toBe(result.stdout);
});

for (const modern of [false, true]) {
  test(`preview matches setup for the installed OpenCode ${modern ? '2' : '1'} CLI`, async () => {
    const root = await temp(),
      project = path.join(root, 'project');
    const binary = path.join(root, 'fake-opencode');
    await writeFile(
      binary,
      `#!${process.execPath}\nconsole.log('--model --format ${modern ? '--standalone' : '--dir'}');\n`,
      { mode: 0o700 },
    );
    const env = { OPENCODE_BIN: binary };
    const preview = await cli(['--project', project, '--dry-run'], env);
    expect(preview.code).toBe(0);
    expect(await readdir(root)).toEqual(['fake-opencode']);
    expect((await cli(['--project', project], env)).code).toBe(0);
    expect(await readFile(path.join(project, CONFIG_NAME), 'utf8')).toBe(preview.stdout);
    const config = JSON.parse(preview.stdout);
    expect(Array.isArray(modern ? config.plugins : config.plugin)).toBe(true);
    expect(preview.stdout).not.toContain(root);
  });
}

test('explicit preview honors model, context, output, steps and force', async () => {
  const root = await temp();
  const result = await cli([
    'setup',
    '--project',
    path.join(root, 'project'),
    '--dry-run',
    '--model',
    'custom:local',
    '--context',
    '32768',
    '--output',
    '2048',
    '--steps',
    '7',
    '--force',
  ]);
  expect(result.code).toBe(0);
  expect(result.stdout).toBe(
    JSON.stringify(
      buildConfig({ model: 'custom:local', context: 32768, output: 2048, steps: 7 }),
      null,
      2,
    ) + '\n',
  );
  expect(await readdir(root)).toEqual([]);
});

test('preview preserves existing invalid configs and symlinked hook directories', async () => {
  const project = await temp();
  const target = await temp();
  await writeFile(path.join(project, CONFIG_NAME), 'keep generated config\n');
  await writeFile(path.join(project, 'opencode.json'), 'keep user config\n');
  await symlink(target, path.join(project, '.zinc'));
  const result = await cli(['--project', project, '--dry-run', '--force']);
  expect(result.code).toBe(0);
  expect(result.stdout).toBe(JSON.stringify(buildConfig(), null, 2) + '\n');
  expect(await readFile(path.join(project, CONFIG_NAME), 'utf8')).toBe('keep generated config\n');
  expect(await readFile(path.join(project, 'opencode.json'), 'utf8')).toBe('keep user config\n');
  expect(await readdir(target)).toEqual([]);
});

test('preview ignores runtime endpoints and keys and needs no usable project directory', async () => {
  const root = await temp();
  const file = path.join(root, 'file');
  await writeFile(file, 'keep');
  const result = await cli(['--project', path.join(file, 'child'), '--dry-run'], {
    ZINC_API_BASE_URL: 'https://fixture-user:fixture-secret@example.invalid/v1?token=fixture',
    ZINC_API_KEY: 'fixture-key-do-not-capture',
  });
  expect(result.code).toBe(0);
  expect(result.stdout).toBe(JSON.stringify(buildConfig(), null, 2) + '\n');
  expect(result.stdout).not.toContain('fixture-secret');
  expect(result.stdout).not.toContain('fixture-key-do-not-capture');
  expect(await readFile(file, 'utf8')).toBe('keep');
});

test('check and run reject preview before config access or requiring a task', async () => {
  const root = await temp();
  for (const args of [['check'], ['run'], ['run', '--', 'Task must not execute']]) {
    const [command, ...rest] = args;
    const result = await cli([
      command,
      '--project',
      path.join(root, 'missing'),
      '--dry-run',
      ...rest,
    ]);
    expect(result.code).toBe(1);
    expect(result.stdout).toBe('');
    expect(result.stderr).toMatch(/dry.run.*setup|setup.*dry.run/i);
  }
  expect(await readdir(root)).toEqual([]);
});

test('invalid preview budgets fail without creating configuration or hook files', async () => {
  const root = await temp();
  for (const args of [
    ['--model', '../bad'],
    ['--context', '4096'],
    ['--output', '0'],
    ['--steps', '101'],
  ]) {
    const result = await cli(['--project', path.join(root, 'missing'), '--dry-run', ...args]);
    expect(result.code).toBe(1);
    expect(result.stdout).toBe('');
  }
  expect(await readdir(root)).toEqual([]);
});
