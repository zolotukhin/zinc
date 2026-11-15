#!/usr/bin/env node
// Exercise the built package from an empty cwd. Only --help and --version run:
// this CPU-only check must never initialize a GPU, open a model or start a server.
import assert from 'node:assert/strict';
import { cp, mkdir, mkdtemp, readdir, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

export async function smokeInstalledCli(prefix) {
  const temporary = await mkdtemp(join(tmpdir(), 'zinc installed cli '));
  try {
    const packageDir = join(temporary, 'installed package');
    const cwd = join(temporary, 'empty working directory');
    await mkdir(join(packageDir, 'bin'), { recursive: true });
    await mkdir(cwd);
    await cp(join(resolve(prefix), 'bin/zinc'), join(packageDir, 'bin/zinc'), {
      dereference: true,
    });
    const run = (args) => {
      const result = spawnSync(join(packageDir, 'bin/zinc'), args, {
        cwd,
        encoding: 'utf8',
        timeout: 15_000,
        maxBuffer: 256 * 1024,
      });
      assert.equal(result.error, undefined, 'installed CLI failed to execute');
      assert.equal(result.status, 0, `installed CLI ${args.join(' ')} failed`);
      return result.stdout;
    };
    const first = JSON.parse(run(['--version', '--json']));
    assert.equal(first.schema_version, 1);
    for (const field of ['version', 'commit', 'target', 'optimize', 'backend']) {
      assert.equal(typeof first.build[field], 'string', `missing build.${field}`);
      assert.ok(first.build[field].length > 0, `empty build.${field}`);
    }
    assert.deepEqual(
      JSON.parse(run(['--json', '--version'])),
      first,
      'version flag order changed metadata',
    );
    assert.ok(run(['--version']).startsWith(`zinc ${first.build.version}\n`));
    for (const flag of ['--help', '--help-all']) {
      const help = run([flag]);
      for (const command of [
        'zinc serve',
        '--no-browser',
        '--check [--json]',
        '--version [--json]',
      ])
        assert.ok(help.includes(command), `missing help: ${command}`);
    }
    // Copy the runtime assets as a package, rather than treating the executable
    // as a single-file Vulkan/Metal distribution. No shader/GPU code is executed.
    if (['vulkan', 'metal'].includes(first.build.backend)) {
      const shaders = join(packageDir, 'share/zinc/shaders');
      await cp(join(resolve(prefix), 'share/zinc/shaders'), shaders, {
        recursive: true,
        dereference: true,
      });
      const assetDir = first.build.backend === 'metal' ? join(shaders, 'metal') : shaders;
      const extension = first.build.backend === 'metal' ? '.metal' : '.spv';
      assert.ok(
        (await readdir(assetDir)).some((name) => name.endsWith(extension)),
        `package has no ${extension} assets`,
      );
      assert.ok((await stat(assetDir)).isDirectory());
    }
    assert.deepEqual(
      await readdir(cwd),
      [],
      'CLI metadata commands wrote into the caller directory',
    );
    return { schema_version: 1, build: first.build, isolated_cli: 'pass', assets: 'pass' };
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.length !== 4 || process.argv[2] !== '--prefix')
    throw new Error('Usage: smoke_installed_cli.mjs --prefix <installation-prefix>');
  console.log(JSON.stringify(await smokeInstalledCli(process.argv[3])));
}
