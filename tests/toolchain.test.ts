import { expect, test } from 'bun:test';
import { execFileSync } from 'node:child_process';
import {
  chmodSync,
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { remoteZigToolchainDir } from '../tools/performance_suite.mjs';

const root = resolve(import.meta.dir, '..');
const version = readFileSync(join(root, '.zig-version'), 'utf8').trim();

test('source, release, Nix and every CI compiler setup use the same repository pin', () => {
  const minimum = readFileSync(join(root, 'build.zig.zon'), 'utf8').match(
    /\.minimum_zig_version\s*=\s*"([^"]+)"/,
  )?.[1];
  expect(minimum).toBe(version);
  const release = JSON.parse(readFileSync(join(root, 'config/release/targets.json'), 'utf8'));
  expect(release.toolchain.zig).toBe(version);
  const nix = JSON.parse(readFileSync(join(root, 'nix/zig-release.json'), 'utf8'));
  expect(nix.version).toBe(version);
  expect(Object.keys(nix.platforms).sort()).toEqual([
    'aarch64-darwin',
    'aarch64-linux',
    'x86_64-darwin',
    'x86_64-linux',
  ]);
  for (const [platform, dist] of Object.entries(nix.platforms) as [string, any][]) {
    expect(dist.url).toBe(
      `https://ziglang.org/download/${version}/zig-${platform.replace('-darwin', '-macos')}-${version}.tar.xz`,
    );
    expect(dist.hash).toMatch(/^sha256-[A-Za-z0-9+/]{43}=$/);
  }
  let setups = 0;
  for (const name of readdirSync(join(root, '.github/workflows')).filter((name) =>
    name.endsWith('.yml'),
  )) {
    const workflow = Bun.YAML.parse(
      readFileSync(join(root, '.github/workflows', name), 'utf8'),
    ) as any;
    for (const job of Object.values(workflow.jobs) as any[]) {
      let sourceCheckedOut = false;
      for (const step of job.steps ?? []) {
        if (step.uses?.startsWith('actions/checkout@')) sourceCheckedOut = true;
        if (!step.uses?.startsWith('mlugg/setup-zig@')) continue;
        expect(sourceCheckedOut).toBe(true);
        // setup-zig's documented default reads minimum_zig_version from the
        // checked-out source, including a dispatched historical release SHA.
        expect(step.with?.version ?? minimum).toBe(version);
        setups++;
      }
    }
  }
  expect(setups).toBeGreaterThan(0);
});

test('remote compiler defaults follow the pin and retain explicit worker overrides', () => {
  expect(remoteZigToolchainDir('/home/example')).toBe(
    `/home/example/.cache/zinc-toolchains/zig-${version}`,
  );
  expect(remoteZigToolchainDir('/home/example', '/opt/custom-zig', '/opt/other/zig')).toBe(
    '/opt/custom-zig',
  );
  expect(remoteZigToolchainDir('/home/example', undefined, '/opt/worker tools/zig')).toBe(
    '/opt/worker tools',
  );
});

test('compiler wrapper selects the pin over a mismatched system compiler and preserves arguments', () => {
  const fixture = mkdtempSync(join(tmpdir(), 'zinc-toolchain-'));
  try {
    mkdirSync(join(fixture, 'scripts'));
    mkdirSync(join(fixture, 'system-bin'));
    const cache = join(fixture, 'compiler cache');
    const pinned = join(cache, `zig-${version}`);
    mkdirSync(pinned, { recursive: true });
    copyFileSync(join(root, 'scripts/zig.sh'), join(fixture, 'scripts/zig.sh'));
    writeFileSync(join(fixture, '.zig-version'), version + '\n');
    function fakeCompiler(path: string, reportedVersion: string, marker: string) {
      writeFileSync(
        path,
        `#!/bin/sh\nif [ "$1" = version ]; then printf '%s\\n' '${reportedVersion}'; else printf '%s\\n' '${marker}'; printf '<%s>\\n' "$@"; fi\n`,
      );
      chmodSync(path, 0o755);
    }
    const system = join(fixture, 'system-bin/zig');
    fakeCompiler(system, '0.0.0', 'system');
    fakeCompiler(join(pinned, 'zig'), version, 'pinned');
    const env = {
      ...process.env,
      PATH: `${join(fixture, 'system-bin')}:/usr/bin:/bin`,
      ZINC_TOOLCHAIN_DIR: cache,
    };
    const run = (...args: string[]) =>
      execFileSync('sh', [join(fixture, 'scripts/zig.sh'), ...args], { env, encoding: 'utf8' });
    expect(run('version').trim()).toBe(version);
    expect(run('build', '--name', 'two words; $HOME')).toBe(
      'pinned\n<build>\n<--name>\n<two words; $HOME>\n',
    );
    fakeCompiler(system, version, 'system');
    expect(run('build')).toBe('system\n<build>\n');
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});
