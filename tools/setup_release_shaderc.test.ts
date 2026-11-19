import { expect, test } from 'bun:test';
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

const script = resolve(import.meta.dir, '../scripts/setup_release_shaderc.sh');

function fixture(run: (root: string, env: Record<string, string | undefined>) => void) {
  const root = mkdtempSync(join(tmpdir(), 'zinc shader setup '));
  try {
    mkdirSync(join(root, 'bin'));
    mkdirSync(join(root, 'empty-sources'));
    const env = {
      ...process.env,
      PATH: `${join(root, 'bin')}:${process.env.PATH}`,
      GITHUB_ACTIONS: '',
      ZINC_RELEASE_SETUP: '',
      SHADERC_TEST_LOG: join(root, 'calls.json'),
    };
    run(root, env);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

function executable(root: string, name: string, body: string) {
  const path = join(root, 'bin', name);
  writeFileSync(path, body);
  chmodSync(path, 0o755);
}

function call(env: Record<string, string | undefined>, command: string, ...args: string[]) {
  return Bun.spawnSync(['bash', '-c', `source "$1"; ${command}`, '--', script, ...args], {
    env,
    stdout: 'pipe',
    stderr: 'pipe',
    timeout: 5000,
  });
}

test('shader setup uses authenticated direct HTTPS Ubuntu sources and retains security updates', () => {
  fixture((root, env) => {
    const path = join(root, 'sources.list');
    expect(call(env, 'shaderc_write_sources "$2"', path).exitCode).toBe(0);
    const lines = readFileSync(path, 'utf8').trim().split('\n');
    expect(lines).toHaveLength(3);
    const rows = lines.map((line) => {
      const match =
        /^deb \[arch=amd64 signed-by=\/usr\/share\/keyrings\/ubuntu-archive-keyring.gpg\] (https:\/\/[^ ]+) (noble(?:-updates|-security)?) main universe$/.exec(
          line,
        );
      expect(match).not.toBeNull();
      return { url: match![1], suite: match![2] };
    });
    expect(rows).toEqual([
      { url: 'https://archive.ubuntu.com/ubuntu', suite: 'noble' },
      { url: 'https://archive.ubuntu.com/ubuntu', suite: 'noble-updates' },
      { url: 'https://security.ubuntu.com/ubuntu', suite: 'noble-security' },
    ]);
  });
});

test('every APT operation excludes image mirror lists, bounds both transports and fails closed', () => {
  fixture((root, env) => {
    executable(root, 'sudo', '#!/usr/bin/env bash\nexec "$@"\n');
    executable(
      root,
      'apt-get',
      `#!/usr/bin/env bun
await Bun.write(process.env.SHADERC_TEST_LOG, JSON.stringify({ args: process.argv.slice(2), frontend: process.env.DEBIAN_FRONTEND }));
process.exit(Number(process.env.SHADERC_TEST_EXIT ?? 0));
`,
    );
    const sources = join(root, 'sources.list');
    expect(call(env, 'shaderc_apt "$2" update', sources).exitCode).toBe(0);
    const log = JSON.parse(readFileSync(env.SHADERC_TEST_LOG!, 'utf8'));
    const options = log.args.filter((arg: string, i: number) => log.args[i - 1] === '-o');
    expect(options).toContain(`Dir::Etc::sourcelist=${sources}`);
    expect(options).toContain(`Dir::Etc::sourceparts=${join(root, 'empty-sources')}`);
    expect(options).toContain('Acquire::http::Timeout=30');
    expect(options).toContain('Acquire::https::Timeout=30');
    expect(options).toContain('DPkg::Lock::Timeout=60');
    expect(options).toContain('APT::Update::Error-Mode=any');
    expect(log.frontend).toBe('noninteractive');
    expect(log.args.at(-1)).toBe('update');
    const failed = call(
      { ...env, SHADERC_TEST_EXIT: '100' },
      'shaderc_apt "$2" update; echo unexpected-success',
      sources,
    );
    expect(failed.exitCode).toBe(100);
    expect(failed.stdout.toString()).not.toContain('unexpected-success');
  });
});

test('the whole setup has a deadline and timeout failures propagate', () => {
  fixture((root, env) => {
    // No installation authorization is set. Even a broken timeout wrapper
    // cannot start package operations on the machine running these CPU tests.
    executable(
      root,
      'timeout',
      `#!/usr/bin/env bun
await Bun.write(process.env.SHADERC_TEST_LOG, JSON.stringify(process.argv.slice(2)));
process.exit(124);
`,
    );
    expect(call(env, 'shaderc_bounded_install "$1"').exitCode).toBe(124);
    const args = JSON.parse(readFileSync(env.SHADERC_TEST_LOG!, 'utf8'));
    expect(args.slice(0, 3)).toEqual(['--signal=TERM', '--kill-after=30s', '600s']);
    expect(args[3]).toBe('bash');
    expect(args.at(-1)).toBe(script);
  });
});

test('the executable entry point still requires a disposable-worker opt-in', () => {
  fixture((_root, env) => {
    const result = Bun.spawnSync(['bash', script], { env, timeout: 5000 });
    expect(result.exitCode).toBe(1);
    expect(result.stderr.toString()).toContain('only on a disposable build worker');
  });
});

test.each([0, 3])(
  'the full setup pins the compiler and cleans up after APT failure %i',
  (failAt) => {
    fixture((root, env) => {
      executable(root, 'sudo', '#!/usr/bin/env bash\nexec "$@"\n');
      executable(
        root,
        'apt-get',
        `#!/usr/bin/env bun
import { appendFileSync, existsSync, readFileSync, readdirSync } from 'node:fs';
const args = process.argv.slice(2);
const option = name => args.find(arg => arg.startsWith(name + '='))?.slice(name.length + 1);
const list = option('Dir::Etc::sourcelist');
const log = process.env.SHADERC_TEST_LOG;
const count = existsSync(log) ? readFileSync(log, 'utf8').trim().split('\\n').length : 0;
appendFileSync(log, JSON.stringify({args, list, sources: readFileSync(list, 'utf8'), parts: readdirSync(option('Dir::Etc::sourceparts'))}) + '\\n');
process.exit(count + 1 === Number(process.env.SHADERC_TEST_FAIL_AT) ? 100 : 0);
`,
      );
      for (const name of ['curl', 'gpg']) executable(root, name, '#!/bin/sh\nexit 0\n');
      executable(root, 'glslc', '#!/bin/sh\nprintf called > "$SHADERC_TEST_VERSION_CALLED"\n');
      const marker = join(root, 'version-called');
      const result = call(
        { ...env, SHADERC_TEST_FAIL_AT: String(failAt), SHADERC_TEST_VERSION_CALLED: marker },
        // Only the distro/authorization checks are replaced. Every package and
        // network command is mocked; temporary sources and cleanup run for real.
        'shaderc_require_disposable() { :; }; shaderc_codename() { printf noble; }; shaderc_install "$1"',
      );
      expect(result.exitCode).toBe(failAt ? 100 : 0);
      const rows = readFileSync(env.SHADERC_TEST_LOG!, 'utf8')
        .trim()
        .split('\n')
        .map((line) => JSON.parse(line));
      expect(rows).toHaveLength(failAt || 4);
      expect(rows[0].args.at(-1)).toBe('update');
      expect(rows[1].args).toContain('libvulkan-dev');
      expect(rows[0].sources).not.toContain('packages.lunarg.com');
      expect(rows[2].sources).toContain(
        '[arch=amd64 signed-by=/usr/share/keyrings/zinc-lunarg.gpg] https://packages.lunarg.com/vulkan/1.4.313 noble main',
      );
      for (const row of rows) expect(row.parts).toEqual([]);
      if (!failAt) {
        const pin = JSON.parse(
          readFileSync(resolve(import.meta.dir, '../config/release/targets.json'), 'utf8'),
        ).toolchain.shaderc;
        expect(rows[3].args.at(-1)).toBe('shaderc=' + pin);
      }
      expect(existsSync(marker)).toBe(!failAt);
      expect(existsSync(dirname(rows[0].list))).toBe(false);
    });
  },
);
