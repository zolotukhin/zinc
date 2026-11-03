import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// Linux has the Vulkan headers/linker used by production. The fixture replaces
// all called entrypoints, including teardown, and never creates a GPU instance.
test.skipIf(process.platform !== 'linux')(
  'upload copies retain pending resources on wait failures',
  () => {
    const prefix = mkdtempSync(join(tmpdir(), 'zinc-upload-contract-'));
    try {
      const build = spawnSync(
        './scripts/zig.sh',
        [
          'build',
          'upload-copy-contract',
          '-Dshaders=false',
          '-Doptimize=Debug',
          '--prefix',
          prefix,
        ],
        { cwd: new URL('..', import.meta.url), encoding: 'utf8', timeout: 120_000 },
      );
      expect(build.status, build.stdout + build.stderr).toBe(0);
      const executable = join(prefix, 'bin/zinc-upload-copy-contract');
      const run = (mode: string) => {
        const result = spawnSync(executable, [mode], { encoding: 'utf8', timeout: 5_000 });
        return { ...result, output: result.stdout + result.stderr };
      };
      const success = run('success');
      expect(success.status, success.output).toBe(0);
      expect(success.output).toContain('event=verified-exact-copy-and-guards');
      expect(success.output.indexOf('event=wait')).toBeLessThan(
        success.output.indexOf('event=free-command'),
      );
      expect(success.output.match(/event=destroy-buffer/g)?.length).toBe(2);
      expect(success.output).not.toContain('unsafe-pending-cleanup');

      for (const [mode, error, forbidden] of [
        ['allocate', 'AllocCmdBufFailed', 'free-command'],
        ['begin', 'BeginCommandBufferFailed', 'record-copy'],
        ['end', 'EndCommandBufferFailed', 'submit'],
        ['submit', 'QueueSubmitFailed', 'wait'],
      ]) {
        const result = run(mode);
        expect(result.status, result.output).toBe(0);
        expect(result.output).toContain(`result=${error}`);
        expect(result.output).not.toContain(`event=${forbidden}`);
        expect(result.output.match(/event=destroy-buffer/g)?.length).toBe(2);
        if (mode !== 'allocate') expect(result.output.match(/event=free-command/g)?.length).toBe(1);
      }
      for (const mode of ['wait-host-oom', 'wait-device-oom', 'wait-device-lost', 'wait-unknown']) {
        const result = run(mode);
        expect(result.status, result.output).toBe(1);
        expect(result.output).toContain('event=wait');
        expect(result.output).toContain('Vulkan upload completion failed');
        for (const name of [
          'free-command',
          'destroy-buffer',
          'free-memory',
          'unmap',
          'unsafe-pending-cleanup',
        ])
          expect(result.output).not.toContain(`event=${name}`);
        expect(result.output).not.toContain('result=');
      }
    } finally {
      rmSync(prefix, { recursive: true, force: true });
    }
  },
  180_000,
);
