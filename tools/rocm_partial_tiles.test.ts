import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('ROCm native-reference GPU fixtures retain a decisive partial-tile signal', () => {
  const directory = mkdtempSync(join(tmpdir(), 'zinc-partial-tile-fixtures-'));
  try {
    const executable = join(directory, 'reference-fixtures');
    const compile = spawnSync(
      'cc',
      [
        '-std=c11',
        '-O2',
        '-Wall',
        '-Wextra',
        '-Werror',
        '-DZINC_REFERENCE_ONLY',
        'tools/validate_rocm_partial_tiles.c',
        '-lm',
        '-o',
        executable,
      ],
      { cwd: new URL('..', import.meta.url), encoding: 'utf8', timeout: 30_000 },
    );
    expect(compile.status, compile.stdout + compile.stderr).toBe(0);
    const run = spawnSync(executable, [], { encoding: 'utf8', timeout: 10_000 });
    expect(run.status, run.stdout + run.stderr).toBe(0);
    expect(run.stdout).toContain('PASS 22 Q5_1/Q8_0 packing and decisive-tail reference fixtures');
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
