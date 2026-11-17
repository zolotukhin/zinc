import { expect, test } from 'bun:test';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

// Exercise the shipped Bash selector without downloading or activating a release.
const script = readFileSync(join(import.meta.dir, '../scripts/install.sh'), 'utf8');
const selector = script.match(/^linux_backend\(\) \{[\s\S]*?^\}/m)![0];

function select(vendors: string[]) {
  const root = mkdtempSync(join(tmpdir(), 'zinc-gpu-choice-'));
  try {
    vendors.forEach((vendor, index) => {
      const device = join(root, `card${index}`, 'device');
      mkdirSync(device, { recursive: true });
      writeFileSync(join(device, 'vendor'), vendor + '\n');
    });
    return spawnSync('bash', ['-c', selector + '\nlinux_backend'], {
      env: { ...process.env, ZINC_DRM_SYSFS_ROOT: root },
      encoding: 'utf8',
    });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test('Linux installer defaults to HIP on AMD-only machines, including multiple AMD devices', () => {
  for (const vendors of [['0x1002'], ['0x1002', '0x1002']]) {
    const result = select(vendors);
    expect(result.status).toBe(0);
    expect(result.stdout.trim()).toBe('rocm');
  }
});

test('Intel and GPU-less packaging hosts keep the Vulkan download', () => {
  for (const vendors of [[], ['0x8086'], ['0x1234']]) {
    const result = select(vendors);
    expect(result.status).toBe(0);
    expect(result.stdout.trim()).toBe('vulkan');
  }
});

test('an AMD APU must not silently select HIP over a different discrete GPU', () => {
  for (const vendors of [
    ['0x1002', '0x8086'],
    ['0x10de', '0x1002'],
  ]) {
    const result = select(vendors);
    expect(result.status).not.toBe(0);
    expect(result.stdout).toBe('');
    expect(result.stderr).toContain('ZINC_BACKEND=rocm');
    expect(result.stderr).toContain('ZINC_BACKEND=vulkan');
  }
});

test('explicit backend selection and rollback bypass mixed-GPU auto-detection', () => {
  const root = mkdtempSync(join(tmpdir(), 'zinc-explicit-gpu-'));
  try {
    const bin = join(root, 'bin');
    mkdirSync(bin);
    writeFileSync(
      join(bin, 'uname'),
      '#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo x86_64;; esac\n',
      { mode: 0o755 },
    );
    for (const [card, vendor] of [
      ['card0', '0x1002'],
      ['card1', '0x8086'],
    ]) {
      const device = join(root, 'drm', card, 'device');
      mkdirSync(device, { recursive: true });
      writeFileSync(join(device, 'vendor'), vendor + '\n');
    }
    for (const backend of ['rocm', 'vulkan', '']) {
      const rollback = backend === '';
      const result = spawnSync(
        'bash',
        [join(import.meta.dir, '../scripts/install.sh'), ...(rollback ? ['--rollback'] : [])],
        {
          env: {
            ...process.env,
            PATH: bin + ':' + process.env.PATH,
            ZINC_BACKEND: backend,
            ZINC_DRM_SYSFS_ROOT: join(root, 'drm'),
            ZINC_INSTALL_DIR: join(root, 'install'),
            ZINC_BIN_DIR: join(root, 'commands'),
            ZINC_VERSION: 'invalid',
          },
          encoding: 'utf8',
        },
      );
      expect(result.status).not.toBe(0);
      expect(result.stderr).not.toContain('Multiple GPU vendors');
      expect(result.stderr).toContain(
        rollback ? 'no previous installation' : 'invalid release version',
      );
    }
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
