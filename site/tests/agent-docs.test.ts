import { expect, test } from 'bun:test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { renderLlmsTxt, renderLlmsFullTxt } from '../src/lib/agent-docs';
import {
  docsReleaseStatus,
  STABLE_RELEASE,
  STABLE_VALIDATION_PATH,
} from '../src/lib/docs-release-status';
import type { ZigApiIndex } from '../src/lib/zig-api-loader';

const publishedRelease = await Bun.file(
  resolve(import.meta.dir, '../public' + STABLE_VALIDATION_PATH),
).json();

const index = {
  sections: [],
  moduleCount: 1,
  codeLineCount: 10,
  exportCount: 1,
  memberCount: 1,
  lastmod: '2026-11-15',
} as unknown as ZigApiIndex;

test('stable client guidance follows the published maintenance certificate', () => {
  expect(publishedRelease.published).toBe(true);
  expect(publishedRelease.status).toBe('published');
  expect(STABLE_RELEASE).toBe(publishedRelease.version);
  expect(publishedRelease.publication.status).toBe('passed');
  expect(publishedRelease.publication.public_install_jobs).toHaveLength(2);
});

test('active OpenCode install recipes match the final-archive client cohort', async () => {
  const pins = [
    ...new Set(publishedRelease.supplementary.actual_clients.map((row: any) => row.opencode)),
  ];
  expect(pins).toHaveLength(1);
  for (const name of ['README.md', 'docs/OPENCODE.md', 'site/src/pages/zinc/index.astro']) {
    const source = await readFile(resolve(import.meta.dir, '../..', name), 'utf8');
    const recipes = [...source.matchAll(/@opencode\/cli@(\d+\.\d+\.\d+)/g)].map(
      (match) => match[1],
    );
    expect(recipes.length, name).toBeGreaterThan(0);
    expect([...new Set(recipes)], name).toEqual(pins);
  }
});

test('both agent indexes put installed client setup before implementation references', () => {
  for (const render of [renderLlmsTxt, renderLlmsFullTxt]) {
    const text = render(index, [], 'https://example.test/');
    expect(text).toContain('https://example.test/zinc/docs/running-zinc/');
    expect(text).not.toContain('/zinc/docs/agent-setup/');
    expect(text).not.toContain('/zinc/docs/binary-releases/');
    expect(text.indexOf('## Run ZINC or connect a client')).toBeLessThan(
      text.indexOf('ZINC Zig API:') < 0
        ? text.indexOf('Zig API root:')
        : text.indexOf('ZINC Zig API:'),
    );
    expect(text).toContain(
      'zinc serve --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090',
    );
    expect(text).toContain('http://localhost:9090/v1/models');
    expect(text).toContain('standalone helper');
    expect(text).toContain(`Stable Linux release: ${STABLE_RELEASE}`);
    expect(text).toContain('AMD RDNA: prefer HIP / ROCm');
    expect(text).toContain('Intel Arc uses ZINC_BACKEND=vulkan');
    expect(text).toContain('https://example.test' + STABLE_VALIDATION_PATH);
    expect(text).toContain('Structured discovery (0.1.1+)');
    expect(text).toContain('are released features');
    expect(text).not.toContain('Development additions');
    expect(text).toContain('absent capabilities are unknown');
    expect(text).toContain('AGENTS.md');
    expect(text).not.toContain('prefer the generated Zig API reference first');
  }
});

test('release notices distinguish stable installation, mixed guides and current source', () => {
  for (const slug of ['getting-started', 'openclaw', 'opencode']) {
    expect(docsReleaseStatus(slug).channel).toBe('stable');
  }
  for (const slug of ['api', 'running-zinc', 'hardware-requirements', 'rocm']) {
    expect(docsReleaseStatus(slug).channel).toBe('mixed');
  }
  expect(docsReleaseStatus('zig-api/compute-forward').channel).toBe('development');
});
