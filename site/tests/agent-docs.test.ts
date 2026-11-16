import { expect, test } from 'bun:test';
import { renderLlmsTxt, renderLlmsFullTxt } from '../src/lib/agent-docs';
import { docsReleaseStatus, STABLE_RELEASE } from '../src/lib/docs-release-status';
import type { ZigApiIndex } from '../src/lib/zig-api-loader';

const index = {
  sections: [],
  moduleCount: 1,
  codeLineCount: 10,
  exportCount: 1,
  memberCount: 1,
  lastmod: '2026-11-15',
} as unknown as ZigApiIndex;

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
    expect(text).toContain('zinc --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090');
    expect(text).toContain('http://localhost:9090/v1/models');
    expect(text).toContain('standalone helper');
    expect(text).toContain(`Stable Linux release: ${STABLE_RELEASE}`);
    expect(text).toContain('Development additions (newer than stable 0.1.0)');
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
