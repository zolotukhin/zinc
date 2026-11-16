import { expect, test } from 'bun:test';
import canonicalHost from '../../workers/canonical-host.mjs';

test('www requests permanently redirect to HTTPS while preserving paths and query strings', () => {
  for (const protocol of ['http:', 'https:']) {
    const response = canonicalHost.fetch(
      new Request(`${protocol}//www.zolotukhin.ai/zinc/benchmarks/?backend=rdna&scenario=long_context`),
    );
    expect(response.status).toBe(301);
    expect(response.headers.get('location')).toBe(
      'https://zolotukhin.ai/zinc/benchmarks/?backend=rdna&scenario=long_context',
    );
  }
});

test('file URLs and encoded paths are preserved', () => {
  for (const path of ['/feed.xml', '/zinc/harnesses/setup-openclaw.mjs', '/a%20b/?q=%2F']) {
    const response = canonicalHost.fetch(new Request(`https://www.zolotukhin.ai${path}`));
    expect(response.headers.get('location')).toBe(`https://zolotukhin.ai${path}`);
  }
});

test('a misrouted request cannot create an apex redirect loop or redirect another host', () => {
  for (const hostname of ['zolotukhin.ai', 'unrelated.example']) {
    const response = canonicalHost.fetch(new Request(`https://${hostname}/`));
    expect(response.status).toBe(404);
    expect(response.headers.has('location')).toBe(false);
  }
});
