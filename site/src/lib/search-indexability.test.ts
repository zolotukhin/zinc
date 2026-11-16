import { expect, test } from 'bun:test';
import { checkHtmlIndexability } from '../../tools/check-indexability';

const canonical = 'https://zolotukhin.ai/zinc/docs/api/';
const page = `<head><title>ZINC API</title><link rel="canonical" href="${canonical}"></head><main><h1>API</h1></main>`;

test('accepts crawlable content with a canonical URL and no dependence on JavaScript', async () => {
  expect(await checkHtmlIndexability(page, canonical)).toEqual([]);
  const code = '<pre>&lt;meta name="robots" content="noindex"&gt;</pre>';
  expect(await checkHtmlIndexability(page + code, canonical)).toEqual([]);
});

test('detects accidental indexing exclusions in Googlebot, general meta, and HTTP headers', async () => {
  for (const name of ['robots', 'googlebot']) {
    const errors = await checkHtmlIndexability(
      page + `<meta name="${name}" content="noindex, follow">`,
      canonical,
    );
    expect(errors.join(' ')).toContain('noindex');
  }
  expect((await checkHtmlIndexability(page, canonical, 'googlebot: noindex')).join(' ')).toContain(
    'noindex',
  );
});

test('rejects a redirecting canonical variant and contradictory duplicate canonicals', async () => {
  expect((await checkHtmlIndexability(page, canonical.slice(0, -1))).join(' ')).toContain(
    'canonical',
  );
  expect(
    (
      await checkHtmlIndexability(page + `<link rel="canonical" href="${canonical}">`, canonical)
    ).join(' '),
  ).toContain('canonical');
});

test('detects empty page shells that still return HTTP 200', async () => {
  const shell = `<link rel="canonical" href="${canonical}"><script>renderArticle()</script>`;
  const errors = await checkHtmlIndexability(shell, canonical);
  expect(errors.join(' ')).toContain('title');
  expect(errors.join(' ')).toContain('heading');
});
