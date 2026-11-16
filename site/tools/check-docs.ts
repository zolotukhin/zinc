import assert from 'node:assert/strict';
import { resolve, sep } from 'node:path';
import { chromium } from 'playwright';
import axe from 'axe-core';
import { docNavSections, mergedDocs } from '../src/lib/docs-navigation';
import { docHeadingAliases } from '../src/lib/docs-heading-aliases';

// Verify the published user journey and retained inbound links without inference.
const dist = resolve(import.meta.dir, '../dist');
const redirects = (await Bun.file(resolve(dist, '_redirects')).text())
  .split('\n')
  .filter((line) => line.trim() && !line.startsWith('#'))
  .map((line) => line.trim().split(/\s+/));
const search = await Bun.file(resolve(dist, 'zinc/docs/search')).json();
for (const doc of mergedDocs) {
  const target = `/zinc/docs/${doc.target}/`;
  for (const suffix of ['', '/']) {
    const rule = redirects.find(([from]) => from === `/zinc/docs/${doc.slug}${suffix}`);
    assert.deepEqual(rule, [`/zinc/docs/${doc.slug}${suffix}`, target, '301']);
  }
  assert.equal(await Bun.file(resolve(dist, `zinc/docs/${doc.slug}/index.html`)).exists(), false);
  assert.equal(
    search.entries.some((entry: { url: string }) => entry.url.includes(`/docs/${doc.slug}/`)),
    false,
  );
}
// Every retained section link must exist, and an alias must not collide with a
// new heading. This protects external bookmarks that ordinary link scans miss.
for (const [slug, headings] of Object.entries(docHeadingAliases)) {
  const html = await Bun.file(resolve(dist, `zinc/docs/${slug}/index.html`)).text();
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(ids.length, new Set(ids).size, `${slug}: duplicate anchor`);
  for (const [canonical, aliases] of Object.entries(headings)) {
    for (const id of [canonical, ...aliases]) assert.ok(ids.includes(id), `${slug}#${id}`);
  }
}

const server = Bun.serve({
  hostname: '127.0.0.1',
  port: 0,
  async fetch(request) {
    const pathname = decodeURIComponent(new URL(request.url).pathname);
    const filename = resolve(dist, `.${pathname}${pathname.endsWith('/') ? 'index.html' : ''}`);
    if (!filename.startsWith(dist + sep)) return new Response('', { status: 404 });
    const file = Bun.file(filename);
    return (await file.exists()) ? new Response(file) : new Response('', { status: 404 });
  },
});
const base = `http://127.0.0.1:${server.port}`;
const browser = await chromium.launch();
let views = 0;
try {
  for (const width of [320, 1440]) {
    for (const javaScriptEnabled of [false, true]) {
      const context = await browser.newContext({
        viewport: { width, height: 960 },
        javaScriptEnabled,
        reducedMotion: 'reduce',
      });
      await context.route('**/*', (route) =>
        new URL(route.request().url()).origin === base ? route.continue() : route.abort(),
      );
      const page = await context.newPage();
      const errors: string[] = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.goto(`${base}/zinc/docs/`);
      assert.equal(await page.locator('[data-docs-search-input]').isVisible(), javaScriptEnabled);
      assert.equal(await page.locator('[data-docs-search-clear]').isVisible(), false);
      const start = page.locator('section[aria-labelledby="start-heading"]');
      assert.equal(await start.locator('a').count(), 2);
      // Follow the actual start link, then the next-step link in the guide.
      await start.locator('a[href="/zinc/docs/getting-started/"]').click();
      await page.waitForURL(`${base}/zinc/docs/getting-started/`);
      assert.equal(await page.locator('[data-doc-channel="stable"]').count(), 1);
      await page.locator('.doc-content a[href="/zinc/docs/running-zinc/"]').first().click();
      await page.waitForURL(`${base}/zinc/docs/running-zinc/`);
      for (const slug of [
        'getting-started',
        'running-zinc',
        'openclaw',
        'opencode',
        'api',
        'hardware-requirements',
        'rocm',
        'development',
      ]) {
        const response = await page.goto(`${base}/zinc/docs/${slug}/`);
        assert.equal(response?.status(), 200);
        assert.equal(await page.locator('.doc-content h1').count(), 1);
        if (width === 320) await page.locator('label[for="doc-nav-toggle"]').click();
        const nav = page.locator('.doc-nav-panel');
        assert.equal(
          await nav.locator('a[aria-current="page"]').getAttribute('href'),
          `/zinc/docs/${slug}/`,
        );
        for (const group of docNavSections.filter((group) => group.id !== 'engineering')) {
          for (const item of group.items)
            assert.equal(await nav.locator(`a[href="/zinc/docs/${item.slug}/"]`).isVisible(), true);
        }
        assert.equal(
          await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
          true,
          `${slug} fits ${width}px, JS=${javaScriptEnabled}`,
        );
        if (javaScriptEnabled && ['getting-started', 'running-zinc'].includes(slug)) {
          await page.addScriptTag({ content: axe.source });
          const violations = await page.evaluate(async () => {
            const result = await (window as any).axe.run(document, {
              runOnly: ['wcag2a', 'wcag2aa', 'wcag21aa'],
            });
            return result.violations
              .filter((v: any) => ['serious', 'critical'].includes(v.impact))
              .map((v: any) => v.id);
          });
          assert.deepEqual(violations, [], `${slug} accessibility at ${width}px`);
        }
        views++;
      }
      if (javaScriptEnabled) {
        await page.goto(`${base}/zinc/docs/`);
        await page.locator('[data-docs-search-input]').fill('upgrade roll back');
        const result = page.locator(
          '[data-docs-search-results] a[href="/zinc/docs/getting-started/#upgrade-or-roll-back"]',
        );
        await result.waitFor({ state: 'visible' });
        assert.equal(await page.locator('[data-docs-search-clear]').isVisible(), true);
        await result.click();
        await page.waitForURL(`${base}/zinc/docs/getting-started/#upgrade-or-roll-back`);
        assert.equal(await page.locator('#upgrade-or-roll-back').count(), 1);
      }
      assert.deepEqual(errors, []);
      await context.close();
    }
  }
  console.log(
    `Docs verified: ${views} mobile/desktop and JavaScript/no-JavaScript views; task navigation, search, accessibility, merged routes and retained anchors.`,
  );
} finally {
  await browser.close();
  server.stop(true);
}
