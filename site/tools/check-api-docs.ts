import assert from 'node:assert/strict';
import { resolve, sep } from 'node:path';
import { chromium } from 'playwright';
import axe from 'axe-core';

// Exercise generated signatures with a keyboard, including narrow screens
// where the full public API requires horizontal scrolling.
const dist = resolve(process.env.API_DOCS_DIST ?? resolve(import.meta.dir, '../dist'));
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
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } });
    await context.route('**/*', (route) =>
      new URL(route.request().url()).origin === base ? route.continue() : route.abort(),
    );
    const page = await context.newPage();
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    for (const module of ['matrix-bounds', 'buffer']) {
      const response = await page.goto(`${base}/zinc/docs/zig-api/${module}/`);
      assert.equal(response?.status(), 200);
      await page.evaluate(() => document.fonts.ready);
      const signatures = page.locator('.symbol-card pre');
      assert.ok((await signatures.count()) > 0, `${module} has no signatures`);
      let keyboardScrolls = 0;
      for (const signature of await signatures.all()) {
        assert.equal(await signature.getAttribute('tabindex'), '0');
        assert.equal(await signature.getAttribute('role'), 'region');
        assert.ok(await signature.getAttribute('aria-label'));
        if (
          keyboardScrolls ||
          !(await signature.evaluate((pre) => pre.scrollWidth > pre.clientWidth))
        )
          continue;
        await signature.evaluate((pre) => {
          const card = pre.closest('.member-card, .symbol-card');
          card?.querySelector<HTMLAnchorElement>('.symbol-heading .anchor-link')?.focus();
        });
        for (let tab = 0; tab < 5; tab++) {
          await page.keyboard.press('Tab');
          if (await signature.evaluate((pre) => document.activeElement === pre)) break;
        }
        assert.equal(await signature.evaluate((pre) => document.activeElement === pre), true);
        await signature.evaluate((pre) => {
          pre.scrollLeft = 0;
        });
        await page.keyboard.press('ArrowRight');
        await page.waitForFunction(() => (document.activeElement?.scrollLeft ?? 0) > 0);
        keyboardScrolls++;
      }
      if (width === 390)
        assert.equal(keyboardScrolls, 1, `${module} did not exercise keyboard scrolling`);
      assert.equal(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
        true,
      );
      await page.addScriptTag({ content: axe.source });
      const audit = await page.evaluate(
        async () =>
          await axe.run(document, {
            runOnly: ['wcag2a', 'wcag2aa', 'wcag21aa'],
          }),
      );
      assert.deepEqual(
        audit.violations
          .filter((v) => ['serious', 'critical'].includes(v.impact ?? ''))
          .map((v) => v.id),
        [],
        `${module} accessibility at ${width}px`,
      );
      const headings = await page.evaluate(
        async () => await axe.run(document, { runOnly: ['heading-order'] }),
      );
      assert.deepEqual(
        headings.violations.map((v) => v.id),
        [],
      );
      assert.deepEqual(errors, []);
      console.log(
        JSON.stringify({ module, viewport: width, keyboardScrolls, accessibility: 'pass' }),
      );
    }
    await context.close();
  }
} finally {
  await browser.close();
  server.stop(true);
}
