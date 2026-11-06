import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { chromium } from 'playwright';

const dist = resolve(import.meta.dir, '../dist');
const server = Bun.serve({
  hostname: '127.0.0.1',
  port: 0,
  async fetch(request) {
    const pathname = decodeURIComponent(new URL(request.url).pathname);
    const filePath = resolve(dist, `.${pathname}${pathname.endsWith('/') ? 'index.html' : ''}`);
    if (!filePath.startsWith(`${dist}${sep}`)) return new Response('Not found', { status: 404 });
    const file = Bun.file(filePath);
    return (await file.exists()) ? new Response(file) : new Response('Not found', { status: 404 });
  },
});
const base = `http://127.0.0.1:${server.port}`;
const posts = [
  '2026-03-26-building-a-local-ai-rig',
  '2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm',
];
let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined;
let views = 0;
try {
  browser = await chromium.launch();
  for (const javaScriptEnabled of [false, true]) {
    for (const width of [390, 1440]) {
      const context = await browser.newContext({
        viewport: { width, height: 960 },
        javaScriptEnabled,
      });
      await context.route('**/*', (route) =>
        new URL(route.request().url()).origin === base ? route.continue() : route.abort(),
      );
      const page = await context.newPage();
      for (const slug of posts) {
        const response = await page.goto(`${base}/blog/${slug}/`);
        assert.equal(response?.status(), 200);
        const images = page.locator('.post-content img');
        assert.ok((await images.count()) > 0);
        await images.evaluateAll(async (elements) => {
          const images = elements as HTMLImageElement[];
          images.forEach((image) => (image.loading = 'eager'));
          await Promise.all(images.map((image) => image.decode()));
        });
        const state = await page.evaluate(() => ({
          overflow: document.documentElement.scrollWidth > innerWidth + 1,
          emptyPairs: [...document.querySelectorAll('.img-pair')].filter(
            (pair) => !pair.querySelector('img'),
          ).length,
        }));
        assert.equal(state.overflow, false, `${slug} at ${width}px must not widen the page`);
        assert.equal(state.emptyPairs, 0, `${slug} must not retain empty photo galleries`);
        if (slug === posts[0]) {
          assert.match(await page.locator('.post-content table').innerText(), /9800X3D/);
          const table = page.locator('.post-content table');
          const scrolls = await table.evaluate((element) => {
            if (element.scrollWidth <= element.clientWidth) return true;
            element.scrollLeft = element.scrollWidth;
            return element.scrollLeft > 0;
          });
          assert.equal(scrolls, true, 'wide table content must remain reachable');
        }
        if (process.env.POST_SCREENSHOTS) {
          await mkdir(process.env.POST_SCREENSHOTS, { recursive: true });
          await page.screenshot({
            path: resolve(
              process.env.POST_SCREENSHOTS,
              `${slug}-${width}-${javaScriptEnabled}.png`,
            ),
            fullPage: true,
          });
        }
        views++;
      }
      await context.close();
    }
  }
  console.log(
    `Articles verified: ${views} desktop/mobile and JavaScript/no-JavaScript views, working images and contained tables.`,
  );
} finally {
  await browser?.close();
  server.stop(true);
}
