import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { chromium } from 'playwright';
import axe from 'axe-core';
import data from '../src/data/zinc-performance.json';
import {
  aggregate,
  formatNumber,
  formatRatio,
  hasMtpVariants,
  overallRatio,
  rowsFor,
  scenarioOrder,
} from '../src/lib/benchmark-view';

// Run against a dev or production-preview server. External services are blocked
// so this verifies the page and its local resources without analytics/network noise.
const dist = resolve(import.meta.dir, '../dist');
const server = process.env.BENCHMARK_BASE_URL
  ? null
  : Bun.serve({
      hostname: '127.0.0.1',
      port: 0,
      async fetch(request) {
        const pathname = decodeURIComponent(new URL(request.url).pathname);
        const filePath = resolve(dist, `.${pathname}${pathname.endsWith('/') ? 'index.html' : ''}`);
        if (!filePath.startsWith(`${dist}${sep}`))
          return new Response('Not found', { status: 404 });
        const file = Bun.file(filePath);
        return (await file.exists())
          ? new Response(file)
          : new Response('Not found', { status: 404 });
      },
    });
const base = process.env.BENCHMARK_BASE_URL ?? `http://127.0.0.1:${server!.port}`;
const screenshots = process.env.BENCHMARK_SCREENSHOTS;
const browser = await chromium.launch();
let views = 0;
try {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    permissions: ['clipboard-read', 'clipboard-write'],
  });
  await context.route('**/*', (route) =>
    new URL(route.request().url()).origin === new URL(base).origin
      ? route.continue()
      : route.abort(),
  );
  const page = await context.newPage();
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const response = await page.goto(`${base}/zinc/benchmarks/`);
  assert.equal(response?.status(), 200);
  await page.waitForSelector('[data-benchmarks][data-initialized="true"]');
  assert.equal(await page.locator('[data-benchmarks] h1').textContent(), 'Benchmarks');
  assert.equal(await page.locator('[data-bench-target-panel]:visible').count(), 1);

  const jsonResponse = await context.request.get(`${base}/zinc/benchmarks.json`);
  assert.equal(jsonResponse.status(), 200);
  assert.match(jsonResponse.headers()['content-type'], /application\/json/);
  assert.deepEqual(await jsonResponse.json(), data);
  const scripts = await page.locator('script[type="application/ld+json"]').allTextContents();
  const dataset = scripts
    .flatMap((script) => JSON.parse(script)['@graph'] ?? [])
    .find((item) => item['@type'] === 'Dataset');
  assert.equal(new URL(dataset.distribution.contentUrl).pathname, '/zinc/benchmarks.json');

  for (const target of data.targets) {
    await page.locator(`[data-bench-target-tab="${target.id}"]`).click();
    const panel = page.locator(`[data-bench-target-panel="${target.id}"]`);
    assert.equal(await panel.isVisible(), true);
    assert.match(await panel.locator('.backend-comparison').innerText(), /ZINC:.*llama.cpp:/s);
    for (const mode of hasMtpVariants(target) ? (['off', 'on'] as const) : (['off'] as const)) {
      if (hasMtpVariants(target)) await panel.locator(`[data-mtp-mode="${mode}"]`).click();
      for (const scenarioId of scenarioOrder) {
        const button = panel.locator(`[data-bench-scenario="${scenarioId}"]`);
        if ((await button.count()) === 0) continue;
        await button.click();
        const visible = panel.locator('[data-bench-scenario-panel]:visible');
        assert.equal(await visible.count(), 1);
        const rows = rowsFor(target, scenarioId, mode);
        const summary = aggregate(rows);
        assert.match(
          await visible.locator('[data-summary="prefill"]').innerText(),
          new RegExp(formatNumber(summary.prefill).replace('.', '\\.')),
        );
        assert.match(
          await visible.locator('[data-summary="decode"]').innerText(),
          new RegExp(formatNumber(summary.decode).replace('.', '\\.')),
        );
        assert.equal(
          await visible.locator('[data-summary="overall"]').innerText(),
          summary.overall == null ? '—' : formatRatio(summary.overall),
        );
        for (const row of rows) {
          const card = visible.locator(`[data-bench-model="${row.model.id}"]`);
          for (const phase of ['prefill_tps', 'decode_tps']) {
            for (const engine of ['zinc', 'baseline']) {
              const value = row[engine]?.[phase]?.median ?? row[engine]?.[phase]?.avg;
              assert.equal(
                await card.locator(`[data-phase="${phase}"] [data-engine="${engine}"]`).innerText(),
                value == null ? '—' : value.toFixed(1),
              );
            }
          }
          const overall = overallRatio(row);
          assert.equal(await card.locator('[data-overall]').innerText(), formatRatio(overall));
          assert.equal(
            await card.locator('[data-overall]').getAttribute('class'),
            overall == null ? 'neutral' : overall >= 1 ? 'ahead' : 'behind',
          );
          await card.locator('.row-evidence > summary').click();
          const measured = await card.locator('tbody tr').nth(3).locator('td').first().innerText();
          assert.equal(measured, String(row.zinc?.decode_tps?.samples?.length ?? 'Not recorded'));
          const ms = row.zinc?.ms_per_token?.median ?? row.zinc?.ms_per_token?.avg;
          assert.equal(
            await card.locator('tbody tr').nth(5).locator('td').first().innerText(),
            ms == null ? 'Not recorded' : ms.toFixed(1),
          );
          assert.equal(
            await card.locator('.prompt code').textContent(),
            row.prompt ?? 'Not recorded',
          );
          assert.equal(
            await card.locator('.mtp-badge').count(),
            row.zinc?.speculative_decoding?.enabled ? 1 : 0,
          );
          await card.locator('.row-evidence > summary').click();
        }
        const url = new URL(page.url());
        assert.equal(url.hash, `#${target.id}`);
        assert.equal(url.searchParams.get('workload') ?? 'core', scenarioId);
        assert.equal(url.searchParams.get('mtp') ?? 'off', mode);
        views++;
      }
    }
  }

  // Shared URLs, history, invalid parameters, and the original #rdna links.
  await page.goto(`${base}/zinc/benchmarks/?workload=context-long&mtp=on#rdna`);
  await page.waitForSelector('[data-benchmarks][data-initialized="true"]');
  const amd = page.locator('[data-bench-target-panel="rdna"]');
  assert.equal(await amd.locator('[data-mtp-mode="on"]').getAttribute('aria-pressed'), 'true');
  assert.equal(
    await amd.locator('[data-bench-scenario="context-long"]').getAttribute('aria-pressed'),
    'true',
  );
  await amd.locator('[data-bench-scenario="core"]').click();
  await page.goBack();
  assert.equal(
    await amd.locator('[data-bench-scenario="context-long"]').getAttribute('aria-pressed'),
    'true',
  );
  await amd.locator('[data-bench-share]').click();
  assert.equal(await page.evaluate(() => navigator.clipboard.readText()), page.url());
  await page.goto(`${base}/zinc/benchmarks/?workload=invalid&mtp=invalid#invalid`);
  await page.waitForSelector('[data-benchmarks][data-initialized="true"]');
  assert.equal(await page.locator('[data-bench-target-panel]:visible').count(), 1);
  const firstTab = page.locator('[data-bench-target-tab]').first();
  await firstTab.focus();
  await page.keyboard.press('End');
  assert.equal(
    await page.locator('[data-bench-target-tab]').last().getAttribute('aria-selected'),
    'true',
  );
  await page.keyboard.press('Home');
  await page.keyboard.press('ArrowRight');
  assert.equal(
    await page.locator('[data-bench-target-tab]').nth(1).getAttribute('aria-selected'),
    'true',
  );
  assert.equal(await page.locator('[data-bench-target-tab][tabindex="0"]').count(), 1);

  // Every internal resource link resolves, including the rendered guide.
  const localLinks = await page
    .locator('[data-benchmarks] a[href^="/"]')
    .evaluateAll((links) => [
      ...new Set(links.map((link) => (link as HTMLAnchorElement).getAttribute('href')!)),
    ]);
  for (const href of localLinks)
    assert.equal((await context.request.get(new URL(href, base).href)).status(), 200, href);

  if (screenshots) await mkdir(screenshots, { recursive: true });
  await page.goto(`${base}/zinc/benchmarks/#rdna-rocm`);
  await page.waitForSelector('[data-benchmarks][data-initialized="true"]');
  await page.addScriptTag({ content: axe.source });
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
    for (const width of [1440, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      for (const target of data.targets) {
        await page.locator(`[data-bench-target-tab="${target.id}"]`).click();
        const current = page.locator(`[data-bench-target-panel="${target.id}"]`);
        // Coding review also exercises early-stop and no-comparable-row states.
        await current.locator('[data-bench-scenario="context-medium"]').click();
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth,
        );
        assert.equal(overflow, false, `${target.id} ${width}px ${colorScheme}: page overflows`);
        if (screenshots && target.id === 'rdna-rocm' && [1440, 390].includes(width))
          await page.screenshot({
            path: `${screenshots}/${width}-${colorScheme}.png`,
            fullPage: true,
          });
        const panel = page.locator('[data-bench-target-panel]:visible');
        await panel.locator('.configuration > summary').click();
        await panel.locator('.row-evidence > summary:visible').first().click();
        assert.equal(
          await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),
          false,
          `${target.id} ${width}px ${colorScheme}: expanded evidence overflows`,
        );
        const accessibility = await page.evaluate(async () =>
          (window as any).axe.run(document, {
            runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice'] },
          }),
        );
        assert.deepEqual(
          accessibility.violations.map((issue: any) => ({
            id: issue.id,
            nodes: issue.nodes.map((node: any) => ({
              target: node.target,
              reason: node.failureSummary,
            })),
          })),
          [],
          `${target.id} ${width}px ${colorScheme}: accessibility violations`,
        );
        if (screenshots && target.id === 'rdna-rocm' && width === 390)
          await page.screenshot({
            path: `${screenshots}/390-${colorScheme}-evidence.png`,
            fullPage: true,
          });
        await panel.locator('.configuration > summary').click();
        await panel.locator('.row-evidence > summary:visible').first().click();
      }
    }
  }
  assert.deepEqual(errors, []);

  const noJs = await browser.newContext({ javaScriptEnabled: false });
  const plain = await noJs.newPage();
  await plain.goto(`${base}/zinc/benchmarks/`);
  assert.equal(
    await plain.locator('[data-bench-target-panel]:visible').count(),
    data.targets.length,
  );
  assert.equal(await plain.locator('noscript .notice').isVisible(), true);
  await noJs.close();
  console.log(
    `Benchmarks verified: ${views} workload/MTP views, exact fixture data, source details, links, sharing, keyboard controls, no-JS content, and 50 responsive/theme/accessibility layouts.`,
  );
} finally {
  await browser.close();
  server?.stop(true);
}
