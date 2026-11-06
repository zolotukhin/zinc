import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
// Both packages are already declared and locked in site/package.json and site/bun.lock.
import { chromium } from 'playwright';
import axe from 'axe-core';

// Exercise the built onboarding pages without starting a client or loading a model.
const dist = resolve(import.meta.dir, '../dist');
const compilerVersion = (
  await Bun.file(resolve(import.meta.dir, '../../.zig-version')).text()
).trim();
// Loopback-only fixture server for this browser check; not a production server.
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
let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined;
let views = 0;
try {
  browser = await chromium.launch();
  // Downloads must remain the tested source, not a hand-maintained website copy.
  for (const client of ['openclaw', 'opencode']) {
    const download = await fetch(`${base}/zinc/harnesses/setup-${client}.mjs`);
    assert.equal(download.status, 200);
    assert.equal(
      await download.text(),
      await Bun.file(resolve(import.meta.dir, `../../tools/setup_${client}.mjs`)).text(),
    );
  }
  const evidencePath = '/zinc/validation/openclaw-amd-2026-11-05.json';
  const evidence = await fetch(`${base}${evidencePath}`);
  assert.equal(evidence.status, 200);
  assert.deepEqual(
    await evidence.json(),
    await Bun.file(resolve(import.meta.dir, `../public${evidencePath}`)).json(),
  );

  for (const javaScriptEnabled of [false, true]) {
    for (const width of [320, 390, 1440]) {
      for (const colorScheme of ['light', 'dark'] as const) {
        const context = await browser.newContext({
          viewport: { width, height: 960 },
          colorScheme,
          javaScriptEnabled,
          permissions: ['clipboard-read', 'clipboard-write'],
          reducedMotion: 'reduce',
        });
        await context.route('**/*', (route) =>
          new URL(route.request().url()).origin === base ? route.continue() : route.abort(),
        );
        const page = await context.newPage();
        const errors: string[] = [];
        page.on('pageerror', (error) => errors.push(error.message));
        const response = await page.goto(`${base}/zinc/`);
        assert.equal(response?.status(), 200);
        assert.equal(
          await page.locator('.build-note').innerText(),
          `One native binary · Zig ${compilerVersion} · MIT licensed`,
          'The compiler badge must display the exact repository pin',
        );
        const section = page.locator('[data-coding-harnesses]');
        assert.match(await section.innerText(), /OpenClaw or OpenCode/);
        assert.match(await section.innerText(), /existing project/);
        assert.match(await section.innerText(), /Stop the earlier chat server/);
        assert.match(
          await section.locator('[data-harness-server] code').first().innerText(),
          /-c 16384 --parallel 1 --port 9090/,
        );
        const claw = section.locator('[data-coding-harness="openclaw"]');
        assert.match(await claw.innerText(), /24\.16\+.*24\.x.*26\.1\+/);
        assert.match(await claw.innerText(), /independently run your project's tests/);
        assert.match(await claw.innerText(), /small starter model did not qualify/);
        assert.match(await claw.innerText(), /does not qualify Intel or Metal/);
        assert.match(await claw.locator('pre code').innerText(), /openclaw@2026\.11\.1/);
        assert.match(
          await claw.locator('pre code').innerText(),
          /setup_openclaw\.mjs (setup|check|run)/,
        );
        const code = section.locator('[data-coding-harness="opencode"]');
        assert.match(await code.locator('pre code').innerText(), /--auto --verify 'bun test'/);

        for (const client of ['openclaw', 'opencode']) {
          const card = section.locator(`[data-coding-harness="${client}"]`);
          assert.equal(await card.locator(`a[href="/zinc/docs/${client}/"]`).isVisible(), true);
          const download = card.locator('a[download]');
          assert.equal(await download.getAttribute('download'), `setup-${client}.mjs`);
          assert.equal(await download.getAttribute('href'), `/zinc/harnesses/setup-${client}.mjs`);
        }
        // These are native links: the setup and evidence must also work without JavaScript.
        for (const href of await section
          .locator('a')
          .evaluateAll((links) => links.map((a) => a.getAttribute('href')!))) {
          assert.equal((await context.request.get(`${base}${href}`)).status(), 200, href);
        }
        assert.equal(
          await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1),
          false,
          `Onboarding must fit ${width}px (${colorScheme}, JS=${javaScriptEnabled})`,
        );
        if (javaScriptEnabled) {
          // Copy exactly the displayed shell text, including continuation newlines.
          for (const terminal of await section.locator('[data-terminal-window]').all()) {
            const expected = (await terminal.locator('pre code').textContent())!.trimEnd();
            const copy = terminal.locator('[data-terminal-copy]');
            await copy.focus();
            await page.keyboard.press('Enter');
            await page.waitForFunction(
              async (expected) => (await navigator.clipboard.readText()) === expected,
              expected,
            );
          }
          await page.addScriptTag({ content: axe.source });
          const violations = await page.evaluate(async () => {
            const result = await (window as any).axe.run('[data-coding-harnesses]', {
              runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] },
            });
            return result.violations.map((violation: any) => ({
              id: violation.id,
              targets: violation.nodes.map((node: any) => node.target),
            }));
          });
          assert.deepEqual(
            violations,
            [],
            `Accessible harness instructions at ${width}px (${colorScheme})`,
          );
        }
        if (process.env.HARNESS_SCREENSHOTS && javaScriptEnabled) {
          await mkdir(process.env.HARNESS_SCREENSHOTS, { recursive: true });
          await section.screenshot({
            path: resolve(process.env.HARNESS_SCREENSHOTS, `harnesses-${width}-${colorScheme}.png`),
          });
        }

        await claw.locator('a[href="/zinc/docs/openclaw/"]').click();
        await page.waitForURL(`${base}/zinc/docs/openclaw/`);
        assert.match(await page.locator('.doc-content').innerText(), /Node.js 24\.22\.2/);
        assert.match(
          (await page.locator('meta[name="description"]').getAttribute('content')) ?? '',
          /Pinned client setup/,
        );
        // The mobile docs menu is CSS-driven, so it must work with scripts disabled too.
        if (width < 1440) await page.locator('label[for="doc-nav-toggle"]').click();
        const nav = page.locator('.doc-nav-panel');
        assert.equal(await nav.locator('a[href="/zinc/docs/openclaw/"]').isVisible(), true);
        assert.equal(
          await nav.locator('a[href="/zinc/docs/openclaw/"]').getAttribute('aria-current'),
          'page',
        );
        assert.equal(await nav.locator('a[href="/zinc/docs/opencode/"]').isVisible(), true);
        assert.deepEqual(errors, []);
        views++;
        await context.close();
      }
    }
  }

  // All onboarding entry points must offer both clients without opening reference drawers.
  const page = await browser.newPage();
  await page.route('**/*', (route) =>
    new URL(route.request().url()).origin === base ? route.continue() : route.abort(),
  );
  for (const slug of ['', 'getting-started/', 'api/']) {
    await page.goto(`${base}/zinc/docs/${slug}`);
    const content = page.locator(slug ? '.doc-content' : '.setup-list');
    for (const client of ['openclaw', 'opencode']) {
      assert.equal(
        await content.locator(`a[href="/zinc/docs/${client}/"]`).first().isVisible(),
        true,
      );
    }
  }
  await page.goto(`${base}/zinc/docs/openclaw-amd-validation/`);
  const report = await page.locator('.doc-content').innerText();
  const reportNav = page.locator('.doc-nav-panel a[href="/zinc/docs/openclaw-amd-validation/"]');
  assert.equal(await reportNav.isVisible(), true);
  assert.equal(await reportNav.getAttribute('aria-current'), 'page');
  assert.match(report, /task.failed/i);
  assert.match(report, /unsupported/i);
  assert.match(report, /Intel.*Metal/s);
  console.log(
    `Coding clients verified: ${views} mobile/desktop, light/dark and JavaScript/no-JavaScript views; working navigation, exact downloads, keyboard copy and accessibility.`,
  );
} finally {
  await browser?.close();
  server.stop(true);
}
