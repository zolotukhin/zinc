// Run from site/: bun tools/render-hocdb-social.ts
// Keep the editable SVG beside the PNG that social crawlers actually fetch.
// Review the regenerated PNG: browser font fallbacks can differ by platform.
import { chromium } from 'playwright';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const source = resolve(import.meta.dir, '../public/hocdb/social-card-v1.svg');
const output = source.replace(/\.svg$/, '.png');
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({
    viewport: { width: 1200, height: 630 },
    deviceScaleFactor: 1,
  });
  // The source has no remote images, scripts, fonts, or other dependencies.
  await page.route(/^https?:/, (route) => route.abort());
  await page.goto(pathToFileURL(source).href);
  await page.evaluate(() => document.fonts.ready);
  await page.locator('svg').screenshot({ path: output, type: 'png' });
  console.log(`Rendered ${output}`);
} finally {
  await browser.close();
}
