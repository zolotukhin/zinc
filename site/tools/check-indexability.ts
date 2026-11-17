import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const site = 'https://zolotukhin.ai';

/** Social crawlers consume the static head and PNG, without running scripts. */
export async function checkHocdbSocialPreview(html: string, png: Uint8Array) {
  const metadata = new Map<string, string[]>();
  await new HTMLRewriter()
    .on('head meta', {
      element(element) {
        const key = element.getAttribute('property') ?? element.getAttribute('name') ?? '';
        metadata.set(key, [...(metadata.get(key) ?? []), element.getAttribute('content') ?? '']);
      },
    })
    .transform(new Response(html))
    .text();

  const image = `${site}/hocdb/social-card-v1.png`;
  for (const [key, value] of Object.entries({
    'og:image': image,
    'og:image:secure_url': image,
    'og:image:type': 'image/png',
    'og:image:width': '1200',
    'og:image:height': '630',
    'twitter:card': 'summary_large_image',
    'twitter:image': image,
    'twitter:image:src': image,
  })) {
    assert.deepEqual(metadata.get(key), [value], `HOCDB social preview: ${key}`);
  }
  const alt = metadata.get('og:image:alt');
  assert.equal(alt?.length, 1, 'HOCDB preview needs one descriptive alt text');
  assert.match(alt![0], /HOCDB.*time-series storage/);
  assert.deepEqual(metadata.get('twitter:image:alt'), alt, 'social preview alt text must agree');

  // Catch a missing/wrong image and metadata that no longer matches its pixels.
  const bytes = Buffer.from(png);
  assert.ok(bytes.length > 24, 'HOCDB preview PNG must not be empty');
  assert.equal(bytes.subarray(0, 8).toString('hex'), '89504e470d0a1a0a', 'PNG signature');
  // PNG requires IHDR first, immediately after the eight-byte signature.
  assert.equal(bytes.subarray(12, 16).toString(), 'IHDR', 'PNG image header');
  assert.equal(bytes.readUInt32BE(16), 1200, 'PNG width must match metadata');
  assert.equal(bytes.readUInt32BE(20), 630, 'PNG height must match metadata');
}

/** Check the original HTML: indexing must not depend on client-side scripts. */
export async function checkHtmlIndexability(html: string, canonical: string, headerRobots = '') {
  const canonicals: string[] = [];
  const robots = [headerRobots];
  let title = '';
  let titles = 0;
  let headings = 0;
  await new HTMLRewriter()
    .on('link[rel="canonical"]', {
      element(element) {
        canonicals.push(element.getAttribute('href') ?? '');
      },
    })
    .on('meta', {
      element(element) {
        if (/^(robots|googlebot)$/i.test(element.getAttribute('name') ?? '')) {
          robots.push(element.getAttribute('content') ?? '');
        }
      },
    })
    .on('title', {
      element() {
        titles++;
      },
      text(chunk) {
        title += chunk.text;
      },
    })
    .on('h1', {
      element() {
        headings++;
      },
    })
    .transform(new Response(html))
    .text();

  const errors: string[] = [];
  if (canonicals.length !== 1 || canonicals[0] !== canonical)
    errors.push('canonical must match the published URL exactly once');
  if (robots.some((value) => /\b(noindex|none)\b/i.test(value)))
    errors.push('noindex prevents this content page from appearing in search');
  if (titles !== 1 || !title.trim()) errors.push('a nonempty title is required');
  if (headings !== 1) errors.push('one primary heading is required');
  return errors;
}

// Cloudflare's path wildcard spans slashes. A broad noindex must never leak
// from download rules onto the articles and documentation that link to them.
function robotHeaders(headers: string, pathname: string): string {
  let matches = false;
  const values: string[] = [];
  for (const line of headers.split('\n')) {
    if (line.startsWith('/')) {
      const pattern = line
        .trim()
        .split('*')
        .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('.*');
      matches = new RegExp(`^${pattern}$`).test(pathname);
    } else if (matches && /^\s+X-Robots-Tag:/i.test(line)) {
      values.push(line.split(':').slice(1).join(':').trim());
    }
  }
  return values.join(', ');
}

function xmlLinks(xml: string, tag: string) {
  return [...xml.matchAll(new RegExp(`<${tag}>([^<]+)</${tag}>`, 'g'))].map((match) => match[1]);
}

export async function checkBuiltIndexability(dist = resolve(import.meta.dir, '../dist')) {
  const read = (path: string) => readFile(resolve(dist, `.${path}`), 'utf8');
  const headers = await read('/_headers');
  const pages = new Set<string>();
  const errors: string[] = [];
  for await (const filename of new Bun.Glob('**/*.html').scan({ cwd: dist, onlyFiles: true })) {
    const html = await read(`/${filename}`);
    if (filename === '404.html') {
      assert.match(
        html,
        /<meta name="robots" content="noindex/,
        'the error page must stay out of search',
      );
      continue;
    }
    const path = `/${filename.replace(/index\.html$/, '')}`;
    const canonical = site + path;
    pages.add(canonical);
    for (const error of await checkHtmlIndexability(html, canonical, robotHeaders(headers, path))) {
      errors.push(`${path}: ${error}`);
    }
  }
  assert.ok(pages.size > 0, 'the built site must contain content pages');

  const sitemap = new Set<string>();
  for (const url of xmlLinks(await read('/sitemap-index.xml'), 'loc')) {
    assert.equal(new URL(url).origin, site, 'sitemap files must belong to this site');
    for (const page of xmlLinks(await read(new URL(url).pathname), 'loc')) {
      if (sitemap.has(page)) errors.push(`duplicate sitemap URL: ${page}`);
      sitemap.add(page);
    }
  }
  for (const url of pages) if (!sitemap.has(url)) errors.push(`missing from sitemap: ${url}`);
  for (const url of sitemap)
    if (!pages.has(url)) errors.push(`sitemap URL is not a canonical HTML page: ${url}`);

  const feedLinks = xmlLinks(await read('/feed.xml'), 'link');
  assert.ok(
    feedLinks.some((url) => new URL(url).pathname.startsWith('/blog/')),
    'RSS must include articles',
  );
  for (const url of feedLinks)
    if (!pages.has(url)) errors.push(`RSS must link directly to a canonical page: ${url}`);

  const downloads = [
    '/zinc/docs/search',
    '/zinc/benchmarks.json',
    '/zinc/docs/zig-api.json',
    '/zinc/docs/zig-api.txt',
    '/llms.txt',
    '/llms-full.txt',
  ];
  for (const pattern of ['zinc/harnesses/*.mjs', 'zinc/validation/*.json']) {
    for await (const filename of new Bun.Glob(pattern).scan({ cwd: dist, onlyFiles: true }))
      downloads.push(`/${filename}`);
  }
  for (const path of downloads) {
    if (!/\bnoindex\b/i.test(robotHeaders(headers, path)))
      errors.push(`download is missing noindex: ${path}`);
  }
  await checkHocdbSocialPreview(
    await read('/hocdb/index.html'),
    await readFile(resolve(dist, 'hocdb/social-card-v1.png')),
  );
  assert.deepEqual(errors, [], 'built pages, sitemap, feed, and download directives must agree');
  return {
    pages: pages.size,
    sitemap: sitemap.size,
    feedLinks: feedLinks.length,
    downloads: downloads.length,
  };
}

if (import.meta.main) console.log('Indexability checks passed:', await checkBuiltIndexability());
