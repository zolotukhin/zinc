import { expect, test } from 'bun:test';
import { checkHocdbSocialPreview, checkHtmlIndexability } from '../../tools/check-indexability';

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

const socialImage = 'https://zolotukhin.ai/hocdb/social-card-v1.png';
const socialHead = `<head>${Object.entries({
  'og:image': socialImage,
  'og:image:secure_url': socialImage,
  'og:image:type': 'image/png',
  'og:image:width': '1200',
  'og:image:height': '630',
  'og:image:alt': 'HOCDB — time-series storage for apps and agents.',
  'twitter:card': 'summary_large_image',
  'twitter:image': socialImage,
  'twitter:image:src': socialImage,
  'twitter:image:alt': 'HOCDB — time-series storage for apps and agents.',
})
  .map(
    ([key, value]) =>
      `<meta ${key.startsWith('og:') ? 'property' : 'name'}="${key}" content="${value}">`,
  )
  .join('')}</head>`;
const socialPng = () =>
  Bun.file(new URL('../../public/hocdb/social-card-v1.png', import.meta.url)).bytes();

test('HOCDB social cards use the dedicated PNG with matching dimensions and accessible descriptions', async () => {
  await checkHocdbSocialPreview(socialHead, await socialPng());
});

test('rejects fallback images, inconsistent cards, and duplicate social metadata', async () => {
  const png = await socialPng();
  await expect(
    checkHocdbSocialPreview(
      socialHead.replace(socialImage, 'https://zolotukhin.ai/og-image-v2.png'),
      png,
    ),
  ).rejects.toThrow('og:image');
  await expect(
    checkHocdbSocialPreview(socialHead.replace('content="630"', 'content="600"'), png),
  ).rejects.toThrow('og:image:height');
  await expect(
    checkHocdbSocialPreview(
      socialHead.replace('</head>', `<meta property="og:image" content="${socialImage}"></head>`),
      png,
    ),
  ).rejects.toThrow('og:image');
});

test('rejects HTML error pages and PNG dimensions that disagree with social metadata', async () => {
  await expect(
    checkHocdbSocialPreview(socialHead, new TextEncoder().encode('<html>Image not found</html>')),
  ).rejects.toThrow('PNG signature');
  const wrongSize = Buffer.from(await socialPng());
  wrongSize.writeUInt32BE(768, 16);
  await expect(checkHocdbSocialPreview(socialHead, wrongSize)).rejects.toThrow('PNG width');
});
