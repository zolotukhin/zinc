import { describe, expect, it } from 'bun:test';
import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const postsDirectory = resolve(import.meta.dir, '../content/posts');

function nonCanonicalPageUrls(source: string): string[] {
  const urls =
    source.match(/https:\/\/zolotukhin\.ai\/(?:blog|topics|zinc(?:\/docs)?)\/[^\s)>'"]+/g) ?? [];
  return urls.filter((url) => {
    const pathname = new URL(url).pathname;
    // These generated downloads are files, including when pasted into a shell
    // example. Adding a slash would break the copyable setup command.
    const harnessDownload = /^\/zinc\/harnesses\/setup-(?:opencode|openclaw)\.mjs$/.test(pathname);
    return !harnessDownload && !pathname.endsWith('/');
  });
}

describe('crawlable internal links', () => {
  it('links directly to canonical trailing-slash URLs', () => {
    const offenders: string[] = [];

    for (const filename of readdirSync(postsDirectory).filter((entry) => entry.endsWith('.md'))) {
      const source = readFileSync(resolve(postsDirectory, filename), 'utf8');
      for (const url of nonCanonicalPageUrls(source)) offenders.push(`${filename}: ${url}`);
    }

    expect(offenders).toEqual([]);
  });

  it('keeps exact harness downloads while rejecting noncanonical page links', () => {
    const pages = [
      'https://zolotukhin.ai/blog/article',
      'https://zolotukhin.ai/topics/rdna4',
      'https://zolotukhin.ai/zinc/docs/openclaw',
      'https://zolotukhin.ai/zinc/benchmarks',
    ];
    const downloads = [
      'https://zolotukhin.ai/zinc/harnesses/setup-openclaw.mjs',
      'https://zolotukhin.ai/zinc/harnesses/setup-opencode.mjs',
    ];
    expect(
      nonCanonicalPageUrls([...downloads, ...pages, ...pages.map((url) => url + '/')].join('\n')),
    ).toEqual(pages);
    // The exception must not swallow a whole subtree of malformed page links.
    expect(nonCanonicalPageUrls('https://zolotukhin.ai/zinc/harnesses/missing-page')).toEqual([
      'https://zolotukhin.ai/zinc/harnesses/missing-page',
    ]);
  });
});
