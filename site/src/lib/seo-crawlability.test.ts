import { describe, expect, it } from 'bun:test';
import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const postsDirectory = resolve(import.meta.dir, '../content/posts');

describe('crawlable internal links', () => {
  it('links directly to canonical trailing-slash URLs', () => {
    const offenders: string[] = [];

    for (const filename of readdirSync(postsDirectory).filter((entry) => entry.endsWith('.md'))) {
      const source = readFileSync(resolve(postsDirectory, filename), 'utf8');
      const internalUrls = source.match(/https:\/\/zolotukhin\.ai\/(?:blog|topics|zinc(?:\/docs)?)\/[^\s)>'"]+/g) ?? [];

      for (const url of internalUrls) {
        const pathname = new URL(url).pathname;
        if (!pathname.endsWith('/')) offenders.push(`${filename}: ${url}`);
      }
    }

    expect(offenders).toEqual([]);
  });
});
