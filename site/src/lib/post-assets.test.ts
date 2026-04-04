import { describe, expect, it } from 'bun:test';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

function resolveSiteRoot(): string {
  const cwd = process.cwd();
  if (existsSync(join(cwd, 'src', 'content', 'posts')) && existsSync(join(cwd, 'public', 'blog'))) {
    return cwd;
  }

  const nestedSiteRoot = join(cwd, 'site');
  if (existsSync(join(nestedSiteRoot, 'src', 'content', 'posts')) && existsSync(join(nestedSiteRoot, 'public', 'blog'))) {
    return nestedSiteRoot;
  }

  throw new Error(`Unable to resolve site root from ${cwd}`);
}

describe('blog post assets', () => {
  it('keeps every local /blog image reference backed by a public asset', () => {
    const siteRoot = resolveSiteRoot();
    const postsDir = join(siteRoot, 'src', 'content', 'posts');
    const publicBlogDir = join(siteRoot, 'public', 'blog');
    const assetPattern = /\/blog\/([A-Za-z0-9._-]+\.(?:svg|png|jpe?g|gif|webp))/g;

    const missing: string[] = [];

    for (const postFile of readdirSync(postsDir).filter(name => name.endsWith('.md'))) {
      const content = readFileSync(join(postsDir, postFile), 'utf8');
      for (const match of content.matchAll(assetPattern)) {
        const assetName = match[1];
        if (!existsSync(join(publicBlogDir, assetName))) {
          missing.push(`${postFile} -> ${assetName}`);
        }
      }
    }

    expect(missing).toEqual([]);
  });
});
