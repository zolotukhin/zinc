// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { execFileSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const siteRoot = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = resolve(siteRoot, '..');
const postDirectory = resolve(siteRoot, 'src/content/posts');
const docsDirectory = resolve(repoRoot, 'docs');

function gitLastmod(relativeSourcePath, fallback) {
  try {
    const value = execFileSync(
      'git',
      ['log', '-1', '--format=%aI', '--', relativeSourcePath],
      { cwd: repoRoot, encoding: 'utf8' },
    ).trim();

    if (value) return value;
  } catch {
    // Source archives and shallow checkouts may not contain git history.
  }

  return fallback;
}

const postLastmods = new Map(
  readdirSync(postDirectory)
    .filter((filename) => filename.endsWith('.md'))
    .map((filename) => {
      const slug = filename.slice(0, -3);
      const datedSlug = slug.match(/^(\d{4}-\d{2}-\d{2})-/);
      const fallback = datedSlug?.[1] ?? '2026-03-25';
      const relativeSourcePath = `site/src/content/posts/${filename}`;
      return [slug, gitLastmod(relativeSourcePath, fallback)];
    }),
);

const latestPostLastmod = [...postLastmods.values()].sort().at(-1);
const docLastmods = new Map(
  readdirSync(docsDirectory)
    .filter((filename) => filename.endsWith('.md'))
    .map((filename) => {
      const slug = filename.slice(0, -3).toLowerCase().replaceAll('_', '-');
      return [slug, gitLastmod(`docs/${filename}`, undefined)];
    }),
);
const latestDocLastmod = [...docLastmods.values()].filter(Boolean).sort().at(-1);

export default defineConfig({
  site: 'https://zolotukhin.ai',
  integrations: [
    mdx(),
    sitemap({
      serialize(item) {
        const pathname = new URL(item.url).pathname;
        const postMatch = pathname.match(/^\/blog\/([^/]+)\/$/);

        if (postMatch) {
          return { ...item, lastmod: postLastmods.get(postMatch[1]) };
        }

        const docMatch = pathname.match(/^\/zinc\/docs\/([^/]+)\/$/);
        if (docMatch && docLastmods.has(docMatch[1])) {
          return { ...item, lastmod: docLastmods.get(docMatch[1]) };
        }

        if (latestDocLastmod && pathname === '/zinc/docs/') {
          return { ...item, lastmod: latestDocLastmod };
        }

        if (latestPostLastmod && (pathname === '/' || pathname === '/blog/' || pathname.startsWith('/topics/'))) {
          return { ...item, lastmod: latestPostLastmod };
        }

        return item;
      },
    }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      defaultColor: 'light',
      themes: {
        light: 'github-light',
        dark: 'vitesse-dark',
      },
      langs: ['zig', 'glsl', 'json', 'bash', 'typescript', 'c'],
    },
  },
});
