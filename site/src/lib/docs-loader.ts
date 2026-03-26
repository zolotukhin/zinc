import { readdir, readFile, stat } from 'node:fs/promises';
import { join, basename, resolve } from 'node:path';
import { execSync } from 'node:child_process';

export interface DocPage {
  slug: string;
  title: string;
  content: string;
  lastmod: string;
  sourcePath: string;
}

// Astro runs from site/, both locally and in CI/Pages. Use cwd so bundling
// into dist/.prerender does not break repo-root resolution.
const SITE_ROOT = process.cwd();
const REPO_ROOT = resolve(SITE_ROOT, '..');
const DOCS_DIR = join(REPO_ROOT, 'docs');

function slugFromFilename(filename: string): string {
  return basename(filename, '.md').toLowerCase().replace(/_/g, '-');
}

function extractTitle(content: string): string {
  const match = content.match(/^#\s+(.+)$/m);
  return match ? match[1] : 'Untitled';
}

async function getGitLastmod(filepath: string): Promise<string> {
  try {
    const date = execSync(`git log -1 --format=%aI -- "${filepath}"`, {
      encoding: 'utf-8',
      cwd: REPO_ROOT,
    }).trim();
    if (date) return date.split('T')[0];
  } catch {
    // git not available or shallow clone — fall through
  }

  // Fallback: file mtime
  try {
    const { mtime } = await stat(filepath);
    return mtime.toISOString().split('T')[0];
  } catch {
    return new Date().toISOString().split('T')[0];
  }
}

export async function loadDocs(): Promise<DocPage[]> {
  const files = await readdir(DOCS_DIR);
  const mdFiles = files.filter(f => f.endsWith('.md'));

  const docs: DocPage[] = [];

  for (const file of mdFiles) {
    const filepath = join(DOCS_DIR, file);
    const raw = await readFile(filepath, 'utf-8');

    docs.push({
      slug: slugFromFilename(file),
      title: extractTitle(raw),
      content: raw,
      lastmod: await getGitLastmod(filepath),
      sourcePath: `docs/${file}`,
    });
  }

  return docs.sort((a, b) => a.slug.localeCompare(b.slug));
}
