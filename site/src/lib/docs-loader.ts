import { readdir, readFile, stat } from 'node:fs/promises';
import { join, basename, dirname } from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export interface DocPage {
  slug: string;
  title: string;
  content: string;
  lastmod: string;
  sourcePath: string;
}

// Resolve relative to this file, not cwd — works on Cloudflare Pages where cwd is site/
const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..', '..', '..');
const DOCS_DIR = join(REPO_ROOT, 'docs');

function slugFromFilename(filename: string): string {
  return basename(filename, '.md').toLowerCase().replace(/_/g, '-');
}

function extractTitle(content: string): string {
  const match = content.match(/^#\s+(.+)$/m);
  return match ? match[1] : 'Untitled';
}

function getGitLastmod(filepath: string): string {
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
    const { mtimeMs } = require('node:fs').statSync(filepath);
    return new Date(mtimeMs).toISOString().split('T')[0];
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
      lastmod: getGitLastmod(filepath),
      sourcePath: `docs/${file}`,
    });
  }

  return docs.sort((a, b) => a.slug.localeCompare(b.slug));
}
