// Run in site-check.yml after installing site dependencies. Core Bun filters
// include any tools/ or site/src/ path, so this test lives in site/tests/.
import { describe, expect, test } from 'bun:test';
import { readdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkRehype from 'remark-rehype';
import rehypeStringify from 'rehype-stringify';

const parser = unified().use(remarkParse).use(remarkGfm);
const renderer = unified().use(remarkParse).use(remarkGfm).use(remarkRehype).use(rehypeStringify);

function rawTables(node: any): boolean {
  return (
    (node.type === 'html' && /<table\b/i.test(node.value ?? '')) ||
    (node.children ?? []).some(rawTables)
  );
}

describe('published guide table authoring', () => {
  test('uses GFM tables because the docs pipeline omits raw HTML', async () => {
    const markdown = '| Model | Result |\n| --- | --- |\n| fixture | pass |\n';
    expect(String(await renderer.process(markdown))).toContain('<td>fixture</td>');
    const html = '<table><tr><td>hidden fixture</td></tr></table>';
    expect(rawTables(parser.parse(html))).toBe(true);
    expect(String(await renderer.process(html))).not.toContain('hidden fixture');
    // Code examples may describe HTML without being authored HTML tables.
    expect(rawTables(parser.parse('```html\n' + html + '\n```'))).toBe(false);
  });

  test('does not silently drop an authored comparison table from any guide', async () => {
    const directory = resolve(import.meta.dir, '../../docs');
    const unsupported: string[] = [];
    for (const name of await readdir(directory)) {
      if (!name.endsWith('.md')) continue;
      if (rawTables(parser.parse(await readFile(resolve(directory, name), 'utf8'))))
        unsupported.push(name);
    }
    // A raw <table> previously hid the entire Intel client-results matrix.
    // Posts use Astro's different renderer; this contract covers guides only.
    expect(unsupported).toEqual([]);
  });
});
