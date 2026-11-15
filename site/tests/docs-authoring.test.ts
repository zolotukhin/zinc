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

function text(node: any): string {
  return node.value ?? (node.children ?? []).map(text).join('');
}

function consumerShellExamples(markdown: string): string[] {
  const examples: string[] = [];
  function visit(parent: any, inherited: string[] = []) {
    const headings = [...inherited];
    for (const node of parent.children ?? []) {
      if (node.type === 'heading') {
        headings.length = node.depth - 1;
        headings.push(text(node));
      }
      if (node.type === 'code' && /^(bash|sh|shell|zsh)$/.test(node.lang ?? '')) {
        // Build outputs remain correct in contributor workflows and dated
        // captures. Installed usage must also work outside a ZINC checkout.
        const source =
          /\b(build|source|development|historical|checkout|benchmark)\b/i.test(
            headings.join(' '),
          ) || /scripts\/zig\.sh build/.test(node.value);
        if (!source) examples.push(node.value);
      }
      if (node.type !== 'heading') visit(node, headings);
    }
  }
  visit(parser.parse(markdown));
  return examples;
}

describe('installed command documentation', () => {
  test('distinguishes source-build commands from installed usage without rewriting captures', () => {
    const markdown =
      '## Run\n```bash\nzinc --check\n```\n' +
      '## Build from source\n```bash\n./zig-out/bin/zinc --check\n```\n' +
      '## History\n### Historical development diagnostics\n```bash\n./zig-out/bin/zinc --check\n```\n' +
      '## Models\n```bash\nzinc model list\n```\n';
    expect(consumerShellExamples(markdown)).toEqual(['zinc --check', 'zinc model list']);
  });

  test('includes nested examples without leaking a quoted source heading into later usage', () => {
    const markdown =
      '## Run\n> ### Build from source\n> ```bash\n> ./zig-out/bin/zinc --check\n> ```\n\n' +
      '- First step:\n\n  ```bash\n  zinc model list\n  ```\n\n' +
      '```bash\nzinc --check\n```\n';
    expect(consumerShellExamples(markdown)).toEqual(['zinc model list', 'zinc --check']);
  });

  test('legacy compiler commands stay visible to the installed-usage guard', () => {
    const command = 'ROCM_PATH=/opt/rocm zig build -Dbackend=rocm';
    expect(consumerShellExamples('## Run\n```bash\n' + command + '\nzinc --check\n```')).toEqual([
      command + '\nzinc --check',
    ]);
  });

  test('consumer examples use zinc and downloaded helpers instead of a checkout', async () => {
    const files = [
      'README.md',
      'docs/GETTING_STARTED.md',
      'docs/BINARY_RELEASES.md',
      'docs/RUNNING_ZINC.md',
      'docs/API.md',
      'docs/AGENT_SETUP.md',
      'docs/HARDWARE_REQUIREMENTS.md',
      'docs/ROCM.md',
      'docs/OPENCLAW.md',
      'docs/OPENCODE.md',
    ];
    for (const file of files) {
      const examples = consumerShellExamples(
        await readFile(resolve(import.meta.dir, '../..', file), 'utf8'),
      );
      expect(examples.length, file).toBeGreaterThan(0);
      for (const example of examples) {
        expect(example, file).not.toMatch(
          /zig-out\/bin\/zinc|(?:node|bun) (?:\.\/)?tools\/setup_|(?:zig|scripts\/zig\.sh)\s+build/,
        );
      }
    }
  });

  test('the first-model flow installs a release before the optional source instructions', async () => {
    const guide = await readFile(resolve(import.meta.dir, '../../docs/GETTING_STARTED.md'), 'utf8');
    const [usage, source] = guide.split('## Build from source');
    expect(usage).toContain('ZINC_VERSION=v0.1.0 bash install.sh');
    expect(usage).toContain('zinc --check');
    expect(usage).not.toMatch(/git clone|scripts\/zig\.sh build|zig-out\/bin\/zinc/);
    expect(source).toContain('./scripts/zig.sh build -Doptimize=ReleaseFast');
    expect(source).toContain('./zig-out/bin/zinc');
  });
});

test('stable examples do not silently require development-only CLI and discovery features', async () => {
  for (const file of ['README.md', 'docs/AGENT_SETUP.md', 'docs/RUNNING_ZINC.md', 'docs/API.md']) {
    const markdown = await readFile(resolve(import.meta.dir, '../..', file), 'utf8');
    for (const example of consumerShellExamples(markdown)) {
      expect(example, file).not.toMatch(
        /\bzinc serve\b|--no-browser|\bzinc (?:--version|--check) --json|\/v1\/capabilities/,
      );
    }
  }
});

test('shared harness introductions do not classify OpenClaw as a coding-only client', async () => {
  const root = resolve(import.meta.dir, '../..');
  const readme = await readFile(resolve(root, 'README.md'), 'utf8');
  const page = await readFile(resolve(root, 'site/src/pages/zinc/index.astro'), 'utf8');
  expect(readme).toContain('## Connect harnesses');
  expect(page).toContain('Connect harnesses');
  expect(readme).not.toContain('## Coding harnesses');
  expect(page).not.toContain('>Coding harnesses<');
  expect(await readFile(resolve(root, 'docs/OPENCLAW.md'), 'utf8')).toStartWith(
    '# OpenClaw with ZINC\n\nConnect OpenClaw to a model served by ZINC.',
  );
});
