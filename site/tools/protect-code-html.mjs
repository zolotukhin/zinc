#!/usr/bin/env bun
import { fileURLToPath } from 'node:url';
import { protectCodeHtml } from '../src/lib/code-html.mjs';

const output = fileURLToPath(new URL('../dist/', import.meta.url));
let pages = 0;
let snippets = 0;
// Astro rebuilds dist first. Protect the final HTML from every renderer:
// terminal components, Markdown, MDX and generated API reference templates.
for await (const path of new Bun.Glob('**/*.html').scan({ cwd: output, absolute: true })) {
  const result = await protectCodeHtml(await Bun.file(path).text());
  if (result.codeCount > 0) await Bun.write(path, result.html);
  pages += 1;
  snippets += result.codeCount;
}
console.log(`Preserved literal code in ${snippets} snippets across ${pages} pages.`);
