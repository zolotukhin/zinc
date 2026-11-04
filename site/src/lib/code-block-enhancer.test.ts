import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

const component = readFileSync(
  new URL('../components/CodeBlockEnhancer.astro', import.meta.url),
  'utf8',
);
const script = component.match(/<script>([\s\S]*?)<\/script>/)?.[1];
if (!script) throw new Error('Code block enhancer script is missing');

for (const fixture of [
  { name: 'Astro data-language', preLanguage: 'mermaid' },
  { name: 'mixed-case data-language', preLanguage: 'Mermaid' },
  { name: 'code data-language', codeLanguage: 'mermaid' },
  { name: 'rehype language class', codeClass: 'language-mermaid' },
]) {
  test(`Mermaid ${fixture.name} remains unwrapped for the diagram renderer`, () => {
    const content = 'flowchart TD\n  A[Build] --> B[Publish]';
    const code = {
      classList: fixture.codeClass ? [fixture.codeClass] : [],
      getAttribute: (name: string) =>
        name === 'data-language' ? (fixture.codeLanguage ?? null) : null,
      textContent: content,
    };
    const pre = {
      dataset: {} as Record<string, string>,
      classList: ['astro-code'],
      getAttribute: (name: string) =>
        name === 'data-language' ? (fixture.preLanguage ?? null) : null,
      closest: () => null,
      querySelector: () => code,
      textContent: content,
    };
    // Run the actual enhancement entrypoint. A diagram must not reach toolbar
    // creation, regardless of whether its language has a visible code label.
    const document = {
      readyState: 'complete',
      querySelectorAll: () => [pre],
      createElement: () => {
        throw new Error('Mermaid diagram received a code toolbar');
      },
    };
    expect(() => runInNewContext(script, { document })).not.toThrow();
    expect(pre.dataset.codeEnhanced).toBeUndefined();
  });
}
