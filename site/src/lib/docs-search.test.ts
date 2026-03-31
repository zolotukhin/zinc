import { buildDocsSearchIndex, type DocsSearchEntry } from './docs-search';
import type { DocPage } from './docs-loader';
import type { ZigApiIndex } from './zig-api-loader';

function findEntry(entries: DocsSearchEntry[], predicate: (entry: DocsSearchEntry) => boolean): DocsSearchEntry | undefined {
  return entries.find(predicate);
}

describe('buildDocsSearchIndex', () => {
  const docs: DocPage[] = [
    {
      slug: 'running-zinc',
      title: 'Running ZINC',
      lastmod: '2026-03-30',
      sourcePath: 'docs/RUNNING_ZINC.md',
      content: `# Running ZINC

Use ZINC from the CLI or server mode.

## Use with OpenAI SDKs

Point the OpenAI base URL at ZINC.

## Inspect the Managed Model Catalog

Use zinc model list to inspect supported models.`,
    },
  ];

  const zigApi: ZigApiIndex = {
    generatedAt: '2026-03-30',
    lastmod: '2026-03-30',
    moduleCount: 1,
    exportCount: 1,
    memberCount: 1,
    symbolCount: 2,
    sections: [
      {
        slug: 'inference-runtime',
        title: 'Inference Runtime',
        description: 'Decode loop and active inference state.',
        order: 1,
        symbolCount: 2,
        modules: [
          {
            slug: 'forward',
            title: 'forward',
            section: 'Inference Runtime',
            sectionSlug: 'inference-runtime',
            sectionOrder: 1,
            sectionDescription: 'Decode loop and active inference state.',
            sourcePath: 'src/compute/forward.zig',
            summary: 'Core decode path.',
            overview: ['Inference engine state and per-token decode.'],
            href: '/zinc/docs/zig-api/forward',
            exportCount: 1,
            memberCount: 1,
            symbolCount: 2,
            symbols: [
              {
                name: 'InferenceEngine',
                declarationKind: 'const',
                symbolKind: 'struct',
                signature: 'pub const InferenceEngine = struct',
                sourcePath: 'src/compute/forward.zig',
                line: 120,
                anchor: 'inferenceengine',
                href: '/zinc/docs/zig-api/forward#inferenceengine',
                qualifiedName: 'InferenceEngine',
                size: 128,
                alignment: 8,
                fields: [],
                doc: {
                  raw: 'Core runtime object.',
                  summary: 'Core runtime object.',
                  description: ['Owns the decode buffers and dispatch state.'],
                  params: [],
                  notes: [],
                },
                members: [
                  {
                    name: 'init',
                    declarationKind: 'fn',
                    symbolKind: 'function',
                    signature: 'pub fn init(...)',
                    sourcePath: 'src/compute/forward.zig',
                    line: 160,
                    anchor: 'inferenceengine-init',
                    href: '/zinc/docs/zig-api/forward#inferenceengine-init',
                    qualifiedName: 'InferenceEngine.init',
                    doc: {
                      raw: 'Initialize the engine.',
                      summary: 'Initialize the engine.',
                      description: ['Allocates the decode state.'],
                      params: [],
                      notes: [],
                    },
                  },
                ],
              },
            ],
          },
        ],
      },
    ],
    modules: [],
  };

  it('indexes markdown pages and heading anchors', () => {
    const entries = buildDocsSearchIndex(docs, zigApi);

    expect(findEntry(entries, entry => entry.kind === 'Guide' && entry.url === '/zinc/docs/running-zinc')?.title).toBe('Running ZINC');
    expect(findEntry(entries, entry => entry.kind === 'Section' && entry.url === '/zinc/docs/running-zinc#use-with-openai-sdks')?.title).toBe('Use with OpenAI SDKs');
    expect(findEntry(entries, entry => entry.kind === 'Section' && entry.url === '/zinc/docs/running-zinc#inspect-the-managed-model-catalog')?.preview).toContain('Use zinc model list');
  });

  it('indexes Zig API sections, modules, symbols, and methods', () => {
    const entries = buildDocsSearchIndex(docs, zigApi);

    expect(findEntry(entries, entry => entry.kind === 'API Section' && entry.url === '/zinc/docs/zig-api#inference-runtime')?.title).toBe('Inference Runtime');
    expect(findEntry(entries, entry => entry.kind === 'API Module' && entry.url === '/zinc/docs/zig-api/forward')?.searchText).toContain('InferenceEngine');
    expect(findEntry(entries, entry => entry.kind === 'API Symbol' && entry.url === '/zinc/docs/zig-api/forward#inferenceengine')?.title).toBe('InferenceEngine');
    expect(findEntry(entries, entry => entry.kind === 'API Method' && entry.url === '/zinc/docs/zig-api/forward#inferenceengine-init')?.preview).toContain('Allocates the decode state');
  });
});
