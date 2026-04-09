import type { DocPage } from './docs-loader';
import type { ZigApiIndex } from './zig-api-loader';

interface SupplementalDocMeta {
  slug: string;
  title: string;
  description: string;
  order: number;
}

const SUPPLEMENTAL_DOCS: SupplementalDocMeta[] = [
  {
    slug: 'spec',
    title: 'ZINC Technical Specification',
    description: 'Architecture, GPU kernels, model support, scheduler behavior, and system design context around the runtime.',
    order: 1,
  },
  {
    slug: 'turboquant-spec',
    title: 'TurboQuant KV Cache Compression',
    description: 'Two-stage vector quantization, residual correction, and memory-reduction tradeoffs for the paged KV-cache path.',
    order: 2,
  },
  {
    slug: 'rdna4-tuning',
    title: 'RDNA4 GPU Tuning Guide',
    description: 'Performance profiling for AMD GPUs, cooperative matrix behavior, SPIR-V toolchain notes, and bandwidth tuning.',
    order: 3,
  },
  {
    slug: 'api',
    title: 'Serving HTTP API',
    description: 'OpenAI-compatible request and response shapes for clients talking to ZINC. This is the network API, not the Zig code reference.',
    order: 4,
  },
];

function siteBaseUrl(siteUrl: string): string {
  return siteUrl.replace(/\/$/, '');
}

function absoluteUrl(siteUrl: string, path: string): string {
  const base = siteBaseUrl(siteUrl);
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

function supplementalDocs(docs: DocPage[], siteUrl: string) {
  const bySlug = new Map(SUPPLEMENTAL_DOCS.map(doc => [doc.slug, doc] as const));

  return [...docs]
    .sort((a, b) => (bySlug.get(a.slug)?.order ?? 99) - (bySlug.get(b.slug)?.order ?? 99))
    .map(doc => {
      const meta = bySlug.get(doc.slug);
      return {
        slug: doc.slug,
        title: meta?.title ?? doc.title,
        description: meta?.description ?? 'ZINC documentation.',
        url: absoluteUrl(siteUrl, `/zinc/docs/${doc.slug}`),
        last_updated: doc.lastmod,
      };
    });
}

export function renderLlmsTxt(zigApi: ZigApiIndex, docs: DocPage[], siteUrl: string): string {
  const base = siteBaseUrl(siteUrl);
  const supplemental = supplementalDocs(docs, base);

  return [
    '# zolotukhin.ai',
    '',
    '> Technical writing and documentation for ZINC, local LLM inference on AMD GPUs, Vulkan compute, Zig, and systems engineering.',
    '',
    'For ZINC, prefer the generated Zig API reference first. It is the canonical internal runtime reference.',
    '',
    '## Primary ZINC Documentation',
    `- ZINC Zig API: ${absoluteUrl(base, '/zinc/docs/zig-api')}`,
    `- ZINC Zig API JSON: ${absoluteUrl(base, '/zinc/docs/zig-api.json')}`,
    `- ZINC Zig API Text: ${absoluteUrl(base, '/zinc/docs/zig-api.txt')}`,
    `- ZINC Docs Landing: ${absoluteUrl(base, '/zinc/docs')}`,
    '',
    `Zig API coverage: ${zigApi.sections.length} sections, ${zigApi.moduleCount} modules, ${zigApi.codeLineCount} Zig code lines, ${zigApi.exportCount} exports, ${zigApi.memberCount} methods.`,
    '',
    '## Supplemental ZINC Documentation',
    ...supplemental.map(doc => `- ${doc.title}: ${doc.url}`),
    '',
    '## Other Surfaces',
    `- Blog: ${absoluteUrl(base, '/blog')}`,
    `- RSS: ${absoluteUrl(base, '/feed.xml')}`,
    '',
    '## Guidance',
    '- Use the Zig API for internal implementation details, symbols, signatures, and module relationships.',
    '- Use the Serving HTTP API page for client-facing request and response contracts.',
    '- Prefer the JSON export when building tools or agents that need structured access.',
    '',
  ].join('\n');
}

export function renderLlmsFullTxt(zigApi: ZigApiIndex, docs: DocPage[], siteUrl: string): string {
  const base = siteBaseUrl(siteUrl);
  const supplemental = supplementalDocs(docs, base);
  const lines: string[] = [
    '# zolotukhin.ai Full Documentation Index',
    '',
    '> Agent-oriented index for the ZINC documentation set.',
    '',
    `Site: ${base}`,
    `LLMs index: ${absoluteUrl(base, '/llms.txt')}`,
    `Zig API root: ${absoluteUrl(base, '/zinc/docs/zig-api')}`,
    `Zig API JSON: ${absoluteUrl(base, '/zinc/docs/zig-api.json')}`,
    `Zig API Text: ${absoluteUrl(base, '/zinc/docs/zig-api.txt')}`,
    '',
    `Last updated: ${zigApi.lastmod}`,
    '',
    '## Guidance',
    '- Treat the generated Zig API as the primary and canonical reference for ZINC internals.',
    '- Treat the Serving HTTP API as the external network surface only.',
    '- Prefer the JSON export for structured ingestion and the text export for lightweight retrieval.',
    '',
    '## Zig API Sections',
  ];

  for (const section of zigApi.sections) {
    lines.push(
      '',
      `### ${section.title}`,
      section.description,
      `URL: ${absoluteUrl(base, `/zinc/docs/zig-api#${section.slug}`)}`,
      `Coverage: ${section.modules.length} modules, ${section.symbolCount} symbols`
    );

    for (const module of section.modules) {
      lines.push(`- ${module.title}: ${absoluteUrl(base, module.href)}`);
      lines.push(`  Source: ${module.sourcePath}`);
      lines.push(`  Summary: ${module.summary}`);
    }
  }

  lines.push('', '## Supplemental ZINC Docs');
  for (const doc of supplemental) {
    lines.push(`- ${doc.title}: ${doc.url}`);
    lines.push(`  Summary: ${doc.description}`);
  }

  lines.push('', '## Other Surfaces');
  lines.push(`- Blog: ${absoluteUrl(base, '/blog')}`);
  lines.push(`- RSS: ${absoluteUrl(base, '/feed.xml')}`);
  lines.push('');

  return lines.join('\n');
}
