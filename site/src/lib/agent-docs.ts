import type { DocPage } from './docs-loader';
import type { ZigApiIndex } from './zig-api-loader';
import { STABLE_RELEASE, STABLE_VALIDATION_PATH } from './docs-release-status';

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
    description:
      'Architecture, GPU kernels, model support, scheduler behavior, and system design context around the runtime.',
    order: 1,
  },
  {
    slug: 'turboquant-spec',
    title: 'TurboQuant KV Cache Compression',
    description:
      'Two-stage vector quantization, residual correction, and memory-reduction tradeoffs for the paged KV-cache path.',
    order: 2,
  },
  {
    slug: 'rdna4-tuning',
    title: 'RDNA4 GPU Tuning Guide',
    description:
      'Performance profiling for AMD GPUs, cooperative matrix behavior, SPIR-V toolchain notes, and bandwidth tuning.',
    order: 3,
  },
  {
    slug: 'api',
    title: 'Serving HTTP API',
    description:
      'OpenAI-compatible request and response shapes for clients talking to ZINC. This is the network API, not the Zig code reference.',
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
  const bySlug = new Map(SUPPLEMENTAL_DOCS.map((doc) => [doc.slug, doc] as const));

  return [...docs]
    .sort((a, b) => (bySlug.get(a.slug)?.order ?? 99) - (bySlug.get(b.slug)?.order ?? 99))
    .map((doc) => {
      const meta = bySlug.get(doc.slug);
      return {
        slug: doc.slug,
        title: meta?.title ?? doc.title,
        description: meta?.description ?? 'ZINC documentation.',
        url: absoluteUrl(siteUrl, `/zinc/docs/${doc.slug}/`),
        last_updated: doc.lastmod,
      };
    });
}

function clientGuidance(base: string): string[] {
  return [
    '## Run ZINC or connect a client',
    `- Run models and connect apps: ${absoluteUrl(base, '/zinc/docs/running-zinc/')}`,
    `- Install and run your first model: ${absoluteUrl(base, '/zinc/docs/getting-started/')}`,
    `- HTTP API and backend coverage: ${absoluteUrl(base, '/zinc/docs/api/')}`,
    `- OpenClaw setup: ${absoluteUrl(base, '/zinc/docs/openclaw/')}`,
    `- OpenCode setup: ${absoluteUrl(base, '/zinc/docs/opencode/')}`,
    `- Published release qualification: ${absoluteUrl(base, STABLE_VALIDATION_PATH)}`,
    '',
    `Stable Linux release: ${STABLE_RELEASE}. Use the installed zinc command; no Zig compiler or source checkout is needed. Each executable contains one backend. Metal is a preview; CUDA is experimental.`,
    'AMD RDNA: prefer HIP / ROCm and install with ZINC_BACKEND=rocm. Intel Arc uses ZINC_BACKEND=vulkan; Vulkan is also an AMD alternative. Check the HIP runtime prerequisites and exact qualified hardware before installing.',
    '1. Inspect zinc --version --json, run zinc --check, and use zinc model list --json to choose a model that fits and has the required validation.',
    '2. Pull qwen35-9b-q4k-m, then start zinc serve --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090. Keep the server running while the client uses it.',
    '3. Check http://localhost:9090/health and discover the loaded ID and context at http://localhost:9090/v1/models. API base: http://localhost:9090/v1. Set stream explicitly.',
    '4. Use the standalone helper linked from the client guide. Run its setup and check commands before a task; verify the outcome independently.',
    '',
    'Structured discovery (0.1.1+): zinc serve, --no-browser, --version --json, --check --json and GET /v1/capabilities are released features. Check zinc --help for the installed version. Discovery includes build identity, backend, active_model, context_length and capabilities. On older servers, use /health and /v1/models; absent capabilities are unknown.',
    'Stable 0.1.1 HIP provides CLI/HTTP chat without browser UI or model switching. Current HIP source adds browser chat; current HIP/CUDA source advertises installed-model switching and unload. Enable controls through live capability discovery.',
    'Memory fit and protocol capabilities do not prove task quality. Never assume context size, model switching, tools or sampling support from a backend name alone.',
    '',
  ];
}

export function renderLlmsTxt(zigApi: ZigApiIndex, docs: DocPage[], siteUrl: string): string {
  const base = siteBaseUrl(siteUrl);
  const supplemental = supplementalDocs(docs, base);

  return [
    '# zolotukhin.ai',
    '',
    '> Technical writing and documentation for ZINC, local LLM inference on AMD (Vulkan or ROCm/HIP), Intel Arc (Vulkan), Apple Silicon (Metal preview), experimental NVIDIA CUDA, Zig, and systems engineering.',
    '',
    ...clientGuidance(base),
    'For changes to ZINC itself, use the development guide, repository AGENTS.md and generated Zig API.',
    '',
    '## ZINC Implementation Reference',
    `- ZINC Zig API: ${absoluteUrl(base, '/zinc/docs/zig-api/')}`,
    `- ZINC Zig API JSON: ${absoluteUrl(base, '/zinc/docs/zig-api.json')}`,
    `- ZINC Zig API Text: ${absoluteUrl(base, '/zinc/docs/zig-api.txt')}`,
    `- ZINC Docs Landing: ${absoluteUrl(base, '/zinc/docs/')}`,
    '',
    `Zig API coverage: ${zigApi.sections.length} sections, ${zigApi.moduleCount} modules, ${zigApi.codeLineCount} Zig code lines, ${zigApi.exportCount} exports, ${zigApi.memberCount} methods.`,
    '',
    '## Supplemental ZINC Documentation',
    ...supplemental.map((doc) => `- ${doc.title}: ${doc.url}`),
    '',
    '## HOCDB',
    '- Project and quickstart: https://zolotukhin.ai/hocdb/',
    '- Agent guide: https://zolotukhin.ai/hocdb/agent-guide.txt',
    '- CLI command contract: https://zolotukhin.ai/hocdb/commands.json',
    '- Releases: https://github.com/o1dstaRs/hocdb/releases',
    'HOCDB is a separate embedded time-series database. Use hocdb describe for its JSON CLI contract; its CLI is not an HTTP API.',
    '',
    '## Other Surfaces',
    `- Blog: ${absoluteUrl(base, '/blog/')}`,
    `- RSS: ${absoluteUrl(base, '/feed.xml')}`,
    '',
    '## Guidance',
    '- Use the Zig API for internal implementation details, symbols, signatures, and module relationships.',
    '- Use the Serving HTTP API page and its backend coverage matrix for client-facing contracts.',
    '- For code analysis, use the Zig API JSON export. For a running server, use health, models and capability discovery.',
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
    '',
    ...clientGuidance(base),
    `Zig API root: ${absoluteUrl(base, '/zinc/docs/zig-api/')}`,
    `Zig API JSON: ${absoluteUrl(base, '/zinc/docs/zig-api.json')}`,
    `Zig API Text: ${absoluteUrl(base, '/zinc/docs/zig-api.txt')}`,
    '',
    `Last updated: ${zigApi.lastmod}`,
    '',
    '## Implementation guidance',
    '- For source changes, read the repository AGENTS.md and Development guide first. Use the generated Zig API as the canonical reference for ZINC internals.',
    '- Treat the Serving HTTP API as the external network surface only.',
    '- For code analysis, prefer the Zig API JSON or text exports. For a running server, use health, models and capability discovery.',
    '',
    '## Zig API Sections',
  ];

  for (const section of zigApi.sections) {
    lines.push(
      '',
      `### ${section.title}`,
      section.description,
      `URL: ${absoluteUrl(base, `/zinc/docs/zig-api/#${section.slug}`)}`,
      `Coverage: ${section.modules.length} modules, ${section.symbolCount} symbols`,
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

  lines.push(
    '',
    '## HOCDB',
    '- Project: https://zolotukhin.ai/hocdb/',
    '- Agent guide: https://zolotukhin.ai/hocdb/agent-guide.txt',
    '- CLI command contract: https://zolotukhin.ai/hocdb/commands.json',
    '- Releases: https://github.com/o1dstaRs/hocdb/releases',
  );
  lines.push('', '## Other Surfaces');
  lines.push(`- Blog: ${absoluteUrl(base, '/blog/')}`);
  lines.push(`- RSS: ${absoluteUrl(base, '/feed.xml')}`);
  lines.push('');

  return lines.join('\n');
}
