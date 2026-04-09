import { existsSync } from 'node:fs';
import { readdir, readFile, stat } from 'node:fs/promises';
import { basename, join } from 'node:path';

export interface ZigApiParam {
  name: string;
  description: string;
}

export interface ZigApiDocBlock {
  raw: string;
  summary: string;
  description: string[];
  section?: string;
  params: ZigApiParam[];
  returns?: string;
  notes: string[];
}

export interface ZigApiMember {
  name: string;
  declarationKind: 'fn';
  symbolKind: 'function';
  signature: string;
  doc: ZigApiDocBlock;
  sourcePath: string;
  line: number;
  anchor: string;
  href: string;
  qualifiedName: string;
}

export interface ZigApiStructField {
  name: string;
  type: string;
  size: number;
  alignment: number;
  offset: number;
}

export interface ZigApiSymbol {
  name: string;
  declarationKind: 'const' | 'fn' | 'var';
  symbolKind: 'function' | 'struct' | 'enum' | 'union' | 'constant' | 'variable';
  signature: string;
  doc: ZigApiDocBlock;
  sourcePath: string;
  line: number;
  anchor: string;
  href: string;
  qualifiedName: string;
  members: ZigApiMember[];
  size?: number;
  alignment?: number;
  fields?: ZigApiStructField[];
}

export interface ZigApiModule {
  slug: string;
  title: string;
  section: string;
  sectionSlug: string;
  sectionOrder: number;
  sectionDescription: string;
  sourcePath: string;
  summary: string;
  overview: string[];
  href: string;
  symbols: ZigApiSymbol[];
  codeLineCount: number;
  exportCount: number;
  memberCount: number;
  symbolCount: number;
}

export interface ZigApiSection {
  slug: string;
  title: string;
  description: string;
  order: number;
  modules: ZigApiModule[];
  symbolCount: number;
}

export interface ZigApiIndex {
  generatedAt: string;
  lastmod: string;
  moduleCount: number;
  codeLineCount: number;
  exportCount: number;
  memberCount: number;
  symbolCount: number;
  sections: ZigApiSection[];
  modules: ZigApiModule[];
}

interface ParsedComment extends ZigApiDocBlock { }

const EXCLUDED_MODULES = new Set(['vulkan/vk.zig']);

const SECTION_META = new Map<string, { title: string; description: string; order: number }>([
  [
    'cli-entrypoints',
    {
      title: 'CLI & Entrypoints',
      description: 'Startup, argument parsing, and the top-level process path that wires model loading, tokenization, and generation together.',
      order: 0,
    },
  ],
  [
    'model-format-loading',
    {
      title: 'Model Format & Loading',
      description: 'GGUF parsing, metadata normalization, and the runtime structures that move weights from disk into GPU-resident buffers.',
      order: 1,
    },
  ],
  [
    'tokenization',
    {
      title: 'Tokenization',
      description: 'Prompt and output text conversion between UTF-8 strings and token IDs used by the decode loop.',
      order: 2,
    },
  ],
  [
    'decode-planning',
    {
      title: 'Decode Planning',
      description: 'Static graph construction and dependency ordering for the per-token compute work that the runtime records and submits.',
      order: 3,
    },
  ],
  [
    'inference-runtime',
    {
      title: 'Inference Runtime',
      description: 'Decode state, pipeline ownership, command recording, and token sampling inside the active inference loop.',
      order: 4,
    },
  ],
  [
    'shader-dispatch',
    {
      title: 'Shader Dispatch',
      description: 'Typed wrappers around the compute shaders that prepare push constants, descriptor layouts, and per-op dispatch dimensions.',
      order: 5,
    },
  ],
  [
    'hardware-detection',
    {
      title: 'Hardware Detection',
      description: 'Vendor and architecture heuristics that translate raw Vulkan properties into tuning defaults for AMD, NVIDIA, and Intel GPUs.',
      order: 6,
    },
  ],
  [
    'vulkan-runtime',
    {
      title: 'Vulkan Runtime',
      description: 'Low-level Vulkan setup, memory allocation, buffers, pipelines, and command submission primitives used throughout the engine.',
      order: 7,
    },
  ],
  [
    'scheduler',
    {
      title: 'Scheduler',
      description: 'Continuous batching scheduler, paged KV cache management, and request lifecycle tracking for concurrent inference serving.',
      order: 8,
    },
  ],
  [
    'api-server',
    {
      title: 'API Server',
      description: 'OpenAI-compatible HTTP server, route dispatch, SSE streaming, and session management for serving inference over the network.',
      order: 9,
    },
  ],
]);

function resolveSiteRoot(): string {
  const cwd = process.cwd();

  if (existsSync(join(cwd, 'astro.config.mjs')) && existsSync(join(cwd, 'src'))) {
    return cwd;
  }

  const nestedSiteRoot = join(cwd, 'site');
  if (existsSync(join(nestedSiteRoot, 'astro.config.mjs')) && existsSync(join(nestedSiteRoot, 'src'))) {
    return nestedSiteRoot;
  }

  throw new Error(`Unable to resolve site root from cwd: ${cwd}`);
}

const SITE_ROOT = resolveSiteRoot();
const REPO_ROOT = join(SITE_ROOT, '..');
const SRC_ROOT = join(REPO_ROOT, 'src');
let zigApiPromise: Promise<ZigApiIndex> | null = null;

function toSlugPart(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
}

function prettyToken(token: string): string {
  const upperTokens = new Map([
    ['gguf', 'GGUF'],
    ['gpu', 'GPU'],
    ['rdna4', 'RDNA4'],
    ['dmmv', 'DMMV'],
    ['kv', 'KV'],
    ['cli', 'CLI'],
  ]);

  return token
    .split('_')
    .map(part => upperTokens.get(part) ?? `${part[0]?.toUpperCase() ?? ''}${part.slice(1)}`)
    .join(' ');
}

function titleFromPath(relativePath: string): string {
  const filename = basename(relativePath, '.zig');
  if (filename === 'main') return 'CLI';
  return prettyToken(filename);
}

function slugFromPath(relativePath: string): string {
  return basename(relativePath, '.zig').toLowerCase().replace(/_/g, '-');
}

function modulePath(relativePath: string): string {
  return `src/${relativePath}`;
}

function moduleHref(slug: string): string {
  return `/zinc/docs/zig-api/${slug}`;
}

function sourceHref(sourcePath: string, line: number): string {
  return `https://github.com/zolotukhin/zinc/blob/main/${sourcePath}#L${line}`;
}

function siteBaseUrl(siteUrl: string): string {
  return siteUrl.replace(/\/$/, '');
}

function absoluteUrl(siteUrl: string, path: string): string {
  if (/^https?:\/\//.test(path)) return path;
  const base = siteBaseUrl(siteUrl);
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

function collapseWhitespace(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

function normalizeDocLines(lines: string[]): string {
  return lines.join('\n').replace(/\n{3,}/g, '\n\n').trim();
}

function toParagraphs(raw: string): string[] {
  if (!raw) return [];

  return raw
    .split(/\n{2,}/)
    .map(paragraph => collapseWhitespace(paragraph.replace(/\n/g, ' ')))
    .filter(Boolean);
}

function splitSummaryAndDescription(raw: string): Pick<ZigApiDocBlock, 'summary' | 'description'> {
  const paragraphs = toParagraphs(raw);

  if (paragraphs.length > 1) {
    return {
      summary: paragraphs[0],
      description: paragraphs.slice(1),
    };
  }

  const singleParagraph = paragraphs[0] ?? '';
  const sentenceMatch = singleParagraph.match(/^(.+?[.!?])(?:\s+)(.+)$/);

  if (sentenceMatch) {
    return {
      summary: sentenceMatch[1].trim(),
      description: [sentenceMatch[2].trim()],
    };
  }

  return {
    summary: singleParagraph,
    description: [],
  };
}

function parseDocComment(lines: string[]): ParsedComment {
  const textLines: string[] = [];
  const params: ZigApiParam[] = [];
  const notes: string[] = [];
  let returns: string | undefined;
  let section: string | undefined;

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (line === '') {
      textLines.push('');
      continue;
    }

    if (line.startsWith('@section ')) {
      section = line.slice('@section '.length).trim();
      continue;
    }

    const paramMatch = line.match(/^@param\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+(.+)$/);
    if (paramMatch) {
      params.push({
        name: paramMatch[1],
        description: paramMatch[2].trim(),
      });
      continue;
    }

    const returnsMatch = line.match(/^@returns\s+(.+)$/);
    if (returnsMatch) {
      returns = returnsMatch[1].trim();
      continue;
    }

    const noteMatch = line.match(/^@note\s+(.+)$/);
    if (noteMatch) {
      notes.push(noteMatch[1].trim());
      continue;
    }

    textLines.push(line);
  }

  const raw = normalizeDocLines(textLines);
  const { summary, description } = splitSummaryAndDescription(raw);

  return {
    raw,
    summary,
    description,
    section,
    params,
    returns,
    notes,
  };
}

function stripLineComment(line: string): string {
  let result = '';
  let inString = false;
  let inChar = false;
  let escaped = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];

    if (!inString && !inChar && char === '/' && next === '/') {
      break;
    }

    result += char;

    if (escaped) {
      escaped = false;
      continue;
    }

    if ((inString || inChar) && char === '\\') {
      escaped = true;
      continue;
    }

    if (!inChar && char === '"') {
      inString = !inString;
      continue;
    }

    if (!inString && char === '\'') {
      inChar = !inChar;
    }
  }

  return result;
}

function braceDelta(line: string): number {
  const source = stripLineComment(line);
  let delta = 0;

  for (const char of source) {
    if (char === '{') delta += 1;
    if (char === '}') delta -= 1;
  }

  return delta;
}

function captureDeclaration(lines: string[], startIndex: number): { signature: string; endIndex: number; hasBlock: boolean } {
  const declaration: string[] = [];

  for (let i = startIndex; i < lines.length; i += 1) {
    const source = stripLineComment(lines[i]);
    declaration.push(source.trim());

    if (source.includes('{') || source.trimEnd().endsWith(';')) {
      return {
        signature: collapseWhitespace(
          declaration
            .join(' ')
            .replace(/\s+\{/g, '')
            .replace(/;$/, '')
        ),
        endIndex: i,
        hasBlock: source.includes('{'),
      };
    }
  }

  return {
    signature: collapseWhitespace(declaration.join(' ')),
    endIndex: lines.length - 1,
    hasBlock: false,
  };
}

function captureBlockEnd(lines: string[], startIndex: number): number {
  let depth = 0;
  let opened = false;

  for (let i = startIndex; i < lines.length; i += 1) {
    depth += braceDelta(lines[i]);
    if (depth > 0) opened = true;
    if (opened && depth === 0) return i;
  }

  return startIndex;
}

function classifySymbol(signature: string, declarationKind: ZigApiSymbol['declarationKind']): ZigApiSymbol['symbolKind'] {
  if (declarationKind === 'fn') return 'function';
  if (/\=\s*(?:extern\s+)?struct\b/.test(signature)) return 'struct';
  if (/\=\s*(?:extern\s+)?enum\b/.test(signature)) return 'enum';
  if (/\=\s*(?:extern\s+)?union\b/.test(signature)) return 'union';
  return declarationKind === 'var' ? 'variable' : 'constant';
}

function isContainerSymbol(symbolKind: ZigApiSymbol['symbolKind']): boolean {
  return symbolKind === 'struct' || symbolKind === 'enum' || symbolKind === 'union';
}

function anchorFromParts(parts: string[]): string {
  return parts.map(toSlugPart).join('-');
}

function parseModuleComment(lines: string[]): ParsedComment {
  const moduleLines: string[] = [];
  let seenDoc = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (!seenDoc && trimmed === '') continue;

    if (trimmed.startsWith('//!')) {
      seenDoc = true;
      moduleLines.push(trimmed.replace(/^\/\/!\s?/, ''));
      continue;
    }

    if (seenDoc && trimmed === '') {
      moduleLines.push('');
      continue;
    }

    break;
  }

  return parseDocComment(moduleLines);
}

function defaultSectionForPath(relativePath: string): string {
  switch (relativePath) {
    case 'main.zig':
      return 'CLI & Entrypoints';
    case 'model/gguf.zig':
    case 'model/loader.zig':
      return 'Model Format & Loading';
    case 'model/tokenizer.zig':
      return 'Tokenization';
    case 'model/architecture.zig':
    case 'compute/graph.zig':
      return 'Decode Planning';
    case 'compute/forward.zig':
      return 'Inference Runtime';
    case 'compute/attention.zig':
    case 'compute/dmmv.zig':
    case 'compute/elementwise.zig':
      return 'Shader Dispatch';
    case 'vulkan/gpu_detect.zig':
      return 'Hardware Detection';
    case 'scheduler/scheduler.zig':
    case 'scheduler/request.zig':
    case 'scheduler/kv_cache.zig':
      return 'Scheduler';
    case 'server/http.zig':
    case 'server/routes.zig':
    case 'server/session.zig':
      return 'API Server';
    default:
      return 'Vulkan Runtime';
  }
}

function sectionMeta(sectionTitle: string): { title: string; description: string; order: number; slug: string } {
  const slug = toSlugPart(sectionTitle);
  const known = SECTION_META.get(slug);

  if (known) {
    return {
      ...known,
      slug,
    };
  }

  return {
    title: sectionTitle,
    description: `${sectionTitle} modules and helpers.`,
    order: 99,
    slug,
  };
}

function parseContainerMembers(
  lines: string[],
  containerStart: number,
  containerEnd: number,
  relativePath: string,
  moduleSlug: string,
  parentName: string
): ZigApiMember[] {
  const members: ZigApiMember[] = [];
  let pendingDocs: string[] = [];
  let depth = 1;

  for (let i = containerStart + 1; i < containerEnd; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    if (depth === 1 && trimmed.startsWith('///')) {
      pendingDocs.push(trimmed.replace(/^\/\/\/\s?/, ''));
      continue;
    }

    if (depth === 1 && /^pub fn\b/.test(trimmed)) {
      const match = trimmed.match(/^pub fn\s+([A-Za-z_][A-Za-z0-9_]*)/);
      const { signature, endIndex, hasBlock } = captureDeclaration(lines, i);
      const bodyEnd = hasBlock ? captureBlockEnd(lines, i) : endIndex;

      if (match) {
        const name = match[1];
        const anchor = anchorFromParts([parentName, name]);
        const href = `${moduleHref(moduleSlug)}#${anchor}`;

        members.push({
          name,
          declarationKind: 'fn',
          symbolKind: 'function',
          signature,
          doc: parseDocComment(pendingDocs),
          sourcePath: modulePath(relativePath),
          line: i + 1,
          anchor,
          href,
          qualifiedName: `${parentName}.${name}`,
        });
      }

      pendingDocs = [];
      i = bodyEnd;
      continue;
    }

    if (depth === 1 && pendingDocs.length > 0) {
      if (trimmed === '') {
        pendingDocs.push('');
      } else if (!trimmed.startsWith('//')) {
        pendingDocs = [];
      }
    }

    depth += braceDelta(line);
    if (depth < 1) break;
  }

  return members;
}

function parseZigModuleContent(content: string, relativePath: string): ZigApiModule | null {
  const lines = content.split(/\r?\n/);
  const moduleDoc = parseModuleComment(lines);
  const slug = slugFromPath(relativePath);
  const href = moduleHref(slug);
  const symbols: ZigApiSymbol[] = [];
  let pendingDocs: string[] = [];
  let depth = 0;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    if (depth === 0 && trimmed.startsWith('///')) {
      pendingDocs.push(trimmed.replace(/^\/\/\/\s?/, ''));
      continue;
    }

    if (depth === 0 && /^(pub (const|fn|var))\b/.test(trimmed)) {
      const match = trimmed.match(/^pub (const|fn|var)\s+([A-Za-z_][A-Za-z0-9_]*)/);
      const { signature, endIndex, hasBlock } = captureDeclaration(lines, i);

      if (match) {
        const declarationKind = match[1] as ZigApiSymbol['declarationKind'];
        const name = match[2];
        const symbolKind = classifySymbol(signature, declarationKind);
        const blockEnd = hasBlock ? captureBlockEnd(lines, i) : endIndex;
        const members = isContainerSymbol(symbolKind)
          ? parseContainerMembers(lines, i, blockEnd, relativePath, slug, name)
          : [];
        const anchor = anchorFromParts([name]);

        symbols.push({
          name,
          declarationKind,
          symbolKind,
          signature,
          doc: parseDocComment(pendingDocs),
          sourcePath: modulePath(relativePath),
          line: i + 1,
          anchor,
          href: `${href}#${anchor}`,
          qualifiedName: name,
          members,
        });

        pendingDocs = [];
        i = blockEnd;
        continue;
      }
    }

    if (depth === 0 && pendingDocs.length > 0) {
      if (trimmed === '') {
        pendingDocs.push('');
      } else if (!trimmed.startsWith('//')) {
        pendingDocs = [];
      }
    }

    depth += braceDelta(line);
    if (depth < 0) depth = 0;
  }

  if (symbols.length === 0) return null;

  const sectionTitle = moduleDoc.section || defaultSectionForPath(relativePath);
  const section = sectionMeta(sectionTitle);
  const memberCount = symbols.reduce((count, symbol) => count + symbol.members.length, 0);
  const summary =
    moduleDoc.summary ||
    symbols.find(symbol => symbol.doc.summary)?.doc.summary ||
    `Public API surface for ${modulePath(relativePath)}.`;

  return {
    slug,
    title: titleFromPath(relativePath),
    section: section.title,
    sectionSlug: section.slug,
    sectionOrder: section.order,
    sectionDescription: section.description,
    sourcePath: modulePath(relativePath),
    summary,
    overview: moduleDoc.description,
    href,
    symbols,
    codeLineCount: countZigCodeLines(content),
    exportCount: symbols.length,
    memberCount,
    symbolCount: symbols.length + memberCount,
  };
}

async function collectZigFiles(directory: string, prefix = ''): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const results: string[] = [];

  for (const entry of entries.toSorted((a, b) => a.name.localeCompare(b.name))) {
    if (entry.name.startsWith('.')) continue;

    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
    const fullPath = join(directory, entry.name);

    if (entry.isDirectory()) {
      results.push(...await collectZigFiles(fullPath, relativePath));
      continue;
    }

    if (!entry.isFile() || !entry.name.endsWith('.zig')) continue;
    if (EXCLUDED_MODULES.has(relativePath)) continue;
    results.push(relativePath);
  }

  return results;
}

export function parseZigModule(content: string, relativePath: string): ZigApiModule | null {
  return parseZigModuleContent(content, relativePath);
}

export function getSourceHref(sourcePath: string, line: number): string {
  return sourceHref(sourcePath, line);
}

function normalizeDocBlock(doc: ZigApiDocBlock) {
  return {
    summary: doc.summary,
    description: doc.description,
    params: doc.params,
    returns: doc.returns ?? null,
    notes: doc.notes,
    raw: doc.raw,
  };
}

function countZigCodeLines(content: string): number {
  const lines = content.split(/\r?\n/);
  let count = 0;
  let inBlockComment = false;

  for (const line of lines) {
    let index = 0;
    let hasCode = false;

    while (index < line.length) {
      if (inBlockComment) {
        const blockEnd = line.indexOf('*/', index);
        if (blockEnd === -1) {
          index = line.length;
          break;
        }
        inBlockComment = false;
        index = blockEnd + 2;
        continue;
      }

      const ch = line[index];
      if (ch === ' ' || ch === '\t') {
        index += 1;
        continue;
      }

      if (line.startsWith('//', index)) break;

      if (line.startsWith('/*', index)) {
        inBlockComment = true;
        index += 2;
        continue;
      }

      hasCode = true;
      break;
    }

    if (hasCode) count += 1;
  }

  return count;
}

export function createZigApiAgentPayload(api: ZigApiIndex, siteUrl: string) {
  const base = siteBaseUrl(siteUrl);

  return {
    schema_version: 'zinc-zig-api-v1',
    title: 'ZINC Zig API',
    description: 'Primary generated reference for the ZINC Zig runtime and supporting Vulkan compute modules.',
    site: base,
    root_url: absoluteUrl(base, '/zinc/docs/zig-api'),
    json_url: absoluteUrl(base, '/zinc/docs/zig-api.json'),
    text_url: absoluteUrl(base, '/zinc/docs/zig-api.txt'),
    llms_url: absoluteUrl(base, '/llms.txt'),
    generated_at: api.generatedAt,
    last_updated: api.lastmod,
    counts: {
      sections: api.sections.length,
      modules: api.moduleCount,
      code_lines: api.codeLineCount,
      exports: api.exportCount,
      methods: api.memberCount,
      symbols: api.symbolCount,
    },
    sections: api.sections.map(section => ({
      slug: section.slug,
      title: section.title,
      description: section.description,
      url: absoluteUrl(base, `/zinc/docs/zig-api#${section.slug}`),
      module_count: section.modules.length,
      symbol_count: section.symbolCount,
      modules: section.modules.map(module => ({
        slug: module.slug,
        title: module.title,
        section: module.section,
        summary: module.summary,
        overview: module.overview,
        url: absoluteUrl(base, module.href),
        source_path: module.sourcePath,
        source_url: sourceHref(module.sourcePath, module.symbols[0]?.line ?? 1),
        counts: {
          code_lines: module.codeLineCount,
          exports: module.exportCount,
          methods: module.memberCount,
          symbols: module.symbolCount,
        },
        symbols: module.symbols.map(symbol => ({
          name: symbol.name,
          qualified_name: symbol.qualifiedName,
          declaration_kind: symbol.declarationKind,
          kind: symbol.symbolKind,
          signature: symbol.signature,
          anchor: symbol.anchor,
          url: absoluteUrl(base, symbol.href),
          source_path: symbol.sourcePath,
          source_line: symbol.line,
          source_url: sourceHref(symbol.sourcePath, symbol.line),
          doc: normalizeDocBlock(symbol.doc),
          ...(symbol.symbolKind === 'struct' && symbol.size !== undefined ? {
            size: symbol.size,
            alignment: symbol.alignment,
            fields: symbol.fields,
          } : {}),
          members: symbol.members.map(member => ({
            name: member.name,
            qualified_name: member.qualifiedName,
            declaration_kind: member.declarationKind,
            kind: member.symbolKind,
            signature: member.signature,
            anchor: member.anchor,
            url: absoluteUrl(base, member.href),
            source_path: member.sourcePath,
            source_line: member.line,
            source_url: sourceHref(member.sourcePath, member.line),
            doc: normalizeDocBlock(member.doc),
          })),
        })),
      })),
    })),
  };
}

export function renderZigApiAgentText(api: ZigApiIndex, siteUrl: string): string {
  const payload = createZigApiAgentPayload(api, siteUrl);
  const lines: string[] = [
    '# ZINC Zig API',
    '',
    payload.description,
    '',
    `Primary docs: ${payload.root_url}`,
    `JSON export: ${payload.json_url}`,
    `Text export: ${payload.text_url}`,
    `LLMs index: ${payload.llms_url}`,
    `Last updated: ${payload.last_updated}`,
    `Counts: ${payload.counts.sections} sections, ${payload.counts.modules} modules, ${payload.counts.code_lines} Zig code lines, ${payload.counts.exports} exports, ${payload.counts.methods} methods.`,
    '',
    'Guidance: Use the generated Zig API as the canonical internal runtime reference. Use the Serving HTTP API doc only for the client-facing network protocol.',
  ];

  for (const section of payload.sections) {
    lines.push('', `## ${section.title}`, '', section.description, `URL: ${section.url}`);

    for (const module of section.modules) {
      lines.push(
        '',
        `### Module: ${module.title}`,
        `URL: ${module.url}`,
        `Source: ${module.source_path}`,
        `Code lines: ${module.counts.code_lines}`,
        `Summary: ${module.summary}`
      );

      for (const paragraph of module.overview) {
        lines.push(`Overview: ${paragraph}`);
      }

      for (const symbol of module.symbols) {
        lines.push(
          '',
          `- ${symbol.qualified_name} [${symbol.kind}]`,
          `  URL: ${symbol.url}`,
          `  Signature: ${symbol.signature}`
        );

        if (symbol.doc.summary) lines.push(`  Summary: ${symbol.doc.summary}`);
        for (const paragraph of symbol.doc.description) {
          lines.push(`  Description: ${paragraph}`);
        }
        for (const param of symbol.doc.params) {
          lines.push(`  Param ${param.name}: ${param.description}`);
        }
        if (symbol.doc.returns) lines.push(`  Returns: ${symbol.doc.returns}`);
        for (const note of symbol.doc.notes) {
          lines.push(`  Note: ${note}`);
        }

        if (symbol.kind === 'struct' && symbol.size !== undefined) {
          lines.push(`  Size: ${symbol.size} bytes (alignment: ${symbol.alignment})`);
          if (symbol.fields && symbol.fields.length > 0) {
            lines.push(`  Fields:`);
            for (const field of symbol.fields) {
              lines.push(`    - ${field.name}: ${field.type} (offset: ${field.offset}, size: ${field.size}, align: ${field.alignment})`);
            }
          }
        }

        for (const member of symbol.members) {
          lines.push(
            `  - ${member.qualified_name} [method]`,
            `    URL: ${member.url}`,
            `    Signature: ${member.signature}`
          );

          if (member.doc.summary) lines.push(`    Summary: ${member.doc.summary}`);
          for (const paragraph of member.doc.description) {
            lines.push(`    Description: ${paragraph}`);
          }
          for (const param of member.doc.params) {
            lines.push(`    Param ${param.name}: ${param.description}`);
          }
          if (member.doc.returns) lines.push(`    Returns: ${member.doc.returns}`);
          for (const note of member.doc.notes) {
            lines.push(`    Note: ${note}`);
          }
        }
      }
    }
  }

  return `${lines.join('\n').trim()}\n`;
}

async function loadZigApiImpl(): Promise<ZigApiIndex> {
  const files = await collectZigFiles(SRC_ROOT);
  const modules: ZigApiModule[] = [];
  let latestMtime = 0;

  for (const relativePath of files) {
    const absolutePath = join(SRC_ROOT, relativePath);
    const [content, fileStat] = await Promise.all([
      readFile(absolutePath, 'utf-8'),
      stat(absolutePath),
    ]);
    latestMtime = Math.max(latestMtime, fileStat.mtimeMs);

    const parsed = parseZigModuleContent(content, relativePath);
    if (parsed) modules.push(parsed);
  }

  const structSymbols: { modulePath: string; qualifiedName: string; symbol: ZigApiSymbol }[] = [];
  for (const mod of modules) {
    for (const sym of mod.symbols) {
      if (sym.symbolKind === 'struct') {
        structSymbols.push({ modulePath: sym.sourcePath, qualifiedName: sym.qualifiedName, symbol: sym });
      }
    }
  }

  if (structSymbols.length > 0) {
    const structModulePaths = [...new Set(structSymbols.map(s => s.modulePath))];
    const zigScriptContent = `const std = @import("std");

fn dumpStruct(comptime T: type, name: []const u8, first: *bool, w: anytype) !void {
    const type_info = @typeInfo(T);
    if (type_info != .@"struct") return;
    if (!first.*) try w.print(",", .{});
    first.* = false;
    
    try w.print("\\"{s}\\":{{", .{name});
    try w.print("\\"size\\":{d},\\"alignment\\":{d},\\"fields\\":[", .{ @sizeOf(T), @alignOf(T) });
    
    var firstField = true;
    inline for (type_info.@"struct".fields) |f| {
        if (@sizeOf(f.type) > 0 and !f.is_comptime) {
            if (!firstField) try w.print(",", .{});
            try w.print("{{\\"name\\":\\"{s}\\",\\"type\\":\\"{s}\\",\\"size\\":{d},\\"alignment\\":{d},\\"offset\\":{d}}}", .{
                f.name, @typeName(f.type), @sizeOf(f.type), f.alignment, @offsetOf(T, f.name)
            });
            firstField = false;
        }
    }
    try w.print("]}}", .{});
}

pub fn main() !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    const out = &stdout.interface;
    try out.print("{{", .{});
    var first = true;
${structSymbols.map((s) => {
      const modIdx = structModulePaths.indexOf(s.modulePath);
      return `    if (@TypeOf(mod${modIdx}.${s.qualifiedName}) == type) {\n        dumpStruct(mod${modIdx}.${s.qualifiedName}, "${s.qualifiedName}", &first, out) catch {};\n    }`;
    }).join('\n')}
    try out.print("}}", .{});
    try out.flush();
}

${structModulePaths.map((path, idx) => `const mod${idx} = @import("../${path.replace(/^src\//, '')}");`).join('\n')}
`;

    const { mkdir, writeFile } = await import('node:fs/promises');
    const { execFile } = await import('node:child_process');
    const { promisify } = await import('node:util');
    const execFileAsync = promisify(execFile);

    const cacheRoot = join(REPO_ROOT, 'src', '.zig-api-cache');
    const scriptPath = join(cacheRoot, 'zig-struct-analyzer.generated.zig');
    const runnerPath = join(REPO_ROOT, 'src', 'zig-struct-analyzer.zig');
    const globalCache = join(cacheRoot, 'global');
    const localCache = join(cacheRoot, 'local');
    await mkdir(globalCache, { recursive: true });
    await mkdir(localCache, { recursive: true });
    if (!existsSync(scriptPath) || (await readFile(scriptPath, 'utf-8')) !== zigScriptContent) {
      await writeFile(scriptPath, zigScriptContent, 'utf-8');
    }

    try {
      const zigArgs = ['run', '-lc'];
      zigArgs.push('-I', join(REPO_ROOT, 'src', 'metal'));

      if (process.platform === 'darwin') {
        zigArgs.push('-I', '/opt/homebrew/include', '-L', '/opt/homebrew/lib', '-lvulkan');
      } else if (process.platform === 'win32') {
        const vulkanSdk = process.env.VULKAN_SDK ?? process.env.VK_SDK_PATH;
        if (vulkanSdk) {
          zigArgs.push(
            '-I',
            join(vulkanSdk, 'Include'),
            '-L',
            join(vulkanSdk, process.arch === 'ia32' ? 'Lib32' : 'Lib'),
            '-lvulkan-1'
          );
        }
      } else {
        zigArgs.push('-lvulkan');
      }

      zigArgs.push(runnerPath);

      const { stdout } = await execFileAsync('zig', zigArgs, {
        cwd: REPO_ROOT,
        maxBuffer: 10 * 1024 * 1024,
        env: {
          ...process.env,
          ZIG_GLOBAL_CACHE_DIR: globalCache,
          ZIG_LOCAL_CACHE_DIR: localCache,
        },
      });
      const structData = JSON.parse(stdout);

      for (const { qualifiedName, symbol } of structSymbols) {
        if (structData[qualifiedName]) {
          symbol.size = structData[qualifiedName].size;
          symbol.alignment = structData[qualifiedName].alignment;
          symbol.fields = structData[qualifiedName].fields;
        }
      }
    } catch (err) {
      console.warn('Failed to extract struct layouts:', err);
    }
  }

  modules.sort((a, b) => {
    if (a.sectionOrder !== b.sectionOrder) return a.sectionOrder - b.sectionOrder;
    return a.sourcePath.localeCompare(b.sourcePath);
  });

  const sections = modules.reduce<Map<string, ZigApiSection>>((groups, module) => {
    const existing = groups.get(module.sectionSlug);

    if (existing) {
      existing.modules.push(module);
      existing.symbolCount += module.symbolCount;
      return groups;
    }

    groups.set(module.sectionSlug, {
      slug: module.sectionSlug,
      title: module.section,
      description: module.sectionDescription,
      order: module.sectionOrder,
      modules: [module],
      symbolCount: module.symbolCount,
    });

    return groups;
  }, new Map());

  const codeLineCount = modules.reduce((count, module) => count + module.codeLineCount, 0);
  const exportCount = modules.reduce((count, module) => count + module.exportCount, 0);
  const memberCount = modules.reduce((count, module) => count + module.memberCount, 0);

  return {
    generatedAt: new Date().toISOString().split('T')[0],
    lastmod: latestMtime ? new Date(latestMtime).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
    moduleCount: modules.length,
    codeLineCount,
    exportCount,
    memberCount,
    symbolCount: exportCount + memberCount,
    sections: [...sections.values()].sort((a, b) => a.order - b.order),
    modules,
  };
}

export async function loadZigApi(): Promise<ZigApiIndex> {
  zigApiPromise ??= loadZigApiImpl();
  return zigApiPromise;
}
