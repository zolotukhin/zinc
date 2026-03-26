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
  exportCount: number;
  memberCount: number;
  symbolCount: number;
  sections: ZigApiSection[];
  modules: ZigApiModule[];
}

interface ParsedComment extends ZigApiDocBlock {}

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

export async function loadZigApi(): Promise<ZigApiIndex> {
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

  const exportCount = modules.reduce((count, module) => count + module.exportCount, 0);
  const memberCount = modules.reduce((count, module) => count + module.memberCount, 0);

  return {
    generatedAt: new Date().toISOString().split('T')[0],
    lastmod: latestMtime ? new Date(latestMtime).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
    moduleCount: modules.length,
    exportCount,
    memberCount,
    symbolCount: exportCount + memberCount,
    sections: [...sections.values()].sort((a, b) => a.order - b.order),
    modules,
  };
}
