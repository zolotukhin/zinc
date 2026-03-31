import type { DocPage } from './docs-loader';
import type { ZigApiIndex, ZigApiDocBlock, ZigApiMember, ZigApiModule, ZigApiSection, ZigApiSymbol } from './zig-api-loader';

export interface DocsSearchEntry {
  title: string;
  url: string;
  kind: 'Guide' | 'Section' | 'API Section' | 'API Module' | 'API Symbol' | 'API Method';
  group: string;
  preview: string;
  searchText: string;
}

interface MarkdownSection {
  heading: string;
  url: string;
  preview: string;
}

function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

function stripMarkdown(value: string): string {
  return normalizeWhitespace(
    value
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/<[^>]+>/g, ' ')
      .replace(/[*_~>#-]+/g, ' ')
  );
}

function truncate(value: string, maxChars: number): string {
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars - 1).trimEnd()}…`;
}

function slugifyHeading(text: string): string {
  const normalized = text
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return normalized || 'section';
}

function collectMarkdownSections(doc: DocPage): MarkdownSection[] {
  const sections: MarkdownSection[] = [];
  const seen = new Map<string, number>();
  const lines = doc.content.split(/\r?\n/);

  let currentHeading = '';
  let currentSlug = '';
  let currentDepth = 0;
  let currentBody: string[] = [];

  const flushCurrent = () => {
    if (currentDepth < 2 || currentHeading.length === 0) return;
    const preview = truncate(stripMarkdown(currentBody.join('\n')), 220);
    sections.push({
      heading: currentHeading,
      url: `/zinc/docs/${doc.slug}#${currentSlug}`,
      preview,
    });
  };

  for (const line of lines) {
    const headingMatch = /^(#{1,6})\s+(.+?)\s*$/.exec(line);
    if (headingMatch) {
      flushCurrent();

      currentDepth = headingMatch[1].length;
      currentHeading = stripMarkdown(headingMatch[2]);
      currentBody = [];

      const baseSlug = slugifyHeading(currentHeading);
      const seenCount = seen.get(baseSlug) ?? 0;
      seen.set(baseSlug, seenCount + 1);
      currentSlug = seenCount > 0 ? `${baseSlug}-${seenCount + 1}` : baseSlug;
      continue;
    }

    if (currentHeading.length > 0) currentBody.push(line);
  }

  flushCurrent();
  return sections;
}

function searchDocText(docBlock: ZigApiDocBlock): string {
  return normalizeWhitespace([
    docBlock.summary,
    ...docBlock.description,
    ...docBlock.params.map(param => `${param.name} ${param.description}`),
    docBlock.returns ?? '',
    ...docBlock.notes,
  ].join(' '));
}

function symbolPreview(symbol: ZigApiSymbol): string {
  const fieldSummary = symbol.fields?.map(field => `${field.name} ${field.type}`).join(' ') ?? '';
  const memberSummary = symbol.members.map(member => member.qualifiedName).join(' ');
  return truncate(normalizeWhitespace([
    symbol.signature,
    searchDocText(symbol.doc),
    fieldSummary,
    memberSummary,
  ].join(' ')), 220);
}

function memberPreview(member: ZigApiMember): string {
  return truncate(normalizeWhitespace([
    member.signature,
    searchDocText(member.doc),
  ].join(' ')), 220);
}

function modulePreview(module: ZigApiModule): string {
  return truncate(normalizeWhitespace([
    module.summary,
    ...module.overview,
    module.symbols.map(symbol => symbol.qualifiedName).join(' '),
  ].join(' ')), 220);
}

function pagePreview(doc: DocPage): string {
  const withoutFirstHeading = doc.content.replace(/^#\s+.+$/m, '');
  return truncate(stripMarkdown(withoutFirstHeading), 220);
}

function uniqueByUrlAndTitle(entries: DocsSearchEntry[]): DocsSearchEntry[] {
  const seen = new Set<string>();
  return entries.filter(entry => {
    const key = `${entry.url}::${entry.title}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function buildDocsSearchIndex(docs: DocPage[], zigApi: ZigApiIndex): DocsSearchEntry[] {
  const entries: DocsSearchEntry[] = [];

  for (const doc of docs) {
    const title = doc.title.trim();
    const preview = pagePreview(doc);
    entries.push({
      title,
      url: `/zinc/docs/${doc.slug}`,
      kind: 'Guide',
      group: 'ZINC Docs',
      preview,
      searchText: normalizeWhitespace(`${title} ${preview}`),
    });

    for (const section of collectMarkdownSections(doc)) {
      entries.push({
        title: section.heading,
        url: section.url,
        kind: 'Section',
        group: title,
        preview: section.preview,
        searchText: normalizeWhitespace(`${title} ${section.heading} ${section.preview}`),
      });
    }
  }

  for (const section of zigApi.sections) {
    entries.push({
      title: section.title,
      url: `/zinc/docs/zig-api#${section.slug}`,
      kind: 'API Section',
      group: 'Zig API',
      preview: truncate(section.description, 220),
      searchText: normalizeWhitespace(`${section.title} ${section.description}`),
    });

    for (const module of section.modules) {
      entries.push({
        title: module.title,
        url: module.href,
        kind: 'API Module',
        group: section.title,
        preview: modulePreview(module),
        searchText: normalizeWhitespace([
          section.title,
          module.title,
          module.sourcePath,
          module.summary,
          ...module.overview,
          module.symbols.map(symbol => symbol.qualifiedName).join(' '),
        ].join(' ')),
      });

      for (const symbol of module.symbols) {
        entries.push({
          title: symbol.qualifiedName,
          url: symbol.href,
          kind: 'API Symbol',
          group: module.title,
          preview: symbolPreview(symbol),
          searchText: normalizeWhitespace([
            section.title,
            module.title,
            symbol.qualifiedName,
            symbol.signature,
            searchDocText(symbol.doc),
            symbol.members.map(member => member.qualifiedName).join(' '),
          ].join(' ')),
        });

        for (const member of symbol.members) {
          entries.push({
            title: member.qualifiedName,
            url: member.href,
            kind: 'API Method',
            group: module.title,
            preview: memberPreview(member),
            searchText: normalizeWhitespace([
              section.title,
              module.title,
              symbol.qualifiedName,
              member.qualifiedName,
              member.signature,
              searchDocText(member.doc),
            ].join(' ')),
          });
        }
      }
    }
  }

  return uniqueByUrlAndTitle(entries);
}
