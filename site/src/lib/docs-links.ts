type DocLinkTarget = { slug: string; sourcePath: string };
type DocLinkOptions = { sourcePath: string; docs: readonly DocLinkTarget[] };

const repositoryDirectories = [
  'src/',
  'benchmarks/',
  'tools/',
  'scripts/',
  'tests/',
  'loops/',
  'site/',
];
const repositoryFiles = new Set([
  'build.zig',
  'build.zig.zon',
  'package.json',
  'bun.lock',
  'bunfig.toml',
  'flake.nix',
  'flake.lock',
  '.zig-version',
  '.bun-version',
  'LICENSE',
]);

function isRepositorySource(path: string): boolean {
  return (
    repositoryFiles.has(path) ||
    repositoryDirectories.some((directory) => path.startsWith(directory))
  );
}

export function createDocLinkResolver({ sourcePath, docs }: DocLinkOptions) {
  const routes = new Map(docs.map((doc) => [doc.sourcePath, `/zinc/docs/${doc.slug}/`]));
  return (href: string): string => {
    if (!href || /^(?:[a-z][a-z\d+.-]*:|\/|#)/i.test(href)) return href;
    try {
      // Authors also use bare repository paths such as src/server/runtime.zig
      // or build.zig. Resolve those from the repo root, not the published URL.
      const barePath = decodeURIComponent(href.split(/[?#]/, 1)[0].replace(/^\.\//, ''));
      const base =
        isRepositorySource(barePath) || routes.has(barePath)
          ? 'https://repo.invalid/'
          : `https://repo.invalid/${sourcePath}`;
      const target = new URL(href, base);
      const path = decodeURIComponent(target.pathname.slice(1));
      if (/\.md$/i.test(target.pathname)) {
        const route = routes.get(path);
        if (route) return `${route}${target.search}${target.hash}`;
      } else if (!isRepositorySource(path)) return href;
      const kind = target.pathname.endsWith('/') ? 'tree' : 'blob';
      return `https://github.com/zolotukhin/zinc/${kind}/main${target.pathname}${target.search}${target.hash}`;
    } catch {
      return href;
    }
  };
}

export function rehypeDocLinks(options: DocLinkOptions) {
  const resolve = createDocLinkResolver(options);
  return (tree: any) => {
    const walk = (node: any) => {
      if (!node || typeof node !== 'object') return;
      if (
        node.type === 'element' &&
        node.tagName === 'a' &&
        typeof node.properties?.href === 'string'
      ) {
        node.properties.href = resolve(node.properties.href);
      }
      if (Array.isArray(node.children)) node.children.forEach(walk);
    };
    walk(tree);
  };
}
