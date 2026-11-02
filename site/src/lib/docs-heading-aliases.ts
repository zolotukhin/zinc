type HeadingAliases = Record<string, Record<string, string[]>>;

// Keep incoming section links when a guide corrects or scopes an old heading.
// IDs are route-specific; canonical navigation/search still uses the new title.
const aliases: HeadingAliases = {
  api: {
    'backend-coverage': ['compatibility-notes'],
    'status-and-compatibility-limits': [
      'status-known-gaps-read-first',
      'status--known-gaps-read-first',
    ],
  },
  'running-zinc': {
    'kv-cache-configuration': ['enable-turboquant-kv-compression'],
  },
  spec: {
    '4-4-shared-runtime-behavior': ['4-3-shared-runtime-behavior', '43-shared-runtime-behavior'],
    '10-3-vulkan-metal-memory-planner': ['10-3-shared-memory-planner', '103-shared-memory-planner'],
  },
};

/// Runs after heading navigation has assigned canonical IDs. Markdown raw HTML
/// stays disabled; these fixed alias spans are generated from the trusted map.
export function rehypeDocHeadingAliases({ slug }: { slug: string }) {
  return (tree: any) => {
    const walk = (node: any) => {
      if (!node || typeof node !== 'object') return;
      if (node.type === 'element' && /^h[1-6]$/.test(node.tagName)) {
        const ids = aliases[slug]?.[node.properties?.id] ?? [];
        node.children = [
          ...(node.children ?? []),
          ...ids.map((id) => ({
            type: 'element',
            tagName: 'span',
            properties: { id },
            children: [],
          })),
        ];
      }
      if (Array.isArray(node.children)) node.children.forEach(walk);
    };
    walk(tree);
  };
}
