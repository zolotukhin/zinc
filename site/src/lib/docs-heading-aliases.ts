type HeadingAliases = Record<string, Record<string, string[]>>;

// Keep incoming section links when a guide corrects or scopes an old heading.
// IDs are route-specific; canonical navigation/search still uses the new title.
export const docHeadingAliases: HeadingAliases = {
  api: {
    'backend-coverage': ['compatibility-notes'],
    'capability-discovery': ['development-capability-discovery'],
    'hip-browser-chat-and-model-lifecycle': ['development-hip-browser-chat'],
    'status-and-compatibility-limits': [
      'status-known-gaps-read-first',
      'status--known-gaps-read-first',
    ],
  },
  benchmarking: {
    'latest-llama-cpp-and-matched-mtp-pairs': ['latest-llamacpp-and-matched-mtp-pairs'],
  },
  spec: {
    '4-4-shared-runtime-behavior': ['4-3-shared-runtime-behavior', '43-shared-runtime-behavior'],
    '10-3-vulkan-metal-memory-planner': ['10-3-shared-memory-planner', '103-shared-memory-planner'],
  },
  'getting-started': {
    'install-zinc': ['install-a-published-version'],
    'choose-your-backend': ['choose-an-archive'],
    'choose-a-model': ['check-the-installation-and-choose-a-model', 'models-in-current-work'],
    'what-to-do-next': [
      'start-the-chat-ui',
      'connect-a-coding-agent',
      'use-an-existing-gguf',
      'manage-cached-models',
      'what-to-read-next',
    ],
    'build-from-source': [
      'install-dependencies',
      'macos-apple-silicon',
      'linux-amd-or-intel-arc',
      'clone-and-build',
    ],
    'install-zinc-and-run-your-first-model': ['getting-started-with-zinc', 'zinc-binary-releases'],
  },
  'running-zinc': {
    'run-models-and-connect-apps': [
      'running-zinc',
      'connect-an-agent-or-application-to-zinc',
      'install-the-command',
      'build-the-binary',
    ],
    'structured-discovery': [
      'development-structured-discovery',
      'development-headless-serving-and-json-diagnostics',
      'development-structured-discovery-and-headless-serving',
    ],
    troubleshooting: [
      'run-a-preflight-check-first',
      'historical-development-node-diagnostics',
      'sample-outputs-from-the-shared-rdna4-node',
      'troubleshooting-start-here-if-something-fails',
      '1-the-first-prompt-is-silent-wrong-or-extremely-slow-on-rdna4',
      '2-no-vulkan-device-found-loader-init-failed',
      '3-model-does-not-fit-oom-during-load',
      '4-macos-metal-build-succeeds-but-check-reports-no-device',
      '5-still-stuck',
      'what-a-healthy-run-looks-like',
      'common-early-mistakes',
      'vulkan-is-present-but-the-wrong-device-is-selected',
      'the-model-path-is-wrong',
      'the-machine-builds-but-runtime-is-unstable',
      'fast-troubleshooting-commands',
    ],
    'models-and-cache': [
      'inspect-the-managed-model-catalog',
      'list-models',
      'activate-a-managed-model-in-a-running-server',
      'remove-a-managed-model-from-a-running-server',
    ],
    'terminal-prompts': ['run-a-single-prompt-from-the-terminal'],
    'hip-browser-chat': ['development-hip-browser-chat'],
    'device-context-and-kv-settings': [
      'pick-a-specific-gpu',
      'change-context-length',
      'kv-cache-configuration',
      'enable-turboquant-kv-compression',
    ],
    'start-the-api': ['run-zinc-as-a-server', 'stable-linux-0-1-0-workflow'],
    'connect-an-app': [
      'openai-compatible-api',
      'streaming-chat-completion',
      'non-streaming-chat-completion',
      'use-with-openai-sdks',
      'all-endpoints',
    ],
    'working-on-zinc-itself': ['rdna4-test-node-deploy', 'export-the-decode-graph', 'related-docs'],
  },
  'hardware-requirements': {
    'hardware-and-memory': ['zinc-hardware-requirements', 'shortest-path-to-success'],
    'supported-platforms': [
      'amd-gpus-linux',
      'intel-arc-gpus-linux',
      'other-vulkan-gpus',
      'nvidia-rtx-gpus-experimental-cuda',
      'apple-silicon-macos',
    ],
    'driver-checks': [
      'amd-vulkan-requirements',
      'amd-rocm-hip-requirements',
      'intel-requirements',
      'apple-silicon-requirements',
      'preflight-check',
      'quick-sanity-check',
      'linux-amd-vulkan',
      'linux-amd-rocm',
      'linux-intel-arc',
      'macos-apple-silicon',
      'linux-or-wsl2-nvidia-cuda-experimental',
      'on-linux-with-an-amd-gpu-through-vulkan',
      'on-linux-with-an-amd-rdna4-gpu-through-rocm',
      'on-linux-with-an-intel-arc-gpu',
      'on-macos-with-apple-silicon',
    ],
    'model-memory': [
      'amd-vram-guide',
      'intel-vram-guide',
      'apple-silicon-memory-guide',
      'model-catalog',
      'system-requirements-both-platforms',
    ],
  },
  rocm: {
    'browser-chat-on-hip': ['development-browser-chat-on-hip'],
    'amd-hip-rocm-setup': ['amd-rocm-setup', 'rocm-backend', 'install-and-run', 'run-a-model'],
    'runtime-prerequisites': ['validated-stack'],
    'serving-limits': ['http-api-coverage', 'api-client-notes'],
    'context-kv-cache-and-mtp': ['long-coding-sessions', 'fair-llama-cpp-comparison'],
    'build-from-source': ['build-and-verify'],
  },
  releasing: {
    'published-0-1-1': ['prepared-0-1-1-candidate'],
  },
  openclaw: {
    'tested-scope': ['repeatable-validation'],
  },
  opencode: {
    'opencode-with-zinc': ['opencode-coding-harness'],
    'context-and-tools': ['prompt-reuse-and-project-directory'],
    'run-a-headless-task': ['try-a-real-task-in-a-worktree'],
    'tested-scope': ['check-the-harness-without-a-model', 'evaluate-coding-tasks'],
  },
};

/// Runs after heading navigation has assigned canonical IDs. Markdown raw HTML
/// stays disabled; these fixed alias spans are generated from the trusted map.
export function rehypeDocHeadingAliases({ slug }: { slug: string }) {
  return (tree: any) => {
    const walk = (node: any) => {
      if (!node || typeof node !== 'object') return;
      if (node.type === 'element' && /^h[1-6]$/.test(node.tagName)) {
        const ids = docHeadingAliases[slug]?.[node.properties?.id] ?? [];
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
