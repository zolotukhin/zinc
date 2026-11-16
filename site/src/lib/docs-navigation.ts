// One task map for the docs hub and every guide's sidebar.
export const docNavSections = [
  {
    id: 'start',
    label: 'Start here',
    items: [
      {
        slug: 'getting-started',
        label: 'Install and first run',
        description: 'Choose a Linux release, download a model, and get your first response.',
      },
      {
        slug: 'running-zinc',
        label: 'Run and connect',
        description: 'Chat, manage models, start the API, and connect an app or agent.',
      },
    ],
  },
  {
    id: 'clients',
    label: 'Connect a client',
    items: [
      {
        slug: 'openclaw',
        label: 'OpenClaw',
        description: 'Configure a project, check the connection, and run a task.',
      },
      {
        slug: 'opencode',
        label: 'OpenCode',
        description: 'Set up local coding with bounded tools and independent project checks.',
      },
    ],
  },
  {
    id: 'reference',
    label: 'Reference',
    items: [
      {
        slug: 'api',
        label: 'HTTP API',
        description: 'Request fields, streaming, endpoints, errors, and backend differences.',
      },
      {
        slug: 'hardware-requirements',
        label: 'Hardware and memory',
        description: 'Qualified hardware, driver checks, and model/context sizing.',
      },
      {
        slug: 'rocm',
        label: 'AMD ROCm',
        description: 'Runtime dependencies, native GPU selection, and troubleshooting.',
      },
    ],
  },
  {
    id: 'development',
    label: 'Build and contribute',
    items: [
      {
        slug: 'development',
        label: 'Development',
        description: 'Build from source, run tests, debug, and validate client helpers.',
      },
      {
        slug: 'codebase-map',
        label: 'Codebase map',
        description: 'Find the source files that own each part of the engine.',
      },
      {
        slug: 'zig-api',
        label: 'Zig API',
        description: 'Generated implementation reference for modules, types and functions.',
      },
    ],
  },
  {
    id: 'engineering',
    label: 'Engineering and reports',
    items: [
      {
        slug: 'openclaw-amd-validation',
        label: 'OpenClaw AMD validation',
        description: 'Recorded client acceptance.',
      },
      {
        slug: 'intel-opencode-validation',
        label: 'Intel client validation',
        description: 'Recorded model and client outcomes.',
      },
      { slug: 'spec', label: 'Technical specification', description: 'Runtime architecture.' },
      { slug: 'rdna4-tuning', label: 'RDNA4 tuning', description: 'AMD performance controls.' },
      { slug: 'amd-gpu-reference', label: 'AMD GPU', description: 'Hardware reference.' },
      { slug: 'intel-gpu-reference', label: 'Intel GPU', description: 'Hardware reference.' },
      { slug: 'nvidia-gpu-reference', label: 'NVIDIA GPU', description: 'Hardware reference.' },
      {
        slug: 'apple-silicon-reference',
        label: 'Apple Silicon',
        description: 'Platform reference.',
      },
      { slug: 'apple-metal-reference', label: 'Apple Metal', description: 'Shader reference.' },
      { slug: 'turboquant-spec', label: 'TurboQuant', description: 'KV compression design.' },
    ],
  },
] as const;

// Retired pages are redirects, not duplicate documents in search or the sitemap.
export const mergedDocs = [
  { slug: 'binary-releases', sourcePath: 'docs/BINARY_RELEASES.md', target: 'getting-started' },
  { slug: 'agent-setup', sourcePath: 'docs/AGENT_SETUP.md', target: 'running-zinc' },
] as const;
