export type TopicHubLink = {
  title: string;
  href: string;
  description: string;
};

export type TopicHubFaq = {
  question: string;
  answer: string;
};

export type TopicHubBriefItem = {
  label: string;
  detail: string;
};

type TopicHubBase = {
  slug: string;
  title: string;
  shortTitle: string;
  description: string;
  keywords: string;
  summary: string;
  updated?: string;
  readNext: TopicHubLink[];
  docs: TopicHubLink[];
  related: string[];
  faqs: TopicHubFaq[];
};

export type TopicHub = TopicHubBase & ({ guide: 'qwen38' } | {
  guide?: undefined;
  practicalAnswer: string;
  bestUse: string;
  status: TopicHubBriefItem[];
  actionPlan: TopicHubBriefItem[];
  checklist: TopicHubBriefItem[];
  measurements: TopicHubBriefItem[];
  articleIdeas: TopicHubBriefItem[];
  pitfalls: TopicHubBriefItem[];
  whatMatters: string[];
});

export const topicHubs: TopicHub[] = [
  {
    slug: 'openclaw-local-models',
    title: 'OpenClaw with Local Models and ZINC',
    shortTitle: 'OpenClaw',
    description: 'Connect OpenClaw to local models with ZINC: AMD GPU setup, model selection, live context discovery, tool calls, and tested R9700 workflows.',
    keywords: 'OpenClaw local models, OpenClaw ZINC, OpenClaw AMD GPU, OpenClaw R9700, OpenClaw RX 9070 XT, OpenClaw local inference, OpenAI compatible API',
    summary: 'Connect OpenClaw to a model running on your GPU. ZINC provides the local inference API; OpenClaw manages the agent workflow and executes tools. Start with the setup guide, then follow the hardware guidance and recorded model tests.',
    practicalAnswer: 'Install ZINC for your GPU, pull a suitable model, and start its HTTP API with enough context for the agent. Download the standalone OpenClaw helper, run setup for an existing project, and run its connection check before a task. The helper discovers the loaded model and actual context capacity; it needs no ZINC source checkout.',
    bestUse: 'Use this page to connect OpenClaw to a local model, choose a context budget, or diagnose a failed tool round trip. File-edit tasks are examples of the integration; OpenClaw is an agent platform with broader uses.',
    status: [
      {
        label: 'Connection path',
        detail: 'OpenClaw connects to the ZINC chat-completions API. The setup guide records the tested client version and Node.js requirements.',
      },
      {
        label: 'Hardware evidence',
        detail: 'The AMD acceptance report records R9700 tests for Vulkan and ROCm. The RX 9070 XT walkthrough provides memory guidance, not a separate qualification of that card.',
      },
    ],
    actionPlan: [
      { label: 'Start the model server', detail: 'Follow the binary installation and OpenClaw setup guides. Qwen 3.5 9B is a starting point when it fits your GPU; the guide uses 16K context and one request slot.' },
      { label: 'Configure OpenClaw', detail: 'Download the standalone helper and run setup for your project. It creates a dedicated configuration without replacing your global OpenClaw settings.' },
      { label: 'Check before a task', detail: 'Run the helper check to verify the client configuration, ZINC health, model identity and available context. Check task outcomes independently afterward.' },
    ],
    checklist: [
      { label: 'Use the loaded model ID', detail: 'Discover the server model instead of guessing its name. Native ROCm discovery can use a GGUF filename as the ID.' },
      { label: 'Match the live context', detail: 'Configure server memory first. A model’s advertised context window does not allocate that capacity on your GPU.' },
      { label: 'Keep configuration portable', detail: 'Use the documented environment variables for custom endpoints and API keys. Rediscover model and context after changing the server.' },
    ],
    measurements: [
      { label: 'Tool round trips', detail: 'Check that the model produces complete tool calls and can use the returned results in the next turn.' },
      { label: 'Task outcome', detail: 'Verify the requested result separately from the agent’s final message. The published file-edit trials include independent tests and server recovery checks.' },
      { label: 'Latency and memory', detail: 'Record time to first useful action, total task time, context and GPU memory. Token throughput alone does not establish agent reliability.' },
    ],
    articleIdeas: [
      { label: 'Choose a model and context', detail: 'Use the acceptance matrix and memory guidance to select a starting configuration for your own workflow.' },
      { label: 'Connect a remote GPU', detail: 'Follow the setup guide’s custom-endpoint instructions when OpenClaw and ZINC run on different machines.' },
    ],
    pitfalls: [
      { label: 'Assuming every model is suitable', detail: 'Passing an inference test does not establish agent task quality. The small installation-test model is not recommended for agent tasks.' },
      { label: 'Copying a larger card’s settings', detail: 'The R9700 has a different memory budget from a 16 GB RX 9070 XT. Check model weights, context and runtime memory before copying a configuration.' },
      { label: 'Confusing inference and tool execution', detail: 'ZINC returns model responses and proposed tool calls. OpenClaw executes tools on the machine where the client runs.' },
    ],
    whatMatters: [
      'Use chat completions and set streaming explicitly when configuring a client directly.',
      'The standalone helper checks configuration and discovery before running an agent task.',
      'The recorded AMD results belong to their exact model, backend, context and client settings.',
    ],
    readNext: [
      {
        title: 'OpenCode and OpenClaw on AMD: R9700 and RX 9070 XT',
        href: '/blog/2026-11-07-opencode-openclaw-amd-r9700-rx-9070-xt/',
        description: 'A Linux walkthrough covering local models, GPU memory and the tests behind the setup.',
      },
      {
        title: 'OpenClaw AMD model acceptance',
        href: '/zinc/docs/openclaw-amd-validation/',
        description: 'Recorded model outcomes, context settings, failures and qualification limits.',
      },
    ],
    docs: [
      { title: 'Configure OpenClaw with ZINC', href: '/zinc/docs/openclaw/', description: 'Client requirements, server startup, setup, connection checks and task execution.' },
      { title: 'Download the OpenClaw helper', href: '/zinc/harnesses/setup-openclaw.mjs', description: 'Standalone setup, checks and task execution using Node’s standard library.' },
      { title: 'Agent and app setup', href: '/zinc/docs/agent-setup/', description: 'An installed-first workflow with model discovery and release compatibility guidance.' },
      { title: 'Hardware and memory requirements', href: '/zinc/docs/hardware-requirements/', description: 'Choose a backend and size the model and context for your GPU.' },
    ],
    related: ['opencode-local-coding', 'amd-rdna4-llm-inference', 'qwen3-8-local-inference'],
    faqs: [
      { question: 'Can OpenClaw use a local model served by ZINC?', answer: 'Yes. The OpenClaw setup guide connects a dedicated client configuration to ZINC’s OpenAI-compatible chat API. Run the helper check with your loaded model before submitting a task.' },
      { question: 'Do I need a ZINC source checkout?', answer: 'No. Install the published Linux binary for your backend and download the standalone OpenClaw helper. Features marked Development in the guides require a newer source build.' },
      { question: 'Can I use an AMD GPU without CUDA?', answer: 'Yes. ZINC provides Vulkan and ROCm backends for AMD. OpenClaw talks to the API; choose the GPU runtime and driver requirements documented for your backend.' },
      { question: 'Does a successful connection prove the model can finish a task?', answer: 'No. Configuration, health and model discovery checks establish a working connection. Use the recorded acceptance results as a starting point and verify your own task outcomes independently.' },
    ],
  },
  {
    slug: 'opencode-local-coding',
    title: 'OpenCode Local Coding with Qwen and ZINC',
    shortTitle: 'OpenCode',
    description: 'A practical guide to OpenCode local coding with Qwen and ZINC: local LLM provider config, tool calls, thinking mode, context limits, SSH tunnels, and trace proxy debugging.',
    keywords: 'OpenCode local coding, OpenCode local LLM, OpenCode Qwen, Qwen coding model, OpenCode ZINC, local coding agent, OpenCode OpenAI compatible provider, ZINC tool calling, self-hosted coding assistant',
    summary: 'OpenCode can use ZINC and Qwen through the same OpenAI-compatible `/v1/chat/completions` API that powers the browser chat UI. Run ZINC, use the portable OpenCode setup script, verify chat and streaming tool calls, and run a headless task. Optional transparent traces help diagnose failures.',
    practicalAnswer: 'For OpenCode local coding, run ZINC on a localhost `/v1` endpoint and configure a custom `@ai-sdk/openai-compatible` provider. Use a Qwen-family model with stable thinking and tool behavior, keep `ZINC_TOOL_CALLING` enabled, and match the actual server context, with a 16384-token context and 4096-token output as the setup defaults. If OpenCode runs on a laptop while ZINC runs on a GPU box, use an SSH tunnel so the OpenCode config still contains only `127.0.0.1`.',
    bestUse: 'Use this page when you want a local coding assistant backed by ZINC rather than a hosted API. It is focused on operator setup, not benchmark marketing: provider config, local ports, tool calls, thinking, context budgets, and trace-based debugging.',
    status: [
      {
        label: 'Best current answer',
        detail: 'OpenCode works best through ZINC\'s OpenAI-compatible chat endpoint, with the optional trace proxy in front during compatibility testing.',
      },
      {
        label: 'Reader problem',
        detail: 'Users need the exact local-provider shape without leaking private GPU-node addresses or hardcoding machine-specific secrets into a repo.',
      },
      {
        label: 'Main bottleneck',
        detail: 'Correct agent behavior depends on tool-call round trips, enough output budget, and enough context for source snapshots. Raw tokens per second is only one part of usability.',
      },
    ],
    actionPlan: [
      {
        label: 'Start ZINC locally',
        detail: 'Start a Qwen model with a context large enough for source files and tool results, then run the setup connection check.',
      },
      {
        label: 'Configure one provider',
        detail: 'Use `tools/setup_opencode.mjs` to create `opencode.zinc.json` and keep runtime endpoints and API keys outside the saved config.',
      },
      {
        label: 'Keep tools on',
        detail: 'Do not disable `ZINC_TOOL_CALLING` for coding sessions. ZINC parses model tool calls; OpenCode executes local tools.',
      },
      {
        label: 'Tunnel remote servers',
        detail: 'If the GPU is remote, forward it with SSH and keep the OpenCode config pointed at localhost. Never publish private hosts, ports, usernames, or model paths.',
      },
    ],
    checklist: [
      {
        label: 'Health check first',
        detail: 'Run the setup script with `check` to verify health, JSON completions, and JSON/SSE tool calls from the OpenCode machine.',
      },
      {
        label: 'Model shape',
        detail: 'Use a Qwen/ChatML-family coding model when tool calling matters, and keep the `model` id consistent between ZINC and OpenCode.',
      },
      {
        label: 'Context budget',
        detail: 'Set OpenCode `limit.context` at or below the context length ZINC actually has available for the loaded model.',
      },
      {
        label: 'Trace locally',
        detail: 'The trace proxy forwards requests and responses unchanged by default. Keep its synthetic compatibility mode out of coding evaluations.',
      },
    ],
    measurements: [
      {
        label: 'Tool-call success',
        detail: 'The key coding metric is whether OpenCode can read, edit, write, run tests, and incorporate tool results in the next turn.',
      },
      {
        label: 'First useful action',
        detail: 'Measure how long it takes to produce the first correct read or edit tool call, not only final answer latency.',
      },
      {
        label: 'Context usage',
        detail: 'Track prompt tokens, visible source snapshots, and output budget so local runs fail predictably instead of silently truncating.',
      },
      {
        label: 'Trace deltas',
        detail: 'Compare direct ZINC requests against proxy traces to see whether failures come from model output, client behavior, or compatibility repair.',
      },
    ],
    articleIdeas: [
      {
        label: 'Local coding agent smoke test',
        detail: 'A small reproducible project where OpenCode must read files, patch a bug, and run tests against ZINC.',
      },
      {
        label: 'Tool calling on Qwen in practice',
        detail: 'Show the exact request, generated `<tool_call>`, structured OpenAI response, and follow-up tool-result prompt.',
      },
      {
        label: 'Which Qwen model for local OpenCode',
        detail: 'Compare Qwen3.6 35B-A3B, Qwen3.6 dense, and smaller Qwen targets for local coding latency, tool-call quality, and context headroom.',
      },
      {
        label: 'Remote GPU, local editor',
        detail: 'A security-focused walkthrough for SSH tunnels, localhost configs, traces, and avoiding credential leaks.',
      },
    ],
    pitfalls: [
      {
        label: 'Publishing private endpoints',
        detail: 'Do not paste GPU-node IPs, SSH ports, usernames, raw model paths, API keys, or trace files into docs, commits, screenshots, or public issues.',
      },
      {
        label: 'Using raw completions',
        detail: 'OpenCode needs chat completions and tool semantics. `/v1/completions` is not the right endpoint for agentic coding.',
      },
      {
        label: 'Undersizing output',
        detail: 'A tiny output cap can look like model failure because the response stops before the tool call or final summary is complete.',
      },
    ],
    whatMatters: [
      'Use `/v1/chat/completions`; it is the endpoint with streaming, thinking, and tool-call support.',
      'ZINC never executes tools. OpenCode executes tools locally and sends tool results back to ZINC.',
      'Thinking mode is a request setting; the trace proxy can force it when the client does not expose a direct toggle.',
      'A localhost SSH tunnel is the safest way to use a remote GPU box from a laptop without leaking private infrastructure.',
    ],
    readNext: [
      {
        title: 'OpenAI-compatible tool calling design',
        href: '/zinc/docs/api/#tool-calling',
        description: 'The ZINC API behavior that OpenCode relies on for tool definitions, tool calls, and tool results.',
      },
      {
        title: 'Qwen 3.8 local inference',
        href: '/topics/qwen3-8-local-inference/',
        description: 'Why Qwen-family models are the main target for thinking and tool-capable local coding sessions.',
      },
      {
        title: 'ZINC performance status',
        href: '/zinc/benchmarks/',
        description: 'Current AMD, Intel Arc, Metal, and CUDA results for the models you may use as local coding backends.',
      },
    ],
    docs: [
      {
        title: 'Download the OpenCode setup script',
        href: '/zinc/harnesses/setup-opencode.mjs',
        description: 'Portable configuration, API checks, and headless execution in one Bun script.',
      },
      {
        title: 'Configure OpenCode with ZINC',
        href: '/zinc/docs/opencode/',
        description: 'Step-by-step OpenCode local LLM provider config for ZINC and Qwen, including proxy setup, thinking, tool calling, context limits, and troubleshooting.',
      },
      {
        title: 'Serving HTTP API',
        href: '/zinc/docs/api/',
        description: 'OpenAI-compatible chat completions, streaming, tools, thinking, models, and health endpoints.',
      },
      {
        title: 'Running ZINC',
        href: '/zinc/docs/running-zinc/',
        description: 'Start the local server, choose managed models, set context, and verify the runtime path.',
      },
    ],
    related: ['openclaw-local-models', 'qwen3-8-local-inference', 'amd-rdna4-llm-inference', 'gemma-local-inference'],
    faqs: [
      {
        question: 'Can OpenCode use ZINC as a local model backend?',
        answer: 'Yes. Configure OpenCode with an OpenAI-compatible provider whose base URL points at the ZINC `/v1` endpoint, usually `http://127.0.0.1:9090/v1` or a local trace proxy.',
      },
      {
        question: 'Which Qwen model should I use for OpenCode local coding?',
        answer: 'Start with a Qwen/ChatML-family model because ZINC validates thinking and tool calling on that template family. Use a smaller Qwen model for setup tests and a larger Qwen3.6 target when you have enough memory and want better coding behavior.',
      },
      {
        question: 'Should I expose the GPU server directly to OpenCode?',
        answer: 'Usually no. Keep ZINC bound to localhost on the GPU machine and use an SSH tunnel from the OpenCode machine. The committed config should still contain only localhost URLs and environment placeholders.',
      },
      {
        question: 'Does ZINC run the tools?',
        answer: 'No. ZINC renders tool definitions and returns structured tool calls. OpenCode executes local tools, then sends the tool results back to ZINC in the next chat request.',
      },
    ],
  },
  {
    slug: 'gemma-local-inference',
    title: 'Gemma 4 Local Inference',
    shortTitle: 'Gemma 4',
    description: 'A clean guide to Gemma 4 local inference: model fit, MoE vs dense variants, sliding-window attention, asymmetric GQA, Vulkan, Metal, and ZINC.',
    keywords: 'Gemma 4 local inference, Gemma 4 AMD GPU, Gemma 4 RDNA4, Gemma 4 Metal, Gemma 4 Vulkan, Gemma MoE inference, local LLM Gemma',
    summary: 'Gemma 4 is a useful local inference target because it stresses the parts of an engine that simple Llama-shaped models do not: sliding-window attention, asymmetric grouped-query attention, Gemma-specific normalization, and MoE routing on the A4B variant.',
    practicalAnswer: 'If you want to run Gemma locally, treat Gemma 4 as an architecture port, not just another GGUF file. Dense Gemma 4 is mostly a memory and attention-shape problem. Gemma 4 26B-A4B adds sparse routing and Gemma-specific FFN behavior. On ZINC, the practical path is to use the managed Gemma model ids, verify the benchmark dashboard for the current backend, and expect the RDNA4 and Metal paths to improve as the Gemma prefill work lands.',
    bestUse: 'Use this page when you are deciding whether Gemma is a model-family port, a benchmark target, or a writing cluster. The useful reader intent is specific: what breaks differently from Qwen, what fits locally, and what an AMD or Metal engine has to optimize next.',
    status: [
      {
        label: 'Best current answer',
        detail: 'Gemma is the right second pillar for ZINC coverage because it proves the engine is not only tuned for Qwen-shaped models.',
      },
      {
        label: 'Reader problem',
        detail: 'People landing here need to know which Gemma variant fits, why it is architecturally different, and what performance number is credible on their GPU.',
      },
      {
        label: 'Main bottleneck',
        detail: 'For larger Gemma runs, time-to-first-token is the risk area: sliding-window attention, asymmetric Q/KV dimensions, and command submission overhead show up before decode looks bad.',
      },
    ],
    actionPlan: [
      {
        label: 'Start from the model shape',
        detail: 'Separate dense Gemma runs from A4B MoE runs before comparing numbers. They stress different kernels and memory paths.',
      },
      {
        label: 'Measure prefill first',
        detail: 'Gemma can look healthy in decode while still losing badly on time-to-first-token because prompt batching and attention shape are the hard parts.',
      },
      {
        label: 'Keep Gemma posts practical',
        detail: 'The strongest future posts should answer a concrete local-running question: head dimensions, sliding windows, MoE routing, Metal parity, or RDNA4 prefill.',
      },
      {
        label: 'Link back to benchmarks',
        detail: 'Readers landing from search need the current ZINC result, the llama.cpp baseline if available, and the exact hardware class.',
      },
    ],
    checklist: [
      {
        label: 'Name the checkpoint',
        detail: 'Use the exact model id, quantization, and whether it is dense or A4B MoE before making any claim.',
      },
      {
        label: 'Confirm backend support',
        detail: 'Check whether the run is using the batched Gemma path or a fallback path; the difference changes the conclusion.',
      },
      {
        label: 'Capture fit and context',
        detail: 'Record VRAM budget, reserved KV cache, active context, and whether any experts or tensors are offloaded.',
      },
      {
        label: 'Compare user-visible latency',
        detail: 'Report TTFT and prompt throughput before decode tokens per second. Gemma pain is often visible before generation starts.',
      },
    ],
    measurements: [
      {
        label: 'TTFT',
        detail: 'The first metric for Gemma posts should be time-to-first-token on a fixed prompt length.',
      },
      {
        label: 'Prompt tok/s',
        detail: 'Prefill throughput tells whether the sliding-window and full-attention layers are batched correctly.',
      },
      {
        label: 'Decode tok/s',
        detail: 'Decode still matters, but it should be paired with model size, active parameters, and context length.',
      },
      {
        label: 'Memory shape',
        detail: 'Show weights, runtime buffers, reserved KV cache, and any offload mode separately.',
      },
    ],
    articleIdeas: [
      {
        label: 'Gemma 4 on 16 GB vs 32 GB RDNA4',
        detail: 'A practical fit guide for RX 9070 XT-class cards versus R9700-class cards.',
      },
      {
        label: 'Sliding-window attention in local inference',
        detail: 'Explain what Gemma saves, what it does not save, and where full-attention layers still dominate.',
      },
      {
        label: 'Gemma A4B MoE versus Qwen A3B MoE',
        detail: 'A direct comparison of routing, active parameters, memory residency, and prefill behavior.',
      },
    ],
    pitfalls: [
      {
        label: 'Assuming Qwen kernels are enough',
        detail: 'Gemma changes attention dimensions, activation behavior, norm placement, and sometimes MoE layout. A generic transformer path is not a full answer.',
      },
      {
        label: 'Publishing decode-only conclusions',
        detail: 'Large Gemma prompts are often prefill-bound. Decode tokens per second alone can make the local experience look better than it is.',
      },
      {
        label: 'Writing generic model coverage',
        detail: 'Searchers do not need another broad Gemma overview. The opportunity is local inference mechanics on real hardware.',
      },
    ],
    whatMatters: [
      'Gemma 4 uses sliding-window attention for most layers, so long-context memory does not grow the same way on every layer.',
      'Full-attention Gemma layers can use different Q and KV head dimensions, which breaks kernels that assume one shared head_dim.',
      'Gemma MoE is not identical to Qwen MoE: activations, norms, and residual placement differ.',
      'Prefill performance depends on batching the prompt correctly; a per-token path wastes bandwidth on large Gemma models.',
    ],
    readNext: [
      {
        title: 'The single push constant blocking Gemma 4 prefill on RDNA4',
        href: '/blog/2026-04-24-the-single-push-constant-blocking-gemma-4-prefill-on-rdna4/',
        description: 'Why Gemma 4 forces a split between Q and KV head dimensions in Vulkan flash attention.',
      },
      {
        title: 'Why one vkQueueSubmit per prompt matters for Gemma 4',
        href: '/blog/2026-04-25-why-one-vkqueuesubmit-per-prompt-is-the-next-quiet-rdna4-prefill-unlock/',
        description: 'How 60-layer Gemma prefill exposes command submission overhead on AMD.',
      },
      {
        title: 'How MoE models work in ZINC',
        href: '/blog/2026-04-04-how-moe-models-work-in-zinc/',
        description: 'The shared routing ideas and the Gemma-specific differences in sparse inference.',
      },
      {
        title: 'Why FP16 KV cache is the wrong default for 128k context',
        href: '/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4/',
        description: 'How Gemma sliding-window attention changes, but does not remove, the KV memory problem.',
      },
    ],
    docs: [
      {
        title: 'Getting Started with ZINC',
        href: '/zinc/docs/getting-started/',
        description: 'Build ZINC, pull a managed model, and run local inference on AMD, Intel Arc, or Apple Silicon.',
      },
      {
        title: 'ZINC Benchmarks',
        href: '/zinc/benchmarks/',
        description: 'Current same-machine benchmark results for Gemma, Qwen, AMD, Metal, and baseline comparisons.',
      },
      {
        title: 'Running ZINC',
        href: '/zinc/docs/running-zinc/',
        description: 'CLI and server usage, managed model ids, backend selection, and runtime flags.',
      },
    ],
    related: ['qwen3-8-local-inference', 'amd-rdna4-llm-inference', 'kv-cache-quantization'],
    faqs: [
      {
        question: 'Can Gemma 4 run locally in ZINC?',
        answer: 'Yes. ZINC has managed Gemma 4 targets in the catalog, including dense and MoE variants. The exact performance depends on backend, memory budget, and whether the path is using per-token fallback or batched prefill.',
      },
      {
        question: 'Why is Gemma harder than a normal dense transformer?',
        answer: 'Gemma 4 combines sliding-window attention, asymmetric grouped-query attention on full-attention layers, Gemma-specific norms, and in some variants MoE routing. Each of those changes assumptions inside inference kernels.',
      },
      {
        question: 'Should future blog posts focus on Gemma?',
        answer: 'Yes. Gemma is a strong follow-up cluster because it is technically distinct from Qwen and creates useful comparisons around sliding-window attention, memory, MoE routing, Vulkan, and Metal.',
      },
    ],
  },
  {
    guide: 'qwen38',
    slug: 'qwen3-8-local-inference',
    title: 'Qwen 3.8 27B local inference',
    shortTitle: 'Qwen 3.8',
    updated: '2026-11-15',
    description: 'Run Qwen 3.8 27B locally with ZINC: GGUF setup, AMD and Intel support, Apple Silicon preview, matched benchmarks, MTP and context memory.',
    keywords: 'Qwen 3.8, Qwen3.8 27B, Qwen 3.8 local inference, Qwen3.8 GGUF, Qwen 3.8 AMD, Qwen 3.8 Intel Arc, Qwen 3.8 Apple Silicon, Qwen MTP, ZINC',
    summary: 'Qwen 3.8 27B is a dense hybrid model for local chat and coding. Start with the pinned Q4_K_M download, choose a backend for your GPU, and size the context for the memory you actually have.',
    readNext: [
      {
        title: 'Matched MTP against llama.cpp',
        href: '/zinc/docs/matched-mtp-llama-cpp/',
        description: 'The November 2 campaign: equal speculation settings, nine samples per cell, exact-output controls and retained failures.',
      },
      {
        title: 'Client validation and task outcomes',
        href: '/zinc/docs/opencode-release-validation/',
        description: 'Model-by-model protocol checks and coding fixtures, with failed tasks and narrower retries kept visible.',
      },
      {
        title: 'Intel qualification and follow-up work',
        href: '/zinc/docs/intel-performance/',
        description: 'Native-reference checks, seven-model coverage and the managed-catalog correction after the stable release.',
      },
    ],
    docs: [
      {
        title: 'Install ZINC',
        href: '/zinc/docs/binary-releases/',
        description: 'Choose the Linux Vulkan or ROCm package; check driver requirements and the separate Metal preview status.',
      },
      {
        title: 'Connect a coding agent',
        href: '/zinc/docs/agent-setup/',
        description: 'Stable server commands, model discovery, OpenCode and OpenClaw setup, and development-only capabilities.',
      },
      {
        title: 'ROCm context and Q8 KV',
        href: '/zinc/docs/rocm-native-context/',
        description: 'Memory accounting, cache precision and the measured limits of near-native-context retrieval.',
      },
      {
        title: 'All measured workloads',
        href: '/zinc/benchmarks/',
        description: 'Compare backends and MTP modes with the source revisions, correctness findings and original samples visible.',
      },
    ],
    related: ['opencode-local-coding', 'amd-rdna4-llm-inference', 'kv-cache-quantization'],
    faqs: [
      {
        question: 'How much memory does Qwen 3.8 27B need?',
        answer: 'The pinned Q4_K_M file is 17.1 GB, about 15.9 GiB. Runtime state and context need additional memory. ZINC records a 20 GiB requirement at the managed 4,096-token default; plan for a 24 GB-class discrete GPU or a 32 GB-class Apple Silicon machine, and check the actual backend and available memory before increasing context.',
      },
      {
        question: 'Can I run the managed model on Intel Arc?',
        answer: 'The Arc Pro B70 Vulkan qualification includes Qwen 3.8. The published 0.1.0 catalog omitted its Intel profile; current source fixes that selection problem. Use the development build for the managed model ID on Intel and keep MTP disabled.',
      },
      {
        question: 'Does a 262,144-token model window mean I can use it immediately?',
        answer: 'No. The managed default is 4,096 tokens. Native capacity with NextN needs 34 GiB of F32 KV alone, or 9.5625 GiB with the opt-in ROCm Q8 cache, before weights and scratch. The long-context report documents a separate precision mode and retrieval test; it does not establish full-window coding quality.',
      },
      {
        question: 'Should I enable MTP for the first run?',
        answer: 'Start with ZINC_MTP=0. The matched campaign found exact-output differences in one Vulkan MTP workload and every ROCm MTP-on workload. Evaluate the current implementation and your own task before treating speculative throughput as a benefit.',
      },
      {
        question: 'Does this setup accept images and video?',
        answer: 'This guide covers the text GGUF path. The upstream model has vision-language capabilities, but ZINC does not currently expose image input through its OpenAI-compatible API.',
      },
    ],
  },
  {
    slug: 'amd-rdna4-llm-inference',
    title: 'AMD RDNA4 LLM Inference',
    shortTitle: 'AMD RDNA4',
    description: 'A practical guide to local LLM inference on AMD RDNA4 GPUs: R9700, RX 9070 XT, Vulkan, ROCm, llama.cpp comparisons, prefill, decode, and ZINC.',
    keywords: 'AMD RDNA4 LLM inference, Radeon AI PRO R9700 inference, RX 9070 XT LLM, AMD GPU local LLM, Vulkan LLM inference, ROCm HIP inference, llama.cpp RDNA4',
    summary: 'ZINC supports RDNA4 consumer and workstation GPUs through both Vulkan and ROCm.',
    practicalAnswer: 'If you want local LLM inference on AMD RDNA4, ZINC offers independently tuned Vulkan and ROCm paths. Vulkan is the broad compatibility route; ROCm uses HIP and dedicated gfx1201 kernels. The important performance split is decode versus prefill: decode is mostly memory and scheduling, while prefill needs batched kernels and model-aware tiling.',
    bestUse: 'Use this page for readers choosing AMD hardware or validating whether Vulkan local inference is real on RDNA4. The page should answer the hardware, driver, benchmark, and baseline questions before sending them to deep posts.',
    status: [
      {
        label: 'Best current answer',
        detail: 'RDNA4 local inference is real through both Vulkan and ROCm. The practical question is which backend, memory class, and driver stack best fit the workload.',
      },
      {
        label: 'Reader problem',
        detail: 'Readers need hardware fit, driver caveats, reproducible benchmark method, and a clear llama.cpp comparison before they trust the result.',
      },
      {
        label: 'Main bottleneck',
        detail: 'Decode is usually weight and scheduling bound. Prefill is a different workload and often needs batched kernels plus fewer host-side submits.',
      },
    ],
    actionPlan: [
      {
        label: 'Pick the memory budget first',
        detail: 'The R9700-class 32 GB card changes which Qwen and Gemma models are realistic. RX 9070-class cards share RDNA4 behavior but fit fewer large targets.',
      },
      {
        label: 'Benchmark against llama.cpp',
        detail: 'Use the same machine, model file, quantization, prompt, output cap, warmup policy, and backend residency. Cross-machine screenshots are not evidence.',
      },
      {
        label: 'Tune the platform before the shader',
        detail: 'Driver version, GECC, cooperative matrix flags, ASPM, and stale GPU processes can move results enough to hide real kernel work.',
      },
      {
        label: 'Report prefill and decode separately',
        detail: 'RDNA4 decode can look strong while prefill remains the user-visible latency limiter. A good post shows both phases.',
      },
    ],
    checklist: [
      {
        label: 'Record the exact GPU',
        detail: 'Name the card, VRAM, driver, Mesa version, and Vulkan device. RDNA4 is not a single performance number.',
      },
      {
        label: 'Clean the node',
        detail: 'Stop stale zinc, llama.cpp, and benchmark processes before publishing a comparison.',
      },
      {
        label: 'Lock the run shape',
        detail: 'Use the same model file, quantization, prompt, max tokens, context, warmup, and endpoint across engines.',
      },
      {
        label: 'Separate CLI from server',
        detail: 'CLI decode, raw HTTP completion, and chat completion measure different parts of the stack.',
      },
    ],
    measurements: [
      {
        label: 'Prompt tok/s',
        detail: 'Shows whether prefill kernels and command submission are healthy.',
      },
      {
        label: 'Decode tok/s',
        detail: 'Useful only when the model, quantization, prompt, and output length are included.',
      },
      {
        label: 'Latency distribution',
        detail: 'For server mode, include TTFT plus p50 and p95 request latency under the stated concurrency.',
      },
      {
        label: 'Platform state',
        detail: 'Record driver version, GECC, cooperative matrix flag, ASPM policy, and thermal throttling clues.',
      },
    ],
    articleIdeas: [
      {
        label: 'RX 9070 XT local LLM guide',
        detail: 'A 16 GB practical guide for what fits, what needs offload, and what ZINC runs well today.',
      },
      {
        label: 'R9700 versus RX 9070 XT',
        detail: 'A memory-class comparison using the same Qwen and Gemma prompts.',
      },
      {
        label: 'RDNA4 benchmark hygiene checklist',
        detail: 'A reproducibility post covering drivers, stale processes, endpoint choice, and prompt shape.',
      },
    ],
    pitfalls: [
      {
        label: 'Blending backend results',
        detail: 'Vulkan and ROCm have different runtime and kernel behavior. Publish them as separate datasets instead of mixing their best numbers.',
      },
      {
        label: 'Comparing dirty runs',
        detail: 'Background GPU users, cold builds, one-token completions, and changed prompts will swamp the signal.',
      },
      {
        label: 'Treating RDNA4 as one number',
        detail: 'R9700, RX 9070 XT, and future workstation cards share architecture but not memory capacity, clocks, thermals, or deployment fit.',
      },
    ],
    whatMatters: [
      'RDNA4 can run useful local LLMs through either ZINC Vulkan or ZINC ROCm, with each backend measured independently.',
      'The R9700 is the strongest ZINC tuning target because 32 GB VRAM changes which Qwen and Gemma models fit.',
      'Prefill and decode are different workloads; a fast decode loop does not imply fast time-to-first-token.',
      'llama.cpp is the comparison baseline, while ZINC maintains dedicated AMD kernels for its supported backends.',
    ],
    readNext: [
      {
        title: 'How we made AMD LLM inference 4x faster on a single GPU',
        href: '/blog/2026-03-30-how-we-moved-zinc-from-7-tok-s-to-33-tok-s-on-amd-rdna4/',
        description: 'The early RDNA4 speedup story and what changed in the decode path.',
      },
      {
        title: 'What broke first in local LLM inference on AMD RDNA4',
        href: '/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4/',
        description: 'The first correctness failures: attention, KV cache, RoPE, MoE, SSM, and tokenizer drift.',
      },
      {
        title: 'The broken Vulkan shaders keeping AMD RDNA4 inference stuck at 4 tok/s',
        href: '/blog/2026-03-29-the-shaders-standing-between-4-tok-s-and-27-tok-s/',
        description: 'How shader debugging moved work back onto the GPU.',
      },
      {
        title: 'Why RDNA4 prefill for Qwen3.5-35B is stuck at 25 tok/s',
        href: '/blog/2026-04-18-why-rdna4-prefill-for-qwen-3-5-is-stuck-at-25-tok-s/',
        description: 'Why prefill needs different kernels than decode on AMD.',
      },
      {
        title: 'How AMD Qwen decode passed llama.cpp in six weeks',
        href: '/blog/2026-05-09-how-we-made-amd-qwen-inference-faster-than-llama-cpp-in-six-weeks-on-the-radeon-ai-pro-r9700/',
        description: 'The public comparison point against llama.cpp on the R9700.',
      },
    ],
    docs: [
      {
        title: 'Run LLMs on AMD GPUs with ROCm or Vulkan',
        href: '/zinc/docs/getting-started/',
        description: 'The fastest path from clone to local AMD, Intel Arc, or Apple Silicon inference.',
      },
      {
        title: 'AMD RDNA3/RDNA4 GPU Reference',
        href: '/zinc/docs/amd-gpu-reference/',
        description: 'Hardware, memory, wave execution, and Vulkan compute details.',
      },
      {
        title: 'RDNA4 Tuning Guide',
        href: '/zinc/docs/rdna4-tuning/',
        description: 'Driver, shader, cooperative matrix, and benchmark tuning notes.',
      },
      {
        title: 'ZINC Benchmarks',
        href: '/zinc/benchmarks/',
        description: 'Current public measurements across AMD, Metal, and baseline engines.',
      },
    ],
    related: ['qwen3-8-local-inference', 'gemma-local-inference', 'kv-cache-quantization'],
    faqs: [
      {
        question: 'Should I use ROCm or Vulkan on AMD RDNA4?',
        answer: 'Both are supported ZINC backends. ROCm is the primary Radeon path, using HIP with dedicated gfx1201 kernels, and it posts by far the widest prompt-processing margins; Vulkan needs no ROCm install and currently generates tokens faster on several models. Compare the separate backend tabs for the model and phase you care about.',
      },
      {
        question: 'Which RDNA4 GPU is the best target for ZINC?',
        answer: 'The Radeon AI PRO R9700 is the primary tuning target because it combines RDNA4 behavior with a 32 GB memory budget. RX 9070-class cards share the architecture but fit fewer large models.',
      },
      {
        question: 'Why compare to llama.cpp?',
        answer: 'llama.cpp is the strongest widely used local baseline with Vulkan and Metal support. ZINC compares against it on the same machine to avoid misleading cross-hardware numbers.',
      },
    ],
  },
  {
    slug: 'kv-cache-quantization',
    title: 'KV Cache Quantization for Local LLMs',
    shortTitle: 'KV Cache',
    description: 'A practical guide to KV cache quantization for local LLM inference: FP16 vs INT8/Q4, 128k context, 32 GB GPUs, TurboQuant, and ZINC.',
    keywords: 'KV cache quantization, local LLM long context, TurboQuant, FP16 KV cache, INT8 KV cache, Q4 KV cache, 128k context, RDNA4 VRAM',
    summary: 'KV cache quantization is the long-context memory lever. Once a model fits, the prompt length and concurrent sessions are usually limited by K/V bytes per token, not by the static weight file.',
    practicalAnswer: 'For local long-context inference, FP16 KV cache is a debug-friendly default, not the right long-term default. INT8 K/V, asymmetric Q4 K plus higher-precision V, and TurboQuant-style compression all attack the same bottleneck: K/V memory grows linearly with tokens. On 16 GB and 32 GB GPUs, that growth decides whether 128k context is real or just a model-card number.',
    bestUse: 'Use this page for readers trying to understand why a model that fits in VRAM still fails at long context. The search intent is practical memory math: bytes per token, which precision is safe, and what an engine must implement.',
    status: [
      {
        label: 'Best current answer',
        detail: 'KV cache precision is the lever that turns model-card context into usable local context after the weights already fit.',
      },
      {
        label: 'Reader problem',
        detail: 'Readers need to compute bytes per token, choose a precision strategy, and understand the quality and bandwidth tradeoff.',
      },
      {
        label: 'Main bottleneck',
        detail: 'At long context, attention reads can become the dominant decode traffic. A smaller cache only helps when kernels read the compressed layout directly.',
      },
    ],
    actionPlan: [
      {
        label: 'Compute bytes per token',
        detail: 'Start from layers, KV heads, head dimension, and precision. That number explains context limits better than model size alone.',
      },
      {
        label: 'Prefer asymmetric precision',
        detail: 'K often tolerates lower precision than V. INT8 or Q4 K with higher-precision V is usually a better first target than uniformly shrinking both.',
      },
      {
        label: 'Read compressed KV directly',
        detail: 'A long-context speedup disappears if the attention path dequantizes the whole cache into FP16 scratch before every read.',
      },
      {
        label: 'Validate at long context',
        detail: 'Short prompts rarely reveal KV quantization regressions. Test perplexity, retrieval, and decode throughput at 16k, 32k, and beyond.',
      },
    ],
    checklist: [
      {
        label: 'Write the formula down',
        detail: 'Use layers times KV heads times head dimension times K/V precision times token count, then compare it with the actual VRAM budget.',
      },
      {
        label: 'Choose K and V separately',
        detail: 'Do not assume symmetric quantization. K and V often have different error tolerance and bandwidth value.',
      },
      {
        label: 'Include allocator behavior',
        detail: 'Paging, contiguous arenas, prefix reuse, and fragmentation determine whether the theoretical saving is usable.',
      },
      {
        label: 'Test retrieval, not only perplexity',
        detail: 'Long-context failures often show up as missed facts or degraded attention over distant tokens.',
      },
    ],
    measurements: [
      {
        label: 'Bytes per token',
        detail: 'The most useful KV number because it directly predicts maximum context and session capacity.',
      },
      {
        label: 'Attention bandwidth',
        detail: 'Measure whether the compressed format actually reduces memory traffic in decode.',
      },
      {
        label: 'Quality delta',
        detail: 'Compare retrieval, perplexity, and answer stability at the target context length.',
      },
      {
        label: 'Cache residency',
        detail: 'Show reserved versus active KV bytes and whether pages are shared, evicted, or compacted.',
      },
    ],
    articleIdeas: [
      {
        label: 'Long-context budgets by GPU memory',
        detail: 'A table for 16 GB, 24 GB, 32 GB, Intel Arc, and Apple Silicon using common Qwen and Gemma shapes.',
      },
      {
        label: 'FP8 versus Q4 KV cache',
        detail: 'A practical comparison of decode bandwidth, implementation complexity, and quality risk.',
      },
      {
        label: 'Prefix cache plus KV quantization',
        detail: 'Explain where prompt caching changes memory pressure and where it only moves the cost.',
      },
    ],
    pitfalls: [
      {
        label: 'Calling FP16 the default answer',
        detail: 'FP16 is useful for correctness bring-up, but it is too expensive for serious 128k local context on 16 GB and 32 GB cards.',
      },
      {
        label: 'Ignoring page layout',
        detail: 'Quantization, paging, and prefix reuse interact. A smaller cache still needs an allocation layout that serves the workload.',
      },
      {
        label: 'Optimizing memory but losing bandwidth',
        detail: 'The format only helps if the kernel spends less time reading memory after scale loads, unpacking, and correction are included.',
      },
    ],
    whatMatters: [
      'Weights are static after load; KV cache grows with every prompt token and generated token.',
      'Grouped-query attention reduces KV cost, but long context still makes KV memory large.',
      'K vectors usually tolerate lower precision than V vectors, so asymmetric formats are attractive.',
      'A good attention kernel should read quantized KV directly rather than dequantizing into a full FP16 scratch copy.',
    ],
    readNext: [
      {
        title: 'Why FP16 KV cache is the wrong default for 128k context',
        href: '/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4/',
        description: 'The memory math for long context on a 32 GB RDNA4 card.',
      },
      {
        title: 'The 16k crossover where KV reads outweigh active weights',
        href: '/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode/',
        description: 'Why decode becomes KV-bandwidth dominated at long context.',
      },
      {
        title: 'Paged KV cache is the serving fix a single-user local engine can mostly skip',
        href: '/blog/2026-05-26-paged-kv-cache-is-the-serving-fix-a-single-user-local-engine-can-mostly-skip/',
        description: 'Where paging matters, where contiguous cache is simpler, and what prefix sharing changes.',
      },
      {
        title: 'FP8 KV cache is the next decode bandwidth cut RDNA4 already has the WMMA for',
        href: '/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for/',
        description: 'Why lower-precision K/V storage also becomes a throughput optimization.',
      },
      {
        title: 'Why GQA is not the last KV cache shape for local 32 GB long context',
        href: '/blog/2026-05-03-why-gqa-is-not-the-last-kv-cache-shape-for-local-32gb-long-context/',
        description: 'Why the next reductions are architectural and memory-layout driven.',
      },
    ],
    docs: [
      {
        title: 'TurboQuant KV Cache Compression',
        href: '/zinc/docs/turboquant-spec/',
        description: 'ZINC design notes for 2-4 bit K/V pages and QJL correction.',
      },
      {
        title: 'ZINC Technical Specification',
        href: '/zinc/docs/spec/',
        description: 'Paged KV cache, request scheduling, and attention engine architecture.',
      },
      {
        title: 'Running ZINC',
        href: '/zinc/docs/running-zinc/',
        description: 'Runtime flags and model-running details, including KV-related options.',
      },
    ],
    related: ['amd-rdna4-llm-inference', 'qwen3-8-local-inference', 'gemma-local-inference'],
    faqs: [
      {
        question: 'Why does KV cache matter for local LLMs?',
        answer: 'KV cache stores the keys and values needed for attention over prior tokens. It grows linearly with context length and can become the largest moving memory cost after the model weights fit.',
      },
      {
        question: 'Is FP16 KV cache still useful?',
        answer: 'Yes, as a simple correctness baseline and debug mode. For long context on 16 GB or 32 GB cards, it is usually too expensive to be the practical default.',
      },
      {
        question: 'What is TurboQuant doing differently?',
        answer: 'TurboQuant combines vector quantization with residual correction for keys so attention inner products stay low-bias at very low bit widths. It is a design target for future compressed KV paths in ZINC.',
      },
    ],
  },
];

export function getTopicHub(slug: string) {
  return topicHubs.find((hub) => hub.slug === slug);
}
