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

export type TopicHub = {
  slug: string;
  title: string;
  shortTitle: string;
  description: string;
  keywords: string;
  summary: string;
  practicalAnswer: string;
  bestUse: string;
  status: TopicHubBriefItem[];
  actionPlan: TopicHubBriefItem[];
  checklist: TopicHubBriefItem[];
  measurements: TopicHubBriefItem[];
  articleIdeas: TopicHubBriefItem[];
  pitfalls: TopicHubBriefItem[];
  whatMatters: string[];
  readNext: TopicHubLink[];
  docs: TopicHubLink[];
  related: string[];
  faqs: TopicHubFaq[];
};

export const topicHubs: TopicHub[] = [
  {
    slug: 'opencode-local-coding',
    title: 'OpenCode Local Coding with Qwen and ZINC',
    shortTitle: 'OpenCode',
    description: 'A practical guide to OpenCode local coding with Qwen and ZINC: local LLM provider config, tool calls, thinking mode, context limits, SSH tunnels, and trace proxy debugging.',
    keywords: 'OpenCode local coding, OpenCode local LLM, OpenCode Qwen, Qwen coding model, OpenCode ZINC, local coding agent, OpenCode OpenAI compatible provider, ZINC tool calling, self-hosted coding assistant',
    summary: 'OpenCode can use ZINC and Qwen through the same OpenAI-compatible `/v1/chat/completions` API that powers the browser chat UI. The useful setup is local and boring: run ZINC, point OpenCode at localhost, keep tools enabled, set honest context limits, and use the trace proxy while testing coding workflows.',
    practicalAnswer: 'For OpenCode local coding, run ZINC on a localhost `/v1` endpoint and configure a custom `@ai-sdk/openai-compatible` provider. Use a Qwen-family model with stable thinking and tool behavior, keep `ZINC_TOOL_CALLING` enabled, and start with conservative context/output limits such as 4096/2048. If OpenCode runs on a laptop while ZINC runs on a GPU box, use an SSH tunnel so the OpenCode config still contains only `127.0.0.1`.',
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
        detail: 'Run the server on `127.0.0.1`, usually port 9090, and verify `/health` plus `/v1/models` before opening OpenCode.',
      },
      {
        label: 'Configure one provider',
        detail: 'Use OpenCode\'s OpenAI-compatible provider config with `baseURL` set to `http://127.0.0.1:9090/v1` or to the local trace proxy.',
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
        detail: '`curl http://127.0.0.1:9090/health` should pass from the same machine that runs OpenCode.',
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
        detail: 'Use `tools/opencode_trace_proxy.mjs` when diagnosing repeated reads, malformed tool paths, short answers, or thinking/tool-choice behavior.',
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
        title: 'Qwen3.6 local inference',
        href: '/topics/qwen3-6-local-inference/',
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
    related: ['qwen3-6-local-inference', 'amd-rdna4-llm-inference', 'gemma-local-inference'],
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
    related: ['qwen3-6-local-inference', 'amd-rdna4-llm-inference', 'kv-cache-quantization'],
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
    slug: 'qwen3-6-local-inference',
    title: 'Qwen3.6 Local Inference',
    shortTitle: 'Qwen3.6',
    description: 'A practical guide to Qwen3.6 local inference: architecture details, GGUF status, sparse MoE, SSM, MTP, speculative decoding, RDNA4, and Metal.',
    keywords: 'Qwen3.6 local inference, Qwen3.6 architecture, Qwen3.6 GGUF, Qwen 3.6 AMD GPU, Qwen3.6 RDNA4, Qwen3.6 speculative decoding, Qwen3.6 MTP',
    summary: 'Qwen3.6 is the core search cluster for ZINC because it combines model-architecture curiosity with practical local inference intent. Readers want to know what the model is, whether it exists as GGUF, and what an engine has to do to run it well.',
    practicalAnswer: 'For local Qwen3.6 inference, the important distinction is between model availability and engine readiness. ZINC already supports managed Qwen3.6 GGUF targets, but the fastest path depends on the variant. Dense Qwen3.6 is a batched-prefill and attention problem. A3B-style Qwen3.6 is a hybrid MoE plus SSM problem, where router scheduling, recurrent state, MTP, and KV memory all matter.',
    bestUse: 'Use this page as the entry point for people searching "can I run Qwen3.6 locally" or "what does Qwen3.6 require from an inference engine." It should turn model-name curiosity into exact local-running guidance.',
    status: [
      {
        label: 'Best current answer',
        detail: 'Qwen3.6 is a fit-and-engine-readiness question, not only a download question. The right answer depends on dense versus A3B-style sparse variants.',
      },
      {
        label: 'Reader problem',
        detail: 'Searchers need a practical path from model name to local run: GGUF availability, managed model id, hardware class, and which backend path is mature.',
      },
      {
        label: 'Main bottleneck',
        detail: 'For sparse Qwen, the hard surface is MoE routing plus recurrent or SSM state in prefill; for dense Qwen, batched attention and LM head cost dominate.',
      },
    ],
    actionPlan: [
      {
        label: 'Identify the variant',
        detail: 'Dense, A3B MoE, and Next-style hybrid models do not have the same bottlenecks. Name the exact GGUF or managed model id before discussing performance.',
      },
      {
        label: 'Split readiness from availability',
        detail: 'A model file can exist before the fast path is ready. State what runs today, then call out which kernels still decide production quality.',
      },
      {
        label: 'Track prefill, decode, and context',
        detail: 'Qwen can win in one phase and lose in another. The useful comparison shows prompt throughput, decode throughput, latency, and context length together.',
      },
      {
        label: 'Treat MTP as the speculation path',
        detail: 'Generic draft-model speculation is fragile on sparse MoE. Target-attached MTP is the more credible story for Qwen A3B models.',
      },
    ],
    checklist: [
      {
        label: 'Verify metadata',
        detail: 'Confirm architecture string, layer count, attention layout, expert count, active expert count, vocab size, and context target.',
      },
      {
        label: 'Pick the right hardware bucket',
        detail: 'Separate 16 GB, 32 GB, Intel Arc, and Apple Silicon runs. A model that technically loads may still reserve too little context to be useful.',
      },
      {
        label: 'Run the baseline',
        detail: 'Use llama.cpp on the same machine when possible, with the same quantization and prompt policy.',
      },
      {
        label: 'State the readiness gap',
        detail: 'Call out whether the limiting work is model loading, prefill batching, SSM state, MoE routing, sampling, or KV memory.',
      },
    ],
    measurements: [
      {
        label: 'Prompt tok/s and TTFT',
        detail: 'Qwen posts should show whether the engine can turn long prompts into the first token quickly.',
      },
      {
        label: 'Decode tok/s',
        detail: 'Report steady-state generation separately from chat endpoint latency and early-stop behavior.',
      },
      {
        label: 'VRAM residency',
        detail: 'Show weight bytes, runtime bytes, reserved KV cache, offload status, and context capacity.',
      },
      {
        label: 'Speculation cost',
        detail: 'For MTP or draft-model posts, include acceptance rate, verifier cost, and state rewind cost.',
      },
    ],
    articleIdeas: [
      {
        label: 'Which Qwen3.6 variant should local users run?',
        detail: 'A practical dense versus A3B fit guide by GPU memory class.',
      },
      {
        label: 'Why MTP is the Qwen speculation path',
        detail: 'Turn the prior speculative-decoding analysis into a concise implementation guide.',
      },
      {
        label: 'Qwen long-context budget on 16 GB and 32 GB',
        detail: 'Tie model fit, KV memory, sampling overhead, and useful context into one decision page.',
      },
    ],
    pitfalls: [
      {
        label: 'Mixing Qwen names',
        detail: 'Qwen3, Qwen3.5, Qwen3.6, and Qwen3-Next attract overlapping searches. The page should keep model family, checkpoint, and architecture separate.',
      },
      {
        label: 'Overpromising long context',
        detail: 'A model-card context length is not the same as useful local context. KV memory, recurrent state, and prefill time still decide the run.',
      },
      {
        label: 'Assuming speculation is free',
        detail: 'Sparse expert verification can wake more experts than the draft saved. Any speedup claim needs acceptance rate and verifier cost.',
      },
    ],
    whatMatters: [
      'Qwen3.6 interest is split between architecture details, GGUF availability, local runtime support, and performance.',
      'Sparse A3B variants decode like small active models but fill memory like large total-parameter models.',
      'Speculative decoding is not automatically useful on sparse MoE models because verification can wake many experts.',
      'MTP-style drafts are more promising than generic draft models when hidden-state alignment is available.',
    ],
    readNext: [
      {
        title: 'Qwen 3.6 architecture and local inference in ZINC',
        href: '/blog/2026-04-05-qwen-3-6-architecture-and-what-it-would-take-to-bring-it-into-zinc/',
        description: 'The main architecture explainer for Qwen3.6, MoE, SSM, context, and local engine implications.',
      },
      {
        title: 'Qwen3.6-35B-A3B GGUF on AMD, Intel Arc, and Metal',
        href: '/blog/2026-04-17-qwen-3-6-is-now-generally-available-in-zinc/',
        description: 'Managed model support, AMD RDNA4, Intel Arc, Apple Silicon notes, and practical run guidance.',
      },
      {
        title: 'Why speculative decoding does not net out on Qwen 35B-A3B',
        href: '/blog/2026-04-28-why-speculative-decoding-does-not-net-out-on-qwen-35b-a3b/',
        description: 'Why generic draft-model speculation can lose on sparse Qwen verification.',
      },
      {
        title: 'Why MTP heads are the speculative decode draft Qwen3 A3B deserves',
        href: '/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves/',
        description: 'Why target-attached multi-token prediction is a better fit than separate draft models.',
      },
      {
        title: 'The gate that keeps Qwen 35B prefill at half of llama.cpp on RDNA4',
        href: '/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4/',
        description: 'The structural MoE plus SSM prefill work needed for flagship Qwen performance.',
      },
    ],
    docs: [
      {
        title: 'Getting Started with ZINC',
        href: '/zinc/docs/getting-started/',
        description: 'Pull a managed Qwen model and run local inference through the CLI or chat path.',
      },
      {
        title: 'ZINC Benchmarks',
        href: '/zinc/benchmarks/',
        description: 'Current Qwen3.6 results against same-machine llama.cpp baselines.',
      },
      {
        title: 'Running ZINC',
        href: '/zinc/docs/running-zinc/',
        description: 'Model ids, CLI flags, chat mode, server mode, and KV quantization options.',
      },
      {
        title: 'Configure OpenCode with ZINC',
        href: '/zinc/docs/opencode/',
        description: 'Use Qwen through ZINC as a local OpenCode coding backend with tools, thinking, and context limits.',
      },
    ],
    related: ['amd-rdna4-llm-inference', 'kv-cache-quantization', 'gemma-local-inference'],
    faqs: [
      {
        question: 'Can Qwen3.6 run locally?',
        answer: 'Yes, when a supported GGUF target exists and the machine has enough memory. In ZINC, the managed model catalog is the preferred path because it keeps local files, defaults, and backend support aligned.',
      },
      {
        question: 'Why is Qwen3.6 hard for local inference engines?',
        answer: 'The hard cases combine sparse experts, recurrent or state-space blocks, large context, and a large vocabulary. That pushes work into routing, recurrent state updates, KV memory, sampling, and prefill scheduling.',
      },
      {
        question: 'Is speculative decoding a clear win on Qwen3.6?',
        answer: 'Not with a generic draft model. Sparse expert verification can erase the win. MTP-style target-attached drafts are the more credible direction for Qwen A3B models.',
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
        title: 'Run LLMs on AMD GPUs with Vulkan or ROCm',
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
    related: ['qwen3-6-local-inference', 'gemma-local-inference', 'kv-cache-quantization'],
    faqs: [
      {
        question: 'Should I use Vulkan or ROCm on AMD RDNA4?',
        answer: 'Both are supported ZINC backends. Vulkan offers broad driver compatibility, while ROCm uses HIP and dedicated gfx1201 kernels. Compare the separate backend tabs for the model and phase you care about.',
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
    related: ['amd-rdna4-llm-inference', 'qwen3-6-local-inference', 'gemma-local-inference'],
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
