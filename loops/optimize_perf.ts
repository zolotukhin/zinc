#!/usr/bin/env bun
/**
 * ZINC Performance Optimization Loop
 *
 * Implements multi-hour optimization efforts defined in loops/efforts/MULTI_HOUR_EFFORT_*.md
 * documents. Each cycle:
 *   1. Read the optimization plan document
 *   2. Build & benchmark baseline on remote RDNA4 node
 *   3. Spawn AI agent to implement ONE concrete step from the plan
 *   4. Build, run tests, benchmark
 *   5. If tok/s improved AND output correct -> commit, update plan
 *   6. If regressed or broken -> revert, log what went wrong
 *   7. Loop back to 3
 *
 * Usage:
 *   bun loops/optimize_perf.ts --effort 1                        # Push descriptors
 *   bun loops/optimize_perf.ts --effort 2 --model qwen36b       # Fused gate+up on Qwen 35B
 *   bun loops/optimize_perf.ts --effort 3 --agent codex         # Batch prefill with Codex
 *   bun loops/optimize_perf.ts --effort 6 --model qwen36b       # RDNA prefill recovery on Qwen 35B
 *   bun loops/optimize_perf.ts --effort 1 --resume               # Resume previous run
 *   bun loops/optimize_perf.ts --effort 1 --cycles 10 --dry-run  # Baseline only
 */

import { spawn, execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, writeFile, mkdir, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
  parseTokPerSec,
  parsePrefillTokPerSec,
  parsePrefillTokenCount,
  parseBandwidthUtil,
  parsePrefillPhaseBudget,
  type PrefillPhaseBudget,
} from "./optimize_zinc";
import { formatElapsed } from "./optimize_llm_tps";

// -- Config ------------------------------------------------------------------

const REPO_ROOT = resolve(import.meta.dir, "..");
const EFFORTS_DIR = resolve(REPO_ROOT, "loops", "efforts");
const RESULTS_DIR = resolve(REPO_ROOT, ".perf_optimize");
const CLAUDE_EFFORT = "max";
// Pin to the 1M-context Opus variant. Cycle prompts run 8-12KB on their own
// (plan + phase budget + swing ideas + known-flat + cycle ledger + failed
// approaches + idea bank) and the agent frequently reads forward.zig
// (8.5K lines) plus shaders and reference-implementation sources, so the
// 1M-context variant is the right default for these cycles. Overridable
// via ZINC_CLAUDE_MODEL in case a future run needs Sonnet / Haiku.
const CLAUDE_MODEL = process.env.ZINC_CLAUDE_MODEL ?? "claude-opus-4-7[1m]";
const CODEX_MODEL = process.env.ZINC_CODEX_MODEL ?? "gpt-5.5";
const CODEX_REASONING_EFFORT = process.env.ZINC_CODEX_REASONING_EFFORT ?? "xhigh";
// opencode (https://opencode.ai) headless agent. Drives the same loop as
// claude/codex via `opencode run`. Default model is GLM-5.2 at max reasoning
// effort (matches the codex `xhigh` / claude `max` defaults); override model
// with ZINC_OPENCODE_MODEL, reasoning variant with ZINC_OPENCODE_VARIANT
// (e.g. max|high|minimal — empty string omits the flag), and optionally pick
// an opencode agent persona with ZINC_OPENCODE_AGENT.
const OPENCODE_MODEL = process.env.ZINC_OPENCODE_MODEL ?? "zai-coding-plan/glm-5.2";
const OPENCODE_VARIANT = process.env.ZINC_OPENCODE_VARIANT ?? "max";
const OPENCODE_AGENT = process.env.ZINC_OPENCODE_AGENT ?? "";

function loadEnv(): Record<string, string> {
  const envPath = join(REPO_ROOT, ".env");
  const vars: Record<string, string> = {};
  if (existsSync(envPath)) {
    const content = require("fs").readFileSync(envPath, "utf8") as string;
    for (const line of content.split("\n")) {
      const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.+?)\s*$/);
      if (m) vars[m[1]] = m[2];
    }
  }
  return vars;
}

const ENV = loadEnv();

function envValue(...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = process.env[key] ?? ENV[key];
    if (value != null && value !== "") return value;
  }
  return undefined;
}

const SELECTED_RDNA_NODE = envValue("ZINC_RDNA_NODE", "ZINC_NODE");

function rdnaNodeEnvKey(node: string | undefined, suffix: string): string | null {
  const normalized = (node ?? "")
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return normalized ? `ZINC_${normalized}_${suffix}` : null;
}

function rdnaEnvValue(suffix: string, ...fallbackKeys: string[]): string | undefined {
  const nodeKey = rdnaNodeEnvKey(SELECTED_RDNA_NODE, suffix);
  return envValue(...(nodeKey ? [nodeKey] : []), ...fallbackKeys);
}

const ZINC_HOST = rdnaEnvValue("HOST", "ZINC_RDNA_HOST", "ZINC_HOST") ?? "127.0.0.1";
const ZINC_PORT = Number(rdnaEnvValue("PORT", "ZINC_RDNA_PORT", "ZINC_PORT") ?? "22");
const ZINC_USER = rdnaEnvValue("USER", "ZINC_RDNA_USER", "ZINC_USER") ?? "root";
const REMOTE_DIR = rdnaEnvValue("REMOTE_DIR", "ZINC_RDNA_REMOTE_DIR", "ZINC_REMOTE_DIR") ?? "/root/zinc";
const AGENT_RSYNC_EXCLUDE_ARGS = [
  "--exclude='.zig-cache'",
  "--exclude='zig-out'",
  "--exclude='node_modules'",
  "--exclude='.git'",
  "--exclude='.perf_optimize'",
  "--exclude='.zinc_optimize'",
  "--exclude='site'",
  "--exclude='.DS_Store'",
  "--exclude='.env'",
  "--exclude='.env.*'",
  "--exclude='*.swp'",
  "--exclude='*.swo'",
].join(" ");
const DIAGNOSTIC_RUNTIME_FLAGS = new Set(["ZINC_PREFILL_PROFILE"]);

type PromptMode = "raw" | "chat";

type ModelTarget = {
  key: string;
  name: string;
  path: string;
  promptMode: PromptMode;
  systemPrompt?: string;
  coherencePromptMode?: PromptMode;
  envVar: string;
  coherenceMaxTokens?: number;
};

function envOrDefault(name: string, fallback: string): string {
  const rdnaPrefix = "ZINC_RDNA_";
  const suffix = name.startsWith(rdnaPrefix) ? name.slice(rdnaPrefix.length) : name;
  const nodeKey = rdnaNodeEnvKey(SELECTED_RDNA_NODE, suffix);
  return envValue(...(nodeKey ? [nodeKey] : []), name) ?? fallback;
}

const MODELS: Record<string, ModelTarget> = {
  qwen36b: {
    key: "qwen36b",
    name: "Qwen3.6-35B",
    path: envOrDefault("ZINC_RDNA_QWEN36_35B_MODEL", "/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
    promptMode: "raw",
    // Keep throughput benchmarking on the raw decode path, but run coherence
    // prompts through ChatML so Qwen gets the expected closed-think scaffold.
    coherencePromptMode: "chat",
    envVar: "ZINC_RDNA_QWEN36_35B_MODEL",
  },
  qwen3827b: {
    key: "qwen3827b",
    name: "Qwen3.8-27B",
    path: envOrDefault("ZINC_RDNA_QWEN38_27B_MODEL", "/root/models/Qwen3.8-27B-Q4_K_M.gguf"),
    promptMode: "chat",
    // Match the default system turn injected by POST /v1/chat/completions so
    // the optimizer and the published reusable-server scenario tokenize the
    // same workload.
    systemPrompt: "You are a helpful assistant. Answer directly. Do not show analysis.",
    coherencePromptMode: "chat",
    envVar: "ZINC_RDNA_QWEN38_27B_MODEL",
  },
  qwen359b: {
    key: "qwen359b",
    name: "Qwen3.5-9B",
    // Node-aware: prefer an explicit model key, then the selected node's own
    // REMOTE_MODEL (on rdna2 that is the managed 9B cache path), then the
    // /root/models layout used by rdna1. Do NOT fall back to the generic
    // ZINC_RDNA_REMOTE_MODEL — on this repo it points at the 8B, not the 9B.
    path: envOrDefault("ZINC_RDNA_QWEN35_9B_MODEL", rdnaEnvValue("REMOTE_MODEL") ?? "/root/models/Qwen3.5-9B-Q4_K_M.gguf"),
    promptMode: "raw",
    coherencePromptMode: "chat",
    envVar: "ZINC_RDNA_QWEN35_9B_MODEL",
  },
  qwen8b: {
    key: "qwen8b",
    name: "Qwen3-8B",
    path: envOrDefault("ZINC_RDNA_QWEN3_8B_MODEL", "/root/models/Qwen3-8B-Q4_K_M.gguf"),
    promptMode: "raw",
    envVar: "ZINC_RDNA_QWEN3_8B_MODEL",
  },
  gemma431b: {
    key: "gemma431b",
    name: "Gemma4-31B",
    path: envOrDefault("ZINC_RDNA_GEMMA4_31B_MODEL", "/root/models/gemma-4-31B-it-Q4_K_M.gguf"),
    promptMode: "chat",
    envVar: "ZINC_RDNA_GEMMA4_31B_MODEL",
  },
  gemma426ba4b: {
    key: "gemma426ba4b",
    name: "Gemma4-26B-A4B",
    path: envOrDefault("ZINC_RDNA_GEMMA4_12B_MODEL", "/root/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
    promptMode: "chat",
    envVar: "ZINC_RDNA_GEMMA4_12B_MODEL",
  },
};

const MODEL_KEYS = Object.keys(MODELS).join(", ");

const REMOTE_ZINC_ENV = "RADV_PERFTEST=coop_matrix";
const REMOTE_VULKAN_DEVICE_INDEX = (() => {
  const raw = process.env.ZINC_RDNA_DEVICE_INDEX
    ?? ENV.ZINC_RDNA_DEVICE_INDEX
    ?? process.env.ZINC_VULKAN_DEVICE_INDEX
    ?? ENV.ZINC_VULKAN_DEVICE_INDEX
    ?? "1";
  const parsed = Number(raw);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : 1;
})();
const LONG_CONTEXT_BENCH_SENTENCE =
  "Benchmark context only. alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.";

function repeatContext(lines: number): string {
  return Array.from({ length: lines }, () => LONG_CONTEXT_BENCH_SENTENCE).join(" ");
}

function shellQuote(value: string): string {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

const PREFILL_BENCHMARK_PROMPT = [
  "Long reference packet for benchmark purposes only:",
  "",
  repeatContext(6),
  "",
  "Important fact near the end: Paris is the capital of France.",
  "",
  "Ignore unrelated filler and answer from the reference packet.",
  "",
  "Based on the reference above, the capital of France is",
].join("\n");

const CODING_REVIEW_SNIPPET = [
  "File: src/cache.ts",
  "```ts",
  "const cache = new Map<string, string>();",
  "const pending = new Map<string, Promise<string>>();",
  "",
  "export async function getValue(key: string, load: () => Promise<string>) {",
  "  if (cache.has(key)) return cache.get(key)!;",
  "  if (pending.has(key)) return cache.get(key)!;",
  "",
  "  const task = load().then((value) => {",
  "    cache.set(key, value);",
  "    pending.delete(key);",
  "    return value;",
  "  });",
  "  pending.set(key, task);",
  "  return task;",
  "}",
  "```",
].join("\n");

const QWEN38_27B_CONTEXT_MEDIUM_PREFILL_PROMPT = [
  "You are reviewing a small TypeScript helper in a production service.",
  "Identify the bug, explain why it appears under concurrent requests, and provide a corrected version.",
  "",
  CODING_REVIEW_SNIPPET,
].join("\n");

// Long-context decode benchmark for Effort 11. The prompt is a single
// English narrative excerpt designed to tokenize to ~1500 tokens on
// Qwen 3 8B (no chat-template overhead, no list/code tokenizer
// quirks). Decode at L≈1500 is where the user-visible curve drop
// hurts most in chat sessions, and it's still under the L=2300 GPU
// hang we observed during manual cycles 71-73 of flash_attn.comp.
const LONG_CONTEXT_DECODE_PROMPT = [
  "Once upon a time, in a small village nestled between rolling hills and a meandering river, there lived a young blacksmith named Tomas. His forge stood at the edge of the marketplace, its chimney trailing thin grey smoke into the morning air. Every dawn he rose before the sun, lit the coals, and shaped iron into the tools and trinkets the villagers needed: horseshoes, kettles, hinges, plowshares. Tomas was not yet thirty, but the lines around his eyes told of long days and patient craft. He had inherited the forge from his father, who had inherited it from his father before him, three generations of black iron and orange sparks. The villagers respected him, though few understood why he often paused mid-strike to listen to the wind, or why on certain summer evenings he would walk alone along the riverbank, far past the willow trees, to a place no one else cared to go. The river there was deeper, its water darker, and the reeds grew taller than a man.",
  "",
  "Tomas had been going to that place since he was a boy, ever since the day his mother had taken him there to teach him the names of the herbs that grew along the bank. She had died the following winter, and the place had become his alone, a kind of memorial that did not need a stone. On this particular morning, however, Tomas did not go to the river. Instead, when he opened the forge, he found something unusual lying on the cold anvil: a sealed letter, its red wax stamped with a sigil he did not recognize. There was no draft, no ash disturbed, no footprint in the soot. The letter had simply appeared. Tomas turned it over in his rough hands. The paper was thick, expensive, and the wax had not yet hardened completely. Whoever left it had done so within the last hour.",
  "",
  "He carried it outside into the daylight and broke the seal with his thumb. The handwriting inside was elegant, precise, and the words were short: \"Come tonight, when the moon is over the willow. Bring nothing. Tell no one.\" There was no signature. Tomas read it twice, then a third time, and the more he read it the more he felt the weight of the morning shift around him. The wind, which had been still, began to stir. The smoke from his chimney bent westward toward the river. A horse in the marketplace whinnied without reason. Tomas folded the letter carefully and slipped it into the leather pouch at his belt, the one where he kept his grandfather's small iron compass and a single silver coin from a country no one in the village had heard of.",
  "",
  "He returned to the forge and worked through the day as he always did, but his mind was not on the iron. He shoed two horses for the miller, repaired a broken latch for the inn, and shaped four new nails for the carpenter, but he did all of it as if he were a man underwater. The customers noticed nothing. The village turned its slow wheel of bread and gossip and bargain and rest. The sun climbed, paused, and began its descent. When the bells of the small chapel rang for evening prayer, Tomas wiped his hands on his leather apron, banked the coals, and locked the forge for the night. He did not eat supper. He walked to the river path and waited for the moon.",
  "",
  "The moon rose late that evening, slow and full, the color of old brass. By the time it crested the willow at the bend of the river, Tomas had been waiting for nearly an hour. The reeds whispered. An owl called once and was silent. Tomas was about to turn back when a figure stepped out from behind the largest willow trunk. It was a woman he had never seen before, tall, with hair the color of river silt and a long traveling cloak the color of moss. She held no lantern, yet her face was clearly visible in the moonlight, as if she carried her own quiet light.",
  "",
  "Continue the story, describing what the woman said next and how Tomas responded.",
].join("\n");

const GEMMA_LONG_DECODE_PROMPT =
  "Write six short bullet points explaining why local LLM benchmark reports should separate prefill throughput from decode throughput.";

const LONG_CODING_PLAN_PROMPT =
  "Write an implementation plan for adding a stable benchmark preset to a local LLM CLI. Include the command shape, warmup policy, metrics to collect, failure handling, llama.cpp comparison, and how the site should display prefill, decode, latency, and overall prompt+decode throughput.";

const GEMMA_LONG_DRAFT_PREFILL_PROMPT = LONG_CODING_PLAN_PROMPT;

const QWEN35_9B_LONG_DRAFT_PREFILL_PROMPT =
  `${LONG_CODING_PLAN_PROMPT}\n\nPlan:\n1.`;

type MetricMode = "decode" | "prefill";

type EffortSpec = {
  doc: string;
  summary: string;
  metricMode: MetricMode;
  primaryMetricLabel: string;
  benchmarkPrompt: string;
  benchmarkMaxTokens: number;
  benchmarkMethod: string;
  defaultModel?: string;
  // Optional sanity floor for the baseline benchmark. This catches cases
  // where the RDNA node is effectively not using the GPU path, is badly
  // contaminated by stale processes, or has fallen into a driver/runtime
  // state that makes optimization results meaningless.
  minHealthyTokPerSec?: number;
  // Optional per-effort controller hints. These are rendered into the agent
  // prompt so the loop can encode knowledge the base plan document doesn't
  // (or shouldn't) encode itself.
  //
  // knownFlatCategories: descriptions of change patterns that have been
  // empirically shown not to move the number on the target hardware. The
  // agent is told not to re-attempt these without new supporting evidence.
  //
  // structuralSwingIdeas: concrete, scoped ideas the agent should pick from
  // when the controller is in a stall-break mode (stalled >= threshold, or
  // one foundation keep has already been banked). These are meant to be
  // real engineering steps, not idea-bank candy — each one should be
  // actionable in a single cycle and should compose with future cycles.
  knownFlatCategories?: string[];
  structuralSwingIdeas?: string[];
  // Reference implementations the agent can read on disk. Only surfaced in
  // the prompt once the loop has stalled — before that, the guidance is to
  // work from the plan document and the in-tree code. The agent is free to
  // ignore these; they are presented as options, not obligations.
  referenceImplementations?: Array<{ path: string; focus: string }>;
  // llama.cpp baselines per scenario, used to render a "beat llama.cpp"
  // delta block in the agent prompt. The loop's per-cycle benchmark only
  // measures ONE scenario (the controller's primaryMetric), but the actual
  // success goal is to beat llama.cpp on all four scenarios. Showing the
  // gap explicitly stops the loop from optimizing in a single-metric
  // vacuum and surfaces secondary scenarios the agent isn't measuring
  // directly. The primary-metric scenario also gets a "to beat llama.cpp,
  // you need X more tok/s (+Y%)" target that the agent can reason against.
  llamaCppBaselines?: LlamaCppBaseline[];
  llamaCppSuccessRule?: string;
};

export type LlamaCppBaseline = {
  scenario: string;
  prefillTokPerSec: number;
  decodeTokPerSec: number;
  promptTokens?: number;
  // Which baseline corresponds to the loop's primaryMetric. Matched
  // case-insensitively against primaryMetricLabel (e.g. a label of
  // "Qwen3.8-27B prefill tok/s" matches isPrimary when metricMode is
  // "prefill" and scenario contains "context-medium" — the controller's
  // benchmark prompt is the site-aligned context-medium prefill).
  isPrimary?: boolean;
};

const EFFORT_SPECS: Record<number, EffortSpec> = {
  1: {
    doc: "MULTI_HOUR_EFFORT_1_PUSH_DESCRIPTORS.md",
    summary: "Push descriptors (~2.5% decode speedup)",
    metricMode: "decode",
    primaryMetricLabel: "decode tok/s",
    benchmarkPrompt: "Write a detailed essay about the history of computing, from mechanical calculators to modern artificial intelligence.",
    benchmarkMaxTokens: 200,
    benchmarkMethod: "200-token decode benchmark on the primary model",
  },
  2: {
    doc: "MULTI_HOUR_EFFORT_2_FUSED_GATE_UP.md",
    summary: "Fused gate+up DMMV (~1-2% decode speedup)",
    metricMode: "decode",
    primaryMetricLabel: "decode tok/s",
    benchmarkPrompt: "Write a detailed essay about the history of computing, from mechanical calculators to modern artificial intelligence.",
    benchmarkMaxTokens: 200,
    benchmarkMethod: "200-token decode benchmark on the primary model",
  },
  3: {
    doc: "MULTI_HOUR_EFFORT_3_BATCH_PREFILL.md",
    summary: "Batch prefill (~4-8x prefill speedup)",
    metricMode: "prefill",
    primaryMetricLabel: "prefill tok/s",
    benchmarkPrompt: PREFILL_BENCHMARK_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "long-context prefill benchmark aligned with the site report",
  },
  4: {
    doc: "MULTI_HOUR_EFFORT_4_PREFILL_RECOVERY.md",
    summary: "Prefill recovery (close the long-context gap vs llama.cpp)",
    metricMode: "prefill",
    primaryMetricLabel: "prefill tok/s",
    benchmarkPrompt: PREFILL_BENCHMARK_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "long-context prefill benchmark aligned with the site report",
  },
  10: {
    doc: "MULTI_HOUR_EFFORT_10_QWEN36_DECODE.md",
    summary: "Qwen 3.6 35B-A3B decode + prefill speedups on RDNA4 (cross-token batched MoE, parallel-scan SSM, GEMM mmq)",
    metricMode: "decode",
    primaryMetricLabel: "decode tok/s",
    benchmarkPrompt: "Write a detailed essay about the history of computing, from mechanical calculators to modern artificial intelligence.",
    benchmarkMaxTokens: 200,
    benchmarkMethod: "200-token decode benchmark on Qwen 3.6 35B-A3B, with --profile to track per-phase budgets",
    knownFlatCategories: [
      "Q4_K × Q8_1 mmq for SSM proj GEMV. Built in commit 27f0c76, wired behind ZINC_MMQ_SSM=1 in 3fef46e — measured zero speedup on Qwen 3.6 (SSM phase 15.94 ms either way). The shader is correct; the path is bandwidth-bound on the *weight* side, not on activation bandwidth or dequant compute. Don't re-attempt unless the dispatch is in a GEMM context (multi-token amortizing weight reads).",
      "Fusing alpha+beta SSM proj DMMVs via dmmv_q4k_fused_gate_up. Reverted in commit 3fef46e (the comment in forward.zig:7557+ explains). The four SSM proj DMMVs already overlap on RDNA4 since there are no inter-DMMV barriers — fusing saves a dispatch but loses no wall time. Distinct from the cycle-13 fused-RMS+alpha+beta which won by adding the RMS norm into the same dispatch — that was a separate dispatch reduction, not just a fusion of already-overlapping DMMVs.",
      "Dense fused gate+up (dmmv_q4k_fused_gate_up.comp landed in 339c886). Regresses Gemma 4 31B decode by +11% from doubled per-WG register pressure on wide inter_dim=25600. Pipeline + helper available, but not wired and not a candidate for re-wire unless a NUM_ROWS=1 variant is built that fits the register budget.",
      "Adding a NUM_ROWS=4 medium variant of dmmv_q4k for SSM out (M=2048). The SSM out projection at 3.10 ms already runs near peak occupancy with NUM_ROWS=2 → 1024 WGs on a 2048-WG-capacity device; dropping to 512 WGs (NUM_ROWS=4) underutilizes. Don't add this without measuring the SSM out is genuinely under-saturated.",
      "Barrier narrowing computeBarrier() → computeBufferBarrier() on already-overlapping dispatches. Cycles 1, 16, 18 all measured at-or-near noise floor on RADV. The driver doesn't appear to differentiate the access masks the way the hypothesis assumed. Don't re-attempt without first proving via VK_KHR_synchronization2 / VkCmdPipelineBarrier2 that the access mask precision actually changes the GPU pipeline behavior on RADV.",
      "Register-caching delta_net_output across the two passes in ssm_gated_norm.comp (cycle 5). Pass-2 global re-read is not the bottleneck.",
      "Fused MoE down + SwiGLU (cycle 9, kpar+swiglu and triple-fused down+swiglu+acc both regressed) and Q4_K MoE gate+up+SwiGLU forward fusion (cycle 19, +0.12% — noise). Don't re-attempt MoE-side SwiGLU fusion variants.",
      "f16-quantized MoE router weight + fused rms_norm_dmmv_f16_router (cycle 17). Per-layer device-local f16 buffers built from f32 ffn_gate_inp at engine init. Negative result. The cycle-8 f32 fused-router shader is the right shape and is already shipped.",
      "Three-way RMS+K+V fusion (cycle 14, rms_norm_dmmv_q8_0_kv shader). Reverted — too much register pressure in one workgroup. Two-way KV-only fusion or an attention-side fused-RMS+single-DMMV (e.g., RMS+Q proj alone) is still open.",
      "Wide NUM_ROWS variants on the SSM proj wqkv path (cycle 11 — Q8_0 NUM_ROWS=4 + register-tiled activation reads). Neutral. NUM_ROWS=4 wide kpar variant of dmmv_q4k_moe_kpar for MoE gate/up (cycle 12) measured -0.05 tok/s and gate_up phase +0.10 ms. Don't re-attempt wide NUM_ROWS variants on M ≤ ~10000 dispatches.",
      "Cycle-6/7 fused MoE down + weighted_acc shader (Q4_K + Q5_K). +0.16 vs checkpoint, below override threshold. Already a correct shader; the bottleneck moved away from this fusion.",
    ],
    structuralSwingIdeas: [
      "More fused-RMS+DMMV shaders. The pattern that delivers wins on this effort: fold the RMS norm into the immediately-consuming DMMV. Cycle 8 shipped rms_norm_dmmv_f32 (+0.61 tok/s) for the f32 router. Cycle 13 shipped rms_norm_dmmv_q4k_alpha_beta (+0.57 tok/s) for the SSM proj alpha+beta pair. Concrete remaining candidates: (a) attn_norm + wqkv DMMV (the SSM proj's biggest output, M≈6144) — cycle 10 attempted this and measured small flag-on gain falling short of checkpoint due to redundant per-WG RMS reduction work; the fix is to compute the RMS reduction ONCE per workgroup via shared memory and reuse across all NUM_ROWS rows (cycle 13's shader does this correctly — read it). (b) attn_norm + ssm_z DMMV (M≈4096), same fix. (c) attn_norm + attention Q proj on the 10 attention layers (M=2048, K=2048) — single-output shape is a clean fit for the existing rms_norm_dmmv_f32 layout.",
      "Eliminate dispatches in the SSM tail. Fused ssm_out + FFN-RMS-norm shader: the SSM tail does (Q4_K ssm_out DMMV → residual add → ffn_norm RMS norm) which is structurally the same as the existing rms_norm_add shader (commit a5f1fdc, used by Gemma post_ffw_norm) but with a Q4_K DMMV in front. Eliminates 1 dispatch + 1 barrier per SSM layer × 30 layers = 30 dispatches saved per token. The hidden-buf accumulate pattern is already proven correct in rms_norm_add.comp — extend with a Q4_K weight stream.",
      "KV-cache-write fused into K-projection on attention layers. Saves 10 dispatches + 10 barriers per token. Distinct from cycle 14's failed three-way RMS+K+V (which had too much register pressure) — this fuses just the K projection's dot-product output directly into the cache page write at end-of-kernel, skipping the intermediate k_buf round-trip + the standalone kv_cache_write dispatch. The existing kv_cache_write shader's page-table indexing logic ports cleanly into a Q8_0 / Q4_K K-proj kernel's tail.",
      "Cross-token batched MoE FFN (phase 1.1, prefill lever — won't help decode metric). Shader dmmv_q4k_moe_batched.comp landed in c36bd23 with dispatch grid (M+1)/2, n_experts_used, n_tokens. Pipeline + DmmvDispatch.recordMoeBatchedDispatch helper available. Remaining work: (1) build per-layer routing buffer for all N prompt tokens; (2) allocate [N × n_experts_used × inter] output scratch; (3) dispatch new shader for gate / up / down; (4) per-token weighted accumulation kernel that scatters n_experts_used × inter outputs per token weighted by routing probs back into hidden; (5) relax canUseBatchedPrefillRdna for the Qwen 35B MoE family with per-layer-type detection. Only attempt if the controller's metric mode is 'prefill' — won't move the decode benchmark.",
      "Parallel-scan SSM prefill (also prefill-only). The 30 SSM layers in the Qwen 35B MoE family have token-recurrent state. Blelloch/Hillis-Steele scan over the N-token axis. Without this, even with batched MoE, Qwen 3.6 prefill caps around 35-40 tok/s (SSM stays sequential). With both, beats llama.cpp's 54.5. Reference llama.cpp mamba2 ggml_ssm_scan.",
      "GEMM-style Q4_K mmq (also prefill-only). dmmv_q4k_q8_1.comp from commit 27f0c76 is GEMV-only and proved no-op on RDNA4 GEMV. A GEMM variant where the dispatch axis includes an N-token batch makes integer-dot pay off — arithmetic intensity goes from K to N×K. Pairs with phase 1.1.",
      "VkCmdPipelineBarrier2 with explicit srcStageMask + srcAccessMask precision (vs the current PipelineBarrier1 we emit). Cycles 1/16/18 narrowed full → buffer-scoped barriers and measured flat — but they didn't change the API. PipelineBarrier2's explicit masks are different on RADV's path, and on a sync-bound benchmark like ours the API switch is worth measuring once before declaring all barrier work flat.",
    ],
    referenceImplementations: [
      {
        path: "/Users/stepan/Workspace/llama.cpp",
        focus: "Vulkan backend at ggml/src/ggml-vulkan/. mul_mmq.comp + mul_mmq_funcs.glsl for the GEMM-style mmq pattern (Q4_K and Q5_K stanzas at lines 303-364 of mul_mmq_funcs.glsl). vulkan-shaders/mul_mm.comp for dense matmul. mamba/mamba2 ggml_ssm_scan op in src/llama-graph.cpp + ggml/src/ggml-cuda/ssm-scan.cu (CUDA reference) for parallel-scan SSM prefill. Routing/expert grouping in vulkan-shaders/topk_moe.comp + count_experts.comp.",
      },
      {
        path: "/Users/stepan/Workspace/vllm",
        focus: "Expert routing + fused MoE: vllm/model_executor/layers/fused_moe/. Useful for understanding how production systems group tokens by expert with permutation indices.",
      },
    ],
  },
  6: {
    doc: "MULTI_HOUR_EFFORT_6_RDNA_QWEN36_PREFILL.md",
    summary: "RDNA Qwen36 prefill recovery (restore flagship TTFT and prefill telemetry)",
    metricMode: "prefill",
    primaryMetricLabel: "prefill tok/s",
    benchmarkPrompt: PREFILL_BENCHMARK_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "long-context prefill benchmark on RDNA for the Qwen3.6-35B flagship workload",
    knownFlatCategories: [
      "DORMANT TILED-GEMM FOUNDATIONS. Cycles 14, 16, 18, 21 ported all 4 pieces (count_experts, mul_mm_q4k, mul_mm_id_q4k, mul_mmq_q4k) as foundationKeeps. mul_mm_q4k was wired into the LM head only (N=1, the worst case for tiled GEMM) and measured FLAT (78.14 vs 78.55 noise band). The other 3 stayed dormant with zero callers. Cycle 40 reverted ~1470 LOC of dormant infra after audit. Re-porting the SAME shaders is a known dead-end; the unfinished work is the buffer-layout refactor needed to wire them into SSM proj / MoE FFN prefill, NOT another foundation port. mul_mm_q4k.comp + count_experts.comp survived; mul_mm_id_q4k and mul_mmq_q4k were deleted. See loops/efforts/MULTI_HOUR_EFFORT_6 for the full audit.",
      "GEMV-style cross-token MoE batching via dmmv_q4k_moe_batched.comp (commit c36bd23). Spent 9 wire-in cycles producing flat or negative results. Architecturally wrong for RDNA4: dispatches 1.7M workgroups (M × n_experts_used × n_tokens) where the device caps at ~1024 in flight. Pipeline + helper stay in tree as record but should not be wired. The structural answer is per-expert grouped GEMM (one WG per expert × tile, not per token × expert × row).",
      "REGISTER-CACHE EXTENSIONS BEYOND CYCLE 46. Cycles 47, 48, 49 attempted to apply the cycle-42/46 winning pattern (vec4 + register-cache for x[]) to rms_norm_dmmv_f32 (router), norm_rope.comp (per-head Q/K-norm+RoPE), and ssm_gated_norm.comp — all flat. Reasons vary (head_dim=128 too small for vec4 amortization on the head-pinned WGs; per-head WG already register-saturated; multi-WG reductions starve the cache). The pattern is exhausted on the obvious-shape kernels; future targets need shape-fit verification first (single-WG-per-token, K_v4 ≥ 64, no per-head-pinned dispatch).",
      "Three-way RMS+K+V or RMS+Q+K+V fusion shaders (cycle 11 of effort 6, cycle 14 of effort 10). Register pressure collapses occupancy on RDNA4. Two-way RMS+single-output fusions (cycles 8, 13 of effort 10) are fine — three-way isn't.",
      "Triple-fused MoE swiglu+down+weighted-acc shader (cycle 9 of effort 6). Regressed -0.5%. The existing path's down + moe_weighted_acc dispatches already overlap on RDNA4.",
      "Attention K+V DMMV fusion via dmmv_q4k_fused_gate_up (cycle 35). Flat on flagship. Three reasons: (1) full-attention layers are only ~1/3 of the model; (2) Q-proj and K-proj have different M values; (3) the K+V dispatch overlap on RDNA4 already saturates DRAM.",
      "SSM proj wqkv+z fusion via dmmv_q4k_fused_wqkv_z (cycle 38). When wired (ZINC_SSM_PROJ_FUSED=1) measured -2.2% regression in cycle 40. The shader was deleted. The wqkv and z DMMVs already overlap; fusing them adds register pressure without saving wall time.",
      "Wide NUM_ROWS variants on dmmv_q4k_moe_kpar (cycle 12 of effort 10). NUM_ROWS=4 underutilizes at MoE expert M=1408. uvec4 collapse of Q4_K/Q5_K block headers across three hot MoE shaders (cycle 43): flat.",
      "vec4 SwiGLU shader (cycle 44). Flat → -0.15 tok/s. The elementwise SwiGLU is already DRAM-bound on the X-vector; vec4 doesn't move the bottleneck.",
      "vec4 alias bindings on flash_attn.comp Phase 1 (q.k dot) and dmmv_q8_0.comp x_data reads (cycle 45). Flat. flash_attn's bottleneck is K/V tile reads not Q reads; dmmv_q8_0 X is already amortized.",
      "Narrowing a single compute→compute barrier to buffer-scoped on RADV (cycles 1, 16, 18 of effort 10; multiple cycles of effort 6 runs). Cumulative movement below 0.25 tok/s. Don't re-attempt without VkCmdPipelineBarrier2 explicit access masks.",
      "Adding phase/dispatch profiling without a downstream structural change. ZINC_PREFILL_PROFILE=1 already covers per-phase and MoE/SSM sub-bucket timing; the loop emits it on every cycle.",
      "Re-layering prefill embedding dequant (CPU f32 cache / staging-only / interleaved). Cycles 14, 15, 23 of the first run explored these; current upfront bulk dequant into host-mapped staging is the accepted equilibrium.",
      "Pair-dispatch via recordBatchDispatch(num_cols=2) through dmmv_q8_0_batch / dmmv_q4k_batch. Cycle 8 of the second run: -0.12 tok/s; cycle 9 rewrote with wave64 parallelism: -0.8 tok/s. K-parallel kpar shaders (commit bed8463 + e43da13) are now the canonical inner loop; specialized num_cols=2 variants on top haven't measured a win.",
      "Extending the prefill CB pipeline from 2-deep to 3-deep (cycle 2 of the second run): flat. Submit/wait is already saturated at 2-deep.",
      "POST-759 PLATEAU: do not repeat cycles 44-50 without new evidence. These all failed the official 154-token gate after the 759.18 checkpoint: route-pack sentinel cleanup, terminal attn_norm reuse, grouped Q4_K MoE Q8_1/DP4a, SSM-out branchless full-tile GEMM, fused shared-expert suffix reuse, appending grouped MoE into the SSM command, and SSM-out DP4a with widened Q8 scale scratch.",
      "Terminal full-attention K/V batching is coherence-sensitive. Cycle 41 was faster (762.76) but failed Qwen3.6-35B coherence with '!' and image-link garbage. Cycle 42/43 landed the safe variant. Any further terminal shortcut must run all 5 coherence models and must not bypass the final-token Q path.",
      "Exact-MoE guard below 16 prompt tokens is unsafe on this target. Cycle 33 shortened the guard to 8, broke output ('A.'), and was reverted. Do not reduce ZINC_QWEN36_MOE_PREFILL_TOPK_GUARD unless a validator proves top-k suffix equivalence on the benchmark prompt and the coherence sweep.",
      "SSM-out quantized alternatives are currently negative. Cycle 29 int8 DP4a SSM-out, cycle 47 branchless full-tile SSM-out, and cycle 50 widened-scale DP4a SSM-out all measured below the 759.18 checkpoint. A new SSM-out attempt needs an L2/max_abs validator against the f32 batched path first, not another default-on quantization variant.",
    ],
    structuralSwingIdeas: [
      "POST-759 PROFILE-FIRST CYCLE. The final budget is balanced: ssm 83.8 ms, moe 73.9 ms, attn 58.3 ms, shared 14.9 ms. Before adding another kernel, run one bounded analysis cycle that emits coverage counters for A3B production: layer-major SSM eligibility, grouped prefix/suffix token counts, exact-suffix guard count, terminal KV-only eligibility, and per-sub-bucket timings on the canonical 154-token prompt. Keep only source-visible counters that make the next optimization more targeted; do not add another default-on path in the same cycle.",
      "MoE gate_up/down is now the best structural target if the refreshed budget still shows moe near SSM. Current sub-buckets are gate_up 29.1 ms and down 25.6 ms. Avoid the reverted variants (Q8_1/DP4a grouped Q4_K gate/up, fused grouped Q4_K gate/up, shared-expert suffix reuse). Preferred next step: a route-density diagnostic plus a per-expert grouped exact-suffix plan that preserves top-k weights, or a small verified down-accumulator improvement that directly removes a measured MoE sub-bucket.",
      "SSM remaining work must start with correctness instrumentation. Current sub-buckets are out 30.3 ms, qkv_z 25.9 ms, conv 17.8 ms, gnorm 17.1 ms, delta 13.1 ms. SSM-out DP4a/full-tile attempts regressed after cycle 29/47/50; alpha/beta batching and conv parallelization also regressed. The next SSM cycle should build an L2/max_abs validator for SSM-out or gnorm against the accepted f32 path, then use that evidence to decide between Q8_1 activation, fusing residual+ffn_norm after SSM-out, or leaving SSM alone.",
      "Attention is no longer huge, but terminal-layer work still produced the last real keep. Current attn is 58.3 ms. Safe path: extend the cycle-42/43 terminal K/V reuse with a validator that compares last-token logits before/after any skipped terminal work. Unsafe path: do not repeat cycle-41's faster-but-incoherent final-layer shortcut. If attacking flash attention, use a bounded Q-cache or coopmat experiment behind a flag and validate all five coherence models.",
      "Cross-scenario guard before another large default-on keep. This effort's canonical prompt now reports 759 tok/s, far above the old public 154 tok/s figure. A large new keep should include one extra bounded public-shape smoke (core or context-medium) so the loop does not overfit the Paris benchmark while hurting coding/review prompts.",
      "If stuck for two more cycles, switch target instead of grinding SSM-out. The next high-value work is effort-15/17/18/19 cross-model prefill, not a tenth Qwen35 SSM micro-variant. Emit a measured-dead analysis and stop if no named sub-bucket plan can clear +3.8 tok/s over 759.18.",
    ],
    referenceImplementations: [
      {
        path: "/Users/stepan/Workspace/zinc/src/shaders/mul_mm_q4k.comp",
        focus: "ALREADY IN TREE. Tiled Q4_K GEMM ported in cycle 16, currently wired only for LM head (N=1, worst case for tiled GEMM). The shader is correct and validated. The unfinished work for swing #1 is wiring it into SSM proj prefill where N=batch_size_per_chunk. Read alongside src/compute/dmmv.zig recordMulMmQ4K helper.",
      },
      {
        path: "/Users/stepan/Workspace/zinc/src/shaders/count_experts.comp",
        focus: "ALREADY IN TREE. Per-expert token-count buffer ported in cycle 14, wired in cycle 22 (gated ZINC_COUNT_EXPERTS_PREFILL=1). Useful for swing #2(c) — culling zero-token experts before any MoE FFN dispatch.",
      },
      {
        path: "/Users/stepan/Workspace/zinc/src/shaders/dmmv_q4k_moe_kpar.comp",
        focus: "Current MoE expert FFN inner loop, NUM_ROWS=2 at expert M=1408. Target for swing #3 (cycle-50 wider-threads-per-row pattern). Lines 100-160 contain the Q4_K block decoder reused throughout the MoE shaders.",
      },
      {
        path: "/Users/stepan/Workspace/zinc/src/shaders/ssm_delta_net.comp",
        focus: "Cycle 50's winner — restructured 8 threads/row × 8 rows/tile to 16 threads/row × 4 rows/tile. Read this to understand the exact pattern to replicate on dmmv_q4k_moe_kpar and dmmv_q4k_moe_fused_down_acc. The reg_state[16] → reg_state[8] halving and subgroupShuffleXor 8 reduction are the two key transformations.",
      },
      {
        path: "/Users/stepan/Workspace/zinc/src/shaders/rms_norm_dmmv_f32.comp",
        focus: "Cycle 42's winner (+4.66%) — vec4 reads/writes across hidden, ffn_norm weights, router weights, with WG-0 ffn_norm writeback. Pattern: gate fused path on K%4==0. Read alongside rms_norm_mul.comp (cycle 46's similar +1.01% win with register-cache for x[]). These two shaders show the structure that landed wins on RDNA4.",
      },
      {
        path: "/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp",
        focus: "Reference for swing #1 wire-up. The tiled GEMM in tree (mul_mm_q4k.comp) was modeled on this — the dispatch-shape pattern is gridX=blocks_m × split_k, gridY=ceil(N/BN) tiles, gridZ=1 for dense / expert_idx for MUL_MAT_ID. For SSM proj wire-up, copy the gridY = ceil(154/BN) sizing and the buffer-layout convention for [N × K] activation tiles.",
      },
      {
        path: "/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp",
        focus: "Reference for swing #5 (Q pre-load). Look at the Q-tile shared-memory load at the top of the kernel and how it's reused across the K-tile inner loop. Our flash_attn.comp re-reads Q[head_dim] per K-tile iteration — this is the redundant work cycle 48 should have removed.",
      },
      {
        path: "/Users/stepan/Workspace/llama.cpp/ggml/src/llama-graph.cpp",
        focus: "Reference for swing #4 (parallel-scan SSM). Search for ggml_ssm_scan to find the prefill-time scan op. The CUDA implementation is at ggml/src/ggml-cuda/ssm-scan.cu — it computes the recurrence via Blelloch scan over the token axis. The math is identical between CUDA and Vulkan; what changes is the WG layout (one WG per head per chunk).",
      },
    ],
  },
  11: {
    doc: "MULTI_HOUR_EFFORT_11_RDNA_DECODE_LONG_CONTEXT.md",
    summary: "Flatten the RDNA4 decode-with-context curve on Qwen 3 8B + 35B (target: decode at L=1500 ≥ 60% of empty-context decode)",
    metricMode: "decode",
    primaryMetricLabel: "decode tok/s at L≈846",
    benchmarkPrompt: LONG_CONTEXT_DECODE_PROMPT,
    benchmarkMaxTokens: 32,
    benchmarkMethod: "decode 32 tokens after prefilling the LONG_CONTEXT_DECODE_PROMPT (tokenizes to ~846 tokens on Qwen 3 8B's BPE, NOT 1500 as the prompt comment originally claimed — calibration finding from run-3 cycle 11) on Qwen3-8B-Q4_K_M.gguf with RADV_PERFTEST=coop_matrix; report decode tok/s as the primary metric",
    knownFlatCategories: [
      "108.05 IS A HARD PLATEAU AT L≈846 WITHOUT COOPMAT. Run-5 ran 34 cycles (c24-c57) with ZERO perf-keeps. All 6 prior structuralSwingIdeas were directly DISPROVEN by measurement. The realistic next-tier improvement requires the multi-week flash_attn_cm1.comp KHR coopmat port. Single-cycle work in this regime should target the small-win remainder list in structuralSwingIdeas, not redo the disproven structural attacks below.",
      "FFN SUB-BUCKET CALIBRATION (run-4 c12): gate+up+SwiGLU is 60% of FFN (3.55 ms = 35% of total decode); down_proj is 40% of FFN (2.40 ms = 24% of total). The 'attack down_proj first' premise was wrong AND every direct attack on gate+up+SwiGLU has now also been disproved (see entries below).",
      "GATE+UP SPLIT-K ON K=4096 IS DEAD. Run-5 c37 (N_K_CHUNKS=2/4): -3.1% / -6.1%. Run-5 c55 (N_K_CHUNKS=2 + dedicated merge pass): -1.8%. Failure mode: gate+up already has 6144 WGs at intermediate_dim=12288, so the run-3 c12 occupancy unlock that gave flash_attn +3.2 tok/s (32 → 128 WGs) does not apply. Splitting K just adds merge-pass overhead with a 192-384 KB partials buffer plus an extra barrier. Don't propose split-K on the gate+up shader.",
      "LAST-WG-DOES-NORM CROSS-LAYER FUSION IS DEAD ACROSS 4 PATTERNS NOW. Separate-buffer (run-3 c5, -7%); atomic-counter cross-WG sync (run-3 c15, broken output); merge-shader piggyback (run-4 c7/c28, -67%/-69%); atomicAdd + GL_KHR_memory_scope_semantics device-scope acquire-release (run-5 c54, BIT-CORRECT but -1.4%). The synchronization overhead exceeds the saved dispatch+barrier even with proper memory ordering. Don't attempt o_proj+ffn_norm cross-layer fusion; the 36 dispatch+barrier savings/token is not enough to overcome any cross-WG sync mechanism.",
      "GATE+UP LDS-INPUT BROADCAST IS DEAD. Run-5 c35: cooperatively load 16 KB shared vec4 input at WG start, replace per-thread x_v4 reads with LDS reads. Result: -13.8% (92.95 vs 108.05). The original x_v4 reads were L1-cache hits; LDS staging halved occupancy without recovering anything. Same pattern that killed K/V LDS staging on flash_attn.",
      "HIDDEN-DIM-ROTATED DISPATCH ON down_proj IS DEAD. Run-5 c44 (ROT_STRIDE=137 spec const): -0.34%. The L1/L2 thrash hypothesis was wrong; the existing in-order dispatch is L2-cache-friendly. Don't propose dispatch-order rotations.",
      "Q+K+V MERGED-DISPATCH FUSION REGRESSES. Run-5 c49 (single dispatch, WG-ID branched across 3 Q4_K weight tensors): -0.23%. The three projections already overlap on RDNA4; merging them adds register pressure without saving wall time. Same pattern as the failed K+V fusion (run-2 c10, -1.1%).",
      "MMQ Q4_K × Q8_1 ROUTING FOR down_proj REGRESSES. Run-5 c51: -3.5%. The dmmv_q4k_q8_1.comp pipeline is correct (bias-fix + wave64 spec), but routing the down_proj through it adds activation-quantize overhead that exceeds the int-dot savings at the current decode shape (n_tokens=1).",
      "NUM_ROWS≠2 ON FUSED gate+up+SwiGLU IS DEAD. NUM_ROWS=4 (run-5 c41, -1.5% — halves WG count from 6144 → 3072, under-saturates). NUM_ROWS=1 narrow Q4_K for tall-K down_proj path (run-5 c39, -0.62%). NUM_ROWS=2 is the right shape on every dense Q4_K path; don't tune NUM_ROWS.",
      "ATTENTION-SHADER MICRO-TUNING IS EXHAUSTED ACROSS 5+ RUNS. ALL of these are flat/regress: V-load promotion; drop redundant Phase-4 barriers; subgroupShuffleXor merge variants; register-resident accumulators across blocks; 32-way ILP; narrow buffer-scoped barriers; register-resident max_old/sum_old per-lane (run-4 c19); subgroupBroadcast/Shuffle in split_merge (run-4 c29, run-5 c52); hoist Phase B reads ahead of Phase A (run-4 c30); pack 2 heads per WG cluster-32 (run-4 c14); wave64 pinning on pipeline_q4k (run-4 c31); pre-scale Q during s_q4 staging (run-5 c42, flat); fold sink_val into M_c before cluster-max (run-5 c43, flat); fold inv_L_full into s_weights (run-5 c45, flat); D-axis split on split_merge D_SPLIT=2 (run-5 c52, slight regress). The 16-way ILP + cluster-4/2 reductions are the saturation point.",
      "FFN-SHADER MICRO-TUNES ARE EXHAUSTED. uvec4 alias for header reads (run-4 c8, flat); algebraic refactor with by_sum reuse (run-4 c13, -0.4%); 2-way paired accumulators (run-4 c16, flat); row/block loop swap mirror of c22 (run-4 c24, flat); unpackHalf2x16 paired-halves (run-5 c34, flat); vec2 subgroupAdd packing on dmmv_q4k.comp (run-5 c36, flat); vec2 subgroupAdd packing on fused gate+up (run-5 c47, flat); hoist 12 gate+up u32 weight reads to top of inner block loop (run-5 c46, flat); residual_rms_norm.comp on decode path (run-5 c53, flat-to-regress).",
      "REMOVING gl_NumSubgroups>1 CROSS-SUBGROUP REDUCTION IN flash_attn REGRESSES (-0.44, run-4 c32). The branch appears unused on wave64=1-subgroup but contributes to compiler scheduling decisions.",
      "limit_occupancy_shmem (llama.cpp RDNA HACK) IS FLAT FOR DMMV. Run-5 c38: 24 KB dummy LDS to halve concurrent WGs/CU on dmmv_q4k.comp. Flat-to-slight-regress. The hack only helps flash_attn-style patterns where over-subscription thrashes the cache; DMMV doesn't have that pattern.",
      "BLOCK_SIZE / N_I_CHUNKS TUNING IS LARGELY FLAT. 256→512 / 256→384 / N_I_CHUNKS=8 all flat. EXCEPTION: run-5 c56 N_I_CHUNKS=8 + cluster-quad merge gained +0.41 at L=846 BUT regressed -39% at L=5. Multi-modal blocker for default-switch; would require runtime-conditional dispatch (cluster-2 at short, cluster-quad at long) to ship.",
      "LDS STAGING OF K OR V IN flash_attn IS UNAMBIGUOUSLY DEAD (3 attempts: -44%, -12%, -29%). Plus run-5 c35 confirms LDS staging of FFN input is also dead (-13.8%). DO NOT propose cooperative LDS staging in any form on the hot decode path.",
      "GQA COLLAPSE STARVES R9700 WITHOUT SPLIT-K. Q_PER_KV=4 (-9.9%); Q_PER_KV=2 (-17.5%). Split-K (N_I_CHUNKS=4) restores WG count and won +4.1% (run-3 c12). Combining Br=2 with split-K is the only untried GQA variant — see structuralSwingIdeas.",
      "TPB=32 DENSE Q4_K DMMV IS DEAD ACROSS 3 VARIANTS (run-4 c5/c6/c33: -1.3% / -6.5% / -0.83%). TPB=16 + NUM_ROWS=2 + cycle-22 row/block loop swap is the right shape.",
      "SPLIT-K / K-AXIS SPLIT ON DENSE Q4_K down_proj IS FLAT-TO-REGRESSING. Split-K kpar32_split (run-4 c4, flat); K-axis N_K_CHUNKS=2/4 (-0.83% / -0.97%). down_proj is already 78-85% of bandwidth-floor; merge-pass overhead exceeds the parallelism gain.",
      "rms_norm FOLDED INTO ADJACENT MATVECS IS AT-OR-BELOW THRESHOLD. -0.37 (rms+gate+up+SwiGLU); +0.49 (rms+Q/K/V); +0.16 (ffn_norm into existing fusion). Don't re-attempt.",
      "rms_norm_mul Pass-2 PER-THREAD rms_inv IS FLAT (run-4 c26, -0.12).",
      "V-CACHE FLOAT16 REGRESSES (-1.6%) at the current scalar V-load shape. Could revisit if a coopmat or Br>1 path absorbs unpack cleanly.",
      "WIDE NUM_ROWS DMMV VARIANTS UNDER-SATURATE. NUM_ROWS=8 dense Q4_K (-3.83); NUM_ROWS=8 Q6_K LM head (flat — LM head fires once, can't move metric).",
      "PER-TOKEN COS/SIN PRECOMPUTE FOR RoPE IS BELOW THRESHOLD (+0.29).",
      "GATE+UP-ONLY FUSION (without SwiGLU) IS FLAT. Only wins WITH SwiGLU folded in.",
      "K+V PROJECTION FUSION REGRESSES (-1.1%).",
      "sigmoid_mul (attn_gate) FOLDED INTO SPLIT-K MERGE PASS IS FLAT.",
      "MANUAL-CYCLE + LANDED-LOOP FOUNDATIONS — DON'T REVERT. Q-stage (6ece0a8); s_kv_base_v4 precompute; Phase 4 rescale+V-acc fusion (539b2aa); gate+up+SwiGLU dense FFN fusion (run-2 c8); Q+K norm+rope+kv_cache_write fusion (run-2 c12); split-K flash attention N_I_CHUNKS=4 + merge pass (run-3 c12); v_im uniform shift-scale Q4_K decode (run-4 c1); cluster-4 M/L reduction in split_merge (run-4 c22); cluster-2 chunk-pair split in split_merge (run-4 c23). All bit-correct. The dmmv_q4k_o_proj_merge.comp shader was DELETED in run-5 c40 (per the prior plan's directive); -430 LOC of dead infrastructure removed.",
      "L=2325 GPU HANG IS L≥2300-ONLY. Confirmed across 110+ cycles. Watchdog-duration issue, not correctness.",
      "MEASUREMENT NOTE: run-4's headline jump 99.38 → 108.05 (+8.7%) was MOSTLY system-state shift (thermal/RDNA driver re-baseline at run start), not earned by code. Real run-4 code-driven gains were ~+2.2% (cycle 1: +1.43%; cycle 22: +0.5%; cycle 23: +0.25%). The cycle-12 jump 101.62 → 107.27 was a measurement artifact (HEAD itself measured ~107 at that point). Future runs should expect run-to-run baseline drift of ±3% from system state alone, and only attribute code-wins above that band.",
    ],
    structuralSwingIdeas: [
      "flash_attn_cm1.comp KHR COOPMAT PORT — THE ONLY UNTRIED STRUCTURAL CEILING. Every cheap attack on flash_attn (15+ cycles across 5 runs) has been exhausted; every cheap attack on dense FFN (15+ cycles across runs 4-5) has been exhausted. Cooperative-matrix wmma intrinsics on the Q.K matmul + softmax-V matmul are the only structural shape that hasn't been attempted. RDNA3+ supports VK_KHR_cooperative_matrix and R9700 advertises it. Multi-week port (estimate 5-10 cycles for the shader rewrite + validation + tuning). Reference: /Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp. Read the Br=8/Bc=64 wmma decomposition and the cross-tile softmax reduce. Land behind ZINC_FA_COOPMAT=1, validate against the scalar split-K path at tol=1e-3 on all COHERENCE_MODELS.",
      "Br=2 + N_I_CHUNKS=4 COMBINED SPLIT-K (cycle 50's untried pivot). Standalone Q_PER_KV=2 GQA collapse failed -17.5% in run-3 c4 due to WG-count loss. The unattempted variant pairs it WITH split-K's WG multiplier: dispatch n_kv_heads × Q_PER_KV_BUDGET=2 × N_I_CHUNKS=4 = 8 × 2 × 4 = 64 WGs (vs the standalone Q_PER_KV=2's 16 WGs that starved). At 64 WGs on 64 CUs the SIMD pool is fed AND each WG amortizes K/V reads 2× across two query heads. Mirrors llama.cpp flash_attn.comp's Br/Bc shape. Build behind ZINC_FA_BR2=1.",
      "PROFILE-PHASE DECOMPOSITION OF THE 'OTHER' 33% BUCKET. The cycle-48 attempt added attn_input_norm/ffn_input_norm tags but didn't decompose all of the residual ~3 ms/token. Sub-categorize: attn_norm, ffn_norm, residual, scale_acc, lm_head, host_gap. May reveal a hidden hotspot (e.g., a per-token rms_norm that's actually 8% of decode and not gate+up at all). Cheap diagnostic; valuable if it surfaces an untouched bucket.",
      "CONDITIONAL DISPATCH FOR flash_attn_split_merge (unblocks cycle 56's +0.41 at L=846). Cycle 56 found N_I_CHUNKS=8 + cluster-quad merge gains +0.41 at long context but regresses -39% at L=5. The fix is to switch dispatch parameters at runtime based on L: cluster-2 chunk-pair (current default, fast at L=5) when L<512, cluster-quad (faster at L≥846) when L≥512. Single Zig-side dispatch decision based on seq_len; no shader changes needed. Captures the unrealized cycle-56 win.",
      "dmmv_q5k.comp ROW/BLOCK SWAP for Q5_K LM head (Qwen 3.6 35B). Mirror cycle-22's row/block swap pattern that won +0.48% on dmmv_q4k.comp and was confirmed +0.72% on dmmv_q6k.comp in run-5 c57 (kept-flat by framework's strict gate but the structural shape is right). Qwen 3.6 35B uses Q5_K weights; this would help the 35B coherence-target's decode rate without affecting the qwen8b primary metric. Cycle-sized.",
      "TPB=128 WIDENED Q6_K LM HEAD (cycle 57's nextIdea). LM head dispatches once per decoded token at M=151936; halving the b-loop iterations via wider TPB on the row-swap-restructured shader could shave 0.1-0.3 tok/s. Small absolute but structurally clean.",
      "GATE+UP+SwiGLU INPUT BROADCAST-VIA-LDS WITH NUM_ROWS=4 AS A UNIT (cycle 57's nextIdea). Distinct from the failed run-5 c35 (LDS-only, -13.8%) and run-5 c41 (NUM_ROWS=4-only, -1.5%). The combined hypothesis: NUM_ROWS=4 halves WG count by 2x, but if the same WG-group cooperatively stages 4 KB of input into LDS once and reuses it across 4 rows × gate+up matmuls, the amortization may exceed the LDS-staging penalty that killed the cycle-35 attempt. Risk: the LDS staging cost compounded with the 2x WG count reduction may still under-saturate. Build behind ZINC_GATEUP_NUM_ROWS_4_LDS=1.",
    ],
    referenceImplementations: [
      {
        path: "/Users/stepan/Workspace/zinc/src/shaders/flash_attn.comp",
        focus: "Current state after 17 cycles of optimization (31.19 → 93.68 tok/s at L=1500). Read the entire shader. Key structural shapes that delivered: (1) cycle 5 D-split — pair lanes (tid, tid+32) on the same d4 and split the i-axis to engage all 64 wave lanes when head_dim=128. (2) cycle 8 s_kv_base_v4 — precompute (page_id*page_size+page_off)*n_kv_heads*head_dim+kv_head*head_dim per i once per block, replacing the prior s_page_ids_block. (3) cycles 6/9/12/16 Phase 4 ILP unrolls (2-way → 4-way → 8-way → 16-way, paired vec4 accumulators). (4) cycles 10/14/17 Phase 1 ILP unrolls (4-way → 8-way → 16-way). The current 16-way pattern is the exhausted ceiling on ILP; 32-way (cycle 18) measured flat.",
      },
      {
        path: "/Users/stepan/Workspace/zinc/src/shaders/flash_attn_batched.comp",
        focus: "Mirrors flash_attn.comp's structure with grid.y = n_queries. Every cycle in this effort has mirrored the decode-side change to this shader so prefill and ZINC_BATCH_ATTN=1 paths benefit equally. Read alongside flash_attn.comp to see which lines line up.",
      },
      {
        path: "/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp",
        focus: "Scalar fallback flash_attn. Search for `Br` and `Bc` to see how multiple Q rows per WG are handled — that's the multi-Q-per-WG / GQA collapse target. Lines 44-90 are Q staging (we ported a simpler version). Lines 196-218 are SHMEM_STAGING for cooperative K loads (note: ZINC's cycle-1 attempt at K-parallel-with-subgroupAdd FAILED -44%; but cooperative K staging without the subgroup reduction is structurally different and unattempted). Lines 355-384 are the fused exp+V loop.",
      },
      {
        path: "/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp",
        focus: "Search for `gqa_ratio` and `get_fa_tuning_params_scalar` (~lines 2854-2928 and ~8866). The gqa_ratio collapse is the host-side change needed for multi-Q-per-WG: when n_heads / n_kv_heads = q_per_kv > 1, dispatch grid.x = n_kv_heads (not n_heads) and the shader processes q_per_kv query heads per WG. ZINC currently dispatches grid.x = n_heads.",
      },
      {
        path: "/Users/stepan/Workspace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp",
        focus: "KHR cooperative_matrix variant of flash_attn. Reference for the deferred cooperative-matrix swing. Look at how Q.K and score.V matmuls are decomposed into wmma tiles, what subgroup_size/coopmat_M/coopmat_N constraints apply, and how cross-tile softmax reduction is done.",
      },
      {
        path: "/Users/stepan/Workspace/zinc/src/shaders/dmmv_q4k_moe_kpar.comp",
        focus: "Wave64 K-parallel pattern reference. NOTE: the cycle-1 'K-parallel Phase 1' attempt that reduced ONE row across all 64 threads with subgroupAdd lost -44%. That is NOT the pattern this shader uses. dmmv_q4k_moe_kpar splits 64 threads as THREADS_PER_BLOCK=16 (cooperative on K dim) × NUM_ROWS=4 (parallel rows). For multi-Q-per-WG, the analog is THREADS_PER_BLOCK_PER_Q × Q_PER_WG. Read the inner loop and reduction shape.",
      },
    ],
  },
  13: {
    doc: "MULTI_HOUR_EFFORT_13_RDNA_GEMMA_LONG_DECODE.md",
    summary: "RDNA Gemma 4 26B MoE long-decode parity with llama.cpp (remove Gemma CPU MoE fallback)",
    metricMode: "decode",
    primaryMetricLabel: "Gemma 4 26B long-decode tok/s",
    benchmarkPrompt: GEMMA_LONG_DECODE_PROMPT,
    benchmarkMaxTokens: 32,
    benchmarkMethod: "32-token chat decode-extended benchmark on Gemma 4 26B A4B, matching the public benchmark matrix shape; run with --model gemma426ba4b",
    knownFlatCategories: [
      "Do not optimize the context-long Gemma win first. In the public data, ZINC generated only 2 tokens while llama.cpp generated 8, so it is an early-stop artifact. The primary metric is decode-extended with 32 generated tokens.",
      "Generic DMMV micro-tuning before removing Gemma CPU MoE fallback is the wrong order. The 26B MoE model is at 52% of llama.cpp on sustained decode while dense Gemma 31B is at 86%; the architectural delta points at MoE control flow, not a 1-2% matvec detail.",
      "Do not port llama.cpp grouped-GEMM mul_mat_id first. Decode top-k <= 8 should start with the lighter matvec-ID shape. Grouped GEMM/count_experts is for prefill or true multi-token batches.",
      "Q8_1 activation quant is not a default fix. llama.cpp enables it conditionally, and ZINC already measured Q8_1 regressions on some Qwen decode paths. Only try it after Gemma GPU MoE is coherent and faster.",
      "Do not skip Gemma-specific semantics to make the fast path fit. pre_ffw_norm_2, ffn_gate_inp.scale, fused ffn_gate_up_exps, ffn_down_exps.scale, post_ffw_norm_1/2, and post_ffw_norm are correctness requirements.",
    ],
    structuralSwingIdeas: [
      "Step 0 profile proof. Run Gemma 26B decode-extended with --profile -n 32 --chat and record cpu_moe_fallbacks plus moe_router/moe_topk/moe_gate_up/moe_swiglu/moe_down/moe_weighted_acc/shared/final_lm_head. If MoE is not a top bucket, update the effort doc before changing code.",
      "GPU top-k validation for Gemma. Allow dispatchSoftmaxTopk after Gemma router scaling, but do not consume it yet. Copy router_output_buf for first token/layer and compare IDs/weights against CPU topKSoftmax. Keep CPU fallback as the actual output path until this validates.",
      "Support fused ffn_gate_up_exps. Gemma 26B uses one fused gate+up expert tensor. Either add offset-aware pushDispatch5/6 helpers so the same buffer can be bound twice with binding 1 at up_base_offset, or add a Gemma-specific fused-gate-up MoE shader with an up_base_offset push constant.",
      "Enable GPU-routed Gemma MoE behind a flag. Preserve unit router RMS + ffn_gate_inp.scale, pre_ffw_norm_2 expert input, selected-only normalized softmax weights, ffn_down_exps.scale, and post_ffw_norm ordering. The first accepted implementation only needs to remove router readback and serial expert dispatch while staying coherent.",
      "Collapse per-expert serial dispatch. The target dispatch shape is one all-selected gate/up dispatch, one activation dispatch, one all-selected down dispatch, and one weighted accumulation. That is ZINC's decode-time equivalent of llama.cpp mul_mat_vec_id.",
      "After GPU MoE lands, test Q4_K x Q8_1 activation quant on exact Gemma expert shapes only. Treat it as a measured follow-up, not the first lever.",
    ],
    referenceImplementations: [
      {
        path: "/Users/zolotukhin/Workplace/llama.cpp/src/llama-graph.cpp",
        focus: "Search for build_moe_ffn. This is the graph-side reference for keeping router logits, top-k, selected weights, fused gate_up_exps, per-expert scales, and down projection in the graph.",
      },
      {
        path: "/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp",
        focus: "Search for ggml_vk_topk_moe, ggml_vk_mul_mat_id, ggml_vk_mul_mat_vec_id_q_f16, ggml_vk_use_mul_mat_vec_id, and ggml_vk_should_use_mmvq. These are the Vulkan control-flow pieces ZINC needs to mirror conceptually.",
      },
      {
        path: "/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl",
        focus: "MUL_MAT_ID shader addressing: data_ids selects expert_id; expert_id offsets the stacked expert weight tensor; expert_i0/expert_i1 select top-k slot and token/batch lane.",
      },
      {
        path: "/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/topk_moe.comp",
        focus: "Reference for fused top-k MoE softmax/normalization. ZINC already has dispatchSoftmaxTopk; use this to compare semantics and edge cases.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/src/compute/forward.zig",
        focus: "Current ZINC Gemma fallback. Search for use_gpu_moe, router_staging, topKSoftmax, fused_gate_up, ffn_gate_inp_scale, ffn_down_exps_scale, and post_ffw_norm.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/src/shaders/dmmv_q4k_fused_gate_up_swiglu_moe.comp",
        focus: "Existing selected-expert fused gate/up/SwiGLU shader. It assumes separate gate/up bindings today; adapt or replace for Gemma's fused ffn_gate_up_exps layout.",
      },
    ],
  },
  15: {
    doc: "MULTI_HOUR_EFFORT_15_RDNA_QWEN36_27B_PREFILL_DECODE.md",
    summary: "RDNA4 Qwen 3.8 27B dense-hybrid prefill/decode optimization",
    metricMode: "prefill",
    primaryMetricLabel: "Qwen3.8-27B prefill tok/s",
    defaultModel: "qwen3827b",
    benchmarkPrompt: QWEN38_27B_CONTEXT_MEDIUM_PREFILL_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "site-aligned 212-token context-medium Coding Review chat prefill benchmark on RDNA for Qwen3.8-27B dense Q4_K_M, including the server-default system turn; run with --model qwen3827b",
    minHealthyTokPerSec: 100,
    knownFlatCategories: [
      "Do not optimize against the old synthetic Paris prompt or raw prompt mode. The current Qwen3.8 contract is the site context-medium chat workload: published ZINC prefill is 402.92 tok/s and decode is 31.86 tok/s, versus llama.cpp 308.38 and 30.79 respectively.",
      "Do not relax canUseBatchedPrefillRdna for cfg.ssm_d_inner > 0 as a first step. A prior SSM batched prefill attempt caused QueueSubmitFailed / GPU resets and had a real hidden-state dependency bug.",
      "Do not repeat the widened dense fused gate+up+SwiGLU path for inter_dim=17408. It was mixed or negative on the same-shape dense 27B predecessor across the four-scenario matrix.",
      "Do not repeat Q6_K+Q4_K fused SSM qkv+z pair dispatch. It engaged but regressed the SSM projection bucket.",
      "Do not flip ZINC_SSM_DELTA_COLS8=0 or retry ZINC_SSM_DELTA_NORMED_QK=1 without new evidence. Both were mixed or negative on the full 27B matrix.",
      "Do not retry Q6_K K=17408 dense-down specialization or broad Q4/Q6 wide variants. They were flat or negative on the 27B matrix.",
      "Do not keep sweeping Q6_K dense-down tiled-kernel variants after the 64.87 tok/s checkpoint without fresh shaderstats. Cycle 27 kept the default-on tiled Q6_K path, but later Q6_K force-wave64, BN64, dequant-hoist, and tail-column FMA-skip variants all measured flat or dead.",
      "Do not keep sweeping Q4_K gate/up/SwiGLU tile shapes without fresh shaderstats. BM16, K=5120 specialization, default-shape tweaks, and post-64 tok/s BN retile variants have all been flat/dead or too small to clear the keep threshold.",
      "Do not spend another cycle on submit/barrier cosmetics around the layer-major prefill path unless the edit removes a named measured barrier cost. The cycle-32 scoped barrier keep was real; subsequent scoped-barrier, compute-to-transfer-barrier, and SSM+dense command-buffer fusion attempts were measured dead.",
      "Do not widen the fused attention o-proj merge to hidden_dim=5120. It caused a severe long-context regression.",
      "Do not repeat direct descriptor-offset SSM prefill projection replay. Cycle 36 measured flag OFF 31.34 tok/s vs flag ON 31.22 tok/s and reverted it.",
      "Do not repeat Q5_K row4 SSM-out, SSM delta tile8, or other SSM-side variants while dense_ffn remains the largest phase bucket. Recent SSM cycles produced zero perf keeps.",
      "Do not repeat Q4_K scale-unpack cleanup, BN=64 mul_mm_q4k projection batches, Q6_K batched-kpar chunk tuning, prefix dense-down batched accumulate, or wave32 row1 selector without new dense subphase evidence. These measured flat or negative around the 31.29 tok/s checkpoint.",
      "Do not repeat lower-bound dense segment additions unless a fresh profile proves the first layers are now hot. Layer-1 extension measured old 4-62 override at 50.16 tok/s vs new layer-1 schedule at 50.07 tok/s and reverted; earlier prefix-depth/layer 2/3/4/8 sweeps were also flat or negative.",
      "Do not repeat fusing the partial hidden scratch copy with the first attention-layer RMS norm at full-attention segment handoff. Measured with ZINC_QWEN36_27B_PARTIAL_ATTN_NORM_STORE: OFF median 49.96 tok/s [51.32, 49.82, 49.96] vs ON median 49.53 tok/s [49.53, 49.42, 49.62]; reverted. It also moved attention RMS work outside the normal phase timer, making profiles less trustworthy.",
    ],
    structuralSwingIdeas: [
      "[TOP PRIORITY] Capture a fresh Qwen3.8 phase profile before editing. The predecessor's DP4a fusion neighborhood is saturated, but Qwen3.8 must be profiled on its own GGUF and chat token shape. Only reuse an old dense-27B conclusion when the tensor type, M/K shape, and active production path match the new profile.",
      "The unspent structural lever from the predecessor effort is its validator-first Tracks 1-3: capture per-layer reference tensors against the per-token path, then batch one layer's dense FFN in validated chunks, then batch SSM projections while preserving exact token-order recurrence. Do not propose another DP4a/quantize-activation/fuse-residual variant unless paired shaderstats proves a specific occupancy or memory-instruction problem in the accepted path.",
      "At the current 64.87 tok/s checkpoint, the profile is balanced rather than single-hot: dense_ffn ~=1848 ms, ssm ~=1498 ms, dense gateup ~=934 ms, dense down ~=916 ms, and SSM proj ~=1095 ms. A next jump probably needs a structural change that moves a whole bucket by multiple percent, not another sub-1% tile/barrier variant.",
      "Before more dense kernel rewrites, collect paired RADV_DEBUG=shaderstats for the accepted fused Q4_K gate/up/SwiGLU path and Q6_K tiled dense-down path. Only edit the shader if shaderstats shows a concrete VGPR/SGPR, occupancy, LDS, spill, or memory-instruction problem that maps to the currently largest dense subphase.",
      "Treat the current dense 27B layer-major segment and barrier schedule as provisionally settled. Segment or barrier work now requires a paired old-vs-new control in the same cycle and a profile-backed reason; otherwise switch buckets.",
      "If dense gateup and down remain tied, pivot to SSM projection as the largest single subphase. Revisit batched SSM qkv/z/alpha/beta only as a validated layer-major dataflow step, not the old descriptor-offset replay path, and measure flag OFF/ON in the same cycle.",
      "If pursuing dense down, do not retune the existing Q6_K tile shape again. Either remove the separate residual accumulation with a correctly validated down+acc design, or collect shaderstats proving why the accepted tiled path is leaving occupancy/bandwidth on the table.",
      "After any new keep above 65 tok/s, run the full four-scenario matrix before treating the win as broadly useful. The site-aligned Coding Review prefill benchmark is the controller metric, but the 27B work has already produced changes that helped one scenario while hurting context-long/decode.",
      "Keep production prefill changes behind a 27B-specific flag until ZINC_BATCHED_PREFILL=validate or an equivalent validator proves final logits and intermediate tensors. Flag-gated paths must be measured flag OFF and flag ON in the same cycle.",
    ],
    referenceImplementations: [
      {
        path: "/Users/zolotukhin/Workplace/zinc/src/compute/forward.zig",
        focus: "Read canUseBatchedPrefillRdna, prefillBatchedImpl, prefillBatch, runSsmLayerGpu, dispatchProjectionBatched, dispatchDmmvAcc, and the dense_ffn profile phase before editing.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/loops/efforts/MULTI_HOUR_EFFORT_15_RDNA_QWEN36_27B_PREFILL_DECODE.md",
        focus: "This effort's measured baselines, failed-attempt list, and staged plan. Follow Track 1 before any production SSM prefill change.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/loops/efforts/MULTI_HOUR_EFFORT_6_RDNA_QWEN36_PREFILL.md",
        focus: "Historical RDNA Qwen prefill attempts, especially dormant tiled-GEMM lessons and SSM capture/validation failures.",
      },
    ],
    llamaCppSuccessRule: "Qwen3.8 already beats the validated Vulkan llama.cpp baseline in prefill and decode on all four RDNA scenarios. Optimize beyond the published ZINC medians without giving back correctness or any scenario: core 241.61/32.17, context-medium 402.92/31.86, context-long 431.16/31.79, and decode-extended 300.18/31.69 tok/s (prefill/decode).",
    // Qwen3.8 baselines from the fair reusable-server measurement contract
    // (R9700, RADV gfx1201, RADV_PERFTEST=coop_matrix, current Vulkan
    // llama.cpp build). Refresh whenever the baseline binary changes.
    llamaCppBaselines: [
      { scenario: "core",              promptTokens: 87,  prefillTokPerSec: 198.04, decodeTokPerSec: 30.86 },
      { scenario: "context-medium",    promptTokens: 212, prefillTokPerSec: 308.38, decodeTokPerSec: 30.79, isPrimary: true },
      { scenario: "context-long",      promptTokens: 378, prefillTokPerSec: 367.50, decodeTokPerSec: 30.75 },
      { scenario: "decode-extended",   promptTokens: 109, prefillTokPerSec: 232.26, decodeTokPerSec: 30.73 },
    ],
  },
  17: {
    doc: "MULTI_HOUR_EFFORT_17_RDNA_QWEN35_9B_PREFILL.md",
    summary: "RDNA4 Qwen 3.5 9B prefill gap recovery (site long-draft/core)",
    metricMode: "prefill",
    primaryMetricLabel: "Qwen3.5-9B long-draft prefill tok/s",
    defaultModel: "qwen359b",
    benchmarkPrompt: QWEN35_9B_LONG_DRAFT_PREFILL_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "site-aligned decode-extended Long Coding Plan prefill benchmark on RDNA for Qwen3.5-9B Q4_K_M; run with --model qwen359b",
    minHealthyTokPerSec: 50,
    knownFlatCategories: [
      "Do not chase decode first. Qwen3.5-9B decode is already ahead of llama.cpp on all four published RDNA scenarios (~95-97 tok/s ZINC vs ~85 tok/s llama). The gap is prompt prefill.",
      "Do not optimize a synthetic Paris prompt. The biggest published RDNA gap is the site long-draft raw prompt: 105.91 tok/s ZINC vs 855.82 tok/s llama.cpp. Core prefill is also poor (100.79 vs 548.94).",
      "Do not blindly flip canUseBatchedPrefillRdna for cfg.ssm_d_inner > 0. Qwen3.5 is an SSM+attention hybrid, and earlier SSM batched-prefill attempts on the Qwen3.6 dense hybrid produced GPU resets or hidden-state dependency bugs when validation was skipped.",
      "Do not reuse the legacy isQwen36DenseHybrid27B predicate as-is. It is shape-locked to hidden_dim=5120/intermediate_dim=17408, so every accepted 27B layer-major prefill path is currently disabled on Qwen3.5-9B.",
      "Do not add another single-token DMMV micro-fusion before proving the prefill path is layer-major on this model. A 5-8x prefill gap is a batching/dataflow problem, not a 1% shader clean-up problem.",
    ],
    structuralSwingIdeas: [
      "First cycle should prove the active path. Run the baseline with ZINC_PREFILL_PROFILE=1 and inspect whether Qwen3.5-9B falls through to token-major prefillBatch because canUseBatchedPrefillRdna rejects cfg.ssm_d_inner > 0. If profile phase data is missing, add only the missing labels before changing kernels.",
      "Generalize the Qwen3.6 dense-hybrid prefill enablement to a shape-safe Qwen dense SSM hybrid helper. Replace the hard hidden_dim=5120/intermediate_dim=17408 predicate with a helper that admits qwen35 architecture, n_experts=0, ssm_d_inner>0, full_attn_interval=4, required tensors present, and known supported Q4_K/Q5_K/Q6_K projection types. Keep the old 27B env names working, but add Qwen3.5-neutral or QWEN35_9B-specific flags for risky changes.",
      "Port the validated layer-major prefix/segment plan to the smaller Qwen3.5 shape. Start with a validation/foundation cycle that reuses the existing dense FFN and SSM projection validators on one layer and a 16/32/64-token chunk. Do not make it production-default until final logits or captured tensors match the token-major reference.",
      "Once validation passes, enable the minimum production layer-major path that can move the long-draft prefill metric: batched dense FFN for prefix SSM layers plus batched full-attention segment layers. Qwen3.5-9B has smaller hidden/inter dims than 27B, so setup overhead may dominate at short prompts; test chunk thresholds instead of assuming the 27B thresholds transfer.",
      "Use the phase budget to choose the bucket after the first validated keep. Expected candidates are SSM projection/conv/delta and dense FFN gate/up/down. If dense FFN dominates, reuse the Qwen3.6 DP4a dense gate/up/down machinery only after checking tensor types and n_tokens thresholds. If SSM dominates, prefer layer-major projection batching before delta-scan rewrites.",
      "After any keep above 150 tok/s, run or at least prepare the full four-scenario matrix before declaring success. Core and long-draft prefill expose different shapes: core has 36 prompt tokens, long-draft has 64, context-medium has 174, and context-long has 322. A threshold that helps 64 tokens but regresses 174/322 is not a public win.",
    ],
    referenceImplementations: [
      {
        path: "/Users/zolotukhin/Workplace/zinc/src/compute/forward.zig",
        focus: "Read canUseBatchedPrefillRdna, isQwen36DenseHybrid27B, qwen36DensePrefillPrefixLayers, qwen36DensePrefillSegmentLayers, prefillQwen36DenseFfnPrefix, prefillQwen36RunBatchedDenseFfnLayer, prefillQwen36RunFullAttnLayerToFfnNorm, and dispatchProjectionBatched before editing.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/loops/efforts/MULTI_HOUR_EFFORT_15_RDNA_QWEN36_27B_PREFILL_DECODE.md",
        focus: "Use the staged validator-first approach and failed-attempt list. The mechanism is relevant to Qwen3.5, but the 27B-specific shape constants and thresholds are not.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/tools/performance_suite.mjs",
        focus: "Source of the public prompt matrix. contextualizePrompt(..., decode-extended) defines the Long Coding Plan prompt used as this effort's primary metric.",
      },
    ],
    llamaCppBaselines: [
      { scenario: "core",              promptTokens: 36,  prefillTokPerSec: 548.94, decodeTokPerSec: 85.51 },
      { scenario: "context-medium",    promptTokens: 174, prefillTokPerSec: 202.32, decodeTokPerSec: 85.10 },
      { scenario: "context-long",      promptTokens: 322, prefillTokPerSec: 205.64, decodeTokPerSec: 85.15 },
      { scenario: "decode-extended",   promptTokens: 64,  prefillTokPerSec: 855.82, decodeTokPerSec: 84.96, isPrimary: true },
    ],
    llamaCppSuccessRule: "Project success rule (from MULTI_HOUR_EFFORT_17_RDNA_QWEN35_9B_PREFILL.md): close the Qwen3.5-9B RDNA prefill gap without giving back ZINC's decode lead. Primary target is decode-extended prefill, where llama.cpp is 855.82 tok/s and ZINC is 105.91 tok/s in the published artifact. A keep must preserve coherent output and should be checked against core/context-medium/context-long before being treated as public progress.",
  },
  25: {
    // RDNA2-node sibling of effort 17: consumer RX 9070 XT (GFX1201, Mesa
    // 26.0.0-devel). Select with ZINC_RDNA_NODE=rdna2. Cycle-0 already landed
    // the node-specific DP4a regression fix (6.8 -> ~171 tok/s); this effort
    // chases the remaining prefill gap to llama.cpp's 973 tok/s ceiling.
    doc: "MULTI_HOUR_EFFORT_25_RDNA2_907XT_QWEN9B_PREFILL.md",
    summary: "RDNA2 (RX 9070 XT / Mesa 26) Qwen 3.5 9B prefill recovery (opencode/GLM-5.2)",
    metricMode: "prefill",
    primaryMetricLabel: "Qwen3.5-9B RDNA2 prefill tok/s",
    defaultModel: "qwen359b",
    benchmarkPrompt: QWEN35_9B_LONG_DRAFT_PREFILL_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "site-aligned decode-extended Long Coding Plan prefill benchmark on RDNA2 (RX 9070 XT, Mesa 26.0.0-devel) for Qwen3.5-9B Q4_K_M; run with ZINC_RDNA_NODE=rdna2 --model qwen359b",
    // Fixed cycle-0 baseline is ~171 tok/s; the DP4a regression drops it to
    // ~7. A floor of 80 catches "regression returned or worse" without
    // false-tripping normal variance.
    minHealthyTokPerSec: 80,
    knownFlatCategories: [
      "Do not re-enable DP4a dense FFN for the 9B on this node. The cycle-0 fix deliberately defaults it off when amd_rdna4 + cooperative_matrix are both exposed: the int8 DP4a gate+up+SwiGLU shader runs ~25x slower than the branchless path on Mesa 26.0.0-devel. ZINC_QWEN_DENSE_FFN_DP4A=1 reproduces 6.8 tok/s. Only revisit after Mesa 26.0 stable ships and a paired measurement shows the shader is fast again.",
      "Do not add the coopmat+rdna4 guard to the SSM DP4a gates (qwenDenseSsmOutDp4aEnabled / qwenDenseSsmProjDp4aEnabled / qwenDenseProjectionDp4aEnabled) expecting a win. Tested cycle-0: SSM moved 165.8 -> 163.5 ms (noise). The 9B SSM projections are NOT on the DP4a path.",
      "Do not sweep ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS. Forcing prefix layers 3/8/16/31 changed nothing — the layer-major prefix path already engages at prefix=3; the cost is inside it, not in layer count.",
      "Do not optimize decode. Qwen3.5-9B decode (~39.6 tok/s) already beats llama.cpp (21.7 tok/s) by ~75% on this node. The low absolute number is a card/driver characteristic hitting both runtimes; it is not a ZINC regression.",
      "Do not run on RDNA1. An autonomous zinc_rt_autopilot loop owns RDNA1 and overwrites zig-out/bin/zinc with its ZINC_RT build. This effort is RDNA2-only (ZINC_RDNA_NODE=rdna2).",
      "Do not trust the 'Modeled decode bandwidth' line — its 576 GB/s theoretical is hardcoded for the R9700 and the MB/token model is inflated on this node. Use wall-clock tok/s.",
    ],
    structuralSwingIdeas: [
      "[LARGEST SINGLE SUB-PHASE] Eliminate the standalone dense-FFN residual add (dense_ffn_residual_acc ~74 ms). dispatchScaleAcc(hidden += down_out) runs as its own dispatch + barrier per layer and is launch-overhead-bound for the bytes touched. Either fuse the residual into the dense-down projection epilogue (a down+acc shader variant that writes hidden[i] += result[i], removing 32 dispatches + 32 barriers), or make scale_accumulate.comp process more elements per workgroup via a stride loop. Either way measure prefill AND decode (scale_accumulate is shared) and confirm bit-identical output.",
      "SSM proj (~73 ms) + qkv (~59 ms) are the largest matmul sub-phases and run on the generic batched matmul. This is the structural GEMM gap to llama.cpp (973 tok/s). Multi-cycle work: collect paired RADV_DEBUG=shaderstats for the active SSM projection shader before editing, and prefer layer-major batching of wqkv/z with token-order recurrence preserved over single-shader tile tweaks.",
      "dense_ffn down_matmul (~60 ms) is on the generic path (DP4a down intentionally off here). If the down+acc fusion lands it partially absorbs this bucket; otherwise shaderstats-first like the SSM track.",
      "Before any kernel edit, capture a fresh ZINC_PREFILL_PROFILE=1 run on RDNA2 and confirm the exact sub-phase you intend to move. Add missing phase labels as a foundation step before any optimization commit.",
    ],
    referenceImplementations: [
      {
        path: "/Users/zolotukhin/Workplace/zinc/src/compute/forward.zig",
        focus: "Read qwenDenseFfnDp4aEnabled (the cycle-0 gate), dispatchQwen36DenseDown, dispatchScaleAcc, the dense_ffn_residual_acc phase (~20650), and dispatchQwen36DenseGateUpDp4a before editing.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/src/shaders/scale_accumulate.comp",
        focus: "Vec4-coalesced residual-add shader; candidate for a stride-loop to cut WG launch count, shared by all dispatchScaleAcc callers.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/loops/efforts/MULTI_HOUR_EFFORT_17_RDNA_QWEN35_9B_PREFILL.md",
        focus: "The R9700/RDNA1 sibling. Same model + metric, but its baselines and accepted paths assume Mesa 25.0.7 without cooperative matrix — do not port DP4a enablement from it to this node.",
      },
    ],
    // llama.cpp ceiling measured on THIS node (9070 XT, Mesa 26.0.0-devel,
    // build 9a532ae, Vulkan/RADV). pp512 / tg256 raw. Decode is already won;
    // these frame the prefill gap only.
    llamaCppBaselines: [
      { scenario: "decode-extended",   promptTokens: 64,  prefillTokPerSec: 973.49, decodeTokPerSec: 21.74, isPrimary: true },
    ],
    llamaCppSuccessRule: "Node-specific success rule: close the Qwen3.5-9B prefill gap on the RX 9070 XT (llama.cpp 973 tok/s vs ZINC ~171 tok/s after cycle-0) without regressing decode (ZINC 39.6 already beats llama 21.7). Do not re-enable DP4a dense FFN on this coopmat RDNA4 node.",
  },
  18: {
    doc: "MULTI_HOUR_EFFORT_18_RDNA_GEMMA26_PREFILL.md",
    summary: "RDNA4 Gemma 4 26B-A4B MoE prefill parity with llama.cpp",
    metricMode: "prefill",
    primaryMetricLabel: "Gemma 4 26B-A4B long-draft prefill tok/s",
    defaultModel: "gemma426ba4b",
    benchmarkPrompt: GEMMA_LONG_DRAFT_PREFILL_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "site-aligned decode-extended Long Coding Draft prefill benchmark on RDNA for Gemma 4 26B-A4B MoE Q4_K_M; run with --model gemma426ba4b",
    minHealthyTokPerSec: 40,
    knownFlatCategories: [
      "Do not chase decode first. Gemma 26B decode is still below llama.cpp, but this effort exists for the larger prefill gap: 92.68 tok/s ZINC vs 647.16 tok/s llama.cpp on the published long-draft scenario.",
      "Do not repeat generic single-token DMMV micro-tuning as the first lever. A 7x long-draft prefill gap on an MoE model is an expert grouping / batching problem, not a 1-2% matvec cleanup problem.",
      "Do not treat Effort 13's decode-time Gemma MoE plan as sufficient. Decode top-k <= 8 can use matvec-ID; prefill needs token-grouped expert work across N prompt tokens, with selected-token compaction and scatter/accumulate back to each token.",
      "Do not bypass Gemma semantics to make the fast path fit. pre_ffw_norm_2, ffn_gate_inp.scale, fused ffn_gate_up_exps, ffn_down_exps.scale, post_ffw_norm_1/2, and post_ffw_norm are correctness requirements.",
      "Do not spend cycles re-opening the old Effort 9 Gemma attention validate bugs unless a fresh validator points there. The remaining published gap is after Gemma batched prefill was enabled; the MoE bucket is the likely untreated structural cost.",
    ],
    structuralSwingIdeas: [
      "First prove the active buckets. Run Gemma 26B long-draft prefill with ZINC_PREFILL_PROFILE=1 and record prefill_moe/router/topk/gate_up/swiglu/down/weighted_acc/shared buckets plus cpu_moe_fallbacks. If MoE is not a top bucket, update this effort before changing kernels.",
      "Build a Gemma MoE prefill validator before enabling production. Capture token-major reference outputs for one MoE layer over a short prompt, then compare GPU top-k IDs/weights, fused gate/up outputs, down outputs, per-token weighted accumulation, post norms, and final logits.",
      "Move router/top-k fully on GPU for all prompt tokens. Reuse dispatchSoftmaxTopk only if it matches Gemma's scaled router semantics and selected-only normalization. Keep the CPU fallback as the actual output path until IDs and weights match.",
      "Group prompt tokens by selected expert. The target dataflow is count_experts -> prefix offsets -> compact (token, topk_slot, weight) pairs by expert -> one batched gate/up operation per active expert group -> activation -> one batched down operation -> scatter weighted outputs back to the original token rows.",
      "Support fused ffn_gate_up_exps in the grouped path. Either bind the same expert tensor with an up-base descriptor offset or write a Gemma-specific batched fused-gate-up shader with an up_base_offset push constant. Do not split/copy the 26B expert weights.",
      "After correctness, choose between a lighter matvec-ID-style grouped path and a tiled GEMM path based on token count. Core has ~49 ZINC prompt tokens and long-draft has ~70; context-medium/context-long have ~192/346. A single threshold is probably wrong.",
      "Only after grouped MoE is coherent, test Q4_K x Q8_1 activation quant on the exact Gemma expert shapes. It is a measured follow-up, not the first lever.",
    ],
    referenceImplementations: [
      {
        path: "/Users/zolotukhin/Workplace/llama.cpp/src/llama-graph.cpp",
        focus: "Search for build_moe_ffn. This is the graph-side reference for Gemma router scaling, top-k, fused gate_up_exps, selected weights, and per-expert scales.",
      },
      {
        path: "/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp",
        focus: "Search for ggml_vk_topk_moe, ggml_vk_mul_mat_id, count_experts, and grouped MoE dispatch selection. Decode matvec-ID is useful background, but this effort needs the prefill/batched-token shape.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/loops/efforts/MULTI_HOUR_EFFORT_13_RDNA_GEMMA_LONG_DECODE.md",
        focus: "Gemma MoE semantic checklist. Reuse the correctness requirements, but do not stop at decode-time matvec-ID.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/loops/efforts/MULTI_HOUR_EFFORT_9_RDNA_GEMMA_BATCHED_PREFILL.md",
        focus: "Historical Gemma batched prefill correctness bugs: V RMS norm, use_k_as_v, asymmetric head dims, post norms, and SWA handling.",
      },
    ],
    llamaCppBaselines: [
      { scenario: "core",              promptTokens: 49,  prefillTokPerSec: 497.08, decodeTokPerSec: 102.00 },
      { scenario: "context-medium",    promptTokens: 192, prefillTokPerSec: 186.67, decodeTokPerSec: 100.84 },
      { scenario: "context-long",      promptTokens: 346, prefillTokPerSec: 169.18, decodeTokPerSec: 100.40 },
      { scenario: "decode-extended",   promptTokens: 70,  prefillTokPerSec: 647.16, decodeTokPerSec: 101.05, isPrimary: true },
    ],
    llamaCppSuccessRule: "Project success rule (from MULTI_HOUR_EFFORT_18_RDNA_GEMMA26_PREFILL.md): close the Gemma 4 26B-A4B RDNA prefill gap on all four public scenarios while preserving Gemma-specific correctness. Primary target is decode-extended prefill, where llama.cpp is 647.16 tok/s and ZINC is 92.68 tok/s in the published artifact. Core prefill is also a major target at 497.08 tok/s llama.cpp vs 89.10 tok/s ZINC.",
  },
  19: {
    doc: "MULTI_HOUR_EFFORT_19_RDNA_GEMMA31_PREFILL.md",
    summary: "RDNA4 Gemma 4 31B dense prefill parity with llama.cpp",
    metricMode: "prefill",
    primaryMetricLabel: "Gemma 4 31B long-draft prefill tok/s",
    defaultModel: "gemma431b",
    benchmarkPrompt: GEMMA_LONG_DRAFT_PREFILL_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "site-aligned decode-extended Long Coding Draft prefill benchmark on RDNA for Gemma 4 31B dense Q4_K_M; run with --model gemma431b",
    minHealthyTokPerSec: 20,
    knownFlatCategories: [
      "Do not optimize Gemma 26B MoE work inside this effort. Gemma 31B is dense; the prefill gap is dense projection batching, attention, and fixed overhead, not expert routing.",
      "Do not chase context-medium/context-long first. ZINC already beats llama.cpp on those prefill scenarios in the published matrix. The exposed gaps are core (41.64 vs 201.97) and long-draft (49.24 vs 242.37).",
      "Do not assume Effort 9 fully solved Gemma dense prefill. It opened the batched path and fixed major correctness bugs, but the current published short/long-draft rows remain about 5x behind llama.cpp.",
      "Do not port Qwen shape constants or Qwen-only flags. Gemma has per-layer attention head-dim variance, post attention/FFN norms, SWA layers, and use_k_as_v behavior that the validator must preserve.",
      "Do not micro-tune decode or LM-head work as the first move. The controller metric is prompt prefill with max_tokens=8.",
    ],
    structuralSwingIdeas: [
      "First prove the current path and buckets. Run Gemma 31B long-draft and core prefill with ZINC_PREFILL_PROFILE=1. Confirm batched prefill is active, record dense_ffn/attention/qkv/o_proj/post_norm/submit overhead, and explain why 70-token long-draft remains only 49.24 tok/s.",
      "Add or reuse a Gemma dense prefill validator that can stop after one layer and compare token-major vs batched tensors. Keep final-logit validation, but add intermediate checks for full-attn layers, SWA layers, post_attention_norm, post_ffw_norm, and use_k_as_v.",
      "Replace per-token-shaped batched DMMV in the hot dense projections with a true tiled Q4_K GEMM/MMQ path over prompt tokens. Priority order should follow profile evidence, likely FFN gate/up/down first, then attention Q/K/V/O.",
      "Attack small-prompt overhead separately from long-context throughput. Core has ~49 prompt tokens and long-draft has ~70, where setup cost can dominate. Context-medium/context-long already win, so tune chunk thresholds and command-buffer construction for short prompt batches instead of only chasing 300+ token throughput.",
      "Audit batched flash attention for Gemma's actual shapes. Full-attn layers use asymmetric Q/KV head dims while SWA layers use smaller symmetric dims. If attention is hot, prefer a shape-specific batched flash-attn path over generic shader cleanups.",
      "Measure every production change against both long-draft and core before keeping it. A change that only helps 192/346-token prompts does not close the public gaps this effort owns.",
      "Once a true GEMM path lands, test Q8_1 activation quant only on the profiled Gemma 31B projection shapes. Treat Q8_1 as a follow-up to structural batching, not a substitute.",
    ],
    referenceImplementations: [
      {
        path: "/Users/zolotukhin/Workplace/zinc/loops/efforts/MULTI_HOUR_EFFORT_9_RDNA_GEMMA_BATCHED_PREFILL.md",
        focus: "Historical Gemma 31B batched prefill work and correctness traps. Read before touching full-attn/SWA/post-norm handling.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/src/compute/forward.zig",
        focus: "Read canUseBatchedPrefillRdna, prefillBatchedImpl, Gemma post norm branches, dispatchProjectionBatched, dispatchFlashAttnBatched, and any Gemma-specific validate paths.",
      },
      {
        path: "/Users/zolotukhin/Workplace/zinc/src/shaders/mul_mm_q4k.comp",
        focus: "Existing tiled Q4_K GEMM reference. It is the right structural direction for prompt-token batches if profiles show DMMV-style weight rereads dominate.",
      },
      {
        path: "/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-vulkan",
        focus: "Vulkan Q4_K matmul/MMQ and Gemma graph lowering. Compare when llama.cpp chooses MMQ/GEMM vs matvec paths on small prompt-token batches.",
      },
    ],
    llamaCppBaselines: [
      { scenario: "core",              promptTokens: 49,  prefillTokPerSec: 201.97, decodeTokPerSec: 28.55 },
      { scenario: "context-medium",    promptTokens: 192, prefillTokPerSec: 50.01,  decodeTokPerSec: 28.19 },
      { scenario: "context-long",      promptTokens: 346, prefillTokPerSec: 46.82,  decodeTokPerSec: 28.09 },
      { scenario: "decode-extended",   promptTokens: 70,  prefillTokPerSec: 242.37, decodeTokPerSec: 28.21, isPrimary: true },
    ],
    llamaCppSuccessRule: "Project success rule (from MULTI_HOUR_EFFORT_19_RDNA_GEMMA31_PREFILL.md): close the Gemma 4 31B RDNA prefill gaps without regressing the context-medium/context-long rows where ZINC already beats llama.cpp. Primary target is decode-extended prefill, where llama.cpp is 242.37 tok/s and ZINC is 49.24 tok/s in the published artifact. Core prefill must also move toward 201.97 tok/s.",
  },
};

export function getEffortSpec(effort: number): EffortSpec | null {
  return EFFORT_SPECS[effort] ?? null;
}

function positiveIntEnv(name: string, fallback: number): number {
  const parsed = Number(process.env[name] ?? fallback);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

function minHealthyTokPerSecForSpec(effortSpec: EffortSpec): number | null {
  const override = process.env.ZINC_MIN_HEALTHY_TPS;
  if (override != null && override.trim() !== "") {
    const parsed = Number(override);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
    if (parsed === 0) return null;
  }
  return effortSpec.minHealthyTokPerSec ?? null;
}

const BENCHMARK_MIN_SAMPLES = 3;
const BENCHMARK_MAX_SAMPLES = Math.max(
  BENCHMARK_MIN_SAMPLES,
  positiveIntEnv("ZINC_BENCH_MAX_SAMPLES", 5),
);
// If the primary metric is noisy, collect a couple of extra samples instead
// of making a keep/revert decision from a lucky or unlucky 3-run median.
const BENCHMARK_EXTRA_SAMPLE_SPREAD_PCT = 0.015;
// Absolute floor on a "material improvement" in tok/s. Previously 0.5,
// which rejected three effort-6 cycles (13/16/21) that produced gains of
// 0.29-0.45 tok/s with sample noise well below the gap. Lowered to 0.2 so
// the relative 1% threshold takes over above ~20 tok/s, which is where
// the loop actually operates.
const MIN_IMPROVEMENT_ABS_TPS = 0.2;
const MIN_IMPROVEMENT_PCT = 0.01;
// Noise-aware override: even when the gain is below the normal threshold,
// accept it if it is large relative to the candidate's sample stdev AND
// above this absolute minimum. Cycle 16 produced samples [28.06, 28.06,
// 28.05] — stdev 0.005, gap 0.30 tok/s = 60× noise, an unambiguous win
// that the old threshold rejected.
// Lowered 0.15 → 0.10 after effort-15 run-2 cycle-2's 103.00 plateau:
// the agent kept landing 102.5-102.9 (below best, so correctly rejected),
// but the tightened floor makes future small-clear wins above best easier
// to accept when the loop is searching out of a saturated neighborhood.
const NOISE_OVERRIDE_ABS_MIN_TPS = 0.1;
const NOISE_OVERRIDE_STDEV_MULTIPLIER = 3;
// How often to refresh the prefill phase budget even without a perf keep.
// Previously we only refreshed after perf keeps; a stalled run would stare
// at a budget from the last perf keep which could be many cycles old.
const PHASE_BUDGET_REFRESH_STALL_THRESHOLD = 3;
const PHASE_BUDGET_STALE_PROMPT_AGE = PHASE_BUDGET_REFRESH_STALL_THRESHOLD;
// Echo-chamber warning: if the last N cycles overwhelmingly target a
// single phase bucket AND no perf keep has come from that bucket in the
// same window, surface a warning in the prompt so the agent considers a
// different bucket.
const ECHO_CHAMBER_WINDOW = 8;
const ECHO_CHAMBER_RATIO = 0.7;
// Correctness-failure streak: when N of the last M cycles failed the
// coherence check (correct=false), surface a warning urging the agent
// to DIAGNOSE the failing path before landing more optimizations. The
// run-2 cycles 11-15 wasted 5 cycles trying new wiring on a path that
// was silently broken by a pre-existing fuse_q8+Q4_K-down crash on
// prompts ≥128 tokens; cycle 16 finally found and fixed it, unlocking
// +5.10%. This detector would have triggered earlier and pointed the
// agent at the buggy path 2-3 cycles sooner.
const CORRECTNESS_STREAK_WINDOW = 6;
const CORRECTNESS_STREAK_THRESHOLD = 3;
const HISTORY_LINES_IN_PROMPT = 20;
const RECENT_CYCLES_IN_PROMPT = 12;
const FAILED_APPROACH_LIMIT = 30;
const IDEA_LIMIT = 24;
const REVIEW_SUMMARY_LIMIT = 6;
const SELF_REVIEW_EVERY = 10;
const STALL_WARNING_THRESHOLD = 4;
const FOUNDATION_KEEP_MAX_DROP_TPS = 0.25;
// After one foundation keep, the next cycle must either swing for a real win
// or pick a different hotspot. We saw cycles 21/22/24 compound into a chain
// of neutral barrier-narrowings that just filled the commit log with noise.
const MAX_FOUNDATION_KEEPS_IN_A_ROW = 1;
const MAX_CHANGED_FILES_IN_PROMPT = 10;
// Every Nth cycle, if the loop is stalled, run a pivot prompt instead of a
// normal cycle. The pivot prompt reviews recent committed foundations,
// identifies dead-end dormant infra, and proposes 3 radically different
// directions. The agent picks one and must measure it in-cycle.
const PIVOT_CYCLE_EVERY = 10;
const PIVOT_STALL_THRESHOLD = 3;
// When stalled this long, inject an explicit pointer to reference
// implementations on disk (llama.cpp, vllm) so the agent can steal
// known-good patterns instead of guessing.
const REFERENCE_IMPLS_STALL_THRESHOLD = 4;

function shouldCleanRemoteBenchmarkNode(): boolean {
  return process.env.ZINC_SKIP_REMOTE_CLEAN !== "1";
}

// Multiple prompts to catch different failure modes:
// - Short factual: catches total corruption
// - Arithmetic: catches subtle numeric drift (wrong MoE routing, bad dequant)
// - Listing: catches mid-sequence divergence (broken RoPE, bad KV cache)
type CoherenceCheck = {
  rawPrompt: string;
  chatPrompt: string;
  expect: string[];
};

const COHERENCE_CHECKS: CoherenceCheck[] = [
  {
    rawPrompt: "The capital of France is",
    chatPrompt: "What is the capital of France? Answer in one word.",
    expect: ["Paris"],
  },
  {
    rawPrompt: "2+2 =",
    chatPrompt: "What is 2+2? Answer using one number.",
    expect: ["4"],
  },
  {
    rawPrompt: "Name the first four planets in order:",
    chatPrompt: "Name the first four planets in order. Answer with only the names separated by commas.",
    expect: ["Mercury", "Venus", "Earth", "Mars"],
  },
];

// All models that must produce coherent output after every change.
// The primary model (--model flag) is benchmarked; these are correctness-only.
const COHERENCE_MODELS: ModelTarget[] = [
  MODELS.qwen36b,
  MODELS.qwen3827b,
  MODELS.qwen8b,
  MODELS.gemma431b,
  MODELS.gemma426ba4b,
];

type CoherenceFailure = {
  id: string;
  label: string;
  model: string;
  prompt: string;
  outputText: string;
  kind: "mismatch" | "crash";
};

type CoherenceSweep = {
  failures: CoherenceFailure[];
  failureIds: string[];
};

type CoherenceCase = {
  modelTarget: ModelTarget;
  check: CoherenceCheck;
  promptMode: PromptMode;
  maxTokens: number;
  prompt: string;
  label: string;
  id: string;
};

const BLOCKED_FILE_OPS = [
  "Edit(loops/*)", "Write(loops/*)", "Edit(site/*)", "Write(site/*)",
  "Edit(docs/*)", "Write(docs/*)", "Edit(.env)", "Write(.env)",
  "Edit(AGENTS.md)", "Write(AGENTS.md)", "Edit(CLAUDE.md)", "Write(CLAUDE.md)",
  "Edit(loops/efforts/MULTI_HOUR_EFFORT_*)", "Write(loops/efforts/MULTI_HOUR_EFFORT_*)",
];

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)", "Bash(git revert:*)", "Bash(git restore:*)",
  "Bash(git reset:*)", "Bash(git stash:*)", "Bash(git clean:*)",
  "Bash(git push:*)", "Bash(git commit:*)",
];

// Paths the agent may change (used for selective revert). Keep this aligned
// with the prompt's "Files you may edit" block; otherwise rejected cycles can
// leak non-src edits, such as build.zig shader-install changes, into later
// baselines.
const REVERTABLE_PATHS = ["build.zig", "src/"];

// Rate-limit backoff knobs. Run-scoped; no persistence across runs.
const RATE_LIMIT_MAX_RETRIES = 3;
const RATE_LIMIT_MAX_WAIT_MS = 6 * 60 * 60 * 1000; // never sleep longer than 6 h
const rateLimitRetriesPerCycle = new Map<number, number>();

function isPrefillMetricLabel(label: string | undefined): boolean {
  return /\bprefill\b/i.test(label ?? "");
}

// -- CLI parsing -------------------------------------------------------------

type AgentType = "claude" | "codex" | "opencode";

function parseArgs() {
  const args = process.argv.slice(2);
  let effort = 0;
  let cycles = 20;
  let dryRun = false;
  let model = "qwen36b";
  let modelExplicit = false;
  let resume = false;
  let agent: AgentType = "codex";
  let analyze = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--effort" && args[i + 1]) effort = parseInt(args[++i], 10);
    else if (args[i] === "--cycles" && args[i + 1]) cycles = parseInt(args[++i], 10);
    else if (args[i] === "--dry-run") dryRun = true;
    else if (args[i] === "--resume") resume = true;
    else if (args[i] === "--analyze") analyze = true;
    else if (args[i] === "--model" && args[i + 1]) {
      model = args[++i];
      modelExplicit = true;
    }
    else if (args[i] === "--agent" && args[i + 1]) agent = args[++i] as AgentType;
  }
  if (!effort || !getEffortSpec(effort)) {
    const effortKeys = Object.keys(EFFORT_SPECS).join("|");
    console.error(`Usage: bun loops/optimize_perf.ts --effort <${effortKeys}> [options]`);
    console.error("");
    console.error("Options:");
    console.error(`  --effort <${effortKeys}>         Optimization to run (required)`);
    console.error("  --cycles N               Max cycles (default: 20)");
    console.error(`  --model NAME             Model: ${MODEL_KEYS} (default: effort-specific, else qwen36b)`);
    console.error("  --agent claude|codex|opencode   AI agent to use (default: codex)");
    console.error("  --resume                 Resume from previous run (read history from log)");
    console.error("  --analyze                Print controller analysis from saved run state");
    console.error("  --dry-run                Build+bench baseline only, skip agent");
    console.error("");
    console.error("Efforts:");
    for (const [id, spec] of Object.entries(EFFORT_SPECS)) {
      console.error(`  ${id} = ${spec.summary}`);
    }
    process.exit(1);
  }
  if (agent !== "claude" && agent !== "codex" && agent !== "opencode") {
    console.error(`Unknown agent: ${agent}. Use 'claude', 'codex', or 'opencode'.`);
    process.exit(1);
  }
  if (!(model in MODELS)) {
    console.error(`Unknown model: ${model}. Use one of: ${MODEL_KEYS}.`);
    process.exit(1);
  }
  return { effort, cycles, dryRun, model, modelExplicit, resume, agent, analyze };
}

// -- Display helpers ---------------------------------------------------------

const CLR = process.stdout.isTTY && !("NO_COLOR" in process.env);
const c = (code: string, t: string) => CLR ? `\x1b[${code}m${t}\x1b[0m` : t;
const SEP = "\u2500".repeat(64);
const BOX_INNER_WIDTH = 58;

function boxLine(text: string): string {
  const content = text.slice(0, BOX_INNER_WIDTH - 1);
  return `\u2551 ${content.padEnd(BOX_INNER_WIDTH - 1)}\u2551`;
}

// -- Command runner with streaming -------------------------------------------

type RunResult = { exitCode: number; signal: NodeJS.Signals | null; stdout: string; stderr: string };

/**
 * Detect other `bun loops/optimize_perf.ts` instances running on this host
 * besides the current process. Returns their PIDs (empty when alone).
 *
 * Why this exists: on 2026-05-30 a `screen -X quit` killed only the screen
 * wrapper while the bun child survived and kept rsyncing/building/committing
 * to the same `main` branch concurrently with the freshly-launched second
 * loop. The two runs interleaved cycle commits, corrupted state, and made
 * per-cycle measurements that weren't reproducible because each loop was
 * benchmarking a tree the other was mutating. main() aborts at startup
 * when a sibling is detected — stop it explicitly before relaunching.
 *
 * `extractOptimizePerfPidsFromPs` is exported separately so the parsing
 * logic is unit-testable without spawning a real `ps`.
 */
export function extractOptimizePerfPidsFromPs(psOutput: string, currentPid: number): number[] {
  const pids: number[] = [];
  for (const line of psOutput.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const m = trimmed.match(/^(\d+)\s+(.*)$/);
    if (!m) continue;
    const pid = parseInt(m[1], 10);
    if (!Number.isFinite(pid) || pid === currentPid) continue;
    const cmd = m[2];
    // Require both: a bun-like leader and the script path. The literal
    // "loops/optimize_perf.ts" filters out the wrapping zsh/login/screen
    // command lines so we don't false-positive on those.
    if (/(^|\/)bun(\s|$)/.test(cmd) && /loops\/optimize_perf\.ts/.test(cmd)) {
      pids.push(pid);
    }
  }
  return pids;
}

export function detectExistingOptimizePerfRuns(currentPid: number = process.pid): number[] {
  try {
    const out = execSync("ps -axo pid=,command=", { encoding: "utf8", timeout: 5000 });
    return extractOptimizePerfPidsFromPs(out, currentPid);
  } catch {
    return [];
  }
}

async function runCommand(
  cmd: string,
  args: string[],
  opts: {
    cwd?: string;
    timeout?: number;
    streamOutput?: boolean;
    stdoutLineFormatter?: (line: string) => string | null;
  } = {},
): Promise<RunResult> {
  const streamOutput = opts.streamOutput ?? false;
  return new Promise((res, rej) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout ?? 120_000,
    });
    let stdout = "", stderr = "", lineBuffer = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (!streamOutput) return;
      if (opts.stdoutLineFormatter) {
        lineBuffer += text;
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const f = opts.stdoutLineFormatter(line);
          if (f !== null) process.stdout.write(f);
        }
      } else {
        process.stdout.write(text);
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (streamOutput) process.stderr.write(text);
    });
    child.on("error", rej);
    child.on("close", (code, signal) => {
      if (streamOutput && opts.stdoutLineFormatter && lineBuffer.trim()) {
        const f = opts.stdoutLineFormatter(lineBuffer);
        if (f !== null) process.stdout.write(f);
      }
      res({ exitCode: code ?? 1, signal, stdout, stderr });
    });
  });
}

// -- SSH & rsync -------------------------------------------------------------

async function ssh(command: string, timeout = 120_000): Promise<string> {
  const { stdout, stderr, exitCode } = await runCommand("ssh", [
    "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
    "-p", String(ZINC_PORT), `${ZINC_USER}@${ZINC_HOST}`, command,
  ], { timeout });
  if (exitCode !== 0 && !stderr.includes("Warning"))
    throw new Error(`SSH failed (${exitCode}): ${stderr.slice(0, 500)}`);
  return stdout.trim();
}

async function rsyncToRemote(): Promise<void> {
  const { exitCode, stderr } = await runCommand("rsync", [
    "-avz", "--checksum", "--delete",
    "-e", `ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no`,
    "--exclude", ".zig-cache", "--exclude", "zig-out", "--exclude", "node_modules",
    "--exclude", ".git", "--exclude", ".perf_optimize", "--exclude", ".zinc_optimize",
    "--exclude", "site", "--exclude", ".DS_Store", "--exclude", ".env", "--exclude", ".env.*",
    "--exclude", "*.swp", "--exclude", "*.swo",
    `${REPO_ROOT}/`, `${ZINC_USER}@${ZINC_HOST}:${REMOTE_DIR}/`,
  ], { timeout: 120_000 });
  if (exitCode !== 0) throw new Error(`rsync failed: ${stderr.slice(0, 300)}`);
}

async function cleanRemoteBenchmarkNode(): Promise<void> {
  if (!shouldCleanRemoteBenchmarkNode()) return;
  console.log(c("2", "  Cleaning stale RDNA benchmark processes..."));
  try {
    await ssh(
      [
        "pkill -f '[z]ig-out/bin/zinc' || true",
        "pkill -f '[l]lama-server' || true",
        "pkill -f '[l]lama-cli' || true",
        "sleep 1",
      ].join("; "),
      30_000,
    );
  } catch (e) {
    console.log(c("1;33", `  Warning: remote cleanup failed; benchmark may be contaminated (${String(e).slice(0, 120)})`));
  }
}

// -- Build & benchmark -------------------------------------------------------

export type BenchResult = {
  buildOk: boolean;
  buildOutput: string;
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  promptTokens?: number | null;
  promptTokenSamples?: number[];
  correct: boolean;
  outputText: string;
  bandwidthUtil: number | null;
  bandwidthSamples: number[];
  error: string | null;
};

export type StepKind = "optimization" | "enablement" | "analysis" | "fix" | "rollback" | "unknown";

export type AgentReport = {
  description: string;
  selfAnalysis: string;
  nextIdeas: string[];
  stepKind: StepKind;
  rawText: string;
};

export type CycleRecord = {
  cycle: number;
  timestamp: string;
  description: string;
  selfAnalysis: string;
  nextIdeas: string[];
  stepKind: StepKind;
  changedFiles: string[];
  categoryTags: string[];
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  promptTokens?: number | null;
  promptTokenSamples?: number[];
  bandwidthUtil: number | null;
  bandwidthSamples: number[];
  correct: boolean;
  improved: boolean;
  broken: boolean;
  kept: boolean;
  foundationKeep: boolean;
  decisionReason: string;
  outputText: string;
  commitHash: string | null;
};

export type BenchCheckpoint = {
  cycle: number;
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  promptTokens?: number | null;
  promptTokenSamples?: number[];
  bandwidthUtil: number | null;
  bandwidthSamples: number[];
  outputText: string;
  commitHash: string | null;
};

export type LoopState = {
  effort: number;
  planDoc: string;
  benchmarkSignature?: string;
  runStartedAt: string;
  lastUpdatedAt: string;
  lastCycle: number;
  bestTokPerSec: number;
  bestCycle: number | null;
  bestCommitHash: string | null;
  bestResult: BenchCheckpoint | null;
  stalledCycles: number;
  consecutiveFoundationKeeps: number;
  cycles: CycleRecord[];
  failedApproaches: string[];
  ideas: string[];
  reviewSummaries: string[];
  // Per-phase prefill GPU profile captured at baseline and refreshed after
  // every accepted keep. Only populated when the effort's metricMode is
  // "prefill" and the runtime was invoked with ZINC_PREFILL_PROFILE=1.
  phaseBudget?: PrefillPhaseBudget | null;
  phaseBudgetCycle?: number | null;
  phaseBudgetRefreshFailures?: number;
  phaseBudgetRefreshLastFailure?: string | null;
  phaseBudgetRefreshLastAttemptCycle?: number | null;
  // Aggregated loop-health metrics recomputed on every save. Cheap to
  // recompute from `cycles` but materialized here so a reader can read the
  // health of a run without knowing the full per-cycle schema.
  runMetrics?: RunMetrics;
};

// ── Metrics ────────────────────────────────────────────────────────────────
//
// These are deliberately not tracked per cycle inline; they are re-derived
// from `cycles[]` + `bestTokPerSec` on every save. That lets a loop change
// (new heuristics, new classifiers) retroactively re-score old runs without
// a migration. The per-cycle fields we compute here are:
//
//   durationMs       — time between the previous cycle's timestamp and this
//                      one's. First cycle is measured from runStartedAt.
//   introducedFlag   — true if this cycle's description/self-analysis added
//                      a ZINC_* env flag (see introducesRuntimeFlag).
//   measuredFlagOn   — true if the self-analysis cites a flag-on tok/s
//                      number in the same cycle.
//   citedReference   — true if the self-analysis references one of the
//                      paths in the effort's referenceImplementations list.
//   attackedBucket   — a rough classification of which top-level phase
//                      bucket the cycle was targeting (attn/moe/ssm/...).
//
// Aggregate metrics roll up from these and answer the "is this loop
// actually improving" question quickly.
export type CycleMetrics = {
  cycle: number;
  durationMs: number;
  introducedFlag: boolean;
  measuredFlagOn: boolean;
  citedReference: boolean;
  attackedBucket: string | null;
  // Information-value score the agent can see in the analyze report:
  //   perf_keep       — moved the primary metric.
  //   measured_dead   — flag-on measurement proved a hypothesis wrong.
  //                     This IS progress even though tok/s didn't move.
  //   dormant_keep    — foundation commit without in-cycle flag-on
  //                     measurement. Suspicious; may turn into dead weight.
  //   broken          — candidate build/coherence broke.
  //   no_op           — no source changes.
  //   revert          — not kept; may or may not have produced a finding.
  informationValue:
    | "perf_keep"
    | "measured_dead"
    | "dormant_keep"
    | "broken"
    | "no_op"
    | "revert";
};

export type RunMetrics = {
  totalCycles: number;
  perfKeeps: number;
  foundationKeeps: number;
  reverts: number;
  brokenCycles: number;
  noOpCycles: number;
  // Time accounting.
  totalCycleMs: number;
  averageCycleMs: number;
  // tok/s per hour of agent time. Positive is forward progress; zero means
  // we are burning agent time without moving the number.
  tpsGainPerHour: number;
  absoluteTpsGain: number;
  // Dormant vs real foundation ratio. Dormant = foundation keep without an
  // in-cycle flag-on measurement. A high dormant count is the specific
  // failure mode we want to catch early (effort-6 run 2 cycles 1/3/5/7/9).
  dormantFoundations: number;
  measuredFoundations: number;
  // Phase-bucket coverage: how many cycles attacked each top-level bucket.
  bucketCoverage: Record<string, number>;
  // External references used: by stall-tier cycles, did the agent cite
  // llama.cpp / vllm when the references were surfaced in the prompt?
  cyclesCitingReferences: number;
  // Diagnostic breakdown: how many cycles produced a measured finding
  // (useful information) vs just churn.
  cyclesProducingInformation: number;
};

export function classifyCycleMetrics(
  cycle: CycleRecord,
  previousTimestamp: string,
  referencePaths: string[],
): CycleMetrics {
  const prev = new Date(previousTimestamp).getTime();
  const now = new Date(cycle.timestamp).getTime();
  const durationMs = Number.isFinite(prev) && Number.isFinite(now) && now > prev ? now - prev : 0;

  const pseudoReport: AgentReport = {
    description: cycle.description ?? "",
    selfAnalysis: cycle.selfAnalysis ?? "",
    nextIdeas: cycle.nextIdeas ?? [],
    stepKind: cycle.stepKind ?? "unknown",
    rawText: `${cycle.description}\n${cycle.selfAnalysis}`,
  };
  const introducedFlag = introducesRuntimeFlag(pseudoReport, cycle.changedFiles ?? []);
  const measuredFlagOn = hasFlagOnMeasurementEvidence(pseudoReport);

  const haystack = (cycle.description + "\n" + (cycle.selfAnalysis ?? "")).toLowerCase();
  const citedReference = referencePaths.some((p) => haystack.includes(p.toLowerCase()))
    || /llama\.cpp|vllm/i.test(haystack);

  let attackedBucket: string | null = null;
  if (/ssm[_ ]?proj|ssm_|\bssm\b/.test(haystack)) attackedBucket = "ssm";
  else if (/\bmoe\b|router|expert|gate_up|gate\/up|swiglu|routed/.test(haystack)) attackedBucket = "moe";
  else if (/\battn|attention|flash[_ ]?attn|\bq[_ ]?proj|\bk[_ ]?proj|\bv[_ ]?proj|rope/.test(haystack)) attackedBucket = "attn";
  else if (/shared[_ ]expert|shared[_ ]proj/.test(haystack)) attackedBucket = "shared";
  else if (/\btail\b|final[_ ]norm|lm[_ ]head|output[_ ]layer/.test(haystack)) attackedBucket = "tail";
  else if (/barrier|submit|command[_ ]buffer/.test(haystack)) attackedBucket = "sync";
  else if (/embed|dequant/.test(haystack)) attackedBucket = "embed";

  let informationValue: CycleMetrics["informationValue"] = "revert";
  if (cycle.broken) informationValue = "broken";
  else if (cycle.improved) informationValue = "perf_keep";
  else if (cycle.foundationKeep) informationValue = "dormant_keep";
  else if ((cycle.changedFiles?.length ?? 0) === 0) {
    // Zero changed files after the agent exits is either a true no-op or
    // a revert-after-measurement. The latter produced information and is
    // valuable; the former is churn. The main loop records stepKind =
    // "rollback" (and decisionReason = "measured-dead…") when it detects
    // the valuable case, so we trust those fields here.
    if (pseudoReport.stepKind === "rollback" || /measured[- ]dead/i.test(cycle.decisionReason ?? "")) {
      informationValue = "measured_dead";
    } else {
      informationValue = "no_op";
    }
  } else if (measuredFlagOn) informationValue = "measured_dead";

  if (cycle.foundationKeep && measuredFlagOn) {
    // Foundation keep that actually measured flag-on is more valuable than a
    // pure dormant commit.
    informationValue = "measured_dead";
  }

  return { cycle: cycle.cycle, durationMs, introducedFlag, measuredFlagOn, citedReference, attackedBucket, informationValue };
}

export function computeRunMetrics(state: LoopState, referencePaths: string[] = []): RunMetrics {
  const metrics: CycleMetrics[] = [];
  let prevTs = state.runStartedAt;
  for (const c of state.cycles) {
    metrics.push(classifyCycleMetrics(c, prevTs, referencePaths));
    prevTs = c.timestamp;
  }

  const totalCycleMs = metrics.reduce((s, m) => s + m.durationMs, 0);
  const perfKeeps = state.cycles.filter((c) => c.improved).length;
  const foundationKeeps = state.cycles.filter((c) => c.foundationKeep).length;
  const reverts = state.cycles.filter((c) => !c.kept && !c.broken && (c.changedFiles?.length ?? 0) > 0).length;
  const brokenCycles = state.cycles.filter((c) => c.broken).length;
  const noOpCycles = state.cycles.filter((c) => (c.changedFiles?.length ?? 0) === 0).length;

  const dormantFoundations = state.cycles.filter((c, i) => c.foundationKeep && !metrics[i].measuredFlagOn).length;
  const measuredFoundations = foundationKeeps - dormantFoundations;

  const bucketCoverage: Record<string, number> = {};
  for (const m of metrics) {
    if (!m.attackedBucket) continue;
    bucketCoverage[m.attackedBucket] = (bucketCoverage[m.attackedBucket] ?? 0) + 1;
  }

  const cyclesCitingReferences = metrics.filter((m) => m.citedReference).length;
  const cyclesProducingInformation = metrics.filter((m) =>
    m.informationValue === "perf_keep" || m.informationValue === "measured_dead"
  ).length;

  const baselineTps = state.cycles[0]?.tokPerSec ?? state.bestResult?.tokPerSec ?? state.bestTokPerSec;
  const absoluteTpsGain = state.bestTokPerSec - (state.bestResult?.cycle === 0 ? state.bestResult.tokPerSec ?? 0 : baselineTps ?? 0);
  const hoursElapsed = totalCycleMs / 1000 / 60 / 60;
  const tpsGainPerHour = hoursElapsed > 0 ? absoluteTpsGain / hoursElapsed : 0;

  return {
    totalCycles: state.cycles.length,
    perfKeeps,
    foundationKeeps,
    reverts,
    brokenCycles,
    noOpCycles,
    totalCycleMs,
    averageCycleMs: state.cycles.length > 0 ? totalCycleMs / state.cycles.length : 0,
    tpsGainPerHour,
    absoluteTpsGain,
    dormantFoundations,
    measuredFoundations,
    bucketCoverage,
    cyclesCitingReferences,
    cyclesProducingInformation,
  };
}

export type PromptContext = {
  cycles: CycleRecord[];
  failedApproaches: string[];
  ideas: string[];
  stalledCycles: number;
  consecutiveFoundationKeeps: number;
  reviewSummary: string | null;
  bestPerf: BenchCheckpoint | null;
  // Latest parsed per-phase prefill profile. Null when unavailable (decode
  // efforts, or when the baseline profile run did not emit phase data).
  phaseBudget?: PrefillPhaseBudget | null;
  phaseBudgetCycle?: number | null;
  phaseBudgetRefreshFailures?: number;
  phaseBudgetRefreshLastFailure?: string | null;
  phaseBudgetRefreshLastAttemptCycle?: number | null;
};

function canonicalizeMemoryEntry(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`"'()[\],.:;!?-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function mergeUniqueEntries(existing: string[], incoming: string[], maxEntries: number): string[] {
  const merged: string[] = [];
  const seen = new Set<string>();
  for (const entry of [...existing, ...incoming]) {
    const trimmed = entry.trim();
    if (!trimmed) continue;
    const key = canonicalizeMemoryEntry(trimmed)
      .split(" ")
      .filter(Boolean)
      .sort()
      .join(" ");
    if (!key || seen.has(key)) continue;
    seen.add(key);
    merged.push(trimmed);
    if (merged.length >= maxEntries) break;
  }
  return merged;
}

function trunc(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function benchResultToCheckpoint(result: BenchResult, cycle: number, commitHash: string | null): BenchCheckpoint {
  return {
    cycle,
    tokPerSec: result.tokPerSec,
    tokPerSecSamples: [...result.tokPerSecSamples],
    promptTokens: result.promptTokens ?? null,
    promptTokenSamples: [...(result.promptTokenSamples ?? [])],
    bandwidthUtil: result.bandwidthUtil,
    bandwidthSamples: [...result.bandwidthSamples],
    outputText: result.outputText,
    commitHash,
  };
}

function checkpointToBenchResult(checkpoint: BenchCheckpoint): BenchResult {
  return {
    buildOk: true,
    buildOutput: "",
    tokPerSec: checkpoint.tokPerSec,
    tokPerSecSamples: [...checkpoint.tokPerSecSamples],
    promptTokens: checkpoint.promptTokens ?? null,
    promptTokenSamples: [...(checkpoint.promptTokenSamples ?? [])],
    correct: true,
    outputText: checkpoint.outputText,
    bandwidthUtil: checkpoint.bandwidthUtil,
    bandwidthSamples: [...checkpoint.bandwidthSamples],
    error: null,
  };
}

export function median(samples: number[]): number | null {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? null;
}

function formatSampleList(samples: number[], digits = 2): string {
  if (samples.length === 0) return "";
  return ` [${samples.map((s) => s.toFixed(digits)).join(", ")}]`;
}

function summarizeBenchMetric(value: number | null, samples: number[], unit: string, digits = 2): string {
  if (value == null) return "unknown";
  return `${value.toFixed(digits)} ${unit}${formatSampleList(samples, digits)}`;
}

function summarizePromptTokens(value: number | null | undefined, samples: number[] | undefined): string {
  if (value == null) return "unknown";
  const sampleList = samples && samples.length > 0
    ? ` [${samples.map((s) => String(s)).join(", ")}]`
    : "";
  return `${value} prompt tokens${sampleList}`;
}

export function formatPhaseBudget(
  budget: PrefillPhaseBudget | null | undefined,
  capturedAtCycle: number | null | undefined,
): string {
  if (!budget) {
    return "- (no phase profile captured yet; baseline will collect one)";
  }
  const ordered = Object.entries(budget.totalsMs)
    .filter(([name]) => name !== "embed")
    .sort((a, b) => b[1] - a[1]);
  const lines: string[] = [];
  const age = capturedAtCycle != null ? ` (captured at cycle ${capturedAtCycle})` : "";
  lines.push(`- Top-level totals (ms)${age}:`);
  for (const [name, value] of ordered) {
    const perTok = budget.perTokenMs[name];
    const perTokStr = perTok != null ? `, ${perTok.toFixed(2)} ms/tok avg` : "";
    lines.push(`  ${name}: ${value.toFixed(1)} ms${perTokStr}`);
  }
  if (Object.keys(budget.moeTotalsMs).length > 0) {
    const moe = Object.entries(budget.moeTotalsMs).sort((a, b) => b[1] - a[1]);
    lines.push(`- MoE sub-buckets (ms): ${moe.map(([n, v]) => `${n}=${v.toFixed(1)}`).join(", ")}`);
  }
  const denseTotals = budget.denseTotalsMs ?? {};
  if (Object.keys(denseTotals).length > 0) {
    const dense = Object.entries(denseTotals).sort((a, b) => b[1] - a[1]);
    lines.push(`- Dense FFN sub-buckets (ms): ${dense.map(([n, v]) => `${n}=${v.toFixed(1)}`).join(", ")}`);
  }
  if (Object.keys(budget.ssmTotalsMs).length > 0) {
    const ssm = Object.entries(budget.ssmTotalsMs).sort((a, b) => b[1] - a[1]);
    lines.push(`- SSM sub-buckets (ms): ${ssm.map(([n, v]) => `${n}=${v.toFixed(1)}`).join(", ")}`);
  }
  if (budget.biggestBucket) {
    lines.push(
      `- Biggest top-level bucket: ${budget.biggestBucket.name} (${budget.biggestBucket.totalMs.toFixed(1)} ms). Target this unless a more specific sub-bucket is clearly larger.`,
    );
  }
  return lines.join("\n");
}

export function formatPhaseBudgetFreshness(
  currentCycle: number,
  capturedAtCycle: number | null | undefined,
  refreshFailures = 0,
  lastFailure: string | null | undefined = null,
  lastAttemptCycle: number | null | undefined = null,
): string | null {
  const age = capturedAtCycle != null ? Math.max(0, currentCycle - capturedAtCycle) : null;
  const stale = capturedAtCycle == null || (age != null && age >= PHASE_BUDGET_STALE_PROMPT_AGE);
  if (!stale && refreshFailures <= 0) return null;

  const lines: string[] = [];
  if (capturedAtCycle == null) {
    lines.push("- No parseable prefill phase profile has been captured for this run.");
  } else if (stale) {
    lines.push(`- STALE: phase profile was captured at cycle ${capturedAtCycle}; current cycle is ${currentCycle} (${age} cycles old).`);
  }

  if (refreshFailures > 0) {
    const attempt = lastAttemptCycle != null ? ` at cycle ${lastAttemptCycle}` : "";
    const count = refreshFailures === 1
      ? "1 refresh attempt has failed"
      : `${refreshFailures} refresh attempts have failed`;
    const reason = lastFailure ? `: ${trunc(lastFailure, 180)}` : ".";
    lines.push(`- ${count}${attempt}${reason}`);
  }

  lines.push(
    "- Action: before another production kernel/dataflow edit, collect a fresh `ZINC_PREFILL_PROFILE=1` benchmark sample or fix the missing profile instrumentation/parser. The next self-analysis must cite current top-level and named sub-buckets if it uses the budget.",
  );
  return lines.join("\n");
}

function formatDominantBucketDirective(budget: PrefillPhaseBudget | null | undefined): string | null {
  const biggest = budget?.biggestBucket;
  if (!biggest) return null;
  const sorted = Object.entries(budget.totalsMs)
    .filter(([name]) => name !== "embed")
    .sort((a, b) => b[1] - a[1]);
  const runnerUp = sorted.find(([name]) => name !== biggest.name);
  const runnerUpText = runnerUp ? `; runner-up is ${runnerUp[0]} at ${runnerUp[1].toFixed(1)} ms` : "";
  const lines = [
    `The current profile's largest top-level bucket is ${biggest.name} at ${biggest.totalMs.toFixed(1)} ms${runnerUpText}.`,
    `A cycle that targets another bucket must cite a fresh profile or a concrete dependency that unlocks ${biggest.name}.`,
  ];
  const namedSubBuckets = collectNamedSubBuckets(budget)
    .sort((a, b) => b.ms - a.ms)
    .slice(0, 6);
  if (namedSubBuckets.length > 0) {
    lines.push(
      `Largest named sub-buckets overall: ${namedSubBuckets.map((b) => `${b.bucket}.${b.name}=${b.ms.toFixed(1)} ms`).join(", ")}.`,
    );
  }
  if (runnerUp && runnerUp[1] >= biggest.totalMs * 0.75) {
    lines.push(
      "The top-level buckets are close; prefer a named sub-bucket with fresh evidence over another broad bucket-level guess.",
    );
  }
  if (biggest.name === "dense_ffn") {
    lines.push("For effort 15, prefer dense layer-major segment work, dense gate/up/SwiGLU structural changes, or dense down+acc fusion. Avoid SSM-only work while dense_ffn remains largest.");
  }
  return lines.map((line) => `- ${line}`).join("\n");
}

function collectNamedSubBuckets(budget: PrefillPhaseBudget): Array<{ bucket: string; name: string; ms: number }> {
  const out: Array<{ bucket: string; name: string; ms: number }> = [];
  for (const [name, ms] of Object.entries(budget.moeTotalsMs)) {
    if (ms > 0) out.push({ bucket: "moe", name, ms });
  }
  for (const [name, ms] of Object.entries(budget.ssmTotalsMs)) {
    if (ms > 0) out.push({ bucket: "ssm", name, ms });
  }
  for (const [name, ms] of Object.entries(budget.denseTotalsMs ?? {})) {
    if (ms > 0) out.push({ bucket: "dense_ffn", name, ms });
  }
  return out;
}

function tailHistory(history: string, maxLines = HISTORY_LINES_IN_PROMPT): string {
  const lines = history.split("\n").map((line) => line.trim()).filter(Boolean);
  return lines.slice(-maxLines).join("\n");
}

function inferStepKind(description: string, selfAnalysis: string): StepKind {
  const haystack = `${description}\n${selfAnalysis}`.toLowerCase();
  if (!haystack.trim()) return "unknown";
  if (/\b(fix|compile|build|correctness|crash|error)\b/.test(haystack)) return "fix";
  if (/\b(rollback|revert|undo|back out|step back)\b/.test(haystack)) return "rollback";
  if (/\b(measure|instrument|benchmark|profile|analy[sz]e|study|inspect)\b/.test(haystack)) return "analysis";
  if (/\b(enablement|plumbing|infrastructure|helper|wrapper|scaffold|layout|pipeline plumbing|descriptor plumbing)\b/.test(haystack)) {
    return "enablement";
  }
  if (/\b(push descriptor|descriptor|dispatch helper|call site conversion|pipeline layout)\b/.test(haystack)) {
    return "enablement";
  }
  return "optimization";
}

export function classifyApproachTags(description: string, changedFiles: string[]): string[] {
  const haystack = `${description}\n${changedFiles.join("\n")}`.toLowerCase();
  const tags: string[] = [];
  if (/\bdmmv\b|dmmv\.zig|matmul|q4_k|q5_k|q6_k|q8_0/.test(haystack)) tags.push("dmmv");
  if (/\b(attention|flash_attn|kv cache|kv_cache|rope)\b|attention\.zig/.test(haystack)) tags.push("attention");
  if (/\b(ssm|delta|conv1d|mamba)\b/.test(haystack)) tags.push("ssm");
  if (/\b(elementwise|swiglu|rms norm|sigmoid|softmax topk|scale acc)\b|elementwise\.zig/.test(haystack)) tags.push("elementwise");
  if (/\b(descriptor|push descriptor|pipeline layout)\b|pipeline\.zig|instance\.zig|command\.zig/.test(haystack)) tags.push("descriptor");
  if (/\b(shader|glsl|\.comp\b)\b|src\/shaders\//.test(haystack)) tags.push("shader");
  if (/\b(buffer|pool|alloc|memory|reuse)\b|buffer\.zig/.test(haystack)) tags.push("memory");
  if (/\b(check|test|correctness|coherence|output)\b/.test(haystack)) tags.push("correctness");
  if (/\b(bench|benchmark|measure|profile|instrument)\b/.test(haystack)) tags.push("measurement");
  if (tags.length === 0) tags.push("other");
  return [...new Set(tags)];
}

function isEnablementLike(report: AgentReport, changedFiles: string[]): boolean {
  if (report.stepKind === "enablement") return true;
  const text = `${report.description}\n${report.selfAnalysis}\n${changedFiles.join("\n")}`.toLowerCase();
  return /\b(enablement|plumbing|infrastructure|helper|wrapper|layout|pipeline|descriptor|call site conversion|scaffold|profile|profiling|instrument|instrumentation|phase budget|timing|timestamp)\b/.test(text);
}

function buildCycleHistoryEntry(cycle: CycleRecord): string {
  const outcome = cycle.improved
    ? "KEPT"
    : cycle.foundationKeep
      ? "KEPT-FOUNDATION"
      : cycle.broken
        ? "REVERTED-BROKEN"
        : "REVERTED";
  const metric = cycle.tokPerSec != null ? ` (${cycle.tokPerSec.toFixed(2)} tok/s)` : "";
  const promptShape = cycle.promptTokens != null ? ` prompt=${cycle.promptTokens}tok` : "";
  const tags = cycle.categoryTags.length > 0 ? ` [${cycle.categoryTags.join(", ")}]` : "";
  return `#${cycle.cycle}: ${outcome}${metric}${promptShape}${tags} ${trunc(cycle.description || cycle.decisionReason, 96)}`;
}

function buildHistoryFromCycles(cycles: CycleRecord[]): string {
  if (cycles.length === 0) return "";
  return cycles.slice(-HISTORY_LINES_IN_PROMPT).map(buildCycleHistoryEntry).join("\n");
}

function buildRecentCycleBlock(cycles: CycleRecord[]): string {
  if (cycles.length === 0) return "  (none yet)";
  return cycles.slice(-RECENT_CYCLES_IN_PROMPT).map((cycle) => `  ${buildCycleHistoryEntry(cycle)}`).join("\n");
}

export function buildSelfReview(state: Pick<LoopState, "cycles" | "stalledCycles" | "consecutiveFoundationKeeps">): string {
  const recent = state.cycles.slice(-SELF_REVIEW_EVERY);
  if (recent.length === 0) return "";

  const improved = recent.filter((cycle) => cycle.improved).length;
  const foundation = recent.filter((cycle) => cycle.foundationKeep).length;
  const broken = recent.filter((cycle) => cycle.broken).length;
  const reverted = recent.filter((cycle) => !cycle.kept).length;
  const tagStats = new Map<string, { kept: number; reverted: number }>();

  for (const cycle of recent) {
    for (const tag of cycle.categoryTags) {
      const entry = tagStats.get(tag) ?? { kept: 0, reverted: 0 };
      if (cycle.kept) entry.kept++;
      else entry.reverted++;
      tagStats.set(tag, entry);
    }
  }

  const deadEnds = [...tagStats.entries()]
    .filter(([, stats]) => stats.reverted > 0 && stats.kept === 0)
    .sort((a, b) => b[1].reverted - a[1].reverted)
    .slice(0, 3)
    .map(([tag, stats]) => `${tag}(${stats.reverted})`);

  const productive = [...tagStats.entries()]
    .filter(([, stats]) => stats.kept > 0)
    .sort((a, b) => (b[1].kept - a[1].kept) || (a[1].reverted - b[1].reverted))
    .slice(0, 3)
    .map(([tag, stats]) => `${tag}(${stats.kept} kept/${stats.reverted} reverted)`);

  const lines = [
    `Last ${recent.length} cycles: ${improved} perf keep, ${foundation} foundation keep, ${reverted} reverted, ${broken} broken.`,
  ];

  if (productive.length > 0) lines.push(`Productive directions: ${productive.join(", ")}.`);
  if (deadEnds.length > 0) lines.push(`Repeated dead ends: ${deadEnds.join(", ")}.`);
  if (state.consecutiveFoundationKeeps > 0) {
    lines.push(`Foundation debt: ${state.consecutiveFoundationKeeps} neutral keep(s) in a row; next cycles should either harvest a speed win or step back.`);
  }
  if (state.stalledCycles >= STALL_WARNING_THRESHOLD) {
    lines.push(`Stall warning: ${state.stalledCycles} cycles without a best-perf win. Stop repeating the last rejected category; pick a different hotspot or a smaller prerequisite.`);
  }

  return lines.join("\n");
}

export function buildAnalysisReport(state: LoopState): string {
  const total = state.cycles.length;
  const improved = state.cycles.filter((cycle) => cycle.improved).length;
  const foundation = state.cycles.filter((cycle) => cycle.foundationKeep).length;
  const broken = state.cycles.filter((cycle) => cycle.broken).length;
  const reverted = state.cycles.filter((cycle) => !cycle.kept).length;

  const tagStats = new Map<string, { kept: number; improved: number; reverted: number }>();
  for (const cycle of state.cycles) {
    for (const tag of cycle.categoryTags) {
      const entry = tagStats.get(tag) ?? { kept: 0, improved: 0, reverted: 0 };
      if (cycle.kept) entry.kept++;
      if (cycle.improved) entry.improved++;
      if (!cycle.kept) entry.reverted++;
      tagStats.set(tag, entry);
    }
  }

  const tagLines = [...tagStats.entries()]
    .sort((a, b) => (b[1].improved - a[1].improved) || (b[1].kept - a[1].kept) || (b[1].reverted - a[1].reverted))
    .slice(0, 8)
    .map(([tag, stats]) => `- ${tag}: ${stats.improved} perf keeps, ${stats.kept} total keeps, ${stats.reverted} reverts`);

  const recent = buildRecentCycleBlock(state.cycles);
  const failed = state.failedApproaches.length > 0
    ? state.failedApproaches.slice(-10).map((entry) => `- ${entry}`).join("\n")
    : "- none";
  const ideas = state.ideas.length > 0
    ? state.ideas.slice(-10).map((entry) => `- ${entry}`).join("\n")
    : "- none";
  const review = state.reviewSummaries.at(-1) ?? buildSelfReview(state);
  const metricLabel = getEffortSpec(state.effort)?.primaryMetricLabel ?? "tok/s";

  const spec = getEffortSpec(state.effort);
  const refPaths = spec?.referenceImplementations?.map((r) => r.path) ?? [];
  const runMetrics = state.runMetrics ?? computeRunMetrics(state, refPaths);
  const bucketLines = Object.entries(runMetrics.bucketCoverage)
    .sort((a, b) => b[1] - a[1])
    .map(([bucket, count]) => `  - ${bucket}: ${count}`);
  const hours = runMetrics.totalCycleMs / 1000 / 60 / 60;
  const dormantShare = runMetrics.foundationKeeps > 0
    ? Math.round((runMetrics.dormantFoundations / runMetrics.foundationKeeps) * 100)
    : 0;

  return [
    `Run started: ${state.runStartedAt}`,
    `Cycles: ${total} total, ${improved} perf keeps, ${foundation} foundation keeps, ${reverted} reverted, ${broken} broken`,
    `Best checkpoint (${metricLabel}): ${state.bestTokPerSec.toFixed(2)} tok/s (cycle ${state.bestCycle ?? "?"}${state.bestCommitHash ? `, ${state.bestCommitHash.slice(0, 8)}` : ""})`,
    `Current stall count: ${state.stalledCycles}`,
    "",
    "Loop health:",
    `  Agent time spent: ${hours.toFixed(2)} h (avg ${(runMetrics.averageCycleMs / 1000 / 60).toFixed(1)} min/cycle)`,
    `  tok/s gain per hour of agent time: ${runMetrics.tpsGainPerHour.toFixed(3)}`,
    `  Cycles producing information (perf_keep + measured_dead): ${runMetrics.cyclesProducingInformation}/${runMetrics.totalCycles}`,
    `  Dormant foundations (flag-gated, no in-cycle flag-on measurement): ${runMetrics.dormantFoundations}/${runMetrics.foundationKeeps} (${dormantShare}%)`,
    `  Cycles citing reference implementations: ${runMetrics.cyclesCitingReferences}`,
    "",
    "Phase bucket coverage:",
    bucketLines.length > 0 ? bucketLines.join("\n") : "  (none yet)",
    "",
    "Recent review:",
    review || "No review yet.",
    "",
    "Category stats:",
    tagLines.length > 0 ? tagLines.join("\n") : "- none",
    "",
    "Recent cycles:",
    recent,
    "",
    "Failed approaches:",
    failed,
    "",
    "Idea bank:",
    ideas,
  ].join("\n");
}

export function improvementThreshold(
  currentTokPerSec: number | null,
  stalledCycles: number = 0,
): number {
  const base = currentTokPerSec == null || currentTokPerSec <= 0
    ? MIN_IMPROVEMENT_ABS_TPS
    : Math.max(MIN_IMPROVEMENT_ABS_TPS, currentTokPerSec * MIN_IMPROVEMENT_PCT);
  // Adaptive plateau-escape: when stalled past the warning threshold,
  // halve the bar (with the absolute floor preserved) so a small but
  // clear positive measurement can break out instead of being rejected
  // by a too-strict 1%-of-best threshold that's tuned for early-game
  // climbs. The noise-aware override remains the primary defense
  // against accepting jitter — this only relaxes the flat bar.
  if (stalledCycles >= STALL_WARNING_THRESHOLD) {
    return Math.max(MIN_IMPROVEMENT_ABS_TPS / 2, base / 2);
  }
  return base;
}

export function sampleStdev(samples: number[]): number {
  if (samples.length < 2) return 0;
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;
  return Math.sqrt(variance);
}

export function relativeSampleSpread(samples: number[]): number {
  const med = median(samples);
  if (med == null || med <= 0 || samples.length < 2) return 0;
  const min = Math.min(...samples);
  const max = Math.max(...samples);
  return (max - min) / med;
}

export function shouldCollectExtraBenchSample(
  samples: number[],
  minSamples = BENCHMARK_MIN_SAMPLES,
  maxSamples = BENCHMARK_MAX_SAMPLES,
): boolean {
  if (samples.length < minSamples) return true;
  if (samples.length >= maxSamples) return false;
  return relativeSampleSpread(samples) >= BENCHMARK_EXTRA_SAMPLE_SPREAD_PCT;
}

/**
 * Noise-aware override: when a candidate's sample dispersion is tight and
 * the gain vs best is a multiple of that noise, the measurement is
 * statistically unambiguous. Accept even when the normal threshold says no.
 *
 * The guard `gain > NOISE_OVERRIDE_ABS_MIN_TPS` prevents this path from
 * accepting micro-jitters when the samples happen to cluster tightly.
 */
export type EchoChamberWarning = {
  bucket: string;
  count: number;
  window: number;
  perfKeepsInBucketWindow: number;
  perfKeepsFromOtherBuckets: number;
};

/**
 * Detect "echo chamber" bucket attacks: a long run of cycles attacking the
 * same phase bucket with no perf keeps from that bucket. Effort-6 run 3
 * had 22/25 cycles targeting SSM even though both perf keeps came from
 * MoE — the agent was drawn to SSM by the stale phase budget and by
 * Sunday-driver momentum. Surfacing this as a prompt warning should
 * nudge diversification when the pattern is obvious.
 *
 * Returns a warning only when the dominant bucket exceeds
 * ECHO_CHAMBER_RATIO of the window AND there is at least one perf keep
 * in the window from a *different* bucket — that is the signal that
 * the loop's wins are coming from elsewhere.
 */
export function detectEchoChamber(cycles: CycleRecord[], referencePaths: string[] = []): EchoChamberWarning | null {
  if (cycles.length < ECHO_CHAMBER_WINDOW) return null;
  const recent = cycles.slice(-ECHO_CHAMBER_WINDOW);
  const counts = new Map<string, number>();
  const perfKeepsByBucket = new Map<string, number>();
  let prevTimestamp = recent[0].timestamp;
  for (const cycle of recent) {
    const m = classifyCycleMetrics(cycle, prevTimestamp, referencePaths);
    prevTimestamp = cycle.timestamp;
    if (!m.attackedBucket) continue;
    counts.set(m.attackedBucket, (counts.get(m.attackedBucket) ?? 0) + 1);
    if (cycle.improved) {
      perfKeepsByBucket.set(m.attackedBucket, (perfKeepsByBucket.get(m.attackedBucket) ?? 0) + 1);
    }
  }
  if (counts.size === 0) return null;
  const [topBucket, topCount] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  if (topCount / recent.length < ECHO_CHAMBER_RATIO) return null;
  const perfKeepsInBucket = perfKeepsByBucket.get(topBucket) ?? 0;
  const totalPerfKeeps = [...perfKeepsByBucket.values()].reduce((a, b) => a + b, 0);
  const perfKeepsFromOtherBuckets = totalPerfKeeps - perfKeepsInBucket;
  if (perfKeepsInBucket > 0) return null; // the bucket is actually paying off; not an echo chamber.
  return {
    bucket: topBucket,
    count: topCount,
    window: recent.length,
    perfKeepsInBucketWindow: perfKeepsInBucket,
    perfKeepsFromOtherBuckets,
  };
}

/**
 * Correctness-failure streak warning: when several recent cycles failed
 * coherence (correct=false), the most likely root cause is a pre-existing
 * latent bug in the path the optimizations target, not bad ideas. Surface
 * a hint to DIAGNOSE the failing path with file overlap from the failed
 * cycles as a starting point.
 */
export type CorrectnessStreakWarning = {
  failedCount: number;
  windowSize: number;
  recentFailedCycles: number[];
  sharedFiles: string[];
};

export function detectCorrectnessStreak(cycles: CycleRecord[]): CorrectnessStreakWarning | null {
  if (cycles.length < CORRECTNESS_STREAK_THRESHOLD) return null;
  const recent = cycles.slice(-CORRECTNESS_STREAK_WINDOW);
  const failed = recent.filter((c) => !c.correct);
  if (failed.length < CORRECTNESS_STREAK_THRESHOLD) return null;
  // Files touched by 2+ failed cycles are the most likely buggy-path hints.
  const fileCounts = new Map<string, number>();
  for (const c of failed) {
    for (const f of c.changedFiles) {
      fileCounts.set(f, (fileCounts.get(f) ?? 0) + 1);
    }
  }
  const sharedFiles = [...fileCounts.entries()]
    .filter(([, count]) => count >= 2)
    .sort((a, b) => b[1] - a[1])
    .map(([f]) => f);
  return {
    failedCount: failed.length,
    windowSize: recent.length,
    recentFailedCycles: failed.map((c) => c.cycle),
    sharedFiles,
  };
}

/**
 * Render the "beat llama.cpp" delta block: shows the controller's current
 * best on its primary metric vs llama.cpp's number on the same metric,
 * the absolute and relative gap to close, plus llama numbers on the other
 * scenarios that the controller is NOT measuring directly. The point is
 * to remind the agent that "best ZINC tok/s" is a means, not the end:
 * the success criterion is beating llama.cpp on each scenario.
 *
 * When `bestTokPerSec` is null/0 (no measurement yet), the gap is shown
 * as "—" rather than guessed.
 */
export function formatLlamaCppComparison(
  baselines: LlamaCppBaseline[],
  primaryMetricLabel: string,
  metricMode: MetricMode,
  bestTokPerSec: number | null,
  successRule?: string,
): string {
  const primary = baselines.find((b) => b.isPrimary) ?? baselines[0];
  if (!primary) return "";
  const llamaPrimary = metricMode === "decode" ? primary.decodeTokPerSec : primary.prefillTokPerSec;
  const ratioStr = bestTokPerSec != null && bestTokPerSec > 0
    ? `${((bestTokPerSec / llamaPrimary) * 100).toFixed(1)}%`
    : "—";
  const gapAbs = bestTokPerSec != null && bestTokPerSec > 0
    ? (llamaPrimary - bestTokPerSec).toFixed(2)
    : "—";
  const gapPct = bestTokPerSec != null && bestTokPerSec > 0
    ? `${(((llamaPrimary - bestTokPerSec) / bestTokPerSec) * 100).toFixed(1)}%`
    : "—";
  const primaryAhead = bestTokPerSec != null && bestTokPerSec >= llamaPrimary;
  const gapLine = bestTokPerSec == null || bestTokPerSec <= 0
    ? "  gap to beat:     —"
    : primaryAhead
      ? `  margin over llama.cpp: +${(bestTokPerSec - llamaPrimary).toFixed(2)} tok/s   (+${(((bestTokPerSec - llamaPrimary) / llamaPrimary) * 100).toFixed(1)}% vs llama.cpp)`
      : `  gap to beat:     +${gapAbs} tok/s   (+${gapPct} on current best)`;
  const tier =
    primaryAhead ? "BEATING llama.cpp ✓" :
    bestTokPerSec != null && bestTokPerSec / llamaPrimary >= 0.9 ? "within striking distance (≥90%)" :
    bestTokPerSec != null && bestTokPerSec / llamaPrimary >= 0.7 ? "closing the gap (70-90%)" :
    "structural gap remains (<70%)";
  const otherLines = baselines
    .filter((b) => b !== primary)
    .map((b) => `  - ${b.scenario.padEnd(18)}${b.promptTokens != null ? ` prompt: ${String(b.promptTokens).padStart(3)} tok   ` : " "}${`prefill: ${b.prefillTokPerSec.toFixed(2)} tok/s`.padEnd(27)} decode: ${b.decodeTokPerSec.toFixed(2)} tok/s`)
    .join("\n");
  return [
    `Primary metric (${primaryMetricLabel}):`,
    primary.promptTokens != null ? `  scenario:        ${primary.scenario} (${primary.promptTokens} prompt tokens)` : `  scenario:        ${primary.scenario}`,
    `  ZINC best:       ${bestTokPerSec != null ? bestTokPerSec.toFixed(2) : "—"} tok/s`,
    `  llama.cpp:       ${llamaPrimary.toFixed(2)} tok/s`,
    `  ratio:           ${ratioStr}   (${tier})`,
    gapLine,
    primaryAhead && baselines.length > 1
      ? `  next target:     primary is already ahead; protect/measure the other public scenarios before another primary-only micro-tune.`
      : null,
    "",
    `Other scenarios the loop is NOT measuring per-cycle (also count toward "beat llama.cpp on all 4"):`,
    otherLines,
    "",
    successRule ?? `Project success rule (from MULTI_HOUR_EFFORT_15_RDNA_QWEN36_27B_PREFILL_DECODE.md): beat llama.cpp on at least 3 of 4 decode scenarios AND keep prefill ≤ 20% behind on every context scenario. A change that improves the controller's prompt but regresses an other scenario does not count. If you are within ~10% of llama.cpp on the primary, prefer the structural lever (validated layer-major batched SSM+dense prefill, Tracks 1-3 in the effort doc) over more micro-fusion — micro-fusion compounds slower than structural batching at this scale.`,
  ].filter((line) => line != null).join("\n");
}

export function formatCorrectnessStreakWarning(w: CorrectnessStreakWarning): string {
  const filesHint = w.sharedFiles.length
    ? ` Files touched by multiple failed cycles (most likely buggy path): ${w.sharedFiles.slice(0, 5).join(", ")}.`
    : "";
  return `Correctness regression streak: ${w.failedCount}/${w.windowSize} recent cycles (cycles ${w.recentFailedCycles.join(", ")}) failed the coherence check. Before landing another optimization, DIAGNOSE the failing path — several "rejected" cycles in a row are usually not bad ideas but a pre-existing latent crash or correctness regression in the path the optimizations target (edge cases: prompt length, batch size, layer index, segment boundary, fuse interaction). Run the failing prompt locally, narrow to the specific layer/operator, and propose a defensive fix. A correct fix that unblocks the failing class can deliver wins the original cycles could not measure — this is exactly how effort-15 run-2 cycle 16 (+5.10% from 103 to 108.25) emerged after run-2 cycles 11-15 had been silently blocked by the fuse_q8+Q4_K-down crash on prompts ≥128 tokens.${filesHint}`;
}

export function formatEchoChamberWarning(warning: EchoChamberWarning): string {
  const other = warning.perfKeepsFromOtherBuckets > 0
    ? ` ${warning.perfKeepsFromOtherBuckets} perf keep(s) in this window came from a *different* bucket.`
    : "";
  return `Echo chamber detected: ${warning.count}/${warning.window} recent cycles have attacked the "${warning.bucket}" bucket with zero perf keeps from it.${other} Pick a different top-level bucket this cycle unless you have new evidence that the "${warning.bucket}" bucket just became approachable.`;
}

export function passesNoiseAwareOverride(candidate: BenchResult, currentBest: BenchResult): boolean {
  if (candidate.tokPerSec == null || currentBest.tokPerSec == null) return false;
  const gain = candidate.tokPerSec - currentBest.tokPerSec;
  if (gain <= NOISE_OVERRIDE_ABS_MIN_TPS) return false;
  const stdev = sampleStdev(candidate.tokPerSecSamples);
  if (stdev === 0) {
    // All samples identical — if the gain is above the minimum, accept.
    return true;
  }
  return gain >= NOISE_OVERRIDE_STDEV_MULTIPLIER * stdev;
}

/**
 * Pivot cycles fire every PIVOT_CYCLE_EVERY cycles when the loop has been
 * stalled for at least PIVOT_STALL_THRESHOLD cycles. The goal is to force
 * a review of committed foundations and a deliberate pivot instead of
 * another speculative optimization on top of a pile of dormant wiring.
 *
 * Returns true when cycle N > 0, (N % PIVOT_CYCLE_EVERY) === 0, and the
 * controller is stalled. If the controller is actively making progress
 * (stalled below threshold) we skip the pivot — no need to second-guess
 * a working direction.
 */
export function shouldRunPivotCycle(cycleNum: number, context: PromptContext | null): boolean {
  if (cycleNum <= 0 || cycleNum % PIVOT_CYCLE_EVERY !== 0) return false;
  if (!context) return false;
  return context.stalledCycles >= PIVOT_STALL_THRESHOLD;
}

export function isMaterialImprovement(
  candidate: BenchResult,
  currentBest: BenchResult,
  stalledCycles: number = 0,
): boolean {
  if (candidate.tokPerSec == null) return false;
  const threshold = improvementThreshold(currentBest.tokPerSec, stalledCycles);
  const current = currentBest.tokPerSec ?? 0;
  if (candidate.tokPerSec > current + threshold) return true;
  // Below the normal threshold — fall back to the noise-aware override so
  // tight-variance gains aren't thrown away.
  return passesNoiseAwareOverride(candidate, currentBest);
}

export function buildAgentPrompt(
  plan: string,
  originalBaseline: BenchResult,
  currentBest: BenchResult,
  cycleNum: number,
  history: string,
  model: string,
  context: PromptContext | null = null,
  options: {
    primaryMetricLabel?: string;
    benchmarkMethod?: string;
    knownFlatCategories?: string[];
    structuralSwingIdeas?: string[];
    referenceImplementations?: Array<{ path: string; focus: string }>;
    llamaCppBaselines?: LlamaCppBaseline[];
    llamaCppSuccessRule?: string;
    metricMode?: MetricMode;
    mode?: "normal" | "pivot";
  } = {},
): string {
  if (options.mode === "pivot") {
    return buildPivotPrompt(plan, originalBaseline, currentBest, cycleNum, model, context, options);
  }
  const modelTarget = MODELS[model] ?? MODELS.qwen36b;
  const sanityCheckPrompt = coherencePromptForMode(
    COHERENCE_CHECKS[0],
    coherencePromptModeForModel(modelTarget),
  );
  const primaryMetricLabel = options.primaryMetricLabel ?? "decode tok/s";
  const benchmarkMethod = options.benchmarkMethod ?? "200-token decode benchmark on the primary model";
  const historySummary = tailHistory(history);
  const failedBlock = context?.failedApproaches?.length
    ? context.failedApproaches.slice(-12).map((entry, i) => `${i + 1}. ${trunc(entry, 140)}`).join("\n")
    : "None yet.";
  const ideasBlock = context?.ideas?.length
    ? context.ideas.slice(-10).map((entry, i) => `${i + 1}. ${trunc(entry, 140)}`).join("\n")
    : "None yet.";
  const recentCyclesBlock = context ? buildRecentCycleBlock(context.cycles) : "  (state unavailable)";
  const reviewBlock = context?.reviewSummary || "No self-review yet.";
  const bestPerf = context?.bestPerf ?? benchResultToCheckpoint(currentBest, 0, null);
  const currentVsBestNote = bestPerf.tokPerSec != null && currentBest.tokPerSec != null && bestPerf.tokPerSec > currentBest.tokPerSec + 0.05
    ? `- Note: the current checked-out code is ${currentBest.tokPerSec.toFixed(2)} tok/s, below the best checkpoint ${bestPerf.tokPerSec.toFixed(2)} tok/s. You are editing the current code, but real wins are still judged against the best checkpoint.`
    : "- Note: current code and best checkpoint are effectively the same right now.";
  const controllerMode = context && context.stalledCycles >= STALL_WARNING_THRESHOLD
    ? "STEP_BACK"
    : context && context.consecutiveFoundationKeeps > 0
      ? "HARVEST"
      : "ADVANCE";

  const phaseBudgetBlock = isPrefillMetricLabel(primaryMetricLabel)
    ? formatPhaseBudget(context?.phaseBudget ?? null, context?.phaseBudgetCycle ?? null)
    : null;
  const phaseBudgetFreshnessBlock = isPrefillMetricLabel(primaryMetricLabel)
    ? formatPhaseBudgetFreshness(
        cycleNum,
        context?.phaseBudgetCycle ?? null,
        context?.phaseBudgetRefreshFailures ?? 0,
        context?.phaseBudgetRefreshLastFailure ?? null,
        context?.phaseBudgetRefreshLastAttemptCycle ?? null,
      )
    : null;
  const dominantBucketDirective = isPrefillMetricLabel(primaryMetricLabel)
    ? formatDominantBucketDirective(context?.phaseBudget ?? null)
    : null;

  const echoWarning = context
    ? detectEchoChamber(context.cycles, options.referenceImplementations?.map((r) => r.path) ?? [])
    : null;
  const echoBlock = echoWarning ? formatEchoChamberWarning(echoWarning) : null;

  const correctnessStreak = context ? detectCorrectnessStreak(context.cycles) : null;
  const correctnessStreakBlock = correctnessStreak
    ? formatCorrectnessStreakWarning(correctnessStreak)
    : null;

  const llamaCppBlock = options.llamaCppBaselines && options.llamaCppBaselines.length > 0
    ? formatLlamaCppComparison(
        options.llamaCppBaselines,
        options.primaryMetricLabel ?? "primary metric",
        options.metricMode ?? "prefill",
        bestPerf.tokPerSec,
        options.llamaCppSuccessRule,
      )
    : null;

  const knownFlatBlock = options.knownFlatCategories?.length
    ? options.knownFlatCategories.map((entry, i) => `${i + 1}. ${entry}`).join("\n")
    : null;

  const swingIdeasBlock = options.structuralSwingIdeas?.length
    ? options.structuralSwingIdeas.map((entry, i) => `${i + 1}. ${entry}`).join("\n")
    : null;

  const stalled = context && context.stalledCycles >= STALL_WARNING_THRESHOLD;
  const hasBankedFoundation = context && context.consecutiveFoundationKeeps > 0;
  const mustSwing = stalled || hasBankedFoundation;

  // Reference implementations only surface once we're stalled. Before that,
  // the agent should work from the plan and the in-tree code; surfacing
  // external references earlier creates prompt noise and encourages
  // reflexive "look at llama.cpp" cycles when the local plan is still
  // doing its job.
  const showReferences =
    options.referenceImplementations?.length &&
    context &&
    context.stalledCycles >= REFERENCE_IMPLS_STALL_THRESHOLD;
  const referencesBlock = showReferences
    ? options.referenceImplementations!
        .map((r, i) => `${i + 1}. ${r.path} — ${r.focus}`)
        .join("\n")
    : null;

  const taskDirective = mustSwing && swingIdeasBlock
    ? `STRUCTURAL SWING REQUIRED. The controller is in ${controllerMode} mode (stall=${context?.stalledCycles ?? 0}, banked foundations=${context?.consecutiveFoundationKeeps ?? 0}). You MUST pick ONE idea from the Structural Swing Ideas block below (or an equally-concrete alternative that attacks a named top-level bucket from the Phase Budget above), not another cosmetic micro-optimization. Cycles that come back with another barrier-narrowing / cosmetic variation will be rejected as a repeat dead end.`
    : "Implement ONE concrete step from the optimization plan above. Pick the next unfinished step.";

  return `You are implementing a performance optimization for the ZINC Vulkan inference engine.

## Optimization Plan
${plan}

## Benchmark Focus
- primary metric: ${primaryMetricLabel}
- benchmark method: ${benchmarkMethod}
- measured prompt shape: ${summarizePromptTokens(currentBest.promptTokens, currentBest.promptTokenSamples)}
- success is judged on the primary metric above, not on one lucky decode sample from a different workload.
- prompt-token thresholds are part of the workload contract. If a change helps the measured shape but is likely to miss or regress nearby public prompt lengths, call that out and prefer the general fix.

## Current Checked-Out Code (build on this code)
- primary metric (${primaryMetricLabel}): ${summarizeBenchMetric(currentBest.tokPerSec, currentBest.tokPerSecSamples, "tok/s")}
- prompt tokens: ${summarizePromptTokens(currentBest.promptTokens, currentBest.promptTokenSamples)}
- bandwidth utilization: ${summarizeBenchMetric(currentBest.bandwidthUtil, currentBest.bandwidthSamples, "%", 1)}
- output: "${currentBest.outputText}" (coherence tested with ${COHERENCE_CHECKS.length} prompts on ${COHERENCE_MODELS.length} models after every change)
- This is the performance of the code currently checked out in the worktree.

## Best Accepted Performance Checkpoint
- primary metric (${primaryMetricLabel}): ${summarizeBenchMetric(bestPerf.tokPerSec, bestPerf.tokPerSecSamples, "tok/s")}
- prompt tokens: ${summarizePromptTokens(bestPerf.promptTokens, bestPerf.promptTokenSamples)}
- bandwidth utilization: ${summarizeBenchMetric(bestPerf.bandwidthUtil, bestPerf.bandwidthSamples, "%", 1)}
- output: "${bestPerf.outputText}"
- cycle: ${bestPerf.cycle}${bestPerf.commitHash ? `, commit ${bestPerf.commitHash.slice(0, 8)}` : ""}
${currentVsBestNote}

## Original Run Baseline (for total gain only)
- primary metric (${primaryMetricLabel}): ${summarizeBenchMetric(originalBaseline.tokPerSec, originalBaseline.tokPerSecSamples, "tok/s")}
- prompt tokens: ${summarizePromptTokens(originalBaseline.promptTokens, originalBaseline.promptTokenSamples)}
- bandwidth utilization: ${summarizeBenchMetric(originalBaseline.bandwidthUtil, originalBaseline.bandwidthSamples, "%", 1)}
- output: "${originalBaseline.outputText}"
${llamaCppBlock ? `\n## llama.cpp Comparison (the real success target)\n${llamaCppBlock}\n` : ""}${phaseBudgetBlock ? `\n## Current Prefill Phase Budget (ZINC_PREFILL_PROFILE=1)\n${phaseBudgetBlock}\nUse this budget to pick the biggest remaining bucket. Do not propose batching/kernel work for a bucket whose total is clearly smaller than another untried bucket.\n` : ""}${phaseBudgetFreshnessBlock ? `\n## Phase Budget Freshness\n${phaseBudgetFreshnessBlock}\n` : ""}${dominantBucketDirective ? `\n## Dominant Bucket Directive\n${dominantBucketDirective}\n` : ""}${echoBlock ? `\n## ⚠ Echo Chamber Warning\n${echoBlock}\n` : ""}${correctnessStreakBlock ? `\n## ⚠ Correctness Regression Streak\n${correctnessStreakBlock}\n` : ""}${knownFlatBlock ? `\n## Known Flat Territory on This Target (do not re-attempt without new evidence)\n${knownFlatBlock}\n` : ""}${swingIdeasBlock ? `\n## Structural Swing Ideas (pick one when controller wants a swing)\n${swingIdeasBlock}\n` : ""}${referencesBlock ? `\n## Reference Implementations on Disk (read when stuck)\n${referencesBlock}\n\nThese are full checkouts of production inference engines. Skim the specific files named above; do not copy wholesale, but steal the architectural patterns (pipeline specialization constants, kernel selection thresholds, MoE routing shapes). If a reference makes an idea obvious, say so in your self-analysis so the next cycle knows the pattern came from a proven codebase.\n` : ""}
## Controller State
- mode: ${controllerMode}
- stalled cycles without a new best checkpoint: ${context?.stalledCycles ?? 0}
- consecutive neutral foundation keeps: ${context?.consecutiveFoundationKeeps ?? 0}
- structural swing required this cycle: ${mustSwing ? "YES" : "no"}

## Recent Cycle Ledger
${recentCyclesBlock}

## Reflection (auto-analysis of recent cycles)
${reviewBlock}

## Previous Attempts
${historySummary || "None yet."}

## Failed Approaches (do not repeat)
${failedBlock}

## Idea Bank
${ideasBlock}

## Your Task (Cycle ${cycleNum})
${taskDirective}
Your change must beat the best accepted performance checkpoint above, not the original run baseline.
If controller mode is STEP_BACK, do not repeat the same hotspot as the last rejected cycles. Either choose a smaller prerequisite, finish a kept enablement step, or switch to a different bottleneck category — but it MUST still be a concrete structural step, not a cosmetic variation of a known-flat pattern.
If you intentionally do a plumbing/enabling step that may be performance-neutral this cycle, mark it as enablement and explain exactly which next step it unlocks and which top-level phase bucket it will eventually attack.

**Flag-gated changes must be measured in the same cycle.** If your change introduces a new runtime env flag (ZINC_*), you MUST run the benchmark both with the flag OFF and with it ON, cite both tok/s numbers in your SELF_ANALYSIS, and make an explicit keep/revert decision. Dormant flag-gated infrastructure that is only validated in a later cycle has cost us ~5 committed foundation cycles; the loop now rejects flag-gated foundation keeps that lack a flag-on measurement.

**Agent-side measurement budget.** The controller will sync, build, run the 3-sample primary benchmark, and run coherence after you return. Do not start tools/performance_suite.mjs from inside the agent, and do not run remote ./zig-out/bin/zinc with -n > 32 unless this exact cycle is an analysis/shaderstats cycle and you cite the bounded reason. Manual checks should be limited to shader compile/build plus one short smoke or one paired flag-off/flag-on target sample; long suites and repeated -n 96 / -n 160 diagnostics waste RDNA time and make controller decisions stale.

Do not use sub-agents, delegation, spawn_agent, or wait_agent. Work directly in this repo.
Before editing any file, re-read the exact current contents from disk. Do not rely on stale context, guessed line numbers, or cached snippets.

## CRITICAL RULES — READ CAREFULLY

1. **BUILD MUST PASS.** Before you declare yourself done, you MUST:
   a. rsync your changes to the remote node
   b. Compile shaders: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR}/src/shaders && for f in *.comp; do glslc --target-env=vulkan1.3 -fshader-stage=compute \\$f -o \\$\{f%.comp}.spv 2>&1; done"
   c. Build: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast 2>&1"
   d. If the build fails, FIX THE ERRORS before finishing. Do NOT leave broken code.

2. **Incremental steps.** If the optimization requires changing many call sites (e.g. 60+ descriptor set conversions), break it into compilable stages:
   - Add new infrastructure (new functions, new fields) FIRST — the old code can coexist.
   - Convert call sites in batches, building after each batch to catch errors.
   - Remove old infrastructure LAST.
   - The code MUST compile at every stage.

3. **ONE focused change per cycle.** Don't try to convert the entire codebase in one shot.

4. **Avoid repeated dead ends.**
   - Read the recent cycle ledger and failed approaches first.
   - If the last few rejected cycles hit the same subsystem, do NOT do another cosmetic variation of that same idea.
   - If you are uncertain, add a tiny enabling or measurement step instead of another large speculative refactor.

5. **Test on remote node:**
   rsync -avz --checksum --delete -e "ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no" ${AGENT_RSYNC_EXCLUDE_ARGS} ${REPO_ROOT}/ ${ZINC_USER}@${ZINC_HOST}:${REMOTE_DIR}/
   ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast && ${REMOTE_ZINC_ENV} ./zig-out/bin/zinc ${zincCliArgs(modelTarget, sanityCheckPrompt, 16)}"

6. **Shader compilation:** glslc --target-env=vulkan1.3 -fshader-stage=compute file.comp -o file.spv

Files you may edit:
- src/compute/*.zig (forward.zig, dmmv.zig, elementwise.zig, attention.zig, argmax.zig)
- src/vulkan/*.zig (pipeline.zig, command.zig, buffer.zig, instance.zig)
- src/model/*.zig (tokenizer.zig, loader.zig, config.zig, architecture.zig)
- src/server/*.zig (routes.zig, runtime.zig)
- src/server/chat.html
- src/shaders/*.comp (GLSL compute shaders)
- src/main.zig
- build.zig (only when a shader/pipeline change needs the build to install a new artifact)

## Output Format
After making your change, print these lines:
@@@DESCRIPTION: <one-line summary of the change>
@@@STEP_KIND: <optimization|enablement|analysis|fix|rollback>
@@@SELF_ANALYSIS: <why this direction, expected effect, and what should happen next>
@@@NEXT_IDEAS: <semicolon-separated follow-up ideas>`;
}

/**
 * Pivot cycle prompt. Fires every PIVOT_CYCLE_EVERY cycles when the loop has
 * been stalled for PIVOT_STALL_THRESHOLD+ cycles. Goals:
 *   1. Force the agent to review committed foundations and identify dead-end
 *      dormant infrastructure.
 *   2. Allow and encourage reverting dormant commits that have been disproved
 *      by later measurement cycles.
 *   3. Push the agent to propose 3 radically different directions (drawing
 *      from reference implementations if available) and pick one that can
 *      be MEASURED IN THIS SAME CYCLE, not deferred.
 */
export function buildPivotPrompt(
  plan: string,
  originalBaseline: BenchResult,
  currentBest: BenchResult,
  cycleNum: number,
  model: string,
  context: PromptContext | null,
  options: {
    primaryMetricLabel?: string;
    benchmarkMethod?: string;
    knownFlatCategories?: string[];
    structuralSwingIdeas?: string[];
    referenceImplementations?: Array<{ path: string; focus: string }>;
    llamaCppBaselines?: LlamaCppBaseline[];
    llamaCppSuccessRule?: string;
    metricMode?: MetricMode;
  },
): string {
  const modelTarget = MODELS[model] ?? MODELS.qwen36b;
  const sanityCheckPrompt = coherencePromptForMode(
    COHERENCE_CHECKS[0],
    coherencePromptModeForModel(modelTarget),
  );
  const primaryMetricLabel = options.primaryMetricLabel ?? "decode tok/s";
  const benchmarkMethod = options.benchmarkMethod ?? "primary benchmark";
  const recentCyclesBlock = context ? buildRecentCycleBlock(context.cycles) : "  (state unavailable)";
  const committedFoundations = context
    ? context.cycles
        .filter((c) => c.kept && (c.foundationKeep || c.improved))
        .slice(-10)
        .map((c) => {
          const hash = c.commitHash ? c.commitHash.slice(0, 8) : "?";
          const kind = c.improved ? "PERF" : "FOUND";
          return `  - cycle ${c.cycle} [${kind} ${hash}] ${trunc(c.description, 110)}`;
        })
        .join("\n") || "  (none)"
    : "  (state unavailable)";
  const phaseBudgetBlock = isPrefillMetricLabel(primaryMetricLabel)
    ? formatPhaseBudget(context?.phaseBudget ?? null, context?.phaseBudgetCycle ?? null)
    : null;
  const phaseBudgetFreshnessBlock = isPrefillMetricLabel(primaryMetricLabel)
    ? formatPhaseBudgetFreshness(
        cycleNum,
        context?.phaseBudgetCycle ?? null,
        context?.phaseBudgetRefreshFailures ?? 0,
        context?.phaseBudgetRefreshLastFailure ?? null,
        context?.phaseBudgetRefreshLastAttemptCycle ?? null,
      )
    : null;
  const dominantBucketDirective = isPrefillMetricLabel(primaryMetricLabel)
    ? formatDominantBucketDirective(context?.phaseBudget ?? null)
    : null;
  const knownFlatBlock = options.knownFlatCategories?.length
    ? options.knownFlatCategories.map((entry, i) => `${i + 1}. ${entry}`).join("\n")
    : null;
  const swingIdeasBlock = options.structuralSwingIdeas?.length
    ? options.structuralSwingIdeas.map((entry, i) => `${i + 1}. ${entry}`).join("\n")
    : null;
  const referencesBlock = options.referenceImplementations?.length
    ? options.referenceImplementations
        .map((r, i) => `${i + 1}. ${r.path} — ${r.focus}`)
        .join("\n")
    : null;
  const llamaCppBlock = options.llamaCppBaselines && options.llamaCppBaselines.length > 0
    ? formatLlamaCppComparison(
        options.llamaCppBaselines,
        primaryMetricLabel,
        options.metricMode ?? "prefill",
        currentBest.tokPerSec,
        options.llamaCppSuccessRule,
      )
    : null;

  return `You are in a PIVOT cycle for the ZINC Vulkan inference engine. The loop has been stalled — recent cycles are not moving the primary metric. Before another speculative change, stop and review.

## Optimization Plan
${plan}

## Benchmark Focus
- primary metric: ${primaryMetricLabel}
- benchmark method: ${benchmarkMethod}
- measured prompt shape: ${summarizePromptTokens(currentBest.promptTokens, currentBest.promptTokenSamples)}
- prompt-token thresholds are part of the workload contract; do not pick a pivot that only helps one exact boundary unless it also explains the nearby public prompt lengths.

## Current Best Checkpoint
- ${summarizeBenchMetric(currentBest.tokPerSec, currentBest.tokPerSecSamples, "tok/s")}
- prompt tokens: ${summarizePromptTokens(currentBest.promptTokens, currentBest.promptTokenSamples)}
- stalled for ${context?.stalledCycles ?? 0} cycles
- consecutive neutral foundation keeps: ${context?.consecutiveFoundationKeeps ?? 0}
${llamaCppBlock ? `\n## llama.cpp Comparison (the real success target)\n${llamaCppBlock}\n` : ""}${phaseBudgetBlock ? `\n## Current Prefill Phase Budget\n${phaseBudgetBlock}\n` : ""}${phaseBudgetFreshnessBlock ? `\n## Phase Budget Freshness\n${phaseBudgetFreshnessBlock}\n` : ""}${dominantBucketDirective ? `\n## Dominant Bucket Directive\n${dominantBucketDirective}\n` : ""}
## Committed Foundations From Recent Cycles
${committedFoundations}

## Recent Cycle Ledger
${recentCyclesBlock}
${knownFlatBlock ? `\n## Known Flat Territory (do not re-attempt without new evidence)\n${knownFlatBlock}\n` : ""}${swingIdeasBlock ? `\n## Candidate Directions\n${swingIdeasBlock}\n` : ""}${referencesBlock ? `\n## Reference Implementations on Disk\n${referencesBlock}\n` : ""}
## Your Task (Pivot Cycle ${cycleNum})
This cycle is different from a normal optimization cycle. Do exactly the following in order:

1. **Dead-end audit.** Read the Committed Foundations list above. For each entry, decide: (a) is this wiring actually being used, (b) has a later cycle measured it as net-negative or non-useful, (c) should it be reverted to clean up tech debt? If you identify dead-end commits, prepare a revert of the dead code. Reverting confirmed dead-end foundations IS valid progress for this cycle.

2. **Pivot proposal.** Propose THREE radically different directions the loop has not meaningfully attempted. Each must:
   - Attack a specific named top-level phase bucket from the budget above.
   - Not be a variation of anything in the Known Flat list.
   - Cite either a plan-document step or a specific pattern from a reference implementation when applicable.
   - Have a concrete measurement strategy that fits in ONE cycle.

3. **Pick one and execute.** Choose the most promising of your three proposals. Implement it. Measure. If it regresses, revert in this same cycle and record the finding. If it is flag-gated, measure both flag-off and flag-on in this cycle (dormant wiring is not acceptable). Produce a concrete tok/s number, not a hand-wave.

Agent-side measurement budget: the controller will run the official sync/build/benchmark/coherence gate after you return. Do not launch tools/performance_suite.mjs from inside the agent. Do not run remote ./zig-out/bin/zinc with -n > 32 unless the pivot you picked is explicitly a bounded analysis/shaderstats cycle; if you do, stop after the one planned diagnostic and cite the evidence. Long -n 96 / -n 160 suites belong outside the cycle agent.

Your output must still end with @@@DESCRIPTION / @@@STEP_KIND / @@@SELF_ANALYSIS / @@@NEXT_IDEAS. Valid STEP_KIND values for a pivot cycle include:
- rollback (if you reverted dead-end foundations)
- analysis (if the pivot is measurement/diagnosis only and produced a concrete finding)
- optimization (if your pivot produced a real tok/s improvement)
- enablement (only if you measured flag-on in this same cycle)

## Test on Remote Node
rsync -avz --checksum --delete -e "ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no" ${AGENT_RSYNC_EXCLUDE_ARGS} ${REPO_ROOT}/ ${ZINC_USER}@${ZINC_HOST}:${REMOTE_DIR}/
ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast && ${REMOTE_ZINC_ENV} ./zig-out/bin/zinc ${zincCliArgs(modelTarget, sanityCheckPrompt, 16)}"

Files you may edit: same as a normal cycle (src/compute/*.zig, src/vulkan/*.zig, src/model/*.zig, src/server/*.zig, src/server/chat.html, src/shaders/*.comp, src/main.zig, and build.zig only when a shader/pipeline change needs build installation). You may also remove files that a revert would remove.

## Output Format
@@@DESCRIPTION: <one-line summary of the pivot action you took>
@@@STEP_KIND: <rollback|analysis|optimization|enablement>
@@@SELF_ANALYSIS: <the three pivot proposals, which you picked and why, and the measured outcome>
@@@NEXT_IDEAS: <semicolon-separated follow-up ideas seeded by what you learned>`;
}

function metricParserForSpec(spec: EffortSpec): (output: string) => number | null {
  return spec.metricMode === "prefill" ? parsePrefillTokPerSec : parseTokPerSec;
}

function coherencePromptForMode(check: CoherenceCheck, promptMode: PromptMode): string {
  return promptMode === "chat" ? check.chatPrompt : check.rawPrompt;
}

function coherencePromptModeForModel(modelTarget: ModelTarget): PromptMode {
  return modelTarget.coherencePromptMode ?? modelTarget.promptMode;
}

function coherenceMaxTokensForModel(modelTarget: ModelTarget): number {
  return modelTarget.coherenceMaxTokens ?? 30;
}

export function zincCliArgs(modelTarget: Pick<ModelTarget, "path" | "promptMode" | "systemPrompt">, prompt: string, maxTokens: number, promptMode = modelTarget.promptMode): string {
  const chatFlag = promptMode === "chat" ? " --chat" : "";
  const systemPromptFlag = promptMode === "chat" && modelTarget.systemPrompt
    ? ` --system-prompt ${shellQuote(modelTarget.systemPrompt)}`
    : "";
  return `-m ${shellQuote(modelTarget.path)} -d ${REMOTE_VULKAN_DEVICE_INDEX}${chatFlag}${systemPromptFlag} --prompt ${shellQuote(prompt)} -n ${maxTokens}`;
}

function zincRemoteCommand(modelTarget: ModelTarget, prompt: string, maxTokens: number, promptMode = modelTarget.promptMode): string {
  return `cd ${REMOTE_DIR} && ${REMOTE_ZINC_ENV} ./zig-out/bin/zinc ${zincCliArgs(modelTarget, prompt, maxTokens, promptMode)} 2>&1`;
}

function zincRemoteCommandProfiled(modelTarget: ModelTarget, prompt: string, maxTokens: number, promptMode = modelTarget.promptMode): string {
  return `cd ${REMOTE_DIR} && ${REMOTE_ZINC_ENV} ZINC_PREFILL_PROFILE=1 ./zig-out/bin/zinc ${zincCliArgs(modelTarget, prompt, maxTokens, promptMode)} 2>&1`;
}

type PhaseBudgetCollection = {
  budget: PrefillPhaseBudget | null;
  failureReason: string | null;
};

/**
 * Run one ZINC_PREFILL_PROFILE=1 sample and return the parsed phase budget.
 * Only call this after buildAndBench has already confirmed the build is green
 * and the output is correct. Profiling adds per-token timestamp overhead
 * (~3%) so we don't include it in the main median calculation.
 */
async function collectPhaseBudget(modelTarget: ModelTarget, effortSpec: EffortSpec): Promise<PhaseBudgetCollection> {
  if (effortSpec.metricMode !== "prefill") {
    return { budget: null, failureReason: "phase budget collection is only available for prefill efforts" };
  }
  try {
    const output = await ssh(
      zincRemoteCommandProfiled(modelTarget, effortSpec.benchmarkPrompt, effortSpec.benchmarkMaxTokens),
      300_000,
    );
    const budget = parsePrefillPhaseBudget(output);
    if (budget) return { budget, failureReason: null };
    const hasPrefill = /Prefill complete|prefill/i.test(output);
    return {
      budget: null,
      failureReason: hasPrefill
        ? "profile run completed but emitted no parseable prefill phase budget"
        : "profile run completed without recognizable prefill output",
    };
  } catch (e) {
    return {
      budget: null,
      failureReason: `profile run failed: ${trunc(String(e).replace(/\s+/g, " "), 180)}`,
    };
  }
}

function applyPhaseBudgetCollection(
  state: LoopState,
  collection: PhaseBudgetCollection,
  cycle: number,
  successPrefix: string,
): void {
  state.phaseBudgetRefreshLastAttemptCycle = cycle;
  if (collection.budget) {
    state.phaseBudget = collection.budget;
    state.phaseBudgetCycle = cycle;
    state.phaseBudgetRefreshFailures = 0;
    state.phaseBudgetRefreshLastFailure = null;
    if (collection.budget.biggestBucket) {
      console.log(c(
        "1;36",
        `  ${successPrefix}: ${collection.budget.biggestBucket.name} (${collection.budget.biggestBucket.totalMs.toFixed(1)} ms)`,
      ));
    }
    return;
  }

  state.phaseBudgetRefreshFailures = (state.phaseBudgetRefreshFailures ?? 0) + 1;
  state.phaseBudgetRefreshLastFailure = collection.failureReason ?? "unknown phase budget refresh failure";
  console.log(c(
    "1;33",
    `  Phase budget refresh failed (${state.phaseBudgetRefreshFailures} consecutive): ${trunc(state.phaseBudgetRefreshLastFailure, 180)}`,
  ));
}

async function buildAndBench(modelTarget: ModelTarget, effortSpec: EffortSpec): Promise<BenchResult> {
  // RX 9070 XT (rdna2) in dynamic power mode ramps sclk from 0 MHz on every
  // short prefill burst, producing 25-30% sample-to-sample noise (160 vs 210
  // tok/s on identical code) that buries real improvements under the median.
  // Pin the card to high-performance so the benchmark is stable. Gated to
  // rdna2 — RDNA1 is managed by the zinc_rt_autopilot loop and Mesa 25.0.7
  // there does not show this ramp-on-burst behaviour. Disable with
  // ZINC_FORCE_GPU_HIGH=0. Idempotent + best-effort, so a node reboot or a
  // driver reset mid-loop no longer silently reintroduces the noise.
  if (SELECTED_RDNA_NODE === "rdna2" && process.env.ZINC_FORCE_GPU_HIGH !== "0") {
    try {
      await ssh(
        `for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do echo high > "$f" 2>/dev/null || true; done`,
        15_000,
      );
    } catch {
      // best-effort; the benchmark can proceed regardless
    }
  }
  console.log(c("2", "  Compiling shaders..."));
  try {
    await ssh(`cd ${REMOTE_DIR} && rm -rf zig-out/share/zinc/shaders`, 30_000);
    await ssh(`cd ${REMOTE_DIR}/src/shaders && for f in *.comp; do glslc --target-env=vulkan1.3 -fshader-stage=compute $f -o \${f%.comp}.spv 2>&1; done`, 120_000);
  } catch (e) {
    return {
      buildOk: false,
      buildOutput: String(e),
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText: "",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: "shader compile failed",
    };
  }

  console.log(c("2", "  Building..."));
  let buildOutput: string;
  try {
    buildOutput = await ssh(`cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast 2>&1`, 600_000);
  } catch (e) {
    return {
      buildOk: false,
      buildOutput: String(e),
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText: "",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: "zig build failed",
    };
  }
  if (buildOutput.includes("error:")) {
    return {
      buildOk: false,
      buildOutput,
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText: "",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: "build errors",
    };
  }

  // Shader-install parity guard. `zig build` only installs the shaders listed
  // in build.zig's `shader_sources` tuple into share/zinc/shaders, but the
  // runtime loads its .spv from that install dir. A new src/shaders/*.comp that
  // is wired into Zig but forgotten in shader_sources gets compiled (above) yet
  // never installed, so the engine silently falls back to an older kernel and
  // the benchmark measures the wrong code. That is exactly how effort-15 logged
  // a 79.63 tok/s "win" that did not survive a clean build. Fail loud here so a
  // forgotten shader is an obvious build failure, not a stale-measurement trap.
  let parityOutput: string;
  try {
    parityOutput = await ssh(
      `cd ${REMOTE_DIR} && for f in src/shaders/*.comp; do b=$(basename "$f" .comp); ` +
        `test -f "zig-out/share/zinc/shaders/$b.spv" || echo "$b.comp"; done`,
      30_000,
    );
  } catch (e) {
    parityOutput = String(e);
  }
  const missingShaders = parityOutput.split("\n").map((s) => s.trim()).filter(Boolean);
  if (missingShaders.length > 0) {
    return {
      buildOk: false,
      buildOutput:
        `Shader-install parity check failed: ${missingShaders.length} shader(s) compiled in ` +
        `src/shaders but NOT installed by build.zig. Add each to the shader_sources tuple in ` +
        `build.zig (or delete the unused .comp), otherwise the runtime falls back to older ` +
        `kernels and the benchmark measures the wrong code:\n  ${missingShaders.join("\n  ")}`,
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText: "",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: "shader install parity mismatch",
    };
  }

  // Quick correctness check (short prompt, few tokens)
  console.log(c("2", "  Running correctness test..."));
  let correctnessOutput: string;
  const firstCheck = COHERENCE_CHECKS[0];
  const correctnessPromptMode = coherencePromptModeForModel(modelTarget);
  const correctnessPrompt = coherencePromptForMode(firstCheck, correctnessPromptMode);
  const correctnessMaxTokens = coherenceMaxTokensForModel(modelTarget);
  try {
    correctnessOutput = await ssh(
      zincRemoteCommand(modelTarget, correctnessPrompt, correctnessMaxTokens, correctnessPromptMode),
      180_000,
    );
  } catch (e) {
    return {
      buildOk: true,
      buildOutput,
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText: "",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: `run failed: ${e}`,
    };
  }

  const textMatch = correctnessOutput.match(/Output text:\s*(.+)/i);
  const outputText = textMatch ? textMatch[1].trim() : "";
  const correct = firstCheck.expect.every(e => outputText.toLowerCase().includes(e.toLowerCase()));

  if (!correct) {
    return {
      buildOk: true,
      buildOutput,
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText,
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: "incorrect output",
    };
  }

  const parseMetric = metricParserForSpec(effortSpec);
  const samplePlan = BENCHMARK_MAX_SAMPLES > BENCHMARK_MIN_SAMPLES
    ? `${BENCHMARK_MIN_SAMPLES}-${BENCHMARK_MAX_SAMPLES}`
    : `${BENCHMARK_MIN_SAMPLES}`;
  console.log(c(
    "2",
    `  Benchmarking (${samplePlan} x ${effortSpec.benchmarkMethod}, primary metric: ${effortSpec.primaryMetricLabel})...`,
  ));
  const tokPerSecSamples: number[] = [];
  const promptTokenSamples: number[] = [];
  const bandwidthSamples: number[] = [];
  for (let sample = 0; sample < BENCHMARK_MAX_SAMPLES; sample++) {
    if (sample >= BENCHMARK_MIN_SAMPLES && !shouldCollectExtraBenchSample(tokPerSecSamples)) {
      break;
    }
    let benchOutput: string;
    try {
      benchOutput = await ssh(
        zincRemoteCommand(modelTarget, effortSpec.benchmarkPrompt, effortSpec.benchmarkMaxTokens),
        300_000,
      );
    } catch (e) {
      return {
        buildOk: true,
        buildOutput,
        tokPerSec: null,
        tokPerSecSamples,
        correct: true,
        outputText,
        bandwidthUtil: null,
        bandwidthSamples,
        error: `bench failed: ${e}`,
      };
    }

    const tps = parseMetric(benchOutput);
    const promptTokens = parsePrefillTokenCount(benchOutput);
    const bw = effortSpec.metricMode === "decode" ? parseBandwidthUtil(benchOutput) : null;
    if (tps != null) tokPerSecSamples.push(tps);
    if (promptTokens != null) promptTokenSamples.push(promptTokens);
    if (bw != null) bandwidthSamples.push(bw);
    const sampleLabel = sample < BENCHMARK_MIN_SAMPLES
      ? `${sample + 1}/${BENCHMARK_MIN_SAMPLES}`
      : `extra ${sample + 1}/${BENCHMARK_MAX_SAMPLES}`;
    console.log(c(
      "2",
      `    sample ${sampleLabel}: ${tps?.toFixed(2) ?? "?"} tok/s (${effortSpec.primaryMetricLabel})${promptTokens != null ? `, prompt ${promptTokens} tok` : ""}${bw != null ? `, BW ${bw.toFixed(1)}%` : ""}`,
    ));
  }

  const tokPerSec = median(tokPerSecSamples);
  const promptTokens = median(promptTokenSamples);
  const bandwidthUtil = median(bandwidthSamples);

  return {
    buildOk: true,
    buildOutput,
    tokPerSec,
    tokPerSecSamples,
    promptTokens,
    promptTokenSamples,
    correct,
    outputText,
    bandwidthUtil,
    bandwidthSamples,
    error: tokPerSec == null ? `${effortSpec.primaryMetricLabel} parse failed` : null,
  };
}

/// Run ALL coherence prompts on ALL models.
/// Returns the full sweep so the controller can enforce non-regression against
/// the accepted baseline instead of demanding global cross-model cleanliness.
function coherenceCaseId(model: string, prompt: string): string {
  return `${model}::${prompt}`;
}

function coherenceCaseLabel(model: string, prompt: string): string {
  return `${model} [${prompt.slice(0, 25)}]`;
}

function formatCoherenceFailure(failure: CoherenceFailure): string {
  if (failure.kind === "crash") {
    const detail = failure.outputText.trim().replace(/\s+/g, " ");
    return detail
      ? `${failure.label}: crashed (${trunc(detail, 90)})`
      : `${failure.label}: crashed`;
  }
  return `${failure.label}: "${failure.outputText.slice(0, 50)}"`;
}

export function formatCoherenceFailureList(failures: CoherenceFailure[]): string {
  return failures.map((failure) => formatCoherenceFailure(failure)).join("; ");
}

export function summarizeCoherenceRegression(
  candidate: CoherenceSweep,
  acceptedFailureIds: string[],
): string | null {
  const accepted = new Set(acceptedFailureIds);
  const regressions = candidate.failures.filter((failure) => !accepted.has(failure.id));
  if (regressions.length === 0) return null;
  return `New coherence failures vs accepted baseline: ${formatCoherenceFailureList(regressions)}`;
}

async function runCoherenceCase(testCase: CoherenceCase, timeoutMs: number): Promise<CoherenceFailure | null> {
  try {
    const out = await ssh(
      zincRemoteCommand(testCase.modelTarget, testCase.prompt, testCase.maxTokens, testCase.promptMode),
      timeoutMs,
    );
    const textMatch = out.match(/Output text:\s*(.+)/i);
    const outputText = textMatch ? textMatch[1].trim() : "";
    const pass = testCase.check.expect.every(e => outputText.toLowerCase().includes(e.toLowerCase()));
    if (pass) return null;
    return {
      id: testCase.id,
      label: testCase.label,
      model: testCase.modelTarget.name,
      prompt: testCase.prompt,
      outputText,
      kind: "mismatch",
    };
  } catch (e) {
    return {
      id: testCase.id,
      label: testCase.label,
      model: testCase.modelTarget.name,
      prompt: testCase.prompt,
      outputText: String(e).slice(-500),
      kind: "crash",
    };
  }
}

async function runCoherenceSweep(): Promise<CoherenceSweep> {
  const cases: CoherenceCase[] = [];
  for (const modelTarget of COHERENCE_MODELS) {
    const promptMode = coherencePromptModeForModel(modelTarget);
    const maxTokens = coherenceMaxTokensForModel(modelTarget);
    for (const check of COHERENCE_CHECKS) {
      const prompt = coherencePromptForMode(check, promptMode);
      cases.push({
        modelTarget,
        check,
        promptMode,
        maxTokens,
        prompt,
        label: coherenceCaseLabel(modelTarget.name, prompt),
        id: coherenceCaseId(modelTarget.name, prompt),
      });
    }
  }

  let failures: CoherenceFailure[] = [];
  for (const testCase of cases) {
    const failure = await runCoherenceCase(testCase, 180_000);
    if (failure) failures.push(failure);
  }

  const crashedIds = new Set(failures.filter((failure) => failure.kind === "crash").map((failure) => failure.id));
  if (crashedIds.size > 0) {
    console.log(c("1;33", `  Coherence saw ${crashedIds.size} crash/timeout case(s); cleaning RDNA node and retrying crashed cases once...`));
    await cleanRemoteBenchmarkNode();
    const stableFailures = failures.filter((failure) => failure.kind !== "crash");
    const retriedFailures: CoherenceFailure[] = [];
    for (const testCase of cases) {
      if (!crashedIds.has(testCase.id)) continue;
      const failure = await runCoherenceCase(testCase, 240_000);
      if (failure) retriedFailures.push(failure);
    }
    failures = [...stableFailures, ...retriedFailures];
  }

  for (const modelTarget of COHERENCE_MODELS) {
    if (!failures.some((failure) => failure.model === modelTarget.name)) {
      console.log(c("2", `    ${modelTarget.name}: all ${COHERENCE_CHECKS.length} prompts OK`));
    } else {
      const crashCount = failures.filter((failure) => failure.model === modelTarget.name && failure.kind === "crash").length;
      if (crashCount > 0) {
        console.log(c("1;33", `    ${modelTarget.name}: ${crashCount} crash/timeout case(s) after retry`));
      }
    }
  }
  return {
    failures,
    failureIds: failures.map((failure) => failure.id),
  };
}

// -- Codex stream formatter --------------------------------------------------

export function formatCodexStreamLine(rawLine: string): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return null; }

  const type = event.type as string | undefined;

  if (type === "item.started" || type === "item.completed") {
    const item = event.item as Record<string, unknown> | undefined;
    const itemType = item?.type as string | undefined;
    if (itemType === "agent_message") {
      const text = item?.text ?? item?.message ?? item?.output_text ?? item?.content;
      if (typeof text === "string" && text.trim()) {
        return c("96", text.trim()) + "\n";
      }
      return null;
    }
    if (itemType === "command_execution") {
      const command = item?.command as string | undefined;
      if (type === "item.started" && command) {
        return `\n${c("33", "\uD83D\uDD27 shell")}${c("2", `   $ ${command.length > 120 ? command.slice(0, 120) + "\u2026" : command}`)}\n`;
      }
      const exitCode = item?.exit_code;
      if (type === "item.completed" && typeof exitCode === "number" && exitCode !== 0) {
        return c("1;31", `  command exited with code ${exitCode}\n`);
      }
      return null;
    }
    if (itemType === "reasoning") {
      return c("2", "  \u2026 thinking\n");
    }
    return null;
  }

  // Agent message with text
  if (type === "message" || type === "agent") {
    const content = (event.content ?? event.message) as string | undefined;
    if (content && typeof content === "string" && content.trim()) {
      return c("96", content.trim()) + "\n";
    }
    return null;
  }

  // Tool/function call
  if (type === "function_call" || type === "tool_use" || type === "action") {
    const name = (event.name ?? event.tool ?? event.function) as string | undefined;
    const cmdOrInput = (event.command ?? event.input ?? event.arguments) as string | Record<string, unknown> | undefined;
    if (name === "shell" || name === "bash" || name === "terminal") {
      const cmd = typeof cmdOrInput === "string" ? cmdOrInput : (cmdOrInput as Record<string, unknown>)?.command as string ?? "";
      return `\n${c("33", "\uD83D\uDD27 shell")}${c("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "\u2026" : cmd}`)}\n`;
    }
    if (name === "write" || name === "create_file" || name === "patch" || name === "apply_diff") {
      const fp = typeof cmdOrInput === "string" ? cmdOrInput : (cmdOrInput as Record<string, unknown>)?.path as string ?? (cmdOrInput as Record<string, unknown>)?.file_path as string ?? "";
      const short = fp.split("/").slice(-3).join("/");
      return `\n${c("33", `\uD83D\uDD27 ${name}`)}${c("2", ` \u2192 ${short}`)}\n`;
    }
    if (name === "read" || name === "read_file") {
      const fp = typeof cmdOrInput === "string" ? cmdOrInput : (cmdOrInput as Record<string, unknown>)?.path as string ?? (cmdOrInput as Record<string, unknown>)?.file_path as string ?? "";
      const short = fp.split("/").slice(-3).join("/");
      return `${c("33", `\uD83D\uDD27 ${name}`)}${c("2", ` \u2192 ${short}`)}\n`;
    }
    if (name) {
      return `\n${c("33", `\uD83D\uDD27 ${name}`)}\n`;
    }
    return null;
  }

  // Function call output / result — skip (too verbose)
  if (type === "function_call_output" || type === "action_output" || type === "tool_result") {
    return null;
  }

  // Thinking / reasoning — show brief indicator
  if (type === "thinking" || type === "reasoning") {
    return c("2", "  \u2026 thinking\n");
  }

  return null;
}

// -- Claude stream formatter -------------------------------------------------

export type ClaudeStreamState = {
  currentToolName: string | null;
  currentBlockIsToolUse: boolean;
  inputJsonBuffer: string;
  inTextBlock: boolean;
  sawTextDeltaInCurrentMessage: boolean;
};

export function formatToolInput(name: string, rawJson: string): string {
  let input: Record<string, unknown> = {};
  try { input = JSON.parse(rawJson) as Record<string, unknown>; } catch { /* empty */ }

  const out: string[] = [];
  const shortPath = (fp: string) => fp.split("/").slice(-3).join("/");
  if (name === "edit") {
    out.push(c("2", ` \u2192 ${shortPath((input.file_path as string) ?? "?")}`));
  } else if (name === "write") {
    const lineCount = ((input.content as string) ?? "").split("\n").length;
    out.push(c("2", ` \u2192 ${shortPath((input.file_path as string) ?? "?")} (${lineCount} lines)`));
  } else if (name === "bash") {
    const cmd = (input.command as string) ?? "?";
    out.push(c("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "\u2026" : cmd}`));
  } else if (name === "read") {
    out.push(c("2", ` \u2192 ${shortPath((input.file_path as string) ?? "?")}`));
  } else if (name === "grep") {
    out.push(c("2", ` \u2192 /${(input.pattern as string) ?? "?"}/`));
  } else if (name === "glob") {
    out.push(c("2", ` \u2192 ${(input.pattern as string) ?? "?"}`));
  }
  return out.length > 0 ? out.join("\n") + "\n" : "";
}

export function formatClaudeStreamLine(rawLine: string, state: ClaudeStreamState): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return rawLine + "\n"; }

  if (event.type === "stream_event") {
    const e = event.event as Record<string, unknown> | undefined;
    if (!e) return null;
    if (e.type === "content_block_start") {
      const block = e.content_block as Record<string, unknown> | undefined;
      if (block?.type === "tool_use") {
        state.currentToolName = (block.name as string) ?? "tool";
        state.currentBlockIsToolUse = true;
        state.inputJsonBuffer = "";
        state.inTextBlock = false;
        return `\n${c("33", `\uD83D\uDD27 ${state.currentToolName}`)}`;
      }
      if (block?.type === "text") {
        state.inTextBlock = true;
        state.currentBlockIsToolUse = false;
        return CLR ? "\n\x1b[96m" : "\n";
      }
      state.inTextBlock = false;
      state.currentBlockIsToolUse = false;
      return null;
    }
    if (e.type === "content_block_delta") {
      const delta = e.delta as Record<string, unknown> | undefined;
      if (delta?.type === "input_json_delta") {
        state.inputJsonBuffer += (delta.partial_json as string) ?? "";
        return null;
      }
      if (delta?.type === "text_delta" && state.inTextBlock) {
        state.sawTextDeltaInCurrentMessage = true;
        return delta.text as string;
      }
      return null;
    }
    if (e.type === "content_block_stop") {
      if (state.currentBlockIsToolUse) {
        state.currentBlockIsToolUse = false;
        const detail = formatToolInput(state.currentToolName ?? "", state.inputJsonBuffer);
        state.inputJsonBuffer = "";
        return detail || null;
      }
      if (state.inTextBlock) {
        state.inTextBlock = false;
        return CLR ? "\x1b[0m\n" : "\n";
      }
      return null;
    }
    return null;
  }
  if (event.type === "assistant") {
    const msg = event.message as Record<string, unknown> | undefined;
    if (!msg) return null;
    const content = msg.content;
    if (Array.isArray(content)) {
      const parts: string[] = [];
      for (const block of content) {
        const b = block as Record<string, unknown>;
        if (b?.type === "text" && typeof b.text === "string" && b.text.trim())
          parts.push(b.text);
      }
      const text = parts.join("\n");
      if (!text.trim()) return null;
      if (state.sawTextDeltaInCurrentMessage) {
        state.sawTextDeltaInCurrentMessage = false;
        return null;
      }
      return c("96", text) + "\n";
    }
    return null;
  }
  return null;
}

function extractAgentText(stdout: string): string {
  const texts: string[] = [];
  for (const line of stdout.split("\n")) {
    if (!line.trim()) continue;
    try {
      const evt = JSON.parse(line) as Record<string, unknown>;
      const type = evt.type;
      if (type === "assistant") {
        const content = (evt.message as Record<string, unknown> | undefined)?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            const text = (block as Record<string, unknown>)?.text;
            if (typeof text === "string" && text.trim()) texts.push(text);
          }
        }
      } else if (type === "message" || type === "agent") {
        const text = evt.content ?? evt.message;
        if (typeof text === "string" && text.trim()) texts.push(text);
      } else if (type === "item.completed") {
        const item = evt.item as Record<string, unknown> | undefined;
        if (item?.type === "agent_message") {
          const text = item.text ?? item.message ?? item.output_text ?? item.content;
          if (typeof text === "string" && text.trim()) texts.push(text);
        }
      }
    } catch {
      if (line.trim().startsWith("@@@")) texts.push(line.trim());
    }
  }
  return texts.join("\n");
}

export function parseAgentReport(stdout: string): AgentReport {
  const rawText = extractAgentText(stdout).trim();
  const window = rawText.slice(-4000);
  const description = window.match(/@@@DESCRIPTION:\s*(.+)/im)?.[1]?.trim()
    ?? rawText.split("\n").map((line) => line.trim()).find(Boolean)
    ?? "Agent made changes";
  const selfAnalysis = window.match(/@@@SELF_ANALYSIS:\s*(.+)/im)?.[1]?.trim() ?? "";
  const stepKindRaw = window.match(/@@@STEP_KIND:\s*(.+)/im)?.[1]?.trim().toLowerCase() ?? "";
  const stepKind = ["optimization", "enablement", "analysis", "fix", "rollback"].includes(stepKindRaw)
    ? stepKindRaw as StepKind
    : inferStepKind(description, selfAnalysis);
  const ideasRaw = window.match(/@@@NEXT_IDEAS:\s*(.+)/im)?.[1]?.trim() ?? "";
  const nextIdeas = ideasRaw
    .split(/[;,]/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 3);

  return {
    description,
    selfAnalysis,
    nextIdeas,
    stepKind,
    rawText,
  };
}

async function listChangedFiles(): Promise<string[]> {
  const tracked = await runCommand("git", ["diff", "--name-only", "--", ...REVERTABLE_PATHS], { cwd: REPO_ROOT });
  const untrackedArgs = REVERTABLE_PATHS.filter((path) => path.endsWith("/"));
  const untracked = untrackedArgs.length > 0
    ? await runCommand("git", ["ls-files", "--others", "--exclude-standard", ...untrackedArgs], { cwd: REPO_ROOT })
    : { stdout: "" };
  const files = [
    ...tracked.stdout.split("\n"),
    ...untracked.stdout.split("\n"),
  ].map((entry) => entry.trim()).filter(Boolean);
  return [...new Set(files)].sort();
}

// -- Agent spawn -------------------------------------------------------------

async function spawnAgent(
  _effortDoc: string,
  plan: string,
  originalBaseline: BenchResult,
  currentBest: BenchResult,
  cycleNum: number,
  history: string,
  model: string,
  agent: AgentType = "claude",
  context: PromptContext | null = null,
  effortSpec: EffortSpec | null = null,
): Promise<RunResult> {
  const isPivot = shouldRunPivotCycle(cycleNum, context);
  const prompt = buildAgentPrompt(plan, originalBaseline, currentBest, cycleNum, history, model, context, {
    primaryMetricLabel: effortSpec?.primaryMetricLabel,
    benchmarkMethod: effortSpec?.benchmarkMethod,
    knownFlatCategories: effortSpec?.knownFlatCategories,
    structuralSwingIdeas: effortSpec?.structuralSwingIdeas,
    referenceImplementations: effortSpec?.referenceImplementations,
    llamaCppBaselines: effortSpec?.llamaCppBaselines,
    llamaCppSuccessRule: effortSpec?.llamaCppSuccessRule,
    metricMode: effortSpec?.metricMode,
    mode: isPivot ? "pivot" : "normal",
  });
  if (isPivot) {
    console.log(c("1;35", `  \uD83D\uDD04 PIVOT cycle — stalled ${context?.stalledCycles ?? 0}, reviewing foundations and picking a radically different direction`));
  }

  console.log(c("1;34", SEP));
  console.log(c("1;34", `  \uD83E\uDDE0 Agent cycle ${cycleNum} (${agent})`));
  console.log(c("1;34", SEP));

  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(
      c("2", `\n\u23F3 still running (${formatElapsed(startedAt)} elapsed)...\n`),
    );
  }, 30_000);

  let result: RunResult;

  if (agent === "codex") {
    // Codex: uses `codex exec` with bypass sandbox (needs SSH/rsync to RDNA node)
    result = await runCommand("codex", codexExecArgs(prompt), {
      cwd: REPO_ROOT,
      timeout: 7_200_000,
      streamOutput: true,
      stdoutLineFormatter: (line) => formatCodexStreamLine(line),
    });
  } else if (agent === "opencode") {
    // opencode: headless `opencode run` (default GLM-5.2). Same SSH/rsync
    // contract as codex — the agent edits code locally, the loop syncs to the
    // RDNA node and benchmarks.
    result = await runCommand("opencode", opencodeExecArgs(prompt), {
      cwd: REPO_ROOT,
      timeout: 7_200_000,
      streamOutput: true,
    });
  } else {
    // Claude: uses stream-json for rich tool-use display
    const claudeState: ClaudeStreamState = {
      currentToolName: null,
      currentBlockIsToolUse: false,
      inputJsonBuffer: "",
      inTextBlock: false,
      sawTextDeltaInCurrentMessage: false,
    };

    result = await runCommand("claude", [
      "-p",
      "--verbose",
      "--output-format", "stream-json",
      "--include-partial-messages",
      `--disallowed-tools=${[...BLOCKED_GIT_OPS, ...BLOCKED_FILE_OPS].join(",")}`,
      "--permission-mode", "bypassPermissions",
      "--model", CLAUDE_MODEL,
      "--effort", CLAUDE_EFFORT,
      prompt,
    ], {
      cwd: REPO_ROOT,
      timeout: 7_200_000,
      streamOutput: true,
      stdoutLineFormatter: (line) => formatClaudeStreamLine(line, claudeState),
    });
  }

  clearInterval(heartbeat);
  console.log(c("1;36", SEP));
  console.log(c("1;32", `  \u2705 Agent done in ${formatElapsed(startedAt)}`));
  console.log(c("1;36", SEP));

  if (result.exitCode !== 0 || result.signal) {
    const how = result.signal ? `killed by ${result.signal}` : `exited with code ${result.exitCode}`;
    console.log(c("1;31", `  Agent ${how}`));
    const tailStderr = result.stderr.slice(-2000).trimEnd();
    const tailStdout = result.stdout.slice(-2000).trimEnd();
    if (tailStderr) console.log(c("1;31", `  stderr tail:\n${tailStderr}`));
    if (tailStdout) console.log(c("2", `  stdout tail:\n${tailStdout}`));
  }

  return result;
}

// -- Resume from previous run ------------------------------------------------

type LogEntry = {
  cycle: number;
  effort: number;
  tokPerSec: number | null;
  tokPerSecSamples?: number[];
  promptTokens?: number | null;
  promptTokenSamples?: number[];
  bandwidthUtil: number | null;
  bandwidthSamples?: number[];
  correct: boolean;
  improved: boolean;
  broken: boolean;
  kept?: boolean;
  foundationKeep?: boolean;
  decisionReason?: string;
  description?: string;
  stepKind?: StepKind;
  changedFiles?: string[];
  outputText: string;
  commitHash?: string | null;
  timestamp: string;
};

export function codexExecArgs(prompt: string): string[] {
  return [
    "exec",
    "-c",
    `model_reasoning_effort="${CODEX_REASONING_EFFORT}"`,
    "--dangerously-bypass-approvals-and-sandbox",
    "--json",
    "--model",
    CODEX_MODEL,
    prompt,
  ];
}

export function opencodeExecArgs(prompt: string): string[] {
  // `opencode run` is the headless, non-interactive mode (see
  // https://opencode.ai). It starts its own server, drives the model to a
  // final message, and exits. --dangerously-skip-permissions auto-approves
  // tool use so the loop is not blocked on prompts; the loop itself still
  // owns git keep/revert, and the prompt forbids destructive git ops.
  const args = ["run", "--dangerously-skip-permissions", "--model", OPENCODE_MODEL];
  if (OPENCODE_VARIANT.length > 0) args.push("--variant", OPENCODE_VARIANT);
  if (OPENCODE_AGENT.length > 0) args.push("--agent", OPENCODE_AGENT);
  args.push(prompt);
  return args;
}

function statePathForEffort(effort: number): string {
  return join(RESULTS_DIR, `effort_${effort}_state.json`);
}

function logPathForEffort(effort: number): string {
  return join(RESULTS_DIR, `effort_${effort}_log.jsonl`);
}

export function effortArtifactPaths(effort: number): string[] {
  return [
    statePathForEffort(effort),
    logPathForEffort(effort),
  ];
}

export async function cleanupPreviousRunArtifacts(effort: number): Promise<string[]> {
  const removed: string[] = [];
  for (const path of effortArtifactPaths(effort)) {
    if (!existsSync(path)) continue;
    await rm(path, { force: true });
    removed.push(path);
  }
  return removed;
}

async function loadLoopState(effort: number): Promise<LoopState | null> {
  const statePath = statePathForEffort(effort);
  if (!existsSync(statePath)) return null;
  return JSON.parse(await readFile(statePath, "utf8")) as LoopState;
}

async function saveLoopState(state: LoopState): Promise<void> {
  state.lastUpdatedAt = new Date().toISOString();
  const spec = getEffortSpec(state.effort);
  const refPaths = spec?.referenceImplementations?.map((r) => r.path) ?? [];
  state.runMetrics = computeRunMetrics(state, refPaths);
  await writeFile(statePathForEffort(state.effort), JSON.stringify(state, null, 2));
}

export function benchmarkSignatureForSpec(spec: EffortSpec, modelKey?: string, modelPath?: string): string {
  const selectedModelKey = modelKey ?? spec.defaultModel ?? null;
  const selectedModel = selectedModelKey ? MODELS[selectedModelKey] : null;
  return JSON.stringify({
    doc: spec.doc,
    metricMode: spec.metricMode,
    primaryMetricLabel: spec.primaryMetricLabel,
    modelKey: selectedModelKey,
    modelPath: modelPath ?? (selectedModelKey ? MODELS[selectedModelKey]?.path ?? null : null),
    benchmarkPrompt: spec.benchmarkPrompt,
    benchmarkMaxTokens: spec.benchmarkMaxTokens,
    benchmarkMethod: spec.benchmarkMethod,
    ...(selectedModel?.systemPrompt ? { benchmarkSystemPrompt: selectedModel.systemPrompt } : {}),
  });
}

export function isResumeStateCompatible(saved: LoopState, spec: EffortSpec, modelKey?: string, modelPath?: string): boolean {
  return saved.benchmarkSignature === benchmarkSignatureForSpec(spec, modelKey, modelPath);
}

function createInitialState(
  effort: number,
  planDoc: string,
  baseline: BenchResult,
  headCommit: string | null,
  benchmarkSignature: string,
): LoopState {
  const now = new Date().toISOString();
  return {
    effort,
    planDoc,
    benchmarkSignature,
    runStartedAt: now,
    lastUpdatedAt: now,
    lastCycle: 0,
    bestTokPerSec: baseline.tokPerSec ?? 0,
    bestCycle: 0,
    bestCommitHash: headCommit,
    bestResult: benchResultToCheckpoint(baseline, 0, headCommit),
    stalledCycles: 0,
    consecutiveFoundationKeeps: 0,
    cycles: [],
    failedApproaches: [],
    ideas: [],
    reviewSummaries: [],
  };
}

/**
 * Heuristic: does the change introduce a new runtime env flag? If so,
 * foundation-keep requires evidence that the flag-ON path was measured in
 * the same cycle. Otherwise we accumulate dormant wiring that isn't
 * disproved until a much later cycle (see effort-6 cycles 1/3/5/7 shipping
 * flag-gated pair-batch infra that cycles 8/9 measured as net-negative).
 */
export function introducesRuntimeFlag(report: AgentReport, changedFiles: string[]): boolean {
  const haystack = `${report.description}\n${report.selfAnalysis}\n${report.rawText}`;
  const mentionedFlags = [...haystack.matchAll(/\bZINC_[A-Z0-9_]+\b/g)].map((match) => match[0]);
  if (
    mentionedFlags.length > 0
    && mentionedFlags.every((flag) => DIAGNOSTIC_RUNTIME_FLAGS.has(flag))
    && /\b(profile|profiling|instrument|instrumentation|phase budget|bucket|timing|timestamp)\b/i.test(haystack)
  ) {
    return false;
  }
  if (/ZINC_[A-Z0-9_]+\s*=\s*1|ZINC_[A-Z0-9_]+\b.*flag|flag[-_]?gated|behind .*flag|default(?:s)?\s+off|default(?:s)?\s+on/i.test(haystack)) {
    return true;
  }
  // Scan the change diff text indirectly via changedFiles + typical env patterns.
  return /std\.posix\.getenv/i.test(haystack) || /getenv\("ZINC_/.test(haystack);
}

function isDiagnosticProfileRepair(report: AgentReport): boolean {
  const haystack = `${report.description}\n${report.selfAnalysis}\n${report.rawText}`;
  return /\bZINC_PREFILL_PROFILE\b/i.test(haystack)
    && /\b(profile|profiling|instrument|instrumentation|phase budget|bucket|parseable|emitted|timing|timestamp)\b/i.test(haystack)
    && (/\b\d+\.\d+\s*(?:ms|µs|us)\b/i.test(haystack)
      || /\b(?:dense_ffn|attention|qkv|gate_?up|down|ssm|moe|router|prefill_[a-z0-9_]+)\b/i.test(haystack));
}

/**
 * When the cycle introduces a runtime flag, require that the self-analysis
 * records a measurement of the flag-ON path. The exact wording varies but
 * the self-analysis must cite a concrete tok/s number AND reference the
 * flag-on state. If this evidence is missing, the change is dormant
 * infrastructure that hasn't been validated; don't commit it.
 */
export function hasFlagOnMeasurementEvidence(report: AgentReport): boolean {
  const haystack = `${report.description}\n${report.selfAnalysis}\n${report.rawText}`;
  const citesFlagOn = /flag[- ]?on|flag\s+(?:=\s*)?1|ZINC_[A-Z0-9_]+=1|with\s+flag\s+set|enabled\s+path|flag\s+enabled|flag\s+ON|when\s+enabled/i.test(haystack);
  const citesFlagOnNumber = /\b\d+\.\d+\s*tok\/s\b/.test(haystack);
  return citesFlagOn && citesFlagOnNumber;
}

/**
 * A cycle with zero final changed files is a "no-op" only when the agent
 * genuinely did nothing. When the agent explored a hypothesis, measured
 * it, found it net-negative, and cleaned up (reverted the code) — that
 * produces real information: the hypothesis is now disproved and future
 * cycles should not repeat it. Distinguishing the two matters because
 * treating revert-after-measurement as a stall penalizes the exact
 * behavior the pivot prompt asks for.
 *
 * Heuristic: stepKind=rollback, OR the description/analysis describes
 * the cycle as a revert/rollback after a measurement (with a tok/s
 * number present to prove the measurement happened).
 */
export function isMeasuredDeadRevert(report: AgentReport): boolean {
  if (report.stepKind === "rollback") return true;
  const haystack = `${report.description}\n${report.selfAnalysis}`;
  const mentionsRevert = /\b(reverted|rolled back|cleaned up|undid|removed the)\b/i.test(haystack);
  const mentionsDead = /\b(net[- ]negative|flat|dead[- ]end|no improvement|within noise|no measurable|did not pay|didn'?t help|unchanged\b|within.*noise)\b/i.test(haystack);
  // Accept any evidence of a concrete measurement: a tok/s number, a
  // millisecond number, or a comparison pattern (e.g. "25.63 vs 25.66").
  // Agents phrase measurements differently and the pattern we want to
  // recognize is "did the cycle cite a real number", not a specific unit.
  const citesNumber = /\b\d+\.\d+\s*(?:tok\/s|ms|µs|us)\b/i.test(haystack)
    || /\b\d+\.\d+\s*(?:vs|->|→|versus)\s*\d+\.\d+\b/i.test(haystack);
  return mentionsRevert && mentionsDead && citesNumber;
}

/**
 * Detect agent rate-limit responses in stdout/stderr and return when the limit
 * resets, so the cycle loop can sleep through it instead of burning a no-op.
 * The prior 50-cycle run lost 36/50 cycles (72%) to claude session-limit
 * rejections that masqueraded as no-ops, polluting the stall counter and
 * triggering false pivots. Detection is layered for robustness:
 *   1. Claude API `rate_limit_event` JSON with `resetsAt` (most reliable).
 *   2. Claude plain-text "session limit · resets HH:MM(am|pm)" fallback.
 *   3. Codex plain-text "try again at <Month Day, Year HH:MM AM/PM>".
 *   4. Generic phrase match → +60 min fallback so we never silently no-op
 *      on an unrecognized rate-limit string.
 * Returns null when no rate-limit signature is present (real no-op).
 */
export function detectAgentRateLimit(
  stdout: string,
  stderr: string = "",
  nowMs: number = Date.now(),
): { resetsAtMs: number; source: string } | null {
  const combined = `${stdout}\n${stderr}`;
  // 1. Claude API: {"type":"rate_limit_event",..."resetsAt":<unix-seconds>,...}
  const apiMatch = combined.match(/"rate_limit_event"[\s\S]{0,400}?"resetsAt":\s*(\d{9,12})/);
  if (apiMatch) {
    const seconds = parseInt(apiMatch[1], 10);
    if (Number.isFinite(seconds) && seconds > 1_700_000_000) {
      return { resetsAtMs: seconds * 1000, source: "claude api" };
    }
  }
  // 2. Claude text: "session limit · resets 7:40pm (TZ)"
  const claudeText = combined.match(/session limit[^\n]*?resets\s+(\d{1,2}):(\d{2})\s*(am|pm)/i);
  if (claudeText) {
    const hh12 = parseInt(claudeText[1], 10);
    const mm = parseInt(claudeText[2], 10);
    const isPm = claudeText[3].toLowerCase() === "pm";
    const hh = (hh12 % 12) + (isPm ? 12 : 0);
    const target = new Date(nowMs);
    target.setHours(hh, mm, 0, 0);
    if (target.getTime() <= nowMs) target.setDate(target.getDate() + 1);
    return { resetsAtMs: target.getTime(), source: "claude text" };
  }
  // 3. Codex text: "try again at May 26th, 2026 11:33 AM"
  const codexText = combined.match(
    /try again at ([A-Z][a-z]+\s+\d+)(?:st|nd|rd|th)?,?\s+(\d{4})\s+(\d{1,2}:\d{2}\s*(?:AM|PM))/i,
  );
  if (codexText) {
    const parsed = Date.parse(`${codexText[1]}, ${codexText[2]} ${codexText[3]}`);
    if (Number.isFinite(parsed) && parsed > nowMs) {
      return { resetsAtMs: parsed, source: "codex text" };
    }
  }
  // 4. Generic fallback: clear rate-limit phrase but unparseable timing.
  if (/(session limit|usage limit|\brate_limit\b|api_error_status":\s*429)/i.test(combined)) {
    return { resetsAtMs: nowMs + 60 * 60 * 1000, source: "fallback (+60m)" };
  }
  return null;
}

export function shouldKeepFoundationStep(
  candidate: BenchResult,
  bestPerf: BenchResult,
  stalledCycles: number,
  consecutiveFoundationKeeps: number,
  report: AgentReport,
  changedFiles: string[],
): boolean {
  if (!candidate.buildOk || !candidate.correct || candidate.tokPerSec == null) return false;
  if (!isEnablementLike(report, changedFiles)) return false;
  if (consecutiveFoundationKeeps >= MAX_FOUNDATION_KEEPS_IN_A_ROW) return false;
  if (changedFiles.length === 0) return false;

  const bestTokPerSec = bestPerf.tokPerSec ?? 0;
  if (candidate.tokPerSec > bestTokPerSec + improvementThreshold(bestTokPerSec)) return false;
  if (candidate.tokPerSec < bestTokPerSec - FOUNDATION_KEEP_MAX_DROP_TPS) return false;

  // Flag-gated foundation without flag-on measurement is dormant wiring.
  // The dormant-commit trap: it passes flag-off because the flag-off path is
  // untouched, but the flag-on path may regress (effort-6 cycles 8/9 proved
  // this on pair-dispatch). Require the same cycle to cite a flag-on number.
  if (introducesRuntimeFlag(report, changedFiles) && !hasFlagOnMeasurementEvidence(report)) {
    return false;
  }

  return stalledCycles >= 2 || report.stepKind === "enablement" || isDiagnosticProfileRepair(report);
}

export async function loadPreviousRun(effort: number): Promise<{
  history: string;
  bestTokPerSec: number;
  lastCycle: number;
  bestCycle: number | null;
  bestCommitHash: string | null;
}> {
  const state = await loadLoopState(effort);
  if (state) {
    return {
      history: buildHistoryFromCycles(state.cycles),
      bestTokPerSec: state.bestTokPerSec,
      lastCycle: state.lastCycle,
      bestCycle: state.bestCycle,
      bestCommitHash: state.bestCommitHash,
    };
  }

  const logPath = logPathForEffort(effort);
  let history = "";
  let bestTokPerSec = 0;
  let lastCycle = 0;
  let bestCycle: number | null = null;
  let bestCommitHash: string | null = null;

  try {
    const content = await readFile(logPath, "utf8");
    for (const line of content.split("\n").filter(Boolean)) {
      try {
        const entry = JSON.parse(line) as LogEntry;
        if (entry.effort !== effort) continue;
        lastCycle = Math.max(lastCycle, entry.cycle);
        if (entry.broken) {
          history += `\nCycle ${entry.cycle}: REVERTED \u2014 ${entry.decisionReason ?? `broken (${entry.outputText?.slice(0, 60)})`}`;
        } else if (entry.improved) {
          history += `\nCycle ${entry.cycle}: KEPT \u2014 ${entry.tokPerSec?.toFixed(2)} tok/s${entry.tokPerSecSamples?.length ? ` ${formatSampleList(entry.tokPerSecSamples)}` : ""}`;
          if (entry.tokPerSec != null && entry.tokPerSec > bestTokPerSec) {
            bestTokPerSec = entry.tokPerSec;
            bestCycle = entry.cycle;
            bestCommitHash = entry.commitHash ?? null;
          }
        } else if (entry.foundationKeep) {
          history += `\nCycle ${entry.cycle}: KEPT-FOUNDATION \u2014 ${entry.description ?? entry.decisionReason ?? "enablement step"}`;
        } else {
          history += `\nCycle ${entry.cycle}: REVERTED \u2014 ${entry.decisionReason ?? `no improvement (${entry.tokPerSec?.toFixed(2)} tok/s${entry.tokPerSecSamples?.length ? ` ${formatSampleList(entry.tokPerSecSamples)}` : ""})`}`;
        }
      } catch { /* skip malformed lines */ }
    }
  } catch { /* no log file yet */ }

  return { history, bestTokPerSec, lastCycle, bestCycle, bestCommitHash };
}

// -- Selective revert (only agent-editable perf paths, not loops/site/docs) ---

async function revertAgentChanges(): Promise<void> {
  for (const path of REVERTABLE_PATHS) {
    await runCommand("git", ["checkout", "--", path], { cwd: REPO_ROOT });
  }
  // Also clean any new untracked files the agent may have created under
  // revertable directories.
  const untrackedArgs = REVERTABLE_PATHS.filter((path) => path.endsWith("/"));
  const { stdout: untracked } = untrackedArgs.length > 0
    ? await runCommand("git", ["ls-files", "--others", "--exclude-standard", ...untrackedArgs], { cwd: REPO_ROOT })
    : { stdout: "" };
  for (const f of untracked.split("\n").filter(Boolean)) {
    await runCommand("rm", ["-f", f], { cwd: REPO_ROOT });
  }
  console.log(c("2", "  Reverted agent changes (build.zig/src only)."));
}

// -- Main loop ---------------------------------------------------------------

async function main() {
  // Race-prevention guard: refuse to start if another optimize_perf.ts
  // is already running. Two concurrent loops rsyncing/building/committing
  // to the same main branch corrupts state and makes per-cycle measurements
  // unreproducible (the other run mutates the tree mid-benchmark). Opt out
  // with ZINC_PERF_ALLOW_PARALLEL=1 only if you know what you're doing.
  if (process.env.ZINC_PERF_ALLOW_PARALLEL !== "1") {
    const existing = detectExistingOptimizePerfRuns();
    if (existing.length > 0) {
      console.error(
        `ERROR: another optimize_perf.ts loop is already running (PID${existing.length > 1 ? "s" : ""}: ${existing.join(", ")}). ` +
        `Stop it first (kill ${existing.join(" ")}) or set ZINC_PERF_ALLOW_PARALLEL=1 to override. ` +
        `Two concurrent loops produced unreproducible measurements and corrupted state on 2026-05-30.`,
      );
      process.exit(2);
    }
  }
  const { effort, cycles, dryRun, model: requestedModel, modelExplicit, resume, agent, analyze } = parseArgs();
  const effortSpec = getEffortSpec(effort);
  if (!effortSpec) {
    throw new Error(`Unknown effort: ${effort}`);
  }
  const model = modelExplicit ? requestedModel : (effortSpec.defaultModel ?? requestedModel);
  const modelTarget = MODELS[model] ?? MODELS.qwen36b;
  const effortFile = effortSpec.doc;
  const plan = await readFile(join(EFFORTS_DIR, effortFile), "utf8");

  await mkdir(RESULTS_DIR, { recursive: true });

  if (analyze) {
    const saved = await loadLoopState(effort);
    if (!saved) {
      console.error(c("1;31", `No saved state found for effort ${effort}.`));
      process.exit(1);
    }
    console.log(buildAnalysisReport(saved));
    return;
  }

  console.log(c("1;37", `\n\u2554${"═".repeat(BOX_INNER_WIDTH)}\u2557`));
  console.log(c("1;37", boxLine(`ZINC Performance Optimization Loop — Effort ${effort}`)));
  console.log(c("1;37", boxLine(effortFile)));
  console.log(c("1;37", boxLine(`Model: ${model}`)));
  console.log(c("1;37", boxLine(`RDNA node: ${SELECTED_RDNA_NODE ?? "(default)"}`)));
  const agentDetails = agent === "claude"
    ? ` (${CLAUDE_MODEL} effort=${CLAUDE_EFFORT})`
    : agent === "opencode"
      ? ` (${OPENCODE_MODEL}${OPENCODE_VARIANT.length > 0 ? `, variant=${OPENCODE_VARIANT}` : ""}${OPENCODE_AGENT.length > 0 ? `, agent=${OPENCODE_AGENT}` : ""})`
      : ` (${CODEX_MODEL} effort=${CODEX_REASONING_EFFORT})`;
  console.log(c("1;37", boxLine(`Agent: ${agent}${agentDetails}`)));
  console.log(c("1;37", boxLine(`Cycles this run: ${cycles}`)));
  if (resume) console.log(c("1;37", boxLine("Resuming from previous run")));
  console.log(c("1;37", `\u255A${"═".repeat(BOX_INNER_WIDTH)}\u255D\n`));

  if (!resume) {
    const removedArtifacts = await cleanupPreviousRunArtifacts(effort);
    if (removedArtifacts.length > 0) {
      console.log(c("2", `  Cleaned ${removedArtifacts.length} saved artifact(s) from previous effort-${effort} runs.`));
    }
  }

  // Step 1: Sync and get baseline
  console.log(c("1;33", "\u2500\u2500 Baseline " + "\u2500".repeat(54)));
  await cleanRemoteBenchmarkNode();
  await rsyncToRemote();
  const originalBaseline = await buildAndBench(modelTarget, effortSpec);

  if (!originalBaseline.buildOk) {
    console.error(c("1;31", "Baseline build failed! Fix build errors first."));
    process.exit(1);
  }
  if (!originalBaseline.correct) {
    console.error(c("1;31", `Baseline output incorrect: "${originalBaseline.outputText}". Fix correctness first.`));
    process.exit(1);
  }
  if (originalBaseline.tokPerSec == null) {
    console.error(c("1;31", `Baseline ${effortSpec.primaryMetricLabel} was not parseable. Fix the benchmark command or parser before starting optimization cycles.`));
    process.exit(1);
  }
  const minHealthyTokPerSec = minHealthyTokPerSecForSpec(effortSpec);
  if (minHealthyTokPerSec != null && originalBaseline.tokPerSec < minHealthyTokPerSec) {
    console.error(c(
      "1;31",
      `Baseline ${effortSpec.primaryMetricLabel} ${originalBaseline.tokPerSec.toFixed(2)} tok/s is below the ${minHealthyTokPerSec.toFixed(2)} tok/s health floor.`,
    ));
    console.error(c("1;31", "This usually means the RDNA node is contaminated, the GPU path is not active, or the driver/runtime state is unhealthy. Clean/reboot/fix the node before burning agent cycles."));
    console.error(c("2", "Set ZINC_MIN_HEALTHY_TPS=0 only if you intentionally want to optimize from this degraded baseline."));
    process.exit(1);
  }

  console.log(c("1;32", `  Baseline (${effortSpec.primaryMetricLabel}): ${summarizeBenchMetric(originalBaseline.tokPerSec, originalBaseline.tokPerSecSamples, "tok/s")}, BW: ${summarizeBenchMetric(originalBaseline.bandwidthUtil, originalBaseline.bandwidthSamples, "%", 1)}`));
  console.log(c("1;32", `  Output: "${originalBaseline.outputText.slice(0, 80)}"`));

  let baselinePhaseBudget: PhaseBudgetCollection = { budget: null, failureReason: null };
  if (effortSpec.metricMode === "prefill") {
    console.log(c("2", "  Capturing baseline prefill phase budget (ZINC_PREFILL_PROFILE=1)..."));
    baselinePhaseBudget = await collectPhaseBudget(modelTarget, effortSpec);
    if (baselinePhaseBudget.budget?.biggestBucket) {
      console.log(c("1;36", `  Biggest prefill bucket at baseline: ${baselinePhaseBudget.budget.biggestBucket.name} (${baselinePhaseBudget.budget.biggestBucket.totalMs.toFixed(1)} ms)`));
    } else {
      console.log(c("1;33", `  Phase budget collection did not emit parseable phase data; prompt will note this (${baselinePhaseBudget.failureReason ?? "unknown"}).`));
    }
  }

  const benchmarkSignature = benchmarkSignatureForSpec(effortSpec, model, modelTarget.path);
  let currentCode = originalBaseline;
  let bestPerf = originalBaseline;
  let bestTokPerSec = bestPerf.tokPerSec ?? 0;
  let startCycle = 1;
  const headCommit = (await runCommand("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT })).stdout.trim() || null;
  let state = createInitialState(effort, effortFile, originalBaseline, headCommit, benchmarkSignature);
  state.phaseBudget = baselinePhaseBudget.budget;
  state.phaseBudgetCycle = baselinePhaseBudget.budget ? 0 : null;
  state.phaseBudgetRefreshFailures = baselinePhaseBudget.budget ? 0 : (baselinePhaseBudget.failureReason ? 1 : 0);
  state.phaseBudgetRefreshLastFailure = baselinePhaseBudget.budget ? null : baselinePhaseBudget.failureReason;
  state.phaseBudgetRefreshLastAttemptCycle = effortSpec.metricMode === "prefill" ? 0 : null;

  if (resume) {
    const saved = await loadLoopState(effort);
    if (saved) {
      if (!isResumeStateCompatible(saved, effortSpec, model, modelTarget.path)) {
        console.log(c(
          "1;33",
          "  Resume note: saved state uses an older or different benchmark signature. Ignoring it and starting fresh for this effort.",
        ));
      } else {
        state = saved;
        startCycle = saved.lastCycle + 1;
        if (saved.bestResult) {
          bestPerf = checkpointToBenchResult(saved.bestResult);
          bestTokPerSec = saved.bestTokPerSec;
        }
        console.log(c("1;36", `  Resumed: ${saved.lastCycle} previous cycles, recorded best ${saved.bestTokPerSec.toFixed(2)} tok/s (${effortSpec.primaryMetricLabel})`));
        if (saved.cycles.length > 0) {
          console.log(c("2", `  Recent cycles:\n${buildRecentCycleBlock(saved.cycles)}`));
        }
        if (saved.bestTokPerSec > (currentCode.tokPerSec ?? 0) + improvementThreshold(currentCode.tokPerSec)) {
          const bestCommitNote = saved.bestCommitHash ? ` on commit ${saved.bestCommitHash.slice(0, 8)}` : "";
          console.log(c(
            "1;33",
            `  Resume note: recorded best cycle ${saved.bestCycle ?? "?"}${bestCommitNote} was faster than the current HEAD benchmark. The loop will branch from the code you currently have checked out, not from that historical metric.`,
          ));
        }
        if (saved.reviewSummaries.length > 0) {
          console.log(c("2", `  Latest review:\n${saved.reviewSummaries.at(-1)}`));
        }
      }
    } else {
      console.log(c("2", "  No previous run found, starting fresh."));
    }
  }

  console.log(c("2", "  Capturing accepted coherence baseline..."));
  let acceptedCoherence = await runCoherenceSweep();
  if (acceptedCoherence.failures.length > 0) {
    console.log(c("1;33", "  Accepted baseline already has cross-model failures; enforcing non-regression only."));
    console.log(c("2", `    ${formatCoherenceFailureList(acceptedCoherence.failures)}`));
  }

  let history = buildHistoryFromCycles(state.cycles);

  // Step 2: Optimization cycles
  for (let cycle = startCycle; cycle < startCycle + cycles; cycle++) {
    console.log(c("1;33", `\n\u2500\u2500 Cycle ${cycle} ` + "\u2500".repeat(54)));

    if (dryRun) {
      console.log(c("2", "  Dry run \u2014 skipping agent."));
      break;
    }

    const promptContext: PromptContext = {
      cycles: state.cycles,
      failedApproaches: state.failedApproaches,
      ideas: state.ideas,
      stalledCycles: state.stalledCycles,
      consecutiveFoundationKeeps: state.consecutiveFoundationKeeps,
      reviewSummary: state.reviewSummaries.at(-1) ?? null,
      bestPerf: state.bestResult ?? benchResultToCheckpoint(bestPerf, 0, state.bestCommitHash),
      phaseBudget: state.phaseBudget ?? null,
      phaseBudgetCycle: state.phaseBudgetCycle ?? null,
      phaseBudgetRefreshFailures: state.phaseBudgetRefreshFailures ?? 0,
      phaseBudgetRefreshLastFailure: state.phaseBudgetRefreshLastFailure ?? null,
      phaseBudgetRefreshLastAttemptCycle: state.phaseBudgetRefreshLastAttemptCycle ?? null,
    };

    const agentRun = await spawnAgent(effortFile, plan, originalBaseline, currentCode, cycle, history, model, agent, promptContext, effortSpec);
    const agentReport = parseAgentReport(agentRun.stdout);

    let changedFiles = await listChangedFiles();
    if (changedFiles.length === 0) {
      // Rate-limit backoff. If the agent was rejected by its quota/session
      // limit (rather than genuinely choosing to make no changes), sleep
      // until the reset time and retry the same cycle number. Without this
      // the prior 50-cycle run burned 36 cycles as no-ops on claude session
      // limits, polluted the stall counter, and triggered false pivots.
      const rl = detectAgentRateLimit(agentRun.stdout, agentRun.stderr);
      if (rl) {
        const tries = (rateLimitRetriesPerCycle.get(cycle) ?? 0) + 1;
        if (tries <= RATE_LIMIT_MAX_RETRIES) {
          rateLimitRetriesPerCycle.set(cycle, tries);
          const waitMs = Math.min(
            RATE_LIMIT_MAX_WAIT_MS,
            Math.max(60_000, rl.resetsAtMs - Date.now() + 60_000),
          );
          const eta = new Date(Date.now() + waitMs).toLocaleString();
          const mins = Math.ceil(waitMs / 60_000);
          console.log(c("1;33", `  ⏸ AGENT RATE LIMIT (${rl.source}) — sleeping ${mins} min until ~${eta}, then retrying cycle ${cycle} (attempt ${tries}/${RATE_LIMIT_MAX_RETRIES})`));
          await new Promise((resolve) => setTimeout(resolve, waitMs));
          cycle--; // for-loop will ++ back to the same number
          continue;
        }
        console.log(c("1;31", `  ⚠ Rate-limit retries exhausted for cycle ${cycle}; falling through as no-op`));
      }
      const measuredDead = isMeasuredDeadRevert(agentReport);
      const decisionReason = measuredDead
        ? "measured-dead: agent explored, measured, and reverted after finding the path non-positive"
        : "no source changes; skipped sync and benchmark";
      if (measuredDead) {
        console.log(c("1;36", `  \uD83D\uDD0E MEASURED DEAD: ${decisionReason}`));
        console.log(c("2", `     ${trunc(agentReport.description, 120)}`));
      } else {
        console.log(c("1;33", `  \u26A0 NO-OP: ${decisionReason}`));
      }

      state.failedApproaches = mergeUniqueEntries(
        state.failedApproaches,
        [`${agentReport.description} — ${decisionReason}`],
        FAILED_APPROACH_LIMIT,
      );
      state.ideas = mergeUniqueEntries(state.ideas, agentReport.nextIdeas, IDEA_LIMIT);
      // Revert-after-measurement cycles produce information (they disprove
      // a hypothesis), but they still did not create a new best checkpoint.
      // Count them as stall pressure so long runs of well-measured dead ends
      // still trigger pivot prompts and phase-budget refreshes instead of
      // printing stall=0 forever while the search keeps circling the same
      // local neighborhood.
      state.stalledCycles++;
      state.consecutiveFoundationKeeps = 0;
      state.lastCycle = cycle;

      const cycleRecord: CycleRecord = {
        cycle,
        timestamp: new Date().toISOString(),
        description: agentReport.description,
        selfAnalysis: agentReport.selfAnalysis,
        nextIdeas: agentReport.nextIdeas,
        stepKind: measuredDead ? "rollback" : agentReport.stepKind,
        changedFiles: [],
        categoryTags: classifyApproachTags(agentReport.description, []),
        tokPerSec: null,
        tokPerSecSamples: [],
        bandwidthUtil: null,
        bandwidthSamples: [],
        correct: measuredDead,
        improved: false,
        broken: false,
        kept: false,
        foundationKeep: false,
        decisionReason,
        outputText: "",
        commitHash: null,
      };
      state.cycles.push(cycleRecord);

      if (state.cycles.length % SELF_REVIEW_EVERY === 0) {
        const review = buildSelfReview(state);
        if (review) state.reviewSummaries = [...state.reviewSummaries.slice(-(REVIEW_SUMMARY_LIMIT - 1)), review];
      }

      await saveLoopState(state);
      history = buildHistoryFromCycles(state.cycles);

      const logEntry: LogEntry = {
        cycle,
        effort,
        tokPerSec: null,
        tokPerSecSamples: [],
        bandwidthUtil: null,
        bandwidthSamples: [],
        correct: false,
        improved: false,
        broken: false,
        kept: false,
        foundationKeep: false,
        decisionReason,
        description: agentReport.description,
        stepKind: agentReport.stepKind,
        changedFiles: [],
        outputText: "",
        commitHash: null,
        timestamp: new Date().toISOString(),
      };
      const logPath = logPathForEffort(effort);
      await writeFile(logPath, JSON.stringify(logEntry) + "\n", { flag: "a" });
      console.log(c("2", `  stall=${state.stalledCycles} best=${bestTokPerSec.toFixed(2)} current=${currentCode.tokPerSec?.toFixed(2) ?? "?"}`));
      continue;
    }

    // Sync and benchmark — with up to 2 fix-up retries if build fails
    console.log(c("2", "  Syncing changes..."));
    await cleanRemoteBenchmarkNode();
    await rsyncToRemote();
    let result = await buildAndBench(modelTarget, effortSpec);

    const MAX_FIX_RETRIES = 2;
    for (let fix = 0; fix < MAX_FIX_RETRIES && !result.buildOk; fix++) {
      console.log(c("1;33", `  \u26A0 Build failed — sending errors to agent for fix (retry ${fix + 1}/${MAX_FIX_RETRIES})`));
      const fixPrompt = `The build FAILED after your changes. Fix the errors and make it compile.

## Build errors:
\`\`\`
${result.buildOutput.slice(-2000)}
\`\`\`

## Rules:
- Fix ONLY the build errors. Do not add new features.
- The code must compile: zig build -Doptimize=ReleaseFast must succeed on the remote node.
- Do not use sub-agents, delegation, spawn_agent, or wait_agent.
- Re-read the file right before patching it; do not patch against stale context.
- rsync to remote: rsync -avz --checksum --delete -e "ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no" ${AGENT_RSYNC_EXCLUDE_ARGS} ${REPO_ROOT}/ ${ZINC_USER}@${ZINC_HOST}:${REMOTE_DIR}/
- Build on remote: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast 2>&1"
- Shader compilation: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR}/src/shaders && for f in *.comp; do glslc --target-env=vulkan1.3 -fshader-stage=compute \\$f -o \\$\{f%.comp}.spv 2>&1; done"`;

      if (agent === "codex") {
        await runCommand("codex", codexExecArgs(fixPrompt), {
          cwd: REPO_ROOT, timeout: 600_000, streamOutput: true,
          stdoutLineFormatter: (line) => formatCodexStreamLine(line),
        });
      } else if (agent === "opencode") {
        await runCommand("opencode", opencodeExecArgs(fixPrompt), {
          cwd: REPO_ROOT, timeout: 600_000, streamOutput: true,
        });
      } else {
        const fixState: ClaudeStreamState = {
          currentToolName: null, currentBlockIsToolUse: false,
          inputJsonBuffer: "", inTextBlock: false, sawTextDeltaInCurrentMessage: false,
        };
        await runCommand("claude", [
          "-p", "--verbose", "--output-format", "stream-json", "--include-partial-messages",
          `--disallowed-tools=${[...BLOCKED_GIT_OPS, ...BLOCKED_FILE_OPS].join(",")}`,
          "--permission-mode", "bypassPermissions",
          "--model", CLAUDE_MODEL,
          "--effort", CLAUDE_EFFORT,
          fixPrompt,
        ], {
          cwd: REPO_ROOT, timeout: 600_000, streamOutput: true,
          stdoutLineFormatter: (line) => formatClaudeStreamLine(line, fixState),
        });
      }

      console.log(c("2", "  Re-syncing after fix..."));
      await rsyncToRemote();
      result = await buildAndBench(modelTarget, effortSpec);
    }

    changedFiles = await listChangedFiles();
    const categoryTags = classifyApproachTags(agentReport.description, changedFiles);
    const improved = isMaterialImprovement(result, bestPerf, state.stalledCycles);
    const foundationCandidate = shouldKeepFoundationStep(
      result,
      bestPerf,
      state.stalledCycles,
      state.consecutiveFoundationKeeps,
      agentReport,
      changedFiles,
    );

    let coherenceError: string | null = null;
    let coherenceSweep: CoherenceSweep | null = null;
    if (result.buildOk && result.correct && (improved || foundationCandidate)) {
      console.log(c("2", "  Checking all models for coherence..."));
      coherenceSweep = await runCoherenceSweep();
      coherenceError = summarizeCoherenceRegression(coherenceSweep, acceptedCoherence.failureIds);
      if (coherenceError) {
        console.log(c("1;31", `  ${coherenceError}`));
      } else if (coherenceSweep.failures.length > 0) {
        console.log(c("2", `  Coherence unchanged vs accepted baseline (${coherenceSweep.failures.length} known failing case(s)).`));
      } else if (acceptedCoherence.failures.length > 0) {
        console.log(c("1;36", "  Coherence improved: all accepted-baseline failures cleared."));
      }
    }

    const correct = result.correct && coherenceError == null;
    const broken = !result.buildOk || !correct;
    const threshold = improvementThreshold(bestPerf.tokPerSec, state.stalledCycles);

    const deltaVsBest = result.tokPerSec != null && (bestPerf.tokPerSec ?? 0) > 0
      ? ((result.tokPerSec - (bestPerf.tokPerSec ?? 0)) / (bestPerf.tokPerSec ?? 1) * 100).toFixed(2)
      : "?";

    let kept = false;
    let foundationKeep = false;
    let decisionReason = "";
    let commitHash: string | null = null;

    if (broken) {
      const failureReason = coherenceError ?? result.error ?? "incorrect output";
      console.log(c("1;31", `  \u274C BROKEN: ${failureReason}`));
      console.log(c("1;31", `     Output: "${result.outputText?.slice(0, 80)}"`));
      decisionReason = failureReason;
      state.failedApproaches = mergeUniqueEntries(
        state.failedApproaches,
        [`${agentReport.description} — ${decisionReason}`],
        FAILED_APPROACH_LIMIT,
      );
      state.stalledCycles++;
      state.consecutiveFoundationKeeps = 0;
      await revertAgentChanges();
    } else if (improved) {
      kept = true;
      console.log(c("1;32", `  \u2705 IMPROVED: ${summarizeBenchMetric(result.tokPerSec, result.tokPerSecSamples, "tok/s")} (${effortSpec.primaryMetricLabel}, +${deltaVsBest}%, threshold +${threshold.toFixed(2)} tok/s vs best checkpoint)`));
      currentCode = result;
      bestPerf = result;
      bestTokPerSec = result.tokPerSec!;
      if (coherenceSweep) acceptedCoherence = coherenceSweep;
      state.stalledCycles = 0;
      state.consecutiveFoundationKeeps = 0;
      decisionReason = `improved by ${deltaVsBest}% vs best checkpoint`;

      await runCommand("git", ["add", "src/", "build.zig"], { cwd: REPO_ROOT });
      await runCommand("git", ["commit", "-m", `perf(effort-${effort}): cycle ${cycle} \u2014 ${result.tokPerSec?.toFixed(2)} ${effortSpec.primaryMetricLabel} (+${deltaVsBest}%)`], { cwd: REPO_ROOT });
      commitHash = (await runCommand("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT })).stdout.trim() || headCommit;
      state.bestTokPerSec = bestTokPerSec;
      state.bestCycle = cycle;
      state.bestCommitHash = commitHash;
      state.bestResult = benchResultToCheckpoint(result, cycle, commitHash);
      console.log(c("2", "  Committed."));

      // Refresh the per-phase budget so the next cycle's prompt reflects the
      // new shape of prefill after this structural change landed. Flat
      // keeps (foundation) do not refresh — they don't move phase totals
      // by enough to justify the extra profile run.
      if (effortSpec.metricMode === "prefill") {
        console.log(c("2", "  Refreshing prefill phase budget after keep..."));
        const refreshed = await collectPhaseBudget(modelTarget, effortSpec);
        applyPhaseBudgetCollection(state, refreshed, cycle, "New biggest prefill bucket");
      }
    } else if (foundationCandidate) {
      kept = true;
      foundationKeep = true;
      currentCode = result;
      if (coherenceSweep) acceptedCoherence = coherenceSweep;
      state.stalledCycles++;
      state.consecutiveFoundationKeeps++;
      decisionReason = `kept enablement step within ${FOUNDATION_KEEP_MAX_DROP_TPS.toFixed(2)} tok/s of best checkpoint`;
      console.log(c("1;36", `  \u2248 FOUNDATION KEEP: ${summarizeBenchMetric(result.tokPerSec, result.tokPerSecSamples, "tok/s")} (${effortSpec.primaryMetricLabel}, ${deltaVsBest}% vs best checkpoint)`));
      await runCommand("git", ["add", "src/", "build.zig"], { cwd: REPO_ROOT });
      await runCommand("git", ["commit", "-m", `perf(effort-${effort}): cycle ${cycle} foundation \u2014 ${trunc(agentReport.description, 72)}`], { cwd: REPO_ROOT });
      commitHash = (await runCommand("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT })).stdout.trim() || headCommit;
      console.log(c("2", "  Committed foundation step."));
    } else {
      decisionReason = `no improvement (needed +${threshold.toFixed(2)} tok/s vs best checkpoint)`;
      console.log(c("1;33", `  \u26A0 NO IMPROVEMENT: ${summarizeBenchMetric(result.tokPerSec, result.tokPerSecSamples, "tok/s")} (${effortSpec.primaryMetricLabel}, ${deltaVsBest}%, needed +${threshold.toFixed(2)} tok/s vs best checkpoint)`));
      state.failedApproaches = mergeUniqueEntries(
        state.failedApproaches,
        [`${agentReport.description} — ${decisionReason}`],
        FAILED_APPROACH_LIMIT,
      );
      state.stalledCycles++;
      state.consecutiveFoundationKeeps = 0;
      await revertAgentChanges();
    }

    state.ideas = mergeUniqueEntries(state.ideas, agentReport.nextIdeas, IDEA_LIMIT);
    state.lastCycle = cycle;

    const cycleRecord: CycleRecord = {
      cycle,
      timestamp: new Date().toISOString(),
      description: agentReport.description,
      selfAnalysis: agentReport.selfAnalysis,
      nextIdeas: agentReport.nextIdeas,
      stepKind: agentReport.stepKind,
      changedFiles,
      categoryTags,
      tokPerSec: result.tokPerSec,
      tokPerSecSamples: result.tokPerSecSamples,
      promptTokens: result.promptTokens ?? null,
      promptTokenSamples: result.promptTokenSamples ?? [],
      bandwidthUtil: result.bandwidthUtil,
      bandwidthSamples: result.bandwidthSamples,
      correct,
      improved,
      broken,
      kept,
      foundationKeep,
      decisionReason,
      outputText: result.outputText?.slice(0, 200),
      commitHash,
    };
    state.cycles.push(cycleRecord);

    if (state.cycles.length % SELF_REVIEW_EVERY === 0) {
      const review = buildSelfReview(state);
      if (review) {
        state.reviewSummaries = [...state.reviewSummaries.slice(-(REVIEW_SUMMARY_LIMIT - 1)), review];
        console.log(c("1;35", `  \uD83D\uDD0D Self-review (${state.cycles.length} cycles)`));
        console.log(c("2", review));
      }
    }

    // Stall-triggered phase budget refresh. Without a perf keep, the budget
    // stays frozen at the cycle that produced the last keep. Over a long
    // stall, accepted changes accumulate (foundation keeps, reverts that
    // produced information) and the agent's view of the budget grows
    // increasingly wrong. Refresh every Nth cycle while stalled so the
    // next cycle's prompt has a current view of bucket totals.
    if (
      effortSpec.metricMode === "prefill"
      && state.stalledCycles >= PHASE_BUDGET_REFRESH_STALL_THRESHOLD
      && state.phaseBudgetCycle !== cycle
      && (cycle - (state.phaseBudgetCycle ?? 0)) >= PHASE_BUDGET_REFRESH_STALL_THRESHOLD
    ) {
      console.log(c("2", `  Refreshing prefill phase budget (stall=${state.stalledCycles}, last refresh at cycle ${state.phaseBudgetCycle ?? "baseline"})...`));
      const refreshed = await collectPhaseBudget(modelTarget, effortSpec);
      applyPhaseBudgetCollection(state, refreshed, cycle, "Refreshed biggest prefill bucket");
    }

    await saveLoopState(state);
    history = buildHistoryFromCycles(state.cycles);

    // Log cycle result
    const logEntry: LogEntry = {
      cycle,
      effort,
      tokPerSec: result.tokPerSec,
      tokPerSecSamples: result.tokPerSecSamples,
      promptTokens: result.promptTokens ?? null,
      promptTokenSamples: result.promptTokenSamples ?? [],
      bandwidthUtil: result.bandwidthUtil,
      bandwidthSamples: result.bandwidthSamples,
      correct,
      improved,
      broken,
      kept,
      foundationKeep,
      decisionReason,
      description: agentReport.description,
      stepKind: agentReport.stepKind,
      changedFiles: changedFiles.slice(0, MAX_CHANGED_FILES_IN_PROMPT),
      outputText: result.outputText?.slice(0, 200),
      commitHash,
      timestamp: new Date().toISOString(),
    };
    const logPath = logPathForEffort(effort);
    await writeFile(logPath, JSON.stringify(logEntry) + "\n", { flag: "a" });
    console.log(c("2", `  stall=${state.stalledCycles} best=${bestTokPerSec.toFixed(2)} current=${currentCode.tokPerSec?.toFixed(2) ?? "?"}`));
  }

  // Summary
  console.log(c("1;37", `\n${"═".repeat(58)}`));
  console.log(c("1;37", `  Effort ${effort} complete.`));
  console.log(c("1;37", `  Baseline (${effortSpec.primaryMetricLabel}): ${originalBaseline.tokPerSec?.toFixed(2)} tok/s`));
  console.log(c("1;37", `  Best (${effortSpec.primaryMetricLabel}):     ${bestTokPerSec.toFixed(2)} tok/s`));
  console.log(c("1;37", `  Current (${effortSpec.primaryMetricLabel}):  ${currentCode.tokPerSec?.toFixed(2) ?? "?"} tok/s`));
  if (bestTokPerSec > (originalBaseline.tokPerSec ?? 0)) {
    const gain = ((bestTokPerSec - (originalBaseline.tokPerSec ?? 0)) / (originalBaseline.tokPerSec ?? 1) * 100).toFixed(1);
    console.log(c("1;32", `  Gain:     +${gain}%`));
  }
  console.log(c("1;37", `  Stall:    ${state.stalledCycles} cycles`));
  console.log(c("1;37", `  State:    ${statePathForEffort(effort)}`));
  console.log(c("1;37", `${"═".repeat(58)}\n`));
}

// Only run main when executed directly, not when imported by tests
const isMainModule = typeof Bun !== "undefined"
  ? Bun.main === import.meta.path
  : !process.argv[1]?.includes(".test.");

if (isMainModule) {
  main().catch((e) => {
    console.error(c("1;31", `Fatal: ${e}`));
    process.exit(1);
  });
}
