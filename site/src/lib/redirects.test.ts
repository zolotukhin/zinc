import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

type Redirect = {
  from: string;
  to: string;
};

const redirectsPath = resolve(import.meta.dir, '../../public/_redirects');

function loadRedirects(): Redirect[] {
  return readFileSync(redirectsPath, 'utf8')
    .split('\n')
    .map(line => line.trim())
    .filter(line => line && !line.startsWith('#'))
    .map(line => {
      const [from, to] = line.split(/\s+/);
      return { from, to };
    });
}

function matchRedirect(path: string, rule: Redirect): string | null {
  if (!rule.from.includes('*')) {
    return path === rule.from ? rule.to : null;
  }

  const [prefix, suffix = ''] = rule.from.split('*');
  if (!path.startsWith(prefix) || !path.endsWith(suffix)) return null;

  const splat = path.slice(prefix.length, path.length - suffix.length || undefined);
  return rule.to.replace(':splat', splat);
}

function resolveRedirect(path: string): string | null {
  for (const rule of loadRedirects()) {
    const destination = matchRedirect(path, rule);
    if (destination) return destination;
  }
  return null;
}

describe('legacy search-index redirects', () => {
  const expected = new Map<string, string>([
    ['/zinc/docs/rdna4-batched-prefill-2x/', '/blog/2026-06-05-how-zinc-rdna4-batched-prefill-went-from-42-to-208-tok-s/'],
    ['/zinc/docs/TURBOQUANT_SPEC/', '/zinc/docs/turboquant-spec/'],
    ['/zinc/docs/rdna4-performance-journey/', '/blog/2026-05-09-how-we-made-amd-qwen-inference-faster-than-llama-cpp-in-six-weeks-on-the-radeon-ai-pro-r9700/'],
    ['/zinc/docs/src/server/model_manager_metal.zig', 'https://github.com/zolotukhin/zinc/blob/main/src/server/model_manager_metal.zig'],
    ['/zinc/docs/build.zig', 'https://github.com/zolotukhin/zinc/blob/main/build.zig'],
    ['/zinc/docs/src/model/loader_metal.zig', 'https://github.com/zolotukhin/zinc/blob/main/src/model/loader_metal.zig'],
    ['/zinc/docs/src/server/runtime.zig', 'https://github.com/zolotukhin/zinc/blob/main/src/server/runtime.zig'],
    ['/zinc/docs/src/compute/forward_metal.zig', 'https://github.com/zolotukhin/zinc/blob/main/src/compute/forward_metal.zig'],
    ['/zinc/docs/apple-silicon-metal-enablement/APPLE_METAL_REFERENCE.md', '/zinc/docs/apple-metal-reference/'],
    ['/cdn-cgi/l/email-protection', '/about/'],
    ['/zinc/docs/src/metal/buffer.zig', 'https://github.com/zolotukhin/zinc/blob/main/src/metal/buffer.zig'],
    ['/zinc/docs/src/metal/device.zig', 'https://github.com/zolotukhin/zinc/blob/main/src/metal/device.zig'],
    ['/zinc/docs/benchmarks/metal_inference.zig', 'https://github.com/zolotukhin/zinc/blob/main/benchmarks/metal_inference.zig'],
    ['/zinc/docs/rdna4-performance-plan/', '/zinc/docs/rdna4-tuning/'],
    ['/zinc/docs/performance-gap-analysis/', '/zinc/benchmarks/'],
    ['/zinc/docs/decode-throughput-plan/', '/zinc/benchmarks/'],
    ['/zinc/docs/qwen35-debug/', '/zinc/docs/'],
    ['/zinc/docs/zig-api/session/', '/zinc/docs/zig-api/'],
    ['/zinc/docs/src/metal/pipeline.zig', 'https://github.com/zolotukhin/zinc/blob/main/src/metal/pipeline.zig'],
    ['/zinc/docs/src/metal/shim.h', 'https://github.com/zolotukhin/zinc/blob/main/src/metal/shim.h'],
    ['/zinc/docs/apple-silicon-metal-enablement/APPLE_SILICON_REFERENCE.md', '/zinc/docs/apple-silicon-reference/'],
    ['/zinc/docs/tools/benchmark_api.mjs', 'https://github.com/zolotukhin/zinc/blob/main/tools/benchmark_api.mjs'],
  ]);

  it('maps every Search Console 404 to its closest live replacement', () => {
    for (const [path, destination] of expected) {
      expect(resolveRedirect(path), path).toBe(destination);
    }
  });

  it('keeps obsolete uppercase doc links out of current content', () => {
    const post = readFileSync(
      resolve(import.meta.dir, '../content/posts/2026-03-25-why-we-are-building-zinc.md'),
      'utf8'
    );

    expect(post).not.toContain('/zinc/docs/TURBOQUANT_SPEC/');
    expect(post).not.toContain('/zinc/docs/RDNA4_TUNING/');
  });
});
