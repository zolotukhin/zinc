import { describe, expect, it } from 'bun:test';
import { createZigApiAgentPayload, loadZigApi, parseZigModule, renderZigApiAgentText } from './zig-api-loader';

describe('parseZigModule', () => {
  it('extracts module docs, symbols, and nested methods', () => {
    const module = parseZigModule(
      `//! Example module summary.
//! @section Shader Dispatch
//! Example module overview paragraph.

const std = @import("std");

/// Example struct docs.
pub const Example = struct {
    value: usize,

    /// Initialize example state.
    /// @param value Value to store on the instance.
    /// @returns A new Example value.
    pub fn init(value: usize) Example {
        return .{ .value = value };
    }
};

/// Compute an output value.
/// @param input Value to double.
/// @returns The doubled input.
pub fn compute(
    input: usize,
) usize {
    return input * 2;
}
`,
      'compute/example.zig'
    );

    expect(module).not.toBeNull();
    expect(module?.section).toBe('Shader Dispatch');
    expect(module?.summary).toBe('Example module summary.');
    expect(module?.overview).toEqual(['Example module overview paragraph.']);
    expect(module?.symbols.map(symbol => symbol.name)).toEqual(['Example', 'compute']);
    expect(module?.codeLineCount).toBe(12);
    expect(module?.symbols[0]?.members).toHaveLength(1);
    expect(module?.symbols[0]?.members[0]?.qualifiedName).toBe('Example.init');
    expect(module?.symbols[0]?.members[0]?.doc.params[0]).toEqual({
      name: 'value',
      description: 'Value to store on the instance.',
    });
    expect(module?.symbols[1]?.doc.returns).toBe('The doubled input.');
  });

  it('ignores nested public declarations outside exported type methods', () => {
    const module = parseZigModule(
      `pub const Container = struct {
    pub const Nested = struct {};
    fn hidden() void {}
};`,
      'model/container.zig'
    );

    expect(module).not.toBeNull();
    expect(module?.symbols).toHaveLength(1);
    expect(module?.symbols[0]?.name).toBe('Container');
    expect(module?.symbols[0]?.members).toHaveLength(0);
  });
});

describe('loadZigApi', () => {
  it('loads grouped modules from the current repository', async () => {
    const api = await loadZigApi();

    expect(api.moduleCount).toBeGreaterThan(0);
    expect(api.codeLineCount).toBeGreaterThan(1000);
    expect(api.exportCount).toBeGreaterThan(10);
    expect(api.memberCount).toBeGreaterThan(10);
    expect(api.sections.some(section => section.title === 'Vulkan Runtime')).toBe(true);
    expect(api.modules.some(module => module.href === '/zinc/docs/zig-api/loader/')).toBe(true);
  }, 30000);

  it('gives every generated module and symbol a unique stable route', async () => {
    const api = await loadZigApi();
    const moduleHrefs = api.modules.map(module => module.href);

    expect(new Set(moduleHrefs).size).toBe(api.moduleCount);
    expect(api.modules.find(module => module.sourcePath === 'src/cuda/buffer.zig')?.href)
      .toBe('/zinc/docs/zig-api/buffer/');
    expect(api.modules.find(module => module.sourcePath === 'src/metal/buffer.zig')?.href)
      .toBe('/zinc/docs/zig-api/metal-buffer/');
    expect(api.modules.find(module => module.sourcePath === 'src/vulkan/buffer.zig')?.href)
      .toBe('/zinc/docs/zig-api/vulkan-buffer/');

    for (const module of api.modules) {
      expect(module.symbols.every(symbol => symbol.href.startsWith(`${module.href}#`))).toBe(true);
      expect(module.symbols.flatMap(symbol => symbol.members)
        .every(member => member.href.startsWith(`${module.href}#`))).toBe(true);
    }
  }, 30000);

  it('extracts struct layout metadata for rendered API docs', async () => {
    const api = await loadZigApi();
    const gguf = api.modules.find(module => module.slug === 'gguf');
    const tensorInfo = gguf?.symbols.find(symbol => symbol.qualifiedName === 'TensorInfo');

    expect(gguf).toBeDefined();
    expect(tensorInfo?.symbolKind).toBe('struct');
    // Struct layout data requires platform-specific compilation (Vulkan/Metal headers).
    // Skip size/field assertions when the struct analyzer couldn't run (e.g. CI on Linux).
    if (tensorInfo?.size != null) {
      expect(tensorInfo.size).toBeGreaterThan(0);
      expect(tensorInfo.alignment).toBeGreaterThan(0);
      expect(tensorInfo.fields?.length).toBeGreaterThan(0);
      expect(tensorInfo.fields?.some(field => field.name === 'dims')).toBe(true);
    }
  }, 30000);

  it('skips struct layout extraction for files that import build-system modules', async () => {
    // forward_zinc_rt.zig and the cpu_zig kernels import `gguf` via the build
    // graph, so `zig run` against the standalone analyzer cannot resolve them.
    // The loader computes the transitive @import closure of files reaching a
    // tainted named module and excludes them from struct-layout extraction.
    // The exact set may grow as new gguf-using files land; this test just
    // verifies the known-tainted entrypoints stay properly excluded and that
    // their per-file docs still render.
    const api = await loadZigApi();
    const knownTaintedSeeds = [
      'src/compute/forward_zinc_rt.zig',
      'src/zinc_rt/isa/cpu_zig/dequant.zig',
      'src/zinc_rt/isa/cpu_zig/embed.zig',
      'src/zinc_rt/isa/cpu_zig/lm_head.zig',
      'src/zinc_rt/isa/cpu_zig/matvec.zig',
      'src/zinc_rt/isa/cpu_zig/moe_gate_topk.zig',
    ];
    for (const path of knownTaintedSeeds) {
      const module = api.modules.find(mod => mod.sourcePath === path);
      expect(module, `module ${path} missing from loaded API`).toBeDefined();
      const summary = module?.summary ?? '';
      expect(summary.length).toBeGreaterThan(0);
      for (const sym of module?.symbols ?? []) {
        if (sym.symbolKind === 'struct') {
          expect(sym.size, `struct ${path}::${sym.qualifiedName} should have no extracted size`).toBeUndefined();
        }
      }
    }
  }, 30000);

  it('skips struct layout extraction for the CUDA backend subtree', async () => {
    // src/cuda/c.zig does `@cImport(@cInclude("cuda_shim.h"))`; the CUDA toolkit
    // headers only exist on the NVIDIA build box, so the standalone analyzer
    // cannot compile the cuda subtree off-box. The loader excludes `src/cuda/*`
    // from struct-layout extraction directly (not via the import BFS — cuda
    // imports elsewhere sit behind a comptime-dead `if (is_cuda)` branch). The
    // per-file docs still render; only the @sizeOf/@offsetOf numbers are skipped.
    const api = await loadZigApi();
    const cudaModules = api.modules.filter(mod => mod.sourcePath.startsWith('src/cuda/'));
    expect(cudaModules.length, 'expected CUDA backend modules in the API').toBeGreaterThan(0);
    for (const module of cudaModules) {
      expect(
        (module.summary ?? '').length,
        `cuda module ${module.sourcePath} should still render docs`
      ).toBeGreaterThan(0);
      for (const sym of module.symbols) {
        if (sym.symbolKind === 'struct') {
          expect(
            sym.size,
            `struct ${module.sourcePath}::${sym.qualifiedName} should have no extracted size`
          ).toBeUndefined();
        }
      }
    }
    // The backend abstraction textually imports cuda behind a comptime-dead branch
    // and must stay analyzable — it must not be swept up by the cuda exclusion.
    const iface = api.modules.find(mod => mod.sourcePath === 'src/gpu/interface.zig');
    expect(iface, 'gpu/interface.zig should remain in the analyzable set').toBeDefined();
  }, 30000);

  it('registers every section used by source files in SECTION_META', async () => {
    // Catches the regression where a file declares `//! @section Foo` but the
    // loader has no entry for `foo` and falls back to an auto-generated meta
    // with description `${title} modules and helpers.`.
    const api = await loadZigApi();
    const autoGenerated = api.sections.filter(section =>
      section.description === `${section.title} modules and helpers.`
    );
    expect(
      autoGenerated.map(section => section.title),
      `unregistered @section values: ${autoGenerated.map(s => s.title).join(', ')}`
    ).toEqual([]);
  }, 30000);

  it('serializes agent-friendly JSON and text exports', async () => {
    const api = await loadZigApi();
    const payload = createZigApiAgentPayload(api, 'https://zolotukhin.ai/');
    const text = renderZigApiAgentText(api, 'https://zolotukhin.ai/');

    expect(payload.root_url).toBe('https://zolotukhin.ai/zinc/docs/zig-api');
    expect(payload.json_url).toBe('https://zolotukhin.ai/zinc/docs/zig-api.json');
    expect(payload.counts.code_lines).toBeGreaterThan(1000);
    expect(payload.sections.length).toBeGreaterThan(0);
    expect(payload.sections.some(section => section.modules.some(module => module.symbols.length > 0))).toBe(true);
    expect(text).toContain('# ZINC Zig API');
    expect(text).toContain('JSON export: https://zolotukhin.ai/zinc/docs/zig-api.json');
    expect(text).toContain('Zig code lines');
    expect(text).toContain('Guidance: Use the generated Zig API as the canonical internal runtime reference.');
  }, 30000);

  it('keeps module-level Zig API docs populated for exported modules', async () => {
    const api = await loadZigApi();
    const missingOverview = api.modules.filter(module => module.overview.length === 0);
    const genericSummary = api.modules.filter(module => module.summary.startsWith('Public API surface for '));

    expect(missingOverview.map(module => module.sourcePath)).toEqual([]);
    expect(genericSummary.map(module => module.sourcePath)).toEqual([]);
  }, 30000);
});
