---
name: zig-docgen
description: Use when adding or editing Zig code in ZINC and you want it to remain compatible with the auto-generated Zig API docs. Covers the comment format, supported tags, parser limitations, section naming, and validation steps needed for reliable HTML, JSON, text, and llms exports.
---

# Zig Docgen

Use this when you touch `src/**/*.zig` and the public surface should stay extractable by `site/src/lib/zig-api-loader.ts`.

## Authoring Contract

1. Every documentable module starts with `//!` comments.
   Required shape:
   - first line: one-sentence summary
   - one `//! @section ...` line
   - one or more overview lines

2. Every public symbol gets a `///` doc block immediately above it.
   Include:
   - a one-sentence summary first
   - extra description lines after that if needed
   - `@param name ...` for meaningful parameters
   - `@returns ...` when the return value matters
   - `@note ...` for operational caveats

3. Every `pub fn` inside an exported `struct`, `enum`, or `union` also gets a `///` doc block.

4. Keep public API extraction simple.
   The generator supports:
   - top-level `pub const`, `pub fn`, `pub var`
   - `pub fn` methods inside exported containers

5. Do not rely on nested public declarations other than methods.
   Example:
   - `pub const Nested = struct {}` inside another exported `struct` is not part of the generated docs today.
   - If it should be documented, promote it to top level or extend the loader and tests.

6. Reuse the existing section names unless you are intentionally changing the docs taxonomy.
   Current sections:
   - `CLI & Entrypoints`
   - `Model Format & Loading`
   - `Tokenization`
   - `Decode Planning`
   - `Inference Runtime`
   - `Shader Dispatch`
   - `Hardware Detection`
   - `Vulkan Runtime`

7. If you add a new section name or a new public declaration pattern, update `site/src/lib/zig-api-loader.ts` and its tests in `site/src/lib/zig-api-loader.test.ts` in the same change.

8. Keep raw bindings and generated glue out of the authored API surface.
   `src/vulkan/vk.zig` is intentionally excluded from generated docs.

## Good Pattern

```zig
//! Create reusable compute command pools and command buffers.
//! @section Vulkan Runtime
//! The decode runtime uses these wrappers to record dispatches and synchronize compute work.

/// Command pool for allocating command buffers.
pub const CommandPool = struct {
    /// Create a command pool bound to the selected compute queue family.
    /// @param instance Active Vulkan instance and logical device.
    /// @returns A CommandPool ready to allocate compute command buffers.
    pub fn init(instance: *const Instance) !CommandPool {
        // ...
    }
};
```

## Editing Checklist

- Add or update the module `//!` block.
- Add `///` docs for every changed public symbol.
- Add `///` docs for every changed public method inside exported containers.
- Prefer stable, descriptive names because anchors come from symbol names.
- If the code shape stops matching the extractor, update the loader instead of hand-waving it.

## Validation

Run these after doc-related Zig changes:

```bash
zig build test
cd site && bun test
cd site && npm run build
```

Success means:
- no undocumented public symbols in the generated Zig API surface
- HTML docs still build
- JSON, text, and llms exports stay in sync with the Zig source comments
