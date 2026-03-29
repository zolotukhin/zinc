# Contract: GPU Abstraction Layer

The `src/gpu/` module provides a comptime-resolved abstraction that `compute/` and `model/` use instead of calling Vulkan or Metal directly.

## Interface

```zig
const builtin = @import("builtin");

/// The concrete backend type, resolved at comptime.
/// On macOS → MetalDevice, on Linux → VulkanInstance.
pub const Backend = if (builtin.os.tag == .macos)
    @import("../metal/device.zig").MetalDevice
else
    @import("../vulkan/instance.zig").VulkanInstance;

/// Unified buffer handle.
pub const Buffer = if (builtin.os.tag == .macos)
    @import("../metal/buffer.zig").MetalBuffer
else
    @import("../vulkan/buffer.zig").Buffer;

/// Unified pipeline handle.
pub const Pipeline = if (builtin.os.tag == .macos)
    @import("../metal/pipeline.zig").MetalPipeline
else
    @import("../vulkan/pipeline.zig").Pipeline;
```

## Required Methods on Each Backend

Both `MetalDevice` and `VulkanInstance` must implement these methods (duck-typed via comptime):

### Buffer Creation
```zig
fn createBuffer(self: *@This(), size: usize) Buffer
fn createBufferMapped(self: *@This(), size: usize) struct { buf: Buffer, ptr: [*]u8 }
fn wrapMmap(self: *@This(), ptr: [*]u8, size: usize) Buffer
fn freeBuffer(self: *@This(), buf: Buffer) void
```

### Pipeline Creation
```zig
fn createPipeline(self: *@This(), shader_name: []const u8) Pipeline
fn freePipeline(self: *@This(), pipe: Pipeline) void
```

### Command Recording & Dispatch
```zig
fn beginCommand(self: *@This()) void
fn dispatch(self: *@This(), pipe: Pipeline, grid: [3]u32, block: [3]u32, push: anytype, bufs: []const Buffer) void
fn barrier(self: *@This()) void
fn commitAndWait(self: *@This()) void
```

### Device Queries
```zig
fn maxBufferSize(self: *@This()) usize
fn totalMemory(self: *@This()) u64
```

## Vulkan Backend Notes

The existing Vulkan code exposes different method signatures. The refactor wraps existing methods to match the interface above:

- `wrapMmap` on Vulkan → staging buffer upload (not zero-copy)
- `barrier` on Vulkan → `vkCmdPipelineBarrier` with compute→compute dependency
- `dispatch` push constants → `vkCmdPushConstants` (Vulkan) vs buffer index N (Metal)

## Invariants

1. Only one backend is ever compiled. No runtime dispatch overhead.
2. `compute/forward.zig` must work identically on both backends — same token output for same model + prompt.
3. The abstraction must not add any allocation, indirection, or function pointer overhead vs direct calls.
4. Vulkan path must produce identical binary to pre-refactor (verify with `zig build test` on Linux).
