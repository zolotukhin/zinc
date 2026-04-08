# Optimization 1: Push Descriptors (VK_KHR_push_descriptor)

## Current State (2026-04-05)

- **Qwen3.5-35B-A3B**: 62.4 tok/s (16.0 ms/tok) on RX 9070 RDNA4
- **CPU record time**: 0.55 ms/tok (3.4% of total)
- **Descriptor allocs**: 1022 per token, 1022 writes, 3266 bindings
- **Target**: Reduce CPU overhead from 0.55 ms to ~0.1 ms

## Why

Every dispatch in the decode loop calls `allocDescSet → writeDescSet → vkCmdBindDescriptorSets`. With 1022 allocations per token:
- `vkAllocateDescriptorSets`: ~0.3 µs × 1022 = ~0.3 ms
- `vkUpdateDescriptorSets`: ~0.2 µs × 1022 = ~0.2 ms

Push descriptors write buffer bindings directly into the command buffer via `vkCmdPushDescriptorSetKHR`, bypassing the pool allocator. On RADV, this eliminates per-allocation bookkeeping and pool fragmentation.

**Expected savings**: 0.55 ms → ~0.15 ms CPU record time. At 62.4 tok/s (16.0 ms/tok), this saves ~0.4 ms → **~64.0 tok/s (+2.5%)**.

## GPU Support (verified)

```
VK_KHR_push_descriptor : extension revision 2  (RDNA4, RADV)
```

## Detailed Steps

### Step 1: Load `vkCmdPushDescriptorSetKHR` function pointer

**File: `src/vulkan/instance.zig`**

Add to the Instance struct:

```zig
push_descriptor_fn: ?*const fn (
    vk.c.VkCommandBuffer,
    vk.c.VkPipelineBindPoint,
    vk.c.VkPipelineLayout,
    u32,
    u32,
    [*]const vk.c.VkWriteDescriptorSet,
) callconv(.C) void,
```

Load after device creation:

```zig
self.push_descriptor_fn = @ptrCast(vk.c.vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"));
if (self.push_descriptor_fn != null) {
    log.info("VK_KHR_push_descriptor available", .{});
}
```

**Build check**: `zig build test` must still pass. No behavior change yet.

### Step 2: Enable push descriptor flag in pipeline layouts

**File: `src/vulkan/pipeline.zig`**

In `createDescriptorSetLayout` (or wherever `VkDescriptorSetLayoutCreateInfo` is built):

```zig
// BEFORE:
.flags = 0,

// AFTER:
.flags = if (instance.push_descriptor_fn != null)
    vk.c.VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR
else
    0,
```

**CRITICAL**: Once this flag is set, `vkAllocateDescriptorSets` will FAIL for these layouts. Step 3 MUST be done in the same commit/cycle, or the code breaks.

**Strategy**: Do NOT apply this flag yet. First implement the push dispatch path (Step 3), then flip the flag.

### Step 3: Add `pushDescAndDispatch` to CommandBuffer

**File: `src/vulkan/command.zig`**

```zig
/// Dispatch using push descriptors — no descriptor set allocation needed.
pub fn pushDescAndDispatch(
    self: *const CommandBuffer,
    pipeline: *const Pipeline,
    push_desc_fn: @TypeOf(Instance.push_descriptor_fn),
    buffer_infos: []const vk.c.VkDescriptorBufferInfo,
    push_data: []const u8,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
) void {
    vk.c.vkCmdBindPipeline(self.handle, vk.c.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    if (push_data.len > 0) {
        vk.c.vkCmdPushConstants(self.handle, pipeline.pipeline_layout,
            vk.c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push_data.len), push_data.ptr);
    }

    // Build VkWriteDescriptorSet array on the stack
    var writes: [8]vk.c.VkWriteDescriptorSet = undefined;
    for (buffer_infos, 0..) |_, i| {
        writes[i] = .{
            .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = null,
            .dstSet = null, // push descriptors don't use a set handle
            .dstBinding = @intCast(i),
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = null,
            .pBufferInfo = &buffer_infos[i],
            .pTexelBufferView = null,
        };
    }

    push_desc_fn.?(
        self.handle,
        vk.c.VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipeline_layout,
        0, // set index
        @intCast(buffer_infos.len),
        &writes,
    );

    vk.c.vkCmdDispatch(self.handle, group_count_x, group_count_y, group_count_z);
}
```

**Build check**: `zig build test` — no call sites yet, just the new function.

### Step 4: Add push-descriptor dispatch helpers to InferenceEngine

**File: `src/compute/forward.zig`**

Add helpers that mirror the existing `writeDescSet*` + dispatch pattern:

```zig
fn pushDispatch3(
    self: *InferenceEngine,
    pip: *const Pipeline,
    push_data: []const u8,
    buf0: vk.c.VkBuffer, size0: vk.c.VkDeviceSize,
    buf1: vk.c.VkBuffer, size1: vk.c.VkDeviceSize,
    buf2: vk.c.VkBuffer, size2: vk.c.VkDeviceSize,
    wg_x: u32, wg_y: u32, wg_z: u32,
) void {
    const infos = [3]vk.c.VkDescriptorBufferInfo{
        .{ .buffer = buf0, .offset = 0, .range = size0 },
        .{ .buffer = buf1, .offset = 0, .range = size1 },
        .{ .buffer = buf2, .offset = 0, .range = size2 },
    };
    self.decode_cmd.pushDescAndDispatch(pip, self.instance.push_descriptor_fn,
        &infos, push_data, wg_x, wg_y, wg_z);
}

// Similarly: pushDispatch2, pushDispatch4, pushDispatch5, pushDispatch7
```

**Build check**: `zig build test` — no call sites changed yet.

### Step 5: Convert call sites incrementally

**File: `src/compute/forward.zig`** — 1022 descriptor allocs across ~60 call sites.

Convert in batches of ~10 call sites per cycle, building after each batch:

**Batch A: Elementwise dispatches (RMS norm, scale_acc, swiglu, sigmoid_mul)**

Find all `allocDescSet + writeDescSet3 + recordRmsNorm/recordSwiglu/recordScaleAcc` patterns and replace:

```zig
// BEFORE:
const ds = try self.allocDescSet(pip.descriptor_set_layout);
self.writeDescSet3(ds, buf0, size0, buf1, size1, buf2, size2);
try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, hidden_dim, 1, eps);

// AFTER:
self.pushDispatch3(pip, std.mem.asBytes(&push), buf0, size0, buf1, size1, buf2, size2, 1, 1, 1);
```

Note: `recordRmsNorm` calls `dispatchWithPush(pip, ds, ...)` internally. We need to either:
- (a) Extract the push constant construction from `recordRmsNorm` and call `pushDispatch3` directly, or
- (b) Add `recordRmsNormPush(cmd, push_data, infos)` variants to `elementwise.zig`

**Recommended approach (a)**: Replace the dispatch call inline. This avoids changing the elementwise API.

**Batch B: DMMV dispatches (dispatchDmmv)**

The `dispatchDmmv` function in forward.zig calls `self.dmmv.recordDispatch(...)` which internally does `cmd.dispatchWithPush(pip, ds, ...)`. Convert `recordDispatch` to accept push descriptor parameters.

**Batch C: MoE dispatches**

Convert `recordMoeDispatch`, `recordSoftmaxTopk`, `recordMoeWeightedAcc`.

**Batch D: Flash attention, SSM, RoPE**

These have more complex binding patterns (5-7 bindings).

**Build after EACH batch.** Run correctness test after each batch.

### Step 6: Enable push descriptor layout flag

**File: `src/vulkan/pipeline.zig`**

Once ALL call sites are converted (no more `allocDescSet` calls remain):

```zig
.flags = vk.c.VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
```

### Step 7: Remove old descriptor pool infrastructure

- Remove `shared_pool` field from InferenceEngine
- Remove `vkCreateDescriptorPool` in init
- Remove all `vkResetDescriptorPool` calls (search: 24 occurrences)
- Remove `allocDescSet` function
- Remove `writeDescSet2/3/4/5` functions
- Keep argmax's dedicated pool (1 static set, negligible overhead)

### Step 8: Verify with profiling

```bash
# On RDNA node:
./zig-out/bin/zinc -m /root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
    --prompt 'Write a detailed essay.' --profile -n 200

# Check:
# - avg CPU record should drop from 0.55 ms to ~0.15 ms
# - avg descriptor allocs should show 0 (push descriptors bypass pool)
# - tok/s should improve ~2-3%
```

## Call Site Inventory

Search `self.allocDescSet` in forward.zig to find all sites. Current count: ~60 unique call sites.

Major categories:
- RMS norm dispatches: ~12 sites (pre-attn norm, FFN norm, post-norms, final norm)
- DMMV dispatches (via dispatchDmmv): ~15 sites (Q/K/V/O proj, FFN gate/up/down, SSM proj)
- MoE dispatches: ~8 sites (router, topk, gate/up/down MoE, weighted_acc)
- Shared expert: ~6 sites (gate/up/gate_scalar, swiglu, down, gate_acc)
- Activation: ~4 sites (swiglu, geglu, sigmoid_mul)
- Flash attention: ~2 sites
- SSM: ~8 sites (conv1d, delta-net, gated_norm)
- RoPE: ~2 sites
- Residual/scale: ~6 sites

## Models to Test

| Model | Prompt | Expected |
|-------|--------|----------|
| Qwen3.5-35B | "The capital of France is" | "Paris." |
| Qwen3-8B | "The capital of France is" | "Paris." |
| Gemma3-12B | "The capital of France is" | "Paris." |

## Risk

- **Low correctness risk**: Push descriptors are semantically identical to pool-allocated sets.
- **Medium refactor risk**: 60 call sites. Convert incrementally with builds between batches.
- **Driver compatibility**: VK_KHR_push_descriptor is widely supported. RADV has had it since Mesa 21.0.
