# Optimization 1: Push Descriptors (VK_KHR_push_descriptor)

## Why

Every DMMV, norm, attention, and activation dispatch allocates a descriptor set from a Vulkan pool via `vkAllocateDescriptorSets`. Profiling shows **914 allocations per token** taking **7.27ms of CPU recording time** — that's **4.3% of total decode time** (168ms).

Push descriptors write buffer bindings directly into the command buffer, bypassing the pool allocator entirely. On RADV (Mesa AMD driver), `vkAllocateDescriptorSets` costs ~8us per call. Push descriptors cost ~1-2us (just a memcpy into the command buffer).

Expected savings: 914 × 6us = **~5.5ms per token** → from 168ms to ~163ms → **3.0% decode speedup** (6.05 → 6.23 tok/s for Gemma 3 12B).

## What

Replace all `allocDescSet + writeDescSet* + vkCmdBindDescriptorSets` with `vkCmdPushDescriptorSetKHR`. This is a plumbing change — no shader modifications, no algorithm changes.

### GPU support (verified)
```
VK_KHR_push_descriptor : extension revision 2  (RDNA4)
```

## How

### Step 1: Enable push descriptor layouts in pipeline creation

**File: `src/vulkan/pipeline.zig:103-109`**

Change the descriptor set layout creation to use the push descriptor flag:
```zig
// BEFORE:
.flags = 0,

// AFTER:
.flags = vk.c.VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
```

**IMPORTANT**: With this flag, `vkAllocateDescriptorSets` will FAIL for these layouts. The push descriptor dispatch path MUST be implemented before this change goes live.

### Step 2: Load `vkCmdPushDescriptorSetKHR` function pointer

**File: `src/vulkan/instance.zig` or `src/vulkan/command.zig`**

Load the function pointer at device creation time:
```zig
const vkCmdPushDescriptorSetKHR = @as(
    ?*const fn(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, u32, u32, [*]const VkWriteDescriptorSet) callconv(.C) void,
    @ptrCast(vk.c.vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR"))
);
```

Store this in the Instance or InferenceEngine struct for reuse.

### Step 3: Add `pushAndDispatch` helper to CommandBuffer

**File: `src/vulkan/command.zig`** — add after `dispatchWithPush` (line 236):

```zig
pub fn pushAndDispatch(
    self: *const CommandBuffer,
    pipeline: *const Pipeline,
    writes: []const vk.c.VkWriteDescriptorSet,
    push_data: []const u8,
    group_count_x: u32, group_count_y: u32, group_count_z: u32,
    push_desc_fn: PushDescFnType,
) void {
    vk.c.vkCmdBindPipeline(self.handle, vk.c.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
    vk.c.vkCmdPushConstants(self.handle, pipeline.pipeline_layout,
        vk.c.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push_data.len), push_data.ptr);
    push_desc_fn(self.handle, vk.c.VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.pipeline_layout, 0, @intCast(writes.len), writes.ptr);
    vk.c.vkCmdDispatch(self.handle, group_count_x, group_count_y, group_count_z);
}
```

### Step 4: Replace allocDescSet + writeDescSet + dispatch pattern

**File: `src/compute/forward.zig`** — 45 call sites need updating.

**BEFORE (typical pattern, repeated 45× across the file):**
```zig
const ds = try self.allocDescSet(pip.descriptor_set_layout);
self.writeDescSet3(ds, buf0.handle, buf0.size, buf1.handle, buf1.size, buf2.handle, buf2.size);
self.decode_cmd.dispatchWithPush(pip, ds, push_data, wg_x, 1, 1);
```

**AFTER:**
```zig
var infos = [3]vk.c.VkDescriptorBufferInfo{
    .{ .buffer = buf0.handle, .offset = 0, .range = buf0.size },
    .{ .buffer = buf1.handle, .offset = 0, .range = buf1.size },
    .{ .buffer = buf2.handle, .offset = 0, .range = buf2.size },
};
var writes: [3]vk.c.VkWriteDescriptorSet = undefined;
for (0..3) |i| {
    writes[i] = .{
        .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = null, .dstSet = null, .dstBinding = @intCast(i),
        .dstArrayElement = 0, .descriptorCount = 1,
        .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = null, .pBufferInfo = &infos[i], .pTexelBufferView = null,
    };
}
self.decode_cmd.pushAndDispatch(&pip, &writes, push_data, wg_x, 1, 1, self.push_desc_fn);
```

**To reduce verbosity**: Create helper functions `makePushWrites3`, `makePushWrites5` that build the write arrays from buffer handles/sizes. Then each call site becomes:
```zig
var infos: [3]vk.c.VkDescriptorBufferInfo = undefined;
var writes: [3]vk.c.VkWriteDescriptorSet = undefined;
makePushWrites3(&infos, &writes, buf0, size0, buf1, size1, buf2, size2);
self.decode_cmd.pushAndDispatch(&pip, &writes, push_data, wg_x, 1, 1, self.push_desc_fn);
```

### Step 5: Remove descriptor pool infrastructure

Once all 45 call sites are converted:
- Remove `shared_pool` field from InferenceEngine
- Remove `vkCreateDescriptorPool` in init (lines 1008-1014)
- Remove all 24 `vkResetDescriptorPool` calls
- Remove `allocDescSet` function
- Remove `writeDescSet2/3/4/5` functions (replaced by `makePushWrites*`)

### Step 6: Handle special cases

- **Argmax descriptor set** (`src/compute/argmax.zig:86-99`): Has its own pool. Either convert to push descriptors or keep its dedicated pool (only 1 set, minimal overhead).
- **MoE DMMV** (`src/compute/dmmv.zig`): Has its own descriptor pool. Same treatment as argmax.
- **Debug readback blocks** (lines 1882, 2100, etc.): These reset the shared_pool. With push descriptors, these resets become no-ops (remove them).

## Call sites to update (complete list)

| Location | Function | Bindings | Notes |
|----------|----------|----------|-------|
| forward.zig:1615 | allocDescSet+writeDescSet3 | 3 | Attn input norm |
| forward.zig:1694 | allocDescSet+writeDescSet3 | 3 | V bare norm |
| forward.zig:1779 | allocDescSet+writeDescSet3 | 3 | Q norm |
| forward.zig:1785 | allocDescSet+writeDescSet3 | 3 | K norm |
| forward.zig:1802-1806 | allocDescSet+writeDescSet3 | 3 | RoPE Q+K |
| forward.zig:1826 | allocDescSet+writeDescSet5 | 5 | Flash attention |
| forward.zig:1999 | allocDescSet+writeDescSet3 | 3 | Sigmoid mul gate |
| forward.zig:2018-2027 | allocDescSet+writeDescSet3/2 | 3,2 | Post-attn norm + residual |
| forward.zig:2133 | allocDescSet+writeDescSet3 | 3 | FFN norm |
| forward.zig:2481-2499 | allocDescSet+writeDescSet3 | 3×3 | Gate+Up+Activation+Down |
| forward.zig:2625-2656 | allocDescSet+writeDescSet3/2 | 3,2,3 | Post-FFN norm + residual + scale |
| forward.zig:2811-2829 | allocDescSet+writeDescSet3 | 3 | Final norm + LM head |
| + ~10 more in MoE/SSM/shared expert paths | | | |

## Testing

1. **Correctness**: Run Gemma 3 `--prompt 'What is the capital of France?' --chat -n 16` → must output "The capital of France is **Paris**."
2. **Performance**: Profile with `--profile` → check `avg CPU record` drops from ~7.3ms to ~2ms
3. **Gemma 4**: Same test with Gemma 4 31B to verify softcap + IDP paths still work
4. **Benchmark**: Compare tok/s before/after across 3 runs (expect ~3% improvement)

## Risk

- **Low correctness risk**: Push descriptors are semantically identical to pool-allocated sets — same buffer bindings, same shader access. No compute logic changes.
- **Medium refactor risk**: 45 call sites across a 4000-line file. Easy to miss one or get a binding index wrong. Systematic search for `allocDescSet` catches all sites.
- **Driver compatibility**: VK_KHR_push_descriptor is core in Vulkan 1.3. All target GPUs support it.
