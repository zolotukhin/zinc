// Vulkan dispatch overhead micro-benchmark for RDNA4
// Measures: empty dispatch, small compute, pipeline bind overhead
// Build: gcc -O2 dispatch_bench.c -o dispatch_bench -lvulkan -lm
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CHECK_VK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "Vulkan error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); } } while(0)

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// Minimal compute shader: just writes threadID to buffer
static const uint32_t shader_code[] = {
    // SPIR-V generated from:
    // #version 450
    // layout(local_size_x = 64) in;
    // layout(binding = 0) buffer B { uint data[]; };
    // void main() { data[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x; }
    0x07230203, 0x00010000, 0x00080001, 0x0000001e,
    0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0006000f, 0x00000005, 0x00000004, 0x6e69616d,
    0x00000000, 0x0000000b, 0x00060010, 0x00000004,
    0x00000011, 0x00000040, 0x00000001, 0x00000001,
    0x00030003, 0x00000002, 0x000001c2, 0x00040005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00080005,
    0x00000009, 0x475f6c67, 0x61626f6c, 0x766e496c,
    0x7461636f, 0x496e6f69, 0x00000044, 0x00030005,
    0x0000000b, 0x00000000, 0x00040005, 0x0000000d,
    0x61746164, 0x00000000, 0x00030005, 0x0000000e,
    0x00004200, 0x00040047, 0x00000009, 0x0000000b,
    0x0000001c, 0x00040047, 0x0000000b, 0x0000000b,
    0x0000001c, 0x00050048, 0x0000000e, 0x00000000,
    0x00000023, 0x00000000, 0x00030047, 0x0000000e,
    0x00000003, 0x00040047, 0x0000000d, 0x00000022,
    0x00000000, 0x00040047, 0x0000000d, 0x00000021,
    0x00000000, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00040015, 0x00000006,
    0x00000020, 0x00000000, 0x00040017, 0x00000007,
    0x00000006, 0x00000003, 0x00040020, 0x00000008,
    0x00000001, 0x00000007, 0x0004003b, 0x00000008,
    0x00000009, 0x00000001, 0x00040020, 0x0000000a,
    0x00000001, 0x00000006, 0x0004003b, 0x0000000a,
    0x0000000b, 0x00000001, 0x0003001d, 0x0000000c,
    0x00000006, 0x0003001e, 0x0000000e, 0x0000000c,
    0x00040020, 0x0000000f, 0x00000002, 0x0000000e,
    0x0004003b, 0x0000000f, 0x0000000d, 0x00000002,
    0x00040015, 0x00000010, 0x00000020, 0x00000001,
    0x0004002b, 0x00000010, 0x00000011, 0x00000000,
    0x00040020, 0x00000013, 0x00000002, 0x00000006,
    0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003d,
    0x00000006, 0x00000014, 0x0000000b, 0x00050041,
    0x00000013, 0x00000015, 0x0000000d, 0x00000011,
    // AccessChain + Store
    0x00060041, 0x00000013, 0x00000016, 0x0000000d,
    0x00000011, 0x00000014, 0x0003003e, 0x00000016,
    0x00000014, 0x000100fd, 0x00010038,
};

int main(int argc, char** argv) {
    int gpu_idx = argc > 1 ? atoi(argv[1]) : 1;  // default Vulkan1
    int n_dispatches = argc > 2 ? atoi(argv[2]) : 2000;
    int n_iters = argc > 3 ? atoi(argv[3]) : 10;

    // Create instance
    VkApplicationInfo app_info = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "dispatch_bench", 1, NULL, 0, VK_API_VERSION_1_2};
    VkInstanceCreateInfo inst_info = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app_info, 0, NULL, 0, NULL};
    VkInstance instance;
    CHECK_VK(vkCreateInstance(&inst_info, NULL, &instance));

    // Get physical device
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance, &dev_count, NULL);
    VkPhysicalDevice* devs = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &dev_count, devs);

    if ((uint32_t)gpu_idx >= dev_count) {
        fprintf(stderr, "GPU index %d out of range (have %d)\n", gpu_idx, dev_count);
        return 1;
    }
    VkPhysicalDevice pdev = devs[gpu_idx];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pdev, &props);
    printf("GPU: %s\n", props.deviceName);

    // Find compute queue family
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, NULL);
    VkQueueFamilyProperties* qf_props = malloc(qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qf_props);
    uint32_t compute_qf = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { compute_qf = i; break; }
    }

    // Create device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, compute_qf, 1, &priority};
    VkDeviceCreateInfo dev_create = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &queue_info, 0, NULL, 0, NULL, NULL};
    VkDevice device;
    CHECK_VK(vkCreateDevice(pdev, &dev_create, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, compute_qf, 0, &queue);

    // Create buffer (4KB)
    VkBufferCreateInfo buf_info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer buffer;
    CHECK_VK(vkCreateBuffer(device, &buf_info, NULL, &buffer));
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(device, buffer, &mem_req);
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);
    uint32_t mem_type = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mem_type = i; break;
        }
    }
    VkMemoryAllocateInfo alloc_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mem_req.size, mem_type};
    VkDeviceMemory memory;
    CHECK_VK(vkAllocateMemory(device, &alloc_info, NULL, &memory));
    CHECK_VK(vkBindBufferMemory(device, buffer, memory, 0));

    // Create shader module
    VkShaderModuleCreateInfo shader_info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof(shader_code), shader_code};
    VkShaderModule shader;
    CHECK_VK(vkCreateShaderModule(device, &shader_info, NULL, &shader));

    // Create pipeline
    VkDescriptorSetLayoutBinding binding = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo ds_layout_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &binding};
    VkDescriptorSetLayout ds_layout;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &ds_layout_info, NULL, &ds_layout));

    VkPipelineLayoutCreateInfo pl_layout_info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &ds_layout, 0, NULL};
    VkPipelineLayout pl_layout;
    CHECK_VK(vkCreatePipelineLayout(device, &pl_layout_info, NULL, &pl_layout));

    VkComputePipelineCreateInfo pipe_info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", NULL},
        pl_layout, VK_NULL_HANDLE, -1};
    VkPipeline pipeline;
    CHECK_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &pipeline));

    // Create descriptor pool and set
    VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &pool_size};
    VkDescriptorPool desc_pool;
    CHECK_VK(vkCreateDescriptorPool(device, &pool_info, NULL, &desc_pool));
    VkDescriptorSetAllocateInfo ds_alloc = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, desc_pool, 1, &ds_layout};
    VkDescriptorSet desc_set;
    CHECK_VK(vkAllocateDescriptorSets(device, &ds_alloc, &desc_set));
    VkDescriptorBufferInfo buf_desc = {buffer, 0, 4096};
    VkWriteDescriptorSet write_ds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, desc_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &buf_desc, NULL};
    vkUpdateDescriptorSets(device, 1, &write_ds, 0, NULL);

    // Create command pool
    VkCommandPoolCreateInfo cmd_pool_info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, compute_qf};
    VkCommandPool cmd_pool;
    CHECK_VK(vkCreateCommandPool(device, &cmd_pool_info, NULL, &cmd_pool));
    VkCommandBufferAllocateInfo cmd_alloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd;
    CHECK_VK(vkAllocateCommandBuffers(device, &cmd_alloc, &cmd));

    // Create fence and timestamp query pool
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
    VkFence fence;
    CHECK_VK(vkCreateFence(device, &fence_info, NULL, &fence));

    VkQueryPoolCreateInfo query_info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2, 0};
    VkQueryPool query_pool;
    CHECK_VK(vkCreateQueryPool(device, &query_info, NULL, &query_pool));

    float ts_period = props.limits.timestampPeriod; // ns per tick

    printf("Testing %d dispatches × %d iterations\n\n", n_dispatches, n_iters);

    // === Test 1: Record + Submit + Wait overhead (single dispatch) ===
    printf("=== Test 1: Record+Submit+Wait (1 dispatch) ===\n");
    for (int iter = 0; iter < n_iters; iter++) {
        double t0 = now_us();

        VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(cmd, &begin);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl_layout, 0, 1, &desc_set, 0, NULL);
        vkCmdDispatch(cmd, 1, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd, 0, NULL};
        vkResetFences(device, 1, &fence);
        vkQueueSubmit(queue, 1, &submit, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        double t1 = now_us();
        if (iter >= 2) printf("  iter %d: %.1f us\n", iter, t1 - t0);
    }

    // === Test 2: N dispatches in one command buffer ===
    printf("\n=== Test 2: %d dispatches in one cmd buffer ===\n", n_dispatches);
    for (int iter = 0; iter < n_iters; iter++) {
        double t0 = now_us();

        VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
        vkBeginCommandBuffer(cmd, &begin);
        vkCmdResetQueryPool(cmd, query_pool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl_layout, 0, 1, &desc_set, 0, NULL);
        for (int i = 0; i < n_dispatches; i++) {
            vkCmdDispatch(cmd, 1, 1, 1);
        }
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo sub = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd, 0, NULL};
        vkResetFences(device, 1, &fence);
        vkQueueSubmit(queue, 1, &sub, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

        uint64_t timestamps[2];
        vkGetQueryPoolResults(device, query_pool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double gpu_us = (timestamps[1] - timestamps[0]) * ts_period / 1000.0;
        double wall_us = now_us() - t0;

        if (iter >= 2)
            printf("  iter %d: wall=%.0f us, gpu=%.0f us, per-dispatch=%.2f us (gpu), cpu_overhead=%.0f us\n",
                   iter, wall_us, gpu_us, gpu_us / n_dispatches, wall_us - gpu_us);
    }

    // === Test 3: Replay pre-recorded command buffer ===
    printf("\n=== Test 3: Replay pre-recorded cmd buffer (%d dispatches) ===\n", n_dispatches);
    // Record once
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, 0, NULL};
    vkBeginCommandBuffer(cmd, &begin);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl_layout, 0, 1, &desc_set, 0, NULL);
    for (int i = 0; i < n_dispatches; i++) {
        vkCmdDispatch(cmd, 1, 1, 1);
    }
    vkEndCommandBuffer(cmd);

    for (int iter = 0; iter < n_iters; iter++) {
        double t0 = now_us();
        VkSubmitInfo sub = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd, 0, NULL};
        vkResetFences(device, 1, &fence);
        vkQueueSubmit(queue, 1, &sub, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        double t1 = now_us();
        if (iter >= 2) printf("  iter %d: %.1f us (submit+wait only, no recording)\n", iter, t1 - t0);
    }

    // Cleanup
    vkDestroyQueryPool(device, query_pool, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, cmd_pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, pl_layout, NULL);
    vkDestroyDescriptorPool(device, desc_pool, NULL);
    vkDestroyDescriptorSetLayout(device, ds_layout, NULL);
    vkDestroyShaderModule(device, shader, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    printf("\nDone.\n");
    return 0;
}
