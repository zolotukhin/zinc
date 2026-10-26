/*
 * Remote Vulkan GPU only; no model is loaded. Compare the F32 and legacy MoE
 * shaders against llama.cpp's own quantizers/dequantizers, including inputs
 * wider than SPEC_K. Build against a matching llama.cpp checkout and library:
 *   cc -std=c11 -O2 -Wall -Wextra -Werror tools/validate_dmmv_shared_cache.c \
 *     -I"$LLAMA_CPP_DIR/ggml/include" -L"$LLAMA_CPP_DIR/build/bin" \
 *     -Wl,-rpath,"$LLAMA_CPP_DIR/build/bin" -lggml-base -lvulkan -lm \
 *     -o /tmp/zinc-validate-shared-cache
 *   ZINC_SHADER_DIR=zig-out/share/zinc/shaders /tmp/zinc-validate-shared-cache
 * Optional --fits-only --bench compares the unchanged cache-fitting path with
 * an older shader build. This is a cache-hot kernel diagnostic, not model TPS.
 */
#include "ggml.h"
#include <vulkan/vulkan.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

typedef struct { VkBuffer handle; VkDeviceMemory memory; void *data; size_t size; } Buffer;
typedef struct { uint32_t M, K, expert_stride, x_expert_stride, x_offset, y_offset; } MoePush;
typedef struct { uint32_t M, K, a_offset, x_offset, y_offset, acc_mode; } F32Push;
typedef struct { const char *shader; enum ggml_type type; } Kernel;
static VkInstance instance;
static VkPhysicalDevice physical;
static VkDevice device;
static VkQueue queue;
static VkCommandPool command_pool;
static VkPhysicalDeviceProperties properties;
static VkPhysicalDeviceMemoryProperties memory_properties;
static uint32_t rng = 0x6bd03917, cases;
static int gpu_lock = -1;

static void fail(const char *what) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }
static void check(VkResult result) { if (result != VK_SUCCESS) fail("Vulkan call"); }
static void *allocate(size_t bytes) { void *p = malloc(bytes); if (!p) fail("allocation"); return p; }
static float random_float(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float)(rng & 65535) / 32768.0f - 1.0f;
}
static void init_gpu(void) {
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    check(vkCreateInstance(&info, NULL, &instance));
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, NULL));
    VkPhysicalDevice *devices = allocate(count * sizeof *devices);
    check(vkEnumeratePhysicalDevices(instance, &count, devices));
    uint32_t index = UINT32_MAX;
    const char *requested = getenv("ZINC_GPU");
    if (requested) {
        char *end;
        unsigned long parsed = strtoul(requested, &end, 10);
        if (!*requested || *end || parsed >= count) fail("ZINC_GPU must be a Vulkan device index");
        index = (uint32_t)parsed;
    } else for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { index = i; break; }
    }
    if (index >= count) fail("set ZINC_GPU to an available Vulkan device");
    physical = devices[index]; free(devices);
    char lock_path[80];
    snprintf(lock_path, sizeof lock_path, "/tmp/zinc-gpu-vulkan-%u.lock", index);
    gpu_lock = open(lock_path, O_CREAT | O_RDWR, 0600);
    if (gpu_lock < 0 || flock(gpu_lock, LOCK_EX | LOCK_NB)) fail("GPU already reserved");
    vkGetPhysicalDeviceProperties(physical, &properties);
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    VkQueueFamilyProperties *families = allocate(count * sizeof *families);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
    uint32_t family = 0;
    while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) family++;
    free(families);
    if (family == count) fail("compute queue unavailable");
    float priority = 1;
    VkDeviceQueueCreateInfo qi = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority };
    VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12 };
    VkPhysicalDeviceFeatures2 features = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11 };
    vkGetPhysicalDeviceFeatures2(physical, &features);
    VkDeviceCreateInfo di = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi };
    check(vkCreateDevice(physical, &di, NULL, &device));
    vkGetDeviceQueue(device, family, 0, &queue);
    VkCommandPoolCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family };
    check(vkCreateCommandPool(device, &ci, NULL, &command_pool));
    printf("GPU: %s; shared memory: %u bytes\n", properties.deviceName, properties.limits.maxComputeSharedMemorySize);
}
static Buffer buffer(size_t bytes) {
    Buffer b = { .size = bytes };
    VkBufferCreateInfo info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    check(vkCreateBuffer(device, &info, NULL, &b.handle));
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, b.handle, &requirements);
    uint32_t index = 0;
    const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    while (index < memory_properties.memoryTypeCount &&
           (!(requirements.memoryTypeBits & (1u << index)) ||
            (memory_properties.memoryTypes[index].propertyFlags & flags) != flags)) index++;
    if (index == memory_properties.memoryTypeCount) fail("host-coherent memory unavailable");
    VkMemoryAllocateInfo allocation = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size, .memoryTypeIndex = index };
    check(vkAllocateMemory(device, &allocation, NULL, &b.memory));
    check(vkBindBufferMemory(device, b.handle, b.memory, 0));
    check(vkMapMemory(device, b.memory, 0, bytes, 0, &b.data));
    return b;
}
static void free_buffer(Buffer b) {
    vkUnmapMemory(device, b.memory); vkDestroyBuffer(device, b.handle, NULL); vkFreeMemory(device, b.memory, NULL);
}
static VkShaderModule shader(const char *name) {
    char path[4096];
    const char *dir = getenv("ZINC_SHADER_DIR");
    if (!dir) dir = "zig-out/share/zinc/shaders";
    int n = snprintf(path, sizeof path, "%s/%s.spv", dir, name);
    if (n < 0 || (size_t)n >= sizeof path) fail("shader path too long");
    FILE *f = fopen(path, "rb");
    if (!f || fseek(f, 0, SEEK_END)) fail("shader open");
    long bytes = ftell(f);
    if (bytes <= 0 || bytes % 4 || fseek(f, 0, SEEK_SET)) fail("shader size");
    uint32_t *code = allocate((size_t)bytes);
    if (fread(code, 1, (size_t)bytes, f) != (size_t)bytes) fail("shader read");
    fclose(f);
    VkShaderModuleCreateInfo info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)bytes, .pCode = code };
    VkShaderModule module;
    check(vkCreateShaderModule(device, &info, NULL, &module)); free(code);
    return module;
}
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y);
}
static void run_case(Kernel kernel, uint32_t K, uint32_t capacity, int separate_inputs, int accumulate, int bench) {
    const int f32 = kernel.type == GGML_TYPE_F32;
    const uint32_t M = bench ? 256 : 67, slots = f32 ? 1 : 2, experts = 3, x_start = 5, y_start = 7;
    const uint32_t x_stride = separate_inputs ? K + 11 : 0;
    const size_t row_bytes = ggml_row_size(kernel.type, K);
    const uint32_t expert_stride = (uint32_t)((M * row_bytes + 255) / 256 * 256);
    const size_t x_count = x_start + (slots - 1) * x_stride + K + 9, y_count = y_start + slots * M + 9;
    Buffer b[] = { buffer((size_t)experts * expert_stride), buffer(x_count * 4), buffer(y_count * 4), buffer(slots * 4) };
    float *x = b[1].data, *y = b[2].data;
    uint32_t *routing = b[3].data;
    routing[0] = 2; if (slots > 1) routing[1] = 0;
    float *row = allocate(K * sizeof *row), *decoded = allocate(K * sizeof *decoded);
    memset(b[0].data, 0xA5, b[0].size);
    const struct ggml_type_traits *traits = ggml_get_type_traits(kernel.type);
    ggml_quantize_init(kernel.type);
    for (uint32_t e = 0; e < experts; e++) for (uint32_t r = 0; r < M; r++) {
        for (uint32_t j = 0; j < K; j++) row[j] = random_float() * 0.25f;
        void *dst = (char *)b[0].data + e * expert_stride + r * row_bytes;
        if (f32) memcpy(dst, row, K * 4); else traits->from_float_ref(row, dst, K);
    }
    for (size_t i = 0; i < x_count; i++) x[i] = random_float();
    for (size_t i = 0; i < y_count; i++) y[i] = -999.25f;
    VkDescriptorSetLayoutBinding bindings[4];
    for (uint32_t i = 0; i < 4; i++) bindings[i] = (VkDescriptorSetLayoutBinding){
        .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    VkDescriptorSetLayoutCreateInfo li = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = f32 ? 3 : 4, .pBindings = bindings };
    VkDescriptorSetLayout set_layout;
    check(vkCreateDescriptorSetLayout(device, &li, NULL, &set_layout));
    VkPushConstantRange range = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 24 };
    VkPipelineLayoutCreateInfo pi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &range };
    VkPipelineLayout layout;
    check(vkCreatePipelineLayout(device, &pi, NULL, &layout));
    VkShaderModule module = shader(kernel.shader);
    VkSpecializationMapEntry entry = { .constantID = 1, .size = 4 };
    VkSpecializationInfo spec = { .mapEntryCount = 1, .pMapEntries = &entry, .dataSize = 4, .pData = &capacity };
    VkComputePipelineCreateInfo pipeline_info = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main", .pSpecializationInfo = &spec },
        .layout = layout };
    VkPipeline pipeline;
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline));
    VkDescriptorPoolSize pool_size = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 };
    VkDescriptorPoolCreateInfo pool_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size };
    VkDescriptorPool pool;
    check(vkCreateDescriptorPool(device, &pool_info, NULL, &pool));
    VkDescriptorSetAllocateInfo si = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout };
    VkDescriptorSet set;
    check(vkAllocateDescriptorSets(device, &si, &set));
    VkDescriptorBufferInfo infos[4]; VkWriteDescriptorSet writes[4];
    for (uint32_t i = 0; i < 4; i++) {
        infos[i] = (VkDescriptorBufferInfo){ .buffer = b[i].handle, .range = b[i].size };
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
            .dstBinding = i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[i] };
    }
    vkUpdateDescriptorSets(device, f32 ? 3 : 4, writes, 0, NULL);
    MoePush moe_push = { M, K, expert_stride, x_stride, x_start * 4, y_start * 4 };
    F32Push f32_push = { M, K, expert_stride, x_start * 4, y_start * 4, (uint32_t)accumulate };
    VkCommandBufferAllocateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cmd;
    check(vkAllocateCommandBuffers(device, &ci, &cmd));
    VkQueryPoolCreateInfo query_info = { .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 };
    VkQueryPool queries;
    check(vkCreateQueryPool(device, &query_info, NULL, &queries));
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    check(vkBeginCommandBuffer(cmd, &begin));
    vkCmdResetQueryPool(cmd, queries, 0, 2);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, f32 ? (void *)&f32_push : (void *)&moe_push);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, 0);
    vkCmdDispatch(cmd, f32 ? M : (M + 63) / 64, slots, 1);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries, 1);
    VkMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    check(vkEndCommandBuffer(cmd));
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    double times[15], max_error = 0;
    const unsigned repeats = bench ? 15 : 1;
    for (unsigned repeat = 0; repeat < repeats; repeat++) {
        for (size_t i = 0; i < y_count; i++) y[i] = -999.25f;
        if (accumulate) for (uint32_t s = 0; s < slots; s++) for (uint32_t r = 0; r < M; r++)
            y[y_start + s * M + r] = (float)((s * M + r) % 13) * 0.125f - 0.5f;
        check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE)); check(vkQueueWaitIdle(queue));
        uint64_t ticks[2];
        check(vkGetQueryPoolResults(device, queries, 0, 2, sizeof ticks, ticks, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        times[repeat] = (ticks[1] - ticks[0]) * properties.limits.timestampPeriod / 1000.0;
        if (repeat) continue;
        for (uint32_t s = 0; s < slots; s++) for (uint32_t r = 0; r < M; r++) {
            const void *src = (char *)b[0].data + (f32 ? 1 : routing[s]) * expert_stride + r * row_bytes;
            if (f32) memcpy(decoded, src, K * 4); else traits->to_float(src, decoded, K);
            double expected = accumulate ? (float)((s * M + r) % 13) * 0.125f - 0.5f : 0;
            for (uint32_t j = 0; j < K; j++) expected += (double)decoded[j] * x[x_start + s * x_stride + j];
            float actual = y[y_start + s * M + r];
            double error = fabs(expected - actual);
            if (!isfinite(actual) || error > 5e-4 * (1 + fabs(expected))) {
                fprintf(stderr, "%s K=%u SPEC_K=%u slot=%u row=%u expected=%.12g actual=%.9g\n",
                        kernel.shader, K, capacity, s, r, expected, actual);
                fail("llama.cpp reference");
            }
            if (error > max_error) max_error = error;
        }
        for (size_t i = 0; i < y_count; i++)
            if ((i < y_start || i >= y_start + slots * M) && y[i] != -999.25f) fail("output guard");
    }
    qsort(times, repeats, sizeof times[0], compare_double);
    printf("PASS %-17s K=%5u SPEC_K=%5u stride=%5u acc=%d error=%.6g%s%.3f us\n",
           kernel.shader, K, capacity, x_stride, accumulate, max_error, bench ? " median=" : " dispatch=", times[repeats / 2]);
    cases++;
    vkDestroyQueryPool(device, queries, NULL); vkFreeCommandBuffers(device, command_pool, 1, &cmd);
    vkDestroyDescriptorPool(device, pool, NULL); vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, module, NULL); vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyDescriptorSetLayout(device, set_layout, NULL);
    for (unsigned i = 0; i < 4; i++) free_buffer(b[i]);
    free(row); free(decoded);
}
int main(int argc, char **argv) {
    int fits_only = 0, bench = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fits-only")) fits_only = 1;
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else fail("unknown argument");
    }
    const Kernel kernels[] = {
        { "dmmv_f32", GGML_TYPE_F32 }, { "dmmv_q4k_moe", GGML_TYPE_Q4_K },
        { "dmmv_q5k_moe", GGML_TYPE_Q5_K }, { "dmmv_q6k_moe", GGML_TYPE_Q6_K },
        { "dmmv_q5_1_moe", GGML_TYPE_Q5_1 }, { "dmmv_mxfp4_moe", GGML_TYPE_MXFP4 },
    };
    const uint32_t shapes[][2] = { {256, 256}, {512, 256}, {512, 512}, {8192, 7936}, {16384, 16128}, {18944, 16128}, {2048, 4096} };
    init_gpu();
    for (unsigned k = 0; k < sizeof kernels / sizeof kernels[0]; k++)
        for (unsigned s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
            uint32_t K = shapes[s][0], capacity = shapes[s][1];
            uint32_t device_capacity = (properties.limits.maxComputeSharedMemorySize - 1024) / 4;
            if (capacity > device_capacity) capacity = device_capacity;
            if (fits_only && K > capacity) continue;
            for (int separate = 0; separate < 2; separate++)
                run_case(kernels[k], K, capacity, separate, kernels[k].type == GGML_TYPE_F32 && separate, bench);
        }
    check(vkDeviceWaitIdle(device)); vkDestroyCommandPool(device, command_pool, NULL);
    vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL); close(gpu_lock);
    ggml_quantize_free();
    printf("PASS: %u cache-boundary/reference/offset/stride/guard cases\n", cases);
    return 0;
}
