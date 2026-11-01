/* Remote GPU only; no GGUF is loaded. Reuse the existing device selection,
 * reservation, buffers and shader loader unchanged. Build with the same flags
 * and pinned llama.cpp library as validate_dmmv_shared_cache.c.
 *
 * These full-vector CPU oracles catch subgroup-local RMS/dot reductions even
 * when short generated text remains plausible. Cases include vector tails,
 * batched inputs, expert-id ties, output guards and repeated counter reuse.
 */
#define main zinc_cache_reference_main
#include "validate_dmmv_shared_cache.c"
#undef main

static uint32_t float_bits(float value) {
    uint32_t bits; memcpy(&bits, &value, sizeof bits); return bits;
}
static float bits_float(uint32_t bits) {
    float value; memcpy(&value, &bits, sizeof value); return value;
}
static uint32_t route_key(float value, uint32_t index) {
    uint32_t bits = float_bits(value);
    uint32_t ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    return (ordered & 0xFFFFFF00u) | (255u - index);
}
static float route_value(uint32_t key) {
    uint32_t bits = key & 0xFFFFFF00u;
    return bits_float((bits & 0x80000000u ? bits ^ 0x80000000u : ~bits) & 0xFFFFFF00u);
}
static void near_router(float actual, float expected) {
    if (!isfinite(actual) || fabsf(actual - expected) > 3e-5f * (1.0f + fabsf(expected))) {
        fprintf(stderr, "router mismatch: actual %.9g expected %.9g\n", actual, expected);
        fail("full-workgroup router reference");
    }
}

static void router_case(uint32_t K, uint32_t M, unsigned mode) {
    const char *names[] = { "rms_norm_scale_dmmv_f32", "rms_norm_scale_dmmv_f32_batch", "rms_norm_scale_dmmv_f32_topk" };
    const uint32_t tokens = mode == 1 ? 3 : 1, k = M < 16 ? M : 16, count = mode == 2 ? 6 : 4;
    const float eps = 1e-6f, scale = 1.0f / sqrtf((float)K), guard = -999.25f;
    Buffer b[] = { buffer(tokens * K * 4), buffer(K * 4), buffer(M * K * 4),
                   buffer((tokens * M + 8) * 4), buffer((2 * k + 8) * 4), buffer(4) };
    float *x = b[0].data, *weights = b[1].data, *matrix = b[2].data, *out = b[3].data;
    for (uint32_t i = 0; i < tokens * K; i++) x[i] = random_float() * (1.0f + (float)((i / 4) % 64) / 16.0f);
    for (uint32_t i = 0; i < K; i++) weights[i] = 0.5f + fabsf(random_float());
    for (uint32_t row = 0; row < M; row++) for (uint32_t i = 0; i < K; i++)
        matrix[row * K + i] = random_float() * 0.03f + (row >= M - 2 ? 0.07f * x[i] : 0.0f);
    /* Identical high-scoring rows exercise the lower-expert-id tie break. */
    memcpy(matrix + (M - 1) * K, matrix + (M - 2) * K, K * 4);
    for (uint32_t i = 0; i < tokens * M + 8; i++) out[i] = guard;
    uint32_t *routing = b[4].data;
    for (uint32_t i = 0; i < 2 * k + 8; i++) routing[i] = 0xDEADBEEFu;
    *(uint32_t *)b[5].data = 0;
    float *expected = allocate(tokens * M * 4);
    for (uint32_t tok = 0; tok < tokens; tok++) {
        double energy = 0;
        for (uint32_t i = 0; i < K; i++) energy += (double)x[tok * K + i] * x[tok * K + i];
        float inverse = 1.0f / sqrtf((float)(energy / K) + eps);
        for (uint32_t row = 0; row < M; row++) {
            double dot = 0;
            for (uint32_t i = 0; i < K; i++)
                dot += (double)(x[tok * K + i] * inverse * weights[i]) * matrix[row * K + i];
            expected[tok * M + row] = (float)dot;
        }
    }
    VkDescriptorSetLayoutBinding bindings[6];
    VkDescriptorBufferInfo infos[6];
    VkWriteDescriptorSet writes[6];
    for (uint32_t i = 0; i < count; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
        infos[i] = (VkDescriptorBufferInfo){ .buffer = b[i].handle, .range = b[i].size };
    }
    VkDescriptorSetLayoutCreateInfo li = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = count, .pBindings = bindings };
    VkDescriptorSetLayout set_layout; check(vkCreateDescriptorSetLayout(device, &li, NULL, &set_layout));
    uint32_t push[] = { M, K, float_bits(eps), k, float_bits(scale) };
    uint32_t push_size = (mode == 2 ? 5 : mode == 1 ? 4 : 3) * 4;
    if (mode == 1) { push[2] = tokens; push[3] = float_bits(eps); }
    VkPushConstantRange range = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = push_size };
    VkPipelineLayoutCreateInfo pi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &range };
    VkPipelineLayout layout; check(vkCreatePipelineLayout(device, &pi, NULL, &layout));
    VkShaderModule module = shader(names[mode]);
    VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                   .module = module, .pName = "main" } };
    VkPipeline pipeline; check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, NULL, &pipeline));
    VkDescriptorPoolSize pool_size = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = count };
    VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size };
    VkDescriptorPool pool; check(vkCreateDescriptorPool(device, &dpi, NULL, &pool));
    VkDescriptorSetAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout };
    VkDescriptorSet set; check(vkAllocateDescriptorSets(device, &ai, &set));
    for (uint32_t i = 0; i < count; i++) writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = i,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[i] };
    vkUpdateDescriptorSets(device, count, writes, 0, NULL);
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer command; check(vkAllocateCommandBuffers(device, &cai, &command));
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    check(vkBeginCommandBuffer(command, &begin));
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
    for (unsigned repeat = 0; repeat < 3; repeat++) {
        vkCmdDispatch(command, M, tokens, 1);
        VkMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    }
    VkMemoryBarrier host = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
    check(vkEndCommandBuffer(command));
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; check(vkCreateFence(device, &fi, NULL, &fence));
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command };
    check(vkQueueSubmit(queue, 1, &submit, fence));
    check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    for (uint32_t i = 0; i < tokens * M; i++) near_router(out[i], expected[i]);
    for (uint32_t i = tokens * M; i < tokens * M + 8; i++) if (out[i] != guard) fail("router output guard");
    if (mode == 2) {
        uint32_t selected[16]; float logits[16], sum = 0;
        for (uint32_t j = 0; j < k; j++) {
            uint32_t best = 0, winner = UINT32_MAX;
            for (uint32_t row = 0; row < M; row++) {
                int seen = 0;
                for (uint32_t prev = 0; prev < j; prev++) if (selected[prev] == row) seen = 1;
                uint32_t key = route_key(expected[row], row);
                if (!seen && (winner == UINT32_MAX || key > best)) { best = key; winner = row; }
            }
            selected[j] = winner; logits[j] = route_value(best) * scale;
            if (routing[j] != winner) fail("router expert selection/tie break");
        }
        for (uint32_t j = 0; j < k; j++) sum += expf(logits[j] - logits[0]);
        for (uint32_t j = 0; j < k; j++) near_router(bits_float(routing[k + j]), expf(logits[j] - logits[0]) / sum);
        for (uint32_t i = 2 * k; i < 2 * k + 8; i++) if (routing[i] != 0xDEADBEEFu) fail("router routing guard");
        if (*(uint32_t *)b[5].data != 0) fail("router replay counter was not reset");
    }
    free(expected);
    vkDestroyFence(device, fence, NULL); vkFreeCommandBuffers(device, command_pool, 1, &command);
    vkDestroyDescriptorPool(device, pool, NULL); vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, module, NULL); vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyDescriptorSetLayout(device, set_layout, NULL);
    for (unsigned i = 0; i < 6; i++) free_buffer(b[i]);
    cases++;
}

int main(void) {
    init_gpu();
    const uint32_t widths[] = { 4, 68, 256, 2048 }, experts[] = { 3, 67, 256 };
    for (unsigned i = 0; i < sizeof widths / sizeof *widths; i++)
        for (unsigned j = 0; j < sizeof experts / sizeof *experts; j++)
            for (unsigned mode = 0; mode < 3; mode++) router_case(widths[i], experts[j], mode);
    vkDestroyCommandPool(device, command_pool, NULL); vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
    flock(gpu_lock, LOCK_UN); close(gpu_lock);
    printf("PASS: %u Gemma router workgroup/reference/guard/replay cases\n", cases);
    return 0;
}
