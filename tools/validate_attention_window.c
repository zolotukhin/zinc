/* Remote GPU only. No model is loaded. Match the pinned llama.cpp STANDARD
 * mask: query_position - key_position >= window_size is excluded. Exercise
 * the shipped decode, split/merge and batched kernels on paged F32/F16/Q8 KV.
 * Short model continuations cannot cover Gemma's 1024-token boundary.
 */
#define main zinc_cache_reference_main
#include "validate_dmmv_shared_cache.c"
#undef main

typedef struct {
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkShaderModule module;
    VkPipeline pipeline;
    VkDescriptorPool pool;
    VkDescriptorSet set;
} AttentionKernel;

static AttentionKernel attention_kernel(const char *name, uint32_t count, uint32_t push_bytes, Buffer **buffers) {
    AttentionKernel k = {0};
    VkDescriptorSetLayoutBinding bindings[6];
    VkDescriptorBufferInfo infos[6];
    VkWriteDescriptorSet writes[6];
    for (uint32_t i = 0; i < count; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
        infos[i] = (VkDescriptorBufferInfo){ .buffer = buffers[i]->handle, .range = buffers[i]->size };
    }
    VkDescriptorSetLayoutCreateInfo si = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = count, .pBindings = bindings };
    check(vkCreateDescriptorSetLayout(device, &si, NULL, &k.set_layout));
    VkPushConstantRange range = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = push_bytes };
    VkPipelineLayoutCreateInfo li = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &k.set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &range };
    check(vkCreatePipelineLayout(device, &li, NULL, &k.layout));
    k.module = shader(name);
    VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = k.layout,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = k.module, .pName = "main" } };
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, NULL, &k.pipeline));
    VkDescriptorPoolSize size = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = count };
    VkDescriptorPoolCreateInfo pi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &size };
    check(vkCreateDescriptorPool(device, &pi, NULL, &k.pool));
    VkDescriptorSetAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = k.pool, .descriptorSetCount = 1, .pSetLayouts = &k.set_layout };
    check(vkAllocateDescriptorSets(device, &ai, &k.set));
    for (uint32_t i = 0; i < count; i++) writes[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = k.set, .dstBinding = i,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[i] };
    vkUpdateDescriptorSets(device, count, writes, 0, NULL);
    return k;
}

static void attention_free(AttentionKernel k) {
    vkDestroyDescriptorPool(device, k.pool, NULL);
    vkDestroyPipeline(device, k.pipeline, NULL);
    vkDestroyShaderModule(device, k.module, NULL);
    vkDestroyPipelineLayout(device, k.layout, NULL);
    vkDestroyDescriptorSetLayout(device, k.set_layout, NULL);
}

static void attention_dispatch(VkCommandBuffer cmd, AttentionKernel k, const uint32_t *push,
                               uint32_t bytes, uint32_t x, uint32_t y) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &k.set, 0, NULL);
    vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, push);
    vkCmdDispatch(cmd, x, y, 1);
    VkMemoryBarrier barrier = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
}

static void cache_encode(Buffer b, const float *values, uint32_t elements, unsigned storage) {
    if (storage == 0) { memcpy(b.data, values, elements * 4); return; }
    if (storage == 1) {
        for (uint32_t i = 0; i < elements; i++) ((ggml_fp16_t *)b.data)[i] = ggml_fp32_to_fp16(values[i]);
        return;
    }
    /* ZINC Q8 cache rows use 36 bytes: 32 signed bytes, F16 scale, padding. */
    for (uint32_t i = 0; i < elements; i += 32) {
        unsigned char *p = (unsigned char *)b.data + (i / 32) * 36;
        float amax = 0;
        for (uint32_t j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(values[i + j]));
        float scale = amax / 127;
        ggml_fp16_t half = ggml_fp32_to_fp16(scale);
        for (uint32_t j = 0; j < 32; j++) ((int8_t *)p)[j] = scale > 0 ? (int8_t)roundf(values[i + j] / scale) : 0;
        memcpy(p + 32, &half, 2); p[34] = p[35] = 0;
    }
}

static float cache_value(Buffer b, uint32_t index, unsigned storage) {
    if (storage == 0) return ((float *)b.data)[index];
    if (storage == 1) return ggml_fp16_to_fp32(((ggml_fp16_t *)b.data)[index]);
    const unsigned char *p = (const unsigned char *)b.data + (index / 32) * 36;
    ggml_fp16_t scale; memcpy(&scale, p + 32, 2);
    return ((const int8_t *)p)[index % 32] * ggml_fp16_to_fp32(scale);
}

static uint32_t attention_bits(float value) { uint32_t bits; memcpy(&bits, &value, 4); return bits; }

static void attention_case(uint32_t dim, uint32_t length, uint32_t window, unsigned storage, unsigned mode) {
    const uint32_t heads = 4, kv_heads = 2, page_size = 16, chunks = mode == 1 ? 4 : 1;
    const uint32_t queries = mode == 2 ? 3 : 1, prefix = mode == 2 ? 0 : 16;
    const uint32_t pages = (length + page_size - 1) / page_size, elements = pages * page_size * kv_heads * dim;
    const uint32_t out_elements = prefix + queries * heads * dim;
    const float guard = -999.25f, scale = 1.0f / sqrtf((float)dim);
    const size_t cache_bytes = storage == 0 ? elements * 4 : storage == 1 ? elements * 2 : (elements / 32) * 36;
    Buffer q = buffer((prefix + queries * heads * dim) * 4), k = buffer(cache_bytes), v = buffer(cache_bytes);
    Buffer table = buffer(pages * 4), out = buffer((out_elements + 16) * 4), sinks = buffer(heads * 4);
    Buffer partial = buffer((heads * chunks * (dim + 2) + 16) * 4);
    float *q_data = q.data, *out_data = out.data;
    for (uint32_t i = 0; i < prefix + queries * heads * dim; i++) q_data[i] = random_float() * 0.2f;
    for (uint32_t i = 0; i < out_elements + 16; i++) out_data[i] = guard;
    for (uint32_t i = 0; i < heads; i++) ((float *)sinks.data)[i] = NAN;
    for (uint32_t i = 0; i < pages; i++) ((uint32_t *)table.data)[i] = pages - 1 - i;
    float *keys = allocate(elements * 4), *values = allocate(elements * 4);
    for (uint32_t logical = 0; logical < pages * page_size; logical++) {
        uint32_t slot = ((uint32_t *)table.data)[logical / page_size] * page_size + logical % page_size;
        for (uint32_t h = 0; h < kv_heads; h++) for (uint32_t d = 0; d < dim; d++) {
            uint32_t at = (slot * kv_heads + h) * dim + d;
            keys[at] = random_float() * 0.15f;
            values[at] = random_float() * 0.4f + (logical < length / 2 ? -3.0f : 1.0f);
        }
    }
    cache_encode(k, keys, elements, storage); cache_encode(v, values, elements, storage);
    free(keys); free(values);
    const char *decode[] = { "flash_attn", "flash_attn_f16kv", "flash_attn_q8kv" };
    const char *batched[] = { "flash_attn_batched", "flash_attn_batched_f16kv" };
    Buffer *inputs[] = { &q, &k, &v, &table, mode == 1 ? &partial : &out, &sinks };
    AttentionKernel first = attention_kernel(mode == 2 ? batched[storage] : decode[storage], 6, mode == 2 ? 36 : 44, inputs);
    AttentionKernel merge = {0};
    if (mode == 1) {
        Buffer *merge_inputs[] = { &partial, &out, &sinks };
        merge = attention_kernel("flash_attn_split_merge", 3, 20, merge_inputs);
    }
    uint32_t push[] = { dim, heads, kv_heads, length, page_size, attention_bits(scale), 0, chunks, prefix / 4, prefix / 4, window };
    uint32_t batch_push[] = { dim, heads, kv_heads, length - queries, queries, page_size, attention_bits(scale), 0, window };
    uint32_t merge_push[] = { dim, heads, 0, chunks, prefix / 4 };
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(device, &cai, &cmd));
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    check(vkBeginCommandBuffer(cmd, &begin));
    for (unsigned replay = 0; replay < 2; replay++) {
        attention_dispatch(cmd, first, mode == 2 ? batch_push : push, mode == 2 ? 36 : 44, heads, mode == 2 ? queries : chunks);
        if (mode == 1) attention_dispatch(cmd, merge, merge_push, 20, heads, 1);
    }
    VkMemoryBarrier host = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
    check(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; check(vkCreateFence(device, &fi, NULL, &fence));
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    check(vkQueueSubmit(queue, 1, &submit, fence));
    check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    double *scores = allocate(length * sizeof *scores);
    for (uint32_t query = 0; query < queries; query++) for (uint32_t head = 0; head < heads; head++) {
        uint32_t position = length - queries + query;
        double max_score = -INFINITY, denominator = 0;
        /* Independent reference predicate, not the shader's start formula. */
        for (uint32_t token = 0; token <= position; token++) {
            scores[token] = -INFINITY;
            if (window && position - token >= window) continue;
            uint32_t slot = ((uint32_t *)table.data)[token / page_size] * page_size + token % page_size;
            uint32_t base = (slot * kv_heads + head / (heads / kv_heads)) * dim;
            double dot = 0;
            for (uint32_t d = 0; d < dim; d++) dot += (double)q_data[prefix + (query * heads + head) * dim + d] * cache_value(k, base + d, storage);
            scores[token] = dot * scale; max_score = fmax(max_score, scores[token]);
        }
        for (uint32_t token = 0; token <= position; token++) denominator += exp(scores[token] - max_score);
        for (uint32_t token = 0; token <= position; token++) scores[token] = exp(scores[token] - max_score) / denominator;
        for (uint32_t d = 0; d < dim; d++) {
            double expected = 0;
            for (uint32_t token = 0; token <= position; token++) {
                if (scores[token] == 0) continue;
                uint32_t slot = ((uint32_t *)table.data)[token / page_size] * page_size + token % page_size;
                uint32_t base = (slot * kv_heads + head / (heads / kv_heads)) * dim;
                expected += scores[token] * cache_value(v, base + d, storage);
            }
            float actual = out_data[prefix + (query * heads + head) * dim + d];
            if (!isfinite(actual) || fabs(actual - expected) > 4e-4 * (1.0 + fabs(expected))) {
                fprintf(stderr, "attention mismatch: dim=%u length=%u window=%u storage=%u mode=%u query=%u head=%u d=%u actual=%.8g expected=%.8g\n",
                        dim, length, window, storage, mode, query, head, d, actual, expected);
                fail("sliding-window attention reference");
            }
        }
    }
    for (uint32_t i = 0; i < prefix; i++) if (out_data[i] != guard) fail("attention prefix guard");
    for (uint32_t i = out_elements; i < out_elements + 16; i++) if (out_data[i] != guard) fail("attention tail guard");
    free(scores); attention_free(first); if (mode == 1) attention_free(merge);
    vkDestroyFence(device, fence, NULL); vkFreeCommandBuffers(device, command_pool, 1, &cmd);
    free_buffer(q); free_buffer(k); free_buffer(v); free_buffer(table); free_buffer(out); free_buffer(sinks); free_buffer(partial);
    cases++;
}

int main(void) {
    init_gpu();
    const uint32_t lengths[] = { 1023, 1024, 1025, 1041 }, windows[] = { 0, 1, 17, 1024 };
    for (uint32_t dim = 256; dim <= 512; dim *= 2)
        for (unsigned l = 0; l < 4; l++) for (unsigned w = 0; w < 4; w++)
            for (unsigned storage = 0; storage < 3; storage++) for (unsigned mode = 0; mode < 3; mode++) {
                if (storage == 2 && mode == 2) continue; /* Q8 prefill uses the serial window-aware path. */
                attention_case(dim, lengths[l], windows[w], storage, mode);
            }
    vkDestroyCommandPool(device, command_pool, NULL); vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL);
    flock(gpu_lock, LOCK_UN); close(gpu_lock);
    printf("PASS: %u sliding-window/page/reference/guard/replay cases\n", cases);
    return 0;
}
