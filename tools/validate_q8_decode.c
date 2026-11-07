/*
 * Remote GPU regression for both Q8_0 decode shaders. The shared fixture uses
 * llama.cpp's native Q8_0 quantizer/dequantizer and a double-precision dot
 * reference, with odd row counts, nonzero A/X/Y offsets, accumulation and
 * output guards. No model or CPU inference baseline is involved.
 * Paired inputs must also match the legacy SIMD32 shader bit for bit: small
 * changes of reduction order can exceed model-level near-tie agreement limits.
 *
 * Build as documented in validate_dmmv_shared_cache.c, replacing its source
 * with this file. Run with ZINC_SHADER_DIR pointing to the package's shaders
 * and ZINC_GPU selecting the remote Vulkan device. --bench is a cache-hot
 * kernel diagnostic; it cannot establish whole-model throughput.
 */
#define main shared_cache_fixture_main
#include "validate_dmmv_shared_cache.c"
#undef main

static uint32_t identical_pairs;

static void paired_case(uint32_t K, uint32_t blocks, int accumulate, int bench, uint32_t salt) {
    const Kernel legacy = { "dmmv_q8_0", GGML_TYPE_Q8_0 };
    const Kernel candidate = { "dmmv_q8_0_intel_rows", GGML_TYPE_Q8_0 };
    _Static_assert(ZINC_DMMV_BENCH_ROWS >= ZINC_DMMV_TEST_ROWS, "capture buffer covers both modes");
    float old_y[ZINC_DMMV_BENCH_ROWS], new_y[ZINC_DMMV_BENCH_ROWS];
    const uint32_t seed = 0x6bd03917u ^ K ^ (blocks << 16) ^ ((uint32_t)accumulate << 30) ^ salt;
    rng = seed;
    run_case_capture(legacy, K, K, accumulate, accumulate, bench, blocks, old_y);
    rng = seed;
    run_case_capture(candidate, K, K, accumulate, accumulate, bench, blocks, new_y);
    const uint32_t rows = bench ? ZINC_DMMV_BENCH_ROWS : ZINC_DMMV_TEST_ROWS;
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t old_bits, new_bits;
        memcpy(&old_bits, &old_y[row], sizeof old_bits);
        memcpy(&new_bits, &new_y[row], sizeof new_bits);
        if (old_bits != new_bits) {
            fprintf(stderr, "Q8 sum order K=%u blocks=%u acc=%d row=%u old=%08x new=%08x\n",
                    K, blocks, accumulate, row, old_bits, new_bits);
            fail("Q8 reduction differs from the legacy SIMD32 shader");
        }
    }
    identical_pairs++;
}

int main(int argc, char **argv) {
    int bench = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) bench = 1;
        else fail("unknown argument");
    }
    // Include the managed checkpoints' shared-expert K=512, Gemma K=2112/2816,
    // and NextN K=10240, even when an MTP-off model replay skips the NextN block.
    // The short widths and odd row count separately exercise tails and guards.
    const uint32_t widths[] = { 32, 64, 96, 512, 1024, 2048, 2112, 2816, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 17408, 21504 };
    init_gpu();
    VkPhysicalDeviceSubgroupProperties subgroup = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceProperties2 info = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup };
    vkGetPhysicalDeviceProperties2(physical, &info);
    printf("Q8 legacy subgroup: %u; vendor: %04x\n", subgroup.subgroupSize, properties.vendorID);
    if (properties.vendorID != 0x8086 || subgroup.subgroupSize != 32) fail("paired Q8 qualification requires Intel legacy SIMD32");
    // Salt zero retains the fixed-128 regression that exposed an unfused update.
    // Additional deterministic inputs catch rounding differences hidden by one seed.
    const uint32_t salts[] = { 0u, 0x9e3779b9u, 0xd1b54a35u };
    for (unsigned seed = 0; seed < sizeof salts / sizeof salts[0]; seed++) {
        for (unsigned s = 0; s < sizeof widths / sizeof widths[0]; s++)
            for (int accumulate = 0; accumulate < 2; accumulate++)
                paired_case(widths[s], 0, accumulate, bench, salts[seed]);
        // The production dispatcher also loads fixed 64/128-block variants.
        for (uint32_t blocks = 64; blocks <= 128; blocks *= 2)
            for (int accumulate = 0; accumulate < 2; accumulate++)
                paired_case(blocks * 32, blocks, accumulate, bench, salts[seed]);
    }
    check(vkDeviceWaitIdle(device));
    vkDestroyCommandPool(device, command_pool, NULL);
    vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL); close(gpu_lock);
    ggml_quantize_free();
    printf("PASS: %u Q8 native-reference/offset/accumulation/guard cases\n", cases);
    printf("PASS: %u Q8 identical-reduction pairs\n", identical_pairs);
    return 0;
}
