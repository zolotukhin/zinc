/*
 * Experimental Intel Q4_K row-layout proof. Use the shared native quantizer,
 * independent double-precision dot, nonzero offsets, odd rows, accumulation
 * and guards. Every row must also match the legacy SIMD32 reduction bitwise.
 * Run only on an authorized remote GPU worker under its reservation.
 * --bench is a cache-hot kernel diagnostic, never whole-model throughput.
 */
#define main shared_cache_fixture_main
#include "validate_dmmv_shared_cache.c"
#undef main

static uint32_t identical_pairs;

static void paired_case(uint32_t K, int accumulate, int bench, uint32_t salt) {
    const Kernel legacy = { "dmmv_q4k", GGML_TYPE_Q4_K };
    const Kernel candidate = { "dmmv_q4k_intel_rows", GGML_TYPE_Q4_K };
    _Static_assert(ZINC_DMMV_BENCH_ROWS >= ZINC_DMMV_TEST_ROWS, "capture buffer covers both modes");
    float old_y[ZINC_DMMV_BENCH_ROWS], new_y[ZINC_DMMV_BENCH_ROWS];
    const uint32_t seed = 0x6bd03917u ^ K ^ ((uint32_t)accumulate << 30) ^ salt;
    rng = seed;
    run_case_capture(legacy, K, K, accumulate, accumulate, bench, 0, old_y);
    rng = seed;
    run_case_capture(candidate, K, K, accumulate, accumulate, bench, 0, new_y);
    const uint32_t rows = bench ? ZINC_DMMV_BENCH_ROWS : ZINC_DMMV_TEST_ROWS;
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t old_bits, new_bits;
        memcpy(&old_bits, &old_y[row], sizeof old_bits);
        memcpy(&new_bits, &new_y[row], sizeof new_bits);
        if (old_bits != new_bits) {
            fprintf(stderr, "Q4 sum order K=%u acc=%d row=%u old=%08x new=%08x\n",
                    K, accumulate, row, old_bits, new_bits);
            fail("Q4 reduction differs from the legacy SIMD32 shader");
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
    const uint32_t widths[] = {
        256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 2816,
        3072, 4096, 5120, 5376, 6144, 8192, 10240, 12288, 17408, 21504,
    };
    init_gpu();
    VkPhysicalDeviceSubgroupProperties subgroup = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceProperties2 info = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &subgroup };
    vkGetPhysicalDeviceProperties2(physical, &info);
    if (properties.vendorID != 0x8086 || subgroup.subgroupSize != 32)
        fail("paired Q4 qualification requires Intel legacy SIMD32");
    const uint32_t salts[] = { 0u, 0x9e3779b9u, 0xd1b54a35u };
    for (unsigned seed = 0; seed < sizeof salts / sizeof salts[0]; seed++)
        for (unsigned width = 0; width < sizeof widths / sizeof widths[0]; width++)
            for (int accumulate = 0; accumulate < 2; accumulate++)
                paired_case(widths[width], accumulate, bench, salts[seed]);
    check(vkDeviceWaitIdle(device));
    vkDestroyCommandPool(device, command_pool, NULL);
    vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL); close(gpu_lock);
    ggml_quantize_free();
    printf("PASS: %u Q4 native-reference/offset/accumulation/guard cases\n", cases);
    printf("PASS: %u Q4 identical-reduction pairs\n", identical_pairs);
    return 0;
}
