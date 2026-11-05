/*
 * Remote GPU regression for both Q8_0 decode shaders. The shared fixture uses
 * llama.cpp's native Q8_0 quantizer/dequantizer and a double-precision dot
 * reference, with odd row counts, nonzero A/X/Y offsets, accumulation and
 * output guards. No model or CPU inference baseline is involved.
 *
 * Build as documented in validate_dmmv_shared_cache.c, replacing its source
 * with this file. Run with ZINC_SHADER_DIR pointing to the package's shaders
 * and ZINC_GPU selecting the remote Vulkan device. --bench is a cache-hot
 * kernel diagnostic; it cannot establish whole-model throughput.
 */
#define main shared_cache_fixture_main
#include "validate_dmmv_shared_cache.c"
#undef main

int main(int argc, char **argv) {
    int bench = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bench")) bench = 1;
        else fail("unknown argument");
    }
    const Kernel kernels[] = {
        { "dmmv_q8_0", GGML_TYPE_Q8_0 },
        { "dmmv_q8_0_intel_rows", GGML_TYPE_Q8_0 },
    };
    const uint32_t widths[] = { 32, 64, 96, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 12288, 17408, 21504 };
    init_gpu();
    for (unsigned k = 0; k < sizeof kernels / sizeof kernels[0]; k++)
        for (unsigned s = 0; s < sizeof widths / sizeof widths[0]; s++)
            for (int accumulate = 0; accumulate < 2; accumulate++)
                run_case(kernels[k], widths[s], widths[s], accumulate, accumulate, bench);
    // The production dispatcher also loads fixed 64/128-block variants.
    for (unsigned k = 0; k < sizeof kernels / sizeof kernels[0]; k++)
        for (uint32_t blocks = 64; blocks <= 128; blocks *= 2)
            for (int accumulate = 0; accumulate < 2; accumulate++)
                run_case_specialized(kernels[k], blocks * 32, blocks * 32, accumulate, accumulate, bench, blocks);
    check(vkDeviceWaitIdle(device));
    vkDestroyCommandPool(device, command_pool, NULL);
    vkDestroyDevice(device, NULL); vkDestroyInstance(instance, NULL); close(gpu_lock);
    ggml_quantize_free();
    printf("PASS: %u Q8 native-reference/offset/accumulation/guard cases\n", cases);
    return 0;
}
