// Remote-only numerical GPU probe; no model or inference throughput baseline.
// Repeat llama.cpp's F32 MoE-router layout with and without oneDNN's explicit
// deterministic attribute. The host dot product is a numerical check only.
// Build with icpx -std=c++17 -fsycl -O2 and the isolated SDK's -ldnnl.
// Execute through an owned worker process and retain stderr separately.
#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static std::uint64_t fingerprint(const std::vector<float> &values) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto *bytes = reinterpret_cast<const unsigned char *>(values.data());
    for (std::size_t i = 0; i < values.size() * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void probe(sycl::queue &queue) {
    constexpr int m = 256, n = 58, k = 2048;
    constexpr int guard = 32;
    constexpr float sentinel = 123456.0f;
    auto engine = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto stream = dnnl::sycl_interop::make_stream(engine, queue);
    float *a = nullptr, *b = nullptr, *c = nullptr;
    void *scratch = nullptr;
    auto cleanup = [&]() {
        // No queued primitive may retain a buffer when it is released, even
        // after a submission or device error. The worker also owns the process.
        queue.wait();
        if (scratch) sycl::free(scratch, queue);
        if (c) sycl::free(c, queue);
        if (b) sycl::free(b, queue);
        if (a) sycl::free(a, queue);
    };
    try {
        a = sycl::malloc_shared<float>(m * k, queue);
        b = sycl::malloc_shared<float>(k * n, queue);
        c = sycl::malloc_shared<float>(m * n + 2 * guard, queue);
        if (!a || !b || !c) throw std::runtime_error("allocation failed");
        std::uint32_t state = 0x21937;
        auto value = [&]() {
            state = 1664525u * state + 1013904223u;
            return (static_cast<int>(state >> 20) - 2048) / 2048.0f;
        };
        for (int i = 0; i < m * k; ++i) a[i] = value();
        for (int i = 0; i < k * n; ++i) b[i] = value();
        std::vector<double> reference(m * n);
        for (int column = 0; column < n; ++column) {
            for (int row = 0; row < m; ++row) {
                double sum = 0;
                for (int inner = 0; inner < k; ++inner)
                    sum += static_cast<double>(a[row * k + inner]) * b[column * k + inner];
                reference[column * m + row] = sum;
            }
        }
        using dt = dnnl::memory::data_type;
        using dims = dnnl::memory::dims;
        // These strides match DnnlGemmWrapper::row_gemm's batched descriptors.
        auto ad = dnnl::memory::desc(dims{1, m, k}, dt::f32, dims{m * k, k, 1});
        auto bd = dnnl::memory::desc(dims{1, k, n}, dt::f32, dims{n * k, 1, k});
        auto cd = dnnl::memory::desc(dims{1, m, n}, dt::f32, dims{m * n, 1, m});
        for (bool deterministic : {false, true}) {
            std::vector<float> first;
            for (int run = 0; run < 32; ++run) {
                std::fill(c, c + m * n + 2 * guard, sentinel);
                dnnl::primitive_attr attr;
                attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
                attr.set_deterministic(deterministic);
                // Recreate descriptors/primitives as the upstream wrapper does.
                auto pd = dnnl::matmul::primitive_desc(engine, ad, bd, cd, attr);
                auto bytes = pd.scratchpad_desc().get_size();
                scratch = bytes ? sycl::malloc_device(bytes, queue) : nullptr;
                if (bytes && !scratch) throw std::runtime_error("scratchpad allocation failed");
                auto am = dnnl::memory(ad, engine, a);
                auto bm = dnnl::memory(bd, engine, b);
                auto cm = dnnl::memory(cd, engine, c + guard);
                auto sm = dnnl::memory(pd.scratchpad_desc(), engine, scratch);
                auto op = dnnl::matmul(pd);
                op.execute(stream, {{DNNL_ARG_SRC, am}, {DNNL_ARG_WEIGHTS, bm},
                                    {DNNL_ARG_DST, cm}, {DNNL_ARG_SCRATCHPAD, sm}});
                stream.wait();
                queue.wait_and_throw();
                bool guards = true;
                for (int i = 0; i < guard; ++i)
                    guards = guards && c[i] == sentinel && c[m * n + guard + i] == sentinel;
                std::vector<float> result(c + guard, c + guard + m * n);
                if (first.empty()) first = result;
                double max_error = 0, max_change = 0;
                std::size_t changed = 0, nonfinite = 0;
                for (std::size_t i = 0; i < result.size(); ++i) {
                    nonfinite += !std::isfinite(result[i]);
                    max_error = std::max(max_error, std::abs(static_cast<double>(result[i]) - reference[i]));
                    max_change = std::max(max_change, std::abs(static_cast<double>(result[i]) - first[i]));
                    changed += std::memcmp(&result[i], &first[i], sizeof(float)) != 0;
                }
                std::printf("{\"deterministic\":%s,\"run\":%d,\"fnv1a64\":\"%016llx\","
                            "\"max_absolute_error\":%.17g,\"max_change_from_first\":%.17g,"
                            "\"changed_elements\":%zu,\"nonfinite\":%zu,\"guards_intact\":%s}\n",
                            deterministic ? "true" : "false", run,
                            static_cast<unsigned long long>(fingerprint(result)), max_error,
                            max_change, changed, nonfinite, guards ? "true" : "false");
                std::fflush(stdout);
                if (scratch) sycl::free(scratch, queue);
                scratch = nullptr;
                if (!guards || nonfinite) throw std::runtime_error("numerical output or guard failure");
            }
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

int main(int argc, char **) {
    try {
        if (argc != 1) throw std::runtime_error("probe accepts no arguments");
        sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        if (queue.get_backend() != sycl::backend::ext_oneapi_level_zero ||
            queue.get_device().get_info<sycl::info::device::vendor_id>() != 0x8086)
            throw std::runtime_error("probe needs an Intel Level Zero GPU");
        probe(queue);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "probe failed: %s\n", error.what());
        return 1;
    }
}
