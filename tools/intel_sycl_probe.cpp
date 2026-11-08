// Standalone native-stack bring-up, not a ZINC inference backend or benchmark.
// Build with Intel icpx -fsycl. Run only on the remote Intel test worker.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr std::size_t widths[] = {
    1, 31, 32, 63, 64, 65, 255, 256, 257, 1023, 1024, 65537, 1048583,
};
static constexpr std::size_t guard = 32;
static constexpr float sentinel = 123456.0f;

static std::string quoted(const std::string &value) {
    std::string out = "\"";
    const char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out += '\\';
        if (c < 32) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else out += static_cast<char>(c);
    }
    return out + '"';
}

struct DeviceBuffer {
    sycl::queue &queue;
    float *data;
    DeviceBuffer(sycl::queue &q, std::size_t count)
        : queue(q), data(sycl::malloc_device<float>(count, q)) {
        if (!data) throw std::runtime_error("device allocation failed");
    }
    ~DeviceBuffer() { sycl::free(data, queue); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

static void check_case(sycl::queue &queue, std::size_t count, unsigned round) {
    std::vector<float> x(count), y(count);
    std::vector<float> initial(count + 2 * guard, sentinel);
    std::vector<float> actual(initial.size()), intermediate(initial.size());
    for (std::size_t i = 0; i < count; ++i) {
        // Binary fractions make the independent host answer exactly representable.
        x[i] = static_cast<float>(static_cast<int>((i + round) % 257) - 128) / 16;
        y[i] = static_cast<float>(static_cast<int>((i * 7 + round) % 127) - 63) / 8;
    }
    DeviceBuffer dx(queue, count), dy(queue, count);
    DeviceBuffer temporary(queue, initial.size()), output(queue, initial.size());
    try {
        auto a = queue.memcpy(dx.data, x.data(), count * sizeof(float));
        auto b = queue.memcpy(dy.data, y.data(), count * sizeof(float));
        auto c = queue.memcpy(temporary.data, initial.data(), initial.size() * sizeof(float));
        auto d = queue.memcpy(output.data, initial.data(), initial.size() * sizeof(float));
        float *px = dx.data, *py = dy.data, *pt = temporary.data, *po = output.data;
        const std::size_t global = ((count + 63) / 64) * 64;
        auto first = queue.submit([&](sycl::handler &handler) {
            handler.depends_on({a, b, c});
            handler.parallel_for(sycl::nd_range<1>{global, 64}, [=](sycl::nd_item<1> item) {
                auto i = item.get_global_linear_id();
                if (i < count) pt[guard + i] = 2 * px[i] + py[i];
            });
        });
        auto second = queue.submit([&](sycl::handler &handler) {
            // Test explicit event dependencies on an out-of-order queue too.
            handler.depends_on({first, d});
            handler.parallel_for(sycl::nd_range<1>{global, 64}, [=](sycl::nd_item<1> item) {
                auto i = item.get_global_linear_id();
                if (i < count) po[guard + i] = pt[guard + i] - 3 * py[i];
            });
        });
        auto copied = queue.submit([&](sycl::handler &handler) {
            handler.depends_on(second);
            handler.memcpy(actual.data(), output.data, actual.size() * sizeof(float));
        });
        auto copied_temporary = queue.submit([&](sycl::handler &handler) {
            handler.depends_on(first);
            handler.memcpy(intermediate.data(), temporary.data, intermediate.size() * sizeof(float));
        });
        copied.wait_and_throw();
        copied_temporary.wait_and_throw();
        queue.wait_and_throw();
        for (std::size_t i = 0; i < count; ++i) {
            const double expected = 2.0 * x[i] - 2.0 * y[i];
            const double expected_temporary = 2.0 * x[i] + y[i];
            if (!std::isfinite(actual[guard + i]) || actual[guard + i] != expected ||
                !std::isfinite(intermediate[guard + i]) || intermediate[guard + i] != expected_temporary)
                throw std::runtime_error("native arithmetic or copy mismatch");
        }
        for (std::size_t i = 0; i < guard; ++i) {
            if (actual[i] != sentinel || actual[guard + count + i] != sentinel ||
                intermediate[i] != sentinel || intermediate[guard + count + i] != sentinel)
                throw std::runtime_error("native output guard changed");
        }
    } catch (...) {
        // Finish queued uses before freeing their USM allocations on any error.
        queue.wait();
        throw;
    }
}

int main(int argc, char **) {
    try {
        if (argc != 1) throw std::runtime_error("probe accepts no arguments");
        std::vector<sycl::device> devices;
        for (const auto &device : sycl::device::get_devices(sycl::info::device_type::gpu))
            if (device.get_backend() == sycl::backend::ext_oneapi_level_zero &&
                device.get_info<sycl::info::device::vendor_id>() == 0x8086)
                devices.push_back(device);
        // A selector may narrow multiple GPUs, but picking the first is ambiguous.
        if (devices.size() != 1) throw std::runtime_error("select exactly one Intel Level Zero GPU with ONEAPI_DEVICE_SELECTOR");
        const auto &device = devices.front();
        if (!device.has(sycl::aspect::usm_device_allocations)) throw std::runtime_error("device USM unavailable");
        auto asynchronous = [](sycl::exception_list errors) {
            if (errors.begin() != errors.end()) std::rethrow_exception(*errors.begin());
        };
        std::uint64_t cases = 0, elements = 0;
        for (bool ordered : {false, true}) {
            sycl::queue queue(device, asynchronous, ordered
                ? sycl::property_list{sycl::property::queue::in_order{}} : sycl::property_list{});
            for (unsigned round = 0; round < 3; ++round)
                for (auto count : widths) {
                    check_case(queue, count, round);
                    ++cases;
                    elements += count * 2;
                }
        }
        std::cout << "{\"passed\":true,\"backend\":\"level_zero\",\"device_type\":\"gpu\",\"vendor_id\":32902,"
                  << "\"device_name\":" << quoted(device.get_info<sycl::info::device::name>())
                  << ",\"driver_version\":" << quoted(device.get_info<sycl::info::device::driver_version>())
                  << ",\"cases\":" << cases << ",\"elements_checked\":" << elements
                  << ",\"guards_checked\":" << cases * guard * 4 << ",\"max_absolute_error\":0}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cout << "{\"passed\":false,\"error\":" << quoted(error.what()) << "}\n";
        return 1;
    }
}
