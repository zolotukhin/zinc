//! Backend-specific CLI help, rendered without initializing a GPU.
//! @section CLI & Entrypoints
//! Renders the compiled backend's device selection, context defaults and serving
//! capabilities. Pure string generation keeps all four variants CPU-testable.
const std = @import("std");

pub const Backend = enum { vulkan, metal, rocm, cuda };

/// Keep short and full help on the same source text. The ROCm/CUDA orchestration
/// has different defaults and endpoints from the Vulkan/Metal server.
pub fn render(comptime backend: Backend, full: bool) []const u8 {
    const accelerator = backend == .rocm or backend == .cuda;
    const identity = switch (backend) {
        .vulkan => "Vulkan (Linux)",
        .metal => "Metal (macOS; preview)",
        .rocm => "ROCm/HIP (Linux)",
        .cuda => "CUDA (Linux/WSL2; experimental)",
    };
    const device_options = switch (backend) {
        .vulkan => "  -d, --device <id>        Vulkan device index (default: auto, prefers discrete GPU)\n",
        .metal => "  -d, --device <id>        Metal device index (default: 0)\n",
        .rocm => "  Device selection: set ROCR_VISIBLE_DEVICES before starting ZINC; --device is ignored.\n",
        .cuda => "  Device selection: set CUDA_VISIBLE_DEVICES before starting ZINC; --device is ignored.\n",
    };
    const context_option = if (accelerator)
        "  -c, --context <size>     Context tokens per slot (default: 2048; 0 also uses 2048)\n"
    else
        "  -c, --context <size>     Context tokens (default: auto, sized to GPU memory; 0 = auto)\n";
    const serving_options = if (accelerator)
        \\  --parallel <n>           Request slots (default: 4); supported models batch decode
        \\                           Gemma MoE uses one slot; single-slot MTP needs --parallel 1
        \\  chat                     Start the HTTP chat API on port 9090; no built-in browser UI
        \\
    else
        \\  --parallel <n>           Requests in flight (default: 4); generation is serialized
        \\  chat                     Start the built-in browser chat (default port: 9090)
        \\
    ;
    const common =
        "ZINC — Zig INferenCe Engine for consumer GPUs and Apple Silicon\n" ++
        "Backend: " ++ identity ++ "\n\n" ++
        \\Usage:
        \\  zinc -m <model.gguf> --prompt "Hello"
        \\  zinc -m <model.gguf> [-p 8080]
        \\  zinc chat [-m <model.gguf> | --model-id <id>] [-p 9090]
        \\  zinc --model-id <id> [--prompt "Hello"]
        \\  zinc -hf <owner/model[:quant]> [--prompt "Hello"]
        \\  zinc --check [-m <model.gguf> | --model-id <id>]
        \\  zinc model <list|pull|use|active|rm> [args]
        \\
        \\Common options:
        \\  -m, --model <path>       GGUF model file to load
        \\  --model-id <id>          Managed model id from the local catalog/cache
        \\  -hf, --hf-repo <spec>    Hugging Face repo (owner/model[:quant]) to download and load
        \\  --prompt <text>          Run one prompt in CLI mode instead of starting the server
        \\  --chat                   Apply the model chat template to --prompt
        \\  --system-prompt <text>   Add an explicit system turn (requires --chat)
        \\  --raw                    Do not auto-apply chat templates to --prompt
        \\  -n, --max-tokens <n>     Max generated tokens in CLI mode (default: 256)
        \\
        ++ device_options ++ context_option ++
        \\
        \\Server options:
        \\  -p, --port <port>        Server port (default: 8080)
        \\
        ++ serving_options ++
        \\
        \\Model management:
        \\  model list [--all] [--json] List managed models (--json for machine-readable output)
        \\  model pull <id>          Download a supported managed model into the local cache
        \\  model use <id>           Set the active managed model for future runs
        \\  model active             Print the active managed model
        \\  model rm [-f] <id>       Remove a cached model (catalog or hf-- id)
        \\
        \\Diagnostics:
        \\  --check                  Run system diagnostics and verify dependencies
        \\  --version                Show version and build metadata
        \\  -h, --help               Show the short help
        \\  --help-all               Show the full help
        \\
        ;
    const profiling_option = if (accelerator) "" else "  --profile                Runtime profiling (per-dispatch Vulkan; Metal phase summary)\n";
    const extended =
        \\Analysis and developer options:
        \\  --graph-report <path>    Write decode-graph analysis JSON report
        \\  --graph-dot <path>       Write decode-graph Graphviz DOT from GGUF metadata
        \\
    ++ profiling_option ++
        \\  --debug                  Enable verbose debug logging
        \\  --kv-quant <bits>        Reserved compatibility flag (0/2/3/4); no runtime effect
        \\
        \\Backend capabilities and KV configuration: https://zolotukhin.ai/zinc/docs/api/#backend-coverage
        \\
    ;
    return if (full) common ++ extended else common ++ "Use `--help-all` for analysis and compatibility flags.\n";
}

test "help describes each backend without promising another backend's features" {
    inline for (.{ Backend.vulkan, Backend.metal, Backend.rocm, Backend.cuda }) |backend| {
        inline for (.{ false, true }) |full| {
            const text = render(backend, full);
            if (backend == .rocm or backend == .cuda) {
                try std.testing.expect(std.mem.indexOf(u8, text, "no built-in browser UI") != null);
                try std.testing.expect(std.mem.indexOf(u8, text, "default: 2048") != null);
                try std.testing.expect(std.mem.indexOf(u8, text, "generation is serialized") == null);
                try std.testing.expect(std.mem.indexOf(u8, text, "--profile") == null);
                const selector = if (backend == .rocm) "ROCR_VISIBLE_DEVICES" else "CUDA_VISIBLE_DEVICES";
                try std.testing.expect(std.mem.indexOf(u8, text, selector) != null);
            } else {
                try std.testing.expect(std.mem.indexOf(u8, text, "generation is serialized") != null);
                try std.testing.expect(std.mem.indexOf(u8, text, "Start the built-in browser chat") != null);
            }
            try std.testing.expectEqual(full, std.mem.indexOf(u8, text, "--kv-quant") != null);
            if (full) try std.testing.expect(std.mem.indexOf(u8, text, "no runtime effect") != null);
        }
    }
}
