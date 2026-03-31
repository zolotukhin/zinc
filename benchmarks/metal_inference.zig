const std = @import("std");
const support = @import("zinc_bench_support");
const metal_device = support.metal_device;
const metal_loader = support.metal_loader;
const tokenizer_mod = support.tokenizer_mod;
const forward_metal = support.forward_metal;

pub const std_options = std.Options{
    .log_level = .warn,
};

const Config = struct {
    model_path: ?[]const u8 = null,
    prompt: []const u8 = "The capital of France is",
    max_tokens: u32 = 128,
    runs: u32 = 3,
    warmup_runs: u32 = 1,
    chat: bool = false,
    device_index: u32 = 0,
    show_help: bool = false,
};

const PreparedPrompt = struct {
    text: []const u8,
    owned_buf: ?[]u8 = null,

    fn deinit(self: *PreparedPrompt, allocator: std.mem.Allocator) void {
        if (self.owned_buf) |buf| allocator.free(buf);
        self.* = undefined;
    }
};

const SummaryStats = struct {
    min: f64,
    median: f64,
    max: f64,
    average: f64,
};

fn helpText() []const u8 {
    return 
    \\Usage: zinc-bench-metal -m <model.gguf> [options]
    \\
    \\Runs a ReleaseFast end-to-end Metal inference benchmark on Apple Silicon.
    \\
    \\Options:
    \\  -m, --model <path>       GGUF model path (required)
    \\  --prompt <text>          Prompt text (default: "The capital of France is")
    \\  -n, --max-tokens <n>     Decode tokens per run (default: 128)
    \\  --runs <n>               Measured runs (default: 3)
    \\  --warmup <n>             Warmup runs before measurement (default: 1)
    \\  --chat                   Apply the model chat template to the prompt
    \\  -d, --device <index>     Metal device index (default: 0)
    \\  -h, --help               Show this help text
    \\
    \\Example:
    \\  zig build bench-metal -- -m /Users/zolotukhin/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
    \\
    ;
}

fn parseU32(arg: []const u8) !u32 {
    return std.fmt.parseUnsigned(u32, arg, 10);
}

fn parseArgs(args: []const [:0]const u8) !Config {
    var config = Config{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            config.show_help = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingModelPath;
            config.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingPrompt;
            config.prompt = args[i];
        } else if (std.mem.eql(u8, arg, "-n") or std.mem.eql(u8, arg, "--max-tokens")) {
            i += 1;
            if (i >= args.len) return error.MissingMaxTokens;
            config.max_tokens = try parseU32(args[i]);
        } else if (std.mem.eql(u8, arg, "--runs")) {
            i += 1;
            if (i >= args.len) return error.MissingRuns;
            config.runs = try parseU32(args[i]);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            i += 1;
            if (i >= args.len) return error.MissingWarmupRuns;
            config.warmup_runs = try parseU32(args[i]);
        } else if (std.mem.eql(u8, arg, "--chat")) {
            config.chat = true;
        } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--device")) {
            i += 1;
            if (i >= args.len) return error.MissingDeviceIndex;
            config.device_index = try parseU32(args[i]);
        } else {
            return error.UnknownArgument;
        }
    }

    if (!config.show_help and config.model_path == null) {
        return error.MissingModelPath;
    }
    if (config.runs == 0) return error.InvalidRuns;
    return config;
}

fn preparePrompt(tokenizer: *const tokenizer_mod.Tokenizer, prompt: []const u8, chat: bool, allocator: std.mem.Allocator) !PreparedPrompt {
    if (!chat) return .{ .text = prompt };

    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{prompt};
    const buf = try allocator.alloc(u8, prompt.len + 256);
    errdefer allocator.free(buf);

    const formatted = try tokenizer.applyChatTemplate(&roles, &contents, buf);
    return .{
        .text = formatted,
        .owned_buf = buf,
    };
}

fn trimOutputText(text: []const u8, chat: bool) []const u8 {
    if (!chat) return text;
    if (std.mem.indexOf(u8, text, "<|im_end|>")) |stop_pos| {
        return text[0..stop_pos];
    }
    return text;
}

fn decodeOutputText(
    tokenizer: *const tokenizer_mod.Tokenizer,
    output_tokens: []const u32,
    chat: bool,
    allocator: std.mem.Allocator,
) ![]u8 {
    var text_buf: std.ArrayList(u8) = .{};
    defer text_buf.deinit(allocator);

    for (output_tokens) |tid| {
        var scratch: [256]u8 = undefined;
        const decoded = tokenizer.decodeToken(tid, &scratch);
        if (decoded.len > 0) {
            try text_buf.appendSlice(allocator, decoded);
        }
    }

    const trimmed = trimOutputText(text_buf.items, chat);
    return try allocator.dupe(u8, trimmed);
}

fn previewText(text: []const u8, limit: usize) []const u8 {
    return text[0..@min(text.len, limit)];
}

fn insertionSort(values: []f64) void {
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        const key = values[i];
        var j = i;
        while (j > 0 and values[j - 1] > key) : (j -= 1) {
            values[j] = values[j - 1];
        }
        values[j] = key;
    }
}

fn computeSummaryStats(allocator: std.mem.Allocator, values: []const f64) !SummaryStats {
    if (values.len == 0) return error.EmptyInput;

    const sorted = try allocator.dupe(f64, values);
    defer allocator.free(sorted);
    insertionSort(sorted);

    var sum: f64 = 0.0;
    for (values) |v| sum += v;

    return .{
        .min = sorted[0],
        .median = if (sorted.len % 2 == 1)
            sorted[sorted.len / 2]
        else
            (sorted[(sorted.len / 2) - 1] + sorted[sorted.len / 2]) / 2.0,
        .max = sorted[sorted.len - 1],
        .average = sum / @as(f64, @floatFromInt(values.len)),
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const config = parseArgs(args) catch |err| {
        std.fs.File.stderr().writeAll(helpText()) catch {};
        return err;
    };

    if (config.show_help) {
        try std.fs.File.stdout().writeAll(helpText());
        return;
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);

    var device = try metal_device.MetalDevice.init(allocator, config.device_index);
    defer device.deinit();

    var model = try metal_loader.load(config.model_path.?, device.ctx, allocator);
    defer model.deinit();

    var tokenizer = try tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator);
    defer tokenizer.deinit();

    var prepared_prompt = try preparePrompt(&tokenizer, config.prompt, config.chat, allocator);
    defer prepared_prompt.deinit(allocator);

    const prompt_tokens = try tokenizer.encodePrompt(prepared_prompt.text, allocator);
    defer allocator.free(prompt_tokens);

    var engine = try forward_metal.InferenceEngine.init(&model, &device, allocator, false);
    defer engine.deinit();

    try stdout.interface.print("Metal inference benchmark\n", .{});
    try stdout.interface.print("Model: {s}\n", .{config.model_path.?});
    try stdout.interface.print(
        "GPU: {s} | unified={} | tgmem={d} KiB | working-set={d} GiB\n",
        .{
            @tagName(device.chip),
            device.hasUnifiedMemory(),
            device.maxThreadgroupMemoryLength() / 1024,
            device.recommendedMaxWorkingSetSize() / (1024 * 1024 * 1024),
        },
    );
    try stdout.interface.print(
        "Prompt tokens: {d} | decode tokens: {d} | warmup: {d} | runs: {d} | chat={}\n\n",
        .{ prompt_tokens.len, config.max_tokens, config.warmup_runs, config.runs, config.chat },
    );

    var warmup_idx: u32 = 0;
    while (warmup_idx < config.warmup_runs) : (warmup_idx += 1) {
        var warmup = try forward_metal.generateWithMetrics(&engine, prompt_tokens, config.max_tokens, tokenizer.eosId(), allocator);
        defer warmup.deinit(allocator);
        try stdout.interface.print(
            "Warmup {d}: prefill {d:.1} tok/s | decode {d:.2} tok/s | {d:.1} ms/tok | output {d} tokens\n",
            .{
                warmup_idx + 1,
                warmup.metrics.prefill_tps,
                warmup.metrics.decode_tps,
                warmup.metrics.ms_per_token,
                warmup.metrics.generated_tokens,
            },
        );
    }
    if (config.warmup_runs > 0) {
        try stdout.interface.writeAll("\n");
    }

    const prefill_tps = try allocator.alloc(f64, config.runs);
    defer allocator.free(prefill_tps);
    const decode_tps = try allocator.alloc(f64, config.runs);
    defer allocator.free(decode_tps);
    const ms_per_token = try allocator.alloc(f64, config.runs);
    defer allocator.free(ms_per_token);

    var preview_output: ?[]u8 = null;
    defer if (preview_output) |buf| allocator.free(buf);

    var run_idx: u32 = 0;
    while (run_idx < config.runs) : (run_idx += 1) {
        var run = try forward_metal.generateWithMetrics(&engine, prompt_tokens, config.max_tokens, tokenizer.eosId(), allocator);
        defer run.deinit(allocator);

        prefill_tps[run_idx] = run.metrics.prefill_tps;
        decode_tps[run_idx] = run.metrics.decode_tps;
        ms_per_token[run_idx] = run.metrics.ms_per_token;

        if (preview_output == null) {
            preview_output = try decodeOutputText(&tokenizer, run.output_tokens, config.chat, allocator);
        }

        try stdout.interface.print(
            "Run {d}: prefill {d:.1} tok/s | decode {d:.2} tok/s | {d:.1} ms/tok | output {d} tokens\n",
            .{
                run_idx + 1,
                run.metrics.prefill_tps,
                run.metrics.decode_tps,
                run.metrics.ms_per_token,
                run.metrics.generated_tokens,
            },
        );
    }

    const prefill_stats = try computeSummaryStats(allocator, prefill_tps);
    const decode_stats = try computeSummaryStats(allocator, decode_tps);
    const ms_stats = try computeSummaryStats(allocator, ms_per_token);

    try stdout.interface.writeAll("\nSummary\n");
    try stdout.interface.print(
        "Prefill tok/s: median {d:.1} | avg {d:.1} | best {d:.1} | worst {d:.1}\n",
        .{ prefill_stats.median, prefill_stats.average, prefill_stats.max, prefill_stats.min },
    );
    try stdout.interface.print(
        "Decode tok/s: median {d:.2} | avg {d:.2} | best {d:.2} | worst {d:.2}\n",
        .{ decode_stats.median, decode_stats.average, decode_stats.max, decode_stats.min },
    );
    try stdout.interface.print(
        "Decode ms/tok: median {d:.1} | avg {d:.1} | best {d:.1} | worst {d:.1}\n",
        .{ ms_stats.median, ms_stats.average, ms_stats.min, ms_stats.max },
    );
    if (preview_output) |text| {
        try stdout.interface.print("Output preview: {s}\n", .{previewText(text, 160)});
    }
    try stdout.interface.flush();
}
