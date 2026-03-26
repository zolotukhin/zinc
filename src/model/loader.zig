//! Build runtime model state from GGUF metadata and GPU-resident tensor buffers.
//! @section Model Format & Loading
//! This module translates an on-disk GGUF file into the normalized model
//! configuration and uploaded tensors consumed by the inference runtime.
const std = @import("std");
const gguf = @import("gguf.zig");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const buffer_mod = @import("../vulkan/buffer.zig");
const CommandPool = @import("../vulkan/command.zig").CommandPool;

const log = std.log.scoped(.loader);

/// Supported model families inferred from GGUF architecture metadata.
pub const Architecture = enum {
    llama,
    mistral,
    qwen2,
    qwen2_moe,
    mamba,
    jamba,
    unknown,
};

/// Normalized model dimensions and routing metadata extracted from GGUF fields.
pub const ModelConfig = struct {
    architecture: Architecture,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
    vocab_size: u32,
    context_length: u32,
    rope_freq_base: f32,
    // MoE fields
    n_experts: u32,
    n_experts_used: u32,
};

/// A tensor descriptor paired with the GPU buffer that stores its contents.
pub const LoadedTensor = struct {
    info: gguf.TensorInfo,
    gpu_buffer: Buffer,
};

/// Runtime model state backed by a memory-mapped GGUF file and uploaded tensor buffers.
pub const Model = struct {
    config: ModelConfig,
    gguf_file: gguf.GGUFFile,
    tensors: std.ArrayList(LoadedTensor),
    mmap_data: ?[]align(std.heap.page_size_min) const u8,
    mmap_file: ?std.fs.File,
    allocator: std.mem.Allocator,

    /// Release tensor buffers, GGUF metadata, and the backing file mapping owned by the model.
    /// @param self Model instance to tear down in place.
    /// @param instance Active Vulkan instance that created the device resources.
    pub fn deinit(self: *Model, instance: *const Instance) void {
        _ = instance;
        for (self.tensors.items) |*t| {
            var buf = t.gpu_buffer;
            buf.deinit();
        }
        self.tensors.deinit(self.allocator);

        if (self.mmap_data) |data| {
            std.posix.munmap(data);
        }
        if (self.mmap_file) |f| {
            var file = f;
            file.close();
        }

        self.gguf_file.deinit();
        self.* = undefined;
    }
};

/// Parse architecture string from GGUF metadata.
fn parseArchitecture(arch_str: []const u8) Architecture {
    if (std.mem.eql(u8, arch_str, "llama")) return .llama;
    if (std.mem.eql(u8, arch_str, "mistral")) return .mistral;
    if (std.mem.eql(u8, arch_str, "qwen2")) return .qwen2;
    if (std.mem.eql(u8, arch_str, "qwen2moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen3moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen35moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "mamba")) return .mamba;
    if (std.mem.eql(u8, arch_str, "jamba")) return .jamba;
    return .unknown;
}

/// Extract model configuration from GGUF metadata.
fn extractConfig(gf: *const gguf.GGUFFile) ModelConfig {
    const arch_str = gf.getString("general.architecture") orelse "unknown";
    const arch = parseArchitecture(arch_str);
    const prefix = arch_str;

    // Helper to look up arch-prefixed metadata keys
    var key_buf: [128]u8 = undefined;

    const n_layers = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.block_count", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };

    const n_heads = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.attention.head_count", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };

    const n_kv_heads = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.attention.head_count_kv", .{prefix}) catch break :blk n_heads;
        break :blk gf.getU32(key) orelse n_heads;
    };

    const hidden_dim = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.embedding_length", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };

    const head_dim = if (n_heads > 0) hidden_dim / n_heads else 0;

    const intermediate_dim = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.feed_forward_length", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };

    const vocab_size = blk: {
        // Try metadata first
        const key = std.fmt.bufPrint(&key_buf, "{s}.vocab_size", .{prefix}) catch break :blk @as(u32, 0);
        const from_meta = gf.getU32(key);
        if (from_meta) |v| if (v > 0) break :blk v;
        // Infer from output.weight or token_embd.weight tensor
        if (gf.findTensor("output.weight")) |t| break :blk @as(u32, @intCast(t.dims[1]));
        if (gf.findTensor("token_embd.weight")) |t| break :blk @as(u32, @intCast(t.dims[1]));
        break :blk @as(u32, 0);
    };

    const context_length = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.context_length", .{prefix}) catch break :blk @as(u32, 4096);
        break :blk gf.getU32(key) orelse 4096;
    };

    const n_experts = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.expert_count", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };

    const n_experts_used = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.expert_used_count", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };

    log.info("Architecture: {s} | {d} layers | {d} heads ({d} KV) | dim {d} | vocab {d}", .{
        arch_str, n_layers, n_heads, n_kv_heads, hidden_dim, vocab_size,
    });

    return ModelConfig{
        .architecture = arch,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .hidden_dim = hidden_dim,
        .intermediate_dim = intermediate_dim,
        .vocab_size = vocab_size,
        .context_length = context_length,
        .rope_freq_base = 10000.0, // default, can be overridden
        .n_experts = n_experts,
        .n_experts_used = n_experts_used,
    };
}

/// Load a GGUF model: memory-map the file, parse headers, and DMA tensors to GPU VRAM.
/// @param path Path to the GGUF file on disk.
/// @param instance Active Vulkan instance used for buffer allocation.
/// @param cmd_pool Command pool used for staging copy operations.
/// @param allocator Allocator used for metadata, tensor lists, and temporary state.
/// @returns A fully populated Model with parsed metadata and uploaded tensors.
pub fn load(
    path: []const u8,
    instance: *const Instance,
    cmd_pool: *const CommandPool,
    allocator: std.mem.Allocator,
) !Model {
    log.info("Loading model: {s}", .{path});

    // Open and memory-map the file
    const file = try std.fs.cwd().openFile(path, .{});
    errdefer file.close();

    const stat = try file.stat();
    const file_size = stat.size;
    log.info("File size: {d} MB", .{file_size / (1024 * 1024)});

    const mmap_data = try std.posix.mmap(
        null,
        file_size,
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    errdefer std.posix.munmap(mmap_data);

    // Parse GGUF headers
    var gf = try gguf.parse(mmap_data, allocator);
    errdefer gf.deinit();

    const config = extractConfig(&gf);

    // Load tensors to GPU
    var loaded_tensors: std.ArrayList(LoadedTensor) = .{};
    errdefer {
        for (loaded_tensors.items) |*t| {
            var buf = t.gpu_buffer;
            buf.deinit();
        }
        loaded_tensors.deinit(allocator);
    }

    var total_vram: u64 = 0;
    for (gf.tensors.items) |tensor_info| {
        const tensor_size = tensor_info.sizeBytes();
        const data_offset = gf.tensor_data_offset + tensor_info.offset;

        // Create device-local buffer
        var gpu_buf = try Buffer.initDeviceLocal(
            instance,
            tensor_size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        );
        errdefer gpu_buf.deinit();

        // Stage and copy data to GPU
        const src_data = mmap_data[data_offset..][0..@intCast(tensor_size)];
        var staging = try Buffer.initStaging(instance, tensor_size);
        defer staging.deinit();

        staging.upload(src_data);
        try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &gpu_buf, tensor_size);

        try loaded_tensors.append(allocator, .{
            .info = tensor_info,
            .gpu_buffer = gpu_buf,
        });

        total_vram += tensor_size;
    }

    log.info("Loaded {d} tensors | {d} MB VRAM", .{
        loaded_tensors.items.len,
        total_vram / (1024 * 1024),
    });

    return Model{
        .config = config,
        .gguf_file = gf,
        .tensors = loaded_tensors,
        .mmap_data = mmap_data,
        .mmap_file = file,
        .allocator = allocator,
    };
}

test "parseArchitecture" {
    try std.testing.expectEqual(Architecture.llama, parseArchitecture("llama"));
    try std.testing.expectEqual(Architecture.qwen2, parseArchitecture("qwen2"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen2moe"));
    try std.testing.expectEqual(Architecture.mamba, parseArchitecture("mamba"));
    try std.testing.expectEqual(Architecture.unknown, parseArchitecture("gpt2"));
}
