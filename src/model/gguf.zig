//! Parse GGUF container files and expose the metadata needed by the loader.
//! @section Model Format & Loading
//! The helpers in this module decode GGUF headers, metadata values, tensor
//! offsets, and GGML quantization information without copying the whole file.
const std = @import("std");

const log = std.log.scoped(.gguf);

/// Little-endian magic value expected at the start of every GGUF file.
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as bytes: 'G','G','U','F' → LE u32

/// GGUF container versions recognized by the parser.
pub const GGUFVersion = enum(u32) {
    v2 = 2,
    v3 = 3,
    _,
};

/// Primitive metadata value tags defined by the GGUF format.
pub const GGUFType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

/// GGML tensor storage and quantization formats referenced by GGUF tensors.
pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    _,

    /// Return the number of tensor elements encoded by one storage block.
    /// @param self GGML tensor format to inspect.
    /// @returns The number of logical elements represented by one block of this format.
    /// @note Quantized formats pack many elements into one block, while plain scalar types return `1`.
    pub fn blockSize(self: GGMLType) u32 {
        return switch (self) {
            .f32, .f16, .bf16, .f64 => 1,
            .i8, .i16, .i32, .i64 => 1,
            .q4_0, .q4_1 => 32,
            .q5_0, .q5_1 => 32,
            .q8_0, .q8_1 => 32,
            .q2_k => 256,
            .q3_k => 256,
            .q4_k => 256,
            .q5_k => 256,
            .q6_k => 256,
            .q8_k => 256,
            else => 1,
        };
    }

    /// Return the serialized byte width of one storage block.
    /// @param self GGML tensor format to inspect.
    /// @returns The number of on-disk bytes consumed by one block of this format.
    /// @note Use this together with `blockSize()` to convert element counts into tensor byte ranges.
    pub fn bytesPerBlock(self: GGMLType) u32 {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .bf16 => 2,
            .f64 => 8,
            .i8 => 1,
            .i16 => 2,
            .i32 => 4,
            .i64 => 8,
            .q4_0 => 18, // 32 * 4/8 + 2 (scale)
            .q4_1 => 20, // 32 * 4/8 + 2 + 2 (scale + min)
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34, // 32 + 2 (scale)
            .q8_1 => 36,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,
            else => 0,
        };
    }
};

/// Typed representation of a GGUF metadata value.
pub const MetadataValue = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_: bool,
    string: []const u8,
    uint64: u64,
    int64: i64,
    float64: f64,
    array: []MetadataValue,

    /// Interpret the metadata value as a string slice when the stored type is `.string`.
    /// @returns The string contents, or `null` when the value is not a string.
    pub fn asString(self: MetadataValue) ?[]const u8 {
        return switch (self) {
            .string => |s| s,
            else => null,
        };
    }

    /// Interpret the metadata value as an unsigned 32-bit integer when it fits.
    /// @returns A normalized `u32` or `null` when the stored value cannot be represented as one.
    pub fn asU32(self: MetadataValue) ?u32 {
        return switch (self) {
            .uint32 => |v| v,
            .int32 => |v| if (v >= 0) @intCast(v) else null,
            .uint64 => |v| if (v <= std.math.maxInt(u32)) @intCast(v) else null,
            else => null,
        };
    }

    /// Interpret the metadata value as an unsigned 64-bit integer when it fits.
    /// @returns A normalized `u64` or `null` when the stored value cannot be represented as one.
    pub fn asU64(self: MetadataValue) ?u64 {
        return switch (self) {
            .uint64 => |v| v,
            .uint32 => |v| v,
            .int64 => |v| if (v >= 0) @intCast(v) else null,
            else => null,
        };
    }

    /// Interpret the metadata value as a 32-bit floating-point number when possible.
    /// @returns An `f32` value or `null` when the stored value is not numeric.
    pub fn asF32(self: MetadataValue) ?f32 {
        return switch (self) {
            .float32 => |v| v,
            .float64 => |v| @floatCast(v),
            else => null,
        };
    }
};

/// Tensor descriptor read from the GGUF header.
pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    type_: GGMLType,
    offset: u64, // offset from start of tensor data section

    /// Multiply the active tensor dimensions to get the logical element count.
    /// @param self Tensor descriptor to inspect.
    /// @returns The total number of logical elements across the first `n_dims` entries in `dims`.
    pub fn numElements(self: *const TensorInfo) u64 {
        var n: u64 = 1;
        for (self.dims[0..self.n_dims]) |d| {
            n *= d;
        }
        return n;
    }

    /// Compute the serialized tensor byte size for the descriptor's GGML storage format.
    /// @param self Tensor descriptor to inspect.
    /// @returns The number of bytes occupied by the tensor payload, rounded up to whole quantization blocks.
    /// @note Quantized tensors round element counts up to a full block before multiplying by bytes-per-block.
    pub fn sizeBytes(self: *const TensorInfo) u64 {
        const n = self.numElements();
        const bs = self.type_.blockSize();
        const bpb = self.type_.bytesPerBlock();
        // Number of blocks, rounded up
        const blocks = (n + bs - 1) / bs;
        return blocks * bpb;
    }
};

/// Parsed GGUF header state, metadata map, and tensor table.
pub const GGUFFile = struct {
    version: GGUFVersion,
    tensor_count: u64,
    metadata: std.StringHashMapUnmanaged(MetadataValue),
    tensors: std.ArrayList(TensorInfo),
    tensor_data_offset: u64, // file offset where tensor data begins
    allocator: std.mem.Allocator,

    /// Release metadata keys, metadata payloads, and tensor names owned by the parsed file.
    /// @param self Parsed GGUF file to tear down in place.
    pub fn deinit(self: *GGUFFile) void {
        // Free metadata string values and keys
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            freeMetadataValue(self.allocator, entry.value_ptr.*);
        }
        self.metadata.deinit(self.allocator);

        // Free tensor names
        for (self.tensors.items) |t| {
            self.allocator.free(t.name);
        }
        self.tensors.deinit(self.allocator);
        self.* = undefined;
    }

    /// Look up a metadata string value by key.
    /// @param self Parsed GGUF file.
    /// @param key Metadata key to search for.
    /// @returns The stored string value when present and typed as `.string`.
    pub fn getString(self: *const GGUFFile, key: []const u8) ?[]const u8 {
        const val = self.metadata.get(key) orelse return null;
        return val.asString();
    }

    /// Look up a metadata value as `u32` when it can be normalized to that type.
    /// @param self Parsed GGUF file.
    /// @param key Metadata key to search for.
    /// @returns The normalized integer value when present.
    pub fn getU32(self: *const GGUFFile, key: []const u8) ?u32 {
        const val = self.metadata.get(key) orelse return null;
        return val.asU32();
    }

    /// Find a tensor descriptor by name.
    /// @param self Parsed GGUF file.
    /// @param name Tensor name to look up.
    /// @returns A pointer to the tensor descriptor when the tensor exists.
    pub fn findTensor(self: *const GGUFFile, name: []const u8) ?*const TensorInfo {
        for (self.tensors.items) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }
};

fn freeMetadataValue(allocator: std.mem.Allocator, val: MetadataValue) void {
    switch (val) {
        .string => |s| allocator.free(s),
        .array => |arr| {
            for (arr) |item| {
                freeMetadataValue(allocator, item);
            }
            allocator.free(arr);
        },
        else => {},
    }
}

/// Parse a GGUF file from a byte slice.
/// @param data Raw GGUF bytes, typically from a memory-mapped file.
/// @param allocator Allocator used for metadata strings, arrays, and tensor descriptors.
/// @returns A parsed GGUFFile that borrows from `data` for numeric fields and owns copied strings.
pub fn parse(data: []const u8, allocator: std.mem.Allocator) !GGUFFile {
    var reader = Reader{ .data = data, .pos = 0 };

    // Header
    const magic = reader.readU32();
    if (magic != GGUF_MAGIC) {
        log.err("Invalid GGUF magic: 0x{x:0>8} (expected 0x{x:0>8})", .{ magic, GGUF_MAGIC });
        return error.InvalidMagic;
    }

    const version: GGUFVersion = @enumFromInt(reader.readU32());
    const tensor_count = reader.readU64();
    const metadata_count = reader.readU64();

    log.info("GGUF v{d}: {d} tensors, {d} metadata entries", .{
        @intFromEnum(version), tensor_count, metadata_count,
    });

    // Parse metadata
    var metadata: std.StringHashMapUnmanaged(MetadataValue) = .{};
    errdefer metadata.deinit(allocator);

    for (0..metadata_count) |_| {
        const key = try reader.readString(allocator);
        errdefer allocator.free(key);
        const val = try reader.readMetadataValue(allocator);
        try metadata.put(allocator, key, val);
    }

    // Parse tensor descriptors
    var tensors: std.ArrayList(TensorInfo) = .{};
    errdefer tensors.deinit(allocator);

    for (0..tensor_count) |_| {
        const name = try reader.readString(allocator);
        errdefer allocator.free(name);

        const n_dims = reader.readU32();
        var dims: [4]u64 = .{ 1, 1, 1, 1 };
        for (0..n_dims) |d| {
            dims[d] = reader.readU64();
        }

        const type_: GGMLType = @enumFromInt(reader.readU32());
        const offset = reader.readU64();

        try tensors.append(allocator, .{
            .name = name,
            .n_dims = n_dims,
            .dims = dims,
            .type_ = type_,
            .offset = offset,
        });
    }

    // Tensor data starts at the next alignment boundary after all headers
    const alignment: u64 = blk: {
        if (metadata.get("general.alignment")) |val| {
            break :blk val.asU64() orelse 32;
        }
        break :blk 32;
    };
    const tensor_data_offset = (reader.pos + alignment - 1) & ~(alignment - 1);

    return GGUFFile{
        .version = version,
        .tensor_count = tensor_count,
        .metadata = metadata,
        .tensors = tensors,
        .tensor_data_offset = tensor_data_offset,
        .allocator = allocator,
    };
}

/// Simple sequential reader over a byte slice.
const Reader = struct {
    data: []const u8,
    pos: u64,

    fn readU8(self: *Reader) u8 {
        const val = self.data[@intCast(self.pos)];
        self.pos += 1;
        return val;
    }

    fn readU32(self: *Reader) u32 {
        const p: usize = @intCast(self.pos);
        const val = std.mem.readInt(u32, self.data[p..][0..4], .little);
        self.pos += 4;
        return val;
    }

    fn readI32(self: *Reader) i32 {
        const p: usize = @intCast(self.pos);
        const val = std.mem.readInt(i32, self.data[p..][0..4], .little);
        self.pos += 4;
        return val;
    }

    fn readU64(self: *Reader) u64 {
        const p: usize = @intCast(self.pos);
        const val = std.mem.readInt(u64, self.data[p..][0..8], .little);
        self.pos += 8;
        return val;
    }

    fn readI64(self: *Reader) i64 {
        const p: usize = @intCast(self.pos);
        const val = std.mem.readInt(i64, self.data[p..][0..8], .little);
        self.pos += 8;
        return val;
    }

    fn readF32(self: *Reader) f32 {
        const p: usize = @intCast(self.pos);
        const val = std.mem.readInt(u32, self.data[p..][0..4], .little);
        self.pos += 4;
        return @bitCast(val);
    }

    fn readF64(self: *Reader) f64 {
        const p: usize = @intCast(self.pos);
        const val = std.mem.readInt(u64, self.data[p..][0..8], .little);
        self.pos += 8;
        return @bitCast(val);
    }

    fn readString(self: *Reader, allocator: std.mem.Allocator) ![]const u8 {
        const len = self.readU64();
        const p: usize = @intCast(self.pos);
        const end: usize = @intCast(self.pos + len);
        const str = try allocator.dupe(u8, self.data[p..end]);
        self.pos += len;
        return str;
    }

    fn readMetadataValue(self: *Reader, allocator: std.mem.Allocator) !MetadataValue {
        const type_: GGUFType = @enumFromInt(self.readU32());
        return switch (type_) {
            .uint8 => .{ .uint8 = self.readU8() },
            .int8 => .{ .int8 = @bitCast(self.readU8()) },
            .uint16 => blk: {
                const val = std.mem.readInt(u16, self.data[@intCast(self.pos)..][0..2], .little);
                self.pos += 2;
                break :blk .{ .uint16 = val };
            },
            .int16 => blk: {
                const val = std.mem.readInt(i16, self.data[@intCast(self.pos)..][0..2], .little);
                self.pos += 2;
                break :blk .{ .int16 = val };
            },
            .uint32 => .{ .uint32 = self.readU32() },
            .int32 => .{ .int32 = self.readI32() },
            .float32 => .{ .float32 = self.readF32() },
            .bool_ => .{ .bool_ = self.readU8() != 0 },
            .string => .{ .string = try self.readString(allocator) },
            .uint64 => .{ .uint64 = self.readU64() },
            .int64 => .{ .int64 = self.readI64() },
            .float64 => .{ .float64 = self.readF64() },
            .array => blk: {
                const elem_type: GGUFType = @enumFromInt(self.readU32());
                const count = self.readU64();
                const items = try allocator.alloc(MetadataValue, @intCast(count));
                errdefer allocator.free(items);
                for (items) |*item| {
                    item.* = try self.readTypedValue(elem_type, allocator);
                }
                break :blk .{ .array = items };
            },
            _ => {
                log.err("Unknown GGUF metadata type: {d}", .{@intFromEnum(type_)});
                return error.UnknownMetadataType;
            },
        };
    }

    fn readTypedValue(self: *Reader, type_: GGUFType, allocator: std.mem.Allocator) !MetadataValue {
        return switch (type_) {
            .uint8 => .{ .uint8 = self.readU8() },
            .int8 => .{ .int8 = @bitCast(self.readU8()) },
            .uint16 => blk: {
                const val = std.mem.readInt(u16, self.data[@intCast(self.pos)..][0..2], .little);
                self.pos += 2;
                break :blk .{ .uint16 = val };
            },
            .int16 => blk: {
                const val = std.mem.readInt(i16, self.data[@intCast(self.pos)..][0..2], .little);
                self.pos += 2;
                break :blk .{ .int16 = val };
            },
            .uint32 => .{ .uint32 = self.readU32() },
            .int32 => .{ .int32 = self.readI32() },
            .float32 => .{ .float32 = self.readF32() },
            .bool_ => .{ .bool_ = self.readU8() != 0 },
            .string => .{ .string = try self.readString(allocator) },
            .uint64 => .{ .uint64 = self.readU64() },
            .int64 => .{ .int64 = self.readI64() },
            .float64 => .{ .float64 = self.readF64() },
            else => error.UnknownMetadataType,
        };
    }
};

test "GGMLType block sizes" {
    try std.testing.expectEqual(@as(u32, 256), GGMLType.q4_k.blockSize());
    try std.testing.expectEqual(@as(u32, 144), GGMLType.q4_k.bytesPerBlock());
    try std.testing.expectEqual(@as(u32, 32), GGMLType.q8_0.blockSize());
    try std.testing.expectEqual(@as(u32, 34), GGMLType.q8_0.bytesPerBlock());
    try std.testing.expectEqual(@as(u32, 1), GGMLType.f16.blockSize());
    try std.testing.expectEqual(@as(u32, 2), GGMLType.f16.bytesPerBlock());
}

test "TensorInfo sizeBytes" {
    const t = TensorInfo{
        .name = "test",
        .n_dims = 2,
        .dims = .{ 4096, 4096, 1, 1 },
        .type_ = .f16,
        .offset = 0,
    };
    // 4096 * 4096 * 2 bytes = 33554432
    try std.testing.expectEqual(@as(u64, 33554432), t.sizeBytes());
}

test "TensorInfo sizeBytes q4_k" {
    const t = TensorInfo{
        .name = "test",
        .n_dims = 2,
        .dims = .{ 4096, 4096, 1, 1 },
        .type_ = .q4_k,
        .offset = 0,
    };
    // 4096*4096 = 16777216 elements, 16777216/256 = 65536 blocks, 65536 * 144 = 9437184
    try std.testing.expectEqual(@as(u64, 9437184), t.sizeBytes());
}

test "MetadataValue conversions" {
    const v_str = MetadataValue{ .string = "hello" };
    try std.testing.expectEqualStrings("hello", v_str.asString().?);
    try std.testing.expect(v_str.asU32() == null);

    const v_u32 = MetadataValue{ .uint32 = 42 };
    try std.testing.expectEqual(@as(u32, 42), v_u32.asU32().?);
    try std.testing.expect(v_u32.asString() == null);
}
