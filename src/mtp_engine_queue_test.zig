//! GPU-free checks of the actual serving queue's MTP request propagation.
const std = @import("std");
const host_io = @import("host_io");
const serving = @import("server/cuda_serve.zig");
const Scheduler = @import("scheduler/scheduler.zig").Scheduler;

fn fixture(capability: serving.MtpCapability) !serving.ServeEngine {
    return .{
        .allocator = std.testing.allocator,
        // Queue operations must not touch the GPU forward.
        .fwd = .{ .qwen = undefined },
        .sched = try Scheduler.init(std.testing.allocator, 1),
        .eos = 999,
        .eot = null,
        .slot_ctx = 16,
        .mtp_capability = capability,
    };
}

fn destroyFixture(engine: *serving.ServeEngine) void {
    engine.registry.deinit(std.testing.allocator);
    engine.sched.deinit();
}

test "MTP queue latches on off and default independently for concurrent pending requests" {
    var engine = try fixture(.{ .supports_mtp = true, .default_enabled = true, .unavailable_reason = "" });
    defer destroyFixture(&engine);
    var default = serving.ReqChannel{};
    var off = serving.ReqChannel{};
    var on = serving.ReqChannel{};
    const a = try engine.submit(&.{ 1, 2 }, 4, &default);
    const b = try engine.submitWithOptions(&.{ 3, 4 }, 4, &off, .{ .enable_mtp = false });
    const c = try engine.submitWithOptions(&.{ 5, 6 }, 4, &on, .{ .enable_mtp = true });
    try std.testing.expect(a != b and b != c);
    try std.testing.expect(default.mtp.enabled);
    try std.testing.expect(default.mtp.requested == null);
    try std.testing.expect(!off.mtp.enabled);
    try std.testing.expectEqual(@as(?bool, false), off.mtp.requested);
    try std.testing.expect(on.mtp.enabled);
    try std.testing.expectEqual(@as(?bool, true), on.mtp.requested);
    try std.testing.expect(!default.mtp.used and !off.mtp.used and !on.mtp.used);
    try std.testing.expectEqual(@as(usize, 3), engine.sched.pending.items.len);
    try std.testing.expect(engine.registry.get(a).? == &default);
    try std.testing.expect(engine.registry.get(b).? == &off);
    try std.testing.expect(engine.registry.get(c).? == &on);
    engine.cancel(&off);
    try std.testing.expect(off.cancelled);
    try std.testing.expect(!default.cancelled and !on.cancelled);
    try std.testing.expect(default.mtp.enabled and on.mtp.enabled);
}

test "MTP queue rejects unsupported opt-in before borrowing a prompt or channel" {
    var engine = try fixture(.{ .unavailable_reason = "No NextN layer" });
    defer destroyFixture(&engine);
    var chan = serving.ReqChannel{};
    try std.testing.expectError(error.MtpUnavailable, engine.submitWithOptions(&.{1}, 4, &chan, .{ .enable_mtp = true }));
    try std.testing.expectEqual(@as(usize, 0), engine.sched.pending.items.len);
    try std.testing.expectEqual(@as(u32, 0), engine.registry.count());
    _ = try engine.submitWithOptions(&.{1}, 4, &chan, .{ .enable_mtp = false });
    try std.testing.expect(!chan.mtp.enabled);
    try std.testing.expect(!chan.mtp.used);
}

test "MTP queue supports explicit opt-in when the engine default is off" {
    var engine = try fixture(.{ .supports_mtp = true, .default_enabled = false, .unavailable_reason = "" });
    defer destroyFixture(&engine);
    var ordinary = serving.ReqChannel{};
    var enabled = serving.ReqChannel{};
    _ = try engine.submit(&.{1}, 4, &ordinary);
    _ = try engine.submitWithOptions(&.{2}, 4, &enabled, .{ .enable_mtp = true });
    try std.testing.expect(!ordinary.mtp.enabled);
    try std.testing.expect(enabled.mtp.enabled);
}

test "MTP queue validates context and clamps output before worker admission" {
    var engine = try fixture(.{ .supports_mtp = true, .default_enabled = true, .unavailable_reason = "" });
    defer destroyFixture(&engine);
    var chan = serving.ReqChannel{};
    try std.testing.expectError(error.EmptyPrompt, engine.submit(&.{}, 4, &chan));
    try std.testing.expectError(error.ContextTooSmall, engine.submit(&@as([16]u32, @splat(1)), 4, &chan));
    try std.testing.expectError(error.InvalidMaxTokens, engine.submit(&.{1}, 0, &chan));
    try std.testing.expectEqual(@as(usize, 0), engine.sched.pending.items.len);
    _ = try engine.submitWithOptions(&.{ 1, 2 }, 999, &chan, .{ .enable_mtp = true });
    try std.testing.expectEqual(@as(u32, 14), engine.sched.pending.items[0].params.max_tokens);
}

test "MTP model compatibility and execution preference do not conflate a CUDA default-off policy" {
    const ForwardCuda = @import("compute/forward_cuda.zig").ForwardCuda;
    const loader = @import("model/loader_cuda.zig");
    const build_options = @import("build_options");
    var model: loader.Model = undefined;
    model.config.n_nextn_layers = 0;
    model.tensor_map = std.StringHashMap(*loader.LoadedTensor).init(std.testing.allocator);
    defer model.tensor_map.deinit();
    var forward: ForwardCuda = undefined;
    forward.model = &model;
    forward.d.n_layers = 2;
    forward.d.n_experts = 0;
    forward.mtp_execution_override = null;
    try std.testing.expect(forward.mtpModelUnavailableReason() != null);
    model.config.n_nextn_layers = 1;
    forward.d.n_experts = 2;
    try std.testing.expect(std.mem.indexOf(u8, forward.mtpModelUnavailableReason().?, "dense") != null);
    forward.d.n_experts = 0;
    try std.testing.expect(std.mem.indexOf(u8, forward.mtpModelUnavailableReason().?, "missing weights") != null);
    var tensor: loader.LoadedTensor = undefined;
    for ([_][]const u8{
        "blk.2.nextn.eh_proj.weight",
        "blk.2.nextn.enorm.weight",
        "blk.2.nextn.hnorm.weight",
        "blk.2.attn_q.weight",
        "blk.2.ffn_gate.weight",
    }) |name| try model.tensor_map.put(name, &tensor);
    try std.testing.expect(forward.mtpModelUnavailableReason() == null);
    if (host_io.getenv("ZINC_MTP") == null) {
        try std.testing.expect(forward.mtpSupported());
        try std.testing.expectEqual(std.mem.eql(u8, build_options.backend, "rocm"), forward.mtpDefaultEnabled());
    }
    const allowed = forward.mtpSupported();
    const original = forward.setMtpExecutionOverride(false);
    try std.testing.expect(original == null);
    try std.testing.expect(!forward.mtpEnabled());
    try std.testing.expectEqual(@as(?bool, false), forward.setMtpExecutionOverride(true));
    try std.testing.expectEqual(allowed, forward.mtpEnabled());
    _ = forward.setMtpExecutionOverride(original);
    try std.testing.expectEqual(forward.mtpDefaultEnabled(), forward.mtpEnabled());
}

test "single-slot serving reuses target state only within capacity" {
    const ForwardCuda = @import("compute/forward_cuda.zig").ForwardCuda;
    // The live 27B/64k case must not allocate a second full KV/SSM cache.
    try std.testing.expect(ForwardCuda.canReuseSingleSlotState(1, 65536, 65536));
    try std.testing.expect(ForwardCuda.canReuseSingleSlotState(1, 32768, 65536));
    // More simultaneous sequences and larger caches need distinct allocations.
    try std.testing.expect(!ForwardCuda.canReuseSingleSlotState(2, 32768, 65536));
    try std.testing.expect(!ForwardCuda.canReuseSingleSlotState(1, 65537, 65536));
    try std.testing.expect(!ForwardCuda.canReuseSingleSlotState(0, 65536, 65536));
    try std.testing.expect(!ForwardCuda.canReuseSingleSlotState(1, 0, 65536));
}
