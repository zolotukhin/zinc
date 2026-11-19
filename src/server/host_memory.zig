//! Return unused host allocator pages after an explicit model unload.
//! @section API Server
//! Reclaims freed glibc arena pages after model and request-cache retirement,
//! preserving live allocations and leaving other platforms' allocators alone.
const builtin = @import("builtin");
const std = @import("std");

const has_malloc_trim = builtin.link_libc and builtin.target.os.tag == .linux and builtin.target.abi.isGnu();
extern "c" fn malloc_trim(pad: usize) c_int;

/// Call only after model resources and reusable request context are retired.
/// ReleaseFast uses the C allocator: freeing vocabularies and graphs can leave
/// hundreds of MiB resident in glibc's thread arenas. Return those unused pages
/// at the idle unload boundary, rather than adding work to token generation.
/// glibc 2.8+ reclaims pages across all arenas; only the padding hint is
/// specific to the main heap. See https://man7.org/linux/man-pages/man3/malloc_trim.3.html.
/// Other platforms keep their allocator's normal behavior.
pub fn reclaimIdlePages() void {
    if (comptime has_malloc_trim) {
        _ = malloc_trim(0);
    }
}

test "idle heap reclamation preserves live HTTP allocations" {
    if (comptime !has_malloc_trim) return error.SkipZigTest;
    const allocator = std.heap.c_allocator;
    const live = try allocator.alloc(u8, 64 * 1024);
    defer allocator.free(live);
    @memset(live, 0x5a);
    const retired = try allocator.alloc(u8, 64 * 1024);
    @memset(retired, 0xa5);
    allocator.free(retired);
    reclaimIdlePages();
    for (live) |byte| try std.testing.expectEqual(@as(u8, 0x5a), byte);
}
