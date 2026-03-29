//! Paged KV cache manager for concurrent request serving.
//! Manages a pool of fixed-size pages that are allocated per-request
//! and freed on completion or cancellation.
const std = @import("std");

const log = std.log.scoped(.kv_cache);

/// A single page in the KV cache pool.
pub const KvPage = struct {
    page_id: u32,
    owner: ?u64, // request ID that owns this page, null if free
    token_start: u32, // first token position in this page
    token_count: u32, // tokens currently stored (≤ page_size)
};

/// Manages allocation and deallocation of KV cache pages.
pub const KvPagePool = struct {
    pages: []KvPage,
    free_list: std.ArrayList(u32),
    page_size: u32, // tokens per page
    total_pages: u32,
    allocator: std.mem.Allocator,

    /// Initialize a page pool with the given capacity.
    pub fn init(allocator: std.mem.Allocator, total_pages: u32, page_size: u32) !KvPagePool {
        const pages = try allocator.alloc(KvPage, total_pages);
        var free_list: std.ArrayList(u32) = .{};
        for (0..total_pages) |i| {
            pages[i] = .{
                .page_id = @intCast(i),
                .owner = null,
                .token_start = 0,
                .token_count = 0,
            };
            try free_list.append(allocator, @intCast(i));
        }
        log.info("KV page pool: {d} pages × {d} tokens = {d} total capacity", .{
            total_pages, page_size, total_pages * page_size,
        });
        return KvPagePool{
            .pages = pages,
            .free_list = free_list,
            .page_size = page_size,
            .total_pages = total_pages,
            .allocator = allocator,
        };
    }

    /// Allocate N pages for a request. Returns page IDs or error if pool exhausted.
    pub fn allocPages(self: *KvPagePool, request_id: u64, count: u32) ![]u32 {
        if (self.free_list.items.len < count) return error.KvCacheExhausted;
        const result = try self.allocator.alloc(u32, count);
        for (0..count) |i| {
            const page_id = self.free_list.pop() orelse return error.KvCacheExhausted;
            self.pages[page_id].owner = request_id;
            self.pages[page_id].token_count = 0;
            result[i] = page_id;
        }
        return result;
    }

    /// Free all pages owned by a request.
    pub fn freePages(self: *KvPagePool, request_id: u64) void {
        for (self.pages) |*page| {
            if (page.owner == request_id) {
                page.owner = null;
                page.token_count = 0;
                self.free_list.append(self.allocator, page.page_id) catch {};
            }
        }
    }

    /// Number of free pages available.
    pub fn freeCount(self: *const KvPagePool) u32 {
        return @intCast(self.free_list.items.len);
    }

    pub fn deinit(self: *KvPagePool) void {
        self.free_list.deinit(self.allocator);
        self.allocator.free(self.pages);
    }
};

test "KvPagePool alloc and free" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 4, 256);
    defer pool.deinit();

    try std.testing.expectEqual(@as(u32, 4), pool.freeCount());

    const pages = try pool.allocPages(1, 2);
    defer allocator.free(pages);
    try std.testing.expectEqual(@as(u32, 2), pool.freeCount());
    try std.testing.expectEqual(@as(usize, 2), pages.len);

    pool.freePages(1);
    try std.testing.expectEqual(@as(u32, 4), pool.freeCount());
}

test "KvPagePool exhaustion" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 2, 256);
    defer pool.deinit();

    const p1 = try pool.allocPages(1, 2);
    defer allocator.free(p1);
    try std.testing.expectError(error.KvCacheExhausted, pool.allocPages(2, 1));

    pool.freePages(1);
    const p2 = try pool.allocPages(2, 1);
    defer allocator.free(p2);
    try std.testing.expectEqual(@as(u32, 1), pool.freeCount());
}

test "KvPagePool multiple requests isolated" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 8, 256);
    defer pool.deinit();

    const p1 = try pool.allocPages(100, 3);
    defer allocator.free(p1);
    const p2 = try pool.allocPages(200, 2);
    defer allocator.free(p2);
    try std.testing.expectEqual(@as(u32, 3), pool.freeCount());

    // Freeing request 100 only frees its 3 pages, not request 200's
    pool.freePages(100);
    try std.testing.expectEqual(@as(u32, 6), pool.freeCount());

    // Request 200's pages still allocated
    pool.freePages(200);
    try std.testing.expectEqual(@as(u32, 8), pool.freeCount());
}

test "KvPagePool pages have correct owner after alloc" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 4, 256);
    defer pool.deinit();

    const pages = try pool.allocPages(42, 2);
    defer allocator.free(pages);

    for (pages) |pid| {
        try std.testing.expectEqual(@as(?u64, 42), pool.pages[pid].owner);
    }

    pool.freePages(42);
    for (pages) |pid| {
        try std.testing.expectEqual(@as(?u64, null), pool.pages[pid].owner);
    }
}

test "KvPagePool free nonexistent request is no-op" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 4, 256);
    defer pool.deinit();

    // Freeing a request that never allocated should not change free count
    pool.freePages(999);
    try std.testing.expectEqual(@as(u32, 4), pool.freeCount());
}
