//! Paged KV cache manager for concurrent request serving.
//! @section Scheduler
//! Manages a pool of fixed-size pages that are allocated per-request
//! and freed on completion or cancellation. Each page maps to a contiguous
//! region of the GPU KV cache buffer, giving each request non-overlapping
//! token storage.
const std = @import("std");

const log = std.log.scoped(.kv_cache);

/// A single page in the KV cache pool.
/// Each page maps to a contiguous region of the GPU KV buffer.
pub const KvPage = struct {
    /// Index of this page in the pool.
    page_id: u32,
    /// Request ID that owns this page, or null if the page is free.
    owner: ?u64,
    /// First token position stored in this page.
    token_start: u32,
    /// Number of tokens currently stored (at most page_size).
    token_count: u32,
};

/// Pool-based allocator for KV cache pages.
/// Tracks which pages are free and which are owned by active requests.
pub const KvPagePool = struct {
    /// All pages in the pool, indexed by page_id.
    pages: []KvPage,
    /// Stack of free page IDs available for allocation.
    free_list: std.ArrayList(u32),
    /// Number of tokens each page can hold.
    page_size: u32,
    /// Total number of pages in the pool.
    total_pages: u32,
    /// Allocator for the page array and free list.
    allocator: std.mem.Allocator,

    /// Initialize a page pool with the given number of pages and tokens per page.
    /// @param allocator Allocator for the page array and free list.
    /// @param total_pages Number of pages to create.
    /// @param page_size Number of tokens each page can hold.
    /// @returns A KvPagePool with all pages initially free.
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

    /// Allocate N pages for a request.
    /// @param self Pool to allocate from.
    /// @param request_id Owner request ID to stamp on allocated pages.
    /// @param count Number of pages to allocate.
    /// @returns Slice of allocated page IDs (caller-owned).
    /// @note Returns error.KvCacheExhausted if not enough free pages remain.
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

    /// Free all pages owned by a request, returning them to the free list.
    /// @param self Pool to return pages to.
    /// @param request_id Request whose pages should be freed.
    pub fn freePages(self: *KvPagePool, request_id: u64) void {
        for (self.pages) |*page| {
            if (page.owner == request_id) {
                page.owner = null;
                page.token_count = 0;
                self.free_list.append(self.allocator, page.page_id) catch {};
            }
        }
    }

    /// Get the KV cache token position base for a request's allocated pages.
    /// Each request's tokens start at page_id * page_size, giving non-overlapping regions.
    /// @param self Pool to query.
    /// @param page_ids Allocated page IDs for the request.
    /// @returns Token position offset for the first page.
    pub fn positionBase(self: *const KvPagePool, page_ids: []const u32) u32 {
        if (page_ids.len == 0) return 0;
        return page_ids[0] * self.page_size;
    }

    /// Maximum context length a request can use based on the number of allocated pages.
    /// @param self Pool to query.
    /// @param page_count Number of pages allocated to the request.
    /// @returns Maximum number of tokens that fit in the allocated pages.
    pub fn maxContext(self: *const KvPagePool, page_count: u32) u32 {
        return page_count * self.page_size;
    }

    /// Number of free pages available for allocation.
    /// @param self Pool to query.
    /// @returns Count of unallocated pages.
    pub fn freeCount(self: *const KvPagePool) u32 {
        return @intCast(self.free_list.items.len);
    }

    /// Release the page array and free list.
    /// @param self Pool to tear down.
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

test "KvPagePool positionBase gives non-overlapping regions" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 8, 256);
    defer pool.deinit();

    const p1 = try pool.allocPages(1, 2);
    defer allocator.free(p1);
    const p2 = try pool.allocPages(2, 2);
    defer allocator.free(p2);

    const base1 = pool.positionBase(p1);
    const base2 = pool.positionBase(p2);
    // Regions should not overlap: base1 + 2*256 <= base2 or vice versa
    const end1 = base1 + pool.maxContext(2);
    const end2 = base2 + pool.maxContext(2);
    try std.testing.expect(end1 <= base2 or end2 <= base1);
}

test "KvPagePool free nonexistent request is no-op" {
    const allocator = std.testing.allocator;
    var pool = try KvPagePool.init(allocator, 4, 256);
    defer pool.deinit();

    // Freeing a request that never allocated should not change free count
    pool.freePages(999);
    try std.testing.expectEqual(@as(u32, 4), pool.freeCount());
}
