//! Minimal Thread.Pool + WaitGroup shim for Zig 0.16.
//! Zig 0.16 removed std.Thread.Pool, std.Thread.WaitGroup, and
//! std.Thread.Mutex/Condition/Futex. This provides compatible replacements
//! using raw Thread.spawn and linux futex syscalls.
const std = @import("std");
const linux = std.os.linux;

fn futexWait(ptr: *const i32, expected: i32) void {
    _ = linux.futex_4arg(
        @ptrCast(ptr),
        .{ .cmd = .WAIT, .private = true },
        @bitCast(expected),
        null,
    );
}

fn futexWakeOne(ptr: *const i32) void {
    _ = linux.futex_3arg(
        @ptrCast(ptr),
        .{ .cmd = .WAKE, .private = true },
        1,
    );
}

fn futexWakeAll(ptr: *const i32) void {
    _ = linux.futex_3arg(
        @ptrCast(ptr),
        .{ .cmd = .WAKE, .private = true },
        std.math.maxInt(u32),
    );
}

/// A simple wait group based on atomic counter + futex.
pub const WaitGroup = struct {
    counter: std.atomic.Value(i32) = std.atomic.Value(i32).init(0),

    pub fn start(self: *WaitGroup) void {
        _ = self.counter.fetchAdd(1, .release);
    }

    pub fn finish(self: *WaitGroup) void {
        const prev = self.counter.fetchSub(1, .release);
        if (prev == 1) {
            futexWakeOne(&self.counter.raw);
        }
    }

    pub fn wait(self: *WaitGroup) void {
        while (true) {
            const val = self.counter.load(.acquire);
            if (val <= 0) return;
            futexWait(&self.counter.raw, val);
        }
    }
};

/// Simple futex-based mutex.
const FutexMutex = struct {
    state: std.atomic.Value(i32) = std.atomic.Value(i32).init(0),

    fn lock(self: *FutexMutex) void {
        while (@cmpxchgWeak(i32, &self.state.raw, 0, 1, .acquire, .monotonic) != null) {
            futexWait(&self.state.raw, 1);
        }
    }

    fn unlock(self: *FutexMutex) void {
        @atomicStore(i32, &self.state.raw, 0, .release);
        futexWakeOne(&self.state.raw);
    }
};

/// Minimal thread pool that pre-spawns worker threads.
/// Compatible with the old std.Thread.Pool interface used by ZINC.
pub const Pool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    mutex: FutexMutex = .{},
    pending: std.atomic.Value(i32) = std.atomic.Value(i32).init(0),
    head: ?*Job = null,
    tail: ?*Job = null,
    shutdown: std.atomic.Value(i32) = std.atomic.Value(i32).init(0),

    const Job = struct {
        func: *const fn (*Job) void,
        next: ?*Job = null,
    };

    fn ClosureJob(comptime func: anytype, comptime Args: type) type {
        return struct {
            job: Job,
            args: Args,
            wg: *WaitGroup,
            alloc: std.mem.Allocator,

            fn execute(job: *Job) void {
                const self: *@This() = @fieldParentPtr("job", job);
                const args = self.args;
                const wg = self.wg;
                const a = self.alloc;
                a.destroy(self);
                @call(.auto, func, args);
                wg.finish();
            }
        };
    }

    pub const InitConfig = struct {
        allocator: std.mem.Allocator = std.heap.page_allocator,
        n_jobs: usize = 0,
    };

    pub fn init(self: *Pool, config: InitConfig) !void {
        const n = if (config.n_jobs > 0) config.n_jobs else (std.Thread.getCpuCount() catch 1);
        self.allocator = config.allocator;
        self.mutex = .{};
        self.pending = std.atomic.Value(i32).init(0);
        self.head = null;
        self.tail = null;
        self.shutdown = std.atomic.Value(i32).init(0);
        self.threads = try config.allocator.alloc(std.Thread, n);
        var spawned: usize = 0;
        errdefer {
            self.shutdown.store(1, .release);
            for (0..spawned) |_| futexWakeAll(&self.pending.raw);
            for (self.threads[0..spawned]) |t| t.join();
            config.allocator.free(self.threads);
        }
        for (self.threads) |*t| {
            t.* = try std.Thread.spawn(.{ .allocator = config.allocator }, workerMain, .{self});
            spawned += 1;
        }
    }

    pub fn deinit(self: *Pool) void {
        self.shutdown.store(1, .release);
        for (0..self.threads.len) |_| futexWakeAll(&self.pending.raw);
        for (self.threads) |t| t.join();
        self.allocator.free(self.threads);
    }

    pub fn spawnWg(self: *Pool, wg: *WaitGroup, comptime func: anytype, args: anytype) void {
        const Args = @TypeOf(args);
        const C = ClosureJob(func, Args);
        const closure = self.allocator.create(C) catch return;
        closure.* = .{
            .job = .{ .func = C.execute, .next = null },
            .args = args,
            .wg = wg,
            .alloc = self.allocator,
        };
        wg.start();
        self.enqueue(&closure.job);
    }

    /// Execute queued jobs on the calling thread while waiting for the WaitGroup.
    pub fn waitAndWork(self: *Pool, wg: *WaitGroup) void {
        while (wg.counter.load(.acquire) > 0) {
            if (self.tryDequeue()) |job| {
                job.func(job);
            } else {
                std.atomic.spinLoopHint();
            }
        }
    }

    fn enqueue(self: *Pool, job: *Job) void {
        self.mutex.lock();
        if (self.tail) |t| {
            t.next = job;
        } else {
            self.head = job;
        }
        self.tail = job;
        self.mutex.unlock();
        _ = self.pending.fetchAdd(1, .release);
        futexWakeOne(&self.pending.raw);
    }

    fn tryDequeue(self: *Pool) ?*Job {
        self.mutex.lock();
        defer self.mutex.unlock();
        const job = self.head orelse return null;
        self.head = job.next;
        if (self.head == null) self.tail = null;
        job.next = null;
        return job;
    }

    fn workerMain(self: *Pool) void {
        while (self.shutdown.load(.acquire) == 0) {
            if (self.tryDequeue()) |job| {
                job.func(job);
            } else {
                const p = self.pending.load(.acquire);
                if (p == 0 and self.shutdown.load(.acquire) == 0) {
                    futexWait(&self.pending.raw, p);
                }
            }
        }
        // Drain remaining jobs on shutdown
        while (self.tryDequeue()) |job| {
            job.func(job);
        }
    }
};
