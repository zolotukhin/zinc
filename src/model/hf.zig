//! Resolve and download Hugging Face GGUF models for the `-hf` CLI flag.
//! @section Managed Models
//! Turns an `owner/repo[:quant]` spec into an installed GGUF in the managed
//! model cache, resolving the concrete file name via the Hugging Face
//! `/v2/<repo>/manifests/<tag>` endpoint and reusing the managed download
//! pipeline for transfer, staging, and manifest bookkeeping.
const std = @import("std");
const catalog = @import("catalog.zig");
const config = @import("config.zig");
const managed = @import("managed.zig");

// The ollama-style manifest is a small bounded document (config + layers +
// ggufFile), typically ~1 KiB; the cap only guards against a misbehaving
// server.
const manifest_body_limit = 4 * 1024 * 1024;
const redirect_buffer_len = 4096;

/// GGUF file metadata resolved from the Hugging Face manifest endpoint.
const ResolvedFile = struct {
    /// Repo-relative GGUF file name (`ggufFile.rfilename`).
    file_name: []u8,
    /// Lowercase sha256 hex of the file (`ggufFile.lfs.sha256`), or null
    /// when the manifest does not carry a usable digest.
    sha256: ?[]u8,
    /// File size in bytes, or 0 when unknown.
    size_bytes: u64,

    fn deinit(self: *ResolvedFile, allocator: std.mem.Allocator) void {
        allocator.free(self.file_name);
        if (self.sha256) |s| allocator.free(s);
        self.* = undefined;
    }
};

/// Parsed `-hf` argument: a Hugging Face repo plus a quantization tag.
pub const Spec = struct {
    /// Repo in `owner/name` form, e.g. `unsloth/Qwen3.5-9B-GGUF`.
    repo: []const u8,
    /// Quantization tag (e.g. `Q4_K_M`), or `latest` to let Hugging Face
    /// pick the repo's default quantization.
    tag: []const u8,
};

/// Parses an `-hf` argument of the form `owner/repo[:quant]`.
///
/// The quantization suffix is optional; when absent the tag defaults to
/// `latest`, which the Hugging Face manifest endpoint resolves to the repo's
/// recommended quantization. Each component (owner, repo, tag) must match
/// the Hugging Face naming grammar; anything else fails with
/// `error.InvalidHfSpec`.
/// @param text Raw CLI argument value.
/// @returns A `Spec` whose slices point into `text`.
pub fn parseSpec(text: []const u8) !Spec {
    var repo = text;
    var tag: []const u8 = "latest";
    if (std.mem.lastIndexOfScalar(u8, text, ':')) |pos| {
        repo = text[0..pos];
        tag = text[pos + 1 ..];
        if (tag.len == 0) return error.InvalidHfSpec;
    }
    const slash = std.mem.indexOfScalar(u8, repo, '/') orelse return error.InvalidHfSpec;
    if (slash == 0 or slash == repo.len - 1) return error.InvalidHfSpec;
    if (std.mem.indexOfScalarPos(u8, repo, slash + 1, '/') != null) return error.InvalidHfSpec;
    if (!isValidSpecComponent(repo[0..slash]) or
        !isValidSpecComponent(repo[slash + 1 ..]) or
        !isValidSpecComponent(tag))
    {
        return error.InvalidHfSpec;
    }
    return .{ .repo = repo, .tag = tag };
}

// Hugging Face owner/repo names and quantization tags are alphanumerics
// plus `.`, `_`, `-`, never starting with `.` or `-`. The components are
// embedded verbatim in request URLs and the spec becomes the /v1/models
// display name, so anything outside that grammar — path separators, query
// or fragment markers, quotes, whitespace — is rejected rather than escaped.
fn isValidSpecComponent(s: []const u8) bool {
    if (s.len == 0) return false;
    if (s[0] == '.' or s[0] == '-') return false;
    for (s) |c| {
        switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '.', '_', '-' => {},
            else => return false,
        }
    }
    return true;
}

/// Derives the managed-cache model id for a Hugging Face spec.
///
/// The id is a single filesystem-safe path component so the download can
/// live in the same `<cache_root>/models/<id>/model.gguf` layout as catalog
/// models. Distinct specs map to distinct ids; `:latest` is cached
/// separately from an explicit quantization even if both resolve to the
/// same upstream file.
/// @param allocator Owns the returned id slice.
/// @param spec Parsed Hugging Face spec.
/// @returns Heap-allocated id like `hf--unsloth--qwen3.5-9b-gguf--q4_k_m`.
pub fn cacheId(allocator: std.mem.Allocator, spec: Spec) ![]u8 {
    const raw = try std.fmt.allocPrint(allocator, "hf--{s}--{s}", .{ spec.repo, spec.tag });
    for (raw) |*c| {
        c.* = std.ascii.toLower(c.*);
        switch (c.*) {
            'a'...'z', '0'...'9', '.', '_', '-' => {},
            else => c.* = '-',
        }
    }
    return raw;
}

/// Ensures the model described by an `-hf` spec is installed and returns its path.
///
/// If the spec is already cached the installed path is returned without any
/// network access. Otherwise the GGUF file name and sha256 digest are
/// resolved via the Hugging Face manifest endpoint and the file is
/// downloaded through the managed pull pipeline (staged `.partial` file,
/// progress bar, manifest write), which verifies the download against the
/// pinned digest when the manifest provides one.
/// @param spec_text Raw `-hf` argument (`owner/repo[:quant]`).
/// @param allocator Used for HTTP, path construction, and the returned path.
/// @param writer Receives human-readable status lines and download progress.
/// @returns Heap-allocated absolute path to the installed GGUF; caller owns it.
/// @note Models zinc cannot run are rejected before the download starts:
/// quantizations with no kernels (`error.UnsupportedQuantization`, detected
/// from the tag and the resolved file name) and architectures the loader
/// does not recognise (`error.UnsupportedArchitecture`, detected from repo
/// metadata when available).
/// @note Gated or private repos are not supported: they fail with `error.HfAuthRequired`.
pub fn ensureModel(spec_text: []const u8, allocator: std.mem.Allocator, writer: anytype) ![]u8 {
    const spec = try parseSpec(spec_text);
    const id = try cacheId(allocator, spec);
    defer allocator.free(id);

    // Pre-download validation: reject unsupported quantizations (from the
    // tag, and later from the resolved file name) and unsupported model
    // architectures (from repo metadata) before any bytes are transferred.
    // The tag check runs before the cache lookup on purpose: it needs no
    // network, and a cached copy of a rejected quantization (downloaded by
    // an older zinc or placed manually) would only generate garbage.
    if (findRejectedQuantMarker(spec.tag)) |marker| {
        try printUnsupportedQuant(writer, spec.tag, marker);
        return error.UnsupportedQuantization;
    }

    if (managed.isInstalled(id, allocator)) {
        const path = try managed.resolveInstalledModelPath(id, allocator);
        try writer.print("Using cached Hugging Face model: {s}\n", .{path});
        try writer.flush();
        return path;
    }
    if (try fetchRepoArchitecture(allocator, spec)) |arch_str| {
        defer allocator.free(arch_str);
        if (config.parseArchitecture(arch_str) == .unknown) {
            try writer.print(
                "Model architecture '{s}' is not supported by zinc. Supported architectures: llama/mistral, qwen2/qwen3 (dense and MoE), qwen3.5/3.6/3.8 (dense and MoE), gemma, gpt-oss, mamba, jamba.\n",
                .{arch_str},
            );
            try writer.flush();
            return error.UnsupportedArchitecture;
        }
    }

    var resolved = try resolveGgufFile(allocator, spec, writer);
    defer resolved.deinit(allocator);

    if (findRejectedQuantMarker(resolved.file_name)) |marker| {
        try printUnsupportedQuant(writer, resolved.file_name, marker);
        return error.UnsupportedQuantization;
    }

    const download_url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/{s}/resolve/main/{s}?download=true",
        .{ spec.repo, resolved.file_name },
    );
    defer allocator.free(download_url);
    const homepage_url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}", .{spec.repo});
    defer allocator.free(homepage_url);

    const entry = catalog.CatalogEntry{
        .id = id,
        .display_name = spec_text,
        .release_date = "",
        .family = "huggingface",
        .format = "gguf",
        .quantization = spec.tag,
        .file_name = resolved.file_name,
        .homepage_url = homepage_url,
        .download_url = download_url,
        // Pinning the manifest digest makes the pull pipeline verify the
        // pre-download x-linked-etag and the post-download file hash.
        .sha256 = resolved.sha256 orelse "",
        .size_bytes = resolved.size_bytes,
        .required_vram_bytes = 0,
        .default_context_length = 4096,
        .recommended_for_chat = false,
        .thinking_stable = false,
        .status = .experimental,
        .tested_profiles = &.{},
    };

    try managed.pullModelWithObserver(entry, allocator, writer, null);
    try writer.print("Remove later with: zinc model rm {s}\n", .{id});
    try writer.flush();
    return managed.resolveInstalledModelPath(id, allocator);
}

const HttpBody = struct {
    status: std.http.Status,
    body: []u8,

    fn deinit(self: *HttpBody, allocator: std.mem.Allocator) void {
        allocator.free(self.body);
        self.* = undefined;
    }
};

fn httpGetJson(allocator: std.mem.Allocator, url: []const u8) !HttpBody {
    var client: std.http.Client = .{ .allocator = allocator };
    defer client.deinit();

    const uri = try std.Uri.parse(url);
    // Hugging Face only includes the resolved `ggufFile` object in the
    // manifest response when the User-Agent contains "llama-cpp". The typed
    // header slot must be overridden; an extra_headers entry would be sent
    // in addition to the client's default `zig/x.y.z (std.http)` agent, and
    // Hugging Face matches on the first user-agent header. accept-encoding
    // is forced to identity because `response.reader` does not decompress.
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .user_agent = .{ .override = "zinc (llama-cpp compatible)" },
            .accept_encoding = .{ .override = "identity" },
        },
        .extra_headers = &.{
            .{ .name = "accept", .value = "application/json" },
        },
    });
    defer req.deinit();
    try req.sendBodiless();

    var redirect_buffer: [redirect_buffer_len]u8 = undefined;
    var response = try req.receiveHead(&redirect_buffer);

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    const content_length = response.head.content_length;
    var transfer_buffer: [4096]u8 = undefined;
    var reader = response.reader(&transfer_buffer);
    var chunk: [4096]u8 = undefined;
    while (true) {
        // Guard: stop once all expected bytes are received.
        // Zig 0.15 std.http panics if we read past the content-length boundary.
        if (content_length) |total| {
            if (body.items.len >= total) break;
        }
        if (body.items.len >= manifest_body_limit) return error.HfManifestFailed;
        const n = reader.readSliceShort(&chunk) catch |err| switch (err) {
            error.ReadFailed => return response.bodyErr().?,
        };
        if (n == 0) break;
        try body.appendSlice(allocator, chunk[0..n]);
    }

    return .{
        .status = response.head.status,
        .body = try body.toOwnedSlice(allocator),
    };
}

fn resolveGgufFile(allocator: std.mem.Allocator, spec: Spec, writer: anytype) !ResolvedFile {
    const manifest_url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/v2/{s}/manifests/{s}",
        .{ spec.repo, spec.tag },
    );
    defer allocator.free(manifest_url);

    try writer.print("Resolving Hugging Face model: {s} (tag: {s})\n", .{ spec.repo, spec.tag });
    try writer.flush();

    var fetched = try httpGetJson(allocator, manifest_url);
    defer fetched.deinit(allocator);
    switch (fetched.status.class()) {
        .success => {},
        else => switch (fetched.status) {
            .unauthorized, .forbidden => return error.HfAuthRequired,
            .not_found => return error.HfModelNotFound,
            else => return error.HfManifestFailed,
        },
    }

    return parseManifestBody(allocator, fetched.body);
}

/// Fetches `gguf.architecture` from the Hugging Face model-info API, or
/// returns null when the metadata is unavailable (network error, gated
/// repo, non-GGUF repo). Best-effort: absence never blocks a download.
fn fetchRepoArchitecture(allocator: std.mem.Allocator, spec: Spec) !?[]u8 {
    const info_url = try std.fmt.allocPrint(
        allocator,
        "https://huggingface.co/api/models/{s}",
        .{spec.repo},
    );
    defer allocator.free(info_url);

    var fetched = httpGetJson(allocator, info_url) catch return null;
    defer fetched.deinit(allocator);
    if (fetched.status.class() != .success) return null;
    return extractArchitecture(allocator, fetched.body);
}

fn extractArchitecture(allocator: std.mem.Allocator, body: []const u8) !?[]u8 {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return null;
    defer parsed.deinit();
    const root = switch (parsed.value) {
        .object => |o| o,
        else => return null,
    };
    const gguf = switch (root.get("gguf") orelse return null) {
        .object => |o| o,
        else => return null,
    };
    const arch = switch (gguf.get("architecture") orelse return null) {
        .string => |s| s,
        else => return null,
    };
    if (arch.len == 0) return null;
    return try allocator.dupe(u8, arch);
}

// Quantization families no zinc backend has kernels for; tensors in these
// formats would be zero-filled at inference time (see dequantRow), so the
// download is rejected up front. Formats supported by at least one backend
// (Q4_K, Q5_0/Q5_1, Q5_K, Q6_K, Q8_0/Q8_1, MXFP4, F16, F32) pass through.
const rejected_quant_markers = [_][]const u8{
    "q2_k", "q3_k", "q4_0", "q4_1", "iq1", "iq2", "iq3", "iq4", "bf16", "tq1", "tq2",
};

fn printUnsupportedQuant(writer: anytype, name: []const u8, marker: []const u8) !void {
    try writer.print(
        "Quantization '{s}' is not supported by zinc: no backend has {s} kernels, so its tensors would be zero-filled at inference. Supported quantizations: Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, MXFP4, F16, F32 (e.g. retry with :Q4_K_M or :Q8_0).\n",
        .{ name, marker },
    );
    try writer.flush();
}

// A marker only counts when it is not embedded in a longer alphanumeric
// run: `uniq2` must not match `iq2`, and `Q4_01B` must not match `q4_0`,
// while delimited occurrences (`model-IQ2_XS.gguf`, `Q4_0_4_4`) still do.
// Case-insensitive and unbounded in length, so an unusually long file name
// cannot slip past the gate.
fn findRejectedQuantMarker(text: []const u8) ?[]const u8 {
    for (rejected_quant_markers) |marker| {
        var start: usize = 0;
        while (std.ascii.indexOfIgnoreCasePos(text, start, marker)) |idx| : (start = idx + 1) {
            const end = idx + marker.len;
            const bounded_left = idx == 0 or !std.ascii.isAlphanumeric(text[idx - 1]);
            const bounded_right = end == text.len or !std.ascii.isAlphanumeric(text[end]);
            if (bounded_left and bounded_right) return marker;
        }
    }
    return null;
}

fn parseManifestBody(allocator: std.mem.Allocator, body: []const u8) !ResolvedFile {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch
        return error.HfManifestFailed;
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.HfManifestFailed,
    };
    const gguf = switch (root.get("ggufFile") orelse return error.HfManifestMissingGguf) {
        .object => |o| o,
        else => return error.HfManifestMissingGguf,
    };
    const file_name = switch (gguf.get("rfilename") orelse return error.HfManifestMissingGguf) {
        .string => |s| s,
        else => return error.HfManifestMissingGguf,
    };
    if (file_name.len == 0) return error.HfManifestMissingGguf;

    var size_bytes: u64 = 0;
    if (gguf.get("size")) |v| switch (v) {
        .integer => |n| {
            if (n > 0) size_bytes = @intCast(n);
        },
        else => {},
    };

    var sha256: ?[]u8 = null;
    errdefer if (sha256) |s| allocator.free(s);
    if (gguf.get("lfs")) |lfs_value| switch (lfs_value) {
        .object => |lfs| {
            if (lfs.get("sha256")) |v| switch (v) {
                .string => |s| {
                    if (isSha256Hex(s)) {
                        const owned = try allocator.dupe(u8, s);
                        for (owned) |*c| c.* = std.ascii.toLower(c.*);
                        sha256 = owned;
                    }
                },
                else => {},
            };
        },
        else => {},
    };

    return .{
        .file_name = try allocator.dupe(u8, file_name),
        .sha256 = sha256,
        .size_bytes = size_bytes,
    };
}

fn isSha256Hex(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| {
        if (!std.ascii.isHex(c)) return false;
    }
    return true;
}

test "parseSpec accepts repo without tag" {
    const spec = try parseSpec("unsloth/Qwen3.5-9B-GGUF");
    try std.testing.expectEqualStrings("unsloth/Qwen3.5-9B-GGUF", spec.repo);
    try std.testing.expectEqualStrings("latest", spec.tag);
}

test "parseSpec splits quantization tag" {
    const spec = try parseSpec("unsloth/Qwen3.5-9B-GGUF:Q4_K_M");
    try std.testing.expectEqualStrings("unsloth/Qwen3.5-9B-GGUF", spec.repo);
    try std.testing.expectEqualStrings("Q4_K_M", spec.tag);
}

test "parseSpec rejects malformed specs" {
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("no-slash"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("/leading"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("trailing/"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("a/b/c"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("a/b:"));
}

test "parseSpec rejects components outside the Hugging Face grammar" {
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("owner/repo name"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("owner/repo?download=1"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("owner/repo#frag:Q8_0"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("owner/..:Q8_0"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("owner/repo:Q8 0"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("owner/repo:-Q8_0"));
    try std.testing.expectError(error.InvalidHfSpec, parseSpec("owner/repo\":Q8_0"));
}

test "cacheId is a lowercase single path component" {
    const spec = try parseSpec("unsloth/Qwen3.5-9B-GGUF:Q4_K_M");
    const id = try cacheId(std.testing.allocator, spec);
    defer std.testing.allocator.free(id);
    try std.testing.expectEqualStrings("hf--unsloth-qwen3.5-9b-gguf--q4_k_m", id);
    try std.testing.expect(std.mem.indexOfScalar(u8, id, '/') == null);
}

test "parseManifestBody reads file name, digest, and size from ggufFile" {
    const manifest_body =
        \\{"siblings":[{"rfilename":"README.md"}],"ggufFile":{"rfilename":"Qwen3.5-9B-Q4_K_M.gguf","size":5650000000,"lfs":{"sha256":"9465E63A22ADD5354D9BB4B99E90117043C7124007664907259BD16D043BB031","size":5650000000}},"other":{"rfilename":"mmproj.gguf"}}
    ;
    var resolved = try parseManifestBody(std.testing.allocator, manifest_body);
    defer resolved.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("Qwen3.5-9B-Q4_K_M.gguf", resolved.file_name);
    try std.testing.expectEqualStrings(
        "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
        resolved.sha256.?,
    );
    try std.testing.expectEqual(@as(u64, 5_650_000_000), resolved.size_bytes);
}

test "parseManifestBody tolerates a missing or malformed digest" {
    const manifest_body =
        \\{"ggufFile":{"rfilename":"model.gguf","size":-5,"lfs":{"sha256":"not-a-digest"}}}
    ;
    var resolved = try parseManifestBody(std.testing.allocator, manifest_body);
    defer resolved.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("model.gguf", resolved.file_name);
    try std.testing.expectEqual(@as(?[]u8, null), resolved.sha256);
    try std.testing.expectEqual(@as(u64, 0), resolved.size_bytes);
}

test "parseManifestBody rejects manifests without a ggufFile" {
    try std.testing.expectError(
        error.HfManifestMissingGguf,
        parseManifestBody(std.testing.allocator, "{\"siblings\":[{\"rfilename\":\"README.md\"}]}"),
    );
}

test "findRejectedQuantMarker flags unsupported quantizations" {
    try std.testing.expectEqualStrings("q3_k", findRejectedQuantMarker("Q3_K_M").?);
    try std.testing.expectEqualStrings("q3_k", findRejectedQuantMarker("mistral-7b-instruct-v0.2.Q3_K_L.gguf").?);
    try std.testing.expectEqualStrings("q2_k", findRejectedQuantMarker("Q2_K").?);
    try std.testing.expectEqualStrings("iq4", findRejectedQuantMarker("model-IQ4_XS.gguf").?);
    try std.testing.expectEqualStrings("q4_0", findRejectedQuantMarker("phi-3-mini-Q4_0.gguf").?);
    try std.testing.expectEqualStrings("bf16", findRejectedQuantMarker("model-BF16.gguf").?);
}

test "findRejectedQuantMarker passes supported quantizations" {
    try std.testing.expect(findRejectedQuantMarker("latest") == null);
    try std.testing.expect(findRejectedQuantMarker("Q4_K_M") == null);
    try std.testing.expect(findRejectedQuantMarker("Qwen3-0.6B-Q8_0.gguf") == null);
    try std.testing.expect(findRejectedQuantMarker("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf") == null);
    try std.testing.expect(findRejectedQuantMarker("model-F16.gguf") == null);
    try std.testing.expect(findRejectedQuantMarker("gpt-oss-20b-MXFP4.gguf") == null);
}

test "findRejectedQuantMarker requires token boundaries" {
    // Markers embedded in longer alphanumeric runs are not quantization tags.
    try std.testing.expect(findRejectedQuantMarker("uniq2-model-Q8_0.gguf") == null);
    try std.testing.expect(findRejectedQuantMarker("model-Q4_01B.gguf") == null);
    // Delimited occurrences still reject, including marker-prefixed variants.
    try std.testing.expectEqualStrings("q4_0", findRejectedQuantMarker("model-Q4_0_4_4.gguf").?);
    try std.testing.expectEqualStrings("iq2", findRejectedQuantMarker("model-IQ2_XXS.gguf").?);
}

test "findRejectedQuantMarker scans names longer than 256 bytes" {
    const long_name = "m" ** 300 ++ "-Q2_K.gguf";
    try std.testing.expectEqualStrings("q2_k", findRejectedQuantMarker(long_name).?);
}

test "extractArchitecture reads gguf.architecture from model info" {
    const body =
        \\{"id":"microsoft/Phi-3-mini-4k-instruct-gguf","gguf":{"architecture":"phi3","context_length":4096}}
    ;
    const arch = (try extractArchitecture(std.testing.allocator, body)).?;
    defer std.testing.allocator.free(arch);
    try std.testing.expectEqualStrings("phi3", arch);
}

test "extractArchitecture returns null when metadata is absent or malformed" {
    try std.testing.expectEqual(@as(?[]u8, null), try extractArchitecture(std.testing.allocator, "{\"id\":\"x/y\"}"));
    try std.testing.expectEqual(@as(?[]u8, null), try extractArchitecture(std.testing.allocator, "{\"gguf\":{}}"));
    try std.testing.expectEqual(@as(?[]u8, null), try extractArchitecture(std.testing.allocator, "not json"));
}

test "parseManifestBody rejects non-JSON bodies" {
    try std.testing.expectError(
        error.HfManifestFailed,
        parseManifestBody(std.testing.allocator, "<html>rate limited</html>"),
    );
}
