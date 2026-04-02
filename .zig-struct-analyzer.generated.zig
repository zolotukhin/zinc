const std = @import("std");

fn dumpStruct(comptime T: type, name: []const u8, first: *bool, w: anytype) !void {
    const type_info = @typeInfo(T);
    if (type_info != .@"struct") return;
    if (!first.*) try w.print(",", .{});
    first.* = false;
    
    try w.print("\"{s}\":{{", .{name});
    try w.print("\"size\":{d},\"alignment\":{d},\"fields\":[", .{ @sizeOf(T), @alignOf(T) });
    
    var firstField = true;
    inline for (type_info.@"struct".fields) |f| {
        if (@sizeOf(f.type) > 0 and !f.is_comptime) {
            if (!firstField) try w.print(",", .{});
            try w.print("{{\"name\":\"{s}\",\"type\":\"{s}\",\"size\":{d},\"alignment\":{d},\"offset\":{d}}}", .{
                f.name, @typeName(f.type), @sizeOf(f.type), f.alignment, @offsetOf(T, f.name)
            });
            firstField = false;
        }
    }
    try w.print("]}}", .{});
}

pub fn main() !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    const out = &stdout.interface;
    try out.print("{{", .{});
    var first = true;
    if (@TypeOf(mod0.ArgmaxDispatch) == type) {
        dumpStruct(mod0.ArgmaxDispatch, "ArgmaxDispatch", &first, out) catch {};
    }
    if (@TypeOf(mod1.AttentionDispatch) == type) {
        dumpStruct(mod1.AttentionDispatch, "AttentionDispatch", &first, out) catch {};
    }
    if (@TypeOf(mod2.DmmvDispatch) == type) {
        dumpStruct(mod2.DmmvDispatch, "DmmvDispatch", &first, out) catch {};
    }
    if (@TypeOf(mod3.SsmConv1dPush) == type) {
        dumpStruct(mod3.SsmConv1dPush, "SsmConv1dPush", &first, out) catch {};
    }
    if (@TypeOf(mod3.SsmDeltaNetPush) == type) {
        dumpStruct(mod3.SsmDeltaNetPush, "SsmDeltaNetPush", &first, out) catch {};
    }
    if (@TypeOf(mod3.SsmGatedNormPush) == type) {
        dumpStruct(mod3.SsmGatedNormPush, "SsmGatedNormPush", &first, out) catch {};
    }
    if (@TypeOf(mod3.SoftmaxTopkPush) == type) {
        dumpStruct(mod3.SoftmaxTopkPush, "SoftmaxTopkPush", &first, out) catch {};
    }
    if (@TypeOf(mod3.MoeWeightedAccPush) == type) {
        dumpStruct(mod3.MoeWeightedAccPush, "MoeWeightedAccPush", &first, out) catch {};
    }
    if (@TypeOf(mod3.ElementwiseDispatch) == type) {
        dumpStruct(mod3.ElementwiseDispatch, "ElementwiseDispatch", &first, out) catch {};
    }
    if (@TypeOf(mod4.DecodeState) == type) {
        dumpStruct(mod4.DecodeState, "DecodeState", &first, out) catch {};
    }
    if (@TypeOf(mod4.SamplingParams) == type) {
        dumpStruct(mod4.SamplingParams, "SamplingParams", &first, out) catch {};
    }
    if (@TypeOf(mod4.InferenceEngine) == type) {
        dumpStruct(mod4.InferenceEngine, "InferenceEngine", &first, out) catch {};
    }
    if (@TypeOf(mod5.HardwareInfo) == type) {
        dumpStruct(mod5.HardwareInfo, "HardwareInfo", &first, out) catch {};
    }
    if (@TypeOf(mod5.Node) == type) {
        dumpStruct(mod5.Node, "Node", &first, out) catch {};
    }
    if (@TypeOf(mod5.Edge) == type) {
        dumpStruct(mod5.Edge, "Edge", &first, out) catch {};
    }
    if (@TypeOf(mod5.OpCount) == type) {
        dumpStruct(mod5.OpCount, "OpCount", &first, out) catch {};
    }
    if (@TypeOf(mod5.CriticalPathNode) == type) {
        dumpStruct(mod5.CriticalPathNode, "CriticalPathNode", &first, out) catch {};
    }
    if (@TypeOf(mod5.NodeAnalysis) == type) {
        dumpStruct(mod5.NodeAnalysis, "NodeAnalysis", &first, out) catch {};
    }
    if (@TypeOf(mod5.Hotspot) == type) {
        dumpStruct(mod5.Hotspot, "Hotspot", &first, out) catch {};
    }
    if (@TypeOf(mod5.GraphAnalysis) == type) {
        dumpStruct(mod5.GraphAnalysis, "GraphAnalysis", &first, out) catch {};
    }
    if (@TypeOf(mod5.Graph) == type) {
        dumpStruct(mod5.Graph, "Graph", &first, out) catch {};
    }
    if (@TypeOf(mod6.Options) == type) {
        dumpStruct(mod6.Options, "Options", &first, out) catch {};
    }
    if (@TypeOf(mod6.ManagedModelInfo) == type) {
        dumpStruct(mod6.ManagedModelInfo, "ManagedModelInfo", &first, out) catch {};
    }
    if (@TypeOf(mod6.FitEstimate) == type) {
        dumpStruct(mod6.FitEstimate, "FitEstimate", &first, out) catch {};
    }
    if (@TypeOf(mod7.Config) == type) {
        dumpStruct(mod7.Config, "Config", &first, out) catch {};
    }
    if (@TypeOf(mod8.CatalogEntry) == type) {
        dumpStruct(mod8.CatalogEntry, "CatalogEntry", &first, out) catch {};
    }
    if (@TypeOf(mod9.TensorInfo) == type) {
        dumpStruct(mod9.TensorInfo, "TensorInfo", &first, out) catch {};
    }
    if (@TypeOf(mod9.GGUFFile) == type) {
        dumpStruct(mod9.GGUFFile, "GGUFFile", &first, out) catch {};
    }
    if (@TypeOf(mod9.ParseOptions) == type) {
        dumpStruct(mod9.ParseOptions, "ParseOptions", &first, out) catch {};
    }
    if (@TypeOf(mod10.ModelConfig) == type) {
        dumpStruct(mod10.ModelConfig, "ModelConfig", &first, out) catch {};
    }
    if (@TypeOf(mod10.ModelInspection) == type) {
        dumpStruct(mod10.ModelInspection, "ModelInspection", &first, out) catch {};
    }
    if (@TypeOf(mod10.LoadedTensor) == type) {
        dumpStruct(mod10.LoadedTensor, "LoadedTensor", &first, out) catch {};
    }
    if (@TypeOf(mod10.Model) == type) {
        dumpStruct(mod10.Model, "Model", &first, out) catch {};
    }
    if (@TypeOf(mod11.RuntimePaths) == type) {
        dumpStruct(mod11.RuntimePaths, "RuntimePaths", &first, out) catch {};
    }
    if (@TypeOf(mod11.ModelFit) == type) {
        dumpStruct(mod11.ModelFit, "ModelFit", &first, out) catch {};
    }
    if (@TypeOf(mod11.ActiveSelection) == type) {
        dumpStruct(mod11.ActiveSelection, "ActiveSelection", &first, out) catch {};
    }
    if (@TypeOf(mod11.CachedGpuProfile) == type) {
        dumpStruct(mod11.CachedGpuProfile, "CachedGpuProfile", &first, out) catch {};
    }
    if (@TypeOf(mod11.InstalledManifest) == type) {
        dumpStruct(mod11.InstalledManifest, "InstalledManifest", &first, out) catch {};
    }
    if (@TypeOf(mod11.RemoveInstalledModelResult) == type) {
        dumpStruct(mod11.RemoveInstalledModelResult, "RemoveInstalledModelResult", &first, out) catch {};
    }
    if (@TypeOf(mod11.DownloadObserver) == type) {
        dumpStruct(mod11.DownloadObserver, "DownloadObserver", &first, out) catch {};
    }
    if (@TypeOf(mod12.Tokenizer) == type) {
        dumpStruct(mod12.Tokenizer, "Tokenizer", &first, out) catch {};
    }
    if (@TypeOf(mod13.KvPage) == type) {
        dumpStruct(mod13.KvPage, "KvPage", &first, out) catch {};
    }
    if (@TypeOf(mod13.KvPagePool) == type) {
        dumpStruct(mod13.KvPagePool, "KvPagePool", &first, out) catch {};
    }
    if (@TypeOf(mod14.GenerationParams) == type) {
        dumpStruct(mod14.GenerationParams, "GenerationParams", &first, out) catch {};
    }
    if (@TypeOf(mod14.Request) == type) {
        dumpStruct(mod14.Request, "Request", &first, out) catch {};
    }
    if (@TypeOf(mod15.Scheduler) == type) {
        dumpStruct(mod15.Scheduler, "Scheduler", &first, out) catch {};
    }
    if (@TypeOf(mod16.Connection) == type) {
        dumpStruct(mod16.Connection, "Connection", &first, out) catch {};
    }
    if (@TypeOf(mod16.Server) == type) {
        dumpStruct(mod16.Server, "Server", &first, out) catch {};
    }
    if (@TypeOf(mod17.LoadSpec) == type) {
        dumpStruct(mod17.LoadSpec, "LoadSpec", &first, out) catch {};
    }
    if (@TypeOf(mod17.ModelSummary) == type) {
        dumpStruct(mod17.ModelSummary, "ModelSummary", &first, out) catch {};
    }
    if (@TypeOf(mod17.ModelCatalogView) == type) {
        dumpStruct(mod17.ModelCatalogView, "ModelCatalogView", &first, out) catch {};
    }
    if (@TypeOf(mod17.LoadedResources) == type) {
        dumpStruct(mod17.LoadedResources, "LoadedResources", &first, out) catch {};
    }
    if (@TypeOf(mod17.ModelManager) == type) {
        dumpStruct(mod17.ModelManager, "ModelManager", &first, out) catch {};
    }
    if (@TypeOf(mod18.ServerState) == type) {
        dumpStruct(mod18.ServerState, "ServerState", &first, out) catch {};
    }
    if (@TypeOf(mod19.Session) == type) {
        dumpStruct(mod19.Session, "Session", &first, out) catch {};
    }
    if (@TypeOf(mod20.Buffer) == type) {
        dumpStruct(mod20.Buffer, "Buffer", &first, out) catch {};
    }
    if (@TypeOf(mod21.CommandPool) == type) {
        dumpStruct(mod21.CommandPool, "CommandPool", &first, out) catch {};
    }
    if (@TypeOf(mod21.CommandBuffer) == type) {
        dumpStruct(mod21.CommandBuffer, "CommandBuffer", &first, out) catch {};
    }
    if (@TypeOf(mod22.GpuConfig) == type) {
        dumpStruct(mod22.GpuConfig, "GpuConfig", &first, out) catch {};
    }
    if (@TypeOf(mod23.Instance) == type) {
        dumpStruct(mod23.Instance, "Instance", &first, out) catch {};
    }
    if (@TypeOf(mod24.Pipeline) == type) {
        dumpStruct(mod24.Pipeline, "Pipeline", &first, out) catch {};
    }
    if (@TypeOf(mod24.SpecConst) == type) {
        dumpStruct(mod24.SpecConst, "SpecConst", &first, out) catch {};
    }
    try out.print("}}", .{});
    try out.flush();
}

const mod0 = @import("src/compute/argmax.zig");
const mod1 = @import("src/compute/attention.zig");
const mod2 = @import("src/compute/dmmv.zig");
const mod3 = @import("src/compute/elementwise.zig");
const mod4 = @import("src/compute/forward.zig");
const mod5 = @import("src/compute/graph.zig");
const mod6 = @import("src/diagnostics.zig");
const mod7 = @import("src/main.zig");
const mod8 = @import("src/model/catalog.zig");
const mod9 = @import("src/model/gguf.zig");
const mod10 = @import("src/model/loader.zig");
const mod11 = @import("src/model/managed.zig");
const mod12 = @import("src/model/tokenizer.zig");
const mod13 = @import("src/scheduler/kv_cache.zig");
const mod14 = @import("src/scheduler/request.zig");
const mod15 = @import("src/scheduler/scheduler.zig");
const mod16 = @import("src/server/http.zig");
const mod17 = @import("src/server/model_manager.zig");
const mod18 = @import("src/server/routes.zig");
const mod19 = @import("src/server/session.zig");
const mod20 = @import("src/vulkan/buffer.zig");
const mod21 = @import("src/vulkan/command.zig");
const mod22 = @import("src/vulkan/gpu_detect.zig");
const mod23 = @import("src/vulkan/instance.zig");
const mod24 = @import("src/vulkan/pipeline.zig");
