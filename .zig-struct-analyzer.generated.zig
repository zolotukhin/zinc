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
    if (@TypeOf(mod4.GenerateMetrics) == type) {
        dumpStruct(mod4.GenerateMetrics, "GenerateMetrics", &first, out) catch {};
    }
    if (@TypeOf(mod4.GenerateResult) == type) {
        dumpStruct(mod4.GenerateResult, "GenerateResult", &first, out) catch {};
    }
    if (@TypeOf(mod4.InitOptions) == type) {
        dumpStruct(mod4.InitOptions, "InitOptions", &first, out) catch {};
    }
    if (@TypeOf(mod4.RuntimeProfile) == type) {
        dumpStruct(mod4.RuntimeProfile, "RuntimeProfile", &first, out) catch {};
    }
    if (@TypeOf(mod4.InferenceEngine) == type) {
        dumpStruct(mod4.InferenceEngine, "InferenceEngine", &first, out) catch {};
    }
    if (@TypeOf(mod5.DecodeState) == type) {
        dumpStruct(mod5.DecodeState, "DecodeState", &first, out) catch {};
    }
    if (@TypeOf(mod5.SamplingParams) == type) {
        dumpStruct(mod5.SamplingParams, "SamplingParams", &first, out) catch {};
    }
    if (@TypeOf(mod5.InferenceEngine) == type) {
        dumpStruct(mod5.InferenceEngine, "InferenceEngine", &first, out) catch {};
    }
    if (@TypeOf(mod6.HardwareInfo) == type) {
        dumpStruct(mod6.HardwareInfo, "HardwareInfo", &first, out) catch {};
    }
    if (@TypeOf(mod6.Node) == type) {
        dumpStruct(mod6.Node, "Node", &first, out) catch {};
    }
    if (@TypeOf(mod6.Edge) == type) {
        dumpStruct(mod6.Edge, "Edge", &first, out) catch {};
    }
    if (@TypeOf(mod6.OpCount) == type) {
        dumpStruct(mod6.OpCount, "OpCount", &first, out) catch {};
    }
    if (@TypeOf(mod6.CriticalPathNode) == type) {
        dumpStruct(mod6.CriticalPathNode, "CriticalPathNode", &first, out) catch {};
    }
    if (@TypeOf(mod6.NodeAnalysis) == type) {
        dumpStruct(mod6.NodeAnalysis, "NodeAnalysis", &first, out) catch {};
    }
    if (@TypeOf(mod6.Hotspot) == type) {
        dumpStruct(mod6.Hotspot, "Hotspot", &first, out) catch {};
    }
    if (@TypeOf(mod6.GraphAnalysis) == type) {
        dumpStruct(mod6.GraphAnalysis, "GraphAnalysis", &first, out) catch {};
    }
    if (@TypeOf(mod6.Graph) == type) {
        dumpStruct(mod6.Graph, "Graph", &first, out) catch {};
    }
    if (@TypeOf(mod7.Options) == type) {
        dumpStruct(mod7.Options, "Options", &first, out) catch {};
    }
    if (@TypeOf(mod7.ManagedModelInfo) == type) {
        dumpStruct(mod7.ManagedModelInfo, "ManagedModelInfo", &first, out) catch {};
    }
    if (@TypeOf(mod7.UnifiedFitEstimate) == type) {
        dumpStruct(mod7.UnifiedFitEstimate, "UnifiedFitEstimate", &first, out) catch {};
    }
    if (@TypeOf(mod8.Options) == type) {
        dumpStruct(mod8.Options, "Options", &first, out) catch {};
    }
    if (@TypeOf(mod8.ManagedModelInfo) == type) {
        dumpStruct(mod8.ManagedModelInfo, "ManagedModelInfo", &first, out) catch {};
    }
    if (@TypeOf(mod8.FitEstimate) == type) {
        dumpStruct(mod8.FitEstimate, "FitEstimate", &first, out) catch {};
    }
    if (@TypeOf(mod9.Config) == type) {
        dumpStruct(mod9.Config, "Config", &first, out) catch {};
    }
    if (@TypeOf(mod10.MetalBuffer) == type) {
        dumpStruct(mod10.MetalBuffer, "MetalBuffer", &first, out) catch {};
    }
    if (@TypeOf(mod11.MetalCommand) == type) {
        dumpStruct(mod11.MetalCommand, "MetalCommand", &first, out) catch {};
    }
    if (@TypeOf(mod12.MetalCapabilities) == type) {
        dumpStruct(mod12.MetalCapabilities, "MetalCapabilities", &first, out) catch {};
    }
    if (@TypeOf(mod12.MetalDevice) == type) {
        dumpStruct(mod12.MetalDevice, "MetalDevice", &first, out) catch {};
    }
    if (@TypeOf(mod13.MetalPipeline) == type) {
        dumpStruct(mod13.MetalPipeline, "MetalPipeline", &first, out) catch {};
    }
    if (@TypeOf(mod14.CatalogEntry) == type) {
        dumpStruct(mod14.CatalogEntry, "CatalogEntry", &first, out) catch {};
    }
    if (@TypeOf(mod15.ModelConfig) == type) {
        dumpStruct(mod15.ModelConfig, "ModelConfig", &first, out) catch {};
    }
    if (@TypeOf(mod16.TensorInfo) == type) {
        dumpStruct(mod16.TensorInfo, "TensorInfo", &first, out) catch {};
    }
    if (@TypeOf(mod16.GGUFFile) == type) {
        dumpStruct(mod16.GGUFFile, "GGUFFile", &first, out) catch {};
    }
    if (@TypeOf(mod16.ParseOptions) == type) {
        dumpStruct(mod16.ParseOptions, "ParseOptions", &first, out) catch {};
    }
    if (@TypeOf(mod17.ModelInspection) == type) {
        dumpStruct(mod17.ModelInspection, "ModelInspection", &first, out) catch {};
    }
    if (@TypeOf(mod17.LoadedTensor) == type) {
        dumpStruct(mod17.LoadedTensor, "LoadedTensor", &first, out) catch {};
    }
    if (@TypeOf(mod17.Model) == type) {
        dumpStruct(mod17.Model, "Model", &first, out) catch {};
    }
    if (@TypeOf(mod18.ModelInspection) == type) {
        dumpStruct(mod18.ModelInspection, "ModelInspection", &first, out) catch {};
    }
    if (@TypeOf(mod18.LoadedTensor) == type) {
        dumpStruct(mod18.LoadedTensor, "LoadedTensor", &first, out) catch {};
    }
    if (@TypeOf(mod18.Model) == type) {
        dumpStruct(mod18.Model, "Model", &first, out) catch {};
    }
    if (@TypeOf(mod19.RuntimePaths) == type) {
        dumpStruct(mod19.RuntimePaths, "RuntimePaths", &first, out) catch {};
    }
    if (@TypeOf(mod19.ModelFit) == type) {
        dumpStruct(mod19.ModelFit, "ModelFit", &first, out) catch {};
    }
    if (@TypeOf(mod19.ActiveSelection) == type) {
        dumpStruct(mod19.ActiveSelection, "ActiveSelection", &first, out) catch {};
    }
    if (@TypeOf(mod19.CachedGpuProfile) == type) {
        dumpStruct(mod19.CachedGpuProfile, "CachedGpuProfile", &first, out) catch {};
    }
    if (@TypeOf(mod19.InstalledManifest) == type) {
        dumpStruct(mod19.InstalledManifest, "InstalledManifest", &first, out) catch {};
    }
    if (@TypeOf(mod19.RemoveInstalledModelResult) == type) {
        dumpStruct(mod19.RemoveInstalledModelResult, "RemoveInstalledModelResult", &first, out) catch {};
    }
    if (@TypeOf(mod19.DownloadObserver) == type) {
        dumpStruct(mod19.DownloadObserver, "DownloadObserver", &first, out) catch {};
    }
    if (@TypeOf(mod20.Tokenizer) == type) {
        dumpStruct(mod20.Tokenizer, "Tokenizer", &first, out) catch {};
    }
    if (@TypeOf(mod21.KvPage) == type) {
        dumpStruct(mod21.KvPage, "KvPage", &first, out) catch {};
    }
    if (@TypeOf(mod21.KvPagePool) == type) {
        dumpStruct(mod21.KvPagePool, "KvPagePool", &first, out) catch {};
    }
    if (@TypeOf(mod22.GenerationParams) == type) {
        dumpStruct(mod22.GenerationParams, "GenerationParams", &first, out) catch {};
    }
    if (@TypeOf(mod22.Request) == type) {
        dumpStruct(mod22.Request, "Request", &first, out) catch {};
    }
    if (@TypeOf(mod23.Scheduler) == type) {
        dumpStruct(mod23.Scheduler, "Scheduler", &first, out) catch {};
    }
    if (@TypeOf(mod24.Connection) == type) {
        dumpStruct(mod24.Connection, "Connection", &first, out) catch {};
    }
    if (@TypeOf(mod24.Server) == type) {
        dumpStruct(mod24.Server, "Server", &first, out) catch {};
    }
    if (@TypeOf(mod25.LoadSpec) == type) {
        dumpStruct(mod25.LoadSpec, "LoadSpec", &first, out) catch {};
    }
    if (@TypeOf(mod25.ModelSummary) == type) {
        dumpStruct(mod25.ModelSummary, "ModelSummary", &first, out) catch {};
    }
    if (@TypeOf(mod25.ModelCatalogView) == type) {
        dumpStruct(mod25.ModelCatalogView, "ModelCatalogView", &first, out) catch {};
    }
    if (@TypeOf(mod25.LoadedResources) == type) {
        dumpStruct(mod25.LoadedResources, "LoadedResources", &first, out) catch {};
    }
    if (@TypeOf(mod25.ModelManager) == type) {
        dumpStruct(mod25.ModelManager, "ModelManager", &first, out) catch {};
    }
    if (@TypeOf(mod26.LoadSpec) == type) {
        dumpStruct(mod26.LoadSpec, "LoadSpec", &first, out) catch {};
    }
    if (@TypeOf(mod26.ModelSummary) == type) {
        dumpStruct(mod26.ModelSummary, "ModelSummary", &first, out) catch {};
    }
    if (@TypeOf(mod26.ModelCatalogView) == type) {
        dumpStruct(mod26.ModelCatalogView, "ModelCatalogView", &first, out) catch {};
    }
    if (@TypeOf(mod26.LoadedResources) == type) {
        dumpStruct(mod26.LoadedResources, "LoadedResources", &first, out) catch {};
    }
    if (@TypeOf(mod26.ModelManager) == type) {
        dumpStruct(mod26.ModelManager, "ModelManager", &first, out) catch {};
    }
    if (@TypeOf(mod27.ServerState) == type) {
        dumpStruct(mod27.ServerState, "ServerState", &first, out) catch {};
    }
    if (@TypeOf(mod28.Session) == type) {
        dumpStruct(mod28.Session, "Session", &first, out) catch {};
    }
    if (@TypeOf(mod29.Buffer) == type) {
        dumpStruct(mod29.Buffer, "Buffer", &first, out) catch {};
    }
    if (@TypeOf(mod30.CommandPool) == type) {
        dumpStruct(mod30.CommandPool, "CommandPool", &first, out) catch {};
    }
    if (@TypeOf(mod30.CommandBuffer) == type) {
        dumpStruct(mod30.CommandBuffer, "CommandBuffer", &first, out) catch {};
    }
    if (@TypeOf(mod31.GpuConfig) == type) {
        dumpStruct(mod31.GpuConfig, "GpuConfig", &first, out) catch {};
    }
    if (@TypeOf(mod32.Instance) == type) {
        dumpStruct(mod32.Instance, "Instance", &first, out) catch {};
    }
    if (@TypeOf(mod33.Pipeline) == type) {
        dumpStruct(mod33.Pipeline, "Pipeline", &first, out) catch {};
    }
    if (@TypeOf(mod33.SpecConst) == type) {
        dumpStruct(mod33.SpecConst, "SpecConst", &first, out) catch {};
    }
    try out.print("}}", .{});
    try out.flush();
}

const mod0 = @import("src/compute/argmax.zig");
const mod1 = @import("src/compute/attention.zig");
const mod2 = @import("src/compute/dmmv.zig");
const mod3 = @import("src/compute/elementwise.zig");
const mod4 = @import("src/compute/forward_metal.zig");
const mod5 = @import("src/compute/forward.zig");
const mod6 = @import("src/compute/graph.zig");
const mod7 = @import("src/diagnostics_metal.zig");
const mod8 = @import("src/diagnostics.zig");
const mod9 = @import("src/main.zig");
const mod10 = @import("src/metal/buffer.zig");
const mod11 = @import("src/metal/command.zig");
const mod12 = @import("src/metal/device.zig");
const mod13 = @import("src/metal/pipeline.zig");
const mod14 = @import("src/model/catalog.zig");
const mod15 = @import("src/model/config.zig");
const mod16 = @import("src/model/gguf.zig");
const mod17 = @import("src/model/loader_metal.zig");
const mod18 = @import("src/model/loader.zig");
const mod19 = @import("src/model/managed.zig");
const mod20 = @import("src/model/tokenizer.zig");
const mod21 = @import("src/scheduler/kv_cache.zig");
const mod22 = @import("src/scheduler/request.zig");
const mod23 = @import("src/scheduler/scheduler.zig");
const mod24 = @import("src/server/http.zig");
const mod25 = @import("src/server/model_manager_metal.zig");
const mod26 = @import("src/server/model_manager.zig");
const mod27 = @import("src/server/routes.zig");
const mod28 = @import("src/server/session.zig");
const mod29 = @import("src/vulkan/buffer.zig");
const mod30 = @import("src/vulkan/command.zig");
const mod31 = @import("src/vulkan/gpu_detect.zig");
const mod32 = @import("src/vulkan/instance.zig");
const mod33 = @import("src/vulkan/pipeline.zig");
